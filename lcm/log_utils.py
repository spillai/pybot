"""LCM Log API"""

# Author: Sudeep Pillai <spillai@csail.mit.edu>
# License: MIT

import numpy as np
import cv2
import os.path
import lcm
import zlib
from itertools import islice

from bot_vision.camera_utils import construct_K, DepthCamera
from bot_vision.image_utils import im_resize

from bot_externals.log_utils import Decoder, LogReader, LogController

import bot_core.image_t as image_t
import bot_core.pose_t as pose_t
import bot_param.update_t as update_t

class BotParamDecoder(Decoder): 
    def __init__(self, channel='PARAM_UPDATE', every_k_frames=1): 
        Decoder.__init__(self, channel=channel, every_k_frames=every_k_frames)
        
    def decode(self, data):
        msg = update_t.decode(data)
        return msg

class MicrostrainDecoder(Decoder): 
    def __init__(self, channel='MICROSTRAIN_INS'): 
        Decoder.__init__(self, channel=channel)
        from microstrain import ins_t 
        self.ins_t_decode_ = lambda data : ins_t.decode(data)

    def decode(self, data):
        msg = self.ins_t_decode_(data)
        return msg

class PoseDecoder(Decoder): 
    def __init__(self, channel='POSE', every_k_frames=1): 
        Decoder.__init__(self, channel=channel, every_k_frames=every_k_frames)
        
    def decode(self, data):
        msg = pose_t.decode(data)
        return msg

class ImageDecoder(Decoder): 
    def __init__(self, channel='CAMERA', scale=1., every_k_frames=1): 
        Decoder.__init__(self, channel=channel, every_k_frames=every_k_frames)
        self.scale = scale

    def decode(self, data): 
        msg = image_t.decode(data)
        if msg.pixelformat == image_t.PIXEL_FORMAT_GRAY: 
            return im_resize(np.asarray(bytearray(msg.data), dtype=np.uint8).reshape(msg.height, msg.width), scale=self.scale)
        elif msg.pixelformat == image_t.PIXEL_FORMAT_MJPEG: 
            im = cv2.imdecode(np.asarray(bytearray(msg.data), dtype=np.uint8), -1)
            return im_resize(im, scale=self.scale)
        else: 
            raise RuntimeError('Unknown pixelformat for ImageDecoder')

class StereoImageDecoder(ImageDecoder): 
    def __init__(self, split='vertical', channel='CAMERA', scale=1., every_k_frames=1): 
        ImageDecoder.__init__(self, channel=channel, scale=scale, every_k_frames=every_k_frames)
        if split == 'vertical' or split == 0: 
            self.split = 0
        elif split == 'horizontal' or split == 1: 
            self.split = 1
        else: 
            raise RuntimeError('Unknown image split type')

    def decode(self, data):
        return np.split(super(StereoImageDecoder, self).decode(data), 2, axis=self.split)

class KinectFrame: 
    def __init__(self, timestamp=None, img=None, depth=None, X=None): 
        self.timestamp = timestamp
        self.img = img
        self.depth = depth
        self.X = X

        from pybot_pcl import compute_normals, fast_bilateral_filter, median_filter
        self.compute_normals_ = compute_normals
        self.fast_bilateral_filter_ = fast_bilateral_filter
        self.median_filter_ = median_filter

    @property
    def Xest(self): 
        return self.fast_bilateral_filter_(self.X, sigmaS=20.0, sigmaR=0.05)

    @property
    def N(self): 
        return self.compute_normals(self.Xest, depth_change_factor=0.5, smoothing_size=10.0)

class KinectDecoder(Decoder): 
    kinect_params = dict(fx=576.09757860, fy=576.09757860, cx=319.50, cy=239.50)
    def __init__(self, channel='KINECT_FRAME', scale=1., 
                 extract_rgb=True, extract_depth=True, extract_X=True, bgr=True, every_k_frames=1):
        Decoder.__init__(self, channel=channel, every_k_frames=every_k_frames)
        self.skip = int(1.0 / scale);
        
        assert (self.skip >= 1)
        self.extract_rgb = extract_rgb
        self.extract_depth = extract_depth    
        self.extract_X = extract_X
        self.bgr = bgr

        from kinect.frame_msg_t import frame_msg_t 
        from kinect.image_msg_t import image_msg_t
        from kinect.depth_msg_t import depth_msg_t
        
        self.frame_msg_t_ = frame_msg_t
        self.image_msg_t_ = image_msg_t
        self.depth_msg_t_ = depth_msg_t
        
        # from openni.frame_msg_t import frame_msg_t as self.frame_msg_t_
        # from openni.image_msg_t import image_msg_t as self.image_msg_t_
        # from openni.depth_msg_t import depth_msg_t as self.depth_msg_t_
                
        if self.extract_X: 
            K = construct_K(**KinectDecoder.kinect_params)
            self.camera = DepthCamera(K=K, shape=(480,640), skip=self.skip)

    def decode(self, data):
        img, depth, X = [None] * 3
        frame_msg = self.frame_msg_t_.decode(data)
        if self.extract_rgb: 
            img = self.decode_rgb(frame_msg)
        if self.extract_depth: 
            depth = self.decode_depth(frame_msg)
            if self.extract_X: 
                X = self.camera.reconstruct(depth)
        # frame = AttrDict(timestamp=frame_msg.timestamp, img=img, depth=depth, X=X) 
        return KinectFrame(timestamp=frame_msg.timestamp, img=img, depth=depth, X=X) 

    def decode_rgb(self, data): 
        w, h = data.image.width, data.image.height;
        if data.image.image_data_format == self.image_msg_t_.VIDEO_RGB_JPEG: 
            img = cv2.imdecode(np.asarray(bytearray(data.image.image_data), dtype=np.uint8), -1)
            bgr = img.reshape((h,w,3))[::self.skip, ::self.skip, :]             
        else: 
            img = np.fromstring(data.image.image_data, dtype=np.uint8)
            rgb = img.reshape((h,w,3))[::self.skip, ::self.skip, :] 
            bgr = cv2.cvtColor(rgb, cv2.COLOR_RGB2BGR)
        if not self.bgr: 
            return cv2.cvtColor(bgr, cv2.COLOR_BGR2RGB)
        else: 
            return bgr

    def decode_depth(self, data):
        # Extract depth image
        w, h = data.image.width, data.image.height;
        if data.depth.compression != self.depth_msg_t_.COMPRESSION_NONE: 
            depth = np.fromstring(zlib.decompress(data.depth.depth_data), dtype=np.uint16)
        else: 
            depth = np.fromstring(data.depth.depth_data, dtype=np.uint16)
        depth = np.reshape(depth, (h,w)).astype(np.float32) * 0.001; # in m
        depth = depth[::self.skip, ::self.skip] # skip pixels
        return depth

class LCMLogReader(LogReader): 
    def __init__(self, *args, **kwargs): 
        super(LCMLogReader, self).__init__(*args, **kwargs)
        self._lc = lcm.LCM()

    def load_log(self, filename): 
        return lcm.EventLog(self.filename, 'r')

    def _index(self): 
        utimes = np.array([ev.timestamp for ev in self.log], dtype=np.int64)
        inds = np.array([idx
                         for idx, ev in enumerate(self.log) 
                         if ev.channel == self.decoder.channel], dtype=np.int64)
        try: 
            self.index = utimes[np.maximum(self.start_idx, inds-1)][::self.every_k_frames]
        except: 
            raise RuntimeError('Probably failed to establish start_idx')

    @property
    def length(self): 
        return len(self.index)

    def get_frame_with_timestamp(self, t): 
        self.log.c_eventlog.seek_to_timestamp(t)
        while True: 
            ev = self.log.next()
            res, msg = self.decode_msgs(ev.channel, ev.data, ev.timestamp)
            if res: return msg

            # if ev.channel == self.decoder.channel: 
            #     break

        #         if res: yield msg

        # return self.decoder.decode(ev.data)

    def get_frame_with_index(self, idx): 
        assert(idx >= 0 and idx < len(self.index))
        return self.get_frame_with_timestamp(self.index[idx])

    def iteritems(self, reverse=False): 
        # Indexed iteration
        if self.index is not None: 
            if reverse: 
                for t in self.index[::-1]: 
                    if self.start_idx != 0: 
                        raise RuntimeWarning('No support for start_idx != 0')
                    yield self.get_frame_with_timestamp(t)                    
            else: 
                for t in self.index: 
                    yield self.get_frame_with_timestamp(t)                    

        # Unindexed iteration (usually much faster)
        else: 
            if reverse: 
                raise RuntimeError('Cannot provide items in reverse when file is not indexed')

            
            # iterator = take(self.log, max_length=self.max_length)
            max_length = 1e12 if self.max_length is None else self.max_length
            print('Taking first {:} frames for lcm log'.format(max_length))

            counts = 0
            for self.idx, ev in enumerate(self.log): 
                if counts >= max_length: break
                if self.idx > self.start_idx and \
                   self.idx % self.every_k_frames == 0:
                    res, msg = self.decode_msg(ev.channel, ev.data, ev.timestamp)
                    if res: 
                        yield msg
                        counts += 1

                # if ev.channel == self.decoder.channel: 
                #     self.idx += 1
                #     if idx % self.every_k_frames == 0: 
                #         yield self.decoder.decode(ev.data)

    def iterframes(self):
        return self.iteritems()

    def get_first_frame(self): 
        return self.get_frame_with_index(0)

class LCMLogController(LogController): 
    def __init__(self, dataset): 
        """
        See LogController
        """
        LogController.__init__(self, dataset)

def KinectLCMLogReader(filename=None, every_k_frames=1, **kwargs): 
    return LCMLogReader(filename=filename, every_k_frames=every_k_frames, decoder=KinectDecoder(**kwargs))        
