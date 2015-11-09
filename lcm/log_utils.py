#!/usr/bin/python
import numpy as np
import cv2, os.path, lcm, zlib
from collections import OrderedDict

from bot_utils.db_utils import AttrDict
from bot_vision.camera_utils import construct_K, DepthCamera
from bot_vision.image_utils import im_resize
from bot_vision.imshow_utils import imshow_cv

import bot_core.image_t as image_t
import bot_core.pose_t as pose_t
import bot_param.update_t as update_t

from kinect.frame_msg_t import frame_msg_t
from kinect.image_msg_t import image_msg_t
from kinect.depth_msg_t import depth_msg_t

# from openni.frame_msg_t import frame_msg_t
# from openni.image_msg_t import image_msg_t
# from openni.depth_msg_t import depth_msg_t

from pybot_pcl import compute_normals, fast_bilateral_filter, median_filter

class Decoder(object): 
    def __init__(self, channel=''): 
        self.channel = channel

    def decode(self, data): 
        return None


class BotParamDecoder(Decoder): 
    def __init__(self, channel='PARAM_UPDATE'): 
        Decoder.__init__(self, channel=channel)
        
    def decode(self, data):
        msg = update_t.decode(data)
        return msg

class PoseDecoder(Decoder): 
    def __init__(self, channel='CAMERA'): 
        Decoder.__init__(self, channel=channel)
        
    def decode(self, data):
        msg = pose_t.decode(data)
        return msg

class ImageDecoder(Decoder): 
    def __init__(self, channel='CAMERA', scale=1.): 
        Decoder.__init__(self, channel=channel)
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
    def __init__(self, split='vertical', channel='CAMERA', scale=1.): 
        ImageDecoder.__init__(self, channel=channel, scale=scale)
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

    @property
    def Xest(self): 
        return fast_bilateral_filter(self.X, sigmaS=20.0, sigmaR=0.05)

    @property
    def N(self): 
        return compute_normals(self.Xest, depth_change_factor=0.5, smoothing_size=10.0)

class KinectDecoder(Decoder): 
    kinect_params = AttrDict(fx=576.09757860, fy=576.09757860, cx=319.50, cy=239.50)
    def __init__(self, channel='KINECT_FRAME', scale=1., 
                 extract_rgb=True, extract_depth=True, extract_X=True, bgr=True):
        Decoder.__init__(self, channel=channel)
        self.skip = int(1.0 / scale);
        
        assert (self.skip >= 1)
        self.extract_rgb = extract_rgb
        self.extract_depth = extract_depth    
        self.extract_X = extract_X
        self.bgr = bgr

        if self.extract_X: 
            K = construct_K(**KinectDecoder.kinect_params)
            self.camera = DepthCamera(K=K, shape=(480,640), skip=self.skip)

    def decode(self, data):
        img, depth, X = [None] * 3
        frame_msg = frame_msg_t.decode(data)
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
        if data.image.image_data_format == image_msg_t.VIDEO_RGB_JPEG: 
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
        if data.depth.compression != depth_msg_t.COMPRESSION_NONE: 
            depth = np.fromstring(zlib.decompress(data.depth.depth_data), dtype=np.uint16)
        else: 
            depth = np.fromstring(data.depth.depth_data, dtype=np.uint16)
        depth = np.reshape(depth, (h,w)).astype(np.float32) * 0.001; # in m
        depth = depth[::self.skip, ::self.skip] # skip pixels
        return depth

class LCMLogReader(object): 
    def __init__(self, filename=None, decoder=None, start_idx=0, every_k_frames=1, index=True):
        filename = os.path.expanduser(filename)
        if filename is None or not os.path.exists(os.path.expanduser(filename)):
            raise Exception('Invalid Filename: %s' % filename)
        print 'Kinect Reader: Opening file', filename        

        # Store attributes
        self.filename = filename
        self.decoder = decoder
        self.every_k_frames = every_k_frames
        self.start_idx = start_idx

        # Log specific
        self._lc = lcm.LCM()
        self._log = lcm.EventLog(self.filename, "r")

        # Build index
        self.idx = 0
        if index: 
            self._index()
        else: 
            self.index = None

    def _index(self): 
        utimes = np.array([ev.timestamp for ev in self._log], dtype=np.int64)
        inds = np.array([idx
                         for idx, ev in enumerate(self._log) 
                         if ev.channel == self.decoder.channel], dtype=np.int64)
        try: 
            self.index = utimes[np.maximum(self.start_idx, inds-1)][::self.every_k_frames]
        except: 
            raise RuntimeError('Probably failed to establish start_idx')

    @property
    def length(self): 
        return len(self.index)

    def get_frame_with_timestamp(self, t): 
        self._log.c_eventlog.seek_to_timestamp(t)
        while True: 
            ev = self._log.next()
            res, msg = self.decode_msgs(ev)
            if res: return msg

            # if ev.channel == self.decoder.channel: 
            #     break

        #         if res: yield msg

        # return self.decoder.decode(ev.data)

    def get_frame_with_index(self, idx): 
        assert(idx >= 0 and idx < len(self.index))
        return self.get_frame_with_timestamp(self.index[idx])

    def decode_msg(self, ev, dec):
        if ev.channel == dec.channel: 
            self.idx += 1
            if self.idx >= self.start_idx and self.idx % self.every_k_frames == 0: 
                return True, (ev.channel, dec.decode(ev.data))
        return False, (None, None)

    def decode_msgs(self, ev): 
        if isinstance(self.decoder, list):
            res, msg = False, None
            for dec in self.decoder: 
                res, msg = self.decode_msg(ev, dec)
                if res: break
            return res, msg
        else: 
            # when accessing only single decoding, 
            # return value as is
            res, msg = self.decode_msg(ev, self.decoder)
            return res, msg[1]
            
    def iteritems(self, reverse=False): 
        if self.index is not None: 
            if reverse: 
                for t in self.index[::-1]: 
                    if self.start_idx != 0: 
                        raise RuntimeWarning('No support for start_idx != 0')
                    frame = self.get_frame_with_timestamp(t)
                    yield frame
            else: 
                for t in self.index: 
                    frame = self.get_frame_with_timestamp(t)
                    yield frame
        else: 
            if reverse: 
                raise RuntimeError('Cannot provide items in reverse when file is not indexed')

            for ev in self._log: 
                res, msg = self.decode_msgs(ev)
                if res: yield msg

                # if ev.channel == self.decoder.channel: 
                #     self.idx += 1
                #     if idx % self.every_k_frames == 0: 
                #         yield self.decoder.decode(ev.data)

    def iter_frames(self):
        return self.iteritems()

    def get_first_frame(self): 
        return self.get_frame_with_index(0)

def KinectLCMLogReader(filename=None, every_k_frames=1, **kwargs): 
    return LCMLogReader(filename=filename, every_k_frames=every_k_frames, decoder=KinectDecoder(**kwargs))        

if __name__ == "__main__": 
    import os.path
    from pybot_pcl import compute_normals

    log = KinectLCMLogReader(filename='~/data/2014_06_14_articulation_multibody/lcmlog-2014-06-14.05')
    # for frame in log.iter_frames(): 
    #     imshow_cv('frame', frame.img)
    #     imshow_cv('depth', frame.depth / 15)

    for frame in log.iteritems(reverse=True): 
        imshow_cv('frame', frame.img)
        imshow_cv('depth', frame.depth / 15)


        # # Build Index
        # self._build_index()

    # def _build_index(self): 
    #     """Build utime index for log"""

    #     # self.utime_map = OrderedDict();
    #     st = time.time()
    #     for ev in self.log:
    #         if ev.channel == self.channel:
    #             data = frame_msg_t.decode(ev.data);
    #             self.utime_map[data.timestamp] = ev.timestamp
    #             if count % 100 == 0: print 'Indexed %s frames: sn:%ld ev:%ld' % (count, data.timestamp, ev.timestamp)
    #     print 'Built index: %f seconds' % (time.time() - st)

    #     # Keep frame index
    #     if len(self.utime_map):
    #         self.cur_frame = 0
    #         sensor_utime = self.utime_map.keys()[self.cur_frame]
    #         event_utime = self.find_closest(sensor_utime)
    #         self.log.c_eventlog.seek_to_timestamp(event_utime)    





# class LCMLogPlayer: 
#     def __init__(self, filename=None):
#         if filename is None or not os.path.exists(filename):
#             raise Exception('Invalid Filename: %s' % filename)
#         self.reader = LCMLogReader(filename=filename)
#         self.reader.reset()
#         # print 'NUM FRAMES: ', self.reader.getNumFrames()

#         # # Default option
#         # if every_k_frames is None and k_frames is None and utimes is None: 
#         #     every_k_frames = 1

#         # self.every_k_frames = every_k_frames
#         # self.k_frames = k_frames

#         # self.fno = 0
#         # self.frame_inds = self.get_frame_inds()
#         # self.frame_utimes = utimes

#     def iterframes(self, every_k_frames=1): 
#         fnos = np.arange(0, self.reader.getNumFrames()-1, every_k_frames).astype(int)
#         for fno in fnos: 
#             frame = self.reader.getFrame(fno)
#             assert(frame is not None)
#             yield frame

#     def get_frames(self, every_k_frames=1): 
#         fnos = np.arange(0, self.reader.getNumFrames()-1, every_k_frames).astype(int)
#         return [self.reader.getFrame(fno) for fno in fnos]
        
#     def get_frame_inds(self): 
#         frames = None
#         if self.every_k_frames is not None: 
#             frames = np.arange(0, self.reader.getNumFrames()-1).astype(int)
#             if self.every_k_frames != 1: frames = frames[::self.every_k_frames]
#         elif self.k_frames is not None: 
#             frames = np.linspace(0, self.reader.getNumFrames()-1, self.k_frames).astype(int)
#         return frames
        
#     def reset(self): 
#         self.reader.reset()
#         self.fno = 0

#     def get_frame_with_percent(self, pc): 
#         assert(pc >= 0.0 and pc <= 1.0)
#         seek_to_index = int(pc * 1.0 * len(self.frame_inds))
#         assert(seek_to_index >= 0 and seek_to_index < len(self.frame_inds))
#         print 'SEEK to : ', seek_to_index
#         return self.get_frame_with_index(self.frame_inds[seek_to_index])

#     def get_frame_with_index(self, index): 
#         return self.reader.getFrame(index)

#     def get_next_frame_with_index(self): 
#         if self.fno >= len(self.frame_inds): 
#             return None
#         frame = self.reader.getFrame(self.frame_inds[self.fno])
#         self.fno += 1
#         return frame

#     def get_frame_with_utime(self, utime): 
#         return self.reader.getFrameWithTimestamp(utime)

#     def get_next_frame_with_utime(self): 
#         if self.fno >= len(self.frame_utimes): 
#             return None
#         frame = self.reader.getFrameWithTimestamp(self.frame_utimes[self.fno])
#         self.fno += 1
#         return frame

#     def get_next_frame(self): 
#         if self.frame_utimes is not None: 
#             return self.get_next_frame_with_utime()
#         else: 
#             return self.get_next_frame_with_index()


