#!/usr/bin/python
import numpy as np
import cv2, os.path, lcm, zlib

from PIL import Image
from collections import OrderedDict

from bot_utils.db_utils import AttrDict
from bot_vision.camera_utils import construct_K, DepthCamera
from bot_vision.imshow_utils import imshow_cv

from kinect.frame_msg_t import frame_msg_t
from kinect.image_msg_t import image_msg_t
from kinect.depth_msg_t import depth_msg_t

class KinectDecoder(object): 
    def __init__(self, channel='KINECT_FRAME', scale=1., 
                 extract_rgb=True, extract_depth=True, extract_X=True):
        
        self.channel = channel
        self.skip = int(1.0 / scale);
        
        assert (self.skip >= 1)
        self.extract_rgb = extract_rgb
        self.extract_depth = extract_depth    
        self.extract_X = extract_X

        if self.extract_X: 
            fx = 576.09757860
            K = construct_K(fx=fx, fy=fx, cx=319.50, cy=239.50)
            self.camera = DepthCamera(K=K, shape=(480,640))

    def decode(self, data):
        img, depth, X = [None] * 3
        frame = frame_msg_t.decode(data)
        if self.extract_rgb: 
            img = self.decode_rgb(frame)
        if self.extract_depth: 
            depth = self.decode_depth(frame)
            if self.extract_X: 
                X = self.camera.reconstruct(depth)
        return AttrDict(timestamp=frame.timestamp, img=img, depth=depth, X=X)

    def decode_rgb(self, data): 
        w, h = data.image.width, data.image.height;
        if data.image.image_data_format == image_msg_t.VIDEO_RGB_JPEG: 
            img = cv2.imdecode(np.asarray(bytearray(data.image.image_data), dtype=np.uint8), -1)
        else: 
            img = np.fromstring(data.image.image_data, dtype=np.uint8)
        return img.reshape((h,w,3))[::self.skip, ::self.skip, :] 

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
    def __init__(self, filename=None, decoder=None):
        filename = os.path.expanduser(filename)
        if filename is None or not os.path.exists(os.path.expanduser(filename)):
            raise Exception('Invalid Filename: %s' % filename)
        print 'Kinect Reader: Opening file', filename        

        # Store attributes
        self.filename = filename
        self.decoder = decoder

        # Log specific
        self._lc = lcm.LCM()
        self._log = lcm.EventLog(self.filename, "r")

    def iter_frames(self, every_k_frames=1):
        idx = 0
        for ev in self._log: 
            if ev.channel == self.decoder.channel: 
                idx += 1
                if idx % every_k_frames == 0: 
                    yield self.decoder.decode(ev.data)
                    
        

if __name__ == "__main__": 
    import os.path

    log = LCMLogReader(filename='~/data/2014_06_14_articulation_multibody/lcmlog-2014-06-14.05', 
                 decoder=KinectDecoder())
    for frame in log.iter_frames(): 
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


