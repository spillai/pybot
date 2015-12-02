import numpy as np
import cv2, os.path, lcm, zlib

import roslib
# roslib.load_manifest(PKG)

import rosbag
import rospy

from sensor_msgs.msg import Image
from cv_bridge import CvBridge, CvBridgeError

from bot_externals.log_utils import LogReader
from bot_vision.image_utils import im_resize
from bot_vision.imshow_utils import imshow_cv
from bot_geometry.rigid_transform import RigidTransform

class Decoder(object): 
    def __init__(self, channel='', decode_cb=lambda data: None): 
        self.channel = channel
        self.decode_cb = decode_cb

    def decode(self, data): 
        try: 
            return self.decode_cb(data)
        except: 
            raise RuntimeError('Error decoding channel: %s' % self.channel)

    def can_decode(self, channel): 
        return self.channel == channel

class ImageDecoder(Decoder): 
    def __init__(self, channel='/camera/rgb/image_raw', scale=1.): 
        Decoder.__init__(self, channel=channel)
        self.scale = scale
        self.bridge = CvBridge()

    def decode(self, msg): 
        try:
            im = self.bridge.imgmsg_to_cv2(msg,'bgr8')
            # print("%.6f" % msg.header.stamp.to_sec())
            return im_resize(im, scale=self.scale)
        except CvBridgeError, e:
            print e

def NavMsgDecoder(channel): 
    def odom_decode(data): 
        tvec, ori = data.pose.pose.position, data.pose.pose.orientation
        return RigidTransform(xyzw=[ori.x,ori.y,ori.z,ori.w], tvec=[tvec.x,tvec.y,tvec.z])
    return Decoder(channel=channel, decode_cb=lambda data: odom_decode(data))

def TfDecoder(channel): 
    def tf_decode(data): 
        return None
    return Decoder(channel=channel, decode_cb=lambda data: tf_decode(data))
        
class ROSBagReader(LogReader): 
    def __init__(self, *args, **kwargs): 
        super(ROSBagReader, self).__init__(*args, **kwargs)

    def load_log(self, filename): 
        return rosbag.Bag(filename, 'r')

    def _index(self): 
        raise NotImplementedError()

    def iteritems(self, reverse=False): 
        if self.index is not None: 
            raise RuntimeError('Cannot provide items indexed')
            # if reverse: 
            #     for t in self.index[::-1]: 
            #         if self.start_idx != 0: 
            #             raise RuntimeWarning('No support for start_idx != 0')
            #         frame = self.get_frame_with_timestamp(t)
            #         yield frame
            # else: 
            #     for t in self.index: 
            #         frame = self.get_frame_with_timestamp(t)
            #         yield frame
        else: 
            if reverse: 
                raise RuntimeError('Cannot provide items in reverse when file is not indexed')

            for channel, msg, t in self._log.read_messages():
                res, msg = self.decode_msgs(channel, msg, t)
                if res: yield msg

                # if ev.channel == self.decoder.channel: 
                #     self.idx += 1
                #     if idx % self.every_k_frames == 0: 
                #         yield self.decoder.decode(ev.data)

    def decode_msg(self, channel, data, t, dec):
        if channel == dec.channel: 
            self.idx += 1
            if self.idx >= self.start_idx and self.idx % self.every_k_frames == 0: 
                return True, (channel, dec.decode(data))
        return False, (None, None)

    def decode_msgs(self, channel, data, t): 
        if isinstance(self.decoder, list):
            res, msg = False, None
            for dec in self.decoder: 
                res, msg = self.decode_msg(channel, data, t, dec)
                if res: break
            return res, msg
        else: 
            # when accessing only single decoding, 
            # return value as is
            res, msg = self.decode_msg(channel, msg, t, self.decoder)
            return res, msg[1]

    def iter_frames(self):
        return self.iteritems()
