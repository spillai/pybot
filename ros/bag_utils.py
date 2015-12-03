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
    def __init__(self, channel='', every_k_frames=1, decode_cb=lambda data: None): 
        self.channel = channel
        self.every_k_frames = every_k_frames
        self.decode_cb = decode_cb
        self.idx = 0

    def decode(self, data): 
        try: 
            return self.decode_cb(data)
        except Exception as e:
            print e
            raise RuntimeError('Error decoding channel: %s by %s' % (self.channel, self))

    def can_decode(self, channel): 
        return self.channel == channel

    def should_decode(self): 
        self.idx += 1
        return self.idx % self.every_k_frames == 0 
        
class ImageDecoder(Decoder): 
    """
    Encoding types supported: 
        bgr8, 32FC1
    """
    def __init__(self, channel='/camera/rgb/image_raw', every_k_frames=1, scale=1., encoding='bgr8'): 
        Decoder.__init__(self, channel=channel, every_k_frames=every_k_frames)
        self.scale = scale
        self.encoding = encoding
        self.bridge = CvBridge()

    def decode(self, msg): 
        try:
            im = self.bridge.imgmsg_to_cv2(msg, self.encoding)
            # print("%.6f" % msg.header.stamp.to_sec())
            return im_resize(im, scale=self.scale)
        except CvBridgeError, e:
            print e

class LaserScanDecoder(Decoder): 
    """
    Mostly stripped from 
    https://github.com/ros-perception/laser_geometry/blob/indigo-devel/src/laser_geometry/laser_geometry.py
    """
    def __init__(self, channel='/scan', every_k_frames=1):
        Decoder.__init__(self, channel=channel, every_k_frames=every_k_frames)

        self.__angle_min = 0.0
        self.__angle_max = 0.0
        self.__cos_sin_map = np.array([[]])
                
    def decode(self, msg): 
        try:

            N = len(msg.ranges)

            zeros = np.zeros(shape=(N,1))
            ranges = np.array(msg.ranges)
            ranges = np.array([ranges, ranges])

            if (self.__cos_sin_map.shape[1] != N or
               self.__angle_min != msg.angle_min or
                self.__angle_max != msg.angle_max):
                print("No precomputed map given. Computing one.")

                self.__angle_min = msg.angle_min
                self.__angle_max = msg.angle_max

                cos_map = [np.cos(msg.angle_min + i * msg.angle_increment)
                       for i in range(N)]
                sin_map = [np.sin(msg.angle_min + i * msg.angle_increment)
                        for i in range(N)]

                self.__cos_sin_map = np.array([cos_map, sin_map])

            return np.hstack([(ranges * self.__cos_sin_map).T, zeros])
        except Exception as e:
            print e


def NavMsgDecoder(channel, every_k_frames=1): 
    def odom_decode(data): 
        tvec, ori = data.pose.pose.position, data.pose.pose.orientation
        return RigidTransform(xyzw=[ori.x,ori.y,ori.z,ori.w], tvec=[tvec.x,tvec.y,tvec.z])
    return Decoder(channel=channel, every_k_frames=every_k_frames, decode_cb=lambda data: odom_decode(data))

def TfDecoder(channel, every_k_frames=1): 
    def tf_decode(data): 
        return None
    return Decoder(channel=channel, every_k_frames=every_k_frames, decode_cb=lambda data: tf_decode(data))

class ROSBagReader(LogReader): 
    def __init__(self, *args, **kwargs): 
        super(ROSBagReader, self).__init__(*args, **kwargs)

    def load_log(self, filename): 
        return rosbag.Bag(filename, 'r')

    def _index(self): 
        raise NotImplementedError()

    def iteritems(self, reverse=False): 
        if self.index is not None: 
            raise NotImplementedError('Cannot provide items indexed')
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

            # Decode only messages that are supposed to be decoded 
            for channel, msg, t in self._log.read_messages(topics=self.decoder.keys()):
                res, msg = self.decode_msg(channel, msg, t)
                if res: 
                    yield msg

    def decode_msg(self, channel, data, t): 
        try: 
            # Check if log index has reached desired start index, 
            # and only then check if decode necessary  
            dec = self.decoder[channel]
            if self.should_decode() and dec.should_decode(): 
                return True, (channel, dec.decode(data))
        except Exception as e:
            print e
            # raise RuntimeError('Failed to decode data from channel: %s, mis-specified decoder?' % channel)
        
        return False, (None, None)


    def should_decode(self): 
        self.idx += 1
        return self.idx >= self.start_idx

    def iter_frames(self):
        return self.iteritems()
