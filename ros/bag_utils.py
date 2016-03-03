"""ROS Bag API"""

# Author: Sudeep Pillai <spillai@csail.mit.edu>
# License: MIT

import sys
import numpy as np
import cv2, os.path, lcm, zlib

import roslib
import tf

import rosbag
import rospy

from message_filters import ApproximateTimeSynchronizer

from genpy.rostime import Time
from sensor_msgs.msg import Image
from cv_bridge import CvBridge, CvBridgeError
from tf2_msgs.msg import TFMessage

from bot_externals.log_utils import Decoder, LogReader
from bot_vision.image_utils import im_resize
from bot_vision.imshow_utils import imshow_cv
from bot_geometry.rigid_transform import RigidTransform
from bot_vision.camera_utils import CameraIntrinsic

class GazeboDecoder(Decoder): 
    """
    Model state decoder for gazebo
    """
    def __init__(self, every_k_frames=1): 
        Decoder.__init__(self, channel='/gazebo/model_states', every_k_frames=every_k_frames)
        self.index = None

    def decode(self, msg): 
        if self.index is None: 
            for j, name in enumerate(msg.name): 
                if name == 'mobile_base': 
                    self.index = j
                    break

        pose = msg.pose[self.index]
        tvec, ori = pose.position, pose.orientation
        return RigidTransform(xyzw=[ori.x,ori.y,ori.z,ori.w], tvec=[tvec.x,tvec.y,tvec.z])

class CameraInfoDecoder(Decoder): 
    """
    Basic CameraIntrinsic deocder for ROSBags (from CameraInfo)
    """
    def __init__(self, channel='/camera/rgb/camera_info'): 
        Decoder.__init__(self, channel=channel)

    def decode(self, msg): 
        print dir(msg), self.channel
        return CameraIntrinsic(K=np.float64(msg.K).reshape(3,3), D=np.float64(msg.D).ravel(), 
                               shape=[msg.height, msg.width])
        
class ImageDecoder(Decoder): 
    """
    Encoding types supported: 
        bgr8, 32FC1
    """
    def __init__(self, channel='/camera/rgb/image_raw', every_k_frames=1, scale=1., encoding='bgr8', compressed=False): 
        Decoder.__init__(self, channel=channel, every_k_frames=every_k_frames)
        self.scale = scale
        self.encoding = encoding
        self.bridge = CvBridge()
        self.compressed = compressed

    def decode(self, msg): 
        if self.compressed: 
            im = cv2.imdecode(np.fromstring(msg.data, np.uint8), cv2.CV_LOAD_IMAGE_COLOR)
        else: 
            try: 
                im = self.bridge.imgmsg_to_cv2(msg, self.encoding)
                # print("%.6f" % msg.header.stamp.to_sec())
            except CvBridgeError as e:
                print e

        return im_resize(im, scale=self.scale)


class ApproximateTimeSynchronizerBag(ApproximateTimeSynchronizer): 
    """
    See reference implementation in message_filters.__init__.py 
    for ApproximateTimeSynchronizer
    """

    def __init__(self, topics, queue_size, slop): 
        ApproximateTimeSynchronizer.__init__(self, [], queue_size, slop)
        self.queues = [{} for f in topics]
        self.topic_queue = {topic: self.queues[ind] for ind, topic in enumerate(topics)}

    def add_topic(self, topic, msg):
        self.add(msg, self.topic_queue[topic])

class StereoSynchronizer(object): 
    """
    Time-synchronized stereo image decoder
    """
    def __init__(self, left_channel, right_channel, on_stereo_cb, 
                 every_k_frames=1, scale=1., encoding='bgr8', compressed=False): 

        self.left_channel = left_channel
        self.right_channel = right_channel

        self.decoder = ImageDecoder(every_k_frames=every_k_frames, 
                                    scale=scale, encoding=encoding, compressed=compressed)

        slop_seconds = 0.02
        queue_len = 10 
        self.synch = ApproximateTimeSynchronizerBag([left_channel, right_channel], queue_len, slop_seconds)
        self.synch.registerCallback(self.on_stereo_sync)
        self.on_stereo_cb = on_stereo_cb

        self.on_left = lambda t, msg: self.synch.add_topic(left_channel, msg)
        self.on_right = lambda t, msg: self.synch.add_topic(right_channel, msg)
        
    def on_stereo_sync(self, lmsg, rmsg): 
        limg = self.decoder.decode(lmsg)
        rimg = self.decoder.decode(rmsg)
        return self.on_stereo_cb(limg, rimg)
        
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


class TfDecoderAndPublisher(Decoder): 
    """
    """
    def __init__(self, channel='/tf', every_k_frames=1):
        Decoder.__init__(self, channel=channel, every_k_frames=every_k_frames)
        self.pub_ = None

    def decode(self, msg): 
        if self.pub_ is None: 
            self.pub_ = rospy.Publisher('/tf', TFMessage, queue_size=10, latch=True)
        self.pub_.publish(msg)
        return None

def NavMsgDecoder(channel, every_k_frames=1): 
    def odom_decode(data): 
        tvec, ori = data.pose.pose.position, data.pose.pose.orientation
        return RigidTransform(xyzw=[ori.x,ori.y,ori.z,ori.w], tvec=[tvec.x,tvec.y,tvec.z])
    return Decoder(channel=channel, every_k_frames=every_k_frames, decode_cb=lambda data: odom_decode(data))

def PoseStampedMsgDecoder(channel, every_k_frames=1): 
    def odom_decode(data): 
        tvec, ori = data.pose.position, data.pose.orientation
        return RigidTransform(xyzw=[ori.x,ori.y,ori.z,ori.w], tvec=[tvec.x,tvec.y,tvec.z])
    return Decoder(channel=channel, every_k_frames=every_k_frames, decode_cb=lambda data: odom_decode(data))

class ROSBagReader(LogReader): 
    def __init__(self, filename, decoder=None, start_idx=0, every_k_frames=1, max_length=None, index=False, verbose=False):
        super(ROSBagReader, self).__init__(filename, decoder=decoder, start_idx=start_idx, 
                                           every_k_frames=every_k_frames, max_length=max_length, index=index, verbose=verbose)

        if self.start_idx < 0 or self.start_idx > 100: 
            raise ValueError('start_idx in ROSBagReader expects a percentage [0,100], provided {:}'.format(self.start_idx))

        # TF relations
        self.relations_map = {}
        
        # # Gazebo states (if available)
        # self._publish_gazebo_states()

    # def _publish_gazebo_states(self): 
    #     """
    #     Perform a one-time publish of all the gazebo states
    #      (available via /gazebo/link_states, /gazebo/model_states)
    #     """

    #     from gazebo_msgs.msg import LinkStates
    #     from gazebo_msgs.msg import ModelStates

    #     self.gt_poses = []

    #     # Assuming the index of the model state does not change
    #     ind = None

    #     print('Publish Gazebo states')
    #     for self.idx, (channel, msg, t) in enumerate(self.log.read_messages(topics='/gazebo/model_states')): 
    #         if ind is None: 
    #             for j, name in enumerate(msg.name): 
    #                 if name == 'mobile_base': 
    #                     ind = j
    #                     break
    #         pose = msg.pose[ind]
    #         tvec, ori = pose.position, pose.orientation
    #         self.gt_poses.append(RigidTransform(xyzw=[ori.x,ori.y,ori.z,ori.w], tvec=[tvec.x,tvec.y,tvec.z]))
            
    #     print('Finished publishing gazebo states {:}'.format(len(self.gt_poses)))
        
    #     # import bot_externals.draw_utils as draw_utils
    #     # draw_utils.publish_pose_list('robot_poses', 
    #     #                              self.gt_poses[::10], frame_id='origin', reset=True)


    def load_log(self, filename): 
        return rosbag.Bag(filename, 'r', chunk_threshold=100 * 1024 * 1024)

    def tf(self, from_tf, to_tf): 
        try: 
            return self.relations_map[(from_tf, to_tf)]
        except: 
            raise KeyError('Relations map does not contain {:}=>{:} tranformation'.format(from_tf, to_tf))

    def establish_tfs(self, relations):
        """
        Perform a one-time look up of all the requested
        *static* relations between frames (available via /tf)
        """

        # Init node and tf listener
        rospy.init_node(self.__class__.__name__, disable_signals=True)
        tf_listener = tf.TransformListener()

        # Create tf decoder
        # st, end = self.log.get_start_time(), self.log.get_end_time()
        # start_t = Time(st + (end-st) * self.start_idx / 100.0)
        tf_dec = TfDecoderAndPublisher(channel='/tf')

        # Establish tf relations
        print('Establishing tfs from ROSBag')
        for self.idx, (channel, msg, t) in enumerate(self.log.read_messages(topics='/tf')): 
            tf_dec.decode(msg)

            for (from_tf, to_tf) in relations:  
                try:
                    (trans,rot) = tf_listener.lookupTransform(from_tf, to_tf, t)
                    self.relations_map[(from_tf,to_tf)] = RigidTransform(tvec=trans, xyzw=rot)
                    # print('\tSuccessfully received transform: {:} => {:} {:}'
                    #       .format(from_tf, to_tf, self.relations_map[(from_tf,to_tf)]))
                except (tf.LookupException, tf.ConnectivityException, tf.ExtrapolationException):
                    pass

            # Finish up once we've established all the requested tfs
            if len(self.relations_map) == len(relations): 
                break

        try: 
            tfs = [self.relations_map[(from_tf,to_tf)] for (from_tf, to_tf) in relations] 
            for (from_tf, to_tf) in relations: 
                print('\tSuccessfully received transform: {:} => {:} {:}'
                      .format(from_tf, to_tf, self.relations_map[(from_tf,to_tf)]))

        except: 
            raise RuntimeError('Error concerning tf lookup')
        print('Established {:} relations\n'.format(len(tfs)))
        
        return tfs 

    def retrieve_tf_relations(self, relations): 
        """
        Perform a one-time look up of all the 
        *static* relations between frames (available via /tf)
        and check if the expected relations hold

        Channel => frame_id

        """
        # if not isinstance(relations, map): 
        #     raise RuntimeError('Provided relations map is not a dict')

        # Check tf relations map
        print('Checking tf relations in ROSBag')
        checked = set()
        relations_lut = dict((k,v) for (k,v) in relations)

        # Decoders are non-gazebo sensors
        decoders = [dec for dec in self.decoder.keys() if 'gazebo' not in dec ]
        for self.idx, (channel, msg, t) in enumerate(self.log.read_messages(topics=decoders)):
            try: 
                if relations_lut[channel] == msg.header.frame_id: 
                    checked.add(channel)
                else: 
                    raise RuntimeError('TF Check failed {:} mapped to {:} instead of {:}'
                                       .format(channel, msg.header.frame_id, relations_lut[channel]))
            except Exception as e: 
                raise ValueError('Wrongly defined relations_lut {:}'.format(e))
            
                # Finish up
            if len(checked) == len(relations_lut):
                break
        print('Checked {:} relations\n'.format(len(checked)))
        return  

    def retrieve_camera_calibration(self, topic):
        # Retrieve camera calibration
        dec = CameraInfoDecoder(channel=topic)

        print('Retrieve camera calibration')
        for self.idx, (channel, msg, t) in enumerate(self.log.read_messages(topics=topic)): 
            return dec.decode(msg) 
                    
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
            # print self._log.get_message_count(topic_filters=self.decoder_keys())
            st, end = self.log.get_start_time(), self.log.get_end_time()
            start_t = Time(st + (end-st) * self.start_idx / 100.0)
            
            print('Reading ROSBag from {:3.2f}% onwards'.format(self.start_idx))
            for self.idx, (channel, msg, t) in enumerate(
                    self.log.read_messages(
                        topics=self.decoder.keys(), start_time=start_t
                    )
            ):

                if self.verbose: 
                    print('Channel: {:}, t: {:}'.format(channel, t))
                res, msg = self.decode_msg(channel, msg, t)
                if res: 
                    yield msg
                
    def decode_msg(self, channel, data, t): 
        try: 
            # Check if log index has reached desired start index, 
            # and only then check if decode necessary  
            dec = self.decoder[channel]
            if dec.should_decode():
                return True, (t, channel, dec.decode(data))
        except: 
            import traceback
            traceback.print_exc()
                    
        return False, (None, None)

    def iter_frames(self):
        return self.iteritems()

class ROSBagController(object): 
    def __init__(self, dataset): 
        """
        Setup channel => callbacks so that they are automatically called 
        with appropriate decoded data and timestamp
        """
        self.dataset_ = dataset
        self.ctrl_cb_ = {}
        self.ctrl_idx_ = 0

    def subscribe(self, channel, callback): 
        self.ctrl_cb_[channel] = callback

    def run(self):
        if not len(self.ctrl_cb_): 
            raise RuntimeError('No callbacks registered yet, subscribe to channels first!')

        for self.ctrl_idx_, (t, ch, data) in enumerate(self.dataset_.iter_frames()): 
            if ch in self.ctrl_cb_: 
                self.ctrl_cb_[ch](t, data)

    @property
    def index(self): 
        return self.ctrl_idx_

    @property
    def filename(self): 
        return self.dataset_.filename

    @property
    def controller(self): 
        """
        Should return the dataset (for offline bag-based callbacks), and 
        should return the rosnode (for online/live callbacks)
        """
        return self.dataset_

