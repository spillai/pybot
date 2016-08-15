"""ROS Bag API"""

# Author: Sudeep Pillai <spillai@csail.mit.edu>
# License: MIT

import sys
import numpy as np
import cv2
import time

import tf

import rosbag
import rospy

from message_filters import ApproximateTimeSynchronizer

from genpy.rostime import Time
from sensor_msgs.msg import Image
from cv_bridge import CvBridge, CvBridgeError
from tf2_msgs.msg import TFMessage

from pybot.externals.log_utils import Decoder, LogReader, LogController, LogDB
from pybot.vision.image_utils import im_resize
from pybot.vision.imshow_utils import imshow_cv
from pybot.vision.camera_utils import CameraIntrinsic
from pybot.geometry.rigid_transform import RigidTransform


class GazeboDecoder(Decoder): 
    """
    Model state decoder for gazebo
    """
    def __init__(self, every_k_frames=1): 
        Decoder.__init__(self, channel='/gazebo/model_states', every_k_frames=every_k_frames)
        self.__gazebo_index = None

    def decode(self, msg): 
        if self.__gazebo_index is None: 
            for j, name in enumerate(msg.name): 
                if name == 'mobile_base': 
                    self.__gazebo_index = j
                    break

        pose = msg.pose[self.__gazebo_index]
        tvec, ori = pose.position, pose.orientation
        return RigidTransform(xyzw=[ori.x,ori.y,ori.z,ori.w], tvec=[tvec.x,tvec.y,tvec.z])

class CameraInfoDecoder(Decoder): 
    """
    Basic CameraIntrinsic deocder for ROSBags (from CameraInfo)
    """
    def __init__(self, channel='/camera/rgb/camera_info'): 
        Decoder.__init__(self, channel=channel)

    def decode(self, msg): 
        # print dir(msg), self.channel
        """        
        D, K, P, R, binning_x, binning_y, distortion_model, 
        header, height, roi, width
        """        
        return CameraIntrinsic(K=np.float64(msg.K).reshape(3,3), 
                               D=np.float64(msg.D).ravel(), 
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
            im = cv2.imdecode(np.fromstring(msg.data, np.uint8), 
                              cv2.CV_LOAD_IMAGE_COLOR)
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
                print("{} :: No precomputed map given. Computing one.".format(self.__class__.__name__))

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
        print self.log

        # for channel, ch_info in info.topics.iteritems(): 
        #     print channel, ch_info.message_count
        
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
        
    #     # import pybot.externals.draw_utils as draw_utils
    #     # draw_utils.publish_pose_list('robot_poses', 
    #     #                              self.gt_poses[::10], frame_id='origin', reset=True)

    def length(self, topic): 
        info = self.log.get_type_and_topic_info()
        return info.topics[topic].message_count

    def load_log(self, filename): 
        st = time.time()
        print('{} :: Loading ROSBag {} ...'.format(self.__class__.__name__, filename))
        bag = rosbag.Bag(filename, 'r', chunk_threshold=100 * 1024 * 1024)
        print('{} :: Done loading {} in {:5.2f} seconds'.format(self.__class__.__name__, filename, time.time() - st))
        return bag

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
        print('{} :: Establishing tfs from ROSBag'.format(self.__class__.__name__))
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
        print('{} Established {:} relations\n'.format(self.__class__.__name__, len(tfs)))
        
        return tfs 

    def calib(self, channel=''):
        return self.retrieve_camera_calibration(channel)

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
        print('{} :: Checking tf relations in ROSBag'.format(self.__class__.__name__))
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
        print('{} :: Checked {:} relations\n'.format(self.__class__.__name__, len(checked)))
        return  

    def retrieve_camera_calibration(self, topic):
        # Retrieve camera calibration
        dec = CameraInfoDecoder(channel=topic)

        print('{} :: Retrieve camera calibration for {}'.format(self.__class__.__name__, topic))
        for self.idx, (channel, msg, t) in enumerate(self.log.read_messages(topics=topic)): 
            return dec.decode(msg) 
                    
    def _index(self): 
        raise NotImplementedError()

    def itercursors(self, topics=[], reverse=False):
        if self.index is not None: 
            raise NotImplementedError('Cannot provide items indexed')
        
        if reverse: 
            raise NotImplementedError('Cannot provide items in reverse when file is not indexed')

        # Decode only messages that are supposed to be decoded 
        # print self._log.get_message_count(topic_filters=self.decoder_keys())
        st, end = self.log.get_start_time(), self.log.get_end_time()
        start_t = Time(st + (end-st) * self.start_idx / 100.0)

        print('{} :: Reading ROSBag from {:3.2f}% onwards'.format(self.__class__.__name__, self.start_idx))
        for self.idx, (channel, msg, t) in \
            enumerate(self.log.read_messages(
                topics=self.decoder.keys() if not len(topics) else topics, 
                start_time=start_t)):
            yield (t, channel, msg)

    def iteritems(self, topics=[], reverse=False): 
        for (t, channel, msg) in self.itercursors(topics=topics, reverse=reverse): 
            try: 
                res, (t, ch, data) = self.decode_msg(channel, msg, t)
                if res: 
                    yield (t, ch, data)
            except Exception, e: 
                print('ROSBagReader.iteritems() :: {:}'.format(e))

    def iterframes(self):
        return self.iteritems()

    @property
    def db(self): 
        return BagDB(self)

class ROSBagController(LogController): 
    def __init__(self, dataset): 
        """
        See LogController
        """
        LogController.__init__(self, dataset)

class BagDB(LogDB): 
    def __init__(self, dataset): 
        """
        """
        LogDB.__init__(self, dataset)

    # @property
    # def poses(self): 
    #     return [v.pose for k,v in self.frame_index_.iteritems()]
        
    # def _index(self, pose_channel='/odom', rgb_channel='/camera/rgb/image_raw'): 
    #     """
    #     Constructs a look up table for the following variables: 
        
    #         self.frame_index_:  rgb/img.png -> TangoFrame
    #         self.frame_idx2name_: idx -> rgb/img.png
    #         self.frame_name2idx_: idx -> rgb/img.png

    #     where TangoFrame (index_in_the_dataset, timestamp, )
    #     """

    #     # 1. Iterate through both poses and images, and construct frames
    #     # with look up table for filename str -> (timestamp, pose, annotation) 
    #     poses = []
    #     # pose_decode = lambda msg_item: \
    #     #               self.dataset.decoder[pose_channel].decode(msg_item)

    #     # Note: Control flow for idx is critical since start_idx could
    #     # potentially change the offset and destroy the pose_index
    #     for idx, (t, ch, data) in enumerate(self.dataset.iteritems()): 
    #         pose = None
    #         if ch == pose_channel: 
    #             try: 
    #                 pose = data
    #             except: 
    #                 pose = None
    #         poses.append(pose)

    #     # Find valid and missing poses
    #     # pose_inds: log_index -> closest_valid_index
    #     valid_arr = np.array(
    #         map(lambda item: item is not None, poses), dtype=np.bool)
    #     pose_inds = BagDB._nn_pose_fill(valid_arr)

    #     # Create indexed frames for lookup        
    #     # self.frame_index_:  rgb/img.png -> TangoFrame
    #     # self.frame_idx2name_: idx -> rgb/img.png
    #     # self.frame_name2idx_: rgb/img.png -> idx
    #     img_decode = lambda msg_item: \
    #                 self.dataset.decoder[TangoFile.RGB_CHANNEL].decode(msg_item)
    #     self.frame_index_ = OrderedDict([
    #         (img_msg, TangoFrame(idx, t, img_msg, poses[pose_inds[idx]], 
    #                              self.dataset.annotationdb[img_msg], img_decode))
    #         for idx, (t, ch, img_msg) in enumerate(self.dataset.itercursors()) \
    #         if ch == TangoFile.RGB_CHANNEL
    #     ])
    #     self.frame_idx2name_ = OrderedDict([
    #         (idx, k) for idx, k in enumerate(self.frame_index_.keys())
    #     ])
    #     self.frame_name2idx_ = OrderedDict([
    #         (k, idx) for idx, k in enumerate(self.frame_index_.keys())
    #     ])

    # def iterframes(self): 
    #     """
    #     Ground truth reader interface for Images [with time, pose,
    #     annotation] : lookup corresponding annotation, and filled in
    #     poses from nearest available timestamp
    #     """
    #     # self._check_ground_truth_availability()

    #     # Iterate through both poses and images, and construct frames
    #     # with look up table for filename str -> (timestamp, pose, annotation) 
    #     for img_msg, frame in self.frame_index_.iteritems(): 
    #         yield (frame.timestamp, img_msg, frame)

    # def iterframes_indices(self, inds): 
    #     for ind in inds: 
    #         img_msg = self.frame_idx2name_[ind]
    #         frame = self.frame_index_[img_msg]
    #         yield (frame.timestamp, img_msg, frame)

    # def iterframes_range(self, ind_range): 
    #     assert(isinstance(ind_range, tuple) and len(ind_range) == 2)
    #     st, end = ind_range
    #     inds = np.arange(0 if st < 0 else st, 
    #                      len(self.frame_index_) if end < 0 else end+1)
    #     return self.iterframes_indices(inds)

    # @property
    # def annotated_inds(self): 
    #     return self.dataset.annotationdb.annotated_inds

    # @property
    # def object_annotations(self): 
    #     return self.dataset.annotationdb.object_annotations

    # @property
    # def objects(self): 
    #     return self.dataset.annotationdb.objects

    # def iter_object_annotations(self, target_name=''): 
    #     frame_keys, polygon_inds = self.dataset.annotationdb.find_object_annotations(target_name)
    #     for idx, (fkey,pind) in enumerate(izip(frame_keys, polygon_inds)): 
    #         try: 
    #             f = self[fkey]
    #         except KeyError, e: 
    #             print(e)
    #             continue
    #         assert(f.is_annotated)
    #         yield f, pind
        
    # # def list_annotations(self, target_name=None): 
    # #     " List of lists"
    # #     inds = self.annotated_inds
    # #     return [ filter(lambda frame: 
    # #                     target_name is None or name is in target_name, 
    # #                     self.dataset.annotationdb.iterframes(inds))

    # def print_index_info(self): 
    #     # Retrieve ground truth information
    #     gt_str = '{} frames annotated ({} total annotations)'\
    #         .format(self.dataset.annotationdb.num_frame_annotations, 
    #                 self.dataset.annotationdb.num_annotations) \
    #         if self.dataset.ground_truth_available else 'Not Available'

    #     # Pretty print IndexDB description 
    #     print('\nTango IndexDB \n========\n'
    #           '\tFrames: {:}\n'
    #           '\tGround Truth: {:}\n'
    #           .format(len(self.frame_index_), gt_str)) 

    # @property
    # def index(self): 
    #     return self.frame_index_

