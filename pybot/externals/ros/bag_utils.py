"""ROS Bag API"""

# Author: Sudeep Pillai <spillai@csail.mit.edu>
# License: MIT

import sys
import numpy as np
import cv2
import time
import os.path

import tf
import rosbag
import rospy
from message_filters import ApproximateTimeSynchronizer

from genpy.rostime import Time
from sensor_msgs.msg import Image
from cv_bridge import CvBridge, CvBridgeError
from cv_bridge.boost.cv_bridge_boost import cvtColor2
from tf2_msgs.msg import TFMessage

from pybot.utils.misc import Accumulator
from pybot.externals.log_utils import Decoder, LogReader, LogController, LogDB
from pybot.vision.image_utils import im_resize
from pybot.vision.imshow_utils import imshow_cv
from pybot.vision.camera_utils import CameraIntrinsic
from pybot.geometry.rigid_transform import RigidTransform
from pybot.utils.dataset.sun3d_utils import SUN3DAnnotationDB

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
        

def compressed_imgmsg_to_cv2(cmprs_img_msg, desired_encoding = "passthrough"):
    """
    Convert a sensor_msgs::CompressedImage message to an OpenCV :cpp:type:`cv::Mat`.

    :param cmprs_img_msg:   A :cpp:type:`sensor_msgs::CompressedImage` message
    :param desired_encoding:  The encoding of the image data, one of the following strings:

       * ``"passthrough"``
       * one of the standard strings in sensor_msgs/image_encodings.h

    :rtype: :cpp:type:`cv::Mat`
    :raises CvBridgeError: when conversion is not possible.

    If desired_encoding is ``"passthrough"``, then the returned image has the same format as img_msg.
    Otherwise desired_encoding must be one of the standard image encodings

    This function returns an OpenCV :cpp:type:`cv::Mat` message on success, or raises :exc:`cv_bridge.CvBridgeError` on failure.

    If the image only has one channel, the shape has size 2 (width and height)
    """
    str_msg = cmprs_img_msg.data
    buf = np.ndarray(shape=(1, len(str_msg)),
                      dtype=np.uint8, buffer=cmprs_img_msg.data)
    im = cv2.imdecode(buf, cv2.IMREAD_ANYCOLOR)

    if desired_encoding == "passthrough":
        return im

    try:
        res = cvtColor2(im, "bgr8", desired_encoding)
    except RuntimeError as e:
        raise CvBridgeError(e)

    return res


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
        try: 
            if self.compressed: 
                im = compressed_imgmsg_to_cv2(msg, self.encoding)
            else: 
                im = self.bridge.imgmsg_to_cv2(msg, self.encoding)
        except CvBridgeError as e:
            raise Exception('ImageDecoder.decode :: {}'.format(e))

        return im_resize(im, scale=self.scale)


class ApproximateTimeSynchronizerBag(ApproximateTimeSynchronizer): 
    """
    See reference implementation in message_filters.__init__.py 
    for ApproximateTimeSynchronizer
    """

    def __init__(self, topics, queue_size, slop): 
        ApproximateTimeSynchronizer.__init__(self, [], queue_size, slop)
        self.queues_ = [{} for f in topics]
        self.topic_queue_ = {topic: self.queues_[ind] for ind, topic in enumerate(topics)}

    def add_topic(self, topic, msg):
        self.add(msg, self.topic_queue_[topic])

# class SensorSynchronizer(object): 
#     def __init__(self, channels, cb_names, decoders, on_synced_cb, slop_seconds=0.1, queue_length=10): 
#         self.channels_ = channels
#         self.cb_name_ = cb_names
#         self.decoders_ = decoders
#         self.slop_seconds_ = slop
#         self.queue_length_ = queue_length

#         self.synch_ = ApproximateTimeSynchronizerBag(self.channels_, queue_length, slop_seconds)
#         self.synch_.registerCallback(self.on_sync)
#         self.on_synced_cb = on_synced_cb

#         for (channel, cb_name) in zip(self.channels_, self.cb_names_): 
#             setattr(self, cb_name, lambda t, msg: self.synch_.add_topic(channel, msg))
#             print('{} :: Registering {} with callback'.format(self.__class__.__name__, cb_name))

#     def on_sync(self, *args): 
#         items = [dec.decoder(msg) for msg, dec in izip(*args, self.decoders_)]
#         return self.on_synced_cb(*items)

# def StereoSynchronizer(left_channel, right_channel, left_cb_name, right_cb_name, on_stereo_cb, 
#                        every_k_frames=1, scale=1., encoding='bgr8', compressed=False): 
#     """
#     Time-synchronized stereo image decoder
#     """
#     channels = [left_channel, right_channel]
#     cb_names = [left_cb_name, right_cb_name]
#     decoders = [ImageDecoder(channel=channel, every_k_frames=every_k_frames, 
#                              scale=scale, encoding=encoding, compressed=compressed)
#                 for channel in channels]
#     return SensorSynchronizer(channels, cb_names, decoders, on_stereo_cb)

# def RGBDSynchronizer(left_channel, right_channel, on_stereo_cb, 
#                  every_k_frames=1, scale=1., encoding='bgr8', compressed=False): 
#     """
#     Time-synchronized RGB-D decoder
#     """
#     channels = [rgb_channel, depth_channel]
#     cb_names = [rgb_cb_name, depth_cb_name]
#     decoders = [ImageDecoder(channel=channel, every_k_frames=every_k_frames, 
#                              scale=scale, encoding=encoding, compressed=compressed)
#                 for channel in channels]
#     return SensorSynchronizer(channels, decoders, on_stereo_cb)

                              
# class StereoSynchronizer(object): 
#     """
#     Time-synchronized stereo image decoder
#     """
#     def __init__(self, left_channel, right_channel, on_stereo_cb, 
#                  every_k_frames=1, scale=1., encoding='bgr8', compressed=False): 

#         self.left_channel = left_channel
#         self.right_channel = right_channel

#         self.decoder = ImageDecoder(every_k_frames=every_k_frames, 
#                                     scale=scale, encoding=encoding, compressed=compressed)

#         slop_seconds = 0.02
#         queue_len = 10 
#         self.synch = ApproximateTimeSynchronizerBag([left_channel, right_channel], queue_len, slop_seconds)
#         self.synch.registerCallback(self.on_stereo_sync)
#         self.on_stereo_cb = on_stereo_cb

#         self.on_left = lambda t, msg: self.synch.add_topic(left_channel, msg)
#         self.on_right = lambda t, msg: self.synch.add_topic(right_channel, msg)
        
#     def on_stereo_sync(self, lmsg, rmsg): 
#         limg = self.decoder.decode(lmsg)
#         rimg = self.decoder.decode(rmsg)
#         return self.on_stereo_cb(limg, rimg)
        
class LaserScanDecoder(Decoder): 
    """
    Mostly stripped from 
    https://github.com/ros-perception/laser_geometry/blob/indigo-devel/src/laser_geometry/laser_geometry.py
    """
    def __init__(self, channel='/scan', every_k_frames=1, range_min=0.0, range_max=np.inf):
        Decoder.__init__(self, channel=channel, every_k_frames=every_k_frames)

        self.angle_min_ = 0.0
        self.angle_max_ = 0.0
        self.range_min_ = range_min
        self.range_max_ = range_max
        self.cos_sin_map_ = np.array([[]])
        
    def decode(self, msg): 
        try:
            N = len(msg.ranges)

            zeros = np.zeros(shape=(N,1))
            ranges = np.array(msg.ranges)
            ranges[ranges < self.range_min_] = np.inf
            ranges[ranges > self.range_max_] = np.inf
            ranges = np.array([ranges, ranges])
            
            if (self.cos_sin_map_.shape[1] != N or
               self.angle_min_ != msg.angle_min or
                self.angle_max_ != msg.angle_max):
                # print("{} :: No precomputed map given. Computing one.".format(self.__class__.__name__))

                self.angle_min_ = msg.angle_min
                self.angle_max_ = msg.angle_max

                cos_map = [np.cos(msg.angle_min + i * msg.angle_increment)
                       for i in range(N)]
                sin_map = [np.sin(msg.angle_min + i * msg.angle_increment)
                        for i in range(N)]

                self.cos_sin_map_ = np.array([cos_map, sin_map])

            return np.hstack([(ranges * self.cos_sin_map_).T, zeros])
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
        self.relations_map_ = {}
        print('-' * 120 + '\n{:}\n'.format(self.log) + '-' * 120)
        
        # # Gazebo states (if available)
        # self._publish_gazebo_states()

    def close(self): 
        print('{} :: Closing log file {}'.format(self.__class__.__name__, self.filename))
        self.log.close()

    def __del__(self): 
        self.log.close()

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
        bag = rosbag.Bag(filename, 'r', chunk_threshold=10 * 1024 * 1024)
        print('{} :: Done loading {} in {:5.2f} seconds'.format(self.__class__.__name__, filename, time.time() - st))
        return bag

    def tf(self, from_tf, to_tf): 
        try: 
            return self.relations_map_[(from_tf, to_tf)]
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
        tf_dec = TfDecoderAndPublisher(channel='/tf')

        # Establish tf relations
        print('{} :: Establishing tfs from ROSBag'.format(self.__class__.__name__))
        for self.idx, (channel, msg, t) in enumerate(self.log.read_messages(topics='/tf')): 
            tf_dec.decode(msg)
            for (from_tf, to_tf) in relations:  

                # If the relations have been added already, skip
                if (from_tf, to_tf) in self.relations_map_: 
                    continue
                    
                # Check if the frames exist yet
                if not tf_listener.frameExists(from_tf) or not tf_listener.frameExists(to_tf): 
                    continue

                # Retrieve the transform with common time
                try:
                    tcommon = tf_listener.getLatestCommonTime(from_tf, to_tf)
                    (trans,rot) = tf_listener.lookupTransform(from_tf, to_tf, tcommon)
                    self.relations_map_[(from_tf,to_tf)] = RigidTransform(tvec=trans, xyzw=rot)
                    # print('\tSuccessfully received transform: {:} => {:} {:}'
                    #       .format(from_tf, to_tf, self.relations_map_[(from_tf,to_tf)]))
                except (tf.LookupException, tf.ConnectivityException, tf.ExtrapolationException) as e:
                    # print e
                    pass

            # Finish up once we've established all the requested tfs
            if len(self.relations_map_) == len(relations): 
                break

        try: 
            tfs = [self.relations_map_[(from_tf,to_tf)] for (from_tf, to_tf) in relations] 
            for (from_tf, to_tf) in relations: 
                print('\tSuccessfully received transform:\n\t\t {:} => {:} {:}'
                      .format(from_tf, to_tf, self.relations_map_[(from_tf,to_tf)]))

        except: 
            raise RuntimeError('Error concerning tf lookup')
        print('{} :: Established {:} relations\n'.format(self.__class__.__name__, len(tfs)))
        
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
        try: 
            if not self.length(topic): 
                raise ValueError('Camera calibration unavailable {}'.format(topic))
        except: 
            raise RuntimeError('Failed to retrieve camera calibration {}, \ntopics are {}\n'.format(topic, ', '.join(info.topics)))
            
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

class BagFrame(object): 
    """
    BagFrame to allow for indexed look up with minimal 
    memory overhead; images are only decoded and held in 
    memory only at request, and not when indexed

    BagFrame: 
       .img [np.arr (in-memory only on request)]
       .pose [RigidTransform]
       .annotation [SUN3DAnntotaionFrame]

    """

    def __init__(self, index, t, img, pose, annotation): 
        self.index_ = index
        self.t_ = t
        self.pose_ = pose
        self.annotation_ = annotation
        self.img_ = img

    @property
    def timestamp(self): 
        return self.t_

    @property
    def pose(self): 
        return self.pose_

    @property
    def is_annotated(self): 
        return self.annotation_.is_annotated

    @property
    def annotation(self): 
        return self.annotation_

    @property
    def img(self): 
        return self.img_

    def __repr__(self): 
        return 'BagFrame::img={}'.format(self.img.shape)


class BagDB(LogDB): 
    def __init__(self, dataset): 
        """
        """
        # Load logdb with ground truth metadata
        # Read annotations from index.json {fn -> annotations}
        try: 
            meta_directory = os.path.expanduser(dataset.filename).replace('.bag','')
            meta = SUN3DAnnotationDB.load(meta_directory, shape=None)
        except Exception, e: 
            raise RuntimeError('Failed to open {}, Error: {}'.format(meta_directory, e))
            meta = None
        LogDB.__init__(self, dataset, meta=meta)

    def _index(self): 
        pass

    def iterframes(self): 
        """
        Ground truth reader interface for Images [with time, pose,
        annotation] : lookup corresponding annotation, and filled in
        poses from nearest available timestamp
        """
        self.check_ground_truth_availability()

        for rgb_idx, (t, ch, data) in enumerate(self.dataset.iteritems(topics=['/camera/rgb/image_raw/compressed_triggered'])):
            frame = BagFrame(rgb_idx, t, data, None, self.annotationdb['rgb/{:08d}.jpg'.format(rgb_idx)]) 
            yield (frame.timestamp, rgb_idx, frame)
                   

        # Iterate through both poses and images, and construct frames
        # with look up table for filename str -> (timestamp, pose, annotation) 

        # SensorSynchronizer(channels=[rgb_channel, depth_channel, odom_channel], 
        #                    cb_names=)

        # odom_data = Accumulator(maxlen=1)
        # rgb_data = Accumulator(maxlen=1)
        # for (t,ch,data) in self.iteritems(): 
        #     if ch == odom_channel: 
        #         odom_data.accumulate(data)
        #     elif ch == rgb_channel: 
        #         rgb_data.accumulate(data)
        #     elif ch == depth_channel: 
        #         depth_data.accumulate(data)
        #     if odom_data.length > 0 and rgb_data.length > 0 and depth_data.length > 0: 
        #         yield

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
        
    #     # Create lut for all topics and corresponding connection index
    #     bag = self.dataset.log
    #     lut = {conn.topic : idx for idx, conn in bag._connections.iteritems()}

    #     # Establish index entries for each of the decoder channels
    #     connection_indexes =  {dec.channel: bag._connection_indexes[lut[dec.channel]]
    #                            for dec in self.dataset.decoder.itervalues()}

    #     for k,v in connection_indexes.iteritems(): 
    #         print k, len(v)

    #     for idx, rgb_item in enumerate(connection_indexes[rgb_channel]): 
    #         frame_index_[idx] = rgb_item.time
            

        # # 1. Iterate through both poses and images, and construct frames
        # # with look up table for filename str -> (timestamp, pose, annotation) 
        # poses = []
        # # pose_decode = lambda msg_item: \
        # #               self.dataset.decoder[pose_channel].decode(msg_item)

        # # Note: Control flow for idx is critical since start_idx could
        # # potentially change the offset and destroy the pose_index
        # for idx, (t, ch, data) in enumerate(self.dataset.iteritems()): 
        #     pose = None
        #     if ch == pose_channel: 
        #         try: 
        #             pose = data
        #         except: 
        #             pose = None
        #     poses.append(pose)

        # # Find valid and missing poses
        # # pose_inds: log_index -> closest_valid_index
        # valid_arr = np.array(
        #     map(lambda item: item is not None, poses), dtype=np.bool)
        # pose_inds = BagDB._nn_pose_fill(valid_arr)

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
    #     # self.check_ground_truth_availability()

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
    #         if self.dataset.is_ground_truth_available else 'Not Available'

    #     # Pretty print IndexDB description 
    #     print('\nTango IndexDB \n========\n'
    #           '\tFrames: {:}\n'
    #           '\tGround Truth: {:}\n'
    #           .format(len(self.frame_index_), gt_str)) 

    # @property
    # def index(self): 
    #     return self.frame_index_

