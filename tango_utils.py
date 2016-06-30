"""Tango log reader API"""

# Author: Sudeep Pillai <spillai@csail.mit.edu>
# License: MIT

import cv2
import numpy as np
import os.path
import json

from collections import deque, namedtuple, Counter
from heapq import heappush, heappop
from abc import ABCMeta, abstractmethod

# from bot_vision.mapping.pose_utils import PoseAccumulator
from bot_externals.log_utils import Decoder, LogReader, LogController
from bot_vision.image_utils import im_resize
from bot_geometry.rigid_transform import RigidTransform
from bot_vision.camera_utils import CameraIntrinsic
from bot_utils.dataset.sun3d_utils import SUN3DAnnotationDB

# def test_coords(): 
#     IF = RigidTransform(Quaternion.from_wxyz([0.002592, 0.704923, 0.709254, -0.005954]), tvec=[0.000663, 0.011257, 0.004177])
#     ID = RigidTransform(Quaternion.from_wxyz([0.702596, -0.079740, -0.079740, 0.702596]), tvec=[0.000000, 0.000000, 0.000000])
#     IC = RigidTransform(Quaternion.from_wxyz([0.000585, 0.707940, 0.706271, 0.001000]), tvec=[0.000339, 0.061691, 0.002792])
#     DC = ID.inverse() * IC

# Decode odometry
def odom_decode(data): 
    """ 
    Return Start-of-Service (SS) -> Device (D): 
    x, y, z, qx, qy, qz, qw, status_code, confidence
    """
    p = np.float64(data.split(','))

    # If incomplete data or status code is 0
    if len(p) < 8 or p[7] == 0:
        raise Exception('TangoOdomDecoder.odom_decode :: Failed to retreive pose')

    tvec, ori = p[:3], p[3:7]
    return RigidTransform(xyzw=ori, tvec=tvec)
    
def TangoOdomDecoder(channel, every_k_frames=1, noise=[0,0]): 
    """
    https://developers.google.com/project-tango/overview/coordinate-systems

    DYNAMIC base: COORDINATE_FRAME_START_OF_SERVICE (SS), target: COORDINATE_FRAME_DEVICE (D)	
    Reported measurements

    May 2016
    base: IMU, target: FISHEYE
    frame->camera t: 0.000663, 0.011257, 0.004177 q: 0.002592, 0.704923, 0.709254, -0.005954

    base: IMU, target: DEVICE
    frame->camera t: 0.000000, 0.000000, 0.000000 q: 0.702596, -0.079740, -0.079740, 0.702596

    base: IMU, target: CAMERA_COLOR
    frame->camera t: 0.000339, 0.061691, 0.002792 q: 0.000585, 0.707940, 0.706271, 0.001000

    base: SS, target: DEVICE
    -0.005353, -0.000184, -0.004125 q: 0.814851, 0.578699, 0.019265, -0.027478

    Jan 2016
    STATIC base: IMU (I), target: CAMERA (C)
    t: 0.000339, 0.061691, 0.002792 q: (w) 0.000585, (x)0.707940, (y)0.706271, (z)0.001000

    STATIC base: IMU (I), target: DEVICE (D)
    t: 0.000000, 0.000000, 0.000000 q: (w) 0.702596, (x) -0.079740, (y) -0.079740, (z) 0.702596

    STATIC base: IMU (I), target: DEPTH (P)
    t: 0.000339, 0.061691, 0.002792 q: (w) 0.000585, 0.707940, 0.706271, 0.001000

    STATIC base: IMU (I), target: FISHEYE (F)
    t: 0.000663, 0.011257, 0.004177 q: (w) 0.002592, 0.704923, 0.709254, -0.005954

    BOT: FWD  (X), LEFT (Y), UP  (Z)
    SS:  LEFT (X), FWD  (Y), UP  (Z)
    CAM: LEFT (X), DOWN (Y), FWD (Z)
 
    BOT->CAM: 
    a) (0, 0, -90)   => LEFT (X), FWD  (Y), UP   (Z)
    b) (-90, 0, -90) => LEFT (X), DOWN (Y), FWD (Z) => CAM

    SS->CAM: 
    a) (-90, 0, 0)   => LEFT (X), DOWN (Y), FWD (Z) => CAM

    """
    # IF = RigidTransform(Quaternion.from_wxyz([0.002592, 0.704923, 0.709254, -0.005954]), tvec=[0.000663, 0.011257, 0.004177])
    # ID = RigidTransform(Quaternion.from_wxyz([0.702596, -0.079740, -0.079740, 0.702596]), tvec=[0.000000, 0.000000, 0.000000])
    # IC = RigidTransform(Quaternion.from_wxyz([0.000585, 0.707940, 0.706271, 0.001000]), tvec=[0.000339, 0.061691, 0.002792])


    # p_ID = RigidTransform(tvec=[0,0,0], xyzw=[-0.079740, -0.079740, 0.706271, 0.706271]) # Jan 2016
    p_ID = RigidTransform(tvec=[0,0,0], xyzw=[-0.079740, -0.079740, 0.702596, 0.702596]) # May 2016
    p_IF = RigidTransform(tvec=[0.000662555, 0.011257, 0.0041772], xyzw=[0.70492326,  0.7092538 , -0.00595375,  0.00259168])
    p_IC = RigidTransform(tvec=[0.000339052, 0.0616911, 0.00279207], xyzw=[0.707940, 0.706271, 0.001000, 0.000585])
    p_DC = p_ID.inverse() * p_IC
    p_DF = p_ID.inverse() * p_IF

    # print('\nCalibration\n==============')
    # print('\tp_ID: {}, \n\tp_IC: {}, \n\tp_DC: {}, \n\tp_DF: {}'.format(p_ID, p_IC, p_DC, p_DF))

    # SS->CAM
    p_S_CAM = RigidTransform.from_roll_pitch_yaw_x_y_z(-np.pi/2, 0, 0, 
                                                    0, 0, 0, axes='sxyz')
    p_CAM_S = p_S_CAM.inverse()

    # Decode odometry
    def calibrated_odom_decode(data): 
        p_SD = odom_decode(data)
        p_SC = p_SD * p_DC
        return p_CAM_S * p_SC
    decode_cb = lambda data: calibrated_odom_decode(data)

    np.random.seed(1)
    noise = np.float32(noise)
    def get_noise(): 
        xyz = np.random.normal(0, noise[0], size=3) if noise[0] > 0 else np.zeros(3)
        rpy = np.random.normal(0, noise[1], size=3) if noise[1] > 0 else np.zeros(3)
        return RigidTransform.from_roll_pitch_yaw_x_y_z(rpy[0], rpy[1], rpy[2], xyz[0], xyz[1], xyz[2])

    # Noise injection
    inject_noise = (noise[0] > 0 or noise[1] > 0)
    if inject_noise: 
        p_accumulator = deque(maxlen=2)
        p_accumulator_noisy = deque(maxlen=2)
        def odom_decode_with_noise(data): 
            p_accumulator.append(data)
            if len(p_accumulator) == 1:
                p_accumulator_noisy.append(p_accumulator[-1])
            else: 
                p21 = get_noise() * (p_accumulator[-2].inverse() * p_accumulator[-1])
                last = p_accumulator_noisy[-1]
                p_accumulator_noisy.append(last.oplus(p21))

            return p_accumulator_noisy[-1]

        decode_cb = lambda data: odom_decode_with_noise(calibrated_odom_decode(data))

    return Decoder(channel=channel, every_k_frames=every_k_frames, decode_cb=decode_cb)

class TangoImageDecoder(Decoder): 
    """
    """
    def __init__(self, directory, channel='RGB', color=True, every_k_frames=1, shape=(1280,720)): 
        Decoder.__init__(self, channel=channel, every_k_frames=every_k_frames)
        self.shape_ = shape 
        if self.shape_[0] < self.shape_[1]: 
            raise RuntimeError('W > H requirement failed, W: {}, H: {}'.format(self.shape_[0], self.shape_[1]))
        self.directory_ = directory
        self.color_ = color

    def decode(self, msg): 
        fn = os.path.join(self.directory_, msg)
        if os.path.exists(fn): 
            im = cv2.imread(fn, 
                            cv2.CV_LOAD_IMAGE_COLOR if self.color_ \
                            else cv2.CV_LOAD_IMAGE_GRAYSCALE)
            return im_resize(im, shape=self.shape_)
        else: 
            raise Exception('File does not exist')


# Basic type for image annotations
AnnotatedImage = namedtuple('AnnotatedImage', ['img', 'annotation'])

class TangoFile(object): 

    RGB_CHANNEL = 'RGB'
    VIO_CHANNEL = 'RGB_VIO'

    def __init__(self, filename): 
        self.filename_ = filename

        # Save topics and counts
        ts, topics = self._get_stats()
        c = Counter(topics)
        self.topics_ = list(set(topics))
        self.topic_lengths_ = dict(c.items())
        self.length_ = sum(self.topic_lengths_.values())

        # Get distance traveled (accumulate relative motion)
        distance = self._get_distance_travelled()
        
        messages_str = ', '.join(['{:} ({:})'.format(k,v) 
                                  for k,v in c.iteritems()])
        print('\nTangoFile \n========\n'
              '\tFile: {:}\n'
              '\tTopics: {:}\n'
              '\tMessages: {:}\n'
              '\tDistance Travelled: {:.2f} m\n'
              '\tDuration: {:} s\n'.format(
                  self.filename_, 
                  self.topics_, messages_str, 
                  distance, np.max(ts)-np.min(ts)))
        
    def _get_distance_travelled(self): 
        " Retrieve distance traveled through relative motion "

        prev_pose, tvec = None, 0
        for (_,pose_str,_) in self.read_messages(topics=TangoFile.VIO_CHANNEL): 
            try: 
                pose = odom_decode(pose_str)
            except: 
                continue

            if prev_pose is not None: 
                tvec += np.linalg.norm(prev_pose.tvec-pose.tvec)

            prev_pose = pose

        return tvec

    def _get_stats(self): 
        # Get stats
        # Determine topics that have at least 3 items (timestamp,
        # channel, data) separated by tabs
        with open(self.filename, 'r') as f: 
            data = filter(lambda ch: len(ch) == 3, 
                          map(lambda l: l.replace('\n','').split('\t'), 
                              filter(lambda l: '\n' in l, f.readlines())))

            ts = map(lambda (t,ch, data): float(t) * 1e-9, data)
            topics = map(lambda (t,ch,data): ch, data)

        return ts, topics

    @property
    def filename(self): 
        return self.filename_

    @property
    def length(self): 
        return self.length_

    @property
    def fd(self): 
        """ Open the tango meta data file as a file descriptor """
        return open(self.filename, 'r')
        
    def read_messages(self, topics=[], start_time=0): 
        """
        Read messages with a heap so that the measurements are monotonic, 
        decoded iteratively (or when needed).
        """
        N = 1000
        heap = []
        
        if isinstance(topics, str): 
            topics = [topics]

        topics_set = set(topics)

        # Read messages in ascending order of timestamps
        # Push messages onto the heap and pop such that 
        # the order of timestamps is ensured to be increasing.
        p_t = 0
        for l in self.fd: 
            try: 
                t, ch, data = l.replace('\n', '').split('\t')
            except: 
                continue

            if len(topics_set) and ch not in topics_set: 
                continue

            if len(heap) == N: 
                c_t, c_ch, c_data = heappop(heap)
                # Check monotononic measurements
                assert(c_t >= p_t)
                p_t = c_t
                yield c_ch, c_data, c_t
            
            heappush(heap, (int(t), ch, data))

        # Pop the rest of the heap
        for j in range(len(heap)): 
            c_t, c_ch, c_data = heappop(heap)
            # Check monotononic measurements
            assert(c_t >= p_t)
            p_t = c_t
            yield c_ch, c_data, c_t

class TangoLogReader(LogReader): 

    cam = CameraIntrinsic(
        K=np.float64([1043.75, 0, 638.797, 0, 1043.75, 357.991, 0, 0, 1]).reshape(3,3), 
        D=np.float64([0.234583, -0.689864, 0, 0, 0.679871]), 
        shape=(720, 1280)
    )

    # fisheye_cam = CameraIntrinsic(
    #     K=np.float64([256.207, 0, 326.606, 0, 256.279, 244.754, 0, 0, 1]).reshape(3,3), 
    #     D=np.float64([0.234583, -0.689864, 0, 0, 0.679871]), 
    #     0.925577
    #     shape=(480, 640)
    # )
    """
    TangoLogReader that uses TangoFile as a log to read and decode
    text strings from tango_data.txt and convert to appropriate 
    objects for log reading. 

    TODO: 
    1. Support for every_k_frames in iteritems
    """

    # fisheye_cam = CameraIntrinsic(K=)
    def __init__(self, directory, scale=1., start_idx=0, every_k_frames=1, 
                 noise=[0,0], with_ground_truth=False): 

        # Set directory and filename for time synchronized log reads 
        self.directory_ = os.path.expanduser(directory)
        self.filename_ = os.path.join(self.directory_, 'tango_data.txt')
            
        self.scale_ = scale
        self.calib_ = TangoLogReader.cam.scaled(self.scale_)
        self.shape_ = self.calib_.shape

        assert(self.shape_[0] % 2 == 0 and self.shape_[1] % 2 == 0)
        self.start_idx_ = start_idx

        # Initialize TangoLogReader with appropriate decoders
        H, W = self.shape_

        # Load ground truth filename
        # Read annotations from index.json {fn -> annotations}
        if with_ground_truth: 
            self.meta_ = SUN3DAnnotationDB.load(self.directory_, shape=(W,H))
            print('\nGround Truth\n========\n'
                  '\tAnnotations: {}\n'
                  '\tObjects: {}'.format(self.meta_.num_annotations, 
                                         self.meta_.num_objects))
        else: 
            self.meta_ = None

        # Setup log (calls load_log, and initializes decoders)
        pose_decoder = TangoOdomDecoder(channel=TangoFile.VIO_CHANNEL, 
                                        every_k_frames=every_k_frames, 
                                        noise=noise)
        img_decoder = TangoImageDecoder(
            self.directory_, channel=TangoFile.RGB_CHANNEL, color=True, 
            shape=(W,H), every_k_frames=every_k_frames)

        super(TangoLogReader, self).\__init__(
            self.filename_, decoder=[pose_decoder, img_decoder]
        )
        
        # Check start index
        if isinstance(self.start_idx_, float):
            raise ValueError('start_idx in TangoReader expects an integer,'
                             'provided {:}'.format(self.start_idx_))


        # Define tango frame for known decoders
        class TangoFrame(object): 
            """
            TangoFrame to allow for indexed look up with minimal 
            memory overhead; images are only decoded and held in 
            memory only at request, and not when indexed
            """

            def __init__(self, t, img_msg, pose_msg, annotation): 
                print 'should be tangoframe self: ', self
                print 'pose_decoder', pose_decoder
                print 'img_decoder', img_decoder

                self.t_ = t
                self.img_msg_ = img_msg
                self.pose_ = pose_decoder.decode(pose_msg)
                self.annotation_ = annotation

            @property
            def timestamp(self): 
                return self.t_

            @property
            def pose(self): 
                return self.pose_

            @property
            def annotation(self): 
                return self.annotation_

            @property
            def img(self): 
                """
                Decoded only at request, avoids in-memory storage
                """
                return img_decoder.decode(self.img_msg_)

    @property
    def annotationdb(self): 
        return self.meta_

    @property
    def ground_truth_available(self): 
        return self.meta_ is not None

    def _check_ground_truth_availability(self):
        if not self.ground_truth_available: 
            raise RuntimeError('Ground truth dataset not loaded')

    @property
    def directory(self): 
        return self.directory_

    @property
    def calib(self): 
        """
        https://developers.google.com/project-tango/apis/c/
        reference/group/camera#group___camera_1ga61f047f290983da9d16522371977cecf

        See /sdcard/config/calibration.xml
        1043.75;   1043.69;   638.797;   357.991;   0.234583;   -0.689864;   0.679871
        """
        return self.calib_ 

    def load_log(self, filename): 
        return TangoFile(filename)
                
    def decode_msg(self, channel, msg, t): 
        try: 
            # Check if log index has reached desired start index, 
            # and only then check if decode necessary  
            dec = self.decoder[channel]
            if dec.should_decode():
                return True, (t, channel, dec.decode(msg))
        except Exception as e:
            # print e
            # raise RuntimeError('Failed to decode data from channel: %s, mis-specified decoder?' % channel)
            pass
        
        return False, (None, None, None)

    def itercursors(self, topics=[], reverse=False): 
        if self.index is not None: 
            raise NotImplementedError('Cannot provide items indexed')
        
        if reverse: 
            raise NotImplementedError('Cannot provide items in reverse when file is not indexed')

        # Decode only messages that are supposed to be decoded 
        print('Reading TangoFile from index={:} onwards'.format(self.start_idx_))
        for self.idx, (channel, msg, t) in enumerate(self.log.read_messages(topics=topics)):
            if self.idx < self.start_idx_: 
                continue
            yield (t, channel, msg)

    def iteritems(self, topics=[], reverse=False): 
        for (t, channel, msg) in self.itercursors(topics=topics, reverse=reverse): 
            try: 
                res, (t, ch, data) = self.decode_msg(channel, msg, t)
                if res: 
                    yield (t, ch, data)
            except Exception, e: 
                print('TangLog.iteritems() :: {:}'.format(e))

    def iterframes(self, reverse=False): 
        """
        Ground truth reader interface for 
        Images [with time, pose, annotation]  
        : lookup corresponding annotation, and 
        fill in poses from previous timestamp
        """
        self._check_ground_truth_availability()

        # Iterate through both poses and images, and construct frames
        # with look up table for filename str -> (timestamp, pose, annotation) 
        for (t, channel, msg) in self.itercursors(topics=TangoFile.RGB_CHANNEL, 
reverse=reverse): 
            try: 
                res, (t, ch, data) = self.decode_msg(channel, msg, t)

                # Annotations
                # Available entries: polygon, bbox, class_label, class_id, instance_id
                if res: 
                    assert(msg in self.meta_)
                    yield (t, ch, AnnotatedImage(img=data, annotation=self.meta_[msg]))

            except Exception, e: 
                print('TangLog.iteritems() :: {:}'.format(e))

    def roidb(self, target_hash, targets=[], every_k_frames=1, verbose=True, skip_empty=True): 
        """
        Returns (img, bbox, targets [unique text])
        """

        self._check_ground_truth_availability()

        if every_k_frames > 1 and skip_empty: 
            raise RuntimeError('roidb not meant for skipping frames,'
                               'and skipping empty simultaneously ')

        # Iterate through all images
        for idx, (t,ch,data) in enumerate(self.iterframes()): 

            # Skip every k frames, if requested
            if idx % every_k_frames != 0: 
                continue

            # Annotations may be empty, if 
            # unlabeled, however we can request
            # to yield if its empty or not
            bboxes = data.annotation.bboxes
            if not len(bboxes) and skip_empty: 
                continue
            target_names = data.annotation.pretty_names

            if len(targets): 
                inds, = np.where([np.any([t in name for t in targets]) for name in target_names])

                target_names = [target_names[ind] for ind in inds]
                bboxes = bboxes[inds]

            print target_names, bboxes.shape
            yield data.img, bboxes, np.int32(map(lambda key: target_hash[key], target_names))

    @property
    def annotated_indices(self): 
        assert(self.ground_truth_available)
        assert(self.start_idx_ == 0)
        inds, = np.where(self.meta_.annotation_sizes)
        return inds

def iter_tango_logs(directory, logs, topics=[]):
    for log in logs: 
        directory = os.path.expanduser(os.path.join(args.directory, log))
        print('Accessing Tango directory {:}'.format(directory))
        dataset = TangoLogReader(directory=directory, scale=im_scale) 
        for item in dataset.iter_frames(topics=topics): 
            bboxes = item.bboxes
            targets = item.coords

class LogDB(object): 
    def __init__(self, dataset): 
        self.dataset_ = dataset
        self.frame_index_ = None
        self._index()
        self.print_index_info()

    def _index(self): 
        raise NotImplementedError()

    def print_index_info(): 
        raise NotImplementedError()

    @property
    def dataset(self): 
        return self.dataset_

class TangoDB(LogDB): 
    def __init__(self, dataset): 
        """
        TangoFrame: 
           .img [np.arr (in-memory only on request)]
           .pose [RigidTransform]
           .annotation [SUN3DAnntotaionFrame]
        """
        LogDB.__init__(self, dataset)

    @property
    def index(self): 
        return self.frame_index_

    def _index(self): 

        # Iterate through both poses and images, and construct frames
        # with look up table for filename str -> (timestamp, pose, annotation) 
        pose_msgs = [msg if ch == TangoFile.VIO_CHANNEL else None\
                     for idx, (t, ch, msg) in enumerate(self.dataset.itercursors())]

        # Find valid and missing poses
        valid_arr = np.array(
            map(lambda item: item is not None, pose_msgs), dtype=np.bool)
        pose_inds = TangoDB._pose_index(valid_arr)

        # Create indexed frames for lookup        
        self.frame_index_ = [
            TangoFrame(t, img_msg, pose_msgs[pose_inds[idx]], 
                       dataset.annotationdb[img_msg]) \
            for idx, (t, ch, img_msg) in enumerate(self.dataset.itercursors()) \
                                    if ch == TangoFile.RGB_CHANNEL]
        assert(dataset.num_frames == len(self.frame_index_))


    def find_annotated_inds(self): 
        " Select all frames that are annotated "
        inds, = np.where(self.dataset.annotationdb.annotation_sizes > 0)
        return inds

    def list_annotations(self, target_name=None): 
        " List of lists"
        inds = self.find_annotated_inds()
        filtered_names = 
        return [ filter(
            lambda name: 
            target_name is None or name is in target_name, 
            self.dataset.annotationdb[ind].pretty_names) for ind in inds ]

    def print_index_info(self): 
        # Retrieve ground truth information
        gt_str = '{} frames annotated ({} total annotations)'
        .format(self.dataset.annotationdb.num_frame_annotations, 
                self.dataset.annotationdb.num_annotations) \
            if self.dataset.ground_truth_available else 'Not Available'

        # Pretty print IndexDB description 
        print('\nTango IndexDB \n========\n'
              '\tFrames: {:}\n'
              '\tPoses: {:}\n'
              '\tGround Truth: {:}\n'
              .format(len(self.frame_index_), 
                      len(pose_msgs), gt_str)) 
                      

    @staticmethod
    def _pose_index(valid): 
        """
        Looks up closest True for each False and returns
        indices for fill-in-lookup
        In: [True, False, True, ... , False, True]
        Out: [0, 0, 2, ..., 212, 212]
        """
        
        valid_inds,  = np.where(valid)
        invalid_inds,  = np.where(~valid)

        all_inds = np.arange(len(valid))
        all_inds[invalid_inds] = -1

        for j in range(10): 
            fwd_inds = valid_inds + j
            bwd_inds = valid_inds - j

            # Forward fill
            invalid_inds, = np.where(all_inds < 0)
            fwd_fill_inds = np.intersect1d(fwd_inds, invalid_inds)
            all_inds[fwd_fill_inds] = all_inds[fwd_fill_inds-j]

            # Backward fill
            invalid_inds, = np.where(all_inds < 0)
            if not len(invalid_inds): break
            bwd_fill_inds = np.intersect1d(bwd_inds, invalid_inds)
            all_inds[bwd_fill_inds] = all_inds[bwd_fill_inds+j]

            # Check if any missing 
            invalid_inds, = np.where(all_inds < 0)
            if not len(invalid_inds): break

        # np.set_printoptions(threshold=np.nan)

        # print valid.astype(np.int)
        # print np.array_str(all_inds)
        # print np.where(all_inds < 0)

        return all_inds

    @property
    def index(self): 
        return self.frame_index_


    # def _build_graph(self): 
    #     # Keep a queue of finite length to ensure 
    #     # time-sync with RGB and IMU
    #     self.__pose_q = deque(maxlen=10)

    #     self.nodes_ = []

    #     for (t,ch,data) in self.dataset_.itercursors(topics=[]): 
    #         if ch == TangoFile.VIO_CHANNEL: 
    #             self.__pose_q.append(data)
    #             continue
            
    #         if not len(self.__pose_q): 
    #             continue

    #         assert(ch == TangoFile.RGB_CHANNEL)
    #         self.nodes_.append(dict(img=data, pose=self.__pose_q[-1]))
            

# Basic type for tango frame (includes pose, image, timestamp)
Frame = namedtuple('Frame', ['img', 'pose', 't_pose', 't_img'])
AnnotatedFrame = namedtuple('AnnotatedFrame', ['img', 'pose', 't_pose', 't_img', 'bboxes'])

class TangoLogController(LogController): 
    def __init__(self, dataset): 
        super(TangoLogController, self).__init__(dataset)

        print('\nSubscriptions\n==============')
        if not self.controller.ground_truth_available: 
            self.subscribe(TangoFile.RGB_CHANNEL, self.on_rgb)
        else: 
            print('\tGround Truth available, subscribe to LogController.on_rgb_gt')
            self.subscribe(TangoFile.RGB_CHANNEL, self.on_rgb_gt)

        self.subscribe(TangoFile.VIO_CHANNEL, self.on_pose)
        print('')

        # Keep a queue of finite lenght to ensure 
        # time-sync with RGB and IMU
        self.__pose_q = deque(maxlen=10)

    def on_rgb_gt(self, t_img, ann_img): 
        if not len(self.__pose_q):
            return
        t_pose, pose = self.__pose_q[-1]
        self.on_frame(AnnotatedFrame(img=ann_img.img, pose=pose, t_pose=t_pose, t_img=t_img, bboxes=ann_img.annotation.bboxes))
        
    def on_rgb(self, t_img, img): 
        if not len(self.__pose_q):
            return
        t_pose, pose = self.__pose_q[-1]
        self.on_frame(AnnotatedFrame(img=img, pose=pose, t_pose=t_pose, t_img=t_img, bboxes=[]))

    def on_pose(self, t, pose): 
        # self.__item_q.append((1, t, pose))
        self.__pose_q.append((t,pose))

        # # If RGB_VIO, RGB, RGB_VIO in stream, then interpolate pose
        # # b/w the 1st and 3rd timestamps to match RGB timestamps
        # if len(self.__item_q) >= 3 and \
        #    self.__item_q[-1][0] == self.__item_q[-3][0] == 1 and \
        #    self.__item_q[-2][0] == 0: 
        #     t1,t2,t3 = self.__item_q[-3][1], self.__item_q[-2][1], self.__item_q[-1][1]
        #     w2, w1 = np.float32([t2-t1, t3-t2]) / (t3-t1)
        #     p1,p3 = self.__item_q[-3][2], self.__item_q[-1][2]
        #     p2 = p1.interpolate(p3, w1)
        #     self.on_frame(t2, t2, p2, self.__item_q[-2][2])
        #     print np.array_str(np.float64([t1, t2, t3]) * 1e-14, precision=6, suppress_small=True), \
        #         (t2-t1) * 1e-6, (t3-t2) * 1e-6, w1, w2, p2

    def on_frame(self, frame): 
        raise NotImplementedError()

    # @abstractmethod
    # def on_rgb(self, t, img): 
    #     raise NotImplementedError()

    # @abstractmethod
    # def on_pose(self, t, pose): 
    #     raise NotImplementedError()
