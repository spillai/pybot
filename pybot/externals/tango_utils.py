"""Tango log reader API"""

# Author: Sudeep Pillai <spillai@csail.mit.edu>
# License: MIT

import cv2
import numpy as np
import os.path
import json

from itertools import izip
from collections import deque, namedtuple, OrderedDict
from abc import ABCMeta, abstractmethod

from pybot.externals.log_utils import Decoder, LogFile, LogReader, LogController, LogDB
from pybot.vision.image_utils import im_resize
from pybot.vision.camera_utils import CameraIntrinsic
from pybot.geometry.rigid_transform import RigidTransform
from pybot.utils.dataset.sun3d_utils import SUN3DAnnotationDB
from pybot.utils.pose_utils import PoseSampler

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
    p_ID = RigidTransform(tvec=[0,0,0], 
                          xyzw=[-0.079740, -0.079740, 0.702596, 0.702596])
    p_IF = RigidTransform(tvec=[0.000662555, 0.011257, 0.0041772], 
                          xyzw=[0.70492326,  0.7092538 , -0.00595375,  0.00259168])
    p_IC = RigidTransform(tvec=[0.000339052, 0.0616911, 0.00279207], 
                          xyzw=[0.707940, 0.706271, 0.001000, 0.000585])
    p_DC = p_ID.inverse() * p_IC
    p_DF = p_ID.inverse() * p_IF
    # print('\nCalibration\n==============')
    # print('\tp_ID: {}, \n\tp_IC: {}, \n\tp_DC: {}, \n\tp_DF: {}'
    #       .format(p_ID, p_IC, p_DC, p_DF))

    # SS->CAM
    p_S_CAM = RigidTransform.from_rpyxyz(
        -np.pi/2, 0, 0, 0, 0, 0, axes='sxyz')
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
        xyz = np.random.normal(0, noise[0], size=3) \
              if noise[0] > 0 else np.zeros(3)
        rpy = np.random.normal(0, noise[1], size=3) \
              if noise[1] > 0 else np.zeros(3)
        return RigidTransform.from_rpyxyz(
            rpy[0], rpy[1], rpy[2], xyz[0], xyz[1], xyz[2])

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
                p21 = get_noise() * \
                      (p_accumulator[-2].inverse() * p_accumulator[-1])
                last = p_accumulator_noisy[-1]
                p_accumulator_noisy.append(last.oplus(p21))

            return p_accumulator_noisy[-1]

        decode_cb = lambda data: \
                odom_decode_with_noise(calibrated_odom_decode(data))

    return Decoder(channel=channel, every_k_frames=every_k_frames, 
                   decode_cb=decode_cb)

class TangoImageDecoder(Decoder): 
    """
    """
    def __init__(self, directory, channel='RGB', color=True, 
                 every_k_frames=1, shape=(1280,720)): 
        Decoder.__init__(self, channel=channel, every_k_frames=every_k_frames)
        self.shape_ = shape 
        if self.shape_[0] < self.shape_[1]: 
            raise RuntimeError('W > H requirement failed, W: {}, H: {}'
                               .format(self.shape_[0], self.shape_[1]))
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

class TangoFile(LogFile): 

    RGB_CHANNEL = 'RGB'
    VIO_CHANNEL = 'RGB_VIO'

    def __init__(self, filename): 
        LogFile.__init__(self, filename)

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

    Calibration: 
        https://developers.google.com/project-tango/apis/c/
        reference/group/camera#group___camera_1ga61f047f290983da9d16522371977cecf

        See /sdcard/config/calibration.xml
        1043.75;   1043.69;   638.797;   357.991;   0.234583;   -0.689864;   0.679871

    """

    def __init__(self, directory, scale=1., start_idx=0, every_k_frames=1, 
                 noise=[0,0], with_ground_truth=False, meta_file='tango_data.txt'): 

        # Set directory and filename for time synchronized log reads 
        self.directory_ = os.path.expanduser(directory)
        self.filename_ = os.path.join(self.directory_, meta_file)
            
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
        # Initialize as a log reader with an associated filename
        # (calls load_log on filename) and appropriate decoders
        super(TangoLogReader, self).__init__(
            self.filename_, 
            decoder=[
                TangoOdomDecoder(channel=TangoFile.VIO_CHANNEL, 
                                 every_k_frames=every_k_frames, 
                                 noise=noise), 
                TangoImageDecoder(
                    self.directory_, channel=TangoFile.RGB_CHANNEL, color=True, 
                    shape=(W,H), every_k_frames=every_k_frames)]
        )

        # Check start index
        if isinstance(self.start_idx_, float):
            raise ValueError('start_idx in TangoReader expects an integer,'
                             'provided {:}'.format(self.start_idx_))


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
        return self.calib_ 

    def load_log(self, filename): 
        return TangoFile(filename)

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

    # def iterframes(self, reverse=False): 
    #     """
    #     Ground truth reader interface for 
    #     Images [with time, pose, annotation]  
    #     : lookup corresponding annotation, and 
    #     fill in poses from previous timestamp
    #     """
    #     # self._check_ground_truth_availability()

    #     # Iterate through both poses and images, and construct frames
    #     # with look up table for filename str -> (timestamp, pose, annotation) 
    #     for (t, channel, msg) in self.itercursors(topics=TangoFile.RGB_CHANNEL, 
    #                                               reverse=reverse): 
    #         try: 
    #             res, (t, ch, data) = self.decode_msg(channel, msg, t)

    #             # Annotations
    #             # Available entries: polygon, bbox, class_label, class_id, instance_id
    #             if res: 
    #                 if self.ground_truth_available: 
    #                     assert(msg in self.meta_)
    #                     yield (t, ch, AnnotatedImage(img=data, annotation=self.meta_[msg]))
    #                 else: 
    #                     yield (t, ch, AnnotatedImage(img=data, annotation=None))

    #         except Exception, e: 
    #             print('TangLog.iteritems() :: {:}'.format(e))


    def iterframes(self, reverse=False): 
        """
        Ground truth reader interface for 
        Images [with time, pose, annotation]  
        : lookup corresponding annotation, and 
        fill in poses from previous timestamp
        """
        
        # Iterate through both poses and images, and construct frames
        # with look up table for filename str -> (timestamp, pose, annotation) 
        for (t, channel, frame) in self.db.iterframes(): 
            yield (t, channel, frame)


    def keyframedb(self, theta=np.deg2rad(20), displacement=0.25, lookup_history=10): 
        sampler = PoseSampler(theta=theta, displacement=displacement, lookup_history=lookup_history, 
                              get_sample=lambda (t, channel, frame): frame.pose, verbose=True)
        self.iterframes = lambda: sampler.iteritems(self.db.iterframes())
        return self

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

            yield (data.img, bboxes, np.int32(map(lambda key: target_hash[key], target_names)))

    @property
    def db(self): 
        return TangoDB(self)

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
        for item in dataset.iterframes(topics=topics): 
            bboxes = item.bboxes
            targets = item.coords


# Define tango frame for known decoders
class TangoFrame(object): 
    """
    TangoFrame to allow for indexed look up with minimal 
    memory overhead; images are only decoded and held in 
    memory only at request, and not when indexed

    TangoFrame: 
       .img [np.arr (in-memory only on request)]
       .pose [RigidTransform]
       .annotation [SUN3DAnntotaionFrame]

    """

    def __init__(self, index, t, img_msg, pose, annotation, img_decode): 
        # print 'should be tangoframe self: ', self
        # print 'img_decoder', img_decode
        self.index_ = index
        self.t_ = t
        self.img_msg_ = img_msg
        self.pose_ = pose
        self.annotation_ = annotation

        self.img_decode = img_decode

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
    def img_filename(self): 
        return self.img_msg_

    @property
    def img(self): 
        """
        Decoded only at request, avoids in-memory storage
        """
        return self.img_decode(self.img_msg_)

    # def __repr__(self): 
    #     return 'img={}, t={}, pose={}'.format(self.img_msg_, self.timestamp, self.pose)

    def __repr__(self): 
        return 'TangoFrame::img={}'.format(self.img_msg_)

class TangoDB(LogDB): 
    def __init__(self, dataset): 
        """
        """
        LogDB.__init__(self, dataset)

    @property
    def poses(self): 
        return [v.pose for k,v in self.frame_index_.iteritems()]
        
    def _index(self, pose_channel=TangoFile.VIO_CHANNEL, rgb_channel=TangoFile.RGB_CHANNEL): 
        """
        Constructs a look up table for the following variables: 
        
            self.frame_index_:  rgb/img.png -> TangoFrame
            self.frame_idx2name_: idx -> rgb/img.png
            self.frame_name2idx_: idx -> rgb/img.png

        where TangoFrame (index_in_the_dataset, timestamp, )
        """

        # 1. Iterate through both poses and images, and construct frames
        # with look up table for filename str -> (timestamp, pose, annotation) 
        poses = []
        pose_decode = lambda msg_item: \
                      self.dataset.decoder[pose_channel].decode(msg_item)

        # Note: Control flow for idx is critical since start_idx could
        # potentially change the offset and destroy the pose_index
        for idx, (t, ch, msg) in enumerate(self.dataset.itercursors()): 
            pose = None
            if ch == pose_channel: 
                try: 
                    pose = pose_decode(msg)
                except: 
                    pose = None
            poses.append(pose)

        # Find valid and missing poses
        # pose_inds: log_index -> closest_valid_index
        valid_arr = np.array(
            map(lambda item: item is not None, poses), dtype=np.bool)
        pose_inds = TangoDB._nn_pose_fill(valid_arr)
        if np.any(pose_inds < 0): 
            print('{} :: TangoDB poses are not fully synchronized, '
                  'skipping few'.format(self.__class__.__name__))

        # Create indexed frames for lookup        
        # self.frame_index_:  rgb/img.png -> TangoFrame
        # self.frame_idx2name_: idx -> rgb/img.png
        # self.frame_name2idx_: rgb/img.png -> idx
        img_decode = lambda msg_item: \
                    self.dataset.decoder[rgb_channel].decode(msg_item)
        self.frame_index_ = OrderedDict([
            (img_msg, TangoFrame(idx, t, img_msg, poses[pose_inds[idx]], 
                                 self.dataset.annotationdb[img_msg], img_decode))
            for idx, (t, ch, img_msg) in enumerate(self.dataset.itercursors()) \
            if ch == rgb_channel and pose_inds[idx] >= 0
        ])
        self.frame_idx2name_ = OrderedDict([
            (idx, k) for idx, k in enumerate(self.frame_index_.keys())
        ])
        self.frame_name2idx_ = OrderedDict([
            (k, idx) for idx, k in enumerate(self.frame_index_.keys())
        ])

    def iterframes(self): 
        """
        Ground truth reader interface for Images [with time, pose,
        annotation] : lookup corresponding annotation, and filled in
        poses from nearest available timestamp
        """
        # self._check_ground_truth_availability()

        # Iterate through both poses and images, and construct frames
        # with look up table for filename str -> (timestamp, pose, annotation) 
        for img_msg, frame in self.frame_index_.iteritems(): 
            yield (frame.timestamp, img_msg, frame)

    def iterframes_indices(self, inds): 
        for ind in inds: 
            img_msg = self.frame_idx2name_[ind]
            frame = self.frame_index_[img_msg]
            yield (frame.timestamp, img_msg, frame)

    def iterframes_range(self, ind_range): 
        assert(isinstance(ind_range, tuple) and len(ind_range) == 2)
        st, end = ind_range
        inds = np.arange(0 if st < 0 else st, 
                         len(self.frame_index_) if end < 0 else end+1)
        return self.iterframes_indices(inds)

    @property
    def annotated_inds(self): 
        return self.dataset.annotationdb.annotated_inds

    @property
    def object_annotations(self): 
        return self.dataset.annotationdb.object_annotations

    @property
    def objects(self): 
        return self.dataset.annotationdb.objects

    def iter_object_annotations(self, target_name=''): 
        frame_keys, polygon_inds = self.dataset.annotationdb.find_object_annotations(target_name)
        for idx, (fkey,pind) in enumerate(izip(frame_keys, polygon_inds)): 
            try: 
                f = self[fkey]
            except KeyError, e: 
                print(e)
                continue
            assert(f.is_annotated)
            yield f, pind
        
    # def list_annotations(self, target_name=None): 
    #     " List of lists"
    #     inds = self.annotated_inds
    #     return [ filter(lambda frame: 
    #                     target_name is None or name is in target_name, 
    #                     self.dataset.annotationdb.iterframes(inds))

    def print_index_info(self): 
        # Retrieve ground truth information
        gt_str = '{} frames annotated ({} total annotations)'\
            .format(self.dataset.annotationdb.num_frame_annotations, 
                    self.dataset.annotationdb.num_annotations) \
            if self.dataset.ground_truth_available else 'Not Available'

        # Pretty print IndexDB description 
        print('\nTango IndexDB \n========\n'
              '\tFrames: {:}\n'
              '\tGround Truth: {:}\n'
              .format(len(self.frame_index_), gt_str)) 
                      

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
        self.on_frame(AnnotatedFrame(img=ann_img.img, pose=pose, 
                                     t_pose=t_pose, t_img=t_img, 
                                     bboxes=ann_img.annotation.bboxes))
        
    def on_rgb(self, t_img, img): 
        if not len(self.__pose_q):
            return
        t_pose, pose = self.__pose_q[-1]
        self.on_frame(AnnotatedFrame(img=img, pose=pose, 
                                     t_pose=t_pose, t_img=t_img, 
                                     bboxes=[]))

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
