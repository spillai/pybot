"""Tango log reader API"""

# Author: Sudeep Pillai <spillai@csail.mit.edu>
# License: MIT

import numpy as np
import cv2
import os.path
import json
from collections import deque, namedtuple
from heapq import heappush, heappop
from abc import ABCMeta, abstractmethod

from collections import Counter, deque
from bot_externals.log_utils import Decoder, LogReader, LogController
from bot_vision.image_utils import im_resize
from bot_geometry.rigid_transform import RigidTransform
from bot_vision.camera_utils import CameraIntrinsic

def TangoOdomDecoder(channel, every_k_frames=1, noise=[0,0]): 
    """
    https://developers.google.com/project-tango/overview/coordinate-systems

    DYNAMIC base: COORDINATE_FRAME_START_OF_SERVICE (SS), target: COORDINATE_FRAME_DEVICE (D)	
    Reported measurements

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

    p_IF = RigidTransform(tvec=[0.000663, 0.011257, 0.004177], xyzw=[0.704923, 0.709254, -0.005954, 0.002592])
    p_ID = RigidTransform(tvec=[0,0,0], xyzw=[-0.079740, -0.079740, 0.706271, 0.706271])
    p_IC = RigidTransform(tvec=[0.000339, 0.061691, 0.002792], xyzw=[0.707940, 0.706271, 0.001000, 0.000585])
    p_DC = p_ID.inverse() * p_IC
    p_DF = p_ID.inverse() * p_IF
    print 'p_ID: %s, \np_IC: %s, \np_DC: %s, \np_DF: %s' % (p_ID, p_IC, p_DC, p_DF)

    # SS->CAM
    p_S_CAM = RigidTransform.from_roll_pitch_yaw_x_y_z(-np.pi/2, 0, 0, 
                                                    0, 0, 0, axes='sxyz')
    p_CAM_S = p_S_CAM.inverse()

    # Decode odometry
    def odom_decode(data): 
        """ x, y, z, qx, qy, qz, qw, status_code, confidence """
        p = np.float64(data.split(','))

        # If incomplete data or status code is 0
        if len(p) < 8 or p[7] == 0:
            raise Exception('TangoOdomDecoder.odom_decode :: Failed to retreive pose')

        tvec, ori = p[:3], p[3:7]
        p_SD = RigidTransform(xyzw=ori, tvec=tvec)
        p_SC = p_SD * p_DC
        return p_CAM_S * p_SC
    decode_cb = lambda data: odom_decode(data)

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
                p_accumulator_noisy.append(last * p21)

            return p_accumulator_noisy[-1]

        decode_cb = lambda data: odom_decode_with_noise(odom_decode(data))

    return Decoder(channel=channel, every_k_frames=every_k_frames, decode_cb=decode_cb)

class TangoImageDecoder(Decoder): 
    """
    """
    def __init__(self, directory, channel='RGB', color=True, every_k_frames=1, shape=(720,1280)): 
        Decoder.__init__(self, channel=channel, every_k_frames=every_k_frames)
        self.shape_ = shape 
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
AnnotatedImage = namedtuple('AnnotatedImage', ['im', 'bboxes'])

class TangoGroundTruthImageDecoder(TangoImageDecoder):
    def __init__(self, directory, filename, channel='RGB', color=True, every_k_frames=1, shape=(720,1280)): 
        self.filename_ = os.path.join(directory, filename)
        if not os.path.exists(self.filename_): 
            raise IOError('Cannot load ground truth file {:}'.format(self.filename_))
        
        TangoImageDecoder.__init__(self, directory, channel=channel, color=color, 
                                   every_k_frames=every_k_frames, shape=shape)

        # Look up dictionary {fn -> annotations}
        self.meta_ = {}

        # Read annotations from index.json
        with open(self.filename_) as f: 
            data = json.loads(f.read())
            meta = {fn: frame for (fn, frame) in zip(data['fileList'], data['frames'])}

            target_hash = {label['name']: lid for (lid,label) in enumerate(data['objects'])}
            target_unhash = {lid: label['name'] for (lid,label) in enumerate(data['objects'])}

            # print meta

            # Establish all annotations with appropriate class labels and instances
            for fn, val in meta.iteritems(): 
                try: 
                    polygons = val['polygon']
                except: 
                    continue

                print polygons
                    
                annotations = []
                for poly in polygons:
                    xy = np.vstack([np.float32(poly['x']), np.float32(poly['y'])]).T

                    # Object label as described by target hash
                    lid = poly['object']

                    # Trailing number after hyphen is instance id
                    label = ''.join(target_unhash[lid].split('-')[:-1])
                    instance_id = int(target_unhash[lid].split('-')[-1])
                    annotations.append(
                        dict(polygon=xy, class_label=label, class_id=None, instance_id=instance_id)
                    )
                self.meta_[str(fn)] = annotations

        # unique_objects = set()
        # for k,v in self.meta_iteritems():

        print('\nGround Truth\n========\n'
              '\tAnnotations: {:}\n'.format(len(self.meta_)))
        print self.meta_

    def decode(self, msg): 
        """
        Look up annotations based on basename of image
        """
        im = TangoImageDecoder.decode(self, msg)
        basename = os.path.basename(msg)
        # print('Retrieving annotations for {:}'.format(basename))
        try: 
            bboxes = self.meta_[basename]
        except: 
            bboxes = []
        return AnnotatedImage(im=im, bboxes=bboxes)

class TangoLog(object): 
    def __init__(self, filename): 

        # Determine topics that have at least 3 items (timestamp,
        # channel, data) separated by tabs
        with open(filename, 'r') as f: 
            data = filter(lambda ch: len(ch) == 3, 
                          map(lambda l: l.replace('\n','').split('\t'), 
                              filter(lambda l: '\n' in l, f.readlines())))

            ts = map(lambda (t,ch, data): float(t) * 1e-9, data)
            topics = map(lambda (t,ch,data): ch, data)
            
        # Save topics and counts
        c = Counter(topics)
        self.topics_ = list(set(topics))
        self.topic_lengths_ = dict(c.items())
        self.length_ = sum(self.topic_lengths_.values())

        messages_str = ', '.join(['{:} ({:})'.format(k,v) for k,v in c.iteritems()])
        print('\nTangoLog\n========\n'
              '\tTopics: {:}\n'
              '\tMessages: {:}\n'
              '\tDuration: {:} s\n'.format(self.topics_, messages_str, np.max(ts)-np.min(ts)))

        # Open the tango meta data file
        self.meta_ =  open(filename, 'r')

    @property
    def length(self): 
        return self.length_

    def read_messages(self, topics=None, start_time=0): 

        N = 10000
        heap = []

        # Read messages in ascending order of timestamps
        # Push messages onto the heap and pop such that 
        # the order of timestamps is ensured to be increasing.
        p_t = 0
        for l in self.meta_:
            try: 
                t, ch, data = l.replace('\n', '').split('\t')
            except: 
                continue

            if len(heap) == N: 
                c_t, c_ch, c_data = heappop(heap)
                assert(c_t >= p_t)
                p_t = c_t
                yield c_ch, c_data, c_t
            
            heappush(heap, (int(t), ch, data))

        # Pop the rest of the heap
        for j in range(len(heap)): 
            c_t, c_ch, c_data = heappop(heap)
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
        if with_ground_truth: 
            im_dec = TangoGroundTruthImageDecoder(self.directory_, filename='annotation/index.json', 
                                                   channel='RGB', color=True, 
                                                   shape=(W,H), every_k_frames=every_k_frames)
        else: 
            im_dec = TangoImageDecoder(self.directory_, channel='RGB', color=True, 
                                       shape=(W,H), every_k_frames=every_k_frames)

        # Setup log (calls load_log, and initializes decoders)
        super(TangoLogReader, self).__init__(
            self.filename_, 
            decoder=[
                TangoOdomDecoder(channel='RGB_VIO', every_k_frames=every_k_frames, noise=noise), 
                im_dec
            ])
        
        if isinstance(self.start_idx_, float):
            raise ValueError('start_idx in TangoReader expects an integer, provided {:}'.format(self.start_idx_))

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
        return TangoLog(filename)

    def iteritems(self, reverse=False): 
        if self.index is not None: 
            raise NotImplementedError('Cannot provide items indexed')
        
        if reverse: 
            raise RuntimeError('Cannot provide items in reverse when file is not indexed')

        # Decode only messages that are supposed to be decoded 
        print('Reading TangoLog from index={:} onwards'.format(self.start_idx_))
        for self.idx, (channel, msg, t) in enumerate(self.log.read_messages()):
            if self.idx < self.start_idx_: 
                continue
            try: 
                res, (t, ch, data) = self.decode_msg(channel, msg, t)
                if res: 
                    yield (t, ch, data)
            except Exception, e: 
                print('TangLog.iteritems() :: {:}'.format(e))
                pass


    def iterframes(self, reverse=False): 
        """
        Ground truth reader interface
        Overload decode msg to lookup corresponding annotation
        """
        if self.index is not None: 
            raise NotImplementedError('Cannot provide items indexed')

        if reverse: 
            raise RuntimeError('Cannot provide items in reverse when file is not indexed')

        if not hasattr(self, 'gt_'): 
            raise RuntimeError('Cannot iterate, ground truth dataset not loaded')

        # Decode only messages that are supposed to be decoded 
        print('Reading TangoLog from index={:} onwards'.format(self.start_idx_))
        for self.idx, (channel, msg, t) in enumerate(self.log.read_messages()):
            if self.idx < self.start_idx_: 
                continue
            try: 
                res, (t, ch, data) = self.decode_msg(channel, msg, t)
                if res: 
                    yield (t, ch, data)
            except Exception, e: 
                print('TangLog.iteritems() :: {:}'.format(e))
                pass
                
    def decode_msg(self, channel, msg, t): 
        try: 
            # Check if log index has reached desired start index, 
            # and only then check if decode necessary  
            dec = self.decoder[channel]
            if dec.should_decode():
                return True, (t, channel, dec.decode(msg))
        except Exception as e:
            # pass
            print e
            raise RuntimeError('Failed to decode data from channel: %s, mis-specified decoder?' % channel)
        
        return False, (None, None, None)

    def iter_frames(self):
        return self.iteritems()

def iter_tango_logs(directory, logs):
    for log in logs: 
        directory = os.path.expanduser(os.path.join(args.directory, log))
        print('Accessing Tango directory {:}'.format(directory))
        dataset = TangoLogReader(directory=directory, scale=im_scale) 
        for item in dataset.iter_frames(): 
            yield item

class TangoLogController(LogController): 
    def __init__(self, dataset): 
        super(TangoLogController, self).__init__(dataset)

        self.subscribe('RGB', self.on_rgb)
        self.subscribe('RGB_VIO', self.on_pose)

        # Keep a queue of finite lenght to ensure 
        # time-sync with RGB and IMU
        self.__pose_q = deque(maxlen=10)
        # self.__item_q = deque(maxlen=3)

    def on_rgb(self, t_img, img): 
        # self.__item_q.append((0, t_img, img))

        if not len(self.__pose_q):
            return

        t_pose, pose = self.__pose_q[-1]
        self.on_frame(t_pose, t_img, pose, img)

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

    def on_frame(self, t_pose, t_img, pose, img): 
        raise NotImplementedError()

    # @abstractmethod
    # def on_rgb(self, t, img): 
    #     raise NotImplementedError()

    # @abstractmethod
    # def on_pose(self, t, pose): 
    #     raise NotImplementedError()
