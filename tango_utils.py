"""Tango log reader API"""

# Author: Sudeep Pillai <spillai@csail.mit.edu>
# License: MIT

import numpy as np
import cv2
import os.path
from heapq import heappush, heappop
from abc import ABCMeta, abstractmethod

from collections import Counter
from bot_externals.log_utils import Decoder, LogReader, LogController
from bot_vision.image_utils import im_resize
from bot_geometry.rigid_transform import RigidTransform
from bot_vision.camera_utils import CameraIntrinsic


def TangoOdomDecoder(channel, every_k_frames=1): 
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

    def odom_decode(data): 
        """ x, y, z, qx, qy, qz, qw, status_code, confidence """
        p = np.float64(data.split(','))
        if len(p) < 8:
            raise Exception('TangoOdomDecoder.odom_decode :: Failed to retreive pose')
        if p[7] == 0: 
            raise Warning('TangoOdomDecoder.odom_decode :: Pose initializing.., status_code: 0')

        tvec, ori = p[:3], p[3:7]
        p_SD = RigidTransform(xyzw=ori, tvec=tvec)
        p_SC = p_SD * p_DC

        return p_CAM_S * p_SC
            
    return Decoder(channel=channel, every_k_frames=every_k_frames, decode_cb=lambda data: odom_decode(data))

class TangoImageDecoder(Decoder): 
    """
    """
    def __init__(self, directory, channel='RGB', every_k_frames=1, shape=(720,1280)): 
        Decoder.__init__(self, channel=channel, every_k_frames=every_k_frames)
        self.shape = shape 
        self.directory = directory

    def decode(self, msg): 
        fn = os.path.join(self.directory, msg)
        if os.path.exists(fn): 
            im = cv2.imread(fn, cv2.CV_LOAD_IMAGE_COLOR)
            return im_resize(im, shape=self.shape)
        else: 
            raise Exception('File does not exist')


class TangoLog(object): 
    def __init__(self, filename): 
        self.meta_ =  open(filename, 'r')
        topics = map(lambda (t,ch,data): ch, 
                     filter(lambda ch: len(ch) == 3, 
                            map(lambda l: l.replace('\n','').split('\t'), 
                                filter(lambda l: '\n' in l, self.meta_.readlines()))))

        # Save topics and counts
        c = Counter(topics)
        self.topics_ = list(set(topics))
        self.topic_lengths_ = dict(c.items())
        self.length_ = sum(self.topic_lengths_.values())

        messages_str = ', '.join(['{:} ({:})'.format(k,v) for k,v in c.iteritems()])
        print('\nTangoLog\n========\n\tTopics: {:}\n\tMessages: {:}\n'.format(self.topics_, messages_str))
        self.meta_.seek(0)

    @property
    def length(self): 
        return self.length_

    def read_messages(self, topics=None, start_time=0): 

        N = 10000
        heap = []

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
            
            heappush(heap, (t, ch, data))
        
        for j in range(len(heap)): 
            c_t, c_ch, c_data = heappop(heap)
            assert(c_t >= p_t)
            p_t = c_t
            
            yield c_ch, c_data, c_t

class TangoLogReader(LogReader): 

    H, W = 720, 1280
    K = np.float64([1043.75, 0, 638.797, 0, 1043.75, 357.991, 0, 0, 1]).reshape(3,3)
    D = np.float64([0.234583, -0.689864, 0, 0, 0.679871])
    cam = CameraIntrinsic(K=K, D=D, shape=(H,W))

    def __init__(self, directory, scale=1., start_idx=0, every_k_frames=1): 
        
        # Set directory and filename for time synchronized log reads 
        self.directory_ = os.path.expanduser(directory)
        self.filename_ = os.path.join(self.directory_, 'tango_data.txt')
        self.scale_ = scale
        self.shape_ = (int(1280 * scale), int(720 * scale))
        assert(self.shape_[0] % 2 == 0 and self.shape_[1] % 2 == 0)

        self.start_idx_ = start_idx

        # HARD-CODED to compensate for scaling the logger

        # Initialize TangoLogReader with appropriate decoders
        super(TangoLogReader, self).__init__(self.filename_, 
                                             decoder=[
                                                 TangoOdomDecoder(channel='RGB_VIO', every_k_frames=every_k_frames), 
                                                 TangoImageDecoder(self.directory_, channel='RGB', 
                                                                   shape=self.shape_, every_k_frames=every_k_frames)
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
        return TangoLogReader.cam.scaled(self.scale_)

    def load_log(self, filename): 
        return TangoLog(filename)

    def iteritems(self, reverse=False): 
        if self.index is not None: 
            raise NotImplementedError('Cannot provide items indexed')
        else: 
            if reverse: 
                raise RuntimeError('Cannot provide items in reverse when file is not indexed')

            # # Decode only messages that are supposed to be decoded 
            # # print self._log.get_message_count(topic_filters=self.decoder_keys())
            # st, end = self.log.get_start_time(), self.log.get_end_time()
            # start_t = Time(st + (end-st) * self.start_idx / 100.0)
             
            print('Reading TangoLog from index={:} onwards'.format(self.start_idx_))
            for self.idx, (channel, msg, t) in enumerate(self.log.read_messages()):
                if self.idx < self.start_idx_: 
                    continue
                # self.idx % self.every_k_frames == 0:
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
            pass
            # raise RuntimeError('Failed to decode data from channel: %s, mis-specified decoder?' % channel)
        
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
    __metaclass__ = ABCMeta
    def __init__(self, dataset): 
        super(TangoLogController, self).__init__(dataset)

        self.subscribe('RGB', self.on_rgb)
        self.subscribe('RGB_VIO', self.on_pose)
        
    @abstractmethod
    def on_rgb(self, t, img): 
        raise NotImplementedError()

    @abstractmethod
    def on_pose(self, t, pose): 
        raise NotImplementedError()
