# Author: Sudeep Pillai <spillai@csail.mit.edu>
# License: MIT
from collections import deque, namedtuple
from abc import ABCMeta, abstractmethod

import numpy as np

from pybot.utils.misc import print_green, print_red
from pybot.utils.misc import Counter, Accumulator, CounterWithPeriodicCallback 
from pybot.geometry.rigid_transform import RigidTransform

def inject_noise(poses_iterable, noise=[0,0]):

    np.random.seed(1)
    noise = np.float32(noise)
    def get_noise():
        if len(noise) == 6:
            xyz = np.random.normal(0, noise[:3])
            rpy = np.random.normal(0, noise[3:])
        elif len(noise) == 2: 
            xyz = np.random.normal(0, noise[0], size=3) \
                  if noise[0] > 0 else np.zeros(3)
            rpy = np.random.normal(0, noise[1], size=3) \
                  if noise[1] > 0 else np.zeros(3)
        else:
            raise ValueError('Unknown noise length, either 2, or 6')
        
        return RigidTransform.from_rpyxyz(
            rpy[0], rpy[1], rpy[2], xyz[0], xyz[1], xyz[2])
    
    p_accumulator = deque(maxlen=2)
    p_accumulator_noisy = deque(maxlen=2)
    
    for p in poses_iterable:
        p_accumulator.append(p)
        if len(p_accumulator) == 1:
            p_accumulator_noisy.append(p_accumulator[-1])
        else: 
            p21 = get_noise() * \
                  (p_accumulator[-2].inverse() * p_accumulator[-1])
            last = p_accumulator_noisy[-1]
            p_accumulator_noisy.append(last.oplus(p21))
        yield p_accumulator_noisy[-1]

class Sampler(object): 
    __metaclass__ = ABCMeta
    """
    Sampling based on a specific criteria

    get_sample:
        Provides a map function for sample retrieval
        For e.g. Keyframe={'img': im, 'pose': pose}, get_sample 
        employs the sampling criteria over a specific attribute 
        such as pose, where get_sample=lambda item: item['pose']

    """
    def __init__(self, lookup_history=10, 
                 get_sample=lambda item: item, 
                 on_sampled_cb=lambda index, item: None, verbose=False): 

        if not hasattr(get_sample, '__call__'): 
            raise ValueError('''get_sample is not a function, '''
                             '''Provide an appropriate attribute selection''')
            
        self.get_sample = get_sample

        self.q_ = deque(maxlen=lookup_history)
        self.on_sampled_cb_ = on_sampled_cb
        # self.force_sample_ = False
        
        # Maintain total items pushed and sampled
        self.all_items_ = Counter()
        self.sampled_items_ = Counter()

        self.verbose_ = verbose
        if verbose: 
            self.verbose_printer_ = CounterWithPeriodicCallback(every_k=5, 
                                                                process_cb=self.print_stats)
            self.verbose_printer_.register_callback(self, 'on_sampled_cb_')


            self.verbose_all_ = deque()
            self.verbose_index_ = deque()

    # def force_check(self): 
    #     sample = self.force_sample_
    #     self.force_sample_ = False
    #     return sample
            
    # def force_sample(self): 
    #     self.force_sample_ = True
    
    def length(self, type='samples'): 
        if type=='samples': 
            return self.sampled_items_.length
        elif type == 'all':
            return self.all_items_.length
        else:
            raise ValueError('Request for unknown length attribute')
        
    def print_stats(self, finish=False): 
        print_green('Sampler: Total: {:}, Samples: {:}, Ratio: {:3.2f} %'
                    .format(self.all_items_.index, self.sampled_items_.index, 
                            self.sampled_items_.index * 100.0 / self.all_items_.index))
        # self.visualize(finish=finish)

    # @abstractmethod
    # def visualize(self, finish=False): 
    #     raise NotImplementedError()

    @abstractmethod
    def check_sample(self, item): 
        raise NotImplementedError()

    def iteritems(self, iterable): 
        for item in iterable: 
            if self.append(item): 
                yield self.latest_sample
    
    def append(self, item, force=False): 
        """
        Add item to the sampler, returns the 
        index of the sampled item and the 
        corresponding item.
        """
        if self.verbose_: 
            self.verbose_all_.append(item)

        self.all_items_.count()                    
        ret = self.check_sample(item, force=force) 

        if ret: 
            self.q_.append(item)
            self.on_sampled_cb_(self.all_items_.index, item)
            self.sampled_items_.count()            

            if self.verbose_: 
                self.verbose_index_.append(self.all_items_.index)

        return ret

    @property
    def latest_sample(self): 
        return self.q_[-1]

    def item(self, idx):
        return self.q_[idx]
    
    @classmethod
    def from_items(cls, items, lookup_history=10):
        raise NotImplementedError()

class PoseSampler(Sampler): 
    def __init__(self, theta=np.deg2rad(20), displacement=0.25, lookup_history=10, 
                 get_sample=lambda item: item, 
                 on_sampled_cb=lambda index, item: None, verbose=False): 
        Sampler.__init__(self, lookup_history=lookup_history, 
                         get_sample=get_sample, 
                         on_sampled_cb=on_sampled_cb, verbose=verbose)

        self.displacement_ = displacement
        self.theta_ = theta

    def from_items(self, items, return_indices=False):
        items = [(self.latest_sample, idx) for idx, item in enumerate(items) if self.append(item)]

        # Unzip returns tuples: inds and poses must be lists
        poses, inds = map(list, zip(*items))
        if return_indices: 
            return poses, inds
        
        return poses

    def check_sample(self, item, force=False):
        if force: 
            return True

        pose = self.get_sample(item)
        pinv = pose.inverse()

        # Check starting from new to old items
        # print '------------'
        for p in reversed(self.q_): 
            newp = pinv * self.get_sample(p)
            d, r = np.linalg.norm(newp.tvec), np.fabs(newp.to_rpyxyz()[:3])
            if d < self.displacement_ and (r < self.theta_).all(): 
                return False

        return True

    # def visualize(self, finish=False): 
    #     # # RPY
    #     # rpyxyz = np.vstack(item.to_rpyxyz() for item in self.all_)
    #     # rot = rpyxyz[:,:3]
    #     # trans = rpyxyz[:,3:6]

    #     # Quaternion
    #     rpyxyz = np.hstack([np.vstack(self.get_sample(item).rotation.wxyz for item in self.verbose_all_), 
    #                         np.vstack(self.get_sample(item).translation for item in self.verbose_all_)])
    #     rot = rpyxyz[:,:4]
    #     trans = rpyxyz[:,4:7]

    #     # # Angle axis
    #     # rpyxyz = np.hstack([np.vstack(np.hstack(item.rotation.to_angle_axis()) for item in self.all_), 
    #     #                     np.vstack(item.translation for item in self.all_)])

    #     # # Delta Quaternion
    #     # delta = [self.all_[idx+1].inverse() * self.all_[idx] for idx in xrange(len(self.all_)-1)]
    #     # print delta[0]
    #     # rpyxyz = np.hstack([np.vstack(item.rotation.wxyz for item in delta), 
    #     #                     np.vstack(item.translation for item in delta)])

    #     ts = np.tile(np.arange(len(rpyxyz)).reshape(-1,1), [1,3])
    #     print rpyxyz.shape, ts.shape
    #     if not finish:
    #         return

    #     fig = plt.figure(1, figsize=(8.0,4.5), dpi=100) 
    #     fig.clf()
        
    #     ax1 = plt.subplot(2,1,1)
    #     plt.plot(ts, rot)
    #     plt.vlines(np.int32(self.verbose_index_), ymin=-1, ymax=1, color='k')
    #     ax1.set_xlim([max(0, np.max(ts)-400), np.max(ts)])

    #     ax2 = plt.subplot(2,1,2)
    #     plt.plot(ts, trans)
    #     plt.vlines(np.int32(self.verbose_index_), ymin=-100, ymax=100, color='k')
    #     ax2.set_xlim([max(0, np.max(ts)-400), np.max(ts)])

    #     plt.show(block=finish)


# class FrustumVolumeIntersectionPoseSampler(Sampler): 
#     def __init__(self, iou=0.5, depth=20, fov=np.deg2rad(60), lookup_history=10, 
#                  get_sample=lambda item: item, 
#                  on_sampled_cb=lambda index, item: None, verbose=False): 
#         Sampler.__init__(self, lookup_history=lookup_history, 
#                          get_sample=get_sample, 
#                          on_sampled_cb=on_sampled_cb, verbose=verbose)
#         self.iou_ = iou
#         self.depth_ = depth
#         self.fov_ = fov
        
#         from pybot.geometry.rigid_transform import RigidTransform
#         from pybot.vision.camera_utils import Frustum
#         from bot_graphics.volumes import SweepingFrustum

#         self.volume_ = SweepingFrustum()
#         self.get_frustum = lambda pose: Frustum(pose, zmin=0.05, zmax=self.depth_, fov=self.fov_)

#         # Get canonical volume
#         fverts = self.get_frustum(RigidTransform.identity()).get_vertices()

#         print 'Adding basic shape'
#         self.volume_.add_vertices(fverts)
#         self.fvol_ = self.volume_.get_volume()
#         self.volume_.clear()

#     def check_sample(self, item):
#         if self.force_check(): 
#             return True
        
#         pose = self.get_sample(item)
#         # pinv = pose.inverse()

#         fverts = self.get_frustum(pose).get_vertices()
        
#         # Check starting from new to old items
#         # print '------------'
#         for p in reversed(self.q_): 
#             verts = self.get_frustum(self.get_sample(p)).get_vertices()

#             self.volume_.clear()
#             self.volume_.add_vertices(fverts)
#             self.volume_.add_vertices(verts)
#             assert(self.volume_.volume_.getNumComponents() > 0)
#             intersection = self.volume_.get_volume()
#             union = self.fvol_ * 2 - intersection
#             print intersection, self.fvol_ * 2, self.fvol_
#             # print p, self.volume_.get_volume()

#             # newp = pinv * self.get_sample(p)
#             # d, r = np.linalg.norm(newp.tvec), np.fabs(newp.to_rpyxyz()[:3])
#             # print r, d < self.displacement_, (r < self.theta_).all(), newp
#             # if d < self.displacement_ and (r < self.theta_).all(): 
#             #     return False

#         return True


Keyframe = namedtuple('Keyframe', ['img', 'pose', 'index'], verbose=False)

class KeyframeSampler(PoseSampler):
    def __init__(self, theta=np.deg2rad(20), displacement=0.25, lookup_history=10, 
                 get_sample=lambda item: item.pose,  
                 on_sampled_cb=lambda index, item: None, verbose=False): 
        PoseSampler.__init__(self, displacement=displacement, theta=theta, 
                             lookup_history=lookup_history, 
                             get_sample=get_sample, 
                             on_sampled_cb=on_sampled_cb, verbose=verbose)

# class KeyframeVolumeSampler(FrustumVolumeIntersectionPoseSampler): 
#     def __init__(self, iou=0.5, depth=20, fov=np.deg2rad(60), lookup_history=10, 
#                  get_sample=lambda item: item.pose,  
#                  on_sampled_cb=lambda index, item: None, verbose=False): 
#         FrustumVolumeIntersectionPoseSampler.__init__(self, iou=iou, depth=depth, fov=fov, 
#                                                       lookup_history=lookup_history, 
#                                                       get_sample=get_sample, 
#                                                       on_sampled_cb=on_sampled_cb, verbose=verbose)


class PoseAccumulator(Accumulator): 
    def __init__(self, maxlen=100, relative=False): 
        Accumulator.__init__(self, maxlen=maxlen)

        self.relative_ = relative
        self.init_ = None

    def accumulate(self, pose):
        if self.relative_: 
            if self.init_ is None: 
                self.init_ = pose
            p = self.relative_to_init(pose)
        else: 
            p = pose

        # Call accumulate on base class
        super(PoseAccumulator, self).accumulate(p)
        
    def relative_to_init(self, pose_wt): 
        """ pose of [t] wrt [0]:  p_0t = p_w0.inverse() * p_wt """  
        return (self.init_.inverse()).oplus(pose_wt)

class PoseInterpolator(PoseAccumulator): 
    def __init__(self, maxlen=100, relative=False): 
        PoseAccumulator.__init__(self, maxlen=maxlen, relative=relative)

        self.relative_ = relative
        self.init_ = None
        
    def add(self, pose): 
        super(PoseAccumulator, self).accumulate(pose)

    def query(self, pose): 
        raise NotImplementedError()

class SkippedPoseAccumulator(PoseAccumulator): 
    def __init__(self, skip=10, **kwargs): 
        Counter.__init__(self)
        PoseAccumulator.__init__(self, **kwargs)
        self.skip_ = skip
        self.skipped_ = False

    @property
    def skipped(self): 
        return self.skipped_

    def accumulate(self, pose): 
        self.skipped_ = True
        if self.check_divisibility(self.skip_):
            PoseAccumulator.accumulate(self, pose)
            self.reset()
            self.skipped_ = False
        self.count()

# class PoseInterpolation(object): 
#     def __init__(self, ncontrol=2, nposes=10): 
#         self.q_ = deque(maxlen=ncontrol)
#         self.interp_ = deque(maxlen=nposes)

#         self.ncontrol_ = ncontrol
#         self.nposes_ = nposes

#     def append(self, p): 
#         if not isinstance(p, RigidTransform): 
#             raise TypeError('PoseInterpolation expects RigidTransform')
#         self.q_.append(p)

#     def interpolate(self, p1, p2, w): 
#         assert(w >= 0 and w <= 1.0)
#         return p1.interpolate(p2, w) 

#     def iteritems(self): 
#         if len(self.q_) >= 2: 
#             for w in np.linspace(0,1,self.nposes_): 
#                 yield self.interpolate(self.q_[-2], self.q_[-1], w)
#             self.q_.popleft()
