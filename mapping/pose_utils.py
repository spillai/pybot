import numpy as np
from collections import deque, namedtuple
from abc import ABCMeta, abstractmethod

from bot_utils.misc import print_green, print_red
from bot_utils.misc import Counter, Accumulator, CounterWithPeriodicCallback 

import matplotlib as mpl
import matplotlib.pyplot as plt
mpl.rcParams.update({'font.size':11, 
                     # 'font.family':'sans-serif', 
                     # 'font.sans-serif': 'Helvetica', 
                     'image.cmap':'autumn', 'text.usetex':False})

class Sampler(object): 
    __metaclass__ = ABCMeta

    def __init__(self, lookup_history=10, 
                 on_sampled_cb=lambda index, item: None, verbose=False): 
        self.q_ = deque(maxlen=lookup_history)
        self.on_sampled_cb_ = on_sampled_cb

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

    def length(self, type='samples'): 
        if type=='samples': 
            return self.sampled_items_.length

    def get_item(self, item): 
        return item

    def print_stats(self, finish=False): 
        print_green('Sampler: Total: {:}, Samples: {:}, Ratio: {:3.2f} %'
                    .format(self.all_items_.index, self.sampled_items_.index, 
                            self.sampled_items_.index * 100.0 / self.all_items_.index))
        # self.visualize(finish=finish)

    def visualize(self, finish=False): 
        # # RPY
        # rpyxyz = np.vstack(item.to_roll_pitch_yaw_x_y_z() for item in self.all_)
        # rot = rpyxyz[:,:3]
        # trans = rpyxyz[:,3:6]

        # Quaternion
        rpyxyz = np.hstack([np.vstack(self.get_item(item).rotation.wxyz for item in self.verbose_all_), 
                            np.vstack(self.get_item(item).translation for item in self.verbose_all_)])
        rot = rpyxyz[:,:4]
        trans = rpyxyz[:,4:7]

        # # Angle axis
        # rpyxyz = np.hstack([np.vstack(np.hstack(item.rotation.to_angle_axis()) for item in self.all_), 
        #                     np.vstack(item.translation for item in self.all_)])

        # # Delta Quaternion
        # delta = [self.all_[idx+1].inverse() * self.all_[idx] for idx in xrange(len(self.all_)-1)]
        # print delta[0]
        # rpyxyz = np.hstack([np.vstack(item.rotation.wxyz for item in delta), 
        #                     np.vstack(item.translation for item in delta)])

        ts = np.tile(np.arange(len(rpyxyz)).reshape(-1,1), [1,3])
        print rpyxyz.shape, ts.shape
        if not finish:
            return

        fig = plt.figure(1, figsize=(8.0,4.5), dpi=100) 
        fig.clf()
        
        ax1 = plt.subplot(2,1,1)
        plt.plot(ts, rot)
        plt.vlines(np.int32(self.verbose_index_), ymin=-1, ymax=1, color='k')
        ax1.set_xlim([max(0, np.max(ts)-400), np.max(ts)])

        ax2 = plt.subplot(2,1,2)
        plt.plot(ts, trans)
        plt.vlines(np.int32(self.verbose_index_), ymin=-100, ymax=100, color='k')
        ax2.set_xlim([max(0, np.max(ts)-400), np.max(ts)])

        plt.show(block=finish)

    @abstractmethod
    def check_sample(self, item): 
        raise NotImplementedError()
    
    def append(self, item): 
        """
        Add item to the sampler, returns the 
        index of the sampled item and the 
        corresponding item.
        """
        if self.verbose_: 
            self.verbose_all_.append(item)

        self.all_items_.count()                    
        ret = self.check_sample(item) 

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

class PoseSampler(Sampler): 
    def __init__(self, theta=20, displacement=0.25, lookup_history=10, 
                 on_sampled_cb=lambda index, item: None, verbose=False): 
        Sampler.__init__(self, lookup_history=lookup_history, 
                         on_sampled_cb=on_sampled_cb, verbose=verbose)

        self.displacement_ = displacement
        self.theta_ = np.deg2rad(theta)
        
    def check_sample(self, pose): 
        pinv = self.get_item(pose).inverse()

        # Check starting from new to old items
        # print '------------'
        for p in reversed(self.q_): 
            newp = pinv * self.get_item(p)
            d, r = np.linalg.norm(newp.tvec), np.fabs(newp.to_roll_pitch_yaw_x_y_z()[:3])
            # print r, d < self.displacement_, (r < self.theta_).all(), newp
            if d < self.displacement_ and (r < self.theta_).all(): 
                return False

        return True

Keyframe = namedtuple('Keyframe', ['img', 'pose', 'index'], verbose=False)

class KeyframeSampler(PoseSampler): 
    def __init__(self, theta=20, displacement=0.25, lookup_history=10, 
                 on_sampled_cb=lambda index, item: None, verbose=False): 
        PoseSampler.__init__(self, displacement=displacement, theta=theta, 
                             lookup_history=lookup_history, 
                             on_sampled_cb=on_sampled_cb, verbose=verbose)

    def get_item(self, item): 
        return item.pose
