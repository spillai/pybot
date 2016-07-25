"""Basic API for lcmlogs/rosbags"""

# Author: Sudeep Pillai <spillai@csail.mit.edu>
# License: MIT

import os.path
import numpy as np
from itertools import islice
from abc import ABCMeta, abstractmethod

def take(iterable, max_length=None): 
    return iterable if max_length is None else islice(iterable, max_length)

class Decoder(object): 
    def __init__(self, channel='', every_k_frames=1, decode_cb=lambda data: data): 
        self.channel = channel
        self.every_k_frames = every_k_frames
        self.decode_cb = decode_cb
        self.idx = 0

    def decode(self, data): 
        try: 
            return self.decode_cb(data)
        except Exception, e:
            print e
            raise RuntimeError('Error decoding channel: {} with {}\n'
                               'Data: {}\n '
                               'Can decode: {}'\
                               .format(self.channel, self.decode_cb.func_code, data, 
                                       self.can_decode(self.channel)))

    def can_decode(self, channel): 
        return self.channel == channel

    def should_decode(self): 
        self.idx += 1
        return self.idx % self.every_k_frames == 0 

class LogReader(object): 
    def __init__(self, filename, decoder=None, start_idx=0, every_k_frames=1, 
                 max_length=None, index=False, verbose=False):
        filename = os.path.expanduser(filename)
        if filename is None or not os.path.exists(filename):
            raise RuntimeError('Invalid Filename: %s' % filename)

        # Store attributes
        self.filename = filename
        if isinstance(decoder, list): 
            self.decoder = { dec.channel: dec for dec in decoder } 
        else: 
            self.decoder = { decoder.channel: decoder }
        self.every_k_frames = every_k_frames
        self.start_idx = start_idx
        self.max_length = max_length
        self.verbose = verbose

        # Load the log
        self._init_log()

        # Build index
        if index: 
            self._index()
        else: 
            self.index = None

        # Create Look-up table for subscriptions
        self.cb_ = {}

    def _init_log(self): 
        self.log_ = self.load_log(self.filename)
        self.idx = 0

    def reset(self): 
        self._init_log()

    def _index(self): 
        raise NotImplementedError()

    def length(self, channel): 
        raise NotImplementedError()

    @property
    def log(self): 
        return self.log_

    def calib(self, channel=''): 
        raise NotImplementedError()

    def load_log(self, filename): 
        raise NotImplementedError('load_log not implemented in LogReader')

    def subscribe(self, channel, callback): 
        self.cb_[channel] = callback

    def check_tf_relations(self, relations): 
        raise NotImplementedError()

    def establish_tfs(self, relations): 
        raise NotImplementedError()

    def iteritems(self): 
        raise NotImplementedError()

    def iterframes(self): 
        raise NotImplementedError()

    def run(self):
        if not len(self.cb_): 
            raise RuntimeError('No callbacks registered yet, subscribe to channels first!')

        # Initialize
        self.init()

        # Run
        iterator = take(self.iterframes(), max_length=self.max_length)
        for self.idx, (t, ch, data) in enumerate(iterator): 
            try: 
                self.cb_[ch](t, data)
            except KeyError, e: 
                print e
            except Exception, e: 
                import traceback
                traceback.print_exc()
                raise RuntimeError()

        # Finish up
        self.finish()

    def init(self): 
        pass

    def finish(self): 
        pass

class LogController(object): 
    __metaclass__ = ABCMeta

    """
    Abstract log controller class 

    Registers callbacks based on channel names, 
    and runs the dataset via run()
    """

    @abstractmethod    
    def __init__(self, dataset): 
        """
        Setup channel => callbacks so that they are automatically called 
        with appropriate decoded data and timestamp
        """
        self.dataset_ = dataset
        self.ctrl_cb_ = {}
        self.ctrl_idx_ = 0

    def subscribe(self, channel, callback):
        print('\t{:}: Subscribing to {:} with callback {:}'.format(self.__class__.__name__, channel, callback.im_func.func_name))
        self.ctrl_cb_[channel] = callback

    def run(self):
        if not len(self.ctrl_cb_): 
            raise RuntimeError('{:}: No callbacks registered yet, subscribe to channels first!'
                               .format(self.__class__.__name__))

        # Initialize
        self.init()

        # Run
        print('{:}: run::Reading log {:}'.format(self.__class__.__name__, self.filename))
        for self.ctrl_idx_, (t, ch, data) in enumerate(self.dataset_.iterframes()): 
            if ch in self.ctrl_cb_: 
                self.ctrl_cb_[ch](t, data)

        # Finish up
        self.finish()

    def init(self): 
        """
        Pre-processing for inherited controllers
        """
        print('{:}: init::Initializing controller {:}'.format(self.__class__.__name__, self.filename))
       
    def finish(self): 
        """
        Post-processing for inherited controllers
        """
        print('{:}: finish::Finishing controller {:}'.format(self.__class__.__name__, self.filename))

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

class LogDB(object): 
    def __init__(self, dataset): 
        self.dataset_ = dataset
        self.frame_index_ = None
        self.frame_name2idx_, self.frame_idx2name_ = None, None
        self._index()
        self.print_index_info()

    @property
    def index(self): 
        return self.frame_index_

    def _index(self): 
        raise NotImplementedError()

    def print_index_info(self): 
        pass

    def iterframes(self, reverse=False): 
        raise NotImplementedError()

    def __getitem__(self, basename): 
        try: 
            return self.frame_index_[basename]
        except KeyError, e: 
            raise KeyError('Missing key in LogDB {}'.format(basename))

    def find(self, basename): 
        try: 
            return self.frame_name2idx_[basename]
        except KeyError, e: 
            raise KeyError('Missing key in LogDB {}'.format(basename))

    @staticmethod
    def _nn_pose_fill(valid): 
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
    def dataset(self): 
        return self.dataset_
