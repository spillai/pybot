"""Basic API for lcmlogs/rosbags"""

# Author: Sudeep Pillai <spillai@csail.mit.edu>
# License: MIT

import os.path
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

    def iter_frames(self): 
        raise NotImplementedError()

    def run(self):
        if not len(self.cb_): 
            raise RuntimeError('No callbacks registered yet, subscribe to channels first!')

        # Initialize
        self.init()

        # Run
        iterator = take(self.iter_frames(), max_length=self.max_length)
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
        for self.ctrl_idx_, (t, ch, data) in enumerate(self.dataset_.iter_frames()): 
            if ch in self.ctrl_cb_: 
                self.ctrl_cb_[ch](t, data)

        # Finish up
        self.finish()

    def init(self): 
        """
        Pre-processing for inherited controllers
        """
        print('{:}: Reading log {:}'.format(self.__class__.__name__, self.filename))
       
    def finish(self): 
        """
        Post-processing for inherited controllers
        """
        print('{:}: Finished reading log {:}'.format(self.__class__.__name__, self.filename))

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

