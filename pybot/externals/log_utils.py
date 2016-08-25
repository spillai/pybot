"""Basic API for lcmlogs/rosbags"""

# Author: Sudeep Pillai <spillai@csail.mit.edu>
# License: MIT

import os.path
import numpy as np
from itertools import islice, izip
from abc import ABCMeta, abstractmethod
from collections import Counter
from heapq import heappush, heappop

def take(iterable, max_length=None): 
    return iterable if max_length is None else islice(iterable, max_length)

class Decoder(object): 
    def __init__(self, channel='', every_k_frames=1, decode_cb=lambda data: data): 
        self.channel_ = channel
        self.every_k_frames_ = every_k_frames
        self.decode_cb_ = decode_cb
        self.idx_ = 0

    def decode(self, data): 
        try: 
            return self.decode_cb_(data)
        except Exception, e:
            raise RuntimeError('\nError decoding channel: {} with {}\n'
                               'Data: {}, Can decode: {}\n'
                               'Error: {}\n'\
                               .format(self.channel_, self.decode_cb_.func_name, data, 
                                       self.can_decode(self.channel_), e))

    def can_decode(self, channel): 
        return self.channel_ == channel

    def should_decode(self): 
        if self.every_k_frames_ == 1: 
            return True
        ret = self.idx_ % self.every_k_frames_ == 0
        self.idx_ += 1
        return ret

    @property
    def channel(self): 
        return self.channel_

class LogDecoder(object): 
    """
    Defines a set of decoders to use against the log (either on-line/off-line)
    """
    def __init__(self, decoder=None):
        if isinstance(decoder, list): 
            self.decoder_ = { dec.channel: dec for dec in decoder } 
        else: 
            self.decoder_ = { decoder.channel: decoder }

    def decode_msg(self, channel, data, t): 
        try: 
            dec = self.decoder_[channel]
            if dec.should_decode():
                return True, (t, channel, dec.decode(data))
        except KeyError: 
            pass
        except Exception as e:
            print('{} :: decode_msg :: {}'.format(self.__class__.__name__, e))
            import traceback
            traceback.print_exc()
                    
        return False, (None, None, None)

    @property
    def decoder(self): 
        return self.decoder_

class LogFile(object): 
    """
    Generic interface for log reading. 
    See tango_data/<dataset>/meta_data.txt
    """

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
        print(self)

    def __repr__(self): 
        messages_str = ', '.join(['{:} ({:})'.format(k,v) 
                                  for k,v in self.topic_lengths_.iteritems()])
        return '\n{} \n========\n' \
        '\tFile: {:}\n' \
        '\tTopics: {:}\n' \
        '\tMessages: {:}\n'.format(
            self.__class__.__name__, 
            self.filename_, 
            self.topics_, messages_str)

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

    # @property
    # def fd(self): 
    #     """ Open the tango meta data file as a file descriptor """
    #     return open(self.filename, 'r')
        
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
        with open(self.filename, 'r') as f: 
            for l in f: 
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


class LogReader(LogDecoder): 
    def __init__(self, filename, decoder=None, start_idx=0, every_k_frames=1, 
                 max_length=None, index=False, verbose=False):
        LogDecoder.__init__(self, decoder=decoder)

        filename = os.path.expanduser(filename)
        if not os.path.exists(filename):
            raise RuntimeError('Invalid Filename: %s' % filename)
        print('{} :: OFFLINE mode'.format(self.__class__.__name__))

        # Store attributes
        self.filename_ = filename
        self.every_k_frames_ = every_k_frames
        self.idx_ = 0
        self.start_idx_ = start_idx
        self.max_length_ = max_length

        # Load the log
        self._init_log()

        # Build index
        if index: 
            self._index()
        else: 
            self.index = None

        self.verbose_ = verbose

    @property
    def filename(self): 
        return self.filename_

    @property
    def every_k_frames(self): 
        return self.every_k_frames_

    @property
    def start_idx(self): 
        return self.start_idx_

    @property
    def max_length(self): 
        return self.max_length_

    @property
    def idx(self): 
        return self.idx_

    @idx.setter
    def idx(self, index): 
        self.idx_ = index

    def _init_log(self): 
        self.log_ = self.load_log(self.filename)
        self.idx_ = 0

    def reset(self): 
        self._init_log()

    def _index(self): 
        raise NotImplementedError()

    def length(self, channel): 
        raise NotImplementedError()

    def calib(self, channel=''): 
        raise NotImplementedError()

    def load_log(self, filename): 
        raise NotImplementedError('load_log not implemented in LogReader')

    def check_tf_relations(self, relations): 
        raise NotImplementedError()

    def establish_tfs(self, relations): 
        raise NotImplementedError()

    def iteritems(self): 
        raise NotImplementedError()

    def iterframes(self): 
        raise NotImplementedError()

    @property
    def log(self): 
        return self.log_

    @property
    def controller(self): 
        raise NotImplementedError()

    @property
    def db(self): 
        raise NotImplementedError()

class LogController(object): 
    __metaclass__ = ABCMeta

    """
    Abstract log controller class 
    Setup channel => callbacks so that they are automatically called 
    with appropriate decoded data and timestamp

    Registers callbacks based on channel names, 
    and runs the dataset via run(). 

    init() sets up the controller, and finish() cleans up afterwards
    """

    @abstractmethod    
    def __init__(self, dataset): 
        """
        Setup channel => callbacks so that they are automatically called 
        with appropriate decoded data and timestamp
        """
        self.dataset_ = dataset
        self.controller_cb_ = {}
        self.controller_idx_ = 0

    def subscribe(self, channel, callback):
        func_name = getattr(callback, 'im_func', callback).func_name
        print('\t{:} :: Subscribing to {:} with callback {:}'
              .format(self.__class__.__name__, channel, func_name))
        self.controller_cb_[channel] = callback

    def _run_offline(self): 
        pass

    def run(self):
        if not len(self.controller_cb_): 
            raise RuntimeError('{:} :: No callbacks registered yet,'
                               'subscribe to channels first!'
                               .format(self.__class__.__name__))

        # Initialize
        self.init()

        # Run
        print('{:}: run::Reading log {:}'
              .format(self.__class__.__name__, self.filename))
        # for self.controller_idx_, (t, ch, data) in enumerate(self.dataset_.iterframes()): 
        for self.controller_idx_, (t, ch, data) in enumerate(self.dataset_.iteritems()): 
            if ch in self.controller_cb_: 
                self.controller_cb_[ch](t, data)

        # Finish up
        self.finish()

    def init(self): 
        """
        Pre-processing for inherited controllers
        """
        print('{:} :: Initializing controller {:}'.format(self.__class__.__name__, self.filename))
        print('{:} :: Subscriptions:'.format(self.__class__.__name__))
        for k,v in self.controller_cb_.iteritems(): 
            func_name = getattr(v, 'im_func', v).func_name
            print('\t{} -> {}'.format(k,func_name))
        print('-' * 80)

    def finish(self): 
        """
        Post-processing for inherited controllers
        """
        print('{:}: finish::Finishing controller {:}'.format(self.__class__.__name__, self.filename))

    @property
    def index(self): 
        return self.controller_idx_

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
    def __init__(self, dataset, meta=None): 
        self.dataset_ = dataset
        self.meta_ = meta
        print('{}'.format(meta))

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
        # Retrieve ground truth information
        gt_str = '{} frames annotated ({} total annotations)'\
            .format(self.annotationdb.num_frame_annotations, 
                    self.annotationdb.num_annotations) \
            if self.is_ground_truth_available else 'Not Available'

        # Pretty print IndexDB description 
        print('\nTango IndexDB \n========\n'
              '\tFrames: {:}\n'
              '\tGround Truth: {:}\n'
              .format(self.annotationdb.num_frames, gt_str))

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

    @property
    def meta(self): 
        return self.meta_

    @property
    def annotationdb(self): 
        return self.meta_

    @property
    def annotated_inds(self): 
        return self.annotationdb.annotated_inds

    @property
    def object_annotations(self): 
        return self.annotationdb.object_annotations

    @property
    def objects(self): 
        return self.annotationdb.objects

    @property
    def poses(self): 
        return [v.pose for k,v in self.frame_index_.iteritems()]

    @property
    def is_ground_truth_available(self): 
        return self.meta_ is not None

    def check_ground_truth_availability(self):
        if not self.is_ground_truth_available: 
            raise RuntimeError('Ground truth dataset not loaded')

    @property
    def annotated_indices(self): 
        assert(self.ground_truth_available)
        assert(self.start_idx_ == 0)
        inds, = np.where(self.annotationdb.annotation_sizes)
        return inds

    def keyframedb(self, *args, **kwargs): 
        raise NotImplementedError()

    def roidb(self, target_hash, targets=[], every_k_frames=1, verbose=True, skip_empty=True): 
        """
        Returns (img, bbox, targets [unique text])
        """

        self.check_ground_truth_availability()

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

            yield (data.img, bboxes, np.int32(map(lambda key: target_hash.get(key, -1), target_names)))
