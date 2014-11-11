import cv2
import numpy as np
import os, fnmatch, time
import re
from itertools import izip, imap
from collections import defaultdict, namedtuple

from bot_utils.db_utils import AttrDict
from bot_vision.image_utils import im_resize

def natural_sort(l): 
    convert = lambda text: int(text) if text.isdigit() else text.lower() 
    alphanum_key = lambda key: [ convert(c) for c in re.split('([0-9]+)', key) ] 
    return sorted(l, key = alphanum_key)

def recursive_set_dict(d, splits, value): 
    if len(splits) < 2: 
        d.setdefault(splits[0], value)
        return
    else: 
        d_ = d.setdefault(splits[0], {})
        recursive_set_dict(d_, splits[1:], value)
        return 

def read_dir(directory, pattern='*.png', recursive=True): 
    """
    Recursively read a directory and return a dictionary tree 
    that match file pattern. 
    """

    # Get directory, and filename pattern
    directory = os.path.expanduser(directory)

    if not os.path.exists(directory): 
        raise Exception("""Path %s doesn't exist""" % directory)

    # Build dictionary [full_root_path -> pattern_matched_files]
    fn_map = {}
    for root, dirs, files in os.walk(directory): 

        # Filter only filename matches 
        matches = [os.path.join(root, fn) 
                   for fn in fnmatch.filter(files, pattern)]
        if not len(matches): continue

        base = root[len(directory):]
        splits = filter(lambda x: len(x) > 0, base.split('/'))
        if recursive and len(splits) > 1: 
            recursive_set_dict(fn_map, splits, matches)
        elif len(splits) == 1: 
            fn_map[splits[0]] = matches
        else: 
            fn_map[os.path.basename(root)] = matches

    return fn_map

class DatasetReader(object): 
    """
    Simple Dataset Reader
    Refer to this class and ImageDatasetWriter for input/output
    """

    # Get directory, and filename pattern
    def __init__(self, process_cb=lambda x: x, 
                 template='data_%i.txt', start_idx=0, max_files=10000, 
                 files=None):
        template = os.path.expanduser(template)
        self.process_cb = process_cb

        # Index starts at 1
        if files is None: 
            self.files = [template % idx
                          for idx in range(start_idx, max_files) 
                          if os.path.exists(template % idx)]
        else: 
            self.files = files

    @staticmethod
    def from_filenames(process_cb, files): 
        return DatasetReader(process_cb=process_cb, files=files)
        
    def iteritems(self, every_k_frames=1, reverse=False):
        fnos = np.arange(0, len(self.files), every_k_frames).astype(int)
        if reverse: 
            fnos = fnos[::-1]
        for fno in fnos: 
            yield self.process_cb(self.files[fno])

    @property
    def length(self): 
        return len(self.files)

    @property
    def frames(self): 
        return self.iteritems()

class VelodyneDatasetReader(DatasetReader): 
    """
    Velodyne reader
    
    >> from pybot_vision import read_velodyne_pc
    >> reader = DatasetReader(process_cb=lambda fn: read_velodyne_pc(fn), ...)
    >> reader = DatasetReader(process_cb=lambda fn: read_velodyne_pc(fn), 
                    template='data_%i.txt', start_idx=1, max_files=10000)
    """


    def __init__(self, **kwargs): 
        try: 
            from pybot_vision import read_velodyne_pc
        except: 
            raise RuntimeError('read_velodyne_pc missing in pybot_vision. Compile it first!')

        if 'process_cb' in kwargs: 
            raise RuntimeError('VelodyneDatasetReader does not support defining a process_cb')
        DatasetReader.__init__(self, process_cb=lambda fn: read_velodyne_pc(fn), **kwargs)

class ImageDatasetReader(DatasetReader): 
    """
    ImageDatasetReader

    >> import cv2
    >> reader = DatasetReader(process_cb=lambda fn: cv2.imread(fn, 0), ...)
    >> reader = DatasetReader(process_cb=lambda fn: cv2.imread(fn, 0), 
                    template='data_%i.txt', start_idx=1, max_files=10000)
    """

    @staticmethod
    def imread_process_cb(scale=1.0): 
        if scale != 1.0: 
            return lambda fn: im_resize(cv2.imread(fn, -1), scale)
        else: 
            return lambda fn: cv2.imread(fn, -1)

    def __init__(self, **kwargs): 
        if 'process_cb' in kwargs: 
            raise RuntimeError('ImageDatasetReader does not support defining a process_cb')

        if 'scale' in kwargs: 
            scale = kwargs.pop('scale', 1.0)
            if scale < 1: 
                DatasetReader.__init__(self, 
                                       process_cb=ImageDatasetReader.imread_process_cb(scale), 
                                       **kwargs)
                return 

        DatasetReader.__init__(self, process_cb=ImageDatasetReader.imread_process_cb(), **kwargs)

    @staticmethod
    def from_filenames(files, **kwargs): 
        return DatasetReader(process_cb=ImageDatasetReader.imread_process_cb(), files=files, **kwargs)

class StereoDatasetReader(object): 
    """
    KITTIDatasetReader: ImageDatasetReader (left) + ImageDatasetReader (right)
    """

    def __init__(self, directory='', 
                 left_template='image_0/%06i.png', 
                 right_template='image_1/%06i.png', 
                 start_idx=0, max_files=10000, scale=1.0): 
        self.left = ImageDatasetReader(template=os.path.join(os.path.expanduser(directory),left_template), 
                                       start_idx=start_idx, max_files=max_files, scale=scale)
        self.right = ImageDatasetReader(template=os.path.join(os.path.expanduser(directory),right_template), 
                                        start_idx=start_idx, max_files=max_files, scale=scale)

    def iteritems(self, *args, **kwargs): 
        return izip(self.left.iteritems(*args, **kwargs), 
                    self.right.iteritems(*args, **kwargs))

    def iter_stereo_frames(self, *args, **kwargs):         
        return self.iteritems(*args, **kwargs)

# class BumblebeeStereoDatasetReader: 
#     def __init__(self, directory): 
#         bfiles = read_dir(directory, pattern='*.bumblebee', recursive=False)
#         self.dataset = DatasetReader(process_cb=lambda x: read_bumblebee(x), files=bfiles)
#         self.iter_stereo_frames = lambda : imap(lambda x: self.split_stereo(x), self.dataset.iteritems())

#     def split_stereo(self, im): 
#          h = im.shape[0]/2
#          return im[:h], im[h:]


