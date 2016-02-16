<<<<<<< HEAD
"""
Basic utilities for dataset reader 
"""

# Copyright (c) 2015 Sudeep Pillai <spillai@csail.mit.edu>
# License: BSD 3 clause
=======
"""Basic dataset reader"""
# Author: Sudeep Pillai <spillai@csail.mit.edu>
# License: MIT
>>>>>>> 5f3d1790c4ecc51ff5fca0f48789e7a49803697b

import cv2
import numpy as np
import os, fnmatch, time
import re
from itertools import izip, imap, chain, islice
from collections import defaultdict, namedtuple
from bot_utils.async_utils import async_prefetch

from bot_vision.image_utils import im_resize

def valid_path(path): 
    vpath = os.path.expanduser(path)
    if not os.path.exists(vpath): 
        raise RuntimeError('Path invalid {:}'.format(vpath))
    return vpath

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

def read_files(directory, pattern='*.png'): 
    """
    Recursively read a directory and return all files 
    that match file pattern. 
    """
    matched_files = []
    for root, dirs, files in os.walk(directory): 
        # Filter only filename matches 
        matches = [os.path.join(root, fn) 
                   for fn in fnmatch.filter(files, pattern)]
        if not len(matches): continue
        matched_files.extend(matches)

    return matched_files

def read_dir(directory, pattern='*.png', recursive=True, expected=None, verbose=False, flatten=False): 
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

        # Filter only expected folders if given
        # Go through /root/root_1 and see if root is in expected ["root", "other1", "other2"]
        if expected is not None and not any((root.split('/')[-1]).startswith(exp) for exp in expected): 
            continue

        # Verbose print
        if verbose: 
            print root, dirs

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

    if flatten: 
        return list(chain([fn for fns in fn_map.values() for fn in fns]))
    return fn_map

class FileReader(object): 
    def __init__(self, filename, process_cb, start_idx=0): 
        self.filename_ = filename
        self.start_idx_ = start_idx
        self.items_ = process_cb(filename)

    def iteritems(self, every_k_frames=1, reverse=False): 
        if reverse: 
            raise NotImplementedError
        return islice(self.items_, self.start_idx_, None, every_k_frames)

    @property
    def items(self): 
        return self.items_

class DatasetReader(object): 
    """
    Simple Dataset Reader
    Refer to this class and ImageDatasetWriter for input/output
    """

    # Get directory, and filename pattern
    def __init__(self, process_cb=lambda x: x, 
                 template='template_%i.txt', start_idx=0, max_files=10000, 
                 files=None):
        template = os.path.expanduser(template)
        self.process_cb = process_cb

        # Index starts at 0
        if files is None:

            # Find files with matching patterns within 
            # the directory, and only add up to required
            # number of files (replace %010i with *
            # for fast pattern matching)

            directory, basename = os.path.split(template)
            ext = os.path.splitext(basename)[-1]
            st = basename.find('%')
            end = basename.find('i', st)+1
            pattern = basename.replace(basename[st:end], '*')

            try: 
                files = os.listdir(directory) 
                nmatches = len(fnmatch.filter(files, pattern))
            except: 
                nmatches = start_idx + max_files
            self.files = [template % idx
                          for idx in range(start_idx, min(nmatches, start_idx + max_files))]

            print('Found {:} files with pattern: {:}'.format(nmatches, pattern))
            print('From {:} to {:}'.format(valid_path(self.files[0]), valid_path(self.files[-1])))
        else: 
            self.files = files
        
        # print('First file: {:}: {:}'.format(template % start_idx, 'GOOD' if
        # os.path.exists(template % start_idx) else 'BAD'))

    @staticmethod
    def from_filenames(process_cb, files): 
        return DatasetReader(process_cb=process_cb, files=files)

    def _prefetch(self, fnos): 
        pass
    def yield_data(self): 
        pass

    @staticmethod
    def from_directory(process_cb, directory, pattern='*.png'):
        files = read_dir(directory, pattern=pattern, flatten=True)
        sorted_files = natural_sort(files)
        return DatasetReader.from_filenames(process_cb, sorted_files)

    # @async_prefetch
    def iteritems(self, every_k_frames=1, reverse=False):
        fnos = np.arange(0, len(self.files), every_k_frames).astype(int)
        if reverse: 
            fnos = fnos[::-1]
        for fno in fnos:
            yield self.process_cb(self.files[fno])

    def iterinds(self, inds, reverse=False): 
        fnos = inds.astype(int)
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


    def __init__(self, template='template_%i.txt', start_idx=0, max_files=10000, files=None):
        try: 
            from pybot_vision import read_velodyne_pc
        except: 
            raise RuntimeError('read_velodyne_pc missing in pybot_vision. Compile it first!')
        DatasetReader.__init__(self, process_cb=lambda fn: read_velodyne_pc(fn), template=template, 
                               start_idx=start_idx, max_files=max_files, files=files)
        

class ImageDatasetReader(DatasetReader): 
    """
    ImageDatasetReader

    >> import cv2
    >> reader = DatasetReader(process_cb=lambda fn: cv2.imread(fn, 0), ...)
    >> reader = DatasetReader(process_cb=lambda fn: cv2.imread(fn, 0), 
                    template='data_%i.txt', start_idx=1, max_files=10000)
    """

    @staticmethod
    def imread_process_cb(scale=1.0, grayscale=False):
        return lambda fn: im_resize(cv2.imread(fn, cv2.IMREAD_GRAYSCALE if grayscale else cv2.IMREAD_UNCHANGED), scale=scale)
    
    def __init__(self, template='template_%i.txt', start_idx=0, max_files=10000, files=None, scale=1.0, grayscale=False): 
        DatasetReader.__init__(self, 
                               process_cb=ImageDatasetReader.imread_process_cb(scale=scale, grayscale=grayscale), template=template, 
                               start_idx=start_idx, max_files=max_files, files=files)

    @staticmethod
    def from_filenames(files, **kwargs): 
        return DatasetReader(process_cb=ImageDatasetReader.imread_process_cb(), files=files, **kwargs)

    @staticmethod
    def from_directory(directory, pattern='*.png', **kwargs):
        return DatasetReader.from_directory(process_cb=ImageDatasetReader.imread_process_cb(),
                                          directory=directory, pattern=pattern)
        
class StereoDatasetReader(object): 
    """
    KITTIDatasetReader: ImageDatasetReader (left) + ImageDatasetReader (right)
    """

    def __init__(self, directory='', 
                 left_template='image_0/%06i.png', 
                 right_template='image_1/%06i.png', 
                 start_idx=0, max_files=10000, scale=1.0, grayscale=False): 
        self.left = ImageDatasetReader(template=os.path.join(os.path.expanduser(directory),left_template), 
                                       start_idx=start_idx, max_files=max_files, scale=scale, grayscale=grayscale)
        self.right = ImageDatasetReader(template=os.path.join(os.path.expanduser(directory),right_template), 
                                        start_idx=start_idx, max_files=max_files, scale=scale, grayscale=grayscale)

    @classmethod 
    def from_filenames(cls, left_files, right_files, **kwargs): 
        c = cls()
        c.left = ImageDatasetReader.from_filenames(left_files, **kwargs)
        c.right = ImageDatasetReader.from_filenames(right_files, **kwargs)
        return c

    @classmethod 
    def from_directory(cls, left_directory, right_directory, **kwargs): 
        c = cls()
        c.left = ImageDatasetReader.from_directory(left_directory, **kwargs)
        c.right = ImageDatasetReader.from_directory(right_directory, **kwargs)
        return c

    @property
    def length(self): 
        assert(self.left.length == self.right.length)
        return self.left.length

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


