import cv2
import numpy as np
import os, fnmatch, time
import re
from itertools import izip, imap
from collections import defaultdict, namedtuple

from bot_utils.db_utils import AttrDict

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

        base = root[len(directory)+1:]
        splits = base.split('/')
        if recursive and len(splits) > 1: 
            recursive_set_dict(fn_map, splits, matches)
        else: 
            fn_map[splits[0]] = matches
            # fn_map[os.path.basename(root)] = matches

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
        from bot_vision.image_utils import im_resize

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

class KITTIStereoDatasetReader(object): 
    """
    KITTISTereoDatasetReader: ImageDatasetReader + VelodyneDatasetReader + Calib
    """

    def __init__(self, directory='', 
                 sequence='', 
                 left_template='image_0/%06i.png', 
                 right_template='image_1/%06i.png', 
                 velodyne_template='velodyne/%06i.bin',
                 start_idx=0, max_files=10000, scale=1.0): 

        from bot_utils.kitti_helpers import kitti_stereo_calib_params, kitti_load_poses

        # Set args
        self.sequence = sequence
        self.scale = scale

        # Get calib
        self.calib = kitti_stereo_calib_params(scale=scale)

        # Read poses
        try: 
            pose_fn = os.path.join(os.path.expanduser(directory), 'poses', ''.join([sequence, '.txt']))
            self.poses = kitti_load_poses(fn=pose_fn)
        except: 
            pass

        # Read images
        seq_directory = os.path.join(os.path.expanduser(directory), 'sequences', sequence)
        self.left = ImageDatasetReader(
            template=os.path.join(seq_directory,left_template), 
            start_idx=start_idx, max_files=max_files, scale=scale
        )

        self.right = ImageDatasetReader(
            template=os.path.join(seq_directory,right_template), 
            start_idx=start_idx, max_files=max_files, scale=scale
        )

        # Read velodyne
        self.velodyne = VelodyneDatasetReader(
            template=os.path.join(seq_directory,velodyne_template), 
            start_idx=start_idx, max_files=max_files
        )

        print 'Initialized stereo dataset reader with %f scale' % scale

        
    def iter_stereo_frames(self, *args, **kwargs): 
        return izip(self.left.iteritems(*args, **kwargs), self.right.iteritems(*args, **kwargs))

    def iter_velodyne_frames(self, *args, **kwargs):         
        return self.velodyne.iteritems(*args, **kwargs)

    def iter_stereo_velodyne_frames(self, *args, **kwargs):         
        return izip(self.left.iteritems(*args, **kwargs), 
                    self.right.iteritems(*args, **kwargs), 
                    self.velodyne.iteritems(*args, **kwargs))

    def stereo_frames(self): 
        return self.iter_stereo_frames()

    @property
    def velodyne_frames(self): 
        return self.iter_velodyne_frames()


class StereoDatasetReader(object): 
    """
    KITTISTereoDatasetReader: ImageDatasetReader (left) + ImageDatasetReader (right)
    """

    def __init__(self, directory='', 
                 left_template='image_0/%06i.png', 
                 right_template='image_1/%06i.png', 
                 start_idx=0, max_files=10000): 
        self.left = ImageDatasetReader(template=os.path.join(os.path.expanduser(directory),left_template), 
                                       start_idx=start_idx, max_files=max_files)
        self.right = ImageDatasetReader(template=os.path.join(os.path.expanduser(directory),right_template), 
                                        start_idx=start_idx, max_files=max_files)
        self.iter_stereo_frames = lambda : izip(self.left.iteritems(), self.right.iteritems())

# class BumblebeeStereoDatasetReader: 
#     def __init__(self, directory): 
#         bfiles = read_dir(directory, pattern='*.bumblebee', recursive=False)
#         self.dataset = DatasetReader(process_cb=lambda x: read_bumblebee(x), files=bfiles)
#         self.iter_stereo_frames = lambda : imap(lambda x: self.split_stereo(x), self.dataset.iteritems())

#     def split_stereo(self, im): 
#          h = im.shape[0]/2
#          return im[:h], im[h:]

def natural_sort(l): 
    convert = lambda text: int(text) if text.isdigit() else text.lower() 
    alphanum_key = lambda key: [ convert(c) for c in re.split('([0-9]+)', key) ] 
    return sorted(l, key = alphanum_key)

class UWRGBDDatasetReader(object): 
    def __init__(self, target, files): 
        self.target = target 
        mask_files = natural_sort(filter(lambda fn: '_mask.png' in fn, files))
        depth_files = natural_sort(filter(lambda  fn: '_depth.png' in fn, files))
        rgb_files = natural_sort(list(set(files) - set(mask_files) - set(depth_files)))
        loc_files = natural_sort(map(lambda fn: fn.replace('.png', '_loc.txt'), rgb_files))
        assert(len(mask_files) == len(depth_files) == len(rgb_files) == len(loc_files))

        self.rgb = ImageDatasetReader.from_filenames(rgb_files)
        self.depth = ImageDatasetReader.from_filenames(depth_files)
        self.mask = ImageDatasetReader.from_filenames(mask_files)

    def iteritems(self): 
        for rgb_im, depth_im, mask_im in izip(self.rgb.iteritems(), 
                                              self.depth.iteritems(), 
                                              self.mask.iteritems()): 
            yield AttrDict(target=self.target, img=rgb_im, depth=depth_im, mask=mask_im)

class UWRGBDObjectDatasetReader(object):
    """
    RGB-D Dataset reader 
    http://rgbd-dataset.cs.washington.edu/dataset.html
    """

    def __init__(self, directory='', targets=None, num_targets=None, blacklist=['']):         
        # import scipy.io
        # from utils.frame_utils import Frame, KinectFrame
        self._dataset = read_dir(os.path.expanduser(directory), pattern='*.png', recursive=False)
        self._class_names = np.sort(self._dataset.keys())
        self._class_ids = np.arange(len(self._class_names), dtype=np.int)

        self.target_hash = dict(zip(self._class_names, self._class_ids))
        self.target_unhash = dict(zip(self._class_ids, self._class_names))

        # # Only randomly choose targets if not defined
        # if num_targets is not None and targets is None and \
        #    num_targets > 0 and num_targets < len(self._class_names): 
        #     inds = np.random.randint(len(self._class_names), size=num_targets)
        #     targets = self._class_names[inds]            
        #     print 'Classes: %i' % len(targets)

        print os.path.expanduser(directory)
        print self._dataset.keys()# , self._dataset['coffee_mug']

        # Instantiate a reader for each of the objects
        self.data = {}
        for key, files in self._dataset.iteritems(): 
            if (targets is not None and key not in targets) or key in blacklist: 
                continue

            target_id = self.target_hash[key]
            self.data[key] = UWRGBDDatasetReader(target_id, files)

    def iteritems(self): 
        for key, frames in self.data.iteritems(): 
            for frame in frames.iteritems(): 
                yield frame

        # self.data, self.target = [], []
        # for key, files in self._dataset.iteritems(): 
        #     if (targets is not None and key not in targets) or key in blacklist: 
        #         continue

        #     target_id = self.target_hash[key]

        #     self.data.extend(files)
        #     self.target.extend( [target_id] * len(files) )

        
        # # Build rgbd filename map [object_type -> [(rgb_fn,depth_fn), .... ]
        # self.rgbd_fn_map = defaultdict(list)
        # self.mat_fn_map = {}

        # # Index starts at 1
        # frame_info = namedtuple('frame_info', ['index', 'rgb_fn', 'depth_fn'])
        # for root,v in fn_map.iteritems(): 
        #     idx = 1;
        #     while True: 
        #         rgb_path = os.path.join(root, '%s_%i.png' % 
        #                                 (os.path.basename(root), idx))
        #         depth_path = os.path.join(root, '%s_%i_depth.png' % 
        #                                   (os.path.basename(root), idx))
        #         if not os.path.exists(rgb_path) or not os.path.exists(depth_path): 
        #             break

        #         self.rgbd_fn_map[os.path.basename(root)].append(
        #             frame_info(index=idx, rgb_fn=rgb_path, depth_fn=depth_path)
        #         )
        #         idx += 1

        # # Convert mat files
        # mat_map = read_dir(directory, '*.mat')
        # for files in mat_map.values(): 
        #     for fn in files: 
        #         self.mat_fn_map[os.path.splitext(os.path.basename(fn))[0]] = fn
        # print self.mat_fn_map

        # # Store relevant metadata about dataset
        # self.total_frames = idx
        # self.rgb = True

    # # Retrieve items from a particular target
    # def iteritems(self, target=None): 
    #     if target is None: 
    #         pass
    #     else: 
    #         for files in self._dataset.iteritems(): 

    # Retrieve metadata
    def load_metadata(self, mat_fn): 
        return scipy.io.loadmat(mat_fn, squeeze_me=True, struct_as_record=False)

    # Get metadata for a particular category
    def get_metadata(self, category): 
        matfile = self.mat_fn_map[category]
        return self.load_metadata(matfile)

    # Get bounding boxes for a particular category
    def get_bboxes(self, category): 
        return self.get_metadata(category)['bboxes']

    # Get files for a particular category
    def get_files(self, category): 
        return self.rgbd_fn_map[category]

    # Retrieve frame (would be nice to use generator instead)
    def get_frame(self, rgb_fn, depth_fn): 
        rgb, depth = cv2.imread(rgb_fn, 1), cv2.imread(depth_fn, -1)
        return rgb, depth


class Caltech101DatasetReader(object): 
    """
    Dataset reader written to conform to scikit.data interface
    Attributes: 
      data:         [image_fn1, ...  ]
      target:       [class_id, ... ]
      target_ids:   [0, 1, 2, ..., 101]
      target_names: [car, bike, ... ]
    """
    def __init__(self, directory='', targets=None, num_targets=None, blacklist=['BACKGROUND_Google']): 
        self._dataset = read_dir(os.path.expanduser(directory), pattern='*.jpg')
        self._class_names = np.sort(self._dataset.keys())
        self._class_ids = np.arange(len(self._class_names), dtype=np.int)

        self.target_hash = dict(zip(self._class_names, self._class_ids))
        self.target_unhash = dict(zip(self._class_ids, self._class_names))

        # Only randomly choose targets if not defined
        if num_targets is not None and targets is None and \
           num_targets > 0 and num_targets < len(self._class_names): 
            inds = np.random.randint(len(self._class_names), size=num_targets)
            targets = self._class_names[inds]            
            print 'Classes: %i' % len(targets)

        self.data, self.target = [], []
        for key, files in self._dataset.iteritems(): 
            if (targets is not None and key not in targets) or key in blacklist: 
                continue

            target_id = self.target_hash[key]
            self.data.extend(files)
            self.target.extend( [target_id] * len(files) )

        self.data = np.array(self.data)
        self.target = np.array(self.target)

        self.target_ids = sorted(np.unique(self.target))
        self.target_names = map(lambda tid: self.target_unhash[tid], self.target_ids)



def test_uw_rgbd_object(): 
    # Read dataset
    rgbd_data_uw = UWRGBDObjectDatasetReader(directory='~/data/rgbd_datasets/udub/rgbd-object/rgbd-dataset')
    for f in rgbd_data_uw.iteritems(): 
        imshow_cv('frame', f.img)




# def test_rgbd_uw(): 
#     # Read dataset
#     rgbd_data_uw = RGBDDatasetReaderUW('~/data/rgbd-datasets/udub/rgbd-scenes')




#     # Get metadata for object (bounding boxe)s
#     # Note: length of metadata and frames should be the same
#     matfile = rgbd_data_uw.get_metadata('table_1')
#     bboxes = rgbd_data_uw.get_bboxes('table_1')

#     # Get files for object
#     files = rgbd_data_uw.get_files('table_1')

#     # Construct frames with (rgb, d) filenames
#     runtime = []
#     for fidx, (bbox, f) in enumerate(zip(bboxes, files)): 
#         rgb, depth = rgbd_data_uw.get_frame(f.rgb_fn, f.depth_fn)
#         t1 = time.time()
#         KinectFrame(f.index, rgb, depth, skip=1)
#         print [(bb.category, bb.instance, bb.top, bb.bottom, bb.left, bb.right) 
#                for bb in np.array([bbox]).flatten()]
#         runtime.append((time.time() - t1) * 1e3)
#         if fidx % 10 == 0: print 'Processed ', fidx
#     print 'Done! Processed %i items. %i bboxes' % (len(files), len(bboxes))
#     print 'Average runtime: ', np.mean(runtime), 'ms'


def test_imagedataset_reader(): 
    # Read dataset
    rgb_data = ImageDatasetReader(directory='~/data/rgb-dataset-test', 
                                  pattern='*.png', template='img_%i.png')
    
    while True: 
        im = rgb_data.get_next()
        if im is None: 
            break

        print im.shape

if __name__ == "__main__": 
    # RGBD UW dataset
    test_uw_rgbd_object()

    # ImageDatasetReader
    # test_imagedataset_reader()

    
