import cv2
import numpy as np
import os, fnmatch, time

from itertools import izip, imap
from collections import defaultdict, namedtuple

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

        if recursive: 
            fn_map[os.path.basename(root)] = matches
        else: 
            return matches
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
        self.template = os.path.expanduser(template)
        self.process_cb = process_cb

        # Index starts at 1
        if files is None: 
            self.files = [self.template % idx
                          for idx in range(start_idx, max_files) 
                          if os.path.exists(self.template % idx)]
        else: 
            self.files = files

    @staticmethod
    def from_filenames(process_cb, files): 
        return DatasetReader(process_cb=process_cb, files=files)

    def iteritems(self, every_k_frames=1):
        fnos = np.arange(0, len(self.files)-1, every_k_frames).astype(int)
        for fno in fnos: 
            yield self.process_cb(self.files[fno])


class VelodyneDatasetReader(DatasetReader): 
    """
    Velodyne reader
    
    >> from fs_utils import read_velodyne_pc
    >> reader = DatasetReader(process_cb=lambda fn: read_velodyne_pc(fn), ...)
    >> reader = DatasetReader(process_cb=lambda fn: read_velodyne_pc(fn), 
                    template='data_%i.txt', start_idx=1, max_files=10000)
    """


    def __init__(self, **kwargs): 
        try: 
            from fs_utils import read_velodyne_pc
        except: 
            raise RuntimeError('read_velodyne_pc missing in fs_utils. Compile it first!')

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

    def __init__(self, **kwargs): 
        from bot_vision.image_utils import im_resize
        if 'process_cb' in kwargs: 
            raise RuntimeError('ImageDatasetReader does not support defining a process_cb')

        if 'scale' in kwargs: 
            scale = kwargs.pop('scale', 1.0)
            if scale < 1: 
                DatasetReader.__init__(self, 
                                       process_cb=lambda fn: 
                                       im_resize(cv2.imread(fn, -1), scale), **kwargs)
                return 
        DatasetReader.__init__(self, process_cb=lambda fn: cv2.imread(fn, -1), **kwargs)

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

        self.iter_stereo_frames = lambda : izip(self.left.iteritems(), self.right.iteritems())
        self.iter_velodyne_frames = lambda : self.velodyne.iteritems()
        self.iter_stereo_velodyne_frames = lambda : izip(self.left.iteritems(), 
                                                         self.right.iteritems(), 
                                                         self.velodyne.iteritems())

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
        
class RGBDDatasetReaderUW(object):
    """
    RGB-D Dataset reader 
    http://rgbd-dataset.cs.washington.edu/dataset.html
    """

    def __init__(self, directory): 
        import scipy.io
        from utils.frame_utils import Frame, KinectFrame

        # Initialize dataset reader
        fn_map = read_dir(directory, '*.png')

        # Build rgbd filename map [object_type -> [(rgb_fn,depth_fn), .... ]
        self.rgbd_fn_map = defaultdict(list)
        self.mat_fn_map = {}

        # Index starts at 1
        frame_info = namedtuple('frame_info', ['index', 'rgb_fn', 'depth_fn'])
        for root,v in fn_map.iteritems(): 
            idx = 1;
            while True: 
                rgb_path = os.path.join(root, '%s_%i.png' % 
                                        (os.path.basename(root), idx))
                depth_path = os.path.join(root, '%s_%i_depth.png' % 
                                          (os.path.basename(root), idx))
                if not os.path.exists(rgb_path) or not os.path.exists(depth_path): 
                    break

                self.rgbd_fn_map[os.path.basename(root)].append(
                    frame_info(index=idx, rgb_fn=rgb_path, depth_fn=depth_path)
                )
                idx += 1

        # Convert mat files
        mat_map = read_dir(directory, '*.mat')
        for files in mat_map.values(): 
            for fn in files: 
                self.mat_fn_map[os.path.splitext(os.path.basename(fn))[0]] = fn
        print self.mat_fn_map

        # Store relevant metadata about dataset
        self.total_frames = idx
        self.rgb = True

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


def test_rgbd_uw(): 
    # Read dataset
    rgbd_data_uw = RGBDDatasetReaderUW('~/data/rgbd-datasets/udub/rgbd-scenes')

    # Get metadata for object (bounding boxe)s
    # Note: length of metadata and frames should be the same
    matfile = rgbd_data_uw.get_metadata('table_1')
    bboxes = rgbd_data_uw.get_bboxes('table_1')

    # Get files for object
    files = rgbd_data_uw.get_files('table_1')

    # Construct frames with (rgb, d) filenames
    runtime = []
    for fidx, (bbox, f) in enumerate(zip(bboxes, files)): 
        rgb, depth = rgbd_data_uw.get_frame(f.rgb_fn, f.depth_fn)
        t1 = time.time()
        KinectFrame(f.index, rgb, depth, skip=1)
        print [(bb.category, bb.instance, bb.top, bb.bottom, bb.left, bb.right) 
               for bb in np.array([bbox]).flatten()]
        runtime.append((time.time() - t1) * 1e3)
        if fidx % 10 == 0: print 'Processed ', fidx
    print 'Done! Processed %i items. %i bboxes' % (len(files), len(bboxes))
    print 'Average runtime: ', np.mean(runtime), 'ms'


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
    # test_rgbd_uw()

    # ImageDatasetReader
    test_imagedataset_reader()
    


# # Re-write with dataset reader inheritence
# class ImageDatasetReader: 
#     """
#     Simple Image Dataset Reader
#     Refer to this class and ImageDatasetWriter for input/output
#     """

#     # Get directory, and filename pattern
#     def __init__(self, template='img_%i.png', start_idx=1, max_files=10000):
#         self.template = os.path.expanduser(template)
#         self.start_idx = start_idx
#         self.max_files = max_files

#         # Index starts at 1
#         self.files = [idx
#                       for idx in range(self.start_idx, self.max_files) 
#                       if os.path.exists(self.template % idx)]

#     def iterframes(self, every_k_frames=1):
#         fnos = np.arange(0, len(self.files)-1, every_k_frames).astype(int)
#         for fno in fnos: 
#             fn = self.template % self.files[fno]
#             im = cv2.imread(fn, 0)
#             yield im 
