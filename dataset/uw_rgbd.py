import os
import numpy as np
import cv2
from itertools import izip, imap

from scipy.io import loadmat

from bot_utils.db_utils import AttrDict
from bot_utils.dataset_readers import read_dir, read_files, natural_sort, \
    DatasetReader, ImageDatasetReader

class UWRGBDObjectDataset(object):
    """
    RGB-D Dataset reader 
    http://rgbd-dataset.cs.washington.edu/dataset.html
    """

    class _reader(object): 
        """
        RGB-D reader 
        Given mask, depth, and rgb files build an read iterator with appropriate process_cb
        """

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


    def __init__(self, directory='', targets=None, num_targets=None, blacklist=['']):         
        self._dataset = read_dir(os.path.expanduser(directory), pattern='*.png', recursive=False)
        self._class_names = np.sort(self._dataset.keys())
        self._class_ids = np.arange(len(self._class_names), dtype=np.int)

        self.target_hash = dict(zip(self._class_names, self._class_ids))
        self.target_unhash = dict(zip(self._class_ids, self._class_names))

        # Only randomly choose targets if not defined
        if num_targets is not None and targets is None and \
           num_targets > 0 and num_targets < len(self._class_names): 
            inds = np.random.randint(len(self._class_names), size=num_targets)
            targets = self._class_names[inds]            
        else: 
            targets = self._class_names
        print 'Classes: %i' % len(targets)
        print self._dataset.keys()# , self._dataset['coffee_mug']

        # Instantiate a reader for each of the objects
        self.data = {}
        for key, files in self._dataset.iteritems(): 
            if (targets is not None and key not in targets) or key in blacklist: 
                continue

            target_id = self.target_hash[key]
            self.data[key] = UWRGBDObjectDataset._reader(target_id, files)

    def iteritems(self): 
        for key, frames in self.data.iteritems(): 
            for frame in frames.iteritems(): 
                yield frame

class UWRGBDSceneDataset(object):
    """
    RGB-D Scene Dataset reader 
    http://rgbd-dataset.cs.washington.edu/dataset.html
    """
    class _reader(object): 
        """
        RGB-D reader 
        Given mask, depth, and rgb files build an read iterator with appropriate process_cb
        """
        def __init__(self, files, meta_file, version): 
            rgb_files, depth_files = UWRGBDSceneDataset._reader.scene_files(files, version)
            assert(len(depth_files) == len(rgb_files))

            self.rgb = ImageDatasetReader.from_filenames(rgb_files)

            # TODO: Check depth seems scaled by 256 not 16
            self.depth = ImageDatasetReader.from_filenames(depth_files)

            # Retrieve bounding boxes for each scene
            bboxes = loadmat(meta_file, squeeze_me=True, struct_as_record=True)['bboxes'] \
                     if meta_file is not None else [None] * len(rgb_files)
            self.bboxes = [ [bbox] 
                            if bbox is not None and bbox.size == 1 else bbox 
                            for bbox in bboxes ]
            assert(len(self.bboxes) == len(rgb_files))

        @staticmethod
        def scene_files(files, version): 
            if version == 'v1': 
                depth_files = natural_sort(filter(lambda  fn: '_depth.png' in fn, files))
                rgb_files = natural_sort(list(set(files) - set(depth_files)))
            elif version == 'v2': 
                depth_files = natural_sort(filter(lambda  fn: '-depth.png' in fn, files))
                rgb_files = natural_sort(list(set(files) - set(depth_files)))    
            else: 
                raise ValueError('''Version %s not supported. '''
                                 '''Check dataset and choose either v1 or v2 scene dataset''' % version)
            return rgb_files, depth_files

        @staticmethod
        def meta_files(directory, version): 
            if version == 'v1': 
                mat_files = read_files(os.path.expanduser(directory), pattern='*.mat')
                return dict(((fn.split('/')[-1]).replace('.mat',''), fn) for fn in mat_files)
            else: 
                raise ValueError('''Version %s not supported. '''
                                 '''Check dataset and choose v1 scene dataset''' % version)

        def iteritems(self): 
            print '\n\n===> Version v1 and v2 have discrepancies in depth values, FIX!! <===\n\n'
            for rgb_im, depth_im, bbox in izip(self.rgb.iteritems(), 
                                         self.depth.iteritems(), 
                                         self.bboxes): 
                yield AttrDict(img=rgb_im, depth=depth_im, bbox=bbox if bbox is not None else [])

    def __init__(self, version, directory='', targets=None, num_targets=None, blacklist=['']):         
        if version not in ['v1', 'v2']: 
            raise ValueError('Version not supported. Check dataset and choose either v1 or v2 scene dataset')

        self._dataset = read_dir(os.path.expanduser(directory), pattern='*.png', recursive=False)
        self._meta = UWRGBDSceneDataset._reader.meta_files(directory, version)

        # Instantiate a reader for each of the objects
        self.data = {}
        for key, files in self._dataset.iteritems(): 
            if (targets is not None and key not in targets) or key in blacklist: 
                continue

            # Get mat file for each scene
            meta_file = self._meta.get(key, None)
            print key, meta_file
            # target_id = self.target_hash[key]
            self.data[key] = UWRGBDSceneDataset._reader(files, meta_file, version)

    def iteritems(self): 
        for key, frames in self.data.iteritems(): 
            for frame in frames.iteritems(): 
                yield frame

    @staticmethod
    def annotate(f): 
        vis = f.img.copy()
        for bbox in f.bbox: 
            cv2.rectangle(vis, (bbox['left'], bbox['top']), (bbox['right'], bbox['bottom']), 
                          (50, 50, 50), 2)
            cv2.putText(vis, '%s' % bbox['category'], (bbox['left'], bbox['top']-5), 
                        cv2.FONT_HERSHEY_SIMPLEX, 0.4, (240, 240, 240), thickness = 1)
        return vis


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



def test_uw_rgbd_object(): 
    # Read dataset
    rgbd_data_uw = UWRGBDObjectDataset(directory='~/data/rgbd_datasets/udub/rgbd-object/rgbd-dataset')
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

