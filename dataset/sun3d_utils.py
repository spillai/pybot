import os
import numpy as np

from itertools import izip
from collections import defaultdict
from datetime import datetime
from bot_utils.io_utils import find_files
from bot_utils.db_utils import load_json_dict, save_json_dict

def frame_to_json(bboxes, targets): 
    """
    {'polygon': [{'x': [1,2,3], 'y': [2,3,4], 'object': 3}]}
    Also decorated (see decorate_frame with pretty_names, polygons, targets)
    """

    assert(len(bboxes) == len(targets))
    
    if len(bboxes): 
        bb = bboxes.astype(np.int32)
        return {'polygon': 
                [{'x': [int(b[0]), int(b[0]), int(b[2]), int(b[2])], 
                  'y': [int(b[1]), int(b[3]), int(b[3]), int(b[1])], 
                  'object': int(object_id)} \
                 for object_id, b in zip(targets, bb)]}
    else: 
        return {}

def filter_none(items): 
    return filter(lambda item: item is not None, items)
        
class SUN3DAnnotationFrame(object): 
    def __init__(self, frame=None): 
        """
        Load annotation from json
        """
        self.annotations_ = []

        # Retrieve polygons
        try: 
            polygons = frame['polygon']
        except: 
            return

        # For each polygon
        for poly in polygons: 

            # Get coordinates
            xy = np.vstack([np.float32(poly['x']), 
                            np.float32(poly['y'])]).T

            # Object ID (from local annotation file)
            object_id = poly['object']
            self.add(poly['object'], xy)

    # def __repr__(self): 
    #     return self.annotations_
            
    def add(self, object_id, xy, bbox=None): 
        if bbox is None: 
            bbox = np.int64([xy[:,0].min(), xy[:,1].min(), xy[:,0].max(), xy[:,1].max()])
        self.annotations_.append(dict(object_id=object_id, xy=xy, bbox=bbox))

    @property
    def is_annotated(self): 
        return len(self.annotations_) > 0

    @property
    def num_annotations(self): 
        return len(self.annotations_)

    @property
    def unscaled_bboxes(self): 
        try: 
            return np.vstack([ann['bbox'] for ann in self.annotations_])
        except: 
            return np.empty(shape=(0,4), dtype=np.int64)

    @property
    def unscaled_polygons(self): 
        try: 
            return [np.vstack(ann['xy']) for ann in self.annotations_]
        except: 
            return [np.empty(shape=(0,2), dtype=np.int64) for ann in self.annotations_]

    # @bboxes.setter
    # def bboxes(self, bboxes):
    #     assert(len(bboxes) == self.num_annotations)
    #     for idx in range(self.num_annotations): 
    #         self.annotations_[idx]['bbox']

    @property
    def object_ids(self): 
        return np.int64([ann['object_id'] for ann in self.annotations_])

    # def to_json(): 
    #     """
    #     Write annotation to json
    #     """
    #     pass


class SUN3DAnnotationDB(object):
    """
    SUN3D JSON Format
         {'date': time.time(),
          'name': basename + '/',
          'frames': [{}, {'polygon': [{'x': [1,2,3], 'y': [2,3,4], 'object': 3}]}, {}, {}, ...],
          'objects': [{'name': 'this'}, {'name': 'that'}, {}, ...],
          'conflictList': [null,null, ...],
          'fileList': ['rgb/2394624513.png', ...],
          'img_height': 360, 'img_width': 640}
    """

    def __init__(self, filename, basename, shape=None, data=None):
        if shape is None: 
            import warnings
            warnings.warn('Shape is not set, default scale being used')

        if shape is not None and shape[0] < shape[1]: 
            raise RuntimeError('W > H requirement failed, W: {}, H: {}'.format(shape[0], shape[1]))

        self.filename_ = os.path.expanduser(filename)
        self.basename_ = basename.replace('/','')
        self.shape_ = shape
        self.initialize(data=data)
        
        
    @property
    def initialized(self): 
        return hasattr(self, 'data_')

    def initialize(self, data=None):
        # Initialize data or set data, and
        # check data integrity
        if data is None: 
            if self.shape_ is None: 
                raise ValueError('shape has to be provided for initialization')
            self.data_ = {
                'date': datetime.now().strftime("%a, %d %b %Y %I:%M:%S %Z"),
                'name': self.basename_ + '/',
                'frames': [], 'objects': [], 'conflictList': [],
                'fileList': [], 
                'extrinsics': None, 
                'img_height': self.shape_[1],
                'img_width': self.shape_[0]
            }
        else: 
            self.data_ = data

        # Determine scale (if not already set)
        if self.shape_ is not None: 
            self.scale_ = self.shape_[1] * 1.0 / self.image_height
        else: 
            self.scale_ = 1.0

        # Data integrity check
        assert(self.data_ is not None)
        assert(len(self.data_['fileList']) == len(self.data_['frames']))

        # Generate lookup tables for targets
        # object_name->object_id lookup
        # object_id->object_name lookup
        self._index_objects()

        # Generate lookup tables for filenames
        # img_filename->index (in data['frames']/data['fileList'])
        self._index_files()

        # Generate lookup tables for object annotations
        # object_name->[(frame_index, polygon_index), ... ]
        self._index_object_annotations()

    def set_files(self, files):
        self.data_['fileList'] = files
        self.data_['frames'] = [{} for j in xrange(len(files))]
        self.data_['conflictList'] = [None for j in xrange(len(files))]
        self._index_files()

    def _index_files(self): 
        self.index_ = {fn: idx for (idx, fn) in enumerate(self.files)}

    def set_objects(self, objects): 
        self.data_['objects'] = [{'name': o} for o in objects]
        self._index_objects()

    def _index_objects(self): 
        """
        Generate look up table for objects
        (from the aggregated annotation DB)
        Object ID (oid) -> Object Name (pretty_name)
        """
        self.object_hash_ = { obj: object_id 
                              for (object_id, obj) in enumerate(self.objects) }
        self.object_unhash_ = { object_id: obj 
                                for (object_id, obj) in enumerate(self.objects) }

    def set_frame(self, basename, bboxes, targets):
        index = self.index_[basename]
        self.data_['frames'][index] = frame_to_json(bboxes, targets)
        
    @property
    def annotation_sizes(self): 
        return np.int32([len(f['polygon']) if 'polygon' in f else 0 \
                         for f in self.data_['frames'] ])

    @property
    def annotated_inds(self): 
        " Select all frames that are annotated "
        inds, = np.where(self.annotation_sizes > 0)
        return inds

    @property
    def num_frames(self): 
        return len(self.data_['frames'])

    # @property
    # def has_annotation(self, basename): 
    #     index = self.index_[basename]
    #     frame = self.data_['frames'][index]
        
    @property
    def num_annotations(self): 
        "Return the total number of annotations across all frames" 
        return sum(self.annotation_sizes, 0)

    @property
    def num_frame_annotations(self): 
        "Return the number of frames that have at least 1 annotation" 
        return sum(np.array(self.annotation_sizes, dtype=np.bool).astype(np.int), 0)

    def has_object_name(self, object_name): 
        return object_name in self.object_hash_

    def get_object_name(self, object_id): 
        return self.object_unhash_[object_id]

    def get_object_id(self, object_name): 
        return self.object_hash_[object_name]

    # def _get_object_info(self, object_id): 
    #     """
    #     Get the object information from the object ID in the 
    #     annotations file. Ideally, the object_id should lookup
    #     pretty-name and the expectaion is that the aggregated
    #     pretty-name across datasets are consistent. 

    #     TODO: need to spec this
    #     """
    #     # Trailing number after hyphen is instance id
    #     pretty_name = self.object_unhash_[object_id]
    #     pretty_class_name = ''.join(pretty_name.split('-')[:-1])
    #     # instance_name = self.object_unhash_[object_id].split('-')[-1]

    #     return dict(pretty_name=pretty_name, pretty_class_name=pretty_class_name, 
    #                 class_id=-1, instance_name=instance_name)

    @property
    def scale(self): 
        return self.scale_

    @property
    def name(self): 
        return self.basename_.replace('/', '')

    @property
    def image_height(self): 
        return self.data_['img_height']

    @property
    def image_width(self): 
        return self.data_['img_width']

    @property
    def image_scale(self): 
        lambda height: height * 1.0 / H

    @property
    def objects(self): 
        return map(lambda item: str(item['name']), 
                   filter_none(self.data_['objects']))

    @property
    def num_objects(self): 
        return len(self.objects)

    @property
    def files(self):
        return self.data_['fileList']

    @property
    def num_files(self):
        return len(self.data_['fileList'])

    def _get_prettynames(self, frame): 
        return [self.object_unhash_[oid] for oid in frame.object_ids]

    def _get_targets(self, frame): 
        return np.int64([self.target_hash_[self.object_unhash_[oid]] for oid in frame.object_ids])

    def _get_scaled_polygons(self, frame): 
        try: 
            return [(p * self.scale).astype(np.int64) for p in frame.unscaled_polygons]
        except: 
            return [np.empty(shape=(0,2), dtype=np.int64) for p in frame.unscaled_polygons]

    def _get_scaled_bboxes(self, frame): 
        try: 
            return np.vstack([(bbox * self.scale).astype(np.int64) for bbox in frame.unscaled_bboxes])
        except: 
            return np.empty(shape=(0,4), dtype=np.int64)

    def decorate_frame(self, aframe): 
        aframe.bboxes = self._get_scaled_bboxes(aframe)
        aframe.polygons = self._get_scaled_polygons(aframe)
        aframe.pretty_names = self._get_prettynames(aframe)
        # aframe.targets = self._get_targets(aframe)
        return aframe

    def get_name(self, index): 
        return self.data_['fileList'][index]

    def get_frame(self, index): 
        frame = SUN3DAnnotationFrame(self.data_['frames'][index])
        return self.decorate_frame(frame)

    def __contains__(self, basename): 
        return basename in self.index_

    def __getitem__(self, basename): 
        index = self.index_[basename]
        return self.get_frame(index)

    def __setitem__(self, basename, frame): 
        " Write back to data_['frames'] "
        assert(isinstance(frame, SUN3DAnnotationFrame))
        index = self.index_[basename]
        # self.data_['frames'][index]
        pass

    def _index_object_annotations(self): 
        """
        Generate look up tables for object
        annotations from the DB. 
        obj_name->[(frame_index,polygon_index), ...]
        """
        self.object_annotations_index_ = defaultdict(list)
        for frame_index, frame in izip(self.annotated_inds, 
                                       self.iterframes(self.annotated_inds)): 
            for polygon_index, obj_name in enumerate(frame.pretty_names): 
                self.object_annotations_index_[obj_name].\
                    append((self.get_name(frame_index), polygon_index))

    @property
    def object_annotations(self): 
        return dict(self.object_annotations_index_)

    def iterframes(self, frame_inds): 
        return (self.get_frame(ind) for ind in frame_inds)

    # def iterannotations(self, frame_inds, polygon_inds): 
    #     return (frame.bboxes[pind] 
    #             for pind, frame in izip(polygon_inds, self.iterframes(frame_inds))    

    def filter_target_name(self, pretty_names, target_name=None): 
        return \
            filter(lambda name: target_name in name, pretty_names) \
            if target_name is not None else pretty_names

    def find_object_annotations(self, target_name): 
        """
        Find annotations by target_name/pretty_name. 
        Returns the index and the associated polygon index, 
        for bbox lookup. [(frame_index, polygon_index), ... ]
        returns: frame_inds, polygon_inds
        """
        if self.object_annotations_index_ is None: 
            raise RuntimeError('Cannot find annotations,'
                               'objectdb has not been indexed yet')

        if not isinstance(target_name, str): 
            raise TypeError('target_name has to be str: provided {}'\
                            .format(type(target_name)))
        
        frame_keys, polygon_inds = [], []
        for object_name, items in self.object_annotations_index_.iteritems(): 
            if target_name in object_name:
                fkeys, pinds = zip(*items)
                frame_keys.extend(fkeys)
                polygon_inds.extend(pinds)
        
        return frame_keys, polygon_inds

    def list_annotations(self, target_name=None): 
        inds = self.annotated_inds
        return [ filter(
            lambda frame: 
            filter_target_name(frame.pretty_names, target_name=target_name), 
            self.iterframes(inds)) ]

    @property
    def frames(self): 
        return map(SUN3DAnnotationFrame, self.data_['frames'])

    def save(self): 
        save_json_dict(self.filename_, self.data_)

    @classmethod
    def load(cls, folder, shape=None): 
        filename = os.path.join(os.path.expanduser(folder), 
                                'annotation/index.json')
        _, basename = os.path.split(folder.rstrip('/'))
        data = load_json_dict(filename)
        
        c = cls(filename, basename, shape=shape, data=data)
        return c

class SUN3DObjectDB(object): 
    def __init__(self, directory): 
        files = filter(lambda fn: os.path.splitext(fn)[1] == '.json', \
                       find_files(os.path.expanduser(directory), 
                                  contains='index.json'))

        directories = map(lambda fn: 
                          fn.replace('/annotation/index.json',''), files)
        
        self.target_hash_ = {}
        self.target_unhash_ = {}
        self.annotation_db_ = {}

        # Create object look up table
        # (default background class available)
        self.objects_ = set(['background'])
        for d in directories: 
            basename = os.path.basename(d)
            # print 'Processing', basename, d

            # Insert objects into unified set
            db = SUN3DAnnotationDB.load(d)
            self.objects_.update(db.objects)
            self.annotation_db_[basename] = db

        # Unified object DB
        self.target_unhash_ = {oid: name for oid, name in enumerate(self.objects_)}
        self.target_hash_ = {name: oid for oid, name in enumerate(self.objects_)}
        
        print('Total objects {}'.format(len(self.target_hash_)))

    @property
    def target_hash(self): 
        return self.target_hash_

    @property
    def target_unhash(self): 
        return self.target_unhash_

    @property
    def objects(self): 
        return list(self.objects_)

    def get_object_ids(self, object_name): 
        """
        Returns a look-up-table with dataset->object_id for 
        the requested object_name
        """
        lut = {}
        for basename, db in self.annotation_db_.iteritems():
            if db.has_object_name(object_name): 
                lut[basename] = db.get_object_id(object_name)
        return lut

    @property
    def num_datasets(self): 
        return len(self.annotation_db_)

    @property
    def datasets(self): 
        return self.annotation_db_.keys()

    def get_target_id(self, object_name): 
        return self.target_hash_[object_name]

    def get_category_id(self): 
        raise NotImplementedError()

    def get_category_name(self): 
        raise NotImplementedError()

    def get_instance_id(self): 
        raise NotImplementedError()

    def get_instance_name(self): 
        raise NotImplementedError()

if __name__ == "__main__": 
    import argparse
    parser = argparse.ArgumentParser(
        description='Load sun3d annotations')
    parser.add_argument(
        '-d', '--directory', type=str, required=True, 
        default=None, help='Annotated image directory')
    args = parser.parse_args()

    shape = (640,360)
    db = SUN3DAnnotationDB.load(args.directory, shape)
    frames = db.frames
    # print db.num_annotations, db.num_files, db.num_objects, db.name, db.objects
    # print frames, db.image_width, db.image_height
    files = db.files
    print db[files[2996]]

    # import ipdb; ipdb.set_trace()
