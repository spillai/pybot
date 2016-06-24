import os
import numpy as np

from datetime import datetime
from bot_utils.db_utils import load_json_dict

class SUN3DAnnotation(object):
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

    def __init__(self, filename, basename):
        self.filename_ = os.path.expanduser(filename)
        self.basename_ = basename.replace('/','')
        self.initialize()

    @property
    def initialized(self): 
        return hasattr(self, 'data_')

    def initialize(self, data=None):
        # Initialize data or set data, and
        # check data integrity
        if data is None: 
            self.data_ = {
                'date': datetime.now().strftime("%a, %d %b %Y %I:%M:%S %Z"),
                'name': self.basename_ + '/',
                'frames': [], 'objects': [], 'conflictList': [],
                'fileList': [], 'img_height': 0, 'img_width': 0 
            }
        else: 
            self.data_ = data
        
        # Data integrity check
        assert(self.data_ is not None)

        # Generate lookup tables for targets
        self._generate_target_lut()

    def _generate_target_lut(self): 
        """
        Generate look up table for objects
        (from the aggregated annotation DB)
        """
        print self.objects
        self.target_hash_ = { obj: object_id 
                              for (object_id, obj) in enumerate(self.objects) }
        self.target_unhash_ = { object_id: obj 
                                for (object_id, obj) in enumerate(self.objects) }
        
    def _get_object_info(self, object_id): 
        """
        Get the object information from the object ID in the 
        annotations file. Ideally, the object_id should lookup
        pretty-name and the expectaion is that the aggregated
        pretty-name across datasets are consistent. 

        TODO: need to spec this
        """
        # Trailing number after hyphen is instance id
        pretty_name = self.target_unhash_[object_id]
        pretty_class_name = ''.join(pretty_name.split('-')[:-1])
        instance_name = self.target_unhash_[object_id].split('-')[-1]

        return dict(pretty_name=pretty_name, pretty_class_name=pretty_class_name, 
                    class_id=-1, instance_name=instance_name)

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
        return map(lambda item: item['name'], self.data_['objects'])

    @property
    def num_objects(self): 
        return len(self.data_['objects'])

    @property
    def num_files(self):
        return len(self.data_['fileList'])

    @property
    def num_annotations(self): 
        return sum([len(frame) for frame in self.data_['frames']], 0)

    def parse_frame(self, frame):
        try: 
            polygons = frame['polygon']
        except: 
            return []

        # For each polygon
        annotations = []
        for poly in polygons: 

            # Get coordinates
            xy = np.vstack([np.float32(poly['x']), 
                            np.float32(poly['y'])]).T
            print xy
            # Object label as described by target hash
            object_id = poly['object']

            object_info = self._get_object_info(object_id)
            object_info['xy'] = xy
            object_info['bbox'] = 

            sbbox = bbox['polygon'] * self.get_image_scale(H)
            bboxes[idx]['polygon'] = sbbox
            bboxes[idx]['bbox'] = np.int64([sbbox[:,0].min(), sbbox[:,1].min(), sbbox[:,0].max(), sbbox[:,1].max()])

            annotations.append(object_info)

        return annotations

    @property
    def frames(self): 
        return map(self.parse_frame, self.data_['frames'])

    # def get_bboxes(self, index=None):
    #     if index is None: 
            

    def save(self, filename): 
        save_json_dict(self.filename_.replace('index.json', 'index_new.json'), self.data_)

    @classmethod
    def load(cls, folder): 
        filename = os.path.join(os.path.expanduser(folder), 'annotation/index.json')
        _, basename = os.path.split(folder.rstrip('/'))
        
        c = cls(filename, basename)

        data = load_json_dict(filename)
        c.initialize(data)

        return c


if __name__ == "__main__": 
    import argparse
    parser = argparse.ArgumentParser(
        description='Load sun3d annotations')
    parser.add_argument(
        '-d', '--directory', type=str, required=True, 
        default=None, help='Annotated image directory')
    args = parser.parse_args()

    db = SUN3DAnnotation.load(args.directory)
    print db.num_annotations, db.num_files, db.num_objects, db.name, db.objects
    frames = db.frames
    print frames
