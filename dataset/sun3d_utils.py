import os
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
        self.data_ = {'date': datetime.now().strftime("%a, %d %b %Y %I:%M:%S %Z"),
                      'name': self.basename_ + '/',
                      'frames': [], 'objects': [], 'conflictList': [],
                      'fileList': [], 'img_height': 0, 'img_width': 0}

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
    def objects(self): 
        return self.data_['objects']

    @property
    def num_objects(self): 
        return len(self.data_['objects'])

    @property
    def num_files(self):
        return len(self.data_['fileList'])

    @property
    def num_annotations(self): 
        return sum([len(frame) for frame in self.data_['frames']], 0)

    def save(self, filename): 
        save_json_dict(self.filename_.replace('index.json', 'index_new.json'), self.data_)

    @classmethod
    def load(cls, folder): 
        filename = os.path.join(os.path.expanduser(folder), 'annotation/index.json')
        _, basename = os.path.split(folder.rstrip('/'))
        
        c = cls(filename, basename)
        c.data_ = load_json_dict(filename)
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

    
