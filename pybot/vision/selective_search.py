from pprint import pprint
from skdata.pascal import VOC2007

# {'filename': '/home/spillai/.skdata/pascal/VOC2007/VOCdevkit/VOC2007/JPEGImages/000001.jpg',
#  'id': '000001',
#  'objects': [{'bounding_box': {'x_max': 194,
#                               'x_min': 47,
#                                'y_max': 370,
#                                'y_min': 239},
#               'difficult': False,
#               'name': 'dog',
#               'pose': 'Left',
#               'truncated': True},
#              {'bounding_box': {'x_max': 351,
#                                'x_min': 7,
#                                'y_max': 497,
#                                'y_min': 11},
#               'difficult': False,
#               'name': 'person',
#               'pose': 'Left',
#               'truncated': True}],
#  'owner': {'flickrid': 'Fried Camels', 'name': 'Jinky the Fruit Bat'},
#  'segmented': True,
#  'sha1': '44b44bb98478db4d706c9700eaf22241dd0a7b4a',
#  'shape': {'depth': 3, 'height': 500, 'width': 353},
#  'source': {'annotation': 'PASCAL VOC2007',
#             'database': 'The VOC2007 Database',
#             'flickrid': '341012865',
#             'image': 'flickr'},
#  'split': 'test'
# }

__train_classes__ = ['bird', 'car', 'cat', 'cow', 'dog', 'sheep']

if __name__ == "__main__": 
    voc = VOC2007()
    for data in voc.meta: 
        print pprint(data)
        break
