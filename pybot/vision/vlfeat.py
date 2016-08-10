import time
import numpy as np

from pybot_vlfeat import vl_dsift
# from pyvlfeat import vl_dsift
# from vlfeat.vl_phow.vl_phow as vl_phow

if __name__ == '__main__': 
    import cv2
    im = cv2.imread('/home/spillai/HD1/data/cv_datasets/caltech_101/101_ObjectCategories/Faces/image_0001.jpg')

    st = time.time()
    pts, desc = vl_dsift(im)
    print desc
    print 'Time %5.3f s' % (time.time() - st)
