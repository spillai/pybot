import cv2
import numpy as np
from sklearn.pipeline import Pipeline, FeatureUnion
from sklearn.base import BaseEstimator, TransformerMixin

import bot_vision.image_utils as image_utils
import bot_vision.stereo_utils as stereo_utils
# def make_base_estimator(**kwargs):
#     name = kwargs.pop('name', None)
#     if not isinstance(name, str): 
#         raise AttributeError('make_estimator requires name to be set!')
#     return type(name, (BaseEstimator,), dict(**kwargs))

def make_estimator(**kwargs): 

    # Extract the transform and fit callbacks for the estimator
    transform_cb = kwargs.pop('transform_cb', None)
    fit_cb = kwargs.pop('fit_cb', None)

    # Ensure at least one of them is set, other-wise complain
    if not transform_cb and not fit_cb: 
        raise KeyError('make_estimator requires at least fit_cb or transform_cb to be set')

    # Templated estimator
    # returns self if fit/transform is not defined
    class TemplatedEstimator: 
        _name_ = kwargs.pop('name', 'templated_estimator')

        def __init__(self):
            pass

        if transform_cb is not None: 
            def transform(self, *args): 
                return transform_cb(*args, **kwargs)
        else: 
            def transform(self, *args): 
                return self

        if fit_cb is not None: 
            def fit(self, X): 
                return fit_cb(X, **kwargs)
        else: 
            def fit(self, *args): 
                return self
                
    return TemplatedEstimator


def ImageResizeFilter(**kwargs): 
    return make_estimator(name='ImageResizeFilter', transform_cb=image_utils.im_resize, **kwargs)()

def SGBMFilter(**kwargs): 
    return make_estimator(name='SGBMFilter', transform_cb=stereo_utils.StereoSGBM().compute, **kwargs)()

if __name__ == "__main__": 

    import os

    pp = Pipeline([('image_resize_down', ImageResizeFilter(scale=0.5)), 
                   ('image_resize_up', ImageResizeFilter(scale=4.0))])
    # stereo_union = FeatureUnion([('left_stereo', pp_pipeline), 
    #                              ('right_stereo', pp_pipeline)])
    # stereo_pipeline = Pipeline([('stereo_union', stereo_union), 
    #                             ()stereo_utils])


    # stereo_pipeline = Pipeline([('image_resize', ImageResizeFilter(scale=0.5))])
    # stereo_pipeline = Pipeline([])
    #                                     ('stereo_disparity', SGBMFilter())

    left = cv2.imread(os.path.expanduser('~/data/dataset/sequences/08/image_0/000000.png'), 0)
    right = cv2.imread(os.path.expanduser('~/data/dataset/sequences/08/image_1/000000.png'), 0)

    resize = ImageResizeFilter(scale=0.5)
    left_scaled = pp.transform(left) # resize.transform(left)
    right_scaled = pp.transform(right) # resize.transform(right)

    print left.shape, left_scaled.shape, right.shape, right_scaled.shape

    disp = SGBMFilter().transform(left_scaled, right_scaled)
    print disp.shape


    cv2.imshow('disp', disp.astype(np.float32) / 64)
    cv2.imshow('original', left)
    cv2.imshow('scaled', left_scaled)
    cv2.waitKey(0)


