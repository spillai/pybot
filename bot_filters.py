"""
This implements a templated estimator/filter factory design pattern for rapid 
filter instantiation and chaining
"""
# Author: Sudeep Pillai
# Licence: TODO

def make_estimator(name, fit_cb=None, transform_cb=None, base=object, estimator_kwargs={}): 
    """
    Templated estimator/filter designed to work with the scikit-learn pipline; 
    This class is mostly meant to enable rapid prototyping with existing functions, 
    and classes that are designed to perform simple filter-like transformations. 

    For. eg. a simple pipeline consisting of 
    'image (source) => resize (320x240) => gaussian blur => output (sink)' can be 
    streamlined by creating resize and gaussian blur filters that simply transform
    the image by applying the standard cv operations    
    """

    # Ensure either transform or fit is defined, other-wise complain
    if not transform_cb and not fit_cb: 
        raise KeyError('make_estimator requires at least fit_cb or transform_cb to be defined')

    # Templated estimator
    # returns self if fit/transform is not defined
    class TemplatedEstimator(base): 
        estimator_name = name

        def __init__(self):
            pass

        if transform_cb is not None: 
            def transform(self, *args): 
                return transform_cb(*args, **estimator_kwargs)
        else: 
            def transform(self, *args): 
                return self

        if fit_cb is not None: 
            def fit(self, X): 
                return fit_cb(X, **estimator_kwargs)
        else: 
            def fit(self, *args): 
                return self
                
    return TemplatedEstimator

def get_estimator(func, **kwargs): 
    """
    This function provides a simple way to create scikit-learn compatible estimators
    from existing utility functions / cv operations
    """
    return make_estimator(name=''.join(['regfilter_', func.func_name]), 
                          transform_cb=func, estimator_kwargs=kwargs)()

if __name__ == "__main__": 

    import os, cv2
    import cv2
    import numpy as np
    from sklearn.pipeline import Pipeline, FeatureUnion

    import bot_vision.image_utils as image_utils
    import bot_vision.stereo_utils as stereo_utils

    def im_read_filter(fn): 
        return cv2.imread(os.path.expanduser(fn))

    
    fn = '/home/spillai/data/dataset/sequences/01/image_0/000000.png'
    print 'im_read_filter: Reading image '
    im1 = im_read_filter(fn)

    print 'Estimator::im_read_filter: Reading image'
    est = get_estimator(im_read_filter)
    im2 = est.transform(fn)

    print 'Estimator::im_resize: Resizing image with args scale=0.5'
    est = get_estimator(image_utils.im_resize, scale=0.5)
    im3 = est.transform(im1)


    # def ImageResizeFilter(**kwargs): 
    #     return make_estimator(name='ImageResizeFilter', transform_cb=image_utils.im_resize, **kwargs)()

    # def SGBMFilter(**kwargs): 
    #     return make_estimator(name='SGBMFilter', transform_cb=stereo_utils.StereoSGBM().compute, **kwargs)()

    # pp = Pipeline([('image_resize_down', ImageResizeFilter(scale=0.5)), 
    #                ('image_resize_up', ImageResizeFilter(scale=4.0))])
    # # stereo_union = FeatureUnion([('left_stereo', pp_pipeline), 
    # #                              ('right_stereo', pp_pipeline)])
    # # stereo_pipeline = Pipeline([('stereo_union', stereo_union), 
    # #                             ()stereo_utils])


    # # stereo_pipeline = Pipeline([('image_resize', ImageResizeFilter(scale=0.5))])
    # # stereo_pipeline = Pipeline([])
    # #                                     ('stereo_disparity', SGBMFilter())

    # left = cv2.imread(os.path.expanduser('~/data/dataset/sequences/08/image_0/000000.png'), 0)
    # right = cv2.imread(os.path.expanduser('~/data/dataset/sequences/08/image_1/000000.png'), 0)

    # resize = ImageResizeFilter(scale=0.5)
    # left_scaled = pp.transform(left) # resize.transform(left)
    # right_scaled = pp.transform(right) # resize.transform(right)

    # print left.shape, left_scaled.shape, right.shape, right_scaled.shape

    # disp = SGBMFilter().transform(left_scaled, right_scaled)
    # print disp.shape


    # cv2.imshow('disp', disp.astype(np.float32) / 64)
    # cv2.imshow('original', left)
    # cv2.imshow('scaled', left_scaled)
    # cv2.waitKey(0)


