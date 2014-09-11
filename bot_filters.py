import cv2
from sklearn.pipeline import Pipeline
from sklearn.base import BaseEstimator, TransformerMixin

def make_estimator(**kwargs):
    name = kwargs.get('name', None)
    if not isinstance(name, str): 
        raise AttributeError('make_estimator requires name to be set!')
    return type(name, (BaseEstimator,), dict(**kwargs))

def ImageResizeFilter(**kwargs): 
    return make_estimator(name='ImageResizeFilter', 
                          transform=lambda im: 
                          cv2.resize(im, None,  fx=kwargs.get(scale), 
                                     fy=kwargs.get(scale), 
                                     interpolation=kwargs.get(interpolation, 
                                                              cv2.INTER_LINEAR)))


if __name__ == "__main__": 
    ImageResizeFilter(scale=0.5)
    


# class OpticalFlowFilter(BaseEstimator): 
#     def __init__(self, *args, **kwargs): 
#         OpticalFlowFilter.__init__(self, *args, **kwargs)

#     def fit(self, *): 
#         return self

#     def transform(self, d): 
#         return self



