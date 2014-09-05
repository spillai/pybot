
import cv2
from sklearn.pipeline import Pipeline
from sklearn.base import BaseEstimator, TransformerMixin

class OpticalFlowFilter(BaseEstimator): 
    def __init__(self, *args, **kwargs): 
        OpticalFlowFilter.__init__(self, *args, **kwargs)

    def fit(self, *): 
        return self

    def transform(self, d): 
        return self

