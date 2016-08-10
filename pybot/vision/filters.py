
from .recognition_utils import ImageClassifier, ImageDescription

class ImageDescriptionEstimator(TransformerMixin):
    def __init__(self, **kwargs):
        self.estimator = ImageDescription(**kwargs)

    def fit(self, *args, **kwargs):
        return self

    def transform(self, *args, **kwargs):
        kpts, desc = self.estimator.detect_and_describe(*args, **kwargs)
        return (np.vstack([kp.pt for kp in kpts])).astype(np.int32), desc

# class ImageDescriptionTransformer(TransformerMixin):
#     def transform(self, X, **transform_params):
#         hours = DataFrame(X['datetime'].apply(lambda x: x.hour))
#         return hours

#     def fit(self, X, y=None, **fit_params):
#         return self
