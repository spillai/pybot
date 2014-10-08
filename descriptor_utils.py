import cv2
import numpy as np

class ImageDescription(object): 
    def __init__(self, descriptor='SIFT', dense=True, step=4): 
        if dense: 
            self.detector = cv2.FeatureDetector_create('Dense')
            self.detector.setInt('initXyStep', step)
        else: 
            self.detector = cv2.FeatureDetector_create('FAST')

        self.extractor = cv2.DescriptorExtractor_create(descriptor)

        # self.matcher = cv2.DescriptorMatcher_create("FlannBased")

    # def set_vocabulary(self, vocab): 
    #     self.matcher.add(vocab)

    def describe(self, im): 
        """
        Computes dense/sparse features on an image and describes 
        these keypoints using a feature descriptor
        returns 
           kpts: [cv2.KeyPoint, ... ] 
           desc: [N x D]
        """
        kpts = self.detector.detect(im)
        kpts, desc = self.extractor.compute(im, kpts)
        return desc
