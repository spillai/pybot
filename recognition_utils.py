import numpy as np
import cv2, os, time, random

from sklearn.linear_model import SGDClassifier
from sklearn.neighbors import KNeighborsClassifier

from sklearn.cross_validation import cross_val_score
from sklearn.ensemble import ExtraTreesClassifier, RandomForestClassifier
from sklearn.svm import LinearSVC
from sklearn.multiclass import OneVsRestClassifier
from sklearn.cross_validation import train_test_split
from sklearn.externals import joblib



from bot_vision.image_utils import im_resize
from bot_utils.io_utils import memory_usage_psutil
from bot_utils.db_utils import AttrDict, AttrDictDB
from bot_vision.bow_utils import BOWTrainer

import bot_vision.mser_utils as mser_utils
import bot_utils.io_utils as io_utils
import sklearn.metrics as metrics

# def selective_search_mask(frame, pts): 



class ImageDescription(object): 
    def __init__(self, detector='SIFT', descriptor='SIFT', selective_search=False, step=4, levels=4, scale=2.0): 
        self.dense = dense
        self.step = step
        self.levels = levels
        self.scale = scale

        # Setup feature detector
        if detector == 'dense'
            # self.detector = cv2.PyramidAdaptedFeatureDetector(, maxLevel=levels)
            self.detector = cv2.FeatureDetector_create('Dense')
            self.detector.setInt('initXyStep', step)
            self.detector.setDouble('featureScaleMul', 2.0)
            self.detector.setInt('featureScaleLevels', levels)
            self.detector.setBool('varyImgBoundWithScale', False)
            self.detector.setBool('varyXyStepWithScale', True)
        else: 
            self.detector = cv2.FeatureDetector_create(detector)

        # Setup feature extractor
        self.extractor = cv2.DescriptorExtractor_create(descriptor)

        # Setup selective search
        if selective_search: 
            self.sel_search = mser_utils.MSER()

    def describe(self, im, mask=None): 
        """
        Computes dense/sparse features on an image and describes 
        these keypoints using a feature descriptor
        returns 
           kpts: [cv2.KeyPoint, ... ] 
           desc: [N x D]
        """
        kpts = self.detector.detect(im, mask=mask)
        kpts, desc = self.extractor.compute(im, kpts)
        return desc.astype(np.uint8)

class ImageClassifier(object): 
    """
    Object recognition class from training data
    Attributes: 
      data:         [image_fn1, ...  ]
      target:       [class_id, ... ]
      target_ids:   [0, 1, 2, ..., 101]
      target_names: [car, bike, ... ]
    """

    def __init__(self, dataset, 
                 test_size=10, 
                 training_args = dict(train_size=10, random_state=1),
                 descriptor_args = dict(dense=True, descriptor='SIFT'), 
                 bow_args = dict(K=64, method='vlad', norm_method='square-rooting'), 
                 cache_args = AttrDict(vocab_path='vocab', clf_path='clf', overwrite=False)): 
        # Save dataset
        self.dataset = dataset
        self.cache_args = cache_args
        print 'Memory usage at __init__ start %5.2f MB' % (memory_usage_psutil())

        # Bag-of-words VLAD/VQ
        if cache_args.overwrite or not io_utils.path_exists(cache_args.clf_path): 
            self.bow = BOWTrainer(**bow_args)
        else: 
            self.bow = joblib.load(cache_args.vocab_path)            

        # Image description using Dense SIFT/Descriptor
        self.image_descriptor = ImageDescription(**descriptor_args)

        # Split up training and test
        self.X_train, self.X_test, self.y_train, self.y_test = [], [], [], []

        print np.unique(self.dataset.target)
        for y_t in np.unique(self.dataset.target): 

            inds, = np.where(self.dataset.target == y_t)
            data, target = self.dataset.data[inds], self.dataset.target[inds]

            # Split train, and test (complement of test)
            X_train, X_test, y_train, y_test = train_test_split(data, target, **training_args)

            # Only allow max testing size
            self.X_train.extend(X_train), self.X_test.extend(X_test[:test_size])
            self.y_train.extend(y_train), self.y_test.extend(y_test[:test_size])

        # Setup classifier
        self.clf_pretrained = False
        if not io_utils.path_exists(cache_args.clf_path) or cache_args.overwrite: 
            print 'Building Classifier'
            # self._clf = KNeighborsClassifier(n_neighbors=10)
            self._clf = LinearSVC()
            self.clf = OneVsRestClassifier(self._clf, n_jobs=1)
            # self.clf = ExtraTreesClassifier(n_estimators=10, 
            #                                 max_depth=3, min_samples_split=1, random_state=0)
        else: 
            print 'Loading Classifier'
            self.clf = joblib.load(cache_args.clf_path)
            self.clf_pretrained = True

        print 'Memory usage at __init__ completed %5.2f MB' % (memory_usage_psutil())

    def train(self): 
        if self.clf_pretrained: 
            return 

        st = time.time()

        print 'Memory usage at pre-train %5.2f MB' % (memory_usage_psutil())

        # Extract features
        train_desc = [self.image_descriptor.describe(cv2.imread(x_t)) for x_t in self.X_train]
        print 'Descriptor extraction took %5.3f s' % (time.time() - st)    

        print 'Memory usage at post-describe %5.2f MB' % (memory_usage_psutil())

        # Build BOW
        self.bow.build(train_desc)
        print 'Codebook: %s' % ('GOOD' if np.isfinite(self.bow.vectorizer.codebook).all() else 'BAD')

        print 'Memory usage at post-bow-build %5.2f MB' % (memory_usage_psutil())

        # Histogram of trained features
        train_target = np.array(self.y_train)
        train_histogram = np.vstack([self.bow.project(desc) for desc in train_desc])

        print 'Memory usage at post-project %5.2f MB' % (memory_usage_psutil())

        # Train/Predict one-vs-all classifier
        self.clf.fit(train_histogram, train_target)
        pred_target = self.clf.predict(train_histogram)

        print 'Memory usage at post-fit %5.2f MB' % (memory_usage_psutil())

        # # Save to DB
        # joblib.dump(self.bow, self.cache_args.vocab_path)
        # joblib.dump(self.clf, self.cache_args.clf_path)

        print ' Accuracy score (Training): %4.3f' % (metrics.accuracy_score(train_target, pred_target))
        print ' Report (Training):\n %s' % (metrics.classification_report(train_target, pred_target, 
                                                                          target_names=self.dataset.target_names))

        print 'Memory usage at post-predict %5.2f MB' % (memory_usage_psutil())
        print 'Training took %5.3f s' % (time.time() - st)
        return

    def classify(self): 
        st = time.time()

        # Extract features
        test_desc = [ self.image_descriptor.describe(cv2.imread(x_t)) for x_t in self.X_test ]

        print 'Descriptor extraction took %5.3f s' % (time.time() - st)    
    
        # Histogram of trained features
        test_target = self.y_test
        test_histogram = np.vstack([self.bow.project(desc) for desc in test_desc])

        pred_target = self.clf.predict(test_histogram)

        print ' Accuracy score (Test): %4.3f' % (metrics.accuracy_score(test_target, pred_target))
        print ' Report (Test):\n %s' % (metrics.classification_report(test_target, pred_target, 
                                                                      target_names=self.dataset.target_names))

        print 'Testing took %5.3f s' % (time.time() - st)

        return test_histogram, test_target, pred_target
        
