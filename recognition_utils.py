import numpy as np
import cv2, os, time, random
import itertools

from sklearn.linear_model import SGDClassifier
from sklearn.neighbors import KNeighborsClassifier
from sklearn.svm import LinearSVC
from sklearn.multiclass import OneVsRestClassifier
from sklearn.cross_validation import train_test_split

from bot_utils.io_utils import memory_usage_psutil
from bot_utils.db_utils import AttrDict, AttrDictDB

from bot_vision.image_utils import im_resize, gaussian_blur, median_blur, box_blur
from bot_vision.bow_utils import BoWVectorizer
import bot_vision.mser_utils as mser_utils

import bot_utils.io_utils as io_utils
import sklearn.metrics as metrics

class ImageDescription(object): 
    def __init__(self, detector='dense', descriptor='SIFT', step=4, levels=4, scale=2.0): 
        self.step = step
        self.levels = levels
        self.scale = scale

        # Setup feature detector
        if detector == 'dense': 
            self.detector = cv2.FeatureDetector_create('Dense')
            self.detector.setInt('initXyStep', step)
            self.detector.setDouble('featureScaleMul', scale)
            self.detector.setInt('featureScaleLevels', levels)
            self.detector.setBool('varyImgBoundWithScale', True)
            self.detector.setBool('varyXyStepWithScale', False)
        else: 
            # self.detector = cv2.PyramidAdaptedFeatureDetector(, maxLevel=levels)
            self.detector = cv2.FeatureDetector_create(detector)

        # Setup feature extractor
        self.extractor = cv2.DescriptorExtractor_create(descriptor)

    def detect_and_describe(self, img, mask=None): 
        """
        Computes dense/sparse features on an image and describes 
        these keypoints using a feature descriptor
        returns 
           kpts: [cv2.KeyPoint, ... ] 
           desc: [N x D]
        """

        try: 
            # Descriptor extraction
            kpts = self.detector.detect(img, mask=mask)
            kpts, desc = self.extractor.compute(img, kpts)

            # Extract color information (Lab)
            pts = np.vstack([kp.pt for kp in kpts]).astype(np.int32)
            imgc = cv2.cvtColor(img, cv2.COLOR_BGR2LAB)
            # lab = median_blur(imgc, size=5) 
            cdesc = imgc[pts[:,1], pts[:,0]]

            # vis = lab.copy()
            # vis[pts[:,1], pts[:,0]] = 255
            # from bot_vision.imshow_utils import imshow_cv
            # imshow_cv('vis', vis, block=True)

            return kpts, np.hstack([desc, cdesc])

        except: 
            return None, None

    def describe(self, img, mask=None): 
        """
        Computes dense/sparse features on an image and describes 
        these keypoints using a feature descriptor
        returns 
           kpts: [cv2.KeyPoint, ... ] 
           desc: [N x D]
        """
        kpts, desc = self.detect_and_describe(img, mask=mask)
        return desc

class ImageClassifier(object): 
    """
    Object recognition class from training data
    Attributes: 
      data:         [image_fn1, ...  ]
      target:       [class_id, ... ]
      target_ids:   [0, 1, 2, ..., 101]
      target_names: [car, bike, ... ]
    """
    
    training_params = AttrDict(train_size=10, random_state=1)
    descriptor_params = AttrDict(detector='dense', descriptor='SIFT', step=2, levels=4, scale=2)
    bow_params = AttrDict(K=64, method='vlad', quantizer='kdtree', norm_method='square-rooting')
    cache_params = AttrDict(detector_path='detector.h5', overwrite=False)
    default_params = AttrDict(
        training=training_params, descriptor=descriptor_params, bow=bow_params, cache=cache_params
    )
    def __init__(self, dataset=None, 
                 max_test_size=10, 
                 process_cb=lambda fn: dict(img=cv2.imread(fn), mask=None), 
                 params = default_params): 

        # Save dataset
        self.dataset = dataset
        self.process_cb = process_cb
        self.params = AttrDict(params)
        print 'Memory usage at __init__ start %5.2f MB' % (memory_usage_psutil())

        # Optionally setup training testing
        if dataset is not None: 
            self.setup_training_testing(max_test_size)

            # Persist, and retrieve db if available
            if not io_utils.path_exists(self.params.cache.detector_path) or self.params.cache.overwrite: 
                self.setup_recognition()
                self.clf_pretrained = False
            else: 
                db = AttrDict.load(self.params.cache.detector_path)
                self.setup_recognition_from_dict(db)
                self.clf_pretrained = True

    def setup_recognition(self): 
        # Bag-of-words VLAD/VQ
        self.bow = BoWVectorizer(**self.params.bow)
        
        # Image description using Dense SIFT/Descriptor
        self.image_descriptor = ImageDescription(**self.params.descriptor)

        # Setup classifier
        print 'Building Classifier'
        # self._clf = KNeighborsClassifier(n_neighbors=10)
        self._clf = LinearSVC()
        self.clf = OneVsRestClassifier(self._clf, n_jobs=1)
        # self.clf = ExtraTreesClassifier(n_estimators=10, 
        #                                 max_depth=3, min_samples_split=1, random_state=0)

        print 'Memory usage at __init__ completed %5.2f MB' % (memory_usage_psutil())


    def setup_training_testing(self, max_test_size): 
        # Split up training and test
        self.X_train, self.X_test, self.y_train, self.y_test = [], [], [], []

        print np.unique(self.dataset.target)
        for y_t in np.unique(self.dataset.target): 

            inds, = np.where(self.dataset.target == y_t)
            data, target = self.dataset.data[inds], self.dataset.target[inds]

            # Split train, and test (complement of test)
            X_train, X_test, y_train, y_test = train_test_split(data, target, **self.params.training)

            # Only allow max testing size
            self.X_train.extend(X_train), self.X_test.extend(X_test[:max_test_size])
            self.y_train.extend(y_train), self.y_test.extend(y_test[:max_test_size])

        self.X_all = list(itertools.chain(self.X_train, self.X_test))
        self.y_all = list(itertools.chain(self.y_train, self.y_test))

    def train(self): 
        if self.clf_pretrained: 
            return 

        if not hasattr(self, 'X_train') or not hasattr(self, 'y_train'): 
            raise RuntimeError('Training cannot proceed. Setup training and testing samples first!')            

        print '===> Training '
        st = time.time()

        print 'Memory usage at pre-train %5.2f MB' % (memory_usage_psutil())

        # Extract features
        train_desc = [self.image_descriptor.describe(**self.process_cb(x_t)) for x_t in self.X_train]
        print 'Descriptor extraction took %5.3f s' % (time.time() - st)    

        print 'Memory usage at post-describe %5.2f MB' % (memory_usage_psutil())

        # Build BOW
        self.bow.build(train_desc)
        print 'Codebook: %s' % ('GOOD' if np.isfinite(self.bow.codebook).all() else 'BAD')

        print 'Memory usage at post-bow-build %5.2f MB' % (memory_usage_psutil())

        # Histogram of trained features
        train_target = np.array(self.y_train, dtype=np.int32)
        train_histogram = np.vstack([self.bow.project(desc) for desc in train_desc])

        print 'Memory usage at post-project %5.2f MB' % (memory_usage_psutil())

        # Train/Predict one-vs-all classifier
        self.clf.fit(train_histogram, train_target)
        pred_target = self.clf.predict(train_histogram)

        print 'Memory usage at post-fit %5.2f MB' % (memory_usage_psutil())

        print ' Accuracy score (Training): %4.3f' % (metrics.accuracy_score(train_target, pred_target))
        print ' Report (Training):\n %s' % (metrics.classification_report(train_target, pred_target, 
                                                                          target_names=self.dataset.target_names))

        print 'Memory usage at post-predict %5.2f MB' % (memory_usage_psutil())
        print 'Training took %5.3f s' % (time.time() - st)

        print 'Saving classifier'
        self.save(self.params.cache.detector_path)
        return

    def classify(self, classify_trained=False): 
        print '===> Classification '
        st = time.time()

        # Extract features
        test_data = self.X_all if classify_trained else self.X_test
        test_desc = [ self.image_descriptor.describe(**self.process_cb(x_t)) for x_t in test_data ]

        print 'Descriptor extraction took %5.3f s' % (time.time() - st)    
    
        # Histogram of trained features
        test_target = self.y_all if classify_trained else self.y_test
        test_histogram = np.vstack([self.bow.project(desc) for desc in test_desc])

        pred_target = self.clf.predict(test_histogram)
        pred_score = self.clf.decision_function(test_histogram)

        # print ' Confusion matrix (Test): %s' % (metrics.confusion_matrix(test_target, pred_target))
        print ' Accuracy score (Test): %4.3f' % (metrics.accuracy_score(test_target, pred_target))
        print ' Report (Test):\n %s' % (metrics.classification_report(test_target, pred_target, 
                                                                      target_names=self.dataset.target_names))

        print 'Testing took %5.3f s' % (time.time() - st)

        return AttrDict(test_target=test_target, pred_target=pred_target, pred_score=pred_score, 
                        target_names=self.dataset.target_names)


    def classify_one(self, img, mask): 
        print '===> Classification one '
        st = time.time()

        # Extract features
        test_desc = self.image_descriptor.describe(img, mask=mask) 
        test_histogram = self.bow.project(test_desc)
        pred_target_proba = self.clf.decision_function(test_histogram)
        pred_target, = self.clf.predict(test_histogram)
        # print pred_target_proba, pred_target

        return self.dataset.target_unhash[pred_target]

    def setup_recognition_from_dict(self, db): 
        try: 
            self.params = db.params
            self.image_descriptor = ImageDescription(**db.params.descriptor)
            self.bow = BoWVectorizer.from_dict(db.bow)
            self.clf = db.clf
        except KeyError: 
            raise RuntimeError('DB not setup correctly, try re-training!')
            

    @classmethod
    def from_dict(cls, db): 
        c = cls()
        c.setup_recognition_from_dict(db)
        return c

    @classmethod
    def load(cls, path): 
        db = AttrDict.load(path)
        return cls.from_dict(db)
        
    def save(self, path): 
        db = AttrDict(params=self.params, bow=self.bow.to_dict(), clf=self.clf)
        db.save(path)

        
