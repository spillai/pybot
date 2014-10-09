import numpy as np
import cv2, os, time, random

from sklearn.linear_model import SGDClassifier
from sklearn.neighbors import KNeighborsClassifier
from sklearn.svm import LinearSVC
from sklearn.multiclass import OneVsRestClassifier
from sklearn.cross_validation import train_test_split
from sklearn.externals import joblib

from bot_utils.db_utils import AttrDict, AttrDictDB
from bot_vision.bow_utils import BOWTrainer
from bot_vision.descriptor_utils import ImageDescription

import bot_utils.io_utils as io_utils
import sklearn.metrics as metrics

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
            self.clf = OneVsRestClassifier(KNeighborsClassifier(n_neighbors=10), n_jobs=1)
        else: 
            print 'Loading Classifier'
            self.clf = joblib.load(cache_args.clf_path)
            self.clf_pretrained = True

    def train(self): 
        if self.clf_pretrained: 
            return 

        st = time.time()

        # Extract features
        train_desc = [ self.image_descriptor.describe(cv2.imread(x_t)) for x_t in self.X_train ]

        print 'Descriptor extraction took %5.3f s' % (time.time() - st)    

        # Build BOW
        self.bow.build(train_desc)

        # Histogram of trained features
        train_target = self.y_train
        train_histogram = np.vstack([self.bow.project(desc) for desc in train_desc])

        # Train one-vs-all classifier
        self.clf.fit(train_histogram, train_target)

        # Save to DB
        joblib.dump(self.bow, self.cache_args.vocab_path)
        joblib.dump(self.clf, self.cache_args.clf_path)

        # Predict for accuracy report
        pred_target = self.clf.predict(train_histogram)
        # print ' Train:\n', np.vstack([train_target, pred_target])        
        print ' Accuracy score (Training): %4.3f' % (metrics.accuracy_score(train_target, pred_target))
        print ' Report (Training):\n %s' % (metrics.classification_report(train_target, pred_target, 
                                                                          target_names=self.dataset.target_names))

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
        
