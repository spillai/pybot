"""
Object Recognition utilities
"""
# Author: Sudeep Pillai <spillai@csail.mit.edu>
# License: TODO

import numpy as np
import cv2, os, time, random
from itertools import izip, chain

import pprint
import datetime

from bot_vision.image_utils import im_resize, gaussian_blur, median_blur, box_blur
from bot_vision.bow_utils import BoWVectorizer, bow_codebook, bow_project, flair_project
from pybot_vision import FLAIR_code
# import bot_vision.mser_utils as mser_utils


import bot_utils.io_utils as io_utils
from bot_utils.io_utils import memory_usage_psutil, format_time
from bot_utils.db_utils import AttrDict, IterDB
from bot_utils.itertools_recipes import chunks

import sklearn.metrics as metrics
from sklearn.decomposition import PCA, RandomizedPCA
from sklearn.svm import LinearSVC, SVC
from sklearn.ensemble import GradientBoostingClassifier, ExtraTreesClassifier
from sklearn.linear_model import SGDClassifier
from sklearn.grid_search import GridSearchCV
from sklearn.cross_validation import train_test_split, ShuffleSplit
from sklearn.kernel_approximation import AdditiveChi2Sampler, RBFSampler
from sklearn.pipeline import Pipeline

from sklearn.externals.joblib import Parallel, delayed



# =====================================================================
# Generic utility functions for object recognition
# ---------------------------------------------------------------------

def classification_report(y_true, y_pred, labels=None, target_names=None,
                          sample_weight=None):
    """Build a text report showing the main classification metrics

    Parameters
    ----------
    y_true : array-like or label indicator matrix
        Ground truth (correct) target values.

    y_pred : array-like or label indicator matrix
        Estimated targets as returned by a classifier.

    labels : array, shape = [n_labels]
        Optional list of label indices to include in the report.

    target_names : list of strings
        Optional display names matching the labels (same order).

    sample_weight : array-like of shape = [n_samples], optional
        Sample weights.

    Returns
    -------
    report : string
        Text summary of the precision, recall, F1 score for each class.

    Examples
    --------
    >>> from sklearn.metrics import classification_report
    >>> y_true = [0, 1, 2, 2, 2]
    >>> y_pred = [0, 0, 2, 2, 1]
    >>> target_names = ['class 0', 'class 1', 'class 2']
    >>> print(classification_report(y_true, y_pred, target_names=target_names))
                 precision    recall  f1-score   support
    <BLANKLINE>
        class 0       0.50      1.00      0.67         1
        class 1       0.00      0.00      0.00         1
        class 2       1.00      0.67      0.80         3
    <BLANKLINE>
    avg / total       0.70      0.60      0.61         5
    <BLANKLINE>

    """

    from sklearn.metrics import precision_recall_fscore_support
    from sklearn.utils.multiclass import unique_labels

    if labels is None:
        labels = unique_labels(y_true, y_pred)
    else:
        labels = np.asarray(labels)

    last_line_heading = 'avg / total'

    if target_names is None:
        width = len(last_line_heading)
        target_names = ['%s' % l for l in labels]
    else:
        width = max(len(cn) for cn in target_names)
        width = max(width, len(last_line_heading))

    headers = ["precision", "recall", "f1-score", "support"]
    fmt = '%% %ds' % width  # first column: class name
    fmt += '  '
    fmt += ' '.join(['% 9s' for _ in headers])
    fmt += '\n'

    headers = [""] + headers
    report = fmt % tuple(headers)
    report += '\n'

    p, r, f1, s = precision_recall_fscore_support(y_true, y_pred,
                                                  labels=labels,
                                                  average=None,
                                                  sample_weight=sample_weight)

    for i, label in enumerate(labels):
        values = [target_names[i]]
        for v in (p[i], r[i], f1[i]):
            values += ["{0:0.3f}".format(v)]
        values += ["{0}".format(s[i])]
        report += fmt % tuple(values)

    report += '\n'

    # compute averages
    values = [last_line_heading]
    for v in (np.average(p, weights=s),
              np.average(r, weights=s),
              np.average(f1, weights=s)):
        values += ["{0:0.3f}".format(v)]
    values += ['{0}'.format(np.sum(s))]
    report += fmt % tuple(values)
    return report


def get_dense_detector(step=4, levels=7, scale=np.sqrt(2)): 
    """
    Standalone dense detector instantiation
    """
    detector = cv2.FeatureDetector_create('Dense')
    detector.setInt('initXyStep', step)
    # detector.setDouble('initFeatureScale', 0.5)

    detector.setDouble('featureScaleMul', scale)
    detector.setInt('featureScaleLevels', levels)

    detector.setBool('varyImgBoundWithScale', True)
    detector.setBool('varyXyStepWithScale', False)

    # detector = cv2.PyramidAdaptedFeatureDetector(detector, maxLevel=4)
    return detector

def get_detector(detector='dense', step=4, levels=7, scale=np.sqrt(2)): 
    """ Get opencv dense-sampler or specific feature detector """
    if detector == 'dense': 
        return get_dense_detector(step=step, levels=levels, scale=scale)
    else: 
        detector = cv2.FeatureDetector_create(detector)
        return cv2.PyramidAdaptedFeatureDetector(detector, maxLevel=levels)



def im_detect_and_describe(img, mask=None, detector='dense', descriptor='SIFT', 
                           step=4, levels=7, scale=np.sqrt(2)): 
    """ 
    Describe image using dense sampling / specific detector-descriptor combination. 
    """
    detector = get_detector(detector=detector, step=step, levels=levels, scale=scale)
    extractor = cv2.DescriptorExtractor_create(descriptor)

    try:     
        kpts = detector.detect(img, mask=mask)
        kpts, desc = extractor.compute(img, kpts)
        pts = np.vstack([kp.pt for kp in kpts]).astype(np.int32)

        return pts, desc
    except Exception as e: 
        print 'im_detect_and_describe', e
        return None, None

def im_describe(*args, **kwargs): 
    """ 
    Describe image using dense sampling / specific detector-descriptor combination. 
    Sugar for description-only call. 
    """
    kpts, desc = im_detect_and_describe(*args, **kwargs)
    return desc

def root_sift(kpts, desc): 
    """ Compute Root-SIFT on descriptor """
    desc = np.sqrt(desc.astype(np.float32) / (np.sum(desc, axis=1)).reshape(-1,1))
    inds, = np.where(np.isfinite(desc).all(axis=1))
    kpts, desc = [kpts[ind] for ind in inds], desc[inds]
    return kpts, desc

# def color_codes(img, kpts): 
#     # Extract color information (Lab)
#     pts = np.vstack([kp.pt for kp in kpts]).astype(np.int32)
#     imgc = median_blur(img, size=5) 
#     cdesc = img[pts[:,1], pts[:,0]]
#     return kpts, np.hstack([desc, cdesc])


# =====================================================================
# General-purpose object recognition interfaces, and functions
# ---------------------------------------------------------------------

class HomogenousKernelMap(AdditiveChi2Sampler): 
    """ 
    Additive homogeneous kernel maps for approximate chi2 kernel
    """
    def __init__(self, sample_steps=2, sample_interval=None): 
        AdditiveChi2Sampler.__init__(self, sample_steps=sample_steps, sample_interval=None)
        self.hlength = (self.sample_steps-1) * 2 + 1

    def fit(self, X, y=None): 
        sgn, Xp = np.sign(X), np.fabs(X)
        super(HomogenousKernelMap, self).fit(Xp, y=y)
        return self

    def transform(self, X, y=None): 
        sgn, Xp = np.sign(X), np.fabs(X)
        sgn = np.tile(sgn, (1,self.hlength))
        psix = super(HomogenousKernelMap, self).transform(Xp, y=y)
        return sgn * psix

class ImageDescription(object):  
    def __init__(self, detector='dense', descriptor='SIFT', step=4, levels=7, scale=np.sqrt(2)): 
        self.step = step
        self.levels = levels
        self.scale = scale
        
        # Setup feature detector
        self.detector = get_dense_detector(step=step, levels=levels, scale=scale)

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
            pts = np.vstack([kp.pt for kp in kpts]).astype(np.int32)
            return pts, desc
        except Exception as e: 
            # Catch when masks are small, and insufficient description
            print 'detect_and_describe', e
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

class BOWClassifier(object): 
    """
    Usage: 

        bc = BOWClassifier(params)

        train_dataset = caltech_training
        test_dataset = caltech_testing

        # Extract SIFT features in advance and stores
        bc.preprocess(train_dataset.iteritems(), test_dataset.iteritems(), batch_size=10)

        # Fit and train classifier
        bc.train(train_dataset.iteritems(), batch_size=10)

        # Evaluate on test dataset
        bc.eval(test_dataset.iteritems(), batch_size=10)

    Object recognition class
    Attributes: 
      data:         [image_fn1, ...  ]
      target:       [class_id, ... ]
      target_ids:   [0, 1, 2, ..., 101]
      target_names: [car, bike, ... ]
    
    """
    def __init__(self, params=None, 
                 process_cb=lambda f: dict(img=cv2.imread(f.filename), mask=None), target_names=None): 
        # Setup Params
        self.params_ = params
        self.process_cb_ = process_cb
        self.target_names_ = target_names

        # 1. Image description using Dense SIFT/Descriptor
        self.image_descriptor_ = ImageDescription(**self.params_.descriptor)

        # 2. Setup dim. red
        self.pca_ = RandomizedPCA(**self.params_.pca) if self.params_.do_pca else None

        # 3. Bag-of-words VLAD/VQ
        # a. Traditional BOW, b. FLAIR
        if (self.params_.bow_method).lower() == 'flair': 
            self.bow_ = BoWVectorizer(**self.params_.bow)

            # self.flair_ = FLAIR_code(**self.params_.flair) 
            # raise Exception('FLAIR not setup correctly')

            # if codebook is not None: 
            #     W=UWRGBDDataset.default_rgb_shape[1], H=UWRGBDDataset.default_rgb_shape[0], K=self.bow.dictionary_size, 
            #     step=self.params.descriptor.step, levels=np.array(self.params.bow.levels, dtype=np.int32), encoding={'bow':0, 'vlad':1, 'fisher':2}[self.params.bow.method])

        elif (self.params_.bow_method).lower() == 'bow': 
            self.bow_ = BoWVectorizer(**self.params_.bow)

        else: 
            raise Exception('Unknown bow_method %s' % self.params_.bow_method)

        # 4. Setup kernel feature map
        self.kernel_tf_ = HomogenousKernelMap(2) if self.params_.do_kernel_approximation else None

        # 5. Setup classifier
        print 'Building Classifier'
        if self.params_.classifier == 'svm': 
            self.clf_hyparams_ = {'C':[0.01, 0.1, 0.2, 0.5, 1.0, 2.0, 4.0, 5.0, 10.0]}
            self.clf_ = LinearSVC(random_state=1)
        elif self.params_.classifier == 'sgd': 
            self.clf_hyparams_ = {'loss':['hinge'], 'alpha':[0.0001, 0.001, 0.01, 0.1, 1.0, 10.0], 'class_weight':['auto']}
            self.clf_ = SGDClassifier(loss='hinge', n_jobs=4, n_iter=10)
        elif self.params_.classifier == 'gradient-boosting': 
            self.clf_hyparams_ = {'learning_rate':[0.01, 0.1, 0.2, 0.5]}
            self.clf_ = GradientBoostingClassifier()
        elif self.params_.classifier == 'extra-trees':             
            self.clf_hyparams_ = {'n_estimators':[10, 20, 40, 100]}
            self.clf_ = ExtraTreesClassifier()
        else: 
            raise Exception('Unknown classifier type %s. Choose from [sgd, svm, gradient-boosting, extra-trees]' 
                            % self.params_.classifier)

    def _extract(self, data_iterable, mode, batch_size=10): 
        """ Supports train/test modes """
        if not (mode == 'train' or mode == 'test'): 
            raise Exception('Unknown mode %s' % mode)

        # Extract features, only if not already available
        features_dir = getattr(self.params_.cache, '%s_features_dir' % mode)
        mode_prefix = lambda name: ''.join(['%s_' % mode, name])

        if not os.path.isdir(features_dir): 
            print '====> [COMPUTE] %s: Feature Extraction ' % mode.upper()
            st = time.time()
            # Add vocab_desc as training field
            fields = [mode_prefix('desc'), mode_prefix('target'), mode_prefix('pts'), mode_prefix('shapes')]
            if mode == 'train': fields.append('vocab_desc')

            # Initialize features_db
            features_db = IterDB(filename=features_dir, mode='w', fields=fields, batch_size=batch_size)

            def add_item_to_features_db(features_db, im_desc, target, pts, shapes): 
                features_db.append(mode_prefix('desc'), im_desc)
                features_db.append(mode_prefix('target'), target)
                features_db.append(mode_prefix('pts'), pts)
                features_db.append(mode_prefix('shapes'), shapes) 

            # Parallel Processing (in chunks of batch_size)
            if batch_size is not None and batch_size > 1: 
                for chunk in chunks(data_iterable, batch_size): 
                    res = Parallel(n_jobs=8, verbose=5) (
                        delayed(im_detect_and_describe)
                        (**dict(self.process_cb_(frame), **self.params_.descriptor)) for frame in chunk
                    )
                    for (pts, im_desc), frame in izip(res, chunk): 
                        if im_desc is None: continue

                        if hasattr(frame, 'bbox'): 
                            # Single image, multiple ROIs/targets
                            target = np.array([bbox['target'] for bbox in frame.bbox], dtype=np.int32)
                            shapes = np.vstack([ [bbox['left'], bbox['top'], bbox['right'], bbox['bottom']] for bbox in frame.bbox] )

                            add_item_to_features_db(features_db, im_desc, target, pts, shapes)
                        elif hasattr(frame, 'target'): 
                            # Single image, single target
                            add_item_to_features_db(features_db, im_desc, frame.target, pts, np.array([[0, 0, frame.img.shape[1]-1, frame.img.shape[0]-1]]))
                        else: 
                            raise RuntimeError('Unknown extraction policy')

                        # Randomly sample from descriptors for vocab construction
                        if mode == 'train': 
                            inds = np.random.permutation(int(min(len(im_desc), self.params_.vocab.num_per_image)))
                            features_db.append('vocab_desc', im_desc[inds])
            else: 
                # Serial Processing                    
                for frame in data_iterable: 
                    # Extract and add descriptors to db
                    pts, im_desc = self.image_descriptor_.detect_and_describe(**self.process_cb_(frame))
                    if im_desc is None: continue

                    if hasattr(frame, 'bbox'): 
                        # Single image, multiple ROIs/targets
                        target = np.array([bbox['target'] for bbox in frame.bbox], dtype=np.int32)
                        shapes = np.vstack([ [bbox['left'], bbox['top'], bbox['right'], bbox['bottom']] for bbox in frame.bbox] )

                        add_item_to_features_db(features_db, im_desc, target, pts, shapes)
                    elif hasattr(frame, 'target'): 
                        # Single image, single target
                        add_item_to_features_db(features_db, im_desc, frame.target, pts, np.array([[0, 0, frame.img.shape[1]-1, frame.img.shape[0]-1]]))
                    else: 
                        raise RuntimeError('Unknown extraction policy')

                    # Randomly sample from descriptors for vocab construction
                    if mode == 'train': 
                        inds = np.random.permutation(int(min(len(im_desc), self.params_.vocab.num_per_image)))
                        features_db.append('vocab_desc', im_desc[inds])

            features_db.finalize()
            print '[%s] Descriptor extraction took %s' % (mode.upper(), format_time(time.time() - st))    
        print '-------------------------------'


    def _build(self): 
        if os.path.isdir(self.params_.cache.train_features_dir): 
            print '====> [LOAD] Feature Extraction'        
            features_db = IterDB(filename=self.params_.cache.train_features_dir, mode='r')
        else: 
            raise RuntimeError('Training Features not available %s' % self.params_.cache.train_features_dir)

        # Reduce dimensionality and Build BOW
        if not os.path.exists(self.params_.cache.vocab_path): # or self.params.cache.overwrite: 
            print '====> [COMPUTE] Vocabulary Construction'
            inds = np.random.permutation(features_db.length('train_desc'))[:self.params_.vocab.num_images]
            vocab_desc = np.vstack([item for item in features_db.itervalues('vocab_desc', inds=inds, verbose=True)])
            print 'Codebook data: %i, %i' % (len(inds), len(vocab_desc))

            # Apply dimensionality reduction
            # Fit PCA to subset of data computed
            print '====> MEMORY: PCA dim. reduction before: %4.3f MB' % (vocab_desc.nbytes / 1024 / 1024.0) 
            if self.pca_ is not None: 
                vocab_desc = self.pca_.fit_transform(vocab_desc)
            print '====> MEMORY: PCA dim. reduction after: %4.3f MB' % (vocab_desc.nbytes / 1024 / 1024.0) 
            print '====> MEMORY: Codebook construction: %4.3f MB' % (vocab_desc.nbytes / 1024 / 1024.0) 

            # BOW construction
            if (self.params_.bow_method).lower() == 'flair': 
                # codebook = bow_codebook(vocab_desc, K=self.params_.bow.K)
                # self.flair_.setVocabulary(codebook.astype(np.float32))
                self.bow_.build(vocab_desc)
            elif (self.params_.bow_method).lower() == 'bow': 
                self.bow_.build(vocab_desc)
            else: 
                raise Exception('Unknown bow_method %s' % self.params_.bow_method)

            vocab_desc = None
            vocab_db = AttrDict(params=self.params_, bow=self.bow_.to_dict(), pca=self.pca_)
            vocab_db.save(self.params_.cache.vocab_path)
            print 'Codebook: %s' % ('GOOD' if np.isfinite(self.bow_.codebook).all() else 'BAD')
        print '-------------------------------'


    def _project(self, mode, batch_size=10): 

        features_dir = getattr(self.params_.cache, '%s_features_dir' % mode)
        if os.path.exists(self.params_.cache.vocab_path) and os.path.isdir(features_dir): 
            print '====> [LOAD] Vocabulary Construction'
            features_db = IterDB(filename=features_dir, mode='r')
            vocab_db = AttrDict.load(self.params_.cache.vocab_path)
            self.bow_ = BoWVectorizer.from_dict(vocab_db.bow)
            self.pca_ = vocab_db.pca
        else: 
            raise RuntimeError('Vocabulary or features path not available %s, %s' % (self.params_.cache.vocab_path, features_dir))

        # Histogram of features
        hists_dir = getattr(self.params_.cache, '%s_hists_dir' % mode)
        mode_prefix = lambda name: ''.join(['%s_' % mode, name])
        if not os.path.exists(hists_dir):
            print '====> [COMPUTE] BoVW / VLAD projection '
            hists_db = IterDB(filename=hists_dir, 
                              fields=[mode_prefix('histogram'), mode_prefix('target')], mode='w', batch_size=batch_size)

            if batch_size is not None and batch_size > 1: 
                # Parallel Processing
                for chunk in chunks(features_db.iter_keys_values([
                        mode_prefix('target'), mode_prefix('desc'), mode_prefix('pts'), mode_prefix('shapes')
                ], verbose=True), batch_size): 
                    desc_red = [self.pca_.transform(desc) if self.pca_ is not None else desc for (target, desc, _, _) in chunk] 

                    if (self.params_.bow_method).lower() == 'flair': 
                        # raise RuntimeError('Not implemented for parallel %s' % self.params_.bow_method)
                        res_hist = Parallel(n_jobs=8, verbose=5) (
                            delayed(flair_project)
                            (desc, self.bow_.codebook, pts=pts, shape=shape, 
                             levels=self.params_.bow.levels, method=self.params_.bow.method, step=self.params_.descriptor.step) for desc, (_, _, pts, shape) in izip(desc_red, chunk)
                        )

                    elif (self.params_.bow_method).lower() == 'bow': 
                        res_hist = Parallel(n_jobs=8, verbose=5) (
                            delayed(bow_project)
                            (desc, self.bow_.codebook, pts=pts, shape=shape, levels=self.params_.bow.levels) for desc, (_, _, pts, shape) in izip(desc_red, chunk)
                        )
                    else: 
                        raise RuntimeError('Unknown bow method %s' % self.params_.bow_method)

                    for hist, (target, _, _, _) in izip(res_hist, chunk): 
                        hists_db.append(mode_prefix('histogram'), hist)
                        hists_db.append(mode_prefix('target'), target)

                hists_db.finalize()
            else: 
                # Serial Processing
                for (target, desc, pts, shape) in features_db.iter_keys_values(
                        [mode_prefix('target'), mode_prefix('desc'), mode_prefix('pts'), mode_prefix('shapes')], verbose=True): 
                    desc_red = self.pca_.transform(desc) if self.pca_ is not None else desc
                    if (self.params_.bow_method).lower() == 'flair': 
                        hist = flair_project(desc_red, self.bow_.codebook, pts=pts, shape=shape, 
                                             levels=self.params_.bow.levels, method=self.params_.bow.method, step=self.params_.descriptor.step)
                    elif (self.params_.bow_method).lower() == 'bow': 
                        hist = self.bow_.project(desc_red, pts=pts, shape=shape)
                    else: 
                        raise RuntimeError('Unknown bow method %s' % self.params_.bow_method)
                    hists_db.append(mode_prefix('histogram'), hist)
                    hists_db.append(mode_prefix('target'), target)
                hists_db.finalize()
        print '-------------------------------'


    def _train(self): 
        if os.path.isdir(self.params_.cache.train_hists_dir): 
            print '====> [LOAD] TRAIN BoVW / VLAD projection '
            hists_db = IterDB(filename=self.params_.cache.train_hists_dir, mode='r')
        else: 
            raise RuntimeError('Trained histograms not available %s' % self.params_.cache.train_hists_dir)

        # Homogeneous Kernel map
        train_hists = np.vstack([item for item in hists_db.itervalues('train_histogram', verbose=True)]) 
        train_targets = np.hstack([item for item in hists_db.itervalues('train_target', verbose=True)]).astype(np.int32) 
        if self.kernel_tf_ is not None: 
            print '====> [COMPUTE] TRAIN Kernel Approximator '
            # inds = np.random.permutation(hists_db.length('train_histogram'))[:self.params_.vocab.num_images]
            # hists = np.vstack([item for item in hists_db.itervalues('train_histogram', inds=inds, verbose=True)])
            train_hists = self.kernel_tf_.fit_transform(train_hists)
            print '-------------------------------'        

        # Train/Predict one-vs-all classifier
        print '====> Train classifier '
        st_clf = time.time()

        # Grid search cross-val
        cv = ShuffleSplit(len(train_hists), n_iter=20, test_size=0.5, random_state=4)
        self.clf_ = GridSearchCV(self.clf_, self.clf_hyparams_, cv=cv, n_jobs=8, verbose=4)
        self.clf_.fit(train_hists, train_targets)
        print 'BEST: ', self.clf_.best_score_, self.clf_.best_params_
        # self.clf = self.clf_.best_estimator_
        pred_targets = self.clf_.predict(train_hists)

        print 'Training Classifier took %s' % (format_time(time.time() - st_clf))
        print '-------------------------------'        

        print ' Accuracy score (Training): %4.3f' % (metrics.accuracy_score(train_targets, pred_targets))
        print ' Report (Training):\n %s' % (classification_report(train_targets, pred_targets, 
                                                                  target_names=self.target_names_))

        print 'Training took %s' % format_time(time.time() - st_clf)

        print '====> Saving classifier '
        self.save(self.params_.cache.detector_path)
        print '-------------------------------'

    def _classify(self): 
        if os.path.isdir(self.params_.cache.test_hists_dir): 
            print '====> [LOAD] TEST BoVW / VLAD projection '
            hists_db = IterDB(filename=self.params_.cache.test_hists_dir, mode='r')
        else: 
            raise RuntimeError('Test histograms not available %s' % self.params_.cache.test_hists_dir)

        print '-------------------------------'

        # Homogeneous Kernel map
        test_hists = np.vstack([item for item in hists_db.itervalues('test_histogram', verbose=True)]) 
        test_targets = np.hstack([item for item in hists_db.itervalues('test_target', verbose=True)]).astype(np.int32)
        if self.kernel_tf_ is not None: 
            print '====> [COMPUTE] Kernel Approximator '
            test_hists = self.kernel_tf_.transform(test_hists)
            print '-------------------------------'        

        print '====> Predict using classifer '
        st_clf = time.time()

        pred_targets = self.clf_.predict(test_hists)
        pred_scores = self.clf_.decision_function(test_hists)
        print '-------------------------------'

        # print ' Confusion matrix (Test): %s' % (metrics.confusion_matrix(test_target, pred_target))
        print '=========================================================> '
        print '\n ===> Classification @ ', datetime.datetime.now()
        print 'Params: \n'
        pp = pprint.PrettyPrinter(indent=4)
        pp.pprint(self.params_)
        print '\n'
        print '-----------------------------------------------------------'
        print ' Accuracy score (Test): %4.3f' % (metrics.accuracy_score(test_targets, pred_targets))
        print ' Report (Test):\n %s' % (classification_report(test_targets, pred_targets, 
                                                              target_names=self.target_names_))
        print 'Testing took %s' % format_time(time.time() - st_clf)

        return AttrDict(test_targets=test_targets, pred_targets=pred_targets, pred_scores=pred_scores, 
                        target_names=self.target_names_)


    def preprocess(self, train_data_iterable, test_data_iterable, batch_size=10): 
        self._extract(train_data_iterable, mode='train', batch_size=batch_size)
        self._extract(test_data_iterable, mode='test', batch_size=batch_size)        

    def train(self, data_iterable, batch_size=10): 
        # 1. Extract D-SIFT features 
        self._extract(data_iterable, mode='train', batch_size=batch_size)

        # 2. Construct vocabulary using PCA-SIFT
        self._build()

        # 2. Reduce dimensionality and project onto 
        #    histogram with spatial pyramid pooling
        self._project(mode='train', batch_size=batch_size)

        # 4. Kernel approximate and linear classification training
        self._train()

    def evaluate(self, data_iterable, batch_size=10): 
        # 1. Extract D-SIFT features 
        self._extract(data_iterable, mode='test', batch_size=batch_size)

        # 2. Reduce dimensionality and project onto 
        #    histogram with spatial pyramid pooling
        self._project(mode='test', batch_size=batch_size)

        # 4. Kernel approximate and linear classification eval
        self._classify()

    def classify_with_mask(self, img, mask=None): 
        """ Describe image (with mask) with a BOW histogram  """
        test_desc = self.image_descriptor_.describe(img, mask=mask) 
        if self.pca_ is not None: 
            test_desc = self.pca_.transform(test_desc)
        test_hist = self.bow_.project(test_desc_red)
        if self.kernel_tf_ is not None: 
            test_hist = self.kernel_tf.transform(test_hist) 
        # pred_target_proba = self.clf_.decision_function(test_histogram)
        pred_target, = self.clf_.predict(test_histogram)

        return pred_target

    def describe_with_bboxes(self, img, bboxes): 
        raise RuntimeError('FLAIR not supported yet!')
        pass


    def _setup_from_dict(self, db): 
        try: 
            self.params_ = db.params
            self.image_descriptor_ = ImageDescription(**db.params.descriptor)
            self.bow_ = BoWVectorizer.from_dict(db.bow)
            self.pca_ = db.pca
            self.kernel_tf_ = db.kernel_tf
            self.clf_, self.clf_hyparams_ = db.clf, db.clf_hyparams
        except KeyError: 
            raise RuntimeError('DB not setup correctly, try re-training!')
        
    def save(self, path): 
        db = AttrDict(params=self.params_, 
                      bow=self.bow_.to_dict(), pca=self.pca_, kernel_tf=self.kernel_tf_, 
                      clf=self.clf_, clf_hyparams=self.clf_hyparams_, target_names=self.target_names_)
        db.save(path)

    @classmethod
    def from_dict(cls, db): 
        c = cls(params=db.params, target_names=db.target_names)
        c._setup_from_dict(db)
        return c

    @classmethod
    def load(cls, path): 
        db = AttrDict.load(path)
        return cls.from_dict(db)
        

# class ImageClassifier(object): 
#     training_params = AttrDict(train_size=10, random_state=1)
#     descriptor_params = AttrDict(detector='dense', descriptor='SIFT', step=2, levels=4, scale=2)
#     bow_params = AttrDict(K=64, method='vlad', quantizer='kdtree', norm_method='square-rooting')
#     cache_params = AttrDict(detector_path='detector.h5', overwrite=False)
#     default_params = AttrDict(
#         training=training_params, descriptor=descriptor_params, bow=bow_params, cache=cache_params
#     )
#     def __init__(self, dataset=None, batch=True, 
#                  process_cb=lambda fn: dict(img=cv2.imread(fn), mask=None), 
#                  params = default_params): 

#         # Save dataset
#         self.dataset_ = dataset_
#         self.process_cb_ = process_cb
#         self.params_ = AttrDict(params)

#         # Train in batches or one-by-one
#         self.batch_ = batch
#         self.BATCH_SIZE_ = 5

#         # Setup recognition 
#         self._setup_recognition()

#         # # Optionally setup training testing
#         # if dataset is not None: 
#         #     self.setup_training_testing()

#         #     # Persist, and retrieve db if available
#         #     if not io_utils.path_exists(self.params.cache.detector_path) or self.params.cache.overwrite: 
#         #         self.setup_recognition()
#         #         self.clf_pretrained = False
#         #     else: 
#         #         db = AttrDict.load(self.params.cache.detector_path)
#         #         self.setup_recognition_from_dict(db)
#         #         self.clf_pretrained = True

    # def _setup_recognition(self): 
        
    #     # Support for parallel processing
    #     if not self.batch_: 
    #         # Image description using Dense SIFT/Descriptor
    #         # Bag-of-words VLAD/VQ
    #         self.image_descriptor_ = ImageDescription(**self.params_.descriptor)
    #         self.bow_ = BoWVectorizer(**self.params_.bow)

    #     # Setup dim. red
    #     # Setup kernel feature map
    #     self.pca_ = RandomizedPCA(**self.params_.pca) if self.params_.do_pca else None
    #     self.kernel_tf_ = HomogenousKernelMap(2) if self.params_.do_kernel_approximation else None

    #     # Setup classifier
    #     print 'Building Classifier'
    #     if self.params_.classifier == 'svm': 
    #         self.clf_hyparams_ = {'C':[0.01, 0.1, 0.2, 0.5, 1.0, 2.0, 4.0, 5.0, 10.0]}
    #         self.clf_ = LinearSVC(random_state=1)
    #     elif self.params_.classifier == 'sgd': 
    #         self.clf_hyparams_ = {'loss':['hinge'], 'alpha':[0.0001, 0.001, 0.01, 0.1, 1.0, 10.0], 'class_weight':['auto']}
    #         self.clf_ = SGDClassifier(loss='hinge', n_jobs=4, n_iter=10)
    #     elif self.params_.classifier == 'gradient-boosting': 
    #         self.clf_hyparams_ = {'learning_rate':[0.01, 0.1, 0.2, 0.5]}
    #         self.clf_ = GradientBoostingClassifier()
    #     elif self.params_.classifier == 'extra-trees':             
    #         self.clf_hyparams_ = {'n_estimators':[10, 20, 40, 100]}
    #         self.clf_ = ExtraTreesClassifier()
    #     else: 
    #         raise Exception('Unknown classifier type %s. Choose from [sgd, svm, gradient-boosting, extra-trees]' 
    #                         % self.params_.classifier)

    # def extract_features(self): 
    #     if not self.dataset_: 
    #         raise RuntimeError('Training cannot proceed. Setup dataset first!')            

    #     # Extract training features, only if not already available
    #     if not os.path.isdir(self.params_.cache.train_path): 
    #         print '====> [COMPUTE] TRAINING: Feature Extraction '        
    #         st = time.time()
    #         features_db = IterDB(filename=self.params_.cache.train_path, mode='w', 
    #                              fields=['train_desc', 'train_target', 'train_pts', 'train_shapes', 'vocab_desc'], batch_size=self.BATCH_SIZE)

    #         # Parallel Processing (in chunks of BATCH_SIZE)
    #         if self.batch_: 
    #             for chunk in chunks(self.dataset_, self.BATCH_SIZE): 
    #                 res = Parallel(n_jobs=8, verbose=5) (
    #                     delayed(im_detect_and_describe)
    #                     (**dict(self.process_cb_(x_t), **self.params_.descriptor)) for (x_t,_) in chunk
    #                 )

    #                 for (pts, im_desc), (x_t, y_t) in izip(res, chunk): 
    #                     features_db.append('train_desc', im_desc)
    #                     features_db.append('train_pts', pts)
    #                     features_db.append('train_shapes', np.array([np.min(pts[:,0]), np.min(pts[:,1]), np.max(pts[:,0]), np.max(pts[:,1])]))
    #                     features_db.append('train_target', y_t)

    #                     # im_shape = (self.process_cb(x_t))['img'].shape[:2]
    #                     # features_db.append('train_shapes', [0, 0, im_shape[1], im_shape[0]])

    #                     # Randomly sample from descriptors for vocab construction
    #                     inds = np.random.permutation(int(min(len(im_desc), self.params_.vocab.num_per_image)))
    #                     features_db.append('vocab_desc', im_desc[inds])
    #         else: 
    #             # Serial Processing                    
    #             for (x_t,y_t) in izip(self.X_train, self.y_train): 
    #                 # Extract and add descriptors to db
    #                 im_desc = self.image_descriptor.describe(**self.process_cb(x_t))
    #                 features_db.append('train_desc', im_desc)
    #                 features_db.append('train_target', y_t)

    #                 # Randomly sample from descriptors for vocab construction
    #                 inds = np.random.permutation(int(min(len(im_desc), self.params_.vocab.num_per_image)))
    #                 features_db.append('vocab_desc', im_desc[inds])

    #         features_db.finalize()
    #         print '[TRAIN] Descriptor extraction took %s' % (format_time(time.time() - st))    

    #     print '-------------------------------'

    #     # Extract test features
    #     if not os.path.isdir(self.params.cache.test_path): 
    #         print '====> [COMPUTE] TESTING: Feature Extraction '        
    #         st = time.time()
    #         features_db = IterDB(filename=self.params.cache.test_path, mode='w', 
    #                              fields=['test_desc', 'test_target', 'test_pts', 'test_shapes'], batch_size=self.BATCH_SIZE)


    #         # Parallel Processing
    #         for chunk in chunks(izip(self.X_test, self.y_test), self.BATCH_SIZE): 
    #             res = Parallel(n_jobs=8, verbose=5) (
    #                 delayed(im_detect_and_describe)
    #                 (**dict(self.process_cb(x_t), **self.params.descriptor)) for (x_t,_) in chunk
    #             )
    #             for (pts, im_desc), (_, y_t) in izip(res, chunk): 
    #                 features_db.append('test_desc', im_desc)
    #                 features_db.append('test_pts', pts)
    #                 features_db.append('test_target', y_t)
    #                 features_db.append('test_shapes', np.array([np.min(pts[:,0]), np.min(pts[:,1]), np.max(pts[:,0]), np.max(pts[:,1])]))

    #         # Serial Processing
    #         # for (x_t,y_t) in izip(self.X_test, self.y_test): 
    #         #     features_db.append('test_desc', self.image_descriptor.describe(**self.process_cb(x_t)))
    #         #     features_db.append('test_target', y_t)

    #         features_db.finalize()
    #         print '[TEST] Descriptor extraction took %s' % (format_time(time.time() - st))    
    #     print '-------------------------------'


    # def train(self): 
    #     if self.clf_pretrained: 
    #         return 

    #     if not hasattr(self, 'X_train') or not hasattr(self, 'y_train'): 
    #         raise RuntimeError('Training cannot proceed. Setup training and testing samples first!')            

    #     print '===> Training '
    #     st = time.time()

    #     # Extract features
    #     if not self.params.cache.results_dir: 
    #         raise RuntimeError('Setup results_dir before running training')

        # # Extract features, only if not already available
        # if not os.path.isdir(self.params.cache.train_path): 
        #     pass
        #     # print '====> [COMPUTE] Feature Extraction '        
        #     # features_db = IterDB(filename=self.params.cache.train_path, mode='w', 
        #     #                      fields=['train_desc', 'train_target', 'vocab_desc'])
        #     # # Serial Processing
        #     # for (x_t,y_t) in izip(self.X_train, self.y_train): 
        #     #     # Extract and add descriptors to db
        #     #     im_desc = self.image_descriptor.describe(**self.process_cb(x_t))
        #     #     features_db.append('train_desc', im_desc)
        #     #     features_db.append('train_target', y_t)

        #     #     # Randomly sample from descriptors for vocab construction
        #     #     inds = np.random.permutation(int(min(len(im_desc), self.params.vocab.num_per_image)))
        #     #     features_db.append('vocab_desc', im_desc[inds])

        #     # features_db.finalize()
        #     # print 'Descriptor extraction took %5.3f s' % (time.time() - st)    
        # else: 
        #     print '====> [LOAD] Feature Extraction'        
        #     features_db = IterDB(filename=self.params.cache.train_path, mode='r')
        # print '-------------------------------'

        # # Build BOW
        # if not os.path.exists(self.params.cache.vocab_path): # or self.params.cache.overwrite: 
        #     print '====> [COMPUTE] Vocabulary Construction'
        #     inds = np.random.permutation(len(self.X_train))[:self.params.vocab.num_images]
        #     vocab_desc = np.vstack([item for item in features_db.itervalues('vocab_desc', inds=inds, verbose=True)])
        #     print 'Codebook data: %i, %i' % (len(inds), len(vocab_desc))

        #     # Apply dimensionality reduction
        #     # Fit PCA to subset of data computed
        #     print '====> MEMORY: PCA dim. reduction before: %4.3f MB' % (vocab_desc.nbytes / 1024 / 1024.0) 
        #     if self.params.do_pca: 
        #         vocab_desc = self.pca.fit_transform(vocab_desc)
        #     print '====> MEMORY: PCA dim. reduction after: %4.3f MB' % (vocab_desc.nbytes / 1024 / 1024.0) 

        #     print '====> MEMORY: Codebook construction: %4.3f MB' % (vocab_desc.nbytes / 1024 / 1024.0) 
        #     self.bow.build(vocab_desc)

        #     vocab_desc = None
        #     vocab_db = AttrDict(params=self.params, bow=self.bow.to_dict(), kernel_tf=self.kernel_tf)
        #     vocab_db.save(self.params.cache.vocab_path)
        #     print 'Codebook: %s' % ('GOOD' if np.isfinite(self.bow.codebook).all() else 'BAD')
        # else: 
        #     print '====> [LOAD] Vocabulary Construction'
        #     vocab_db = AttrDict.load(self.params.cache.vocab_path)
        #     self.bow = BoWVectorizer.from_dict(vocab_db.bow)
        # print '-------------------------------'

        # # Histogram of trained features
        # if not os.path.exists(self.params.cache.train_hists_path): #  or self.params.cache.overwrite: 
        #     print '====> [COMPUTE] BoVW / VLAD projection '
        #     train_target = np.array(self.y_train, dtype=np.int32)

        #     # Serial Processing
        #     # train_histogram = np.vstack([self.bow.project(
        #     #     self.pca.transform(desc) if self.params.do_pca else desc, pts=pts, shape=shape
        #     # ) for (desc, pts, shape) in features_db.iter_keys_values(['train_desc', 'train_pts', 'train_shapes'], verbose=True)])

        #     # Parallel Processing
        #     train_histogram = []
        #     for chunk in chunks(features_db.iter_keys_values(['train_desc', 'train_pts', 'train_shapes'], verbose=True), self.BATCH_SIZE): 
        #         res_desc = [self.pca.transform(desc) for (desc, _, _) in chunk]
        #         res_hist = Parallel(n_jobs=8, verbose=5) (
        #             delayed(bow_project)
        #             (desc, self.bow.codebook, pts=pts, shape=shape, levels=self.params.bow.levels) for desc, (_, pts, shape) in izip(res_desc, chunk)
        #         )
        #         train_histogram.extend(res_hist)
        #     train_histogram = np.vstack(train_histogram)

        #     hists_db = AttrDict(train_target=train_target, train_histogram=train_histogram)
        #     hists_db.save(self.params.cache.train_hists_path)
        #     print '====> MEMORY: Histogram: %s %4.3f MB' % (train_histogram.shape, 
        #                                                     train_histogram.nbytes / 1024 / 1024.0) 
        # else: 
        #     print '====> [LOAD] BoVW / VLAD projection '
        #     hists_db = AttrDict.load(self.params.cache.train_hists_path)
        #     train_target, train_histogram = hists_db.train_target, hists_db.train_histogram
        # print '-------------------------------'

        # # PCA dim. red
        # if self.params.do_pca: 
        #     print '====> PCA '            
        #     train_histogram = self.pca.fit_transform(train_histogram)
        #     print '-------------------------------'        

        # # Homogeneous Kernel map
        # if self.params.do_kernel_approximation: 
        #     print '====> Kernel Approximation '
        #     train_histogram = self.kernel_tf.fit_transform(train_histogram)
        #     print '-------------------------------'        

        # # Train/Predict one-vs-all classifier
        # print '====> Train classifier '
        # st_clf = time.time()

        # # Grid search cross-val
        # cv = ShuffleSplit(len(train_histogram), n_iter=20, test_size=0.5, random_state=4)
        # self.clf_ = GridSearchCV(self.clf_, self.clf_hyparams_, cv=cv, n_jobs=8, verbose=4)
        # self.clf_.fit(train_histogram, train_target)
        # print 'BEST: ', self.clf_.best_score_, self.clf_.best_params_
        # # self.clf = self.clf_.best_estimator_
        # pred_target = self.clf_.predict(train_histogram)

        # print 'Training Classifier took %s' % (format_time(time.time() - st_clf))
        # print '-------------------------------'        


        # print ' Accuracy score (Training): %4.3f' % (metrics.accuracy_score(train_target, pred_target))
        # print ' Report (Training):\n %s' % (classification_report(train_target, pred_target, 
        #                                                           target_names=self.dataset.target_names))

        # print 'Training took %s' % format_time(time.time() - st)

        # print '====> Saving classifier '
        # self.save(self.params.cache.detector_path)
        # print '-------------------------------'
        # return

    # def classify(self): 
    #     print '===> Classification '
    #     st = time.time()

    #     # Extract features
    #     if not os.path.isdir(self.params.cache.test_path): 
    #         print '====> [COMPUTE] Feature Extraction '        
    #         features_db = IterDB(filename=self.params.cache.test_path, mode='w', 
    #                              fields=['test_desc', 'test_target'], batch_size=5)
    #         for (x_t,y_t) in izip(self.X_test, self.y_test): 
    #             features_db.append('test_desc', self.image_descriptor.describe(**self.process_cb(x_t)))
    #             features_db.append('test_target', y_t)
    #         features_db.finalize()
    #     else: 
    #         print '====> [LOAD] Feature Extraction'        
    #         features_db = IterDB(filename=self.params.cache.test_path, mode='r')
    #     print '-------------------------------'
    #     print 'Descriptor extraction took %s' % format_time(time.time() - st)    

    #     # Load Vocabulary
    #     if os.path.exists(self.params.cache.vocab_path):
    #         print '====> [LOAD] Vocabulary Construction'
    #         vocab_db = AttrDict.load(self.params.cache.vocab_path)
    #         self.bow = BoWVectorizer.from_dict(vocab_db.bow)
    #     else: 
    #         raise RuntimeError('Vocabulary not built %s' % self.params.cache.vocab_path)

    #     # Histogram of trained features
    #     if not os.path.exists(self.params.cache.test_hists_path): #  or self.params.cache.overwrite: 
    #         print '====> [COMPUTE] BoVW / VLAD projection '
    #         test_target = self.y_test

    #         # # Serial Processing
    #         # test_histogram = np.vstack([self.bow.project(
    #         #     self.pca.transform(desc) if self.params.do_pca else desc, pts=pts, shape=shape
    #         # ) for (desc, pts, shape) in features_db.iter_keys_values(['test_desc', 'test_pts', 'test_shapes'], verbose=True)])

    #         # Parallel Processing
    #         test_histogram = []
    #         for chunk in chunks(features_db.iter_keys_values(['test_desc', 'test_pts', 'test_shapes'], verbose=True), self.BATCH_SIZE): 
    #             res_desc = [self.pca.transform(desc) for (desc, _, _) in chunk]
    #             res_hist = Parallel(n_jobs=8, verbose=5) (
    #                 delayed(bow_project)
    #                 (desc, self.bow.codebook, pts=pts, shape=shape, levels=self.params.bow.levels) for desc, (_, pts, shape) in izip(res_desc, chunk)
    #             )
    #             test_histogram.extend(res_hist)
    #         test_histogram = np.vstack(test_histogram)

    #         hists_db = AttrDict(test_target=test_target, test_histogram=test_histogram)
    #         hists_db.save(self.params.cache.test_hists_path)
    #         print '====> MEMORY: Histogram: %s %4.3f MB' % (test_histogram.shape, 
    #                                                         test_histogram.nbytes / 1024 / 1024.0) 
    #     else: 
    #         print '====> [LOAD] BoVW / VLAD projection '
    #         hists_db = AttrDict.load(self.params.cache.test_hists_path)
    #         test_target, test_histogram = hists_db.test_target, hists_db.test_histogram
    #     print '-------------------------------'

    #     # # PCA dim. red
    #     # if self.params.do_pca: 
    #     #     print '====> PCA '            
    #     #     test_histogram = self.pca.transform(test_histogram)
    #     #     print '-------------------------------'        

    #     if self.params.do_kernel_approximation: 
    #         # Apply homogeneous transform
    #         test_histogram = self.kernel_tf.transform(test_histogram)

    #     print '====> Predict using classifer '
    #     pred_target = self.clf_.predict(test_histogram)
    #     pred_score = self.clf_.decision_function(test_histogram)
    #     print '-------------------------------'

    #     # print ' Confusion matrix (Test): %s' % (metrics.confusion_matrix(test_target, pred_target))
    #     print '=========================================================> '
    #     print '\n ===> Classification @ ', datetime.datetime.now()
    #     print 'Params: \n'
    #     pp = pprint.PrettyPrinter(indent=4)
    #     pp.pprint(self.params)
    #     print '\n'
    #     print '-----------------------------------------------------------'
    #     print ' Accuracy score (Test): %4.3f' % (metrics.accuracy_score(test_target, pred_target))
    #     print ' Report (Test):\n %s' % (classification_report(test_target, pred_target, 
    #                                                           target_names=self.dataset.target_names))

    #     print 'Testing took %s' % format_time(time.time() - st)

    #     return AttrDict(test_target=test_target, pred_target=pred_target, pred_score=pred_score, 
    #                     target_names=self.dataset.target_names)


    # def classify_one(self, img, mask): 
    #     print '===> Classification one '
    #     st = time.time()

    #     # Extract features
    #     test_desc = self.image_descriptor_.describe(img, mask=mask) 
    #     test_histogram = self.bow_.project(test_desc)
    #     pred_target_proba = self.clf_.decision_function(test_histogram)
    #     pred_target, = self.clf_.predict(test_histogram)
    #     # print pred_target_proba, pred_target

    #     return self.dataset.target_unhash[pred_target]

    # def setup_recognition_from_dict(self, db): 
    #     try: 
    #         self.params = db.params
    #         self.image_descriptor = ImageDescription(**db.params.descriptor)
    #         self.bow = BoWVectorizer.from_dict(db.bow)
    #         self.clf = db.clf
    #     except KeyError: 
    #         raise RuntimeError('DB not setup correctly, try re-training!')
            

    # @classmethod
    # def from_dict(cls, db): 
    #     c = cls()
    #     c.setup_recognition_from_dict(db)
    #     return c

    # @classmethod
    # def load(cls, path): 
    #     db = AttrDict.load(path)
    #     return cls.from_dict(db)
        
    # def save(self, path): 
    #     db = AttrDict(params=self.params_, bow=self.bow.to_dict(), clf=self.clf, pca=self.pca)
    #     db.save(path)

        



# from pybot_vlfeat import vl_dsift
# def im_detect_and_describe(img, mask=None, detector='dense', descriptor='SIFT', step=4, levels=7, scale=np.sqrt(2)): 

#     try:     
#         all_pts, all_desc = [], []
#         for l in range(levels): 
#             if l == 0:
#                 im = img.copy()
#                 mask_im = mask.copy() if mask is not None else None
#             else: 
#                 im = im_resize(im, scale=1./scale)
#                 mask_im = im_resize(mask, scale=1./scale) if mask_im is not None else None

#             # Convert to HSV
#             # im = cv2.cvtColor(im, cv2.COLOR_BGR2LAB)
#             cur_scale = scale ** l  

#             # Lab-SIFT
#             ch_desc = []
#             for ch in range(im.shape[2]): 
#                 pts, desc = vl_dsift(im[:,:,])
#                 ch_desc.append(desc)

#             pts, desc = (pts * cur_scale).astype(np.int32), (np.hstack(ch_desc)).astype(np.uint8)
#             all_pts.extend(pts)
#             all_desc.append(desc)

#         pts = np.vstack(all_pts).astype(np.int32)
#         desc = np.vstack(all_desc)

#         return pts, desc
#     except Exception as e: 
#         print e
#         return None, None
