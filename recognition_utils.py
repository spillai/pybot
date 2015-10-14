"""
Object Recognition utilities
"""
# Author: Sudeep Pillai <spillai@csail.mit.edu>
# License: MIT

import numpy as np
import cv2, os, time, random
from itertools import izip, chain

import pprint
import datetime
import pandas as pd

import gop
from pybot_vision import BINGObjectness

from bot_vision.image_utils import im_resize, gaussian_blur, median_blur, box_blur
from bot_vision.bow_utils import BoWVectorizer, bow_codebook, bow_project, flair_project
from bot_vision.caffe.detector import Detector, DetectorFastRCNN
from pybot_vision import FLAIR_code
from fast_rcnn.config import cfg

from bot_utils.io_utils import memory_usage_psutil, format_time
from bot_utils.db_utils import AttrDict, IterDB
from bot_utils.itertools_recipes import chunks

import sklearn.metrics as metrics
from sklearn.preprocessing import normalize
from sklearn.decomposition import PCA, RandomizedPCA
from sklearn.svm import LinearSVC, SVC
from sklearn.calibration import CalibratedClassifierCV
from sklearn.ensemble import GradientBoostingClassifier, ExtraTreesClassifier
from sklearn.linear_model import SGDClassifier
from sklearn.grid_search import GridSearchCV
from sklearn.cross_validation import train_test_split, ShuffleSplit
from sklearn.kernel_approximation import AdditiveChi2Sampler, RBFSampler

from sklearn.externals.joblib import Parallel, delayed

# =====================================================================
# Generic utility functions for object recognition
# ---------------------------------------------------------------------

def multilabel_precision_recall(y_score, y_test, target_map): 
    from sklearn.metrics import precision_recall_curve
    from sklearn.metrics import average_precision_score
    from sklearn.preprocessing import label_binarize

    # Compute Precision-Recall and plot curve
    precision = dict()
    recall = dict()
    average_precision = dict()

    target_ids = target_map.keys()
    target_names = target_map.values()
    print target_names

    y_test_multi = label_binarize(y_test, classes=target_ids)
    N, n_classes = y_score.shape[:2]
    for i,name in enumerate(target_names):
        precision[name], recall[name], _ = precision_recall_curve(y_test_multi[:, i],
                                                                  y_score[:, i])
        average_precision[name] = average_precision_score(y_test_multi[:, i], y_score[:, i])

    # Compute micro-average ROC curve and ROC area
    precision["average"], recall["average"], _ = precision_recall_curve(y_test_multi.ravel(),
        y_score.ravel())
    average_precision["micro"] = average_precision_score(y_test_multi, y_score,
                                                         average="micro") 
    average_precision["macro"] = average_precision_score(y_test_multi, y_score,
                                                         average="macro") 
    return precision, recall, average_precision


def plot_precision_recall(y_score, y_test, target_map, title='Precision-Recall curve'): 
    import matplotlib.pyplot as plt

    target_ids = target_map.keys()
    target_names = target_map.values()

    # Get multilabel precision recall curve
    precision, recall, average_precision = multilabel_precision_recall(y_score, y_test, target_map)

    # Plot Precision-Recall curve for each class
    plt.clf()
    plt.plot(recall["average"], precision["average"],
             label='Average', linewidth=3)
             # label='Average (area = {0:0.2f})'
             #       ''.format(average_precision["micro"]))
    plt.xlabel('Recall')
    plt.ylabel('Precision')
    plt.ylim([0.0, 1.0])
    plt.xlim([0.0, 1.0])
    plt.legend(loc="lower left")
    plt.show()

    for i,name in enumerate(target_names):
        plt.plot(recall[name], precision[name],
                 label='{0}'.format(name.title().replace('_', ' ')))
                 # label='{0} (area = {1:0.2f})'
                 #       ''.format(name.title().replace('_', ' '), average_precision[name]))

    plt.xlim([0.0, 1.0])
    plt.ylim([0.0, 1.0])
    plt.xlabel('Recall')
    plt.ylabel('Precision')
    plt.title(title)
    plt.legend(loc="lower left")
    plt.show(block=False)

def plot_roc(y_score, y_test, target_map, title='ROC curve'): 
    import matplotlib.pyplot as plt
    from sklearn.metrics import roc_curve, auc, precision_recall_curve
    from sklearn.preprocessing import label_binarize

    # Compute Precision-Recall and plot curve
    fpr = dict()
    tpr = dict()
    roc_auc = dict()
    
    target_ids = target_map.keys()
    target_names = target_map.values()
    print target_names

    y_test_multi = label_binarize(y_test, classes=target_ids)
    N, n_classes = y_score.shape[:2]
    for i,name in enumerate(target_names):
        fpr[name], tpr[name], _ = roc_curve(y_test_multi[:, i], y_score[:, i])
        roc_auc[name] = auc(fpr[name], tpr[name]) 

    # Compute micro-average ROC curve and ROC area
    fpr["micro"], tpr["micro"], _ = roc_curve(y_test_multi.ravel(), y_score.ravel())
    roc_auc["micro"] = auc(fpr["micro"], tpr["micro"]) 

    # Plot Precision-Recall curve for each class
    plt.clf()
    plt.plot([0, 1], [0, 1], 'k--')
    plt.plot(fpr["micro"], tpr["micro"],
             label='ROC curve (area = {0:0.2f})'
                   ''.format(roc_auc["micro"]), linewidth=3)
    plt.xlabel('False Positive Rate')
    plt.ylabel('True Positive Rate')
    plt.ylim([0.0, 1.0])
    plt.xlim([0.0, 1.0])
    plt.legend(loc="lower right")
    plt.show()

    for i,name in enumerate(target_names):
        plt.plot(fpr[name], tpr[name],
                 label='{0}'.format(name.title().replace('_', ' ')))
                 # label='{0} (area = {1:0.2f})'
                 #       ''.format(name.title().replace('_', ' '), roc_auc[name]))

    plt.xlim([0.0, 1.0])
    plt.ylim([0.0, 1.0])
    plt.xlabel('False Positive Rate')
    plt.ylabel('True Positive Rate')
    plt.title(title)
    plt.legend(loc="lower right")
    plt.show(block=False)


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

def root_sift(kpts, desc, eps=1e-7): 
    """ Compute Root-SIFT on descriptor """
    desc = desc.astype(np.float32)
    desc = np.sqrt(desc / (np.sum(desc, axis=1)[:,np.newaxis] + eps))
    # desc /= (np.linalg.norm(desc, axis=1)[:,np.newaxis] + eps)

    inds, = np.where(np.isfinite(desc).all(axis=1))
    kpts, desc = [kpts[ind] for ind in inds], desc[inds]
    return kpts, desc

class CNNDescription(object): 
    def __init__(self): 
        
        NETS = {'vgg16': ('VGG16',
                  'vgg16_fast_rcnn_iter_40000.caffemodel'),
        'vgg_cnn_m_1024': ('VGG_CNN_M_1024',
                           'vgg_cnn_m_1024_fast_rcnn_iter_40000.caffemodel'),
        'caffenet': ('CaffeNet',
                     'caffenet_fast_rcnn_iter_40000.caffemodel')}

        rcnn_dir = '/home/spillai/code/recognition-pod/recognition/fast-rcnn/'
        
        demo_net = 'caffenet'
        model_def = os.path.join(rcnn_dir, 'models', NETS[demo_net][0],
                                 'test.prototxt')
        pretrained_model = os.path.join(rcnn_dir, 'data', 'fast_rcnn_models',
                                        NETS[demo_net][1])
        
        self.rcnn_ = DetectorFastRCNN(model_def, pretrained_model)
        
        cfg.TEST.BBOX_REG = False

    def describe(self, img, bboxes):
        return self.rcnn_.detect_bboxes(img, bboxes)

def im_detect_and_describe(img, mask=None, detector='dense', descriptor='SIFT', colorspace='gray',
                           step=4, levels=7, scale=np.sqrt(2)): 
    """ 
    Describe image using dense sampling / specific detector-descriptor combination. 
    """
    detector = get_detector(detector=detector, step=step, levels=levels, scale=scale)
    extractor = cv2.DescriptorExtractor_create(descriptor)

    try:     
        kpts = detector.detect(img, mask=mask)
        kpts, desc = extractor.compute(img, kpts)

        if descriptor == 'SIFT': 
            kpts, desc = root_sift(kpts, desc)

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
    def __init__(self, detector='dense', descriptor='SIFT', colorspace='gray', step=4, levels=7, scale=np.sqrt(2)): 
        self.step = step
        self.levels = levels
        self.scale = scale
        self.colorspace = colorspace
        
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
    def __init__(self, params, target_map, 
                 process_cb=lambda f: dict(img=cv2.imread(f.filename), mask=None)): 

        # Setup Params
        self.params_ = params
        self.process_cb_ = process_cb
        self.target_map_ = target_map
        self.target_ids_ = (np.unique(target_map.keys())).astype(np.int32)
        self.epoch_no_ = 0

        # 1. Image description using Dense SIFT/Descriptor
        self.image_descriptor_ = ImageDescription(**self.params_.descriptor)

        # 2. Setup dim. red
        self.pca_ = RandomizedPCA(**self.params_.pca) if self.params_.do_pca else None

        # 3. Bag-of-words VLAD/VQ
        # a. Traditional BOW, b. FLAIR
        if (self.params_.bow_method).lower() == 'flair': 
            self.bow_ = BoWVectorizer(**self.params_.bow)

        elif (self.params_.bow_method).lower() == 'bow': 
            self.bow_ = BoWVectorizer(**self.params_.bow)

        else: 
            raise Exception('Unknown bow_method %s' % self.params_.bow_method)

        # 4. Setup kernel feature map
        self.kernel_tf_ = HomogenousKernelMap(2) if self.params_.do_kernel_approximation else None

        # 5. Setup classifier
        print '-------------------------------'        
        print '====> Building Classifier, setting class weights' 
        cweights = {cid: 10.0 for cidx,cid in enumerate(self.target_ids_)}
        cweights[51] = 1.0
        print 'Weights: ', self.target_ids_, cweights

        if self.params_.classifier == 'svm': 
            self.clf_hyparams_ = {'C':[0.01, 0.1, 0.5, 1.0, 4.0, 5.0, 10.0]} # , 'class_weight': ['auto']}
            self.clf_base_ = LinearSVC(random_state=1)
        elif self.params_.classifier == 'sgd': 
            self.clf_hyparams_ = {'alpha':[0.0001, 0.001, 0.01, 0.1, 1.0, 10.0], 'class_weight':['auto']} # 'loss':['hinge'], 
            self.clf_ = SGDClassifier(loss='log', n_jobs=4, n_iter=1, verbose=1)
        elif self.params_.classifier == 'gradient-boosting': 
            self.clf_hyparams_ = {'learning_rate':[0.01, 0.1, 0.2, 0.5]}
            self.clf_base_ = GradientBoostingClassifier()
        elif self.params_.classifier == 'extra-trees':             
            self.clf_hyparams_ = {'n_estimators':[10, 20, 40, 100]}
            self.clf_base_ = ExtraTreesClassifier()
        else: 
            raise Exception('Unknown classifier type %s. Choose from [sgd, svm, gradient-boosting, extra-trees]' 
                            % self.params_.classifier)



    def _extract(self, data_iterable, mode, batch_size=10): 
        """ Supports train/test modes """
        if not (mode == 'train' or mode == 'test' or mode == 'neg'): 
            raise Exception('Unknown mode %s' % mode)

        # Extract features, only if not already available
        features_dir = getattr(self.params_.cache, '%s_features_dir' % mode)
        mode_prefix = lambda name: ''.join(['%s_' % mode, name])
        print ''
        if not os.path.isdir(features_dir): 
            print '====> [COMPUTE] %s: Feature Extraction ' % mode.upper()
            st = time.time()
            # Add fields for training 
            fields = [mode_prefix('desc'), mode_prefix('target'), mode_prefix('pts'), mode_prefix('shapes')]

            # Add vocab_desc only for training
            if mode == 'train': fields.append('vocab_desc')

            # Initialize features_db
            features_db = IterDB(filename=features_dir, mode='w', fields=fields, batch_size=batch_size)

            # Closure for sugared item addition
            def add_item_to_features_db(features_db, im_desc, target, pts, shapes): 
                features_db.append(mode_prefix('desc'), im_desc)
                features_db.append(mode_prefix('target'), target)
                features_db.append(mode_prefix('pts'), pts)
                features_db.append(mode_prefix('shapes'), shapes) 

            # Closure for providing ROI to extracted features in image, 
            # and optionally adding to training set for vocab constr.
            def add_bbox(features_db, frame, pts, im_desc, mode, vocab_num_per_image): 
                if hasattr(frame, 'bbox') and len(frame.bbox): 
                    # Single image, multiple ROIs/targets
                    target = np.array([bbox['target'] for bbox in frame.bbox], dtype=np.int32)
                    shapes = np.vstack([ [bbox['left'], bbox['top'], bbox['right'], bbox['bottom']] for bbox in frame.bbox] )

                    add_item_to_features_db(features_db, im_desc, target, pts, shapes)

                elif hasattr(frame, 'target'): 
                    # Single image, single target
                    add_item_to_features_db(features_db, im_desc, frame.target, pts, np.array([[0, 0, frame.img.shape[1]-1, frame.img.shape[0]-1]]))

                elif mode == 'neg' and hasattr(frame, 'bbox_extract') and hasattr(frame, 'bbox_extract_target'): 
                    assert mode == 'neg', "Mode should be neg to continue, not %s" % mode
                    bboxes = frame.bbox_extract(frame)
                    shapes = (bboxes.reshape(-1,4)).astype(np.int32)
                    target = np.ones(len(bboxes), dtype=np.int32) * frame.bbox_extract_target

                    # Add all bboxes extracted via object proposals as bbox_extract_target
                    add_item_to_features_db(features_db, im_desc, target, pts, shapes)
                else: 
                    # print 'Nothing to do, frame.bbox is empty'
                    return 
                    

                # Randomly sample from descriptors for vocab construction
                if mode == 'train': 
                    inds = np.random.permutation(int(min(len(im_desc), vocab_num_per_image)))
                    features_db.append('vocab_desc', im_desc[inds])

            # Parallel Processing (in chunks of batch_size)
            if batch_size is not None and batch_size > 1: 
                for chunk in chunks(data_iterable, batch_size): 
                    res = Parallel(n_jobs=6, verbose=5) (
                        delayed(im_detect_and_describe)
                        (**dict(self.process_cb_(frame), **self.params_.descriptor)) for frame in chunk
                    )
                    for (pts, im_desc), frame in izip(res, chunk): 
                        if im_desc is None: continue

                        # Add bbox info to features db
                        add_bbox(features_db, frame, pts, im_desc, mode, self.params_.vocab.num_per_image)

            else: 
                # Serial Processing                    
                for frame in data_iterable: 
                    # Extract and add descriptors to db
                    pts, im_desc = self.image_descriptor_.detect_and_describe(**self.process_cb_(frame))
                    if im_desc is None: continue

                    # Add bbox info to features db
                    add_bbox(features_db, frame, pts, im_desc, mode, self.params_.vocab.num_per_image)

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
            print 'Codebook data: nimages (%i), vocab(%s)' % (len(inds), vocab_desc.shape)

            # Apply dimensionality reduction
            # Fit PCA to subset of data computed
            print '====> MEMORY: PCA dim. reduction before: %4.3f MB' % (vocab_desc.nbytes / 1024 / 1024.0) 
            if self.pca_ is not None: 
                vocab_desc = self.pca_.fit_transform(vocab_desc)
            print '====> MEMORY: PCA dim. reduction after: %4.3f MB' % (vocab_desc.nbytes / 1024 / 1024.0) 
            print '====> MEMORY: Codebook construction: %4.3f MB' % (vocab_desc.nbytes / 1024 / 1024.0) 

            # BOW construction
            if (self.params_.bow_method).lower() == 'flair': 
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
        hists_dir = hists_dir.replace('EPOCH', 'all')
        mode_prefix = lambda name: ''.join(['%s_' % mode, name])
        if not os.path.exists(hists_dir):
            print '====> [COMPUTE] BoVW / VLAD projection '
            hists_db = IterDB(filename=hists_dir, 
                              fields=[mode_prefix('histogram'), mode_prefix('target')], mode='w', batch_size=batch_size)

            if batch_size is not None and batch_size > 1: 
                # Parallel Processing
                for chunk in chunks(features_db.iter_keys_values([
                        mode_prefix('target'), mode_prefix('desc'), mode_prefix('pts'), mode_prefix('shapes')
                ], verbose=False), batch_size): 
                    desc_red = [self.pca_.transform(desc) if self.pca_ is not None else desc for (target, desc, _, _) in chunk] 

                    if (self.params_.bow_method).lower() == 'flair': 
                        # raise RuntimeError('Not implemented for parallel %s' % self.params_.bow_method)
                        res_hist = Parallel(n_jobs=6, verbose=5) (
                            delayed(flair_project)
                            (desc, self.bow_.codebook, pts=pts, shape=shape, 
                             levels=self.params_.bow.levels, method=self.params_.bow.method, step=self.params_.descriptor.step)
                            for desc, (_, _, pts, shape) in izip(desc_red, chunk)
                        )

                    elif (self.params_.bow_method).lower() == 'bow': 
                        res_hist = Parallel(n_jobs=6, verbose=5) (
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


    def _train(self, mode='train'): 
        # Retrieve mode
        mode_prefix = lambda name: ''.join(['%s_' % mode, name])

        if os.path.isdir(self.params_.cache.train_hists_dir): 
            print '====> [LOAD] TRAIN BoVW / VLAD projection '

            # Fit the training data on the first training pass

            hists_dir = getattr(self.params_.cache, '%s_hists_dir' % mode)
            hists_dir = hists_dir.replace('EPOCH', 'all')
            if not os.path.isdir(hists_dir): 
                raise RuntimeError('Hard Negative histograms not available %s' % hists_dir)

            if self.params_.classifier == 'sgd': 
                hists_db = IterDB(filename=hists_dir, mode='r')
                hists_iterable = hists_db.itervalues(mode_prefix('histogram'))
                targets_iterable = hists_db.itervalues(mode_prefix('target'))
            elif self.params_.classifier == 'svm': 
                # hists_db = IterDB(filename=hists_dir, mode='r')
                # hists_iterable = hists_db.itervalues(mode_prefix('histogram'))
                # targets_iterable = hists_db.itervalues(mode_prefix('target'))

                # Existing samples
                train_db = IterDB(filename=self.params_.cache.train_hists_dir, mode='r')
                hists_iterable = train_db.itervalues('train_histogram')
                targets_iterable = train_db.itervalues('train_target')

                if mode == 'neg': 
                    # Negative hists from each epoch
                    ep_hists_dir = getattr(self.params_.cache, 'neg_hists_dir')

                    for j in np.arange(self.epoch_no_): 
                        epj_hists_dir = ep_hists_dir.replace('EPOCH', '%i' % j)

                        neg_db = IterDB(filename=epj_hists_dir, mode='r')
                        hists_iterable = chain(hists_iterable, neg_db.itervalues('neg_histogram'))
                        targets_iterable = chain(targets_iterable, neg_db.itervalues('neg_target'))

                    # All the negative hists
                    allneg_hists_db = IterDB(filename=hists_dir, mode='r')
                    allneg_hists_iterable = allneg_hists_db.itervalues(mode_prefix('histogram'))
                    allneg_targets_iterable = allneg_hists_db.itervalues(mode_prefix('target'))
            
            else: 
                raise RuntimeError('Unknown classifier mode %s' % self.params_.classifier)
        
        else: 
            raise RuntimeError('Trained histograms not available %s' % self.params_.cache.train_hists_dir)

        # =================================================
        # Batch-wise (out-of-core) training

        # Train one-vs-all classifier
        print '-------------------------------'        
        print '====> Train classifier : %s' % (mode)
        st_clf = time.time()

        def pick_top_false_positives(clf, hists, targets): 
            clf_pred_scores = clf.decision_function(hists)
            clf_pred_targets = clf.predict(hists)
            ninds, = np.where(clf_pred_targets != targets)

            # Pick top 20 predictions
            inds = np.argsort(np.max(clf_pred_scores[ninds], axis=1), axis=0)[-100:]
            inds = ninds[inds]

            return inds

        if self.params_.classifier == 'sgd': 

            pred_targets, train_targets = [], []
            for hists_chunk, targets_chunk in izip(chunks(hists_iterable, 100), 
                                                   chunks(targets_iterable, 100)):
                train_hists_chunk = np.vstack([self.kernel_tf_.fit_transform(hist) 
                                               if self.kernel_tf_ is not None else hist for hist in hists_chunk])
                train_targets_chunk = np.hstack([ target for target in targets_chunk]).astype(np.int32)

                # TODO: Shuffle data

                # Negative mining, add only top k falsely predicted instances
                # ninds: inconsistent targets
                # Otherwise, add all relevant training data
                if mode == 'neg': 
                    # Pick top false positives
                    inds = pick_top_false_positives(self.clf_, train_hists_chunk, train_targets_chunk)

                    train_hists_chunk = train_hists_chunk[inds]
                    train_targets_chunk = train_targets_chunk[inds]

                # Provide all unique targets the classifier will expect, in the first fit
                self.clf_.partial_fit(train_hists_chunk, train_targets_chunk, 
                                      classes=self.target_ids_, #  if mode == 'train' else None, 
                                      sample_weight=None) # np.ones(len(train_hists_chunk)) * 0.01 if mode == 'neg' else None)

                # Predict targets
                pred_targets_chunk = self.clf_.predict(train_hists_chunk)
                pred_targets.append(pred_targets_chunk)
                train_targets.append(train_targets_chunk)

                print 'Adding %i,%i negative samples ' % (len(pred_targets_chunk), len(train_targets_chunk))

            try: 
                pred_targets = np.hstack(pred_targets)
                train_targets = np.hstack(train_targets)
            except: 
                return
            

        # =================================================
        # Batch (in-memory) training

        elif self.params_.classifier == 'svm': 

            # Setup train hists, and targets
            train_hists = np.vstack([item for item in hists_iterable]) 
            train_targets = np.hstack([item for item in targets_iterable]).astype(np.int32) 

            # Find false negatives from negative samples
            if mode == 'neg': 

                # Newly introduced negative examples into training
                ep_hists_dir = getattr(self.params_.cache, 'neg_hists_dir')
                epcurr_hists_dir = ep_hists_dir.replace('EPOCH', '%i' % self.epoch_no_)
                neg_hists_db = IterDB(filename=epcurr_hists_dir, 
                                      fields=['neg_histogram', 'neg_target'], mode='w', batch_size=10)

                neg_hists, neg_targets = [], []
                for hists_chunk, targets_chunk in izip(chunks(allneg_hists_iterable, 100), 
                                                       chunks(allneg_targets_iterable, 100)):
                    orig_hists_chunk = np.vstack([hist for hist in hists_chunk])
                    neg_hists_chunk = self.kernel_tf_.fit_transform(orig_hists_chunk) \
                                      if self.kernel_tf_ else orig_hists_chunk
                    neg_targets_chunk = np.hstack([ target for target in targets_chunk]).astype(np.int32)
                
                    inds = pick_top_false_positives(self.clf_, neg_hists_chunk, neg_targets_chunk)
                    neg_hists_db.append(mode_prefix('histogram'), orig_hists_chunk[inds])
                    neg_hists_db.append(mode_prefix('target'), neg_targets_chunk[inds])

                    neg_hists.append(orig_hists_chunk[inds])
                    neg_targets.append(neg_targets_chunk[inds])

                neg_hists_db.finalize()

                try: 
                    # Stack and train
                    train_hists = np.vstack([train_hists, np.vstack(neg_hists)])
                    train_targets = np.hstack([train_targets, np.hstack(neg_targets)])

                    # TODO: Shuffle data

                except Exception as e: 
                    print e

            # Kernel approximation
            if self.kernel_tf_ is not None: 
                print '====> [COMPUTE] TRAIN Kernel Approximator '
                train_hists = self.kernel_tf_.fit_transform(train_hists)
                print '-------------------------------'        


            # Grid search cross-val (best C param)
            cv = ShuffleSplit(len(train_hists), n_iter=10, test_size=0.3, random_state=4)
            clf_cv = GridSearchCV(self.clf_base_, self.clf_hyparams_, cv=cv, n_jobs=4, verbose=4)

            print '====> Training Classifier (with grid search hyperparam tuning) .. '
            print '====> BATCH Training (in-memory): %4.3f MB' % (train_hists.nbytes / 1024 / 1024.0) 
            clf_cv.fit(train_hists, train_targets)
            print 'BEST: ', clf_cv.best_score_, clf_cv.best_params_

            # Setting clf to best estimator
            self.clf_ = clf_cv.best_estimator_
            pred_targets = self.clf_.predict(train_hists)

            # Calibrating classifier
            print 'Calibrating Classifier ... '
            self.clf_prob_ = CalibratedClassifierCV(self.clf_, cv=cv, method='sigmoid')
            self.clf_prob_.fit(train_hists, train_targets)        

        print 'Training Classifier took %s' % (format_time(time.time() - st_clf))
        print '-------------------------------'        

        print ' Accuracy score (Training): %4.3f' % (metrics.accuracy_score(train_targets, pred_targets))
        print ' Report (Training):\n %s' % (classification_report(train_targets, pred_targets, 
                                                                  labels=self.target_map_.keys(), 
                                                                  target_names=self.target_map_.values()))
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

        # =================================================
        # Batch-wise (out-of-core) evaluation

        # Predict using classifier
        print '-------------------------------'
        print '====> Predict using classifer '
        st_clf = time.time()

        pred_targets, pred_scores, test_targets = [], [], []
        for hists_chunk, targets_chunk in izip(chunks(hists_db.itervalues('test_histogram'), 100), 
                                               chunks(hists_db.itervalues('test_target'), 100)):
            test_hists_chunk = np.vstack([self.kernel_tf_.transform(hist) 
                                           if self.kernel_tf_ is not None else hist for hist in hists_chunk])
            test_targets_chunk = np.hstack([ target for target in targets_chunk]).astype(np.int32)

            # Predict targets
            pred_targets_chunk = self.clf_.predict(test_hists_chunk)
            pred_scores_chunk = self.clf_.decision_function(test_hists_chunk)
            pred_targets.append(pred_targets_chunk)
            pred_scores.append(pred_scores_chunk)
            test_targets.append(test_targets_chunk)

        pred_scores = np.vstack(pred_scores)
        pred_targets = np.hstack(pred_targets)
        test_targets = np.hstack(test_targets)
        print '-------------------------------'

        print '=========================================================> '
        print '\n ===> Classification @ ', datetime.datetime.now()
        # print 'Params: \n'
        # pp = pprint.PrettyPrinter(indent=4)
        # pp.pprint(self.params_)
        # print '\n'
        print '-----------------------------------------------------------'
        print ' Accuracy score (Test): %4.3f' % (metrics.accuracy_score(test_targets, pred_targets))
        print ' Report (Test):\n %s' % (classification_report(test_targets, pred_targets, 
                                                              labels=self.target_map_.keys(), 
                                                              target_names=self.target_map_.values()))
        cmatrix = metrics.confusion_matrix(test_targets, pred_targets, labels=self.target_map_.keys())
        
        # print ' Confusion matrix (Test): \n%s' % (''.join(['{:15s} {:3s}\n'.format(name, cmatrix[idx]) 
        #                                                    for idx,name in enumerate(self.target_map_.values())]))
        print ' Confusion matrix (Test): \n%s' % (pd.DataFrame(cmatrix, 
                                                               columns=self.target_map_.values(), 
                                                               index=self.target_map_.values()))
        
        # plot_precision_recall(pred_scores, test_targets, self.target_map_)
        print 'Testing took %s' % format_time(time.time() - st_clf)

        return AttrDict(test_targets=test_targets, pred_targets=pred_targets, pred_scores=pred_scores, 
                        target_map=self.target_map_)


    def preprocess(self, train_data_iterable, test_data_iterable, bg_data_iterable, batch_size=10): 
        # Extract training and testing features via self.process_cb_
        self._extract(train_data_iterable, mode='train', batch_size=batch_size)
        self._extract(test_data_iterable, mode='test', batch_size=batch_size)
        if self.params_.neg_epochs: 
            self._extract(bg_data_iterable, mode='neg', batch_size=batch_size)

    def train(self, data_iterable, batch_size=10): 
        # 1. Extract D-SIFT features 
        self._extract(data_iterable, mode='train', batch_size=batch_size)

        # 2. Construct vocabulary using PCA-SIFT
        self._build()

        # 2. Reduce dimensionality and project onto 
        #    histogram with spatial pyramid pooling
        self._project(mode='train', batch_size=batch_size)

        # 4. Kernel approximation and linear classification training
        self._train(mode='train')

    def neg_train(self, bg_data_iterable, batch_size=10, viz_cb=lambda e: None): 

        # 5. Hard-neg mining
        for epoch in np.arange(self.params_.neg_epochs): 
            self.epoch_no_ = epoch
            print 'Processing epoch %i' % (epoch)

            if epoch == 0:
                # 1. Extract features
                self._extract(bg_data_iterable, mode='neg', batch_size=batch_size)

                # 2. Project
                self._project(mode='neg', batch_size=batch_size)

            # 3. Re-train
            self._train(mode='neg')

            viz_cb(epoch)
            # training = self._hard_negative_mining(training, epoch)
            # training.save(os.path.join(conf.results_dir, 'training-epoch-%i.h5' % (epoch)))


    def evaluate(self, data_iterable, batch_size=10): 
        # 1. Extract D-SIFT features 
        self._extract(data_iterable, mode='test', batch_size=batch_size)

        # 2. Reduce dimensionality and project onto 
        #    histogram with spatial pyramid pooling
        self._project(mode='test', batch_size=batch_size)

        # 4. Kernel approximate and linear classification eval
        self._classify()

    # def classify_with_mask(self, img, mask=None): 
    #     """ Describe image (with mask) with a BOW histogram  """
    #     test_desc = self.image_descriptor_.describe(img, mask=mask) 
    #     if self.pca_ is not None: 
    #         test_desc = self.pca_.transform(test_desc)
    #     test_hist = self.bow_.project(test_desc_red)
    #     if self.kernel_tf_ is not None: 
    #         test_hist = self.kernel_tf.transform(test_hist) 
    #     # pred_target_proba = self.clf_.decision_function(test_histogram)
    #     pred_target, = self.clf_.predict(test_histogram)

    #     return pred_target

    def setup_flair(self, W, H): 
        if not hasattr(self.bow_, 'codebook') or \
           not hasattr(self.bow_, 'dictionary_size') or \
           not hasattr(self.params_.descriptor, 'step' ) or \
           not hasattr(self.params_.bow, 'levels') or \
           not hasattr(self.params_.bow, 'method'): 
            raise RuntimeError('unknown params for flair setup')

        if self.bow_.codebook is None: 
            raise RuntimeError('Vocabulary not setup')

        # Setup flair encoding
        self.flair_ = FLAIR_code(W=W, H=H, 
                                 K=self.bow_.dictionary_size, step=self.params_.descriptor.step, 
                                 levels=np.array(self.params_.bow.levels, dtype=np.int32), 
                                 encoding={'bow':0, 'vlad':1, 'fisher':2}[self.params_.bow.method])
        
        # Provide vocab for flair encoding
        self.flair_.setVocabulary(self.bow_.codebook.astype(np.float32))

    def describe(self, img, bboxes): 
        """ 
        Describe img regions defined by the bboxes
        FLAIR classification (one-time descriptor evaluation)

        Params: 
             img:    np.uint8
             bboxes: np.float32 [bbox['left'], bbox['top'], bbox['right'], bbox['bottom'], ...] 

        """
        if not hasattr(self, 'flair_'): 
            raise RuntimeError('FLAIR not setup yet, run setup_flair() first!')

        # Construct rectangle info
        im_rect = np.vstack([[bbox['left'], bbox['top'], 
                              bbox['right'], bbox['bottom']] \
                             for bbox in bboxes]).astype(np.float32)
        
        try: 
            # Extract image features
            # if self.params_.profile: self.profiler[self.params_.descriptor.descriptor].start()
            pts, desc = self.image_descriptor_.detect_and_describe(img=img, mask=None)
            # if self.params_.profile: self.profiler[self.params_.descriptor.descriptor].stop()

            # Reduce dimensionality
            # if self.params_.profile: self.profiler['dim_red'].start()
            if self.params_.do_pca: 
                desc = self.pca_.transform(desc)
            # if self.params_.profile: self.profiler['dim_red'].stop()

            # Extract histogram
            # pts: int32, desc: float32, im_rect: float32
            # if self.params_.profile: self.profiler['FLAIR'].start()
            hists = self.flair_.process(pts.astype(np.int32), desc.astype(np.float32), im_rect)
            # if self.params_.profile: self.profiler['FLAIR'].stop()

            # Transform histogram using kernel feature map
            if self.params_.do_kernel_approximation: 
                hists = self.kernel_tf_.transform(hists)
        except Exception as e: 
            # if self.params_.profile: self.profiler[self.params_.descriptor.descriptor].stop(force=True)
            print 'Failed to extract FLAIR and process', e
            return None

        return hists

    def process(self, img, bboxes): 
        """ 
        Describe img and predict with corresponding bboxes 
        """
        if not len(bboxes): 
            return None, None

        # Batch predict targets and visualize
        hists = self.describe(img, bboxes)

        if hists is None: 
            return None, None

        pred_target = self.clf_.predict(hists)
        # if self.params.profile: self.profiler['predict'].start()
        pred_prob = self.clf_prob_.predict_proba(hists) 
        # if self.params.profile: self.profiler['predict'].stop()
        # pred_score = self.clf_.decision_function(hists)

        return pred_prob, pred_target
    
    def get_catgory_id(self, col_ind): 
        return self.clf_.classes_.take(np.asarray(col_ind, dtype=np.intp)) 

    def get_categories(self): 
        return self.clf_.classes_

    def _setup_from_dict(self, db): 
        try: 
            self.params_ = db.params
            self.image_descriptor_ = ImageDescription(**db.params.descriptor)
            self.bow_ = BoWVectorizer.from_dict(db.bow)
            self.pca_ = db.pca
            self.kernel_tf_ = db.kernel_tf
            self.clf_base_, self.clf_, self.clf_hyparams_ = db.clf_base, db.clf, db.clf_hyparams
            self.clf_prob_ = db.clf_prob
        except KeyError: 
            raise RuntimeError('DB not setup correctly, try re-training!')
        
    def save(self, path): 
        db = AttrDict(params=self.params_, 
                      bow=self.bow_.to_dict(), pca=self.pca_, kernel_tf=self.kernel_tf_, 
                      clf=self.clf_, clf_base=self.clf_base_, clf_hyparams=self.clf_hyparams_, 
                      clf_prob=self.clf_prob_, target_map=self.target_map_)
        db.save(path)

    @classmethod
    def from_dict(cls, db): 
        c = cls(params=db.params, target_map=dict((int(key), item) for key,item in db.target_map.iteritems()))
        c._setup_from_dict(db)
        return c

    @classmethod
    def load(cls, path): 
        db = AttrDict.load(path)
        return cls.from_dict(db)


class HistogramClassifier(BOWClassifier): 
    """
    Usage: 

        bc = HistogramClassifier(params)

        train_dataset = caltech_training
        test_dataset = caltech_testing

        # Extract features in advance and stores
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
    def __init__(self, params, target_map, 
                 process_cb=lambda f: dict(img=cv2.imread(f.filename), mask=None)): 

        # Setup Params
        self.params_ = params
        self.process_cb_ = process_cb
        self.target_map_ = target_map
        self.target_ids_ = (np.unique(target_map.keys())).astype(np.int32)
        self.epoch_no_ = 0

        # 1. Image description using Dense SIFT/Descriptor
        self.image_descriptor_ = CNNDescription(**self.params_.descriptor)

        # 2. Setup dim. red
        self.pca_ = RandomizedPCA(**self.params_.pca) if self.params_.do_pca else None

        # # 3. Bag-of-words VLAD/VQ
        # # a. Traditional BOW, b. FLAIR
        # if (self.params_.bow_method).lower() == 'flair': 
        #     self.bow_ = BoWVectorizer(**self.params_.bow)

        # elif (self.params_.bow_method).lower() == 'bow': 
        #     self.bow_ = BoWVectorizer(**self.params_.bow)

        # else: 
        #     raise Exception('Unknown bow_method %s' % self.params_.bow_method)

        # # 4. Setup kernel feature map
        # self.kernel_tf_ = HomogenousKernelMap(2) if self.params_.do_kernel_approximation else None
        self.kernel_tf_ = None

        # 5. Setup classifier
        print '-------------------------------'        
        print '====> Building Classifier, setting class weights' 
        cweights = {cid: 10.0 for cidx,cid in enumerate(self.target_ids_)}
        cweights[51] = 1.0
        print 'Weights: ', self.target_ids_, cweights

        if self.params_.classifier == 'svm': 
            self.clf_hyparams_ = {'C':[0.01, 0.1, 0.5, 1.0, 4.0, 5.0, 10.0]} # , 'class_weight': ['auto']}
            self.clf_base_ = LinearSVC(random_state=1)
        elif self.params_.classifier == 'sgd': 
            self.clf_hyparams_ = {'alpha':[0.0001, 0.001, 0.01, 0.1, 1.0, 10.0], 'class_weight':['auto']} # 'loss':['hinge'], 
            self.clf_ = SGDClassifier(loss='log', n_jobs=4, n_iter=1, verbose=1)
        elif self.params_.classifier == 'gradient-boosting': 
            self.clf_hyparams_ = {'learning_rate':[0.01, 0.1, 0.2, 0.5]}
            self.clf_base_ = GradientBoostingClassifier()
        elif self.params_.classifier == 'extra-trees':             
            self.clf_hyparams_ = {'n_estimators':[10, 20, 40, 100]}
            self.clf_base_ = ExtraTreesClassifier()
        else: 
            raise Exception('Unknown classifier type %s. Choose from [sgd, svm, gradient-boosting, extra-trees]' 
                            % self.params_.classifier)

    def _extract(self, data_iterable, mode, batch_size=10): 
        """ Supports train/test modes """
        if not (mode == 'train' or mode == 'test' or mode == 'neg'): 
            raise Exception('Unknown mode %s' % mode)

        # Extract features, only if not already available
        hists_dir = getattr(self.params_.cache, '%s_hists_dir' % mode)
        hists_dir = hists_dir.replace('EPOCH', 'all')
        mode_prefix = lambda name: ''.join(['%s_' % mode, name])
        if not os.path.isdir(hists_dir): 
            print '====> [COMPUTE] %s: Feature Extraction ' % mode.upper()
            st = time.time()

            # Initialize hists_db
            hists_db = IterDB(filename=hists_dir, 
                              fields=[mode_prefix('histogram'), mode_prefix('target')], mode='w', batch_size=batch_size)

            # Closure for sugared item addition
            def add_item_to_hists_db(hists_db, im_desc, target, bboxes): 
                hists_db.append(mode_prefix('histogram'), im_desc)
                hists_db.append(mode_prefix('target'), target)

            # Closure for providing ROI to extracted features in image, 
            # and optionally adding to training set for vocab constr.
            def add_bbox(hists_db, frame, mode): 
                if hasattr(frame, 'bbox') and len(frame.bbox): 
                    # Single image, multiple ROIs/targets
                    target = np.array([bbox['target'] for bbox in frame.bbox], dtype=np.int32)
                    bboxes = np.vstack([[bbox['left'], bbox['top'], bbox['right'], bbox['bottom']] for bbox in frame.bbox])

                    im_desc = self.image_descriptor_.describe(**self.process_cb_(frame))
                    add_item_to_hists_db(hists_db, im_desc, target, bboxes)

                elif hasattr(frame, 'target'): 
                    # Single image, single target
                    sz = frame.img.shape[:2]
                    bboxes = np.array([[0,0,sz[1]-1, sz[0]-1]])
                    im_desc = self.image_descriptor_.describe(frame.img, bboxes)
                    add_item_to_hists_db(hists_db, im_desc, frame.target, bboxes)

                elif mode == 'neg' and hasattr(frame, 'bbox_extract') and hasattr(frame, 'bbox_extract_target'): 
                    assert mode == 'neg', "Mode should be neg to continue, not %s" % mode
                    bboxes = frame.bbox_extract(frame)
                    target = np.ones(len(bboxes), dtype=np.int32) * frame.bbox_extract_target
                    
                    # Add all bboxes extracted via object proposals as bbox_extract_target
                    im_desc = self.image_descriptor_.describe(frame.img, bboxes)
                    add_item_to_hists_db(hists_db, im_desc, target, bboxes)
                else: 
                    print 'Nothing to do, frame.bbox is empty'
                    return 

            # Parallel Processing (in chunks of batch_size)
            if 0: # batch_size is not None and batch_size > 1: 
                raise RuntimeError('Do not have support for parallel')
                # for chunk in chunks(data_iterable, batch_size): 
                #     res = Parallel(n_jobs=6, verbose=5) (
                #         delayed(im_detect_and_describe)
                #         (**dict(self.process_cb_(frame), **self.params_.descriptor)) for frame in chunk
                #     )
                #     for (pts, im_desc), frame in izip(res, chunk): 
                #         if im_desc is None: continue

                #         # Add bbox info to hists db
                #         add_bbox(hists_db, frame, im_desc, mode, self.params_.vocab.num_per_image)

            else: 
                # Serial Processing                    
                for frame in data_iterable: 
                    # Extract and add descriptors to db
                    add_bbox(hists_db, frame, mode)

            hists_db.finalize()
            print '[%s] Descriptor extraction took %s' % (mode.upper(), format_time(time.time() - st))    
        print '-------------------------------'


    def train(self, data_iterable, batch_size=10): 
        # 1. Extract features 
        self._extract(data_iterable, mode='train', batch_size=batch_size)

        # 4. Linear classification
        self._train(mode='train')

    def evaluate(self, data_iterable, batch_size=10): 
        # 1. Extract D-SIFT features 
        self._extract(data_iterable, mode='test', batch_size=batch_size)
        
        print 'data_iterable', data_iterable

        # 4. Linear classification eval
        self._classify()

    def describe(self, img, bboxes): 
        """ 
        Describe img regions defined by the bboxes
        FLAIR classification (one-time descriptor evaluation)

        Params: 
             img:    np.uint8
             bboxes: np.float32 [bbox['left'], bbox['top'], bbox['right'], bbox['bottom'], ...] 

        """
        return self.image_descriptor_.describe(img, bboxes)

    def process(self, img, bboxes): 
        """ 
        Describe img and predict with corresponding bboxes 
        """
        if not len(bboxes): 
            return None, None

        # Batch predict targets and visualize
        hists = self.describe(img, bboxes)

        if hists is None: 
            return None, None

        pred_target = self.clf_.predict(hists)
        # if self.params.profile: self.profiler['predict'].start()
        pred_prob = self.clf_prob_.predict_proba(hists) 
        # if self.params.profile: self.profiler['predict'].stop()
        # pred_score = self.clf_.decision_function(hists)

        return pred_prob, pred_target
    

    def _setup_from_dict(self, db): 
        try: 
            self.params_ = db.params
            self.image_descriptor_ = CNNDescription(**db.params.descriptor)
            # self.bow_ = BoWVectorizer.from_dict(db.bow)
            self.pca_ = db.pca
            self.kernel_tf_ = db.kernel_tf
            self.clf_base_, self.clf_, self.clf_hyparams_ = db.clf_base, db.clf, db.clf_hyparams
            self.clf_prob_ = db.clf_prob
        except KeyError: 
            raise RuntimeError('DB not setup correctly, try re-training!')
        
    def save(self, path): 
        db = AttrDict(params=self.params_, 
                      # bow=self.bow_.to_dict(), 
                      pca=self.pca_, kernel_tf=self.kernel_tf_, 
                      clf=self.clf_, clf_base=self.clf_base_, clf_hyparams=self.clf_hyparams_, 
                      clf_prob=self.clf_prob_, target_map=self.target_map_)
        db.save(path)

    @classmethod
    def from_dict(cls, db): 
        c = cls(params=db.params, target_map=dict((int(key), item) for key,item in db.target_map.iteritems()))
        c._setup_from_dict(db)
        return c

    @classmethod
    def load(cls, path): 
        db = AttrDict.load(path)
        return cls.from_dict(db)

class ObjectProposer(object): 
    """
    Usage: 

        prop = ObjectProposer.create('GOP', params)
        bboxes = prop.process(im)
        
    """
    def __init__(self): 
        pass

    @staticmethod
    def create(method='GOP', params=None): 
        if method == 'GOP': 
            return GOPObjectProposer(**params)
        elif method == 'BING': 
            return BINGObjectProposer(**params)
        else: 
            raise RuntimeError('Unknown proposals method: %s' % method)

class GOPObjectProposer(ObjectProposer): 
    def __init__(self, num_proposals=1000): 
        from bot_vision.recognition.gop_util import setupLearned as gop_setuplearned

        gop_data_dir = """/home/spillai/perceptual-learning/software/externals/recognition-pod/proposals/gop/data/"""
        self.prop_ = gop.proposals.Proposal( gop_setuplearned( 140, 4, 0.8, gop_data_dir=gop_data_dir ) )

        self.num_proposals_ = num_proposals
        self.segmenter_ = gop.contour.MultiScaleStructuredForest()
        self.segmenter_.load( os.path.join(gop_data_dir, 'sf.dat'))

    def process(self, im): 
        im_ = gop.imgproc.npread(np.copy(im))
        s = gop.segmentation.geodesicKMeans( im_, self.segmenter_, self.num_proposals_ )
        b = self.prop_.propose( s )
        return s.maskToBox( b )

class BINGObjectProposer(ObjectProposer): 
    def __init__(self, voc_dir, num_proposals=10): 
        self.proposer_ = BINGObjectness()
        self.num_proposals_ = num_proposals
        try: 
            self.proposer_.build(voc_dir)
        except: 
            raise RuntimeError('Error with selective search voc dir %s' % params.voc_dir)

    def process(self, im): 
        return self.proposer_.process(im, self.num_proposals_)
            
