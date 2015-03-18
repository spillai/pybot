import numpy as np
import cv2, os, time, random
from itertools import izip, chain

from bot_vision.image_utils import im_resize, gaussian_blur, median_blur, box_blur
from bot_vision.bow_utils import BoWVectorizer
import bot_vision.mser_utils as mser_utils

import bot_utils.io_utils as io_utils
from bot_utils.io_utils import memory_usage_psutil
from bot_utils.db_utils import AttrDict, IterDB
from bot_utils.itertools_recipes import chunks

import sklearn.metrics as metrics
from sklearn.decomposition import PCA, RandomizedPCA
from sklearn.svm import LinearSVC, SVC
from sklearn.linear_model import SGDClassifier
from sklearn.grid_search import GridSearchCV
from sklearn.cross_validation import train_test_split, ShuffleSplit
from sklearn.kernel_approximation import AdditiveChi2Sampler, RBFSampler
from sklearn.pipeline import Pipeline

from sklearn.externals.joblib import Parallel, delayed

class HomogenousKernelMap(AdditiveChi2Sampler): 
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

def get_dense_detector(step=4, levels=7, scale=np.sqrt(2)): 
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
    if detector == 'dense': 
        return get_dense_detector(step=step, levels=levels, scale=scale)
    else: 
        return cv2.FeatureDetector_create(detector)

def im_detect_and_describe(img, mask=None, detector='dense', descriptor='SIFT', step=4, levels=7, scale=np.sqrt(2)): 
    detector = get_detector(detector=detector, step=step, levels=levels, scale=scale)
    extractor = cv2.DescriptorExtractor_create(descriptor)
    
    kpts = detector.detect(img, mask=mask)
    kpts, desc = extractor.compute(img, kpts)
    return kpts, desc

def im_describe(*args, **kwargs): 
    kpts, desc = im_detect_and_describe(*args, **kwargs)
    return desc

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

            # # RootSIFT
            # # pre_shape = desc.shape
            # desc = np.sqrt(desc.astype(np.float32) / (np.sum(desc, axis=1)).reshape(-1,1))
            # inds, = np.where(np.isfinite(desc).all(axis=1))
            # kpts, desc = [kpts[ind] for ind in inds], desc[inds]
            # # post_shape = desc.shape
            # # print pre_shape, post_shape

            # # Extract color information (Lab)
            # pts = np.vstack([kp.pt for kp in kpts]).astype(np.int32)
            # imgc = median_blur(img, size=5) 
            # cdesc = img[pts[:,1], pts[:,0]]
            # return kpts, np.hstack([desc, cdesc])

            return kpts, desc

        except Exception as e: 
            print e
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

        # Setup dim. red
        self.pca = RandomizedPCA(**self.params.pca)

        # Setup kernel feature map
        # kernel_tf = [('chi2_kernel', HomogenousKernelMap(2)), ('rbf_sampler', RBFSampler(gamma=0.2, n_components=40))]
        # self.kernel_tf = Pipeline(kernel_tf) if self.params.do_kernel_approximation else None
        self.kernel_tf = HomogenousKernelMap(2) if self.params.do_kernel_approximation else None

        # Setup classifier
        print 'Building Classifier'
        # cv_params = {'loss':['hinge'], 'alpha':[0.01, 0.1, 1.0, 10.0], 'class_weight':['auto']}
        # self._clf = SGDClassifier(loss='hinge', n_jobs=4)

        # Linear SVM
        self.clf_hyparams = {'C':[0.01, 0.1, 0.5, 1.0, 10.0]}
        self.clf = LinearSVC(random_state=1, tol=1e-4)

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

        self.X_all = list(chain(self.X_train, self.X_test))
        self.y_all = list(chain(self.y_train, self.y_test))

    def extract_features(self): 
        if self.clf_pretrained: 
            return 

        if not hasattr(self, 'X_train') or not hasattr(self, 'y_train'): 
            raise RuntimeError('Training cannot proceed. Setup training and testing samples first!')            

        # Extract training features, only if not already available
        if not os.path.isdir(self.params.cache.train_path): 
            print '====> [COMPUTE] TRAINING: Feature Extraction '        
            st = time.time()
            vocab_construction = AttrDict(num_per_image=1e3)
            features_db = IterDB(filename=self.params.cache.train_path, mode='w', 
                                 fields=['train_desc', 'train_target', 'vocab_desc'], batch_size=50)

            # Parallel Processing
            for chunk in chunks(izip(self.X_train, self.y_train), 50): 
                res = Parallel(n_jobs=8, verbose=5) (
                    delayed(im_describe)
                    (**dict(self.process_cb(x_t), **self.params.descriptor)) for (x_t,_) in chunk
                )

                for im_desc, (_, y_t) in izip(res, chunk): 
                    features_db.append('train_desc', im_desc)
                    features_db.append('train_target', y_t)

                    # Randomly sample from descriptors for vocab construction
                    inds = np.random.permutation(int(min(len(im_desc), vocab_construction.num_per_image)))
                    features_db.append('vocab_desc', im_desc[inds])

            # Serial Processing                    
            # for (x_t,y_t) in izip(self.X_train, self.y_train): 
            #     # Extract and add descriptors to db
            #     im_desc = self.image_descriptor.describe(**self.process_cb(x_t))
            #     features_db.append('train_desc', im_desc)
            #     features_db.append('train_target', y_t)

            #     # Randomly sample from descriptors for vocab construction
            #     inds = np.random.permutation(int(min(len(im_desc), vocab_construction.num_per_image)))
            #     features_db.append('vocab_desc', im_desc[inds])

            features_db.finalize()
            print '[TRAIN] Descriptor extraction took %5.3f s' % (time.time() - st)    

        print '-------------------------------'

        # Extract test features
        if not os.path.isdir(self.params.cache.test_path): 
            print '====> [COMPUTE] TESTING: Feature Extraction '        
            st = time.time()
            features_db = IterDB(filename=self.params.cache.test_path, mode='w', 
                                 fields=['test_desc', 'test_target'], batch_size=50)


            # Parallel Processing
            for chunk in chunks(izip(self.X_test, self.y_test), 50): 
                res = Parallel(n_jobs=8, verbose=5) (
                    delayed(im_describe)
                    (**dict(self.process_cb(x_t), **self.params.descriptor)) for (x_t,_) in chunk
                )
                for im_desc, (_, y_t) in izip(res, chunk): 
                    features_db.append('test_desc', im_desc)
                    features_db.append('test_target', y_t)

            # Serial Processing
            # for (x_t,y_t) in izip(self.X_test, self.y_test): 
            #     features_db.append('test_desc', self.image_descriptor.describe(**self.process_cb(x_t)))
            #     features_db.append('test_target', y_t)

            features_db.finalize()
            print '[TEST] Descriptor extraction took %5.3f s' % (time.time() - st)    
        print '-------------------------------'


    def train(self): 
        if self.clf_pretrained: 
            return 

        if not hasattr(self, 'X_train') or not hasattr(self, 'y_train'): 
            raise RuntimeError('Training cannot proceed. Setup training and testing samples first!')            

        print '===> Training '
        st = time.time()

        # Extract features
        if not self.params.cache.features_dir: 
            raise RuntimeError('Setup features_dir before running training')

        # Extract features, only if not already available
        if not os.path.isdir(self.params.cache.train_path): 
            print '====> [COMPUTE] Feature Extraction '        
            vocab_construction = AttrDict(num_per_image=1e3)
            features_db = IterDB(filename=self.params.cache.train_path, mode='w', 
                                 fields=['train_desc', 'train_target', 'vocab_desc'])
            # Serial Processing
            for (x_t,y_t) in izip(self.X_train, self.y_train): 
                # Extract and add descriptors to db
                im_desc = self.image_descriptor.describe(**self.process_cb(x_t))
                features_db.append('train_desc', im_desc)
                features_db.append('train_target', y_t)

                # Randomly sample from descriptors for vocab construction
                inds = np.random.permutation(int(min(len(im_desc), vocab_construction.num_per_image)))
                features_db.append('vocab_desc', im_desc[inds])

            features_db.finalize()
            print 'Descriptor extraction took %5.3f s' % (time.time() - st)    
        else: 
            print '====> [LOAD] Feature Extraction'        
            features_db = IterDB(filename=self.params.cache.train_path, mode='r')
        print '-------------------------------'

        # Build BOW
        if not os.path.exists(self.params.cache.vocab_path): # or self.params.cache.overwrite: 
            print '====> [COMPUTE] Vocabulary Construction'
            vocab_construction = AttrDict(num_images=1e3)
            inds = np.random.permutation(len(self.X_train))[:vocab_construction.num_images]
            vocab_desc = np.vstack([item for item in features_db.itervalues('vocab_desc', inds=inds, verbose=True)])
            print 'Codebook data: %i, %i' % (len(inds), len(vocab_desc))

            # Apply dimensionality reduction
            # Fit PCA to subset of data computed
            print '====> [REDUCE] PCA dim. reduction before: %4.3f MB' % (vocab_desc.nbytes / 1024 / 1024.0) 
            if self.params.do_pca: 
                vocab_desc = self.pca.fit_transform(vocab_desc)
            print '====> [REDUCE] PCA dim. reduction after: %4.3f MB' % (vocab_desc.nbytes / 1024 / 1024.0) 

            print '====> MEMORY: Codebook construction: %4.3f MB' % (vocab_desc.nbytes / 1024 / 1024.0) 
            self.bow.build(vocab_desc)
            vocab_desc = None
            vocab_db = AttrDict(params=self.params, bow=self.bow.to_dict())
            vocab_db.save(self.params.cache.vocab_path)
            print 'Codebook: %s' % ('GOOD' if np.isfinite(self.bow.codebook).all() else 'BAD')
        else: 
            print '====> [LOAD] Vocabulary Construction'
            vocab_db = AttrDict.load(self.params.cache.vocab_path)
            self.bow = BoWVectorizer.from_dict(vocab_db.bow)
        print '-------------------------------'

        # Histogram of trained features
        if not os.path.exists(self.params.cache.train_hists_path): #  or self.params.cache.overwrite: 
            print '====> [COMPUTE] BoVW / VLAD projection '
            train_target = np.array(self.y_train, dtype=np.int32)
            train_histogram = np.vstack([self.bow.project(
                self.pca.transform(desc) if self.params.do_pca else desc
            ) for desc in features_db.itervalues('train_desc', verbose=True)])

            hists_db = AttrDict(train_target=train_target, train_histogram=train_histogram)
            hists_db.save(self.params.cache.train_hists_path)
            print '====> MEMORY: Histogram: %s %4.3f MB' % (train_histogram.shape, 
                                                            train_histogram.nbytes / 1024 / 1024.0) 
        else: 
            print '====> [LOAD] BoVW / VLAD projection '
            hists_db = AttrDict.load(self.params.cache.train_hists_path)
            train_target, train_histogram = hists_db.train_target, hists_db.train_histogram
        print '-------------------------------'

        # # PCA dim. red
        # if self.params.do_pca: 
        #     print '====> PCA '            
        #     train_histogram = self.pca.fit_transform(train_histogram)
        #     print '-------------------------------'        

        # Homogeneous Kernel map
        if self.params.do_kernel_approximation: 
            print '====> Kernel Approximation '
            train_histogram = self.kernel_tf.fit_transform(train_histogram)
            print '-------------------------------'        

        # Train/Predict one-vs-all classifier
        print '====> Train classifier '
        st_clf = time.time()

        # Grid search cross-val
        cv = ShuffleSplit(len(train_histogram), n_iter=10, test_size=0.3, random_state=4)
        self.clf = GridSearchCV(self.clf, self.clf_hyparams, cv=cv, n_jobs=8, verbose=1)
        self.clf.fit(train_histogram, train_target)
        self.clf = self.clf.best_estimator_
        pred_target = self.clf.predict(train_histogram)

        print 'Training Classifier took %5.3f s' % (time.time() - st_clf)
        print '-------------------------------'        


        print ' Accuracy score (Training): %4.3f' % (metrics.accuracy_score(train_target, pred_target))
        print ' Report (Training):\n %s' % (metrics.classification_report(train_target, pred_target, 
                                                                          target_names=self.dataset.target_names))

        print 'Training took %5.3f s' % (time.time() - st)

        print '====> Saving classifier '
        self.save(self.params.cache.detector_path)
        print '-------------------------------'
        return

    def classify(self): 
        print '===> Classification '
        st = time.time()

        # Extract features
        if not os.path.isdir(self.params.cache.test_path): 
            print '====> [COMPUTE] Feature Extraction '        
            features_db = IterDB(filename=self.params.cache.test_path, mode='w', 
                                 fields=['test_desc', 'test_target'], batch_size=5)
            for (x_t,y_t) in izip(self.X_test, self.y_test): 
                features_db.append('test_desc', self.image_descriptor.describe(**self.process_cb(x_t)))
                features_db.append('test_target', y_t)
            features_db.finalize()
        else: 
            print '====> [LOAD] Feature Extraction'        
            features_db = IterDB(filename=self.params.cache.test_path, mode='r')
        print '-------------------------------'
        print 'Descriptor extraction took %5.3f s' % (time.time() - st)    

        # Load Vocabulary
        if os.path.exists(self.params.cache.vocab_path):
            print '====> [LOAD] Vocabulary Construction'
            vocab_db = AttrDict.load(self.params.cache.vocab_path)
            self.bow = BoWVectorizer.from_dict(vocab_db.bow)
        else: 
            raise RuntimeError('Vocabulary not built %s' % self.params.cache.vocab_path)

        # Histogram of trained features
        if not os.path.exists(self.params.cache.test_hists_path): #  or self.params.cache.overwrite: 
            print '====> [COMPUTE] BoVW / VLAD projection '
            test_target = self.y_test
            test_histogram = np.vstack([self.bow.project(
                self.pca.transform(desc) if self.params.do_pca else desc
            ) for desc in features_db.itervalues('test_desc', verbose=True)])
            hists_db = AttrDict(test_target=test_target, test_histogram=test_histogram)
            hists_db.save(self.params.cache.test_hists_path)
            print '====> MEMORY: Histogram: %s %4.3f MB' % (test_histogram.shape, 
                                                            test_histogram.nbytes / 1024 / 1024.0) 
        else: 
            print '====> [LOAD] BoVW / VLAD projection '
            hists_db = AttrDict.load(self.params.cache.test_hists_path)
            test_target, test_histogram = hists_db.test_target, hists_db.test_histogram
        print '-------------------------------'

        # # PCA dim. red
        # if self.params.do_pca: 
        #     print '====> PCA '            
        #     test_histogram = self.pca.transform(test_histogram)
        #     print '-------------------------------'        

        if self.params.do_kernel_approximation: 
            # Apply homogeneous transform
            test_histogram = self.kernel_tf.transform(test_histogram)

        print '====> Predict using classifer '
        pred_target = self.clf.predict(test_histogram)
        pred_score = self.clf.decision_function(test_histogram)
        print '-------------------------------'

        # print ' Confusion matrix (Test): %s' % (metrics.confusion_matrix(test_target, pred_target))
        print '=========================================================> '
        print 'Params: ', self.params
        print '-----------------------------------------------------------'
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

        
