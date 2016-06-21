from fast_rcnn.config import cfg
import os
import warnings

# Import FastRCNN
with warnings.catch_warnings():
    warnings.simplefilter("ignore")
    from bot_vision.caffe.fast_rcnn_utils import FastRCNNDescription, nms

# Import FasterRCNN
with warnings.catch_warnings():
    warnings.simplefilter("ignore")
    from bot_vision.caffe.faster_rcnn_utils import FasterRCNNDescription, nms

from bot_vision.recognition_utils import BOWClassifier
 
def extract_hypercolums(net, im, layers): 
    hypercolumns = []
    for layer in layers:
        convmap = net.blobs[layer].data
        for fmap in convmap[0]:
            upscaled = sp.misc.imresize(fmap, size=(im.shape[0], im.shape[1]),
                                        mode="F", interp='bilinear')
            hypercolumns.append(upscaled)
    return np.asarray(hypercolumns)

        
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
        self.image_descriptor_ = FastRCNNDescription(**self.params_.descriptor)

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
        print 'HISTS', hists_dir
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
                    # target = np.array([bbox['target'] for bbox in frame.bbox], dtype=np.int32)
                    # bboxes = np.vstack([[bbox['left'], bbox['top'], bbox['right'], bbox['bottom']] for bbox in frame.bbox])
                    target, bboxes = self.process_cb_(frame)

                    im_desc = self.image_descriptor_.describe(frame.img, bboxes)
                    add_item_to_hists_db(hists_db, im_desc, target, bboxes)

                elif hasattr(frame, 'target'): 
                    # Single image, single target
                    sz = frame.img.shape[:2]
                    bboxes = np.array([[0,0,sz[1]-1, sz[0]-1]])
                    im_desc = self.image_descriptor_.describe(frame.img, bboxes)
                    add_item_to_hists_db(hists_db, im_desc, frame.target, bboxes)

                elif mode == 'neg': 
                    target, bboxes = self.process_cb_(frame)
                    
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

    def neg_train(self, bg_data_iterable, batch_size=10, viz_cb=lambda e: None): 

        # 5. Hard-neg mining
        for epoch in np.arange(self.params_.neg_epochs): 
            self.epoch_no_ = epoch
            print 'Processing epoch %i' % (epoch)

            if epoch == 0:
                # 1. Extract features
                self._extract(bg_data_iterable, mode='neg', batch_size=batch_size)

            # 2. Re-train
            self._train(mode='neg')

            # viz_cb(epoch)

    def evaluate(self, data_iterable, batch_size=10): 
        # 1. Extract D-SIFT features 
        self._extract(data_iterable, mode='test', batch_size=batch_size)

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
            self.image_descriptor_ = FastRCNNDescription(**db.params.descriptor)
            self.process_cb_ = None
            self.pca_ = db.pca
            self.kernel_tf_ = db.kernel_tf
            self.clf_base_, self.clf_, self.clf_hyparams_ = db.clf_base, db.clf, db.clf_hyparams
            self.clf_prob_ = db.clf_prob
        except KeyError: 
            raise RuntimeError('DB not setup correctly, try re-training!')
        
    def save(self, path): 
        db = AttrDict(params=self.params_, 
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

class ObjectClassifier(object): 
    """
    Usage: 

        prop = ObjectClassifier.create('fast-rcnn', params)
        probs, targets = prop.process(im, bboxes)
    """
    @staticmethod
    def create(method='fast-rcnn', params=None): 
        if method == 'bow': 
            return BOWClassifier(**params)
        elif method == 'fast-rcnn': 
            return HistogramClassifier(**params)
        else: 
            raise RuntimeError('Unknown classifier method: %s' % method)
