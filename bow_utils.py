import cv2, time
import numpy as np

from scipy.cluster.vq import vq, kmeans2
from scipy.spatial import cKDTree

from sklearn.cluster import KMeans, MiniBatchKMeans
from sklearn.mixture import GMM

from bot_vision.color_utils import get_random_colors
from bot_utils.db_utils import AttrDict

from pybot_vision import flair_code

# =====================================================================
# Generic utility functions for bag-of-visual-words computation
# ---------------------------------------------------------------------

def normalize_hist(hist, norm_method='global-l2'): 
    """
    Various normalization methods

    Refer to: 
    [1] Improving the Fisher Kernel for Large-Scale Image Classifcation, Perronnin et al
    http://www.robots.ox.ac.uk/~vgg/rg/papers/peronnin_etal_ECCV10.pdf

    [2] Segmentation Driven Object Detection with Fisher Vectors, Cinbis et al

    """

    # Component-wise mass normalization 
    if norm_method == 'component-wise-mass': 
        raise NotImplementedError('Component-wise-mass normalization_method not implemented')

    # Component-wise L2 normalization
    elif norm_method == 'component-wise-l2': 
        return hist / np.max(np.linalg.norm(hist, axis=1), 1e-12)

    # Global L2 normalization
    elif norm_method == 'global-l2': 
        return hist / (np.linalg.norm(hist) + 1e-12)

    # Square rooting / Power Normalization with alpha = 0.5
    elif norm_method == 'square-rooting': 
        # Power-normalization followed by L2 normalization as in [2]
        hist = np.sign(hist) * np.sqrt(np.fabs(hist))
        return hist / (np.linalg.norm(hist) + 1e-12)

    else: 
        raise NotImplementedError('Unknown normalization_method %s' % norm_method)            


def bow(data, code, K): 
    """
    BoW histogram with L2 normalization
    """
    code_hist, bin_edges = np.histogram(code, bins=np.arange(K+1) - 0.5)

    # Normalize
    code_hist = normalize_hist(code_hist.astype(np.float32), norm_method='global-l2')

    # Vectorize [1 x K]
    return code_hist.ravel()

def bow_histogram(data, codebook, pts=None, shape=None): 
    code, dist = vq(data, codebook)
    code_hist = bow(data, code, codebook.shape[0])
    return code_hist

def bow_project(data, codebook, pts=None, shape=None, levels=(1,2,4)): 
    """
    Project the descriptions on to the codebook/vocabulary, 
    returning the histogram of words
    [N x 1] => [1 x K] histogram

    Otherwise, if kpts and bbox shape specified, perform spatial pooling
    """

    if pts is None or shape is None: 
        return bow_histogram(data, codebook)
    else: 
        # Compute histogram for each spatial level
        # assert(pts.dtype == np.int32)
        pts = pts.astype(np.int32)
        xmin, ymin = shape[0], shape[1]
        xs, ys = pts[:,0]-xmin, pts[:,1]-ymin

        # Ownership bin 
        # For each pt, and each level find the x and y bin
        # and assign to appropriate bin
        # nbins = np.sum(np.array(self.levels) ** 2)

        # levels = [1, 2, 4]
        assert(len(pts) == len(data))
        hist = []
        for j,level in enumerate(levels): 
            # Determine the bin each point belongs to given level, and assign
            xdim, ydim = int(np.floor((shape[2]-shape[0]) * 1. / level + 1)), int(np.floor((shape[3]-shape[1]) * 1. / level + 1))
            xbin, ybin = xs / xdim, ys / ydim
            bin_idx = ybin * level + xbin

            # Compute histogram for each bin                
            for lbin in range(level * level): 
                inds, = np.where(bin_idx == lbin)
                hist.append(bow_histogram(data[inds], codebook, pts=pts[inds], shape=shape))

        # Stack all histograms together
        return np.hstack(hist)

def bow_codebook(data, K=64): 
    km = MiniBatchKMeans(n_clusters=K, init='k-means++', 
                         compute_labels=False, batch_size=1000, max_iter=150, max_no_improvement=30, 
                         verbose=False).fit(data)
    return km.cluster_centers_

def flair_project(data, codebook, pts=None, shape=None, method='bow', levels=(1,2,4), step=4): 
    W, H = np.max(shape[:, -2:], axis=0)

    return flair_code(descriptors=data.astype(np.float32), pts=pts.astype(np.int32), 
                      rects=shape.astype(np.float32), codebook=codebook.astype(np.float32), 
                      W=int(W+5), H=int(H+5), K=codebook.shape[0], 
                      step=step, levels=np.array(list(levels), dtype=np.int32), encoding={'bow':0, 'vlad':1, 'fisher':2}[method])

# =====================================================================
# General-purpose bag-of-words interfaces
# ---------------------------------------------------------------------

class VocabBuilder(object): 
    def __init__(self, D, K=300, N=100000):
        self.D_ = D
        self.K_ = K
        self.built_ = False
        print('Initializing vocabulary builder K={:}, D={:}'.format(K,D))

        # Binary Vocab builder 
        # self.voc_ = BinaryVocabulary()
        self.vocab_ = None

        # Vocab Training
        self.N_ = N
        if N: 
            self.vocab_len_ = 0
            self.vocab_data_ = np.empty((N, D), dtype=np.uint8)
        
    def add(self, desc):
        if self.built_: 
            return
 
        if self.vocab_len_ < self.N_:
            Nd = len(desc)
            st, end = self.vocab_len_, min(self.vocab_len_ + Nd, self.N_)
            self.vocab_data_[st:end] = desc[:end-st]
            self.vocab_len_ += len(desc)
            print('Vocabulary building: {:}/{:}'.format(self.vocab_len_, self.N_))
        else: 
            print('Vocabulary built')
            self.built_ = True

        # else: 
        #     # Build vocab if not built already
        #     self.voc_.build(self.vocab_data_, self.K_)
        #     self.vocab_ = self.voc_.getCentroids()
            
        #     sz = self.vocab_.shape[:2]
        #     if sz[0] != self.K_ or sz[1] != self.D_: 
        #         raise RuntimeError('Voc error! KxD={:}x{:}, expected'.format(sz[0],sz[1],self.K_,self.D_))

        #     self.save('vocab.yaml.gz')

    @property
    def vocab_data(self): 
        return self.vocab_data_
            
    # def project(self, desc): 
    #     return self.voc_.getClusterAssignments()

    # @classmethod
    # def load(cls, fn): 

    #     voc = ORBVocabulary()
    #     voc.load(fn)
    #     vocab = self.voc.getCentroids()

    #     K, D = vocab.shape[:2]
    #     vocab_builder = cls(D, K=K, N=0)
    #     vocab_builder.voc_ = voc
    #     vocab_builder.vocab_ = vocab
    #     vocab_builder.built_ = True
        
    #     return vocab_builder

    # def save(self, fn): 
    #     fn = os.path.expanduser(fn)
    #     print('Saving vocabulary to {:}'.format(fn))
    #     self.voc_.save(fn)
    #     self.built_ = True

    @property
    def built(self): 
        return self.built_


class BoWVectorizer(object): 
    default_params = AttrDict(K=64, levels=(1,2,4), 
                              method='vlad', quantizer='kdtree', norm_method='square-rooting')
    def __init__(self, K=64, levels=(1,2,4), 
                 method='vlad', quantizer='kdtree', norm_method='square-rooting'): 
        self.K = K
        self.levels = levels
        self.method, self.quantizer = method, quantizer
        self.norm_method = norm_method
        self.codebook = None

    def _build_codebook(self, data): 
        """
        Build [K x D] codebook/vocabulary from data
        """
        st = time.time()
        self.codebook = bow_codebook(data, K=self.K)
        print 'Vocab construction from data %s (%s KB, %s) => codebook %s took %5.3f s' % \
            (data.shape, data.nbytes / 1024, data.dtype, self.codebook.shape, time.time() - st)
        print 'Codebook: %s' % ('GOOD' if np.isfinite(self.codebook).all() else 'BAD')

        # Save codebook, and index
        self.index_codebook()

    def _build_gmm(self, data): 
        """
        Build gmm from data
        """
        st = time.time()

        self.gmm = GMM(n_components=self.K, covariance_type='diag')
        self.gmm.fit(data)

        # Setup codebook for closest center lookup
        self.codebook = self.gmm.means_

        print 'Vocab construction from data %s (%s KB, %s) => GMM %s took %5.3f s' % \
            (data.shape, data.nbytes / 1024, data.dtype, self.gmm.means_.shape, time.time() - st)
        print 'GMM: %s' % ('GOOD' if np.isfinite(self.gmm.means_).all() else 'BAD')

        # Save codebook, and index
        self.index_codebook()

    def ready(self): 
        return self.codebook is not None

    def build(self, data): 
        """
        Build a codebook/vocabulary from data
        """
        assert(len(data) > 0)
        if self.method == 'fisher': 
            self._build_gmm(np.vstack(data))
        else: 
            # self._build_codebook(np.vstack(data))
            self._build_codebook(data)

    def build_incremental(self, data, N=100000): 
        if not hasattr(self, 'vbuilder__'): 
            self.vbuilder__ = VocabBuilder(data.shape[1], K=-1, N=N)

        if not self.vbuilder__.built: 
            self.vbuilder__.add(data)
        else: 
            train_data = self.vbuilder__.vocab_data
            self.build(train_data)
        
    @staticmethod
    def compute_index(codebook): 
        return cKDTree(codebook)

    def index_codebook(self): 
        # Index codebook for quick querying
        st = time.time()
        self.index = BoWVectorizer.compute_index(self.codebook)
        print 'Indexing codebook %s took %5.3f s' % (self.codebook.shape, time.time() - st)

    @classmethod
    def from_dict(cls, db, index=None): 
        bowv = cls(**db.params)
        bowv.codebook = db.codebook
        if index is None: 
            bowv.index_codebook()
        else: 
            bowv.index = index
            print 'BoVW using supplied index', index
        return bowv

    @classmethod
    def load(cls, path):
        db = AttrDict.load(path)
        return cls.from_dict(db)

    def to_dict(self): 
        return AttrDict(codebook=self.codebook, params=AttrDict(K=self.K, levels=self.levels, method=self.method, norm_method=self.norm_method))

    def save(self, path): 
        db = self.to_dict()
        db.save(path)

    def get_code(self, data): 
        """
        Transform the [N x D] data to [N x 1] where n_i \in {1, ... , K}
        returns the cluster indices
        """
        if self.quantizer == 'vq': 
            code, dist = vq(data, self.codebook)
        elif self.quantizer == 'kdtree': 
            dist, code = self.index.query(data, k=1)
        else: 
            raise NotImplementedError('Quantizer %s not implemented. Use vq or kdtree!' % self.quantizer)

        return code

    def get_histogram(self, data): 
        """
        Project the descriptions on to the codebook/vocabulary, 
        returning the histogram of words
        [N x 1] => [1 x K] histogram
        """
        if self.method == 'vq' or self.method == 'bow': 
            code = self.get_code(data)
            code_hist = self.bow(data, code, self.K)
        elif self.method == 'vlad': 
            code = self.get_code(data)
            code_hist = self.vlad(data, code)
        elif self.method == 'fisher': 
            code = self.get_code(data)
            code_hist = self.fisher(data, code)
        else: 
            raise NotImplementedError('''Histogram method %s not implemented. '''
                                      '''Use vq/bow or vlad or fisher!''' % self.method)            
        return code_hist

    def visualize(self, img, data, pts, level=0, code=None): 
        """
        Visualize the quantized words onto the image. 
        
        level: visualize the spatial pooling level
        """
        if not hasattr(self, 'visualize_cols__'): 
            self.visualize_cols__ = (get_random_colors(self.K) * 255).astype(np.int32)[:,:3]

        code_ids = self.get_code(data)
        for (label, pt) in zip(code_ids, pts):
            if code is None or label == code: 
                col = tuple(map(int, self.visualize_cols__[label % self.K].ravel()))
                cv2.circle(img, (pt[0], pt[1]), 2, col, -1)

            
    def project(self, data, pts=None, shape=None): 
        """
        Project the descriptions on to the codebook/vocabulary, 
        returning the histogram of words
        [N x 1] => [1 x K] histogram

        Otherwise, if kpts and bbox shape specified, perform spatial pooling
        """
        
        if pts is None: #  or shape is None: 
            return self.get_histogram(data)

        if shape is None: 
            shape = (np.min(pts[:,0]), np.min(pts[:,1]), np.max(pts[:,0]), np.max(pts[:,1]))

        # Compute histogram for each spatial level
        # assert(pts.dtype == np.int32)
        pts = pts.astype(np.int32)
        xmin, ymin = shape[0], shape[1]
        xs, ys = pts[:,0]-xmin, pts[:,1]-ymin

        # Ownership bin 
        # For each pt, and each level find the x and y bin
        # and assign to appropriate bin
        # nbins = np.sum(np.array(self.levels) ** 2)

        # levels = [1, 2, 4]
        hist = []
        for j,level in enumerate(self.levels): 
            # Determine the bin each point belongs to given level, and assign
            xdim, ydim = (shape[2]-shape[0]+1) / level, (shape[3]-shape[1]+1) / level
            xbin, ybin = xs / xdim, ys / ydim
            bin_idx = ybin * level + xbin

            # Compute histogram for each bin                
            for lbin in range(level * level): 
                inds, = np.where(bin_idx == lbin)
                hist.append(self.get_histogram(data[inds]))

        # Stack all histograms together
        return np.hstack(hist)
                    
                
            
    @staticmethod
    def normalize(hist, norm_method='global-l2'): 
        return normalize_hist(hist, norm_method=norm_method)

    @staticmethod
    def bow(data, code, K): 
        return bow(data, code, K)

    def vlad(self, data, code): 
        """
        Aggregating local descriptors into a compact image representation
        Herve Jegou, Matthijs Douze, Cordelia Schmid and Patrick Perez
        Proc. IEEE CVPR 10, June, 2010.
        """
        residuals = np.zeros(self.codebook.shape, dtype=np.float32)

        # Accumulate residuals [K x D]
        for cidx, c in enumerate(code):
            residuals[c] += data[cidx] - self.codebook[c]
       
        # Normalize [ Component-wise L2 / SSR followed by L2 normalization]
        residuals = BoWVectorizer.normalize(residuals, norm_method=self.norm_method)
        residuals = BoWVectorizer.normalize(residuals, norm_method='global-l2')
            
        # Vectorize [1 x (KD)]
        return residuals.ravel()

    def fisher(self, data, code): 
        """
        [1] Fisher kenrels on visual vocabularies for image categorizaton. 
        F. Perronnin and C. Dance. In Proc. CVPR, 2006.
        [2] Improving the fisher kernel for large-scale image classification. 
        Florent Perronnin, Jorge Sanchez, and Thomas Mensink. In Proc. ECCV, 2010.
        """

        # Fisher vector encoding
        K, D = self.gmm.means_.shape[:2]
        residuals_v = np.zeros(shape=(K,D), dtype=np.float32)
        residuals_u = np.zeros(shape=(K,D), dtype=np.float32)

        # Posterior prob. of data under each mixture [N x K]
        posteriors = self.gmm.predict_proba(data)
        
        # Inverse sqrt of covariance [K x K] 
        sigma_inv = 1.0 / (np.sqrt(self.gmm.covars_) + 1e-12)

        # Accumulate residuals [K x D]
        for cidx, c in enumerate(code):
            residuals_v[c] += posteriors[cidx,c] * (data[cidx] - self.codebook[c]) * sigma_inv[c]
            residuals_u[c] += posteriors[cidx,c] * np.square((data[cidx] - self.codebook[c]) * sigma_inv[c] - 1)

        for c in range(len(residuals_v)):
            residuals_v[c] *= 1.0 / (len(data) * np.sqrt(self.gmm.weights_[c]) + 1e-12)
            residuals_u[c] *= 1.0 / (len(data) * np.sqrt(2 * self.gmm.weights_[c]) + 1e-12)

        # Normalize
        residuals = BoWVectorizer.normalize(np.vstack([residuals_v, residuals_u]), norm_method=self.norm_method)

        # Vectorize [1 x (2KD)]
        return residuals.ravel()

    @property
    def dictionary_size(self): 
        return self.K

    @property
    def dimension_size(self): 
        return self.codebook.shape[1]

