import cv2, time
import numpy as np

from scipy.cluster.vq import vq, kmeans2
from scipy.spatial import cKDTree

from sklearn.cluster import KMeans, MiniBatchKMeans

from bot_utils.db_utils import AttrDict

class BoWVectorizer(object): 
    default_params = AttrDict(K=64, method='vlad', norm_method='square-rooting')
    def __init__(self, K=64, method='vlad', quantizer='kdtree', norm_method='square-rooting'): 
        self.K = K
        self.method, self.quantizer = method, quantizer
        self.norm_method = norm_method

    def _build(self, data): 
        """
        Build [K x D] codebook/vocabulary from data
        """
        st = time.time()

        # Scipy: 1x
        # self.codebook, self.labels = kmeans2(data, self.K)

        # Scikit-learn: 2x
        km = MiniBatchKMeans(n_clusters=self.K, init='k-means++', 
                             compute_labels=False, batch_size=1000, max_iter=150, max_no_improvement=30, 
                             verbose=False).fit(data)
        # Alternate
        # km = KMeans(n_clusters=self.K, n_jobs=4, tol=0.01, verbose=True).fit(data)
        self.codebook = km.cluster_centers_

        # # Opencv: 1x
        # term_crit = (cv2.TERM_CRITERIA_EPS | cv2.TERM_CRITERIA_MAX_ITER, 1, 10)
        # ret, labels, self.codebook = cv2.kmeans(data, self.K, criteria=term_crit, 
        #                                         attempts=10, flags=cv2.KMEANS_PP_CENTERS)

        print 'Vocab construction from data %s (%s KB, %s) => codebook %s took %5.3f s' % \
            (data.shape, data.nbytes / 1024, data.dtype, self.codebook.shape, time.time() - st)

        # Save codebook, and index
        self.index_codebook()

    def build(self, data): 
        """
        Build a codebook/vocabulary from data
        """
        assert(len(data) > 0)
        self._build(np.vstack(data))

    def index_codebook(self): 
        # Index codebook for quick querying
        st = time.time()
        self.index = cKDTree(self.codebook)
        print 'Indexing codebook %s took %5.3f s' % (self.codebook.shape, time.time() - st)


    @classmethod
    def from_dict(cls, db): 
        bowv = cls(**db.params)
        bowv.codebook = db.codebook
        bowv.index_codebook()
        return bowv

    @classmethod
    def load(cls, path):
        db = AttrDict.load(path)
        return cls.from_dict(db)

    def to_dict(self): 
        return AttrDict(codebook=self.codebook, params=AttrDict(K=self.K, method=self.method, norm_method=self.norm_method))

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
            code_hist = self.bow(data, code)
        elif self.method == 'vlad': 
            code = self.get_code(data)
            code_hist = self.vlad(data, code)
        else: 
            raise NotImplementedError('''Histogram method %s not implemented. '''
                                      '''Use vq/bow or vlad!''' % self.method)            
        return code_hist

    def project(self, data): 
        """
        Project the descriptions on to the codebook/vocabulary, 
        returning the histogram of words
        [N x 1] => [1 x K] histogram
        """
        return self.get_histogram(data)

    @staticmethod
    def normalize(hist, norm_method='global-l2'): 
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

    def bow(self, data, code): 
        """
        BoW histogram with L2 normalization
        """
        code_hist, bin_edges = np.histogram(code, bins=np.arange(self.K+1) - 0.5)
        
        # Normalize
        code_hist = BoWVectorizer.normalize(code_hist, norm_method='global-l2')

        # Vectorize [1 x K]
        return code_hist.ravel()
        

    def vlad(self, data, code): 
        """
        Aggregating local descriptors into a compact image representation
        Herve Jegou, Matthijs Douze, Cordelia Schmid and Patrick Perez
        Proc. IEEE CVPR 10, June, 2010.
        """
        residuals = np.zeros(self.codebook.shape)

        # Accumulate residuals [K x D]
        for cidx, c in enumerate(code):
            residuals[c] += data[cidx] - self.codebook[c]
       
        # Normalize
        residuals = BoWVectorizer.normalize(residuals, norm_method=self.norm_method)
            
        # Vectorize [1 x (KD)]
        return residuals.ravel()

    @property
    def dictionary_size(self): 
        return self.K

    @property
    def dimension_size(self): 
        return self.codebook.shape[1]

