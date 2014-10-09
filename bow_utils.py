import cv2, time
import numpy as np
from scipy.cluster.vq import vq, kmeans2

from scipy.spatial import cKDTree
from sklearn.cluster import KMeans, MiniBatchKMeans

class BOWVectorizer(object): 
    def __init__(self, K=100, method='vq', quantizer='kdtree', norm_method='square-rooting'): 
        self.K = K
        self.method, self.quantizer = method, quantizer
        self.norm_method = norm_method

    def build(self, data): 
        """
        Build [K x D] codebook/vocabulary from data
        """
        st = time.time()

        # Scipy: 1x
        # self.codebook, self.labels = kmeans2(data, self.K)

        # Scikit-learn: 2x
        km = MiniBatchKMeans(n_clusters=self.K, init='k-means++', 
                             compute_labels=False, batch_size=1000, max_iter=150, verbose=True).fit(data)
        # km = KMeans(n_clusters=self.K, n_jobs=4, tol=0.01, verbose=True).fit(data)
        self.codebook = km.cluster_centers_

        # # Opencv: 1x
        # ret, labels, self.codebook = cv2.kmeans(data, self.K, 
        #                                         criteria=(cv2.TERM_CRITERIA_EPS | cv2.TERM_CRITERIA_MAX_ITER, 1, 10), 
        #                                         attempts=10, flags=cv2.KMEANS_PP_CENTERS)
        print 'Vocab construction from data %s => codebook %s took %5.3f s' % (data.shape, self.codebook.shape, 
                                                                               time.time() - st)

        # Index codebook for quick querying
        st = time.time()
        self.index = cKDTree(self.codebook)
        print 'Indexing codebook %s took %5.3f s' % (self.codebook.shape, time.time() - st)

        return self.codebook

    def vectorize(self, data): 
        """
        Transform the [N x D] data to [N x 1] where n_i \in {1, ... , K}
        returns the cluster indices
        """
        if self.method == 'vq': 
            if self.quantizer == 'vq': 
                code, dist = vq(data, self.codebook)
            elif self.quantizer == 'kdtree': 
                dist, code = self.index.query(data, k=1)
            else: 
                raise NotImplementedError('Quantizer %s not implemented. Use vq or kdtree!' % self.quantizer)
        elif self.method == 'vlad': 
            code = self.vlad(data)
        else: 
            raise NotImplementedError('Codebook generation method %s not implemented. Use vq or vlad!' % self.method)
        return code

    def vlad(self, data): 
        """
        Aggregating local descriptors into a compact image representation
        Herve Jegou, Matthijs Douze, Cordelia Schmid and Patrick Perez
        Proc. IEEE CVPR 10, June, 2010.
        """
        residuals = np.zeros(self.codebook.shape)
        if self.quantizer == 'kdtree': 
            dist, code = self.index.query(data, k=1)
        elif self.quantizer == 'vq': 
            code, dist = vq(data, self.codebook)
        else: 
            raise NotImplementedError('Quantizer %s not implemented. Use vq or kdtree!' % quantizer)

        # Accumulate residuals [K x D]
        for cidx, c in enumerate(code):
            residuals[c] += data[cidx] - self.codebook[c]
       
        # Normalize
        # Component-wise mass normalization 
        if self.norm_method == 'component-wise-mass': 
            raise NotImplementedError('VLAD normalization_method not implemented')

        # Component-wise L2 normalization
        elif self.norm_method == 'component-wise-l2': 
            residuals /= np.max(np.linalg.norm(residuals, axis=1), 1e-12)

        # Global L2 normalization
        elif self.norm_method == 'global-l2': 
            residuals /= (np.linalg.norm(residuals) + 1e-12)

        # Square rooting
        elif self.norm_method == 'square-rooting': 
            residuals = np.sign(residuals) * np.sqrt(np.abs(residuals))

        else: 
            import warnings
            raise warnings.warn('VLAD un-normalized')
            # raise NotImplementedError('VLAD normalization_method not implemented')
            
        # Vectorize [1 x (KD)]
        return residuals.ravel()

class BOWTrainer(object): 
    def __init__(self, **kwargs): 
        self.vectorizer = BOWVectorizer(**kwargs)
        # self.data = []

    # def add(self, data): 
    #     """
    #     Accumuate the descriptions for codebook/vocabulary construction
    #     """
    #     self.data.append(data)


    # def build(self): 
    #     """
    #     Build a codebook/vocabulary from data
    #     """
    #     assert(len(self.data) > 0)
    #     return self.vectorizer.build(np.vstack(self.data))

    def load(self, path): 
        pass

    def save(self, path): 
        pass


    def build(self, data): 
        """
        Build a codebook/vocabulary from data
        """
        assert(len(data) > 0)
        return self.vectorizer.build(np.vstack(data))

    def get_words(self, data): 
        """
        Project the descriptions on to the codebook/vocabulary, 
        returning the all the words in the description
        [N x D] => [1 x N] where n_i \in {1, ... , K}
        """
        return self.vectorizer.vectorize(data)
        
    def project(self, data): 
        """
        Project the descriptions on to the codebook/vocabulary, 
        returning the histogram of words
        [N x 1] => [1 x K] histogram
        """
        word_hist, bin_edges = np.histogram(self.get_words(data), 
                                            bins=np.arange(self.vectorizer.K), normed=True)
        return word_hist
