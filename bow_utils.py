import cv2
import numpy as np
from scipy.cluster.vq import vq, kmeans2

class BOWVectorizer(object): 
    def __init__(self, K=100, method='vq', norm_method='square-rooting'): 
        self.K = K
        self.method = method
        self.norm_method = norm_method

    def build(self, data): 
        """
        Build [K x D] codebook/vocabulary from data
        """
        self.codebook, self.labels = kmeans2(data, self.K)
        return self.codebook, self.labels

    def vectorize(self, data): 
        """
        Transform the [N x D] data to [N x 1] where n_i \in {1, ... , K}
        returns the cluster indices
        """

        if self.method == 'vq': 
            code, dist = vq(data, self.codebook)
            return code
        elif self.method == 'vlad': 
            return self.vlad(data)
        else: 
            raise NotImplementedError('Unknown method')

    def vlad(self, data): 
        """
        Aggregating local descriptors into a compact image representation
        Herve Jegou, Matthijs Douze, Cordelia Schmid and Patrick Perez
        Proc. IEEE CVPR 10, June, 2010.
        """
        residuals = np.zeros(self.codebook.shape)
        codes, dist = vq(data, self.codebook)

        # Accumulate residuals [K x D]
        for cidx, c in enumerate(codes):
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
