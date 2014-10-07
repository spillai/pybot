import cv2
import numpy as np
from scipy.cluster.vq import vq, kmeans2

class BOWVectorizer(object): 
    def __init__(self, K, method='kmeans'): 
        self.K = K
        self.method = method
        
    def build(self, descriptors): 
        self.codebook, self.labels = kmeans2(descriptors, self.K)
        return self.codebook, self.labels

    def vectorize(self, descriptors): 
        return vq(descriptors, self.codebook)

class BOWTrainer(object): 
    def __init__(self, K=200, method='kmeans'): 
        self.vectorizer = BOWVectorizer(K=K, method=method)
        self.descriptors = []

    def add(self, descriptors): 
        self.descriptors.append(descriptors)

    def cluster(self, descriptors): 
        assert(len(descriptors) > 0)
        return self.vectorizer.build(np.vstack(descriptors))

    def project(self, descriptors): 
        code, dist = self.vectorizer.vectorize(descriptors)
        word_hist, bin_edges = histogram(code, bins=range(codebook.shape[0] + 1), normed=True)
        return word_hist

    

def computeHistograms(codebook, descriptors):
    code, dist = vq.vq(descriptors, codebook)
    word_hist, bin_edges = histogram(code, bins=range(codebook.shape[0] + 1), normed=True)
    return word_hist
