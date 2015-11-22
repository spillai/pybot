#!/usr/bin/env python
# -*- coding: utf-8 -*-

"""Implementation of Fourier, Nystroem, and sparse feature based Hilbert maps."""

import numpy as np
import random

from scipy import sparse

from sklearn.cluster import MiniBatchKMeans
from sklearn.kernel_approximation import RBFSampler, Nystroem
from sklearn.linear_model import SGDClassifier
from sklearn.metrics.pairwise import rbf_kernel
from sklearn.preprocessing import MinMaxScaler


class UsageError(BaseException):

    """Exception raised when the code is used incorrectly."""

    def __init__(self, msg):
        """Creates a new UsageError instance.

        :param msg the error message
        """
        BaseException.__init__(self)
        self.msg = msg

    def __str__(self):
        return "Usage error: {}".format(self.msg)


class IncrementalHilbertMap(object):

    """Incremental version of Hilbert maps using stochastic gradient descent."""

    def __init__(self, method, components, gamma=3.0, use_rkhs=False):
        """Creates a new IncrementalHilbertMap instance.

        :param method the random feature transformation method to use
        :param parameters dictionary of parameters used by the random feature
            transformation method
        :param components the number of random components to use
        """
        if method == "fourier":
            self.sampler = RBFSampler(
                    gamma=gamma,
                    n_components=components
            )
        elif method == "nystroem":
            self.sampler = Nystroem(
                    kernel="rbf",
                    gamma=gamma,
                    n_components=components
            )
        else:
            raise UsageError("Invalid method provided: {}".format(method))

        self.components = components
        self.feature_method = method
        self.scaler = MinMaxScaler()
        self.use_rkhs = use_rkhs

        # SGD version of logistic regression with elasticnet penalty
        self.classifier = SGDClassifier(
                loss="log",
                penalty="elasticnet",
                alpha=0.0001
        )

    def fit(self, data):
        """Uses the provided data to fit the transoformation matrix.

        :param data the data to use in the fitting of the random feature
            transform
        """
        # Scale the data such that they all lie within [0, 1]
        if self.feature_method == "nystroem":
            # Select good inducing points for the Nystroem feature
            mbkm = MiniBatchKMeans(
                    n_clusters=self.components,
                    compute_labels=False,
                    batch_size=1000,
                    init_size=10000
            )
            mbkm.fit(data)
            self.sampler.fit(mbkm.cluster_centers_)
            self.scaler.fit(self.sampler.transform(mbkm.cluster_centers_))
        else:
            self.sampler.fit(data)

            # Build training dataset for the scalar
            indices = random.sample(range(len(data)), min(len(data), 10000))
            scaler_data = [data[i] for i in indices]
            self.scaler.fit(self.sampler.transform(scaler_data))

    def add(self, data, labels):
        """Updates the classifier with new data.

        :param data the raw data points to add
        :param labels the labels associated with the data
        """
        if self.use_rkhs:
            avg_features = []
            for entry in data:
                # Generate 10 samples with fixed variation from the model
                sample_coords = np.random.multivariate_normal(
                        entry,
                        [(0.2, 0.0), (0.0, 0.2)],
                        10
                )
                features = self.sampler.transform(sample_coords)
                avg_features.append(features.mean(axis=0))
                assert(len(avg_features[-1]) == self.components)
            rbf_data = avg_features
        else:
            rbf_data = self.sampler.transform(data)

        # Scale data and update the classifier
        scaled_data = self.scaler.transform(rbf_data)
        self.classifier.partial_fit(scaled_data, labels, classes=[0, 1])


class SparseHilbertMap(object):

    """Hilbert map using the sparse feature."""

    def __init__(self, centers, gamma=1.0, cutoff=0.001, use_rkhs=False):
        """Creates a new sparse hilbert map.

        :param centers the locations at which to create sparse kernels
        :param gamma the gamma parameter of the RBF kernel
        :param cutoff value below which kernel values are set to 0
        :param use_rkhs whether or not to use reproducing kernel hilbert spaces
        """
        self.centers = centers
        self.gamma = gamma
        self.cutoff = cutoff
        self.use_rkhs = use_rkhs
        self.batch_size = 10000
        self.classifier = SGDClassifier(
            loss="log",
            penalty="elasticnet",
            alpha=0.0001,
            class_weight="auto",
            l1_ratio=0.80,
        )

    def add(self, data, labels):
        """Updates the classifier with new data.

        :param data the raw data points to add
        :param labels the labels associated with the data
        """
        if self.use_rkhs:
            process_data = self.rkhs_data(data)
            if process_data.shape[0] < 100:
                self.classifier.fit(process_data, labels)
            else:
                offset = 0
                while offset < process_data.shape[0]:
                    self.classifier.partial_fit(
                            process_data[offset:offset+self.batch_size],
                            labels[offset:offset+self.batch_size],
                            classes=[0, 1]
                    )
                    offset += self.batch_size
        else:
            if len(data) < 100:
                kernel = rbf_kernel(data, self.centers, self.gamma)
                kernel = sparse.csr_matrix(kernel * (kernel > self.cutoff))
                self.classifier.fit(kernel, labels)
            else:
                offset = 0
                while offset < len(data):
                    # Compute kernel matrix and sparsify it
                    kernel = rbf_kernel(data[offset:offset+self.batch_size], self.centers, self.gamma)
                    kernel = sparse.csr_matrix(kernel * (kernel > self.cutoff))

                    # Update the classifier using the kernel matrix
                    self.classifier.partial_fit(
                            kernel,
                            labels[offset:offset+self.batch_size],
                            classes=[0, 1]
                    )
                    offset += self.batch_size

    def rkhs_data(self, data):
        """Return features obtained using the RKHS for the given data.
        
        :param data the data entries for which to create RKHS content
        :return RKHS based features for the given input data
        """
        avg_features = None
        # Compute average feauters for each input point
        for entry in data:
            mean_map_sample_coords = np.random.multivariate_normal(
                    entry,
                    [(0.2, 0.0), (0.0, 0.2)],
                    10
            )
            kernel = rbf_kernel(mean_map_sample_coords, self.centers, self.gamma)
            kernel = kernel.mean(axis=0)
            kernel = sparse.csr_matrix(kernel * (kernel > self.cutoff))

            if avg_features is None:
                avg_features = kernel
            else:
                avg_features = sparse.vstack((avg_features, kernel))
        return avg_features

    def classify(self, query):
        """Returns the probabilistic prediction for the given data points.

        :param query the data points to perform predictions on
        :return predicted values for the given inputs
        """
        if self.use_rkhs:
            kernel = self.rkhs_data(query)
        else:
            kernel = rbf_kernel(query, self.centers, self.gamma)
            kernel = sparse.csr_matrix(kernel * (kernel > self.cutoff))
        return self.classifier.predict_proba(kernel)
