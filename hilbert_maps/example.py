#!/usr/bin/env python
# -*- coding: utf-8 -*-

"""
Example of running hilbert maps on carmen logfiles.
"""

import argparse
import copy
import math
import matplotlib.pyplot as plt
import numpy as np
import sys

import hilbert_map as hm
import util


def train_sparse_hm(data, components, gamma, distance_cutoff):
    """Trains a hilbert map model using the sparse feature.

    :param data the dataset to train on
    :param components the number of components to use
    :param gamma the gamma value to use in the RBF kernel
    :param distance_cutoff the value below which values are set to 0
    :return hilbert map model trained on the given data
    """
    # Extract poses and scans from the data
    poses = data["poses"]
    scans = data["scans"]

    # Limits in metric space based on poses with a 10m buffer zone
    xlim, ylim = util.bounding_box(poses, 10.0)
    # Sampling locations distributed in an even grid over the area
    centers = util.sampling_coordinates(xlim, ylim, math.sqrt(components))

    model = hm.SparseHilbertMap(centers, gamma, distance_cutoff)

    # Train the model with the data
    count = 0
    for data, label in util.data_generator(poses, scans):
        model.add(data, label)

        sys.stdout.write("\rTraining model: {: 6.2f}%".format(count / float(len(poses)) * 100))
        sys.stdout.flush()
        count += 1
    print("")

    return model


def train_incremental_hm(data, components, gamma, feature):
    """Trains a hilbert map model using either the Nystroem or Fourier feature.

    :param data the dataset to train on
    :param components the number of components to use
    :param gamma the gamma value to use in the RBF kernel
    :param feature the type of feature to use
    :return hilbert map model trained on the given data
    """
    # Extract poses and scans from the data
    poses = data["poses"]
    scans = data["scans"]

    model = hm.IncrementalHilbertMap(feature, components, gamma, False)

    # Fit the feature to the dataset
    training_data = []
    for data, label in util.data_generator(poses, scans):
        training_data.extend(data)
    model.fit(np.array(training_data))

    # Train the model
    count = 0
    for data, label in util.data_generator(poses, scans):
        model.add(data, label)

        sys.stdout.write("\rTraining model: {: 6.2f}%".format(count / float(len(poses)) * 100))
        sys.stdout.flush()
        count += 1
    print("")

    return model


def generate_map(model, resolution, limits, fname, verbose=True):
    """Generates a grid map by querying the model at cell locations.

    :param model the hilbert map model to use
    :param resolution the resolution of the produced grid map
    :param limits the limits of the grid map
    :param fname the name of the file in which to store the final grid map
    :param verbose print progress if True
    """
    # Determine query point locations
    x_count = int(math.ceil((limits[1] - limits[0]) / resolution))
    y_count = int(math.ceil((limits[3] - limits[2]) / resolution))
    sample_coords = []
    for x in range(x_count):
        for y in range(y_count):
            sample_coords.append((limits[0] + x*resolution, limits[2] + y*resolution))

    # Obtain predictions in a batch fashion
    predictions = []
    offset = 0
    batch_size = 100
    old_intercept = copy.deepcopy(model.classifier.intercept_)
    model.classifier.intercept_ = 0.1 * model.classifier.intercept_
    while offset < len(sample_coords):
        if isinstance(model, hm.IncrementalHilbertMap):
            query = model.sampler.transform(sample_coords[offset:offset+batch_size])
            predictions.extend(model.classifier.predict_proba(query)[:, 1])
        elif isinstance(model, hm.SparseHilbertMap):
            predictions.extend(model.classify(sample_coords[offset:offset+batch_size])[:, 1])

        if verbose:
            sys.stdout.write("\rQuerying model: {: 6.2f}%".format(offset / float(len(sample_coords)) * 100))
            sys.stdout.flush()
        offset += batch_size
    if verbose:
        print("")
    predictions = np.array(predictions)
    model.classifier.intercept_ = old_intercept

    # Turn predictions into a matrix for visualization
    mat = predictions.reshape(x_count, y_count)
    plt.clf()
    plt.title("Occupancy map")
    plt.imshow(mat.transpose()[::-1, :])
    plt.colorbar()
    plt.savefig(fname)


def main():
    # Setup command line argument parsing
    parser = argparse.ArgumentParser(description="Hilbert maps example")
    parser.add_argument(
            "logfile",
            help="Logfile in CARMEN format to process"
    )
    parser.add_argument(
            "feature",
            choices=["sparse", "fourier", "nystroem"],
            help="The feature to use"
    )
    parser.add_argument(
            "--components",
            default=1000,
            type=int,
            help="Number of components used with the feature"
    )
    parser.add_argument(
            "--gamma",
            default=1,
            type=float,
            help="Gamma value used in the RBF kernel"
    )
    parser.add_argument(
            "--distance_cutoff",
            default=0.001,
            type=float,
            help="Value below which a kernel value will be set to 0"
    )
    parser.add_argument(
            "--resolution",
            default=0.1,
            type=float,
            help="Grid cell resolution of the map"
    )

    args = parser.parse_args()


    # Load data and split it into training and testing data
    train_data, test_data = util.create_test_train_split(args.logfile, 0.1)

    # Train the desired model on the data
    if args.feature == "sparse":
        model = train_sparse_hm(train_data, args.components, args.gamma, args.distance_cutoff)
    else:
        model = train_incremental_hm(train_data, args.components, args.gamma, args.feature)

    # Evaluate the map on the hold out set
    tpr, fpr, auc = util.roc_evaluation(model, test_data)
    print("Area under curve: {:.2f}".format(auc))

    # Produce a grid map based on the trained model
    xlim, ylim = util.bounding_box(train_data["poses"], 10.0)
    generate_map(
            model,
            args.resolution,
            [xlim[0], xlim[1], ylim[0], ylim[1]],
            "hilbert_map.png"
    )

    return 0


if __name__ == "__main__":
    sys.exit(main())
