# from __future__ import absolute_import
# from __future__ import division
# from __future__ import print_function

import os
import argparse
import cv2
import numpy as np
np.random.seed(0)

os.environ['KERAS_BACKEND'] = 'theano'
# os.environ['KERAS_BACKEND'] = 'tensorflow'
import keras.backend as K
import keras.models as models

from keras.layers.noise import GaussianNoise
from keras.layers.core import Layer, Dense, Dropout, Activation, Flatten, Reshape, Merge, Permute
from keras.layers.convolutional import Convolution2D, MaxPooling2D, UpSampling2D, ZeroPadding2D
from keras.layers.normalization import BatchNormalization
from keras.utils import np_utils
from keras.regularizers import ActivityRegularizer
from keras.utils.visualize_util import plot

# from keras.optimizers import SGD

from itertools import imap
from bot_utils.itertools_recipes import chunks

class UnPooling2D(Layer):
    """A 2D Repeat layer"""
    def __init__(self, poolsize=(2, 2)):
        super(UnPooling2D, self).__init__()
        self.poolsize = poolsize

    @property
    def output_shape(self):
        input_shape = self.input_shape
        return (input_shape[0], input_shape[1],
                self.poolsize[0] * input_shape[2],
                self.poolsize[1] * input_shape[3])

    def get_output(self, train):
        X = self.get_input(train)
        s1 = self.poolsize[0]
        s2 = self.poolsize[1]
        output = X.repeat(s1, axis=2).repeat(s2, axis=3)
        return output

    def get_config(self):
        return {"name":self.__class__.__name__,
            "poolsize":self.poolsize}

def get_channel_axis():
    dim_ordering = K.image_dim_ordering()
    if dim_ordering == "th":
        channel_axis = 1
    else:
        channel_axis = -1
    return channel_axis


def segnet(input_shape, num_classes): 
    """
    input_shape: (3, 360, 480)
    """

    pad = 1
    kernel = 3
    pool_size = 2
    filter_size = 64

    C, H, W = input_shape
    num_features = H * W

    model = models.Sequential()
    model.add(Layer(input_shape=input_shape))

    # Encoding layers
    model.add(ZeroPadding2D((pad,pad)))
    model.add(Convolution2D(filter_size, kernel, kernel, border_mode='valid'))
    model.add(BatchNormalization())
    model.add(Activation('relu'))
    model.add(MaxPooling2D((pool_size, pool_size)))

    model.add(ZeroPadding2D((pad,pad)))
    model.add(Convolution2D(128, kernel, kernel, border_mode='valid'))
    model.add(BatchNormalization())
    model.add(Activation('relu'))
    model.add(MaxPooling2D((pool_size, pool_size)))

    model.add(ZeroPadding2D((pad,pad)))
    model.add(Convolution2D(256, kernel, kernel, border_mode='valid'))
    model.add(BatchNormalization())
    model.add(Activation('relu'))
    model.add(MaxPooling2D((pool_size, pool_size)))

    model.add(ZeroPadding2D((pad,pad)))
    model.add(Convolution2D(512, kernel, kernel, border_mode='valid'))
    model.add(BatchNormalization())
    model.add(Activation('relu'))
    # model.add(MaxPooling2D(pool_size=(pool_size, pool_size)))

    # Decoding layers
    #model.add(UpSampling2D(size=(pool_size,pool_size)))
    model.add(ZeroPadding2D((pad,pad)))
    model.add(Convolution2D(512, kernel, kernel, border_mode='valid'))
    model.add(BatchNormalization())

    model.add(UpSampling2D((pool_size,pool_size)))
    model.add(ZeroPadding2D((pad,pad)))
    model.add(Convolution2D(256, kernel, kernel, border_mode='valid'))
    model.add(BatchNormalization())
    
    model.add(UpSampling2D((pool_size,pool_size)))
    model.add(ZeroPadding2D(padding=(pad,pad)))
    model.add(Convolution2D(128, kernel, kernel, border_mode='valid'))
    model.add(BatchNormalization())
    
    model.add(UpSampling2D((pool_size,pool_size)))
    model.add(ZeroPadding2D((pad,pad)))
    model.add(Convolution2D(filter_size, kernel, kernel, border_mode='valid'))
    model.add(BatchNormalization())


    model.add(Convolution2D(num_classes, 1, 1, border_mode='valid',))
    
    # import ipdb; ipdb.set_trace()
    model.add(Reshape((num_classes, num_features), input_shape=(num_classes,H,W)))
    model.add(Permute((2, 1)))
    model.add(Activation('softmax'))

    return model


def normalized(rgb):
    return np.dstack([cv2.equalizeHist(rgb[:,:,j]) for j in range(3)])

def binarylab(labels, num_classes):
    H, W = labels.shape
    x = np.zeros([H,W,num_classes])
    for j in range(num_classes): 
        valid = labels == j
        x[:,:,j][valid] = 1
    return x.reshape(H*W,num_classes)

def data_generator(path, num_classes):
    train_data = []
    train_label = []

    with open(os.path.join(path, 'CamVid/train.txt')) as f:
        txt = f.readlines()
        txt = [line.split(' ') for line in txt]

    for (x,y) in txt:
        full_train_path = os.path.join(path, x[8:])
        full_label_path = os.path.join(path, y[8:][:-1])
        print full_train_path, full_label_path

        X = np.rollaxis(normalized(cv2.imread(full_train_path)),2)
        Y = binarylab(cv2.imread(full_label_path)[:,:,0], num_classes)
        yield X, Y

def chunked_data(iterable, batch_size=10):
    for batch in chunks(iterable, batch_size): 
        X, Y = zip(*batch)
        yield np.array(X), np.array(Y)

if __name__ == "__main__": 

    # Parse arguments
    parser = argparse.ArgumentParser(
        description='Train Segnet using TensorFlow')
    parser.add_argument(
        '-d', '--directory', type=str, required=True, 
        help="Directory")
    args = parser.parse_args()

    # Setup Segnet training
    C, H, W = (3, 360, 480)
    num_classes = 12
    nb_epoch = 100
    samples_per_epoch = 14

    # model = segnet(input_shape=(C, H, W), num_classes=num_classes)
    # model.compile(loss="categorical_crossentropy", optimizer='adadelta')

    current_dir = os.path.dirname(os.path.realpath(__file__))
    model_path = os.path.join(current_dir, "model.png")
    # plot(model_path, to_file=model_path, show_shapes=True)

    datagen = data_generator(os.path.expanduser(args.directory), num_classes)

    chunked_datagen = chunked_data(datagen, batch_size=samples_per_epoch)
    chunked_X = imap(lambda item: item[0], chunked_datagen)
    chunked_Y = imap(lambda item: item[1], chunked_datagen)
    for (x,y) in datagen:
        print x.shape, y.shape
    # class_weight = [0.2595, 0.1826, 4.5640, 0.1417, 0.5051, 0.3826, 9.6446, 1.8418, 6.6823, 6.2478, 3.0, 7.3614]
    # history = model.fit_generator(datagen, samples_per_epoch=samples_per_epoch, nb_epoch=nb_epoch,
    #                               show_accuracy=True, verbose=2, class_weight=class_weight, nb_worker=6) 

