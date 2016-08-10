# from __future__ import absolute_import
# from __future__ import division
# from __future__ import print_function

import os
import argparse
import cv2
import numpy as np
np.random.seed(0)

from collections import OrderedDict
from itertools import imap

# os.environ['KERAS_BACKEND'] = 'theano'
os.environ['KERAS_BACKEND'] = 'tensorflow'

import tensorflow as tf
from keras import backend as K

import keras.models as models
from keras.layers.noise import GaussianNoise
from keras.layers.core import Layer, Dense, Dropout, Activation, Flatten, Reshape, Merge, Permute
from keras.layers.convolutional import Convolution2D, MaxPooling2D, UpSampling2D, ZeroPadding2D
from keras.layers.normalization import BatchNormalization
from keras.utils import np_utils
from keras.regularizers import ActivityRegularizer
from keras.callbacks import ModelCheckpoint, RemoteMonitor, TensorBoard, ProgbarLogger

# import keras.utils.visualize_util as vutil
# from IPython.display import SVG

# from keras.utils.visualize_util import plot
# from keras.optimizers import SGD

from bot_utils.itertools_recipes import chunks
from bot_vision.image_utils import flip_rb
from bot_vision.color_utils import get_random_colors, colormap

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

    
def segnet(input_shape, num_classes): 
    """
    input_shape: (3, 360, 480)
    """

    C, H, W = input_shape
    num_features = H * W

    model = models.Sequential()
    model.add(Layer(input_shape=input_shape))

    # Encoding layers
    model.add(ZeroPadding2D((1,1)))
    model.add(Convolution2D(64, 3, 3, border_mode='valid'))
    model.add(BatchNormalization())
    model.add(Activation('relu'))
    model.add(MaxPooling2D((2, 2)))

    model.add(ZeroPadding2D((1,1)))
    model.add(Convolution2D(128, 3, 3, border_mode='valid'))
    model.add(BatchNormalization())
    model.add(Activation('relu'))
    model.add(MaxPooling2D((2, 2)))

    model.add(ZeroPadding2D((1,1)))
    model.add(Convolution2D(256, 3, 3, border_mode='valid'))
    model.add(BatchNormalization())
    model.add(Activation('relu'))
    model.add(MaxPooling2D((2, 2)))

    model.add(ZeroPadding2D((1,1)))
    model.add(Convolution2D(512, 3, 3, border_mode='valid'))
    model.add(BatchNormalization())
    model.add(Activation('relu'))

    # Decoding layers
    model.add(ZeroPadding2D((1,1)))
    model.add(Convolution2D(512, 3, 3, border_mode='valid'))
    model.add(BatchNormalization())

    model.add(UpSampling2D((2,2)))
    model.add(ZeroPadding2D((1,1)))
    model.add(Convolution2D(256, 3, 3, border_mode='valid'))
    model.add(BatchNormalization())
    
    model.add(UpSampling2D((2,2)))
    model.add(ZeroPadding2D((1,1)))
    model.add(Convolution2D(128, 3, 3, border_mode='valid'))
    model.add(BatchNormalization())
    
    model.add(UpSampling2D((2,2)))
    model.add(ZeroPadding2D((1,1)))
    model.add(Convolution2D(64, 3, 3, border_mode='valid'))
    model.add(BatchNormalization())

    model.add(Convolution2D(num_classes, 1, 1, border_mode='valid',))
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
    with open(os.path.join(path, 'train.txt')) as f:
        txt = f.readlines()
        txt = [line.split(' ') for line in txt]

    while True: 
        for (x,y) in txt:
            full_train_path = os.path.join(path, x[15:])
            full_label_path = os.path.join(path, y[15:][:-1])
            X = np.rollaxis(normalized(cv2.imread(full_train_path)),2)
            Y = binarylab(cv2.imread(full_label_path)[:,:,0], num_classes)
            yield np.asarray([X]), np.asarray([Y])
        
def chunked_data(iterable, batch_size=10):
    for batch in chunks(iterable, batch_size): 
        X, Y = zip(*batch)
        yield np.vstack(X), np.vstack(Y)

if __name__ == "__main__": 

    # Parse arguments
    parser = argparse.ArgumentParser(
        description='Train Segnet using TensorFlow')
    parser.add_argument(
        '-d', '--directory', type=str, 
        default='', required=True, 
        help="Directory")
    parser.add_argument(
        '-m', '--model', type=str, 
        default='', required=False, 
        help="Model with weights")
    parser.add_argument(
        '-c', '--colors', type=str, 
        default='', required=True, 
        help="Colors file (camvid12.png)")
    parser.add_argument(
        '-g', '--gpu', type=str, 
        default='0', required=False, 
        help="GPU device ID")
    args = parser.parse_args()

    # Setup Segnet training
    C, H, W = (3, 360, 480)
    num_classes = 12
    nb_epoch = 3000
    samples_per_epoch = 30
    batch_size = 1

    # Data Generator
    datagen = data_generator(os.path.expanduser(args.directory), num_classes)
    chunked_datagen = chunked_data(datagen, batch_size=batch_size)

    try: 
        os.makedirs('models')
        os.makedirs('debug')
    except:
        pass
    
    # chunked_X = imap(lambda item: item[0], chunked_datagen)
    # chunked_Y = imap(lambda item: item[1], chunked_datagen)
    # for (x,y) in datagen:
    #     print x.shape, y.shape
    
    if len(args.model): 
        model = models.load_model(os.path.expanduser(args.model))
    else: 
        with tf.device('/gpu:' + args.gpu):
            model = segnet(input_shape=(C, H, W), num_classes=num_classes)
            model.compile(loss="categorical_crossentropy", optimizer='adam')

            # current_dir = os.path.dirname(os.path.realpath(__file__))
            # model_path = os.path.join(current_dir, "model.png")
            # plot(model_path, to_file=model_path, show_shapes=True)

            # SVG(vutil.to_graph(model, recursive=True, show_shape=True).create(prog='dot', format="svg"))

            class_weight = [0.2595, 0.1826, 4.5640, 0.1417, 0.5051, 0.3826, 9.6446, 1.8418, 6.6823, 6.2478, 3.0, 7.3614]
            callbacks = [
                ProgbarLogger(),
                TensorBoard(log_dir='./logs', histogram_freq=0, write_graph=True), 
                ModelCheckpoint(filepath="models/weights.{epoch:02d}-{loss:.2f}.hdf5", 
                                verbose=1, monitor='loss', save_weights_only=False, save_best_only=True)
            ]

            history = model.fit_generator(datagen, samples_per_epoch=samples_per_epoch, nb_epoch=nb_epoch,
                                          show_accuracy=True, verbose=2, callbacks=callbacks, class_weight=class_weight)

            model.save('models/model.hdf5')


    # Colors for camvid
    colors = cv2.imread(args.colors).astype(np.uint8)
    
    print('Predict')
    with tf.device('/gpu:' + args.gpu):
        for idx, (im,target) in enumerate(datagen):
            gt = np.squeeze(target).argmax(axis=1).reshape(H,W).astype(np.uint8)
            
            out = model.predict_classes(im)
            out = np.squeeze(out).reshape(H,W).astype(np.uint8)
            colored = cv2.LUT(np.dstack([out, out, out]), colors)
            colored_gt = cv2.LUT(np.dstack([gt, gt, gt]), colors)
            cv2.imwrite('debug/colored-{:04d}.png'.format(idx),
                        np.vstack([np.squeeze(im).transpose(1,2,0),
                                   colored,
                                   colored_gt]))

            if idx == 360:
                break
