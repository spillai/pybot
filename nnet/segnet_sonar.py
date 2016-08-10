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

from bot_utils.itertools_recipes import chunks
# from bot_vision.image_utils import flip_rb
# from bot_vision.color_utils import get_random_colors, colormap

nchannels = 1

def segnet(input_shape, num_classes): 
    """
    input_shape: (nchannels, 360, 480)
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
    return cv2.equalizeHist(rgb)

def binarylab(labels, num_classes):
    H, W = labels.shape
    x = np.zeros([H,W,num_classes])
    for j in range(num_classes): 
        valid = labels == j
        x[:,:,j][valid] = 1
    return x.reshape(H*W,num_classes)

def data_generator(path, num_classes, name='train.txt'):
    txt = np.loadtxt(os.path.join(path, name), dtype=np.str)
    while True: 
        for (x,y) in txt:
            try: 
                full_train_img = cv2.imread(os.path.join(path, x),  
                                            cv2.IMREAD_GRAYSCALE)
                full_label_img = (cv2.imread(os.path.join(path, y), 
                                             cv2.IMREAD_GRAYSCALE) > 0).astype(np.uint8)
            except: 
                continue 
                
            X_ = full_train_img[:,:,np.newaxis]
            X = np.rollaxis(X_, 2)

            Y = binarylab(full_label_img, num_classes)
            W = full_label_img.ravel() * (512 * 96.0 - 1) + 1.0
            # print X.shape, Y.shape, W.shape

            yield np.asarray([X]), np.asarray([Y]) # , np.asarray([W])
        
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
    # parser.add_argument(
    #     '-c', '--colors', type=str, 
    #     default='', required=True, 
    #     help="Colors file (camvid12.png)")
    parser.add_argument(
        '-g', '--gpu', type=str, 
        default='0', required=False, 
        help="GPU device ID")
    args = parser.parse_args()

    # Setup Segnet training
    C, H, W = (nchannels, 512, 96)
    num_classes = 2
    nb_epoch = 1000
    samples_per_epoch = 10
    validation_samples_per_epoch = 5
    batch_size = 1

    # Data Generator
    datagen = data_generator(os.path.expanduser(args.directory), num_classes, name='train.txt')
    validation_datagen = data_generator(os.path.expanduser(args.directory), num_classes, name='test.txt')
    chunked_datagen = chunked_data(datagen, batch_size=batch_size)

    try: 
        os.makedirs('models')
        os.makedirs('debug')
    except:
        pass
    
    # chunked_X = imap(lambda item: item[0], chunked_datagen)
    # chunked_Y = imap(lambda item: item[1], chunked_datagen)
    # for idx, (x,y,s) in enumerate(datagen):
    #     print x.shape, y.shape, s.shape
    #     if idx > 100: 
    #         break

    
    dev_str = '/gpu:' + args.gpu if int(args.gpu) >= 0 else '/cpu:0' 
    if len(args.model): 
        model = models.load_model(os.path.expanduser(args.model))
    else: 
        with tf.device(dev_str):
            model = segnet(input_shape=(C, H, W), num_classes=num_classes)
            model.compile(loss="categorical_crossentropy", metrics=['accuracy'], optimizer='adam', sample_weight_mode=None)

            # current_dir = os.path.dirname(os.path.realpath(__file__))
            # model_path = os.path.join(current_dir, "model.png")
            # plot(model_path, to_file=model_path, show_shapes=True)

            # SVG(vutil.to_graph(model, recursive=True, show_shape=True).create(prog='dot', format="svg"))

            class_weight = [1.0 / (512 * 96), 1.0]
            callbacks = [
                ProgbarLogger(),
                TensorBoard(log_dir='./logs', histogram_freq=0, write_graph=True), 
                ModelCheckpoint(filepath="models/weights.{epoch:02d}-{loss:.2f}.hdf5", 
                                verbose=1, monitor='loss', save_weights_only=False, save_best_only=True)
            ]

            history = model.fit_generator(datagen, samples_per_epoch=samples_per_epoch, nb_epoch=nb_epoch,
                                          show_accuracy=True, verbose=2, callbacks=callbacks, class_weight=class_weight, 
                                          validation_data=validation_datagen, nb_val_samples=validation_samples_per_epoch)

            model.save('models/model.hdf5')


    # # Colors for camvid
    # colors = cv2.imread(args.colors).astype(np.uint8)
    
    print('Predict')
    with tf.device(dev_str):
        for idx, (im,target) in enumerate(datagen):
            gt = np.squeeze(target).argmax(axis=1).reshape(H,W).astype(np.uint8)
            out = model.predict_classes(im)
            out = np.squeeze(out).reshape(H,W).astype(np.uint8) * 255

            img = np.squeeze(im).reshape(H,W).astype(np.uint8)
            cv2.imwrite('debug/colored-{:04d}.png'.format(idx),
                        np.hstack([img, gt, out]))

            if idx == 360:
                break
