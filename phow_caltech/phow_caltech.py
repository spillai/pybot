#!/usr/bin/env python

"""
Python rewrite of http: //www.vlfeat.org/applications/caltech-101-code.html
Stripped from: https://github.com/shackenberg/phow_caltech101.py
"""

import numpy as np
import cv2, os, time, random

from glob import glob
from datetime import datetime

from scipy.cluster.vq import vq
from scipy.io import loadmat, savemat

from sklearn.cluster import k_means
from sklearn.decomposition import PCA

from pyvlfeat import vl_ikmeans
from vl_phow import vl_phow

from bot_vision.image_utils import im_resize
from bot_utils.db_utils import AttrDict

def get_classes(datasetpath, numClasses):
    classes_paths = [files
                     for files in glob(datasetpath + "/*")
                     if os.path.isdir(files)]
    classes_paths.sort()
    classes = [os.path.basename(class_path) for class_path in classes_paths]
    if len(classes) == 0:
       raise ValueError('no classes found')
    if len(classes) < numClasses:
       raise ValueError('conf.numClasses is bigger than the number of folders')
    classes = classes[:numClasses]
    return classes

def get_all_images(classes, conf):
    all_images = []
    all_images_class_labels = []

    def get_imgfiles(path, extensions):
        all_files = []
        all_files.extend([os.path.join(path, os.path.basename(fname))
                         for fname in glob(path + "/*")
                         if os.path.splitext(fname)[-1].lower() in extensions])
        return all_files

    for idx, imageclass in enumerate(classes):
        imgs = get_imgfiles(os.path.join(conf.calDir, imageclass), conf.extensions)
        if len(imgs) == 0:
            raise ValueError('no images for class ' + str(imageclass))

        # Sample few images per class
        imgs = random.sample(imgs, conf.imagesperclass)
        all_images = all_images + imgs
        class_labels = list(idx * np.ones(conf.imagesperclass))
        all_images_class_labels = all_images_class_labels + class_labels

    all_images_class_labels = np.array(all_images_class_labels, np.int32)
    return all_images, all_images_class_labels

def create_split(all_images, conf):
    temp = np.mod(np.arange(len(all_images)), conf.imagesperclass) < conf.numTrain
    selTrain = np.where(temp == True)[0]
    selTest = np.where(temp == False)[0]
    # the '[0]' is there, because 'where' returns tuples, don't know why....
    # the use of the 'temp' variable is not pythonic, but we need the indices 
    # not a boolean array. See Matlab code
    return selTrain, selTest


def standarizeImage(im):
    if im.shape[0] > 480:
        resize_factor = 480.0 / im.shape[0]  # don't remove trailing .0 to avoid integer devision
        im = im_resize(im, scale=resize_factor)
    if np.amax(im) > 1.1:
        im = im / 255.0
    assert((np.amax(im) > 0.01) & (np.amax(im) <= 1))
    assert((np.amin(im) >= 0.00))
    return im


def getPhowFeatures(imagedata, phowOpts):
    im = standarizeImage(imagedata)
    frames, descrs = vl_phow(im,
                             verbose=phowOpts.Verbose,
                             sizes=phowOpts.Sizes,
                             step=phowOpts.Step)
    print 'Get phow: ', descrs.shape, frames.shape
    return frames, descrs

def getImageDescriptor(model, im, conf):
    im = standarizeImage(im)
    height, width = im.shape[:2]
    numWords = model.vocab.shape[1]

    frames, descrs = getPhowFeatures(im, conf.phowOpts)
    descrs = np.array(descrs, order='F')

    print 'Pre-PCA', descrs.shape
    descrs = conf.dim_red.transform(descrs.T)
    print 'Post-PCA', descrs.shape
      
    # quantize appearance
    if model.quantizer == 'vq':
        print descrs.shape, model.vocab.shape
        binsa, _ = vq(descrs.T, model.vocab.T)
    elif model.quantizer == 'kdtree':
        raise ValueError('quantizer kdtree not implemented')
    else:
        raise ValueError('quantizer {0} not known or understood'.format(model.quantizer))

    hist = []
    for n_spatial_bins_x, n_spatial_bins_y in zip(model.numSpatialX, model.numSpatialX):
        binsx, distsx = vq(frames[0, :], np.linspace(0, width, n_spatial_bins_x))
        binsy, distsy = vq(frames[1, :], np.linspace(0, height, n_spatial_bins_y))
        # binsx and binsy list to what spatial bin each feature point belongs to
        if (np.any(distsx < 0)) | (np.any(distsx > (width/n_spatial_bins_x+0.5))):
            print 'something went wrong'
            import pdb; pdb.set_trace()
        if (np.any(distsy < 0)) | (np.any(distsy > (height/n_spatial_bins_y+0.5))):
            print 'something went wrong'
            import pdb; pdb.set_trace()

        # combined quantization
        number_of_bins = n_spatial_bins_x * n_spatial_bins_y * numWords
        temp = np.arange(number_of_bins)
        # update using this: http://stackoverflow.com/questions/15230179/how-to-get-the-linear-index-for-a-numpy-array-sub2ind
        temp = temp.reshape([n_spatial_bins_x, n_spatial_bins_y, numWords])
        bin_comb = temp[binsx, binsy, binsa]
        hist_temp, _ = np.histogram(bin_comb, bins=range(number_of_bins+1), density=True)
        hist.append(hist_temp)

    hist = np.hstack(hist)
    hist = np.array(hist, 'float32') / sum(hist)
    return hist



def trainVocab(selTrain, all_images, conf):
    selTrainFeats = random.sample(selTrain, conf.images_for_histogram)
    descrs = []
    # if MULTIPROCESSING:
    #     raise ValueError('MULTIPROCESSING not implemented')
    #     #pool = Pool(processes=30)  
    #     #list_of_train_images = [all_images[i] for i in selTrainFeats]
    #     #descrs.append(pool.map_async(getPhowFeatures, list_of_train_images).get())        
    # else:
    for i in selTrainFeats:
        im = cv2.imread(all_images[i])
        descrs.append(getPhowFeatures(im, conf.phowOpts)[1])
        # the '[1]' is there because we only want the descriptors and not the frames
    
    descrs = np.hstack(descrs)
    # n_features = descrs.shape[1]
    # sample_indices = random.sample(np.arange(n_features), conf.numbers_of_features_for_histogram)
    # descrs = descrs[:, sample_indices]

    print 'Pre-PCA', descrs.shape, 'n: ', conf.numbers_of_features_for_histogram
    conf.dim_red = PCA(n_components=conf.numbers_of_features_for_histogram, whiten=True)
    descrs = conf.dim_red.fit_transform(descrs.T)
    print 'Post-PCA', descrs.shape

    # Quantize the descriptors to get the visual words
    vocab, _, _ = k_means(descrs.astype(np.float64), 
                          n_clusters=conf.numWords, n_init=5, init='k-means++', verbose=True)

    # vocab, _ = vl_ikmeans(descrs,
    #                       conf.numWords,
    #                       100, 
    #                       'elkan',
    #                       int(conf.verbose))

    return vocab


def computeHistograms(all_images, model, conf):
    hists = []
    for ii, imagefname in enumerate(all_images):
        print('Processing {0} ({1:.2f}%)'.format(imagefname, 100.0 * ii / len(all_images)))
        im = cv2.imread(imagefname)
        hists_temp = getImageDescriptor(model, im, conf)
        hists.append(hists_temp)
    hists = np.vstack(hists)
    return hists


if __name__ == '__main__':

    OVERWRITE = True
    VERBOSE = True

    # Configuration

    conf = AttrDict()
    conf.calDir = os.path.expanduser('~/data/caltech_101/101_ObjectCategories')
    conf.dataDir = 'results/' 
    conf.autoDownloadData = True 
    conf.numTrain = 5
    conf.numTest = 5

    conf.numClasses = 102 
    conf.numWords = 600 
    conf.numSpatialX = 4 
    conf.numSpatialY = 4 
    conf.quantizer = 'vq' 

    conf.svm = AttrDict()
    conf.svm.C = 10 
    conf.svm.solver = 'pegasos' 
    conf.svm.biasMultiplier = 1 
    conf.phowOpts = AttrDict(Step=5)
    conf.clobber = False 
    conf.tinyProblem = True
    conf.prefix = 'baseline' 
    conf.randSeed = 1 

    # more configs
    conf.imagesperclass = conf.numTrain + conf.numTest
    conf.extensions = [".jpg", ".bmp", ".png", ".pgm", ".tif", ".tiff"]
    conf.images_for_histogram = 10
    conf.numbers_of_features_for_histogram = 4096
    conf.verbose = True

    if conf.tinyProblem: 
      conf.prefix = 'tiny' 
      conf.numClasses = 2
      conf.numSpatialX = 2 
      conf.numSpatialY = 2 
      conf.numWords = 300 
      conf.phowOpts = AttrDict(Verbose=2, Sizes=7, Step=3)

    conf.vocabPath = os.path.join(conf.dataDir, ''.join([conf.prefix ,'-vocab.mat'])) 
    conf.histPath = os.path.join(conf.dataDir, ''.join([conf.prefix,  '-hists.mat'])) 
    conf.modelPath = os.path.join(conf.dataDir, ''.join([conf.prefix, '-model.mat'])) 
    conf.resultPath = os.path.join(conf.dataDir, ''.join([conf.prefix, '-result'])) 
    
    # Setup results
    if not os.path.exists(conf.dataDir):
        os.makedirs(conf.dataDir)
        print "Folder " + conf.dataDir + " created"

    # Setup data
    classes = get_classes(conf.calDir, conf.numClasses)
    
    # Get images
    all_images, all_images_class_labels = get_all_images(classes, conf)
    selTrain, selTest = create_split(all_images, conf)

    model = AttrDict()
    model.classes = classes 
    model.phowOpts = conf.phowOpts 
    model.numSpatialX = conf.numSpatialX 
    model.numSpatialY = conf.numSpatialY 
    model.quantizer = conf.quantizer 
    model.vocab = [] 
    model.w = [] 
    model.b = [] 
    # FIX: model.classify = @classify

    if VERBOSE: print str(datetime.now()) + ' start training vocab'
    if (not os.path.exists(conf.vocabPath)) | OVERWRITE:
        vocab = trainVocab(selTrain, all_images, conf)    
        savemat(conf.vocabPath, {'vocab': vocab})
    else:
        if VERBOSE: print 'using old vocab from ' + conf.vocabPath
        vocab = loadmat(conf.vocabPath)['vocab']
    model.vocab = vocab

    if VERBOSE: print str(datetime.now()) + ' start computing hists'
    if (not os.path.exists(conf.histPath)) | OVERWRITE:
        hists = computeHistograms(all_images, model, conf)
        savemat(conf.histPath, {'hists': hists})
    else:
        if VERBOSE: print 'using old hists from ' + conf.histPath
        hists = loadmat(conf.histPath)['hists']


