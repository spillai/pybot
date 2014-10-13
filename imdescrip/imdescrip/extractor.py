# Imdescrip -- a collection of tools to extract descriptors from images.
# Copyright (C) 2013  Daniel M. Steinberg (d.steinberg@acfr.usyd.edu.au)
#
# This program is free software: you can redistribute it and/or modify
# it under the terms of the GNU General Public License as published by
# the Free Software Foundation, either version 3 of the License, or
# (at your option) any later version.
#
# This program is distributed in the hope that it will be useful,
# but WITHOUT ANY WARRANTY; without even the implied warranty of
# MERCHANTABILITY or FITNESS FOR A PARTICULAR PURPOSE.  See the
# GNU General Public License for more details.
#
# You should have received a copy of the GNU General Public License
# along with this program.  If not, see <http://www.gnu.org/licenses/>.

""" Module for image descriptor extraction. """


import os, itertools, cPickle, time
import multiprocessing as mp
from utils.progress import Progress


def extract (imfile, savedir, descobj):
    """ Extract features/descriptors from a single image.

    This function calls an image descripor object on a single image in order to
    extract the images descripor. If a feature/descriptor file already exists
    for the image, this function returns

    Arguments:
        imfile:   str, the path to the image to extract features/a descriptor
                  from. 
        savedir:  str, a directory in which to save all of the image features.
                  They are pickled objects (protocol 2) with the same name as
                  the image file. The object that is pickled is the return from
                  descobj.extract().
        decobj:   An image descriptor object which does the actual extraction
                  work. the method called is descobj.extract(image). See
                  descriptors.Descriptor for an abstract base class. 
    
    Returns:
        False if there were no errors encountered, true if otherwise. See
        "errors.log" in savedir for the actual error encountered. 

    """
    
    imname = os.path.splitext(os.path.split(imfile)[1])[0] # get image name
    feafile = os.path.join(savedir, imname + ".p")

    # Check to see if feature file already exists, continue if so
    if os.path.exists(feafile) == True:
        return False 

    # Extract image descriptors
    try:
        fea = descobj.extract(imfile) # extract image descriptor
    except Exception as e:
        with open(os.path.join(savedir, 'errors.log'), 'a') as l:
            l.write('{0} : {1}\n'.format(imfile, e))
        return True

    # Write pickled feature
    with open(feafile, 'wb') as f:
        cPickle.dump(fea, f, protocol=2)

    return False


def extract_batch (filelist, savedir, descobj, verbose=False):
    """ Extract features/descriptors from a batch of images. Single-threaded. 

    This function calls an image descripor object on a batch of images in order
    to extract the images descripor. If a feature/descriptor file already exists
    for the image, it is skipped. This is a single-threaded pipeline.

    Arguments:
        filelist: A list of files of image names including their paths of images
                  to read and extract descriptors from
        savedir:  A directory in which to save all of the image features. They
                  are pickled objects (protocol 2) with the same name as the
                  image file. The object that is pickled is the return from
                  descobj.extract().
        decobj:   An image descriptor object which does the actual extraction
                  work. the method called is descobj.extract(image). See
                  descriptors.Descriptor for an abstract base class. 
        verbose:  bool, display progress?

    Returns:
        True if there we any errors extracting image features. False otherwise. 
        If there is a problem extracting any image descriptors, a file
        "errors.log" is created in the savedir directory with a list of file
        names, error number and messages.

    """

    # Try to make the save path
    if not os.path.exists(savedir):
        os.mkdir(savedir)

    errflag = False

    # Set up progess updates
    nfiles = len(filelist)
    progbar = Progress(nfiles, title='Extracting descriptors', verbose=verbose)

    # Iterate through all of the images in filelist and extract features
    for i, impath in enumerate(filelist):
        errflag |= extract(impath, savedir, descobj) 
        progbar.update(i)

    progbar.finished()
    
    if errflag == True:
        print('Done with errors. See the "errors.log" file in ' + savedir)


def extract_smp (filelist, savedir, descobj, njobs=None, verbose=False):
    """ Extract features/descriptors from a batch of images. Multi-threaded. 

    This function calls an image descripor object on a batch of images in order
    to extract the images descripor. If a feature/descriptor file already exists
    for the image, it is skipped. This is a multi-threaded (SMP) pipeline
    suitable for running on a single computer.

    Arguments:
        filelist: A list of files of image names including their paths of images
                  to read and extract descriptors from
        savedir:  A directory in which to save all of the image features. They
                  are pickled objects (protocol 2) with the same name as the
                  image file. The object that is pickled is the return from
                  descobj.extract().
        decobj:   An image descriptor object which does the actual extraction
                  work. the method called is descobj.extract(image). See
                  descriptors.Descriptor for an abstract base class. 
        njobs:    int, Number of threads to use. If None, then the number of
                  threads is chosen to be the same as the number of cores.
        verbose:  bool, display progress?

    Returns:
        True if there we any errors extracting image features. False otherwise. 
        If there is a problem extracting any image descriptors, a file
        "errors.log" is created in the savedir directory with a list of file
        names, error number and messages.

    """

    # Try to make the save path
    if not os.path.exists(savedir):
        os.mkdir(savedir)

    # Set up parallel job
    pool = mp.Pool(processes=njobs)

    # Iterate through all of the images in filelist and extract features
    result = pool.map_async(__extract_star, itertools.izip(filelist, 
                    itertools.repeat(savedir), itertools.repeat(descobj)))

    # Set up progess updates
    nfiles = len(filelist)
    progbar = Progress(nfiles, title='Extracting descriptors', verbose=verbose)

    # Get the status
    while ((result.ready() is not True) and (verbose == True)):
        approx_rem = nfiles - result._number_left * result._chunksize
        progbar.update(max(0, approx_rem)) 
        time.sleep(5)

    progbar.finished()

    # Get notification of errors
    errflag = any(result.get())
    pool.close()
    pool.join()

    if errflag == True:
        print('Done, with errors. See the "errors.log" file in ' + savedir)


def __extract_star (args):
    """ Covert args to (file, savedir, descobj) arguments. """

    return extract(*args)
