Imdescrip
=========

A collection of python tools for extracting descriptors from images (whole and
sub-image descriptors).


*Author*: Daniel Steinberg

*Institute*: Australian Centre for Field Robotics, The University of Sydney

*Date*: 20/02/2013

*License*: GPL v3 (See LICENSE)

*References*:

 [1] Yang, J.; Yu, K.; Gong, Y. & Huang, T. Linear spatial pyramid matching
     using sparse coding for image classification Computer Vision and Pattern
     Recognition, 2009. CVPR 2009. IEEE Conference on, 2009, 1794-1801

 [2] D. M. Steinberg, An Unsupervised Approach to Modelling Visual Data, PhD
     Thesis, 2013.

**If you use this package please consider citing [2]**


Functionality
-------------

It's probably easiest to describe what this package can do by breaking down each
sub-folder/package.


### descriptors:

Actual classes for extracting descriptors/features from images. For instance, a
modified versions of Yang et. al.'s sparse code spatial pyramid matching (ScSPM)
[1] descriptor is implemented here. Also provided is an abstract base class for
implementing new descriptor classes that work with the extractor model.


### extractors:

Routines for batch processing images for descriptor extraction. These routines
call the classes in descriptors to actually extract the descriptors from the
images. The extracted descriptors are then saved to individual binary pickled
objects. 


### utils:

Various utilities used by the other modules. These include:

* spatial pyramid pooling (with arbitrary pooling functions, e.g. max and mean)
* dense grid patch extraction (image and SIFT patches)
* training patch (image and SIFT) extraction from a list of images. Useful for
  training dictionaries.
* patch centring and contrast normalisation.
* image reading and resizing in a single routine.
* a simple progress bar -- mainly included to remove some package dependencies

### test:

Unit tests for this package.


Dependencies and Installation
------------

This package has the following dependencies:

System libraries, install these first (use apt-get for ubuntu):
* libatlas-dev      (spams)
* libatlas-base-dev (spams)
* libatlas3gf-base  (spams)
* libboost-python   (pyvlfeat, this worked with v1.49 but not v1.53 on Ubuntu)
* python-opencv     (image reading and manipulation)

Common python libraries (probably pre-compiled/packaged for all OSes)
* scipy
* numpy
* matplotlib (optional)

Manual install
* spams (>=2.3)
* pyvlfeat

Once all of these dependencies have been installed (this needs to be done
manually), this package can simply be placed in your python path or you can use
the setup.py with easy\_install or pip.

### Notes

For pyvlfeat I had to manually download this from pypi and change the setup.py
entry:

    LinkArgs = ['-msse', '-shared', '-lboost_python-mt-py26']
to

    LinkArgs = ['-msse', '-shared', '-lboost_python-mt-py27']

Then I could install this using 

    sudo pip install [download diectory]

Also spams is a manual install: 
  
  1. get the code from: http://spams-devel.gforge.inria.fr/index.html
  2. install the libatlas dependencies
  3. sudo pip install [download dir]


Usage
-----

Have a look at "scripts\example\_extract.py" for some usage examples, and how I
would use the ScSPM descriptor. Typically a work flow consists of:

1. Instantiating and training a descriptor object (e.g. ScSPM)
2. Saving this object with pickle.
3. Loading this descriptor object when a dataset needs to be processed.
4. Calling an extractor routine with this descriptor object on a list of images.

Of course (2) and (3) are optional, but save unnecessary ScSPM dictionary
training.


TODO
----

* Re-write the python interface for vlfeat DSIFT (pyvlfeat is unmaintained)


Thanks
------

* Oscar Pizarro - for finding the occasional problem with the OMP code.
