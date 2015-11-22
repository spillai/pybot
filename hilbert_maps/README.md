# Hilbert Maps #

This repository contains the code which implements the method of the paper "Hilbert maps: scalable continuous occupancy mapping with stochastic gradient descent" by Fabio Ramos and Lionel Ott presented in RSS 2105.

```
@INPROCEEDINGS{Ramos-RSS-15, 
    AUTHOR    = {Fabio Ramos AND Lionel Ott}, 
    TITLE     = {Hilbert maps: scalable continuous occupancy mapping with stochastic gradient descent}, 
    BOOKTITLE = {Proceedings of Robotics: Science and Systems}, 
    YEAR      = {2015}
}
```

# Software Requirements #

To run the code you need the following software components:

* [Python](https://www.python.org/)
* [scikit-learn](http://scikit-learn.org/stable/index.html)
* [matplotlib](http://matplotlib.org/)
* [SciPy](http://www.scipy.org/)
* [NumPy](http://www.numpy.org/)

# Running the Example #

The script `example.py` is a simple way to produce a map using hilbert maps from carmen style logfiles as follows:

```
#!bash

example.py
    [-h]
    [--components COMPONENTS]
    [--gamma GAMMA]
    [--distance_cutoff DISTANCE_CUTOFF]
    [--resolution RESOLUTION]
    logfile
    {sparse,fourier,nystroem}
```
Only the `logfile` parameter and the feature type (`sparse`, `fourier`, or `nystroem`) is required. For more detailed parameter description use `example.py --help`.

For the intel dataset the following component numbers are decent starting choices.

* sparse 1000
* fourier 10000
* nystroem 1000