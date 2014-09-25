#!/usr/bin/env python

import numpy as np
import matplotlib as mpl
import matplotlib.pylab as plt

def colormap(v, scale=255): 
    return plt.cm.hsv(v.ravel())[:,:3] * scale
