#!/usr/bin/env python
# Helper functions for plotting
import numpy as np
import matplotlib.pylab as plt
from matplotlib.colors import colorConverter

# from bot_externals.ros.draw_utils import *
# from bot_externals.lcm.draw_utils import *

def height_map(hX, hmin=-0.20, hmax=5.0): 
    return np.array(plt.cm.hsv((hX-hmin)/(hmax-hmin)))[:,:3]

def color_by_height_axis(axis=2): 
    return height_map(X[:,axis]) * 255

def get_color_arr_label(c, n, color_func=plt.cm.gist_rainbow, palette_size=20): 
    if c < 0: 
        carr = np.tile(np.array([0,0,0,0]), [n,1])
    else: 
        carr = np.tile(np.array(color_func( (c % palette_size) * 1. / palette_size)), [n,1])
    return carr

def reshape_arr(arr):
    """ 
    Reshapes organized point clouds to [Nx3] form
    """
    return arr.reshape(-1,3) if arr.ndim == 3 else arr

def get_color_arr(c, n, flip_rb=False):
    """ 
    Convert string c to carr array (N x 3) format
    """
    carr = None;

    if isinstance(c, str): # single color
        carr = np.tile(np.array(colorConverter.to_rgb(c)), [n,1])
    elif  isinstance(c, float):
        carr = np.tile(np.array(color_func(c)), [n,1])
    else:
        carr = reshape_arr(c)

    if flip_rb: 
        b, r = carr[:,0], carr[:,2]
        carr[:,0], carr[:,2] = r.copy(), b.copy()

    # return floating point with values in [0,1]
    return carr.astype(np.float32) / 255.0 if carr.dtype == np.uint8 else carr.astype(np.float32)

def copy_pointcloud_data(arr, carr, flip_rb=False): 
    # arr, carr = deepcopy(_arr), deepcopy(_carr)
    arr = arr.reshape(-1,3)
    N, D = arr.shape[:2]
    carr = get_color_arr(carr, N, flip_rb=flip_rb);
    return arr, carr



def init(): 
    pass

# def publish_line_segments(*args, **kwargs): 
#     pass

# def publish_cloud(*args, **kwargs): 
#     pass

# def publish_pose_list(*args, **kwargs): 
#     pass

