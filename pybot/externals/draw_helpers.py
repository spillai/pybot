# Author: Sudeep Pillai <spillai@csail.mit.edu>
# License: MIT

# Helper functions for plotting
import numpy as np
from pybot.utils.plot_utils import plt
from matplotlib.colors import colorConverter
from copy import deepcopy

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


def copy_pointcloud_data(_arr, _carr, flip_rb=False): 
    arr, carr = deepcopy(_arr), deepcopy(_carr)
    arr = arr.reshape(-1,3)
    N, D = arr.shape[:2]
    carr = get_color_arr(carr, N, flip_rb=flip_rb);
    return arr, carr


class Frustum(object): 
    """
    Also see pybot.vision.camera_utils.Frustum (need to fix redundancy)
    """

    def __init__(self, pose, zmin=0.0, zmax=0.1, fov=np.deg2rad(60)): 
        # FoV derived from fx,fy,cx,cy=500,500,320,240
        # fovx, fovy = 65.23848614  51.28201165
        rx, ry = 0.638, 0.478

        self.pose = pose
        arr = [np.array([-rx, -ry, 1.]) * zmin,
               np.array([-rx,  ry, 1.]) * zmin,
               np.array([ rx,  ry, 1.]) * zmin,
               np.array([ rx, -ry, 1.]) * zmin,

               np.array([-rx, -ry, 1.]) * zmax,
               np.array([-rx,  ry, 1.]) * zmax,
               np.array([ rx,  ry, 1.]) * zmax,
               np.array([ rx, -ry, 1.]) * zmax]

        # vertices: nul, nll, nlr, nur, ful, fll, flr, fur
        self.vertices_ = self.pose * np.vstack(arr)

        # self.near, self.far = np.array([0,0,zmin]), np.array([0,0,zmax])
        # self.near_off, self.far_off = np.tan(fov / 2) * zmin, np.tan(fov / 2) * zmax

        # arr = [self.near + np.array([-1, -1, 0]) * self.near_off, 
        #        self.near + np.array([1, -1, 0]) * self.near_off, 
        #        self.near + np.array([1, 1, 0]) * self.near_off, 
        #        self.near + np.array([-1, 1, 0]) * self.near_off, 
               
        #        self.far + np.array([-1, -1, 0]) * self.far_off, 
        #        self.far + np.array([1, -1, 0]) * self.far_off, 
        #        self.far + np.array([1, 1, 0]) * self.far_off, 
        #        self.far + np.array([-1, 1, 0]) * self.far_off]
        
        # nll, nlr, nur, nul, fll, flr, fur, ful = self.pose * np.vstack(arr)
        # return nll, nlr, nur, nul, fll, flr, fur, ful
    
    @property
    def vertices(self): 
        return self.vertices_

