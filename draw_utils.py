#!/usr/bin/env python
# Helper functions for plotting

print 'Importing dummy draw_utils'

def height_map(hX, hmin=-0.20, hmax=5.0): 
    return np.array(plt.cm.hsv((hX-hmin)/(hmax-hmin)))[:,:3]

def color_by_height_axis(axis=2): 
    return height_map(X[:,axis]) * 255

def reshape_arr(arr):
    """ 
    Reshapes organized point clouds to [Nx3] form
    """
    if arr.ndim == 3:
        return arr.reshape((-1,3))
    elif arr.ndim == 2: 
        assert(arr.shape[1] == 3)
        return arr
    else: 
        raise Exception('Invalid dimensions %s' % arr.shape)

def get_color_arr(c, n, color_func=plt.cm.gist_rainbow, 
                  color_by='value', palette_size=20, flip_rb=False):
    """ 
    Convert string c to carr array (N x 3) format
    """
    carr = None;

    if color_by == 'value': 
        if isinstance(c, str): # single color
            carr = np.tile(np.array(colorConverter.to_rgb(c)), [n,1])
        elif  isinstance(c, float):
            carr = np.tile(np.array(color_func(c)), [n,1])
        else:
            carr = reshape_arr(c.astype(float) * 1.0)

    elif color_by == 'label': 
        if c < 0: 
            carr = np.tile(np.array([0,0,0,0]), [n,1])
        else: 
            carr = np.tile(np.array(color_func( (c % palette_size) * 1. / palette_size)), [n,1])
    else: 
        raise Exception("unknown color_by argument")

    if flip_rb: 
        r, b = carr[:,0], carr[:,2]
        carr[:,0], carr[:,2] = b.copy(), r.copy()
    return carr        


def init(): 
    pass

def publish_line_segments(*args, **kwargs): 
    pass

def publish_cloud(*args, **kwargs): 
    pass

def publish_pose_list(*args, **kwargs): 
    pass

