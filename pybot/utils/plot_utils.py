# Author: Sudeep Pillai <spillai@csail.mit.edu>
# License: MIT

from __future__ import division

import numpy as np
import matplotlib as mpl; # mpl.use('Agg')
import matplotlib.pyplot as plt; plt.ioff()
from matplotlib import rcParams as rc

rc['font.size'] = 7
rc['font.family'] = 'sans-serif'
rc['font.sans-serif'] = 'Arial'
rc['savefig.bbox'] = 'tight' # 'standard', 'tight'
rc['savefig.pad_inches'] = 0.01

# 'axes.titlesize' : 24
# 'axes.labelsize' : 20
# 'lines.linewidth' : 3
# 'lines.markersize' : 10
# 'xtick.labelsize' : 16
# 'ytick.labelsize' : 16


def colormap(v, scale=255): 
    return plt.cm.hsv(v.ravel())[:,:3] * scale

def moving_average(X, win_size=3): 
    return np.convolve(X, np.repeat(1.0, win_size) / win_size, 'valid')

def plot_confusion_matrix(cm, target_names, title='Confusion matrix', cmap=plt.cm.Greys, block=True):
    # Colormaps: jet, Greys
    cm_normalized = cm.astype(np.float32) / cm.sum(axis=1)[:, np.newaxis]
    plt.imshow(cm_normalized, interpolation='nearest', cmap=cmap)

    # Show confidences
    for i, cas in enumerate(cm): 
        for j, c in enumerate(cas): 
            if c > 0: 
                plt.text(j-0.1, i+0.2, c, fontsize=16, fontweight='bold', color='#b70000')

    f = plt.figure(1)
    f.clf()
    plt.title(title)
    plt.colorbar()
    tick_marks = np.arange(len(target_names))
    plt.xticks(tick_marks, target_names, rotation=45)
    plt.yticks(tick_marks, target_names)
    plt.tight_layout()
    plt.ylabel('True label')
    plt.xlabel('Predicted label')
    plt.show(block=block)


def plot_poses(poses, pose_type='absolute'): 

    f = plt.figure(1)
    f.clf()
    f.subplots_adjust(hspace=0.25)
        
    # , **{'ylabel': 'Position X(m)', 'xlim': [min(xs), max(xs)]}
    pparams = { 'linewidth':1, }
    # font = { 'family': 'normal', 'weight': 'normal', 'size': 12 }
    # mpl.rc('font', **font)

    rpyxyz = np.vstack([pose.to_rpyxyz() for pose in poses])
    ax = f.add_subplot(2, 1, 1)

    if pose_type == 'absolute': 
        for j,label in enumerate(['Roll', 'Pitch', 'Yaw']): 
            ax.plot(np.unwrap(rpyxyz[:,j], discont=np.pi), label=label)
    elif pose_type == 'relative': 
        for j,label in enumerate(['Roll', 'Pitch', 'Yaw']): 
            ax.plot(np.unwrap(rpyxyz[1:,j]-rpyxyz[:-1,j], discont=np.pi), label=label)
    else: 
        raise RuntimeError('Unknown pose_type=%s, use absolute or relative' % pose_type)
    ax.legend(loc='upper right', fancybox=False, ncol=3, 
              # bbox_to_anchor=(1.0,1.2), 
              prop={'size':13})
            
    ax = f.add_subplot(2, 1, 2)
    for j,label in enumerate(['X', 'Y', 'Z']): 
        ax.plot(rpyxyz[:,j+3], label=label)
    ax.legend(loc='upper right', fancybox=False, ncol=3, 
              # bbox_to_anchor=(1.0,1.2), 
              prop={'size':13})

    plt.tight_layout()
    plt.show(block=True)

def draw_histogram_1d(hist, output=None, shape=(20,100)): 
    if output is None: 
        output = np.zeros(shape=shape, dtype=np.uint8)
    H, W = output.shape[:2]
    w = 8
    cols = colormap(np.arange(len(hist)) * 1.0 / len(hist))
    for idx, hval in enumerate((hist * H).astype(int)): 
        output[H - hval:H, int(idx * w) : int((idx + 1) * w)] = cols[idx]
    # output[0,:,:] = 255
    output[-1,:w*len(hist),:] = 200
    # output[:,0,:] = 255
    # output[:,-1,:] = 255
    return output
        
        
    

