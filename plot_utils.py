#!/usr/bin/env python

import numpy as np
import matplotlib as mpl
import matplotlib.pylab as plt

def colormap(v, scale=255): 
    return plt.cm.hsv(v.ravel())[:,:3] * scale


def moving_average(X, win_size=3): 
    return np.convolve(X, np.repeat(1.0, win_size) / win_size, 'valid')

def plot_poses(poses, pose_type='absolute'): 

    f = plt.figure(1)
    f.clf()
    f.subplots_adjust(hspace=0.25)
        
    # , **{'ylabel': 'Position X(m)', 'xlim': [min(xs), max(xs)]}
    pparams = { 'linewidth':1, }
    # font = { 'family': 'normal', 'weight': 'normal', 'size': 12 }
    # mpl.rc('font', **font)

    rpyxyz = np.vstack([pose.to_roll_pitch_yaw_x_y_z() for pose in poses])
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

