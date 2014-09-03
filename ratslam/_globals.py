# =============================================================================
# Federal University of Rio Grande do Sul (UFRGS)
# Connectionist Artificial Intelligence Laboratory (LIAC)
# Renato de Pontes Pereira - rppereira@inf.ufrgs.br
# =============================================================================
# Copyright (c) 2013 Renato de Pontes Pereira, renato.ppontes at gmail dot com
# 
# Permission is hereby granted, free of charge, to any person obtaining a copy 
# of this software and associated documentation files (the "Software"), to deal
# in the Software without restriction, including without limitation the rights 
# to use, copy, modify, merge, publish, distribute, sublicense, and/or sell 
# copies of the Software, and to permit persons to whom the Software is 
# furnished to do so, subject to the following conditions:
#
# The above copyright notice and this permission notice shall be included in 
# all copies or substantial portions of the Software.
#
# THE SOFTWARE IS PROVIDED "AS IS", WITHOUT WARRANTY OF ANY KIND, EXPRESS OR 
# IMPLIED, INCLUDING BUT NOT LIMITED TO THE WARRANTIES OF MERCHANTABILITY, 
# FITNESS FOR A PARTICULAR PURPOSE AND NONINFRINGEMENT. IN NO EVENT SHALL THE 
# AUTHORS OR COPYRIGHT HOLDERS BE LIABLE FOR ANY CLAIM, DAMAGES OR OTHER 
# LIABILITY, WHETHER IN AN ACTION OF CONTRACT, TORT OR OTHERWISE, ARISING FROM,
# OUT OF OR IN CONNECTION WITH THE SOFTWARE OR THE USE OR OTHER DEALINGS IN THE
# SOFTWARE.
# =============================================================================

import numpy as np
import itertools

def min_delta(d1, d2, max_):
    delta = np.min([np.abs(d1-d2), max_-np.abs(d1-d2)])
    return delta

def clip_rad_180(angle):
    while angle > np.pi:
        angle -= 2*np.pi
    while angle <= -np.pi:
        angle += 2*np.pi
    return angle

def clip_rad_360(angle):
    while angle < 0:
        angle += 2*np.pi
    while angle >= 2*np.pi:
        angle -= 2*np.pi
    return angle

def signed_delta_rad(angle1, angle2):
    dir = clip_rad_180(angle2 - angle1)
    
    delta_angle = abs(clip_rad_360(angle1) - clip_rad_360(angle2))
    
    if (delta_angle < (2*np.pi-delta_angle)):
        if (dir>0):
            angle = delta_angle
        else:
            angle = -delta_angle
    else: 
        if (dir>0):
            angle = 2*np.pi - delta_angle
        else:
            angle = -(2*np.pi-delta_angle)
    return angle


def create_pc_weights(dim, var):
    dim_center = int(np.floor(dim/2.))
    
    weight = np.zeros([dim, dim, dim])
    for x, y, z in itertools.product(xrange(dim), xrange(dim), xrange(dim)):
        dx = -(x-dim_center)**2
        dy = -(y-dim_center)**2
        dz = -(z-dim_center)**2
        weight[x, y, z] = 1.0/(var*np.sqrt(2*np.pi))*np.exp((dx+dy+dz)/(2.*var**2))

    weight = weight/np.sum(weight)
    return weight

# @profile
def compare_segments(seg1, seg2, slen):
    cwl = seg1.size

    mindiff = 1e10
    minoffset = 0

    diffs = np.zeros(slen)

    for offset in xrange(slen+1):
        e = (cwl-offset)

        cdiff = np.abs(seg1[offset:cwl] - seg2[:e])
        cdiff = np.sum(cdiff)/e

        if cdiff < mindiff:
            mindiff = cdiff
            minoffset = offset

        cdiff = np.abs(seg1[:e] - seg2[offset:cwl])
        cdiff = np.sum(cdiff)/e

        if cdiff < mindiff:
            mindiff = cdiff
            minoffset = -offset

    return minoffset, mindiff

# CONSTANTS AND ALGORITHM PARAMETERS ==========================================
# NOTE: it is need a refactoring to set these variables as a model parameter
PC_VT_INJECT_ENERGY     = 0.1
PC_DIM_XY               = 61
PC_DIM_TH               = 36
PC_W_E_VAR              = 1
PC_W_E_DIM              = 7
PC_W_I_VAR              = 2
PC_W_I_DIM              = 5
PC_GLOBAL_INHIB         = 0.00002
PC_W_EXCITE             = create_pc_weights(PC_W_E_DIM, PC_W_E_VAR)
PC_W_INHIB              = create_pc_weights(PC_W_I_DIM, PC_W_I_VAR)
PC_W_E_DIM_HALF         = int(np.floor(PC_W_E_DIM/2.))
PC_W_I_DIM_HALF         = int(np.floor(PC_W_I_DIM/2.))
PC_C_SIZE_TH            = (2.*np.pi)/PC_DIM_TH
PC_E_XY_WRAP            = range(PC_DIM_XY-PC_W_E_DIM_HALF, PC_DIM_XY) + range(PC_DIM_XY) + range(PC_W_E_DIM_HALF)
PC_E_TH_WRAP            = range(PC_DIM_TH-PC_W_E_DIM_HALF, PC_DIM_TH) + range(PC_DIM_TH) + range(PC_W_E_DIM_HALF)
PC_I_XY_WRAP            = range(PC_DIM_XY-PC_W_I_DIM_HALF, PC_DIM_XY) + range(PC_DIM_XY) + range(PC_W_I_DIM_HALF)
PC_I_TH_WRAP            = range(PC_DIM_TH-PC_W_I_DIM_HALF, PC_DIM_TH) + range(PC_DIM_TH) + range(PC_W_I_DIM_HALF)            
PC_XY_SUM_SIN_LOOKUP    = np.sin(np.multiply(range(1, PC_DIM_XY+1), (2*np.pi)/PC_DIM_XY))
PC_XY_SUM_COS_LOOKUP    = np.cos(np.multiply(range(1, PC_DIM_XY+1), (2*np.pi)/PC_DIM_XY))
PC_TH_SUM_SIN_LOOKUP    = np.sin(np.multiply(range(1, PC_DIM_TH+1), (2*np.pi)/PC_DIM_TH))
PC_TH_SUM_COS_LOOKUP    = np.cos(np.multiply(range(1, PC_DIM_TH+1), (2*np.pi)/PC_DIM_TH))
PC_CELLS_TO_AVG         = 3;
PC_AVG_XY_WRAP          = range(PC_DIM_XY-PC_CELLS_TO_AVG, PC_DIM_XY) + range(PC_DIM_XY) + range(PC_CELLS_TO_AVG)
PC_AVG_TH_WRAP          = range(PC_DIM_TH-PC_CELLS_TO_AVG, PC_DIM_TH) + range(PC_DIM_TH) + range(PC_CELLS_TO_AVG)
IMAGE_Y_SIZE            = 640
IMAGE_X_SIZE            = 480
IMAGE_VT_Y_RANGE        = slice((480/2 - 80 - 40), (480/2 + 80 - 40))
IMAGE_VT_X_RANGE        = slice((640/2 - 280 + 15), (640/2 + 280 + 15))
IMAGE_VTRANS_Y_RANGE    = slice(270, 430)
IMAGE_VROT_Y_RANGE      = slice(75, 240)
IMAGE_ODO_X_RANGE       = slice(180+15, 460+15)
VT_GLOBAL_DECAY         = 0.1
VT_ACTIVE_DECAY         = 1.0
VT_SHIFT_MATCH          = 20
VT_MATCH_THRESHOLD      = 0.09
EXP_DELTA_PC_THRESHOLD  = 1.0
EXP_CORRECTION          = 0.5
EXP_LOOPS               = 100
VTRANS_SCALE            = 100
VISUAL_ODO_SHIFT_MATCH  = 140
ODO_ROT_SCALING         = np.pi/180./7.
POSECELL_VTRANS_SCALING = 1./10.
# =============================================================================