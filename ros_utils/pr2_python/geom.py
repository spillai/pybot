# Software License Agreement (BSD License)
#
# Copyright (c) 2008, Willow Garage, Inc.
# All rights reserved.
#
# Redistribution and use in source and binary forms, with or without
# modification, are permitted provided that the following conditions
# are met:
#
#  * Redistributions of source code must retain the above copyright
#    notice, this list of conditions and the following disclaimer.
#  * Redistributions in binary form must reproduce the above
#    copyright notice, this list of conditions and the following
#    disclaimer in the documentation and/or other materials provided
#    with the distribution.
#  * Neither the name of Willow Garage, Inc. nor the names of its
#    contributors may be used to endorse or promote products derived
#    from this software without specific prior written permission.
#
# THIS SOFTWARE IS PROVIDED BY THE COPYRIGHT HOLDERS AND CONTRIBUTORS
# "AS IS" AND ANY EXPRESS OR IMPLIED WARRANTIES, INCLUDING, BUT NOT
# LIMITED TO, THE IMPLIED WARRANTIES OF MERCHANTABILITY AND FITNESS
# FOR A PARTICULAR PURPOSE ARE DISCLAIMED. IN NO EVENT SHALL THE
# COPYRIGHT OWNER OR CONTRIBUTORS BE LIABLE FOR ANY DIRECT, INDIRECT,
# INCIDENTAL, SPECIAL, EXEMPLARY, OR CONSEQUENTIAL DAMAGES (INCLUDING,
# BUT NOT LIMITED TO, PROCUREMENT OF SUBSTITUTE GOODS OR SERVICES;
# LOSS OF USE, DATA, OR PROFITS; OR BUSINESS INTERRUPTION) HOWEVER
# CAUSED AND ON ANY THEORY OF LIABILITY, WHETHER IN CONTRACT, STRICT
# LIABILITY, OR TORT (INCLUDING NEGLIGENCE OR OTHERWISE) ARISING IN
# ANY WAY OUT OF THE USE OF THIS SOFTWARE, EVEN IF ADVISED OF THE
# POSSIBILITY OF SUCH DAMAGE.
#
# Author: Jon Binney
'''
Functions for working with geometry using numpy types.
'''

__docformat__ = "restructuredtext en"

import roslib
roslib.load_manifest('pr2_python')

import numpy as np
from tf import transformations

def to_hom(points):
    if len(points.shape) == 1:
        points_hom = np.ones(points.shape[0] + 1,)
        points_hom[:-1] = points
    else:
        points_hom = np.ones((points.shape[0], points.shape[1]+1))
        points_hom[:,:-1] = points
    return points_hom

def from_hom(points):
    if len(points.shape) == 1:
        return points[:-1] / points[-1]
    else:
        return points[:,:-1] / np.tile(np.reshape(points[:,-1], (len(points), 1)), (1, points.shape[1]-1))

def transform_points(points, transform):
    '''
    Transform one or more D dimensional points by the given homogeneous transform.

    points - (D,) or (N, D) array

    transform - (D + 1, D + 1) array
    '''
    return from_hom(np.dot(to_hom(points), np.transpose(np.array(transform))))

def transform_vectors(vectors, transform):
    R = np.array(transform[:3,:3]) / transform[3,3]
    return np.dot(vectors, np.transpose(R))

def crop_to_bbox_mask(points, xmin, xmax, ymin, ymax, zmin, zmax):
    bbox_mask = (points[:,0] > xmin) & (points[:,0] < xmax)
    bbox_mask &= (points[:,1] > ymin) & (points[:,1] < ymax)
    bbox_mask &= (points[:,2] > zmin) & (points[:,2] < zmax)
    return bbox_mask

def crop_to_bbox(points, xmin, xmax, ymin, ymax, zmin, zmax):
    return points[crop_to_bbox_mask(points, xmin, xmax, ymin, ymax, zmin, zmax)]

def hmat_to_trans_rot(hmat):
    '''
    Converts a 4x4 homogenous rigid transformation matrix to a translation and a
    quaternion rotation.
    '''
    scale, shear, angles, trans, persp = transformations.decompose_matrix(hmat)
    rot = transformations.quaternion_from_euler(*angles)
    return trans, rot

def trans_rot_to_hmat(trans, rot):
    '''
    Converts a rotation and translation to a homogeneous transform.

    **Args:**

        **trans (np.array):** Translation (x, y, z).

        **rot (np.array):** Quaternion (x, y, z, w).

    **Returns:**
        H (np.array): 4x4 homogenous transform matrix.
    '''
    H = transformations.quaternion_matrix(rot)
    H[0:3, 3] = trans
    return H

    
class Transform:
    def __init__(self):
        pass

    def from_pose_stamped(self, ps):
        return self.from_pose(ps.pose)

    def from_pose(self, pose):
        if isinstance(pose, PoseStamped):
            pose = pose.pose
        p = pose.position
        q = pose.orientation
        trans = np.array((p.x, p.y, p.z))
        rot = np.array((q.x, q.y, q.z, q.w))
        return self.from_trans_rot(trans, rot)

    def from_trans_rot(self, trans, rot):
        hmat = transformations.quaternion_matrix(rot)
        hmat[0:3, 3] = trans
        return self.from_hmat(hmat)

    def from_hmat(self, hmat):
        self._hmat = hmat
        return self
    
    def to_hmat(self):
        return self._hmat.copy()

    def transform_point(self, p):
        return transform_points(p, self._hmat)

    def get_origin(self):
        return self._hmat[:3,3] / self._hmat[3,3]

    def inverse(self):
        return Transform().from_hmat(linalg.inv(self._hmat))
