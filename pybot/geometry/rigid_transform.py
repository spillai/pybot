"""
General-purpose class for rigid-body transformations.
"""
# Author: Sudeep Pillai <spillai@csail.mit.edu>
# License: MIT

import random as rnd

import numpy as np

from pybot.geometry import transformations as tf
from pybot.geometry.quaternion import Quaternion


def normalize_vec(v): 
    """ Normalizes a vector with its L2 norm """
    return v * 1.0 / np.linalg.norm(v)


def skew(v, return_dv=False):
  """ 
  Returns the skew-symmetric matrix of a vector
  Ref: https://github.com/dreamdragon/Solve3Plus1/blob/master/skew3.m

  Also known as the cross-product matrix [v]_x such that 
  the cross product of (v x w) is equivalent to the 
  matrix multiplication of the cross product matrix of 
  v ([v]_x) and w

  In other words: v x w = [v]_x * w
  """
  sk = np.float32([[0, -v[2], v[1]],
                   [v[2], 0, -v[0]],
                   [-v[1], v[0], 0]])
    
  if return_dv: 
      dV = np.float32([[0, 0, 0], 
                       [0, 0, -1], 
                       [0, 1, 0], 
                       [0, 0, 1], 
                       [0, 0, 0], 
                       [-1, 0, 0], 
                       [0, -1, 0], 
                       [1, 0, 0], 
                       [0, 0, 0]])
      return sk, dV
  else: 
      return sk
        

def tf_construct(vec1,vec2): 
    """
    Construct a reference frame with two vectors

    Align vx along v1, and construct [vx,vy,vz] as follows:
      vx = v1
      vz = v1 x v2
      vy = vz x vx

    TODO: checks for degenerate cases
    """
    assert(np.linalg.norm(vec1)-1.0 < 1e-6)
    assert(np.linalg.norm(vec2)-1.0 < 1e-6)
    v1, v2 = vec1.copy(), vec2.copy()

    v1 = normalize_vec(v1)
    v2 = normalize_vec(v2)
    
    vz = np.cross(v1, v2); 
    vz = normalize_vec(vz)
    
    vy = np.cross(vz, v1)
    vy = normalize_vec(vy)

    R = np.vstack([v1, vy, vz]).T
    
    return R


def tf_construct_3pt(p1, p2, p3, origin=None):
    """ Construct triad with 3-points """
    v1, v2 = p1-p2, p2-p3
    v1 = v1 / np.linalg.norm(v1)
    v2 = v2 / np.linalg.norm(v2)
    R = tf_construct(v1,v2)
    if origin is None: 
        return RigidTransform.from_Rt(R, p2)
    else: 
        return RigidTransform.from_Rt(R, origin)

def tf_compose(R, t): 
    """ Construct [R t; 0 1] transformation matrix from R and t """
    T = np.eye(4);
    T[:3,:3] = R.copy()
    T[:3,3] = t.copy()
    return T


def rpyxyz(roll, pitch, yaw, x, y, z, axes='rxyz'):
    return RigidTransform.from_rpyxyz(roll, pitch, yaw, x, y, z, axes=axes)


def pose(pid, rt):
    assert(isinstance(rt, RigidTransform))
    return Pose.from_rigid_transform(pid, rt)


class RigidTransform(object):
    """
    SE(3) rigid transform class that allows compounding of 6-DOF poses
    and provides common transformations that are commonly seen in geometric problems.
    
    quat: Quaternion/Rotation (xyzw)
    tvec: Translation (xyz)
    """
    def __init__(self, xyzw=[0.,0.,0.,1.], tvec=[0.,0.,0.]):
        """ Initialize a RigidTransform with Quaternion and 3D Position """
        self.quat = Quaternion(xyzw)
        self.tvec = np.float32(tvec)

    def __repr__(self):
        return 'rpy (rxyz): %s tvec: %s' % \
            (np.array_str(self.quat.to_rpy(axes='rxyz'),
                          precision=2, suppress_small=True), 
             np.array_str(self.tvec, precision=2, suppress_small=True))

    def __mul__(self, other):
        """ 
        Left-multiply RigidTransform with another rigid transform
   
        Two variants: 
           RigidTransform: Identical to oplus operation
              - self * other = self.oplus(other)
        
           ndarray: transform [N x 3] point set (X_2 = p_21 * X_1)
        """
        if isinstance(other, RigidTransform):
            return self.oplus(other)
        else:          
            X = np.hstack([other, np.ones((len(other),1))]).T
            return (np.dot(self.matrix, X).T)[:,:3]

    def __rmul__(self, other): 
        raise NotImplementedError('Right multiply not implemented yet!')                    

    def copy(self):
        return RigidTransform(self.quat.copy(), self.tvec.copy())
    
    def inverse(self):
        """
        Returns a new RigidTransform that corresponds to the
        inverse of this one 
        """
        qinv = self.quat.inverse()
        return RigidTransform(qinv, qinv.rotate(-self.tvec))

    def oplus(self, other): 
        if isinstance(other, RigidTransform): 
            t = self.quat.rotate(other.tvec) + self.tvec
            r = self.quat * other.quat
            return RigidTransform(r, t)
        elif isinstance(other, list): 
            return [self.oplus(o) for o in other]
        else: 
            raise TypeError("Type inconsistent", type(other), other.__class__)

    def ominus(self, other):
        raise NotImplementedError()
        # if not isinstance(other, RigidTransform): 
        #     raise TypeError("Type inconsistent")
        # oinv = other.inverse()
        # return oinv.oplus(self)

    def wrt(self, p_tr): 
        """
        self: p_ro, desired: p_to
        o: other, r: reference, t: target
        """
        return self * p_tr

    def rotate_vec(self, v): 
        if v.ndim == 2: 
            return np.vstack([self.quat.rotate(v_) for v_ in v])
        else: 
            assert(v.ndim == 1 or (v.ndim == 2 and v.shape[0] == 1))
            return self.quat.rotate(v)


    def interpolate(self, other, w): 
        """
        LERP interpolation on rotation, and linear interpolation on position 
        Other approaches: 
        https://www.cvl.isy.liu.se/education/graduate/geometry-for-computer-vision-2014/geometry2014/lecture7.pdf
        """
        assert(w >= 0 and w <= 1.0)
        return self.from_Rt(self.rotation.interpolate(other.rotation, w).R,
                            self.t + w * (other.t - self.t))

        # return self.from_Rt(self.R * expm3(w * logm(self.R.T * other.R)), self.t + w * (other.t - self.t))
        # return self.from_matrix(self.matrix * expm(w * logm((self.inverse() * other).matrix)))

    def to_matrix(self):
        """ Returns a 4x4 homogenous matrix of the form [R t; 0 1] """
        result = self.quat.to_matrix()
        result[:3, 3] = self.tvec
        return result

    def to_vector(self):
        """ Returns a 7-vector representation of 
        rigid transform [tx, ty, tz, qx, qy, qz, qw] 
        """
        return np.hstack([self.tvec, self.xyzw])
    
    def to_Rt(self):
        """ Returns rotation R, and translational vector t """
        T = self.to_matrix()
        return T[:3,:3].copy(), T[:3,3].copy()

    def to_rpyxyz(self, axes='rxyz'):
        r, p, y = self.quat.to_rpy(axes=axes)
        return np.array((r, p, y, self.tvec[0], self.tvec[1], self.tvec[2]))

    @classmethod
    def from_rpyxyz(cls, roll, pitch, yaw, x, y, z, axes='rxyz'):
        q = Quaternion.from_rpy(roll, pitch, yaw, axes=axes)
        return cls(q, (x, y, z))

    @classmethod
    def from_Rt(cls, R, t):
        T = np.eye(4)
        T[:3,:3] = R.copy();
        return cls(Quaternion.from_matrix(T), t)

    @classmethod
    def from_matrix(cls, T):
        return cls(Quaternion.from_matrix(T), T[:3,3])

    @classmethod
    def from_vector(cls, v):
        return cls(xyzw=v[3:], tvec=v[:3])

    @classmethod
    def from_triad(cls, pos, v1, v2):
        return RigidTransform.from_matrix(
            tf_compose(tf_construct(v1, v2), pos))

    @classmethod
    def from_angle_axis(cls, angle, axis, tvec): 
        return cls(Quaternion.from_angle_axis(angle, axis), tvec)

    @property
    def wxyz(self):
        return self.quat.wxyz

    @property
    def xyzw(self):
        return self.quat.xyzw

    @property
    def R(self): 
        return self.quat.R

    @property
    def t(self): 
        return self.tvec

    @property
    def orientation(self): 
        return self.quat

    @property
    def rotation(self): 
        return self.quat

    @property
    def rpyxyz(self):
        return self.to_rpyxyz()

    @property
    def translation(self):
        return self.tvec

    @classmethod
    def identity(cls):
        return cls()

    @classmethod
    def random(cls, t=1):
        q_wxyz = [ rnd.random(), rnd.random(), rnd.random(), rnd.random() ]
        qmag = np.sqrt(sum([x*x for x in q_wxyz]))
        q_wxyz = [ x / qmag for x in q_wxyz ]
        translation = [ rnd.uniform(-t, t), rnd.uniform(-t, t), rnd.uniform(-t, t) ]
        return cls(Quaternion.from_wxyz(q_wxyz), translation)
    
    @property
    def matrix(self): 
        return self.to_matrix()

    @property
    def vector(self): 
        return self.to_vector()

    def scaled(self, scale):
        " Returns Sim3 representation of rigid-body transformation with scale "
        return Sim3(xyzw=self.xyzw, tvec=self.tvec, scale=scale)

class Sim3(RigidTransform): 
    def __init__(self, xyzw=[0.,0.,0.,1.], tvec=[0.,0.,0.], scale=1.0):    
        RigidTransform.__init__(self, xyzw=xyzw, tvec=tvec)
        self.scale = scale

    @classmethod
    def from_matrix(cls, T):
        sR_t = np.eye(4)
        sR_t[:3,:3] = T[:3,:3] / T[3,3]
        return cls(Quaternion.from_matrix(sR_t), T[:3,3], scale=1.0 / T[3,3])

    def to_matrix(self):
        result = self.quat.to_matrix()
        result[:3, 3] = self.tvec
        result[3, 3] = 1.0 / self.scale
        result[:3, :3] /= self.scale
        return result

    def __mul__(self, other):
        """ 
        See RigidTransform.__mul__
        Left-multiply RigidTransform with another rigid transform
   
        Two variants: 
           RigidTransform: Identical to oplus operation
            - t = self.t + self.scale * (self.R * o.t)
            - R = o.R * self.R

              - self * other = self.oplus(other)
        """
        if isinstance(other, RigidTransform):
            return self.oplus(other)
        else:
            raise NotImplementedError()
            # X = np.hstack([other, np.ones((len(other),1))]).T
            # return (np.dot(self.matrix, X).T)[:,:3]

    def oplus(self, other): 
        if isinstance(other, RigidTransform): 
            t = self.scale * self.quat.rotate(other.tvec) + self.tvec
            r = self.quat * other.quat
            return RigidTransform(r, t)
        elif isinstance(other, list) and isinstance(other[0], ): 
            return [self.oplus(o) for o in other]
        else: 
            raise TypeError("Type inconsistent", type(other), other.__class__)

        
class Pose(RigidTransform): 
    def __init__(self, pid, xyzw=[0.,0.,0.,1.], tvec=[0.,0.,0.]):
        RigidTransform.__init__(self, xyzw=xyzw, tvec=tvec)
        self.id = pid

    @classmethod
    def from_rigid_transform(cls, pid, pose):
        return cls(pid, pose.quat, pose.tvec)
    
    def __repr__(self): 
        return 'Pose ID: %i, rpy (rxyz): %s tvec: %s' % \
            (self.id, 
             np.array_str(self.quat.to_rpy(axes='rxyz'), precision=2), 
             np.array_str(self.tvec, precision=2))

    
class DualQuaternion(object):
    """
    SE(3) rigid transform class that allows compounding of 6-DOF poses
    and provides common transformations that are commonly seen in geometric problems.
        
    """
    def __init__(self, xyzw=[0.,0.,0.,1.], tvec=[0.,0.,0.]):
        """ Initialize a RigidTransform with Quaternion and 3D Position """
        self.real = Quaternion(xyzw)
        self.dual = ( Quaternion(xyzw=[tvec[0], tvec[1], tvec[2], 0.]) * self.real ) * 0.5

    def __repr__(self):
        return 'real: %s dual: %s' % \
            (np.array_str(self.real, precision=2, suppress_small=True), 
             np.array_str(self.dual, precision=2, suppress_small=True))

    def __add__(self, other): 
        return DualQuaternion.from_dq(self.real + other.real, self.dual + other.dual)

    def __mul__(self, other):
        """ 
        Left-multiply RigidTransform with another rigid transform
        
        Two variants: 
           RigidTransform: Identical to oplus operation
           ndarray: transform [N x 3] point set (X_2 = p_21 * X_1)

        """
        if isinstance(other, DualQuaternion):
            return DualQuaternion.from_dq(other.real * self.real, 
                                          other.dual * self.real + other.real * self.dual)
        elif isinstance(other, float):
            return DualQuaternion.from_dq(self.real * other, self.dual * other)
        # elif isinstance(other, nd.array): 
        #     X = np.hstack([other, np.ones((len(other),1))]).T
        #     return (np.dot(self.matrix, X).T)[:,:3]
        else: 
            raise TypeError('__mul__ typeerror {:}'.format(type(other)))
            
    def __rmul__(self, other): 
        raise NotImplementedError('Right multiply not implemented yet!')                    

    def normalize(self): 
        """ Check validity of unit-quaternion norm """
        self.real.normalize()
        self.dual.noramlize()

    def dot(self, other): 
        return self.real.dot(other.real)

    def inverse(self):
        """ Returns a new RigidTransform that corresponds to the inverse of this one """
        qinv = self.dual.inverse()
        return DualQuaternion.from_dq(qinv.q, qinv.rotate(- self.tvec))

    def conjugate(self): 
        return DualQuaternion.from_dq(self.real.conjugate(), self.dual.conjugate())

    def to_matrix(self):
        """ Returns a 4x4 homogenous matrix of the form [R t; 0 1] """
        result = self.rotation.to_matrix()
        result[:3, 3] = self.translation
        return result
    
    def to_Rt(self):
        """ Returns rotation R, and translational vector t """
        T = self.to_matrix()
        return T[:3,:3].copy(), T[:3,3].copy()

    def to_wxyz(self):
        return self.rotation.to_wxyz()

    def to_xyzw(self):
        return self.rotation.to_xyzw()

    @classmethod
    def from_dq(cls, r, d):
        if not isinstance(r, DualQuaternion): 
            raise TypeError('r is not DualQuaternion')
        if not not isinstance(d, DualQuaternion): 
            raise TypeError('d is not DualQuaternion')

        a = cls()
        a.real = r
        a.dual = d
        return a

    @classmethod
    def from_Rt(cls, R, t):
        T = np.eye(4)
        T[:3,:3] = R.copy()
        T[:3,3] = t.copy()
        return cls.from_matrix(T)

    @classmethod
    def from_matrix(cls, T):
        return cls(Quaternion.from_matrix(T), T[:3,3])

    @property
    def rotation(self): 
        return self.real

    @property
    def translation(self): 
        t = self.dual.q * self.real.conjugate()
        return np.array([t.x, t.y, t.z]) * 2.0

    @classmethod
    def identity(cls):
        return cls()
