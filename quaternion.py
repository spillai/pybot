"""
General-purpose class for quaternion / rotation transformations.
"""
# Author: Sudeep Pillai <spillai@csail.mit.edu>
# License: MIT

import math
import numpy as np
import transformations as tf

###############################################################################
class Quaternion(object):
    """
    Generic Quaternion class
       : (qx, qy, qz, qw)
    
    """
    def __init__ (self, q=[0,0,0,1]):
        if isinstance(q, Quaternion): 
            self.q = q.q.copy()
        else:
            try: 
                self.q = np.array(q, np.float64)
            except:
                raise TypeError("Quaternion can not be initialized from {:}".format(type(q)))
                            
        
        self.normalize()

    def normalize(self): 
        """ Check validity of unit-quaternion norm """
        norm = np.linalg.norm(self.q)
        if abs(norm-1) > 1e-2: 
            raise RuntimeError('Norm computed is %5.3f' % norm)
        if abs(norm - 1) > 1e-2:
            self.q /= norm

    def dot(self, other): 
        return self.q.dot(other.q)

    @classmethod
    def identity(cls):
        return cls()

    @classmethod
    def from_wxyz(cls, q): 
        return cls(np.roll(q, shift=-1))

    @classmethod
    def from_xyzw(cls, q): 
        return cls(q)

    @property
    def x(self): 
        return self.q[0]

    @property
    def y(self): 
        return self.q[1]

    @property
    def z(self): 
        return self.q[2]

    @property
    def w(self): 
        return self.q[3]

    def to_wxyz(self): 
        q = np.roll(self.q, shift=1)
        return q

    def to_xyzw(self): 
        """ Return (x,y,z,w) representation """
        return self.q
    
    def __mul__(self, other):
        """ Multiply quaternion with another """
        if isinstance(other, float): 
            return Quaternion(self.q * other)
        elif isinstance(other, Quaternion): 
            return tf.quaternion_multiply(self.q, other.q)
        else: 
            raise TypeError('Quaternion multiply error')

    def __getitem__ (self, i):
        return self.q[i]

    def __repr__ (self):
        return '%s' % self.q

    def rotate(self, v):
        """ Rotate a vector with this quaternion in reverse """
        qx, qy, qz, qw = self.q

        ab  =  qw*qx
        ac  =  qw*qy
        ad  =  qw*qz
        nbb = -qx*qx
        bc  =  qx*qy
        bd  =  qx*qz
        ncc = -qy*qy
        cd  =  qy*qz
        ndd = -qz*qz

        return np.array((2*( (ncc + ndd)*v[0] + (bc -  ad)*v[1] + (ac + bd)*v[2] ) + v[0],
                         2*( (ad +  bc)*v[0] + (nbb + ndd)*v[1] + (cd - ab)*v[2] ) + v[1],
                         2*( (bd -  ac)*v[0] + (ab +  cd)*v[1] + (nbb + ncc)*v[2] ) + v[2]))

    def rotate_rev (self, vector):
        """ Rotate a vector with this quaternion in reverse """
        qx, qy, qz, qw = q
        b = np.array((0, v[0], v[1], v[2]))
        a = np.array((b[0]*qw - b[1]*qx - b[2]*qy - b[3]*qz,
             b[0]*qx + b[1]*qw + b[2]*qz - b[3]*qy,
             b[0]*qy - b[1]*qz + b[2]*qw + b[3]*qx,
             b[0]*qz + b[1]*qy - b[2]*qx + b[3]*qw))
        b[0] = qw
        b[1:] = -q[:3]
        return np.array((b[0]*a[1] + b[1]*a[0] + b[2]*a[3] - b[3]*a[2],
                         b[0]*a[2] - b[1]*a[3] + b[2]*a[0] + b[3]*a[1],
                         b[0]*a[3] + b[1]*a[2] - b[2]*a[1] + b[3]*a[0]))

    def inverse(self):
        """ Invert rotation """
        return Quaternion(tf.quaternion_inverse(self.q))

    def conjugate(self):
        """ Quaternion conjugate """
        return Quaternion(tf.quaternion_conjugate(self.q))

    @classmethod
    def from_roll_pitch_yaw (cls, roll, pitch, yaw, axes='rxyz'):
        """ Construct Quaternion from axis-angle representation """
        return cls(tf.quaternion_from_euler(roll, pitch, yaw, axes=axes))

    @classmethod
    def from_rpy (cls, rpy, axes='rxyz'):
        """ Construct Quaternion from Euler angle representation """
        return cls.from_roll_pitch_yaw(rpy[0], rpy[1], rpy[2], axes=axes)

    @classmethod
    def from_angle_axis(cls, theta, axis):
        """ Construct Quaternion from axis-angle representation """
        x, y, z = axis
        norm = math.sqrt(x*x + y*y + z*z)
        if 0 == norm:
            return cls([0, 0, 0, 1])
        t = math.sin(theta/2) / norm;
        return cls([x*t, y*t, z*t, math.cos(theta/2)])

    def to_roll_pitch_yaw (self, axes='rxyz'):
        """ Return Euler angle with XYZ convention """
        return np.array(tf.euler_from_quaternion(self.q, axes=axes))

    def to_angle_axis(self):
        """ Return axis-angle representation """
        q = np.roll(self.q, shift=1)
        halftheta = math.acos(q[0])
        if abs(halftheta) < 1e-12:
            return 0, np.array((0, 0, 1))
        else:
            theta = halftheta * 2
            axis = np.array(q[1:4]) / math.sin(halftheta)
            return theta, axis

    @classmethod
    def from_matrix(cls, matrix):
        return tf.quaternion_from_matrix(matrix)

    @classmethod
    def from_homogenous_matrix(cls, matrix):
        return cls.from_matrix(matrix)

    def to_matrix(self):
        return tf.quaternion_matrix(self.q)[:3,:3]

    def to_homogeneous_matrix(self):
        return tf.quaternion_matrix(self.q)

    def interpolate(self, other, this_weight):
        q0, q1 = np.roll(self.q, shift=1), np.roll(other.q, shift=1)
        u = 1 - this_weight
        assert(u >= 0 and u <= 1)
        cos_omega = np.dot(q0, q1)

        if cos_omega < 0:
            result = -q0[:]
            cos_omega = -cos_omega
        else:
            result = q0[:]

        cos_omega = min(cos_omega, 1)

        omega = math.acos(cos_omega)
        sin_omega = math.sin(omega)
        a = math.sin((1-u) * omega)/ sin_omega
        b = math.sin(u * omega) / sin_omega

        if abs(sin_omega) < 1e-6:
            # direct linear interpolation for numerically unstable regions
            result = result * this_weight + q1 * u
            result /= math.sqrt(np.dot(result, result))
        else:
            result = result*a + q1*b
        return Quaternion(np.roll(result, shift=-1))


###############################################################################
if __name__ == "__main__":
    import random
    q = Quaternion.from_roll_pitch_yaw (0, 0, 2 * math.pi / 16)
    v = [ 1, 0, 0 ]
    print 'init: ', v
    for i in range (16):
        t = np.dot(q.to_matrix(), v)
        v = q.rotate (v)

        # Check whether quats, and rotation matrix mult returns same result
        assert(np.all(v - t < 1e-12))

        # print v

    q2 = Quaternion.from_roll_pitch_yaw(0, 0, 0)
    rpy_start = np.array(q.to_roll_pitch_yaw())
    rpy_goal = np.array(q2.to_roll_pitch_yaw())
    print "interpolate from ", q2.to_roll_pitch_yaw(), " to ", q.to_roll_pitch_yaw()
    for i in range(101):
        alpha = i / 100.
        qinterp = q2.interpolate(q, alpha)
        rpy_interp = np.array(qinterp.to_roll_pitch_yaw())
        rpy_expected = (rpy_goal * alpha + rpy_start * (1 - alpha))
        err = rpy_expected - rpy_interp
        for k in [ 0, 1, 2 ]:
            print 'err: ', err[k]
            assert abs(err[k]) < 1e-12

    def mod2pi_positive(vin):
        q = vin / (2*np.pi) + 0.5
        qi = int(q)
        return vin - qi*2*np.pi

    def mod2pi(vin):
        if (vin < 0):
            return -mod2pi_positive(-vin)
        return mod2pi_positive(vin)

    def mod2pi_ref(ref, vin):
        return ref + mod2pi(vin - ref)

    print "testing angle-axis conversion"
    for unused in range(100):
        theta = random.uniform(-np.pi, np.pi)
        axis = np.array([ random.random(), random.random(), random.random() ])
        axis /= np.linalg.norm(axis)
        q = Quaternion.from_angle_axis(theta, axis)
        theta_check, axis_check = q.to_angle_axis()
        if np.dot(axis, axis_check) < 0:
            theta_check *= -1
            axis_check *= -1
        theta_check = mod2pi_ref(theta, theta_check)
        dtheta = theta_check - theta
        daxis = axis - axis_check
        assert abs(dtheta) < 1e-12
        assert np.linalg.norm(daxis) < 1e-9


    def make_random_quaternion(): 
        q_wxyz = [ random.random(), random.random(), random.random(), random.random() ]
        qmag = np.sqrt(sum([x*x for x in q_wxyz]))
        q_wxyz = [ x / qmag for x in q_wxyz ]
        return Quaternion.from_wxyz(q_wxyz)

    print 'rotate by random matrix'
    for _ in range(100): 
        q = make_random_quaternion()
        t = np.dot(q.to_matrix(), v)
        v = q.rotate (v)    
        assert(np.all(v - t < 1e-12))

    print 'Check inverse'
    for _ in range(100): 
        q = make_random_quaternion()
        t = q.to_homogeneous_matrix()
        assert(tf.is_same_transform(q.inverse().to_homogeneous_matrix(), t.T))

    print "OK"
