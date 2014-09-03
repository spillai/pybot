import math
import numpy as np
import ros_transformations as tf
from tf_tests import tf_isequal
np.set_printoptions(precision=2, suppress=True, threshold='nan', linewidth=160)

# q = [x, y, z, w]
class Quaternion:
    def __init__ (self, q=[0,0,0,1]):
        if isinstance(q, Quaternion): 
            self.q = q.q.copy()
        else: 
            try: 
                self.q = np.array(q, np.float64)
            except:
                raise Exception("TypeError")
                # raise TypeError ("invalid initializer")            

        norm = np.linalg.norm(self.q)
        assert abs(norm-1) < 1e-2
        if abs(norm - 1) > 1e-2:
            self.q /= norm

    @staticmethod
    def identity():
        return Quaternion()

    @staticmethod
    def from_wxyz(q): 
        return Quaternion(np.roll(q, shift=-1))

    @staticmethod
    def from_xyzw(q): 
        return Quaternion(q)

    def to_wxyz(self): 
        q = np.roll(self.q, shift=1)
        return q

    def to_xyzw(self): 
        return self.q
    
    def __mul__(self, other): 
        return tf.quaternion_multiply(self.q, other.q)

    def __getitem__ (self, i):
        return self.q[i]

    def __repr__ (self):
        return '%s' % self.q
        # return 'w:%s, x:%s, y:%s, z:%s' % (self.q[3], self.q[0], self.q[1], self.q[2])

    def rotate(self, v):
        # repr. (TEMP. HACK) now in [w,x,y,z] form : originally [x,y,z,w]
        q = np.roll(self.q, shift=1)

        ab  =  q[0]*q[1]
        ac  =  q[0]*q[2]
        ad  =  q[0]*q[3]
        nbb = -q[1]*q[1]
        bc  =  q[1]*q[2]
        bd  =  q[1]*q[3]
        ncc = -q[2]*q[2]
        cd  =  q[2]*q[3]
        ndd = -q[3]*q[3]

        return np.array((2*( (ncc + ndd)*v[0] + (bc -  ad)*v[1] + (ac + bd)*v[2] ) + v[0],
                         2*( (ad +  bc)*v[0] + (nbb + ndd)*v[1] + (cd - ab)*v[2] ) + v[1],
                         2*( (bd -  ac)*v[0] + (ab +  cd)*v[1] + (nbb + ncc)*v[2] ) + v[2]))

    def rotate_rev (self, vector):
        q = np.roll(self.q, shift=1)

        b = np.array((0, v[0], v[1], v[2]))
        a = np.array((b[0]*q[0] - b[1]*q[1] - b[2]*q[2] - b[3]*q[3],
             b[0]*q[1] + b[1]*q[0] + b[2]*q[3] - b[3]*q[2],
             b[0]*q[2] - b[1]*q[3] + b[2]*q[0] + b[3]*q[1],
             b[0]*q[3] + b[1]*q[2] - b[2]*q[1] + b[3]*q[0]))
        b[0] = q[0]
        b[1:] = -q[1:]
        return np.array((b[0]*a[1] + b[1]*a[0] + b[2]*a[3] - b[3]*a[2],
                         b[0]*a[2] - b[1]*a[3] + b[2]*a[0] + b[3]*a[1],
                         b[0]*a[3] + b[1]*a[2] - b[2]*a[1] + b[3]*a[0]))

    def inverse(self):
        return Quaternion(tf.quaternion_inverse(self.q))

    @staticmethod
    def from_roll_pitch_yaw (roll, pitch, yaw, axes='rxyz'):
        return Quaternion(tf.quaternion_from_euler(roll, pitch, yaw, axes=axes))

    @staticmethod
    def from_rpy (rpy, axes='rxyz'):
        return Quaternion.from_roll_pitch_yaw(rpy[0], rpy[1], rpy[2], axes=axes)

    @staticmethod
    def from_angle_axis(theta, axis):
        x, y, z = axis
        norm = math.sqrt(x*x + y*y + z*z)
        if 0 == norm:
            return Quaternion([0, 0, 0, 1])
        t = math.sin(theta/2) / norm;
        return Quaternion([x*t, y*t, z*t, math.cos(theta/2)])

    def to_roll_pitch_yaw (self, axes='rxyz'):
        return np.array(tf.euler_from_quaternion(self.q, axes=axes))

    def to_angle_axis(self):
        q = np.roll(self.q, shift=1)
        halftheta = math.acos(q[0])
        if abs(halftheta) < 1e-12:
            return 0, np.array((0, 0, 1))
        else:
            theta = halftheta * 2
            axis = np.array(q[1:4]) / math.sin(halftheta)
            return theta, axis

    @staticmethod
    def from_matrix(matrix):
        return tf.quaternion_from_matrix(matrix)

    @staticmethod
    def from_homogenous_matrix(matrix):
        return Quaternion.from_matrix(matrix)

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
        t = q.to_matrix()
        assert(tf_isequal(q.inverse().to_matrix(), t.T))

    print "OK"
