import numpy as np
import transformations as tf
from ros_quaternion import Quaternion
from tf_tests import tf_isequal
np.set_printoptions(precision=2, suppress=True, threshold='nan', linewidth=160)

def normalize_vec(v): 
    return v * 1.0 / np.linalg.norm(v)

# Construct a reference frame with two vectors
# TODO: checks for degenerate cases
def tf_construct(vec1,vec2): 
    """Align vx along v1, and construct [vx,vy,vz] as follows: 
    vx = v1
    vz = v1 x v2
    vy = vz x vx
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
    

# Quaternion quat within this class is interpreted as xyzw, 
# similar to ros_quaternion.py file
class RigidTransform(object):
    def __init__(self, xyzw=[0.,0.,0.,1.], tvec=[0.,0.,0.]):
        self.quat = Quaternion(xyzw)
        self.tvec = np.array(tvec)

    def __repr__(self):
        return 'rpy: %s tvec: %s' % (self.quat.to_roll_pitch_yaw(), self.tvec)
        # return 'quat: %s, tvec: %s' % (self.quat, self.tvec)

    def inverse(self):
        """ returns a new RigidTransform that corresponds to the inverse of this one """
        qinv = self.quat.inverse()
        return RigidTransform(qinv, qinv.rotate(- self.tvec))

    def to_homogeneous_matrix(self):
        result = self.quat.to_homogeneous_matrix()
        result[:3, 3] = self.tvec
        return result

    def to_Rt(self): 
        T = self.to_homogeneous_matrix()
        return T[:3,:3].copy(), T[:3,3].copy()

    # def interpolate(self, other_transform, this_weight):
    #     assert this_weight >= 0 and this_weight <= 1
    #     t = self.tvec * this_weight + other_transform.tvec * (1 - this_weight)
    #     r = self.quat.interpolate(other_transform.quat, this_weight)
    #     return RigidTransform(r, t)

    # def ominus(self, other): 
    #     if not isinstance(other, RigidTransform): 
    #         raise TypeError("Type inconsistent")
    #     oinv = other.inverse()
    #     return oinv.oplus(self)

    # left multiply
    def __mul__(self, other):
        if isinstance(other, RigidTransform):
            t = self.quat.rotate(other.tvec) + self.tvec
            r = self.quat * other.quat

            # assert(tf_isequal(np.dot(self.to_homogeneous_matrix(), 
            #                          other.to_homogeneous_matrix()), 
            #        RigidTransform(r,t).to_homogeneous_matrix()))

            return RigidTransform(r, t)

        # multiply with N x 3
        else:
            X = np.hstack([other, np.ones((len(other),1))]).T
            return (np.dot(self.to_homogeneous_matrix(), X).T)[:,:3]
            # olen = len(other)
            # if olen == 3:
            #     r = np.array(self.quat.rotate(other))
            #     return r + self.tvec
            # elif olen == 4:
            #     return np.dot(self.to_homogeneous_matrix(), other)
        # else:
        #         raise ValueError()
    def __rmul__(self, other): 
        raise AssertionError()                    

    def oplus(self, other): 
        if not isinstance(other, RigidTransform): 
            raise TypeError("Type inconsistent", type(other), other.__class__)
        t = self.quat.rotate(other.tvec) + self.tvec
        r = self.quat * other.quat
        return RigidTransform(r, t)

    def to_roll_pitch_yaw_x_y_z(self, axes='rxyz'):
        r, p, y = self.quat.to_roll_pitch_yaw(axes=axes)
        return np.array((r, p, y, self.tvec[0], self.tvec[1], self.tvec[2]))

    def rotate_vec(self, v): 
        if v.ndim == 2 and v.shape[0] > 1: 
            return np.vstack(map(lambda v_: self.quat.rotate(v_), v))
        else: 
            assert(v.ndim == 1 or (v.ndim == 2 and v.shape[0] == 1))
            return self.quat.rotate(v)

    @staticmethod
    def from_bot_core_pose_t(pose): 
        return RigidTransform(Quaternion.from_wxyz(pose.orientation), pose.pos)

    @staticmethod
    def from_roll_pitch_yaw_x_y_z(r, p, yaw, x, y, z, axes='rxyz'):
        q = Quaternion.from_roll_pitch_yaw(r, p, yaw, axes=axes)
        return RigidTransform(q, (x, y, z))

    @staticmethod
    def from_Rt(R, t):
        T = np.eye(4)
        T[:3,:3] = R.copy();
        return RigidTransform(Quaternion.from_homogenous_matrix(T), t)

    @staticmethod
    def from_homogenous_matrix(T):
        return RigidTransform(Quaternion.from_homogenous_matrix(T), T[:3,3])

    @staticmethod
    def from_triad(pos, v1, v2):
        # print v1, v2, type(v1)
        return RigidTransform.from_homogenous_matrix(tf_compose(tf_construct(v1, v2), pos))

    @staticmethod
    def from_angle_axis(angle, axis, tvec): 
        return RigidTransform(Quaternion.from_angle_axis(angle, axis), tvec)

    def wxyz(self):
        return self.quat.to_wxyz()

    def xyzw(self):
        return self.quat.to_xyzw()

    def translation(self):
        return self.tvec

    @staticmethod
    def identity():
        return RigidTransform()

# class Pose(RigidTransform): 
#     def __init__(self, pid, rotation_quat, translation_vec):
#         RigidTransform.__init__(self, rotation_quat, translation_vec)
#         self.id = pid

#     def __repr__(self): 
#         return 'Pose ID: %i, quat: %s, rpy: %s tvec: %s' % (self.id, 
#                                                             self.quat, self.quat.to_roll_pitch_yaw(), self.tvec)

#     @staticmethod
#     def from_triad(pid, pos, v1, v2):
#         rt = RigidTransform.from_triad(pos,v1,v2)
#         return Pose(pid, rt.quat, rt.tvec)

#     @staticmethod
#     def from_rigid_transform(pid, rt):
#         # Quat: [x y z w]
#         return Pose(pid, rt.quat, rt.tvec)

#     # @staticmethod
#     # def from_vec(pid, vec):
#     #     # print vec[-4:], vec[:3]
#     #     return Pose(pid, vec[-4:], vec[:3])
        
#     # def to_vec(self):
#     #     return np.hstack([self.tvec, self.quat.q])


import random
def make_random_transform(t):
    q_wxyz = [ random.random(), random.random(), random.random(), random.random() ]
    qmag = np.sqrt(sum([x*x for x in q_wxyz]))
    q_wxyz = [ x / qmag for x in q_wxyz ]
    translation = [ random.uniform(-t, t), random.uniform(-t, t), random.uniform(-t, t) ]
    return RigidTransform(Quaternion.from_wxyz(q_wxyz), translation)


if __name__ == "__main__":

    q = Quaternion.identity()
    t = [ 1, 2, 3 ]
    m = RigidTransform(q, t)
    print "m"
    print m.to_homogeneous_matrix()
    print "--------------------------"

    q2 = Quaternion.from_roll_pitch_yaw(np.pi / 4, 0, 0)
    t2 = [ 0, 0, 0 ]
    m2 = RigidTransform(q2, t2)
    print "m2"
    print m2.to_homogeneous_matrix()

    print "--------------------------"
    m3 = m * m2
    print "m * m2"
    print m3.to_homogeneous_matrix()
    print np.dot(m.to_homogeneous_matrix(), m2.to_homogeneous_matrix())
    print "--------------------------"

    m4 = m2 * m
    print "m * m2"
    print m4.to_homogeneous_matrix()
    print np.dot(m2.to_homogeneous_matrix(), m.to_homogeneous_matrix())
    print "--------------------------"

    print "Testing inverse"
    identity = np.identity(4)
    for unused in range(100):
        # generate a bunch of random rigid body transforms, then compose them and apply their inverses
        # the result should be the identity transform
        num_transforms = random.randint(0, 10)
        ms = [ make_random_transform() for unused in range(num_transforms) ]
        inverses = [ m.inverse() for m in ms ]
        inverses.reverse()
        r = RigidTransform.identity()
        for m in ms + inverses:
            r *= m
        errs = (identity - r.to_homogeneous_matrix()).flatten().tolist()[0]
        sse = np.dot(errs, errs)
        assert sse < 1e-10
    print "OK"
    

    print 'check inverse'
    for _ in range(10): 
        m = make_random_transform()
        t = m.to_homogeneous_matrix()
        tinv = np.linalg.inv(t) # tf_compose(t[:3,:3].T, np.dot(t[:3,:3],-t[:3,3]))
        print tinv, '\n', tf_compose(t[:3,:3].T, np.dot(t[:3,:3],-t[:3,3]))
        assert(tf_isequal(tinv, m.inverse().to_homogeneous_matrix()))


    print "Testing composition"
    t = RigidTransform.identity()
    m = np.identity(4)
    for unused in range(1000):
#        print "===="
#        print t.to_homogeneous_matrix()
#        print m

        n = make_random_transform()
        # print 't: ', t.to_homogeneous_matrix()
        # print 'm: ', m
#        n.quat = Quaternion(1, 0, 0, 0)
#        print "applying "
#        print n.to_homogeneous_matrix()


        t = t * n
        m = np.dot(m, n.to_homogeneous_matrix())
        errs = (t.to_homogeneous_matrix() - m).flatten().tolist()[0]
        # print errs

        sse = np.dot(errs, errs)


#        print "--"
#        print t.to_homogeneous_matrix()
#        print m
        assert (sse < 1e-10)

    for unused in range(1000):
        t1 = make_random_transform()
        t2 = make_random_transform()
        t1.to_homogeneous_matrix


    # Composition, and inverse
    a = make_random_transform()
    b = make_random_transform()
    c = make_random_transform()

    ba = b * a

    # huh?
    print np.dot(b, a), b * a

    # retrieve relative pose 
    assert(tf_isequal((ba * a.inverse()).to_homogeneous_matrix(), b.to_homogeneous_matrix()))

    print "OK"
