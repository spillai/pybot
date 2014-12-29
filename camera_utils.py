import cv2

import numpy as np
import numpy.matlib as npm
from scipy import linalg

from bot_utils.db_utils import AttrDict
from bot_geometry.rigid_transform import Quaternion, RigidTransform

kinect_v1_params = AttrDict(
    K_depth = np.array([[576.09757860, 0, 319.5],
                        [0, 576.09757860, 239.5],
                        [0, 0, 1]], dtype=np.float64), 
    K_rgb = np.array([[528.49404721, 0, 319.5],
                      [0, 528.49404721, 239.5],
                      [0, 0, 1]], dtype=np.float64), 
    H = 480, W = 640, 
    shift_offset = 1079.4753, 
    projector_depth_baseline = 0.07214
)

def construct_K(fx=500.0, fy=500.0, cx=319.5, cy=239.5): 
    """
    Create camera intrinsics from focal lengths and focal centers
    """
    K = npm.eye(3)
    K[0,0], K[1,1] = fx, fy
    K[0,2], K[1,2] = cx, cy
    return K

class CameraIntrinsic(object): 
    def __init__(self, K, D=np.zeros(4, dtype=np.float64)): 
        """
        Default init
        """
        self.K = npm.mat(K)                # Calibration matrix.
        self.D = D                         # Distortion
        self.cx, self.cy = K[0,2], K[1,2]  # Camera center.
        self.fx, self.fy = K[0,0], K[1,1]  # Focal length

    @classmethod
    def simluate(cls): 
        """
        Simulate a 640x480 camera with 500 focal length
        """
        return cls.from_calib_params(500., 500., 320., 240.)

    @classmethod
    def from_calib_params(cls, fx, fy, cx, cy): 
        return cls(construct_K(fx, fy, cx, cy))

class CameraExtrinsic(RigidTransform): 
    def __init__(self, R=npm.eye(3), t=npm.zeros(3)):
        """
        Default init
        """
        p = RigidTransform.from_Rt(R, t)
        RigidTransform.__init__(self, xyzw=p.quat.to_xyzw(), tvec=p.tvec)

    @property
    def R(self): 
        return self.quat.to_homogeneous_matrix()[:3,:3]

    @property
    def t(self): 
        return self.tvec

    @classmethod
    def identity(cls): 
        """
        Simulate a camera at identity
        """
        return cls()

    @classmethod
    def simulate(cls): 
        """
        Simulate a camera at identity
        """
        return cls.identity()

class Camera(CameraIntrinsic, CameraExtrinsic): 
    def __init__(self, K, R, t, D=np.zeros(4, dtype=np.float64)): 
        CameraIntrinsic.__init__(self, K, D)
        CameraExtrinsic.__init__(self, R, t)

    @property
    def P(self): 
        Rt = self.to_homogeneous_matrix()[:3]
        return self.K * Rt     # Projection matrix

    @classmethod
    def simulate(cls): 
        """
        Simulate camera intrinsics and extrinsics
        """
        return cls.from_intrinsics_extrinsics(CameraIntrinsic.simluate(), CameraExtrinsic.simulate())

    @classmethod
    def from_intrinsics_extrinsics(cls, intrinsic, extrinsic): 
        return cls(intrinsic.K, extrinsic.R, extrinsic.t, intrinsic.D)

    def project(self, X):
        """
        Project [Nx3] points onto 2-D image plane [Nx2]
        """
        R, t = self.to_Rt()
	rvec,_ = cv2.Rodrigues(R)
	proj,_ = cv2.projectPoints(X, rvec, t, self.K, self.D)
	return proj.reshape((-1,2))

    def factor(self): 
        """
        Factor camera matrix P into K, R, t such that P = K[R|t].
        """
        return cv2.decomposeProjectionMatrix(self.P)

    def center(self): 
        """
        Returns the camera center, the point in space projected to (0, 0) on
        screen.
        """
        if self.cx is None: 
            raise AssertionError('cx, cy is not set')
        return npm.matrix([self.cx, self.cy])

def KinectCamera(R=npm.eye(3), t=npm.zeros(3)): 
    return Camera(kinect_v1_params.K_depth, R, t)

class DepthCamera(CameraIntrinsic): 
    def __init__(self, K, shape=(480,640), skip=1, D=np.zeros(4, dtype=np.float64)):
        CameraIntrinsic.__init__(self, K, D)

        # Retain image shape
        self.shape = shape
        self.skip = skip

        # Construct mesh for quick reconstruction
        self._build_mesh(shape=shape)
        
    def _build_mesh(self, shape): 
        H, W = shape
        xs,ys = np.arange(0,W), np.arange(0,H);
        fx_inv = 1.0 / self.fx;

        self.xs = (xs-self.cx) * fx_inv
        self.xs = self.xs[::self.skip] # skip pixels
        self.ys = (ys-self.cy) * fx_inv
        self.ys = self.ys[::self.skip] # skip pixels

        self.xs, self.ys = np.meshgrid(self.xs, self.ys);

    def reconstruct(self, depth): 
        assert(depth.shape == self.xs.shape)
        return np.dstack([self.xs * depth, self.ys * depth, depth])

def KinectDepthCamera(K=kinect_v1_params.K_depth, shape=(480,640)): 
    return DepthCamera(K=K, shape=shape)

def compute_fundamental(x1, x2, method=cv2.FM_RANSAC): 
    """
    Computes the fundamental matrix from corresponding points x1, x2 using
    the 8 point algorithm.

    Options: 
    CV_FM_7POINT for a 7-point algorithm.  N = 7
    CV_FM_8POINT for an 8-point algorithm.  N >= 8
    CV_FM_RANSAC for the RANSAC algorithm.  N >= 8
    CV_FM_LMEDS for the LMedS algorithm.  N >= 8"
    """
    assert(x1.shape == x2.shape)
    F, mask = cv2.findFundamentalMat(x1, x2, method)
    return F, mask

def compute_epipole(F):
    """ Computes the (right) epipole from a 
        fundamental matrix F. 
        (Use with F.T for left epipole.) """
    
    # return null space of F (Fx=0)
    U,S,V = linalg.svd(F)
    e = V[-1]
    return e/e[2]

def compute_essential(F, K): 
    """ Compute the Essential matrix, and R1, R2 """
    return K.T * npm.mat(F) * K



# def plot_epipolar_line(im, F, x, epipole=None, show_epipole=True):
#   """
#   Plot the epipole and epipolar line F * x = 0.
#   """
#   import pylab

#   m, n = im.shape[:2]
#   line = numpy.dot(F, x)

#   t = numpy.linspace(0, n, 100)
#   lt = numpy.array([(line[2] + line[0] * tt) / (-line[1]) for tt in t])

#   ndx = (lt >= 0) & (lt < m)
#   pylab.plot(t[ndx], lt[ndx], linewidth=2)

#   if show_epipole:
#     if epipole is None:
#       epipole = compute_right_epipole(F)
#     pylab.plot(epipole[0] / epipole[2], epipole[1] / epipole[2], 'r*')
