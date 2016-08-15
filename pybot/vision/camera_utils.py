# Author: Sudeep Pillai <spillai@csail.mit.edu>
# License: MIT

import cv2

import numpy as np
from numpy.linalg import det, norm
import numpy.matlib as npm
from scipy import linalg

from pybot.vision.color_utils import get_color_by_label
from pybot.vision.image_utils import to_color
from pybot.utils.db_utils import AttrDict
from pybot.geometry.rigid_transform import Quaternion, RigidTransform

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

def colvec(vec): 
    """ Convert to column vector """
    return vec.reshape(-1,1)

def rowvec(vec): 
    """ Convert to row vector """
    return vec.reshape(1,-1)

def unproject(a):
  """ Homogenize vector """
  return np.append(a, 1)

def project(a):
  """ De-homogenize vector """
  return a[:-1]/float(a[-1])

def unproject_points(pts): 
    assert(pts.ndim == 2 and pts.shape[1] == 2)
    return np.hstack([pts, colvec(np.ones(len(pts)))])

def project_points(pts): 
    z = colvec(pts[:,2])
    return pts[:,:2] / z

def sampson_error(F, pts1, pts2): 
    """
    Computes the sampson error for F, and points pts1, pts2. Sampson
    error is the first order approximation to the geometric error.
    Remember that this is a squared error.
   
    (x'^{T} * F * x)^2
    -----------------
    (F * x)_1^2 + (F * x)_2^2 + (F^T * x')_1^2 + (F^T * x')_2^2

    where (F * x)_i^2 is the square of the i-th entry of the vector Fx
    """
    x1, x2 = unproject_points(pts1).T, unproject_points(pts2).T
    Fx1 = np.dot(F, x1)
    Fx2 = np.dot(F, x2)
       
    # Sampson distance as error measure
    denom = Fx1[0]**2 + Fx1[1]**2 + Fx2[0]**2 + Fx2[1]**2
    return ( np.diag(x1.T.dot(Fx2)) )**2 / denom 
    
def get_baseline(fx, baseline=None, baseline_px=None): 
    """
    Retrieve baseline / baseline(in pixels) using the focal length
    """
    assert(baseline is not None or baseline_px is not None)
    if baseline is not None: 
        baseline_px = baseline * fx
    elif baseline_px is not None: 
        baseline = baseline_px / fx
    else: 
        raise AssertionError('Ambiguous baseline and baseline_px values. Provide either, not both')
    return baseline, baseline_px

def triangulate_points(cam1, pts1, cam2, pts2): 
    """
    Triangulate 2D corresponding points using 2-views 
    """
    assert(isinstance(cam1, Camera) and isinstance(cam2, Camera))
    assert(isinstance(pts1, np.ndarray) and isinstance(pts2, np.ndarray))
    X = cv2.triangulatePoints(cam1.P, cam2.P, pts1.reshape(-1,1,2), pts2.reshape(-1,1,2)).T 
    return X[:,:3] / colvec(X[:,3])
    
def construct_K(fx=500.0, fy=500.0, cx=319.5, cy=239.5): 
    """
    Create camera intrinsics from focal lengths and focal centers
    """
    K = npm.eye(3)
    K[0,0], K[1,1] = fx, fy
    K[0,2], K[1,2] = cx, cy
    return K

def construct_D(k1=0, k2=0, k3=0, p1=0, p2=0): 
    """
    Create camera distortion params
    """
    return np.float64([k1,k2,p1,p2,k3])

def undistort_image(im, K, D): 
    """
    Optionally: 
        newcamera, roi = cv2.getOptimalNewCameraMatrix(self.K, self.D, (W,H), 0) 
    """
    H,W = im.shape[:2]
    Kprime, roi = cv2.getOptimalNewCameraMatrix(K, D, (W,H), 1, (W,H))
    return cv2.undistort(im, K, D, None, K)

# def camera_from_P(P): 
    
def get_calib_params(fx, fy, cx, cy, baseline=None, baseline_px=None): 
    raise RuntimeError('Deprecated, see camera_utils.StereoCamera')

    baseline, baseline_px = get_baseline(fx, baseline=baseline, baseline_px=baseline_px)
    q43 = -1/baseline
    P0 = np.float32([fx, 0.0, cx, 0.0, 
                     0.0, fy, cy, 0.0, 
                     0.0, 0.0, 1.0, 0.0]).reshape((3,4))
    P1 = np.float32([fx, 0.0, cx, -baseline_px, 
                     0.0, fy, cy, 0.0, 
                     0.0, 0.0, 1.0, 0.0]).reshape((3,4))

    K0, K1 = P0[:3,:3], P1[:3,:3]

    R0, R1 = np.eye(3), np.eye(3)
    T0, T1 = np.zeros(3), np.float32([baseline, 0, 0])
    # T0, T1 = np.zeros(3), np.float32([-baseline_px, 0, 0])

    Q = np.float32([[-1, 0, 0, cx],
                    [0, -1, 0, cy], 
                    [0, 0, 0, -fx], 
                    [0, 0, q43,0]])

    D0, D1 = np.zeros(5), np.zeros(5)
    return AttrDict(R0=R0, R1=R1, K0=K0, K1=K1, P0=P0, P1=P1, Q=Q, T0=T0, T1=T1, 
                    D0=D0, D1=D1, fx=fx, fy=fy, cx=cx, cy=cy, baseline=baseline, baseline_px=baseline * fx)

class CameraIntrinsic(object): 
    def __init__(self, K, D=np.zeros(5, dtype=np.float64), shape=None): 
        """
        K: Calibration matrix
        D: Distortion
        shape: Image Size (H,W,C): (480,640,3)
        """
        self.K = K
        self.D = D
        self.shape = np.int32(shape) if shape is not None else None
        
    def __repr__(self): 
        return 'CameraIntrinsic =======>\n K:\n{:}\n D: {:}\n fx: {:}, fy: {:}, '\
            'cx: {:}, cy: {:}, shape: {:}'.format(
                np.array_str(np.array(self.K), precision=2, suppress_small=True), 
                np.array_str(self.D, precision=2),
                self.fx, self.fy, self.cx, self.cy, self.shape) 

    @classmethod
    def simulate(cls): 
        """
        Simulate a 640x480 camera with 500 focal length
        """
        return cls.from_calib_params(500., 500., 320., 240., shape=(480,640))

    @classmethod
    def from_calib_params(cls, fx, fy, cx, cy, k1=0, k2=0, k3=0, p1=0, p2=0, shape=None): 
        return cls(construct_K(fx, fy, cx, cy), 
                   D=construct_D(k1, k2, k3, p1, p2), shape=shape)

    @classmethod
    def from_calib_params_fov(cls, fov, cx, cy, D=np.zeros(5, dtype=np.float64), shape=None): 
        return cls(construct_K(cx / np.tan(fov), cy / np.tan(fov), cx, cy), D=D, shape=shape)

    @property
    def fx(self): 
        return self.K[0,0]

    @property
    def fy(self): 
        return self.K[1,1]

    @property
    def cx(self): 
        return self.K[0,2]

    @property
    def cy(self): 
        return self.K[1,2]

    @property
    def k1(self): 
        return self.D[0]

    @property
    def k2(self): 
        return self.D[1]

    @property
    def k3(self): 
        return self.D[4]

    @property
    def p1(self): 
        return self.D[2]

    @property
    def p2(self): 
        return self.D[3]

    @property
    def fov(self): 
        """
        Returns the field of view for each axis
        """
        return np.float32([np.arctan(self.shape[1] * 0.5 / self.fx), 
                           np.arctan(self.shape[0] * 0.5 / self.fy)]) * 2.0

    def scaled(self, scale): 
        """
        Returns the scaled intrinsics of the camera, that results 
        from resizing the image by the appropriate scale parameter
        """
        shape = np.int32(self.shape * scale) if self.shape is not None else None
        return CameraIntrinsic.from_calib_params(self.fx * scale, self.fy * scale, 
                                                 self.cx * scale, self.cy * scale, 
                                                 k1=self.k1, k2=self.k2, k3=self.k3, 
                                                 p1=self.p1, p2=self.p2, 
                                                 shape=shape)

    def ray(self, pts, undistort=True, rotate=False): 
        """
        Returns the ray corresponding to the points. 
        Optionally undistort (defaults to true), and 
        rotate ray to the camera's viewpoint 
        """
        upts = self.undistort_points(pts) if undistort else pts
        ret = unproject_points(
            np.hstack([ (colvec(upts[:,0])-self.cx) / self.fx, (colvec(upts[:,1])-self.cy) / self.fy ])
        )
        if rotate: 
            ret = self.extrinsics.rotate_vec(ret)
        return ret

    def reconstruct(self, xyZ, undistort=True): 
        """
        Reproject to 3D with calib params
        """
        Z = colvec(xyZ[:,2])
        return self.ray(xyZ[:,:2], undistort=undistort) * Z
        
    def undistort(self, im): 
        return undistort_image(im, self.K, self.D)

    def undistort_points(self, pts): 
        """
        Undistort image points using camera matrix, and distortion coeffs
 
        Have to provide P matrix for appropriate scaling
        http://code.opencv.org/issues/1964#note-2
        """
        out = cv2.undistortPoints(pts.reshape(-1,1,2).astype(np.float32), self.K, self.D, P=self.K)
        return out.reshape(-1,2) 
        
    def undistort_debug(self, im=None): 
        if im is None: 
            im = np.zeros(shape=self.shape, dtype=np.uint8)
        im[::20, :] = 128
        im[:, ::20] = 128
        return self.undistort(im)

    def save(self, filename):
        try: 
            height, width = self.shape[:2]
        except: 
            height, width = '', ''
        AttrDict(
            fx=float(self.fx), fy=float(self.fy), 
            cx=float(self.cx), cy=float(self.cy), 
            k1=float(self.D[0]), k2=float(self.D[1]), k3=float(self.D[4]), p1=float(self.D[2]), p2=float(self.D[3]),
            width=int(width), height=int(height)
        ).save_yaml(filename)
        
    @classmethod
    def load(cls, filename):
        db = AttrDict.load_yaml(filename)
        shape = np.int32([db.width, db.height]) if hasattr(db, 'width') and hasattr(db, 'height') else None 
        return cls.from_calib_params(db.fx, db.fy, db.cx, db.cy, 
                                     k1=db.k1, k2=db.k2, k3=db.k3, p1=db.p1, p2=db.p2, shape=shape)
    
class CameraExtrinsic(RigidTransform): 
    def __init__(self, R=npm.eye(3), t=npm.zeros(3)):
        """
        Pose is defined as p_cw (pose of the world wrt camera)
        """
        p = RigidTransform.from_Rt(R, t)
        RigidTransform.__init__(self, xyzw=p.quat.to_xyzw(), tvec=p.tvec)
        self.__cached_inverse = None

    def inverse(self): 
        if self.__cached_inverse is None: 
            self.__cached_inverse = super(CameraExtrinsic, self).inverse()
        return self.__cached_inverse

    def __repr__(self): 
        return 'CameraExtrinsic =======>\npose = {:}'.format(RigidTransform.__repr__(self))

    def c2w(self, X): 
        """
        Transform points in camera coordinates to world
        """
        return self * X

    def w2c(self, X): 
        """
        Transform points from world coordinates to camera
        """
        return self.inverse() * X
        
    @classmethod
    def from_rigid_transform(cls, p): 
        R, t = p.to_Rt()
        return cls(R, t)
    
    @property
    def R(self): 
        return self.quat.R

    @property
    def Rt(self): 
        return self.matrix[:3]

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

    @property
    def o2w(self): 
        return self

    @property
    def w2o(self): 
        return self.inverse()

    def save(self, filename): 
        R, t = p.to_Rt()
        AttrDict(R=R.tolist(), t=t.tolist()).save_yaml(filename)
        
    @classmethod
    def load(cls, filename):
        db = AttrDict.load_yaml(filename)
        return cls(R=np.float64(db.R), t=np.float64(db.t))
        
class Camera(CameraIntrinsic, CameraExtrinsic): 
    def __init__(self, K, R, t, D=np.zeros(4, dtype=np.float64), shape=None): 
        CameraIntrinsic.__init__(self, K, D, shape=shape)
        CameraExtrinsic.__init__(self, R, t)

    def __repr__(self): 
        return '{:}\n{:}'.format(CameraIntrinsic.__repr__(self),
                                 CameraExtrinsic.__repr__(self))

    @property
    def P(self): 
        """ Projection matrix """
        return self.K.dot(self.Rt)     

    @classmethod
    def simulate(cls): 
        """
        Simulate camera intrinsics and extrinsics
        """
        return cls.from_intrinsics_extrinsics(CameraIntrinsic.simulate(), CameraExtrinsic.simulate())

    @classmethod
    def from_intrinsics_extrinsics(cls, intrinsic, extrinsic): 
        return cls(intrinsic.K, extrinsic.R, extrinsic.t, D=intrinsic.D, shape=intrinsic.shape)

    @classmethod
    def from_intrinsics(cls, intrinsic): 
        return cls.from_intrinsics_extrinsics(intrinsic, CameraExtrinsic.identity())

    @classmethod
    def from_KD(cls, K, D, shape=None): 
        return cls.from_intrinsics(CameraIntrinsic(K, D=D, shape=shape))

    @property
    def intrinsics(self): 
        return CameraIntrinsic(self.K, self.D, shape=self.shape)

    @property
    def extrinsics(self): 
        return CameraExtrinsic(self.R, self.t)

    def project(self, X, check_bounds=False, return_depth=False, min_depth=0.1):
        """
        Project [Nx3] points onto 2-D image plane [Nx2]
        """
        R, t = self.to_Rt()
	rvec,_ = cv2.Rodrigues(R)
	proj,_ = cv2.projectPoints(X, rvec, t, self.K, self.D)
        x = proj.reshape(-1,2)

        # Only return positive depths
        depths = self.depth_from_projection(X)
        valid = depths >= min_depth

        if check_bounds: 
            if self.shape is None: 
                raise ValueError('check_bounds cannot proceed. Camera.shape is not set')
        
            # Only return points within-image bounds
            valid = np.bitwise_and(
                valid, np.bitwise_and(
                    np.bitwise_and(x[:,0] >= 0, x[:,0] < self.shape[1]), \
                    np.bitwise_and(x[:,1] >= 0, x[:,1] < self.shape[0]))
            )

        if return_depth: 
            return x[valid], depths[valid]

        return x[valid]

    def depth_from_projection(self, X): 
        """
        Determine the depth of the [Nx3] points given camera pose
        
        Transform points in camera frame, and check z-vector: 
        [p_c = T_cw * p_w]
        """
        return (self.extrinsics * X)[:,2]

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

    def set_pose(self, pose): 
        """
        Provide extrinsics to the camera
        """
        self.quat = pose.quat
        self.tvec = pose.tvec

    def scaled(self, scale): 
        """
        Returns the scaled (ext/intrinsics) of the camera, that results 
        from resizing the image by the appropriate scale parameter
        
        The [R,t] portion of the extrinsics are unchanged, only the intrinsics
        """
        intrinsics = self.intrinsics.scaled(scale)
        return Camera.from_intrinsics_extrinsics(intrinsics, self.extrinsics)

    def F(self, other): 
        """
        Compute the fundamental matrix with respect to other camera 
        http://www.robots.ox.ac.uk/~vgg/hzbook/code/vgg_multiview/vgg_F_from_P.m

        The computed fundamental matrix, given by the formula 17.3 (p. 412) in
        Hartley & Zisserman book (2nd ed.).
        
        Use as: 
            F_10 = poses[0].F(poses[1])
            l_1 = F_10 * x_0

        """

        X1 = self.P[[1,2],:]
        X2 = self.P[[2,0],:]
        X3 = self.P[[0,1],:]
        Y1 = other.P[[1,2],:]
        Y2 = other.P[[2,0],:]
        Y3 = other.P[[0,1],:]

        F = np.float64([
            [det(np.vstack([X1, Y1])), det(np.vstack([X2, Y1])), det(np.vstack([X3, Y1]))],
            [det(np.vstack([X1, Y2])), det(np.vstack([X2, Y2])), det(np.vstack([X3, Y2]))],
            [det(np.vstack([X1, Y3])), det(np.vstack([X2, Y3])), det(np.vstack([X3, Y3]))] 
        ])

        return F #  / F[2,2]

    def E(self, other): 
        """ 
        Computes the essential matrix between this camera 
        and the other, via E = [t]_x * R
        See HZ, Sec 9.6, p 257
        """
        return skew(self.t).dot(self.R)

    def triangulate(self, pts, other_cam, other_pts): 
        return triangulate_points(self, pts, other_cam, other_pts)

    def frustum(self, zmin=0.01, zmax=0.1, pts=None): 
        assert(self.shape is not None)
        return Frustum.from_camera(self, zmin=zmin, zmax=zmax, pts=pts)

    def check_visibility(self, pts, zmin=0, zmax=100): 
        """
        Check if points are visible given fov of camera

        camera: type Camera
        """
        return check_visibility(self, pts, zmin=zmin, zmax=zmax)

    def save(self, filename): 
        raise NotImplementedError()

    @classmethod
    def load(cls, filename):
        raise NotImplementedError()

class StereoCamera(Camera): 
    def __init__(self, lcamera, rcamera, baseline): 
        if not isinstance(lcamera, Camera) or not isinstance(rcamera, Camera): 
            raise TypeError('lcamera, rcamera are not Camera type')

        Camera.__init__(self, lcamera.K, lcamera.R, lcamera.t, D=lcamera.D, shape=lcamera.shape) 
        if np.linalg.norm(lcamera.t-rcamera.t) < 1e-2:
            raise ValueError('Right camera does not have an offset from the left camera')

        # CameraIntrinsic.__init__(self, K, D, shape=shape)
        # CameraExtrinsic.__init__(self, R, t)

        # self.left = lcamera
        self.right = rcamera
        self.baseline = baseline

    @classmethod
    def simulate(cls): 
        cam = Camera.simulate()
        return cls.from_left_with_baseline(cam, baseline=0.1)

    @property
    def left(self): 
        return self

    def __repr__(self): 
        return \
            """\n############################# Stereo ##############################\n""" \
            """Baseline (in meters): {:}\n""" \
            """\n****************************** LEFT ******************************\n""" \
            """{:}""" \
            """\n******************************************************************\n""" \
            """\n***************************** RIGHT ******************************\n""" \
            """{:}""" \
            """\n******************************************************************\n""" \
            """##################################################################\n""" \
            .format(self.baseline, super(StereoCamera, self).__repr__(), self.right)

    @property 
    def baseline_px(self): 
        return self.baseline * self.left.fx

    @property
    def Q(self): 
        """
        Q = [ 1 0 0      -c_x
              0 1 0      -c_y
              0 0 0      f
              0 0 -1/T_x (c_x - c_x')/T_x ]')]
        """
        q43 = -1.0 / self.baseline
        return np.float32([[-1, 0, 0, self.left.cx],
                           [0, -1, 0, self.left.cy], 
                           [0, 0, 0, -self.left.fx], 
                           [0, 0, q43, 0]])

    @classmethod
    def from_calib_params(cls, fx, fy, cx, cy, k1=0, k2=0, k3=0, p1=0, p2=0, baseline=None, baseline_px=None, shape=None): 
        baseline, baseline_px = get_baseline(fx, baseline=baseline, baseline_px=baseline_px)
        lcamera = Camera.from_intrinsics(
            CameraIntrinsic.from_calib_params(fx, fy, cx, cy, k1=k1, k2=k2, k3=k3, p1=p1, p2=p2, shape=shape))
        return cls.from_left_with_baseline(lcamera, baseline=baseline)

        # rcamera = Camera.from_intrinsics_extrinsics(
        #     CameraIntrinsic.from_calib_params(fx, fy, cx, cy, k1=k1, k2=k2, k3=k3, p1=p1, p2=p2, shape=shape), 
        #     CameraExtrinsic(t=np.float32([baseline, 0, 0])))

        # return cls(lcamera, rcamera, baseline=baseline)

    @classmethod
    def from_left_with_baseline(cls, lcamera, baseline): 
        """ Left extrinsics is assumed to be identity """
        rcamera = Camera.from_intrinsics_extrinsics(
            lcamera.intrinsics, 
            CameraExtrinsic.from_rigid_transform(
                lcamera.oplus(RigidTransform.from_rpyxyz(0,0,0,baseline,0,0))
            )
        )
        return cls(lcamera, rcamera, baseline=baseline)

    @classmethod
    def from_intrinsics_extrinsics(cls, lintrinsics, rintrinsics, rextrinsics): 
        """ Left extrinsics is assumed to be identity """
        raise NotImplementedError()

    @classmethod
    def from_intrinsics(cls, intrinsic): 
        return cls.from_intrinsics_extrinsics(intrinsic, CameraExtrinsic.identity())

    def scaled(self, scale): 
        """
        Returns the scaled (ext/intrinsics) of the camera, that results 
        from resizing the image by the appropriate scale parameter
        
        The [R,t] portion of the extrinsics are unchanged, only the intrinsics
        """
        left = super(StereoCamera, self).scaled(scale)
        right = self.right.scaled(scale)
        return StereoCamera(left, right, baseline=self.baseline)

    def disparity_from_plane(self, rows, height): 
        """
        Computes the disparity image from expected height for each of
        the provided rows
        """
        Z = height * self.left.fy  /  (np.arange(rows) - self.left.cy + 1e-9)
        return self.left.fx * self.baseline / Z 

    def depth_from_disparity(self, disp): 
        """
        Computes the depth given disparity
        """
        return self.left.fx * self.baseline / disp

    def disparity_from_depth(self, depth): 
        """
        Computes the disparity given depth 
        """
        return self.left.fx * self.baseline / depth

    def reconstruct(self, disp): 
        """
        Reproject to 3D with calib params
        """
        X = cv2.reprojectImageTo3D(disp, self.Q)
        return X

    def reconstruct_sparse(self, xyd): 
        """
        Reproject to 3D with calib params
        """
        N, _ = xyd.shape[:2]
        xyd1 = np.hstack([xyd, np.ones(shape=(N,1))])
        XYZW = np.dot(self.Q, xyd1.T).T
        W = (XYZW[:,3]).reshape(-1,1)
        return (XYZW / W)[:,:3]
                          
    def reconstruct_with_texture(self, disp, im, sample=1): 
        """
        Reproject to 3D with calib params and texture mapped
        """
        assert(im.ndim == 3)
        X = cv2.reprojectImageTo3D(disp, self.Q)
        im_pub, X_pub = np.copy(im[::sample,::sample]).reshape(-1,3 if im.ndim == 3 else 1), \
                        np.copy(X[::sample,::sample]).reshape(-1,3)
        return im_pub, X_pub

    def set_pose(self, left_pose): 
        """
        Provide extrinsics to the left camera
        """
        super(StereoCamera, self).set_pose(left_pose)
        self.right.set_pose(
            self.left.oplus(RigidTransform.from_rpyxyz(0,0,0,self.baseline,0,0))
        )
        
    @classmethod
    def load(cls, filename): 
        raise NotImplementedError()

    def save(self, filename): 
        raise NotImplementedError()

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
        self.xs = self.xs[::self.skip]
        self.ys = (ys-self.cy) * fx_inv
        self.ys = self.ys[::self.skip]

        self.xs, self.ys = np.meshgrid(self.xs, self.ys);

    def reconstruct(self, depth): 
        s = self.skip
        depth_sampled = depth[::s,::s]
        assert(depth_sampled.shape == self.xs.shape)
        return np.dstack([self.xs * depth_sampled, self.ys * depth_sampled, depth_sampled])

    def reconstruct_sparse(self, pts, depth): 
        return np.vstack([(pts[:,0] - self.cx) * 1.0 / self.fx * depth,
                          (pts[:,1] - self.cy) * 1.0 / self.fx * depth, 
                          depth]).T

    def save(self, filename): 
        raise NotImplementedError()

    @classmethod
    def load(cls, filename):
        raise NotImplementedError()
        

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
    if len(x1) < 8:
        raise RuntimeError('Fundamental matrix requires N >= 8 pts')

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
    return (K.T).dot(npm.mat(F)).dot(K)

def check_visibility(camera, pts_w, zmin=0, zmax=100): 
    """
    Check if points are visible given fov of camera. 
    This method checks for both horizontal and vertical
    fov.     
    camera: type Camera
    """
    # Transform points in to camera's reference
    # Camera: p_cw
    pts_c = camera.c2w(pts_w.reshape(-1, 3))
    
    # Determine look-at vector, and check angle 
    # subtended with camera's z-vector (3rd column)
    z = pts_c[:,2]
    v = pts_c / np.linalg.norm(pts_c, axis=1).reshape(-1, 1)
    hangle, vangle = np.arctan2(v[:,0], v[:,2]), np.arctan2(-v[:,1], v[:,2])

    # Provides inds mask for all points that are within fov
    return np.fabs(hangle) < camera.fov[0] * 0.5 and \
        np.fabs(vangle) < camera.fov[1] * 0.5 and \
        z >= zmin and z <= zmax

def get_median_depth(camera, pts, subsample=10): 
    """ 
    Get the median depth of points for a camera reference
      Transform points in camera frame, and check z-vector: 
      [p_c = T_cw * p_w]
    
    """
    return np.median((camera * pts[::subsample])[:,2])

def get_bounded_projection(camera, pts, subsample=10): 
    """ Project points and only return points that are within image bounds """

    # Project points
    pts2d = camera.project(pts[::subsample].astype(np.float32))

    # Only return points within-image bounds
    valid = np.bitwise_and(np.bitwise_and(pts2d[:,0] >= 0, pts2d[:,0] < camera.shape[1]), \
                           np.bitwise_and(pts2d[:,1] >= 0, pts2d[:,1] < camera.shape[0]))
    return pts2d[valid], valid

def get_discretized_projection(camera, pts, subsample=10, discretize=4): 

    vis = np.ones(shape=(camera.shape[0]/discretize, camera.shape[1]/discretize), dtype=np.float32) * 10000.0
    pts2d, valid = get_bounded_projection(camera, pts, subsample=subsample)

    if True: 
        pts3d = pts[valid]
        depth = (camera * pts3d[::subsample])[:,2]
        vis[pts2d[::subsample,1], pts2d[::subsample,0]] = depth 
    else: 
        pts2d = pts2d.astype(np.int32) / discretize
        depth = get_median_depth(camera, pts, subsample=subsample)
        vis[pts2d[:,1], pts2d[:,0]] = depth 

    return vis, depth

def get_object_bbox(camera, pts, subsample=10, scale=1.0, min_height=10, min_width=10): 
    """

    Returns: 
       pts2d: Projected points onto camera
       bbox: Bounding box of the projected points [l, t, r, b]
       depth: Median depth of the projected points

    """

    pts2d, valid = get_bounded_projection(camera, pts, subsample=subsample)

    if not len(pts2d): 
        return [None] * 3

    # Min-max bounds
    x0, x1 = int(max(0, np.min(pts2d[:,0]))), int(min(camera.shape[1]-1, np.max(pts2d[:,0])))
    y0, y1 = int(max(0, np.min(pts2d[:,1]))), int(min(camera.shape[0]-1, np.max(pts2d[:,1])))

    # Check median center 
    xmed, ymed = np.median(pts2d[:,0]), np.median(pts2d[:,1])
    if (xmed >= 0 and ymed >= 0 and xmed <= camera.shape[1] and ymed < camera.shape[0]) and \
       (y1-y0) >= min_height and (x1-x0) >= min_width: 

        depth = get_median_depth(camera, pts, subsample=subsample)
        if depth < 0: return [None] * 3
        # assert(depth >= 0), "Depth is less than zero, add check for this."

        if scale != 1.0: 
            w2, h2 = (scale-1.0) * (x1-x0) / 2, (scale-1.0) * (y1-y0) / 2
            x0, x1 = int(max(0, x0 - w2)), int(min(x1 + w2, camera.shape[1]-1))
            y0, y1 = int(max(0, y0 - h2)), int(min(y1 + h2, camera.shape[0]-1))
        return pts2d.astype(np.int32), np.float32([x0, y0, x1, y1]), depth
    else: 
        return [None] * 3

def epipolar_line(F_10, x_0): 
    """
    l_1 = F_10 * x_0
    line = F.dot(np.hstack([x, np.ones(shape=(len(x),1))]).T)
    """
    return cv2.computeCorrespondEpilines(x_0.reshape(-1,1,2), 1, F_10).reshape(-1,3)

def plot_epipolar_line(im_1, F_10, x_0, im_0=None): 
    """
    Plot the epipole and epipolar line F * x = 0.

    l[0] * x + l[1] * y + l[2] = 0
    @ x=0: y = -l[2] / l[1]
    @ x=W: y = (-l[2] -l[0]*W) / l[1]
    """
    
    H,W = im_1.shape[:2]
    lines_1 = epipolar_line(F_10, x_0)

    vis_1 = to_color(im_1)
    vis_0 = to_color(im_0) if im_0 is not None else None
    
    N = 20
    cols = get_color_by_label(np.arange(len(x_0)) % N) * 255
    # for tid, pt in zip(ids, pts): 
    #     cv2.circle(vis, tuple(map(int, pt)), 2, 
    #                tuple(map(int, cols[tid % N])) if colored else (0,240,0),
    #                -1, lineType=cv2.CV_AA)

    # col = (0,255,0)
    for col, l1 in zip(cols, lines_1):
        try: 
            x0, y0 = map(int, [0, -l1[2] / l1[1] ])
            x1, y1 = map(int, [W, -(l1[2] + l1[0] * W) / l1[1] ])
            cv2.line(vis_1, (x0,y0), (x1,y1), col, 1)
        except: 
            pass
            # raise RuntimeWarning('Failed to estimate epipolar line {:s}'.format(l1))

    if vis_0 is not None: 
        for col, x in zip(cols, x_0): 
            cv2.circle(vis_0, tuple(x), 3, col, -1)
        return np.vstack([vis_0, vis_1])
    
    return vis_1

class HalfPlane(object): 
    """
    Half plane class where the plane is defined as passing through
    points (p,q,r) with points in ccw order

    Implemented from reference implementation in OpenMVG (Pierre Moulon). 
    
    """
    def __init__(self, hp): 
        assert(hp.ndim == 1 and len(hp) == 4)
        self.hp_ = hp

    @classmethod
    def from_points(cls, p, q, r): 
        abc = (p-r).cross(q-r)
        hp = np.r_[abc, -abc.dot(r)]
        return cls(hp)

    def intersect(self): 
        """
        [1] Paper: Finding the intersection of n half-spaces in time
        O(n log n).  
        Author: F.P. Preparata, D.E. Muller 
        Published in:
        Theoretical Computer Science, Volume 8, Issue 1, Pages 45-55
        Year: 1979 More: ISSN 0304-3975,
        http://dx.doi.org/10.1016/0304-3975(79)90055-0.  
        """
        pass

def test_HalfPlane(): 
    pass

class Frustum(object): 
    def __init__(self, vertices): 
        """
        Vertices: nll, nlr, nur, nul, fll, flr, fur, ful
        """
        self.vertices_ = vertices

    @property
    def _front_back_vertices(self): 
        N = len(self.vertices_)
        return self.vertices_[:N/2], self.vertices_[N/2:]

    @classmethod
    def from_pose(cls, pose, zmin=0.01, zmax=0.1, fov=np.deg2rad(60)): 
        if fov >= np.pi: 
            raise ValueError('Frustum fov cannot be {} radians'.format(fov))
        if zmin < 0.01: 
            raise ValueError('zmin needs to be finite > 0.01')

        
        # Determine vertices from fov, zmin and zmax
        # 
        #    ful --- fur
        #    |\       |\
        #    | nul ---| nur
        #    fll| --- flr |
        #     \ |       \ |
        #      nll ---  nlr
        # 
        # Order: nul, nll, nlr, nur, ful, fll, flr, fur

        # near, far = np.array([0,0,zmin]), np.array([0,0,zmax])
        # near_off, far_off = np.tan(fov / 2) * zmin, np.tan(fov / 2) * zmax
        # arr = [near + np.array([-1, -1,  0]) * near_off, 
        #        near + np.array([-1,  1,  0]) * near_off, 
        #        near + np.array([ 1,  1,  0]) * near_off, 
        #        near + np.array([ 1, -1,  0]) * near_off, 
               
        #        far +  np.array([-1, -1,  0]) * far_off, 
        #        far +  np.array([-1,  1,  0]) * far_off, 
        #        far +  np.array([ 1,  1,  0]) * far_off, 
        #        far +  np.array([ 1, -1,  0]) * far_off]

        # FoV derived from fx,fy,cx,cy=500,500,320,240
        # fovx, fovy = 65.23848614  51.28201165
        rx, ry = 0.638, 0.478
        arr = [np.array([-rx, -ry, 1.]) * zmin,
               np.array([-rx,  ry, 1.]) * zmin,
               np.array([ rx,  ry, 1.]) * zmin,
               np.array([ rx, -ry, 1.]) * zmin,

               np.array([-rx, -ry, 1.]) * zmax,
               np.array([-rx,  ry, 1.]) * zmax,
               np.array([ rx,  ry, 1.]) * zmax,
               np.array([ rx, -ry, 1.]) * zmax]

        return cls(pose * np.vstack(arr))

    @classmethod
    def from_camera(cls, c, zmin=0.01, zmax=0.1, pts=None):
        if not isinstance(c, Camera): 
            raise TypeError('c is not of Camera type, cannot instantiate frustum')
        if zmin < 0.01: 
            raise ValueError('zmin needs to be finite > 0.01')

        # Construct frustum for full camera field-of-view 
        if pts is None: 
            H, W = c.shape
            pts = np.vstack([[0, 0], [0,H-1], [W-1,H-1], [W-1,0]])
          
        rays = c.ray(pts, undistort=True, rotate=False)
        return cls(c.w2c(np.vstack([rays * zmin, rays * zmax])))           

    @property
    def vertices(self): 
        return self.vertices_

    @property
    def points_and_normals(self): 
        """
        Returns the point/normals parametrization for planes, 
        including clipped zmin and zmax frustums

        Note: points need to be in CCW
        """

        nv1, fv1 = self._front_back_vertices
        nv2 = np.roll(nv1, -1, axis=0)
        fv2 = np.roll(fv1, -1, axis=0)

        vx = np.vstack([fv1-nv1, nv2[0]-nv1[0], fv1[2]-fv1[1]])
        vy = np.vstack([fv2-fv1, nv2[1]-nv2[0], fv1[1]-fv1[0]])
        pts = np.vstack([fv1, nv1[0], fv1[1]])

        # vx += 1e-12
        # vy += 1e-12

        vx /= np.linalg.norm(vx, axis=1).reshape(-1,1)
        vy /= np.linalg.norm(vy, axis=1).reshape(-1,1)
        
        normals = np.cross(vx, vy)
        normals /= np.linalg.norm(normals, axis=1).reshape(-1,1)
        return pts, normals        

    @property
    def planes(self): 
        # plane parameters : [n, -n.p]
        pts, normals = self.points_and_normals
        return np.hstack([normals, -np.sum(np.multiply(pts, normals), axis=1).reshape(-1,1)])


def test_Frustum(): 
    pass


if __name__ == "__main__": 
    cam = Camera.from_intrinsics_extrinsics(CameraIntrinsic.simulate(), 
                                            CameraExtrinsic.simulate())
    print cam.shape
    cam.save('cam_test.yaml')
