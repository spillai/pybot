import cv2

import numpy as np
from numpy.linalg import det, norm
import numpy.matlib as npm
from scipy import linalg

from bot_vision.image_utils import to_color
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

def undistort_image(im, K, D): 
    # newcamera, roi = cv2.getOptimalNewCameraMatrix(self.K, self.D, (W,H), 0) 
    return cv2.undistort(im, K, D, None, K)

def get_calib_params(fx, fy, cx, cy, baseline=None, baseline_px=None): 
    assert(baseline is not None or baseline_px is not None)
    if baseline is not None: 
        baseline_px = baseline * fx
    elif baseline_px is not None: 
        baseline = baseline_px / fx
    else: 
        raise AssertionError('Ambiguous baseline and baseline_px values. Provide either, not both')
        
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
        Default init
        """
        self.K = npm.mat(K)                # Calibration matrix
        self.D = D                         # Distortion
        self.cx, self.cy = K[0,2], K[1,2]  # Camera center
        self.fx, self.fy = K[0,0], K[1,1]  # Focal length
        self.shape = shape                 # Image size (H,W,C): (480,640,3)

    def __repr__(self): 
        return ' K = {:}\n D = {:}\n fx={:}, fy={:}, cx={:}, cy={:}'.format(np.array_str(np.array(self.K), precision=2), 
                                                                            np.array_str(self.D, precision=2),
                                                                            self.fx, self.fy, self.cx, self.cy) 

    @classmethod
    def simulate(cls): 
        """
        Simulate a 640x480 camera with 500 focal length
        """
        return cls.from_calib_params(500., 500., 320., 240.)

    @classmethod
    def from_calib_params(cls, fx, fy, cx, cy): 
        return cls(construct_K(fx, fy, cx, cy))

    @classmethod
    def from_calib_params_fov(cls, fov, cx, cy, D=np.zeros(5, dtype=np.float64), shape=None): 
        return cls(construct_K(cx / np.tan(fov), cy / np.tan(fov), cx, cy), D=D, shape=shape)

    @property
    def fov(self): 
        """
        Returns the field of view for each axis
        """
        return np.array([np.arctan(self.cx / self.fx), np.arctan(self.cy / self.fy)]) * 2

    def undistort(self, im): 
        return undistort_image(im, self.K, self.D)
        
    def undistort_debug(self, im=None): 
        if im is None: 
            im = np.zeros(shape=self.shape, dtype=np.uint8)
        im[::20, :] = 128
        im[:, ::20] = 128
        return self.undistort(im)

    def save(self, filename): 
        AttrDict(
            K=self.K.tolist(), D=self.D.tolist(), 
            shape=self.shape.tolist() if self.shape is not None else None
        ).save_json(filename)
        
    @classmethod
    def load(cls, filename):
        db = AttrDict.load_json(filename)
        return cls(
            K=np.float64(db.K), D=np.float64(db.D), 
            shape=np.int32(db.shape) if db.shape is not None else None
        )
    
class CameraExtrinsic(RigidTransform): 
    def __init__(self, R=npm.eye(3), t=npm.zeros(3)):
        """
        Pose is defined as p_cw (pose of the world wrt camera)
        """
        p = RigidTransform.from_Rt(R, t)
        RigidTransform.__init__(self, xyzw=p.quat.to_xyzw(), tvec=p.tvec)

    @classmethod
    def from_rigid_transform(cls, p): 
        R, t = p.to_Rt()
        return cls(R, t)
    
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

    def save(self, filename): 
        R, t = p.to_Rt()
        AttrDict(R=R.tolist(), t=t.tolist()).save_json(filename)
        
    @classmethod
    def load(cls, filename):
        db = AttrDict.load_json(filename)
        return cls(R=np.float64(db.R), t=np.float64(db.t))
        
class Camera(CameraIntrinsic, CameraExtrinsic): 
    def __init__(self, K, R, t, D=np.zeros(4, dtype=np.float64), shape=None): 
        CameraIntrinsic.__init__(self, K, D, shape=shape)
        CameraExtrinsic.__init__(self, R, t)

    @property
    def P(self): 
        Rt = self.to_homogeneous_matrix()[:3]
        return self.K.dot(Rt)     # Projection matrix

    @classmethod
    def simulate(cls): 
        """
        Simulate camera intrinsics and extrinsics
        """
        return cls.from_intrinsics_extrinsics(CameraIntrinsic.simluate(), CameraExtrinsic.simulate())

    @classmethod
    def from_intrinsics_extrinsics(cls, intrinsic, extrinsic): 
        return cls(intrinsic.K, extrinsic.R, extrinsic.t, D=intrinsic.D, shape=intrinsic.shape)

    @classmethod
    def from_intrinsics(cls, intrinsic): 
        return cls.from_intrinsics_extrinsics(intrinsic, CameraExtrinsic.identity())

    @classmethod
    def from_KD(cls, K, D, shape=None): 
        return cls.from_intrinsics(CameraIntrinsic(K, D=D, shape=shape))

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

    def set_pose(self, pose): 
        """
        Provide extrinsics to the camera
        """
        self.quat = pose.quat
        self.tvec = pose.tvec

    def F(self, other): 
        """
        Compute the fundamental matrix with respect to other camera 
        http://www.robots.ox.ac.uk/~vgg/hzbook/code/vgg_multiview/vgg_F_from_P.m
        
        Use as: 
           F_10 = poses[1].F(poses[0])
           l_1 = F_10 * x_0

        """
        X1 = self.P[[1,2],:]
        X2 = self.P[[2,0],:]
        X3 = self.P[[0,1],:]
        Y1 = other.P[[1,2],:]
        Y2 = other.P[[2,0],:]
        Y3 = other.P[[0,1],:]
        
        return np.float64([[det(np.vstack([X1, Y1])), det(np.vstack([X2, Y1])), det(np.vstack([X3, Y1]))],
                        [det(np.vstack([X1, Y2])), det(np.vstack([X2, Y2])), det(np.vstack([X3, Y2]))],
                        [det(np.vstack([X1, Y3])), det(np.vstack([X2, Y3])), det(np.vstack([X3, Y3]))]])


    def save(self, filename): 
        raise NotImplementedError()

    @classmethod
    def load(cls, filename):
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
        self.xs = self.xs[::self.skip] # skip pixels
        self.ys = (ys-self.cy) * fx_inv
        self.ys = self.ys[::self.skip] # skip pixels

        self.xs, self.ys = np.meshgrid(self.xs, self.ys);

    def reconstruct(self, depth): 
        assert(depth.shape == self.xs.shape)
        return np.dstack([self.xs * depth, self.ys * depth, depth])

    def reconstruct_points(self, pts, depth): 
        return np.vstack([self.xs[pts[:,1], pts[:,0]] * depth,
                          self.ys[pts[:,1], pts[:,0]] * depth,
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

def check_visibility(camera, pts_w): 
    """
    Check if points are visible given fov of camera
    
    camera: type Camera
    """
    # Transform points in to camera's reference
    # Camera: p_cw
    pts_c = camera * pts_w.reshape(-1, 3)

    # Hack: only check max of the fovs
    hfov = np.max(camera.fov) / 2

    # Determine look-at vector, and check angle 
    # subtended with camera's z-vector (3rd column)
    v = pts_c / np.linalg.norm(pts_c, axis=1).reshape(-1, 1)
    thetas = np.arccos(v[:,2])

    # Provides inds mask for all points that are within fov
    return thetas < hfov

def get_median_depth(camera, pts, subsample=10): 
    """ 
    Get the median depth of points for a camera reference
      Transform points in camera frame, and check z-vector: 
      [p_c = T_cw * p_w]
    
    """
    return np.median((camera * pts[::subsample])[:,2])

def get_bounded_projection(camera, pts, subsample=10, min_height=10, min_width=10): 
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

def epipolar_line(F_10, x_1): 
    """
    l_1 = F_10 * x_1
    line = F.dot(np.hstack([x, np.ones(shape=(len(x),1))]).T)
    """
    return cv2.computeCorrespondEpilines(x_1.reshape(-1,1,2), 1, F_10).reshape(-1,3)

def plot_epipolar_line(im_1, F_10, x_0, im_0=None): 
    """
    Plot the epipole and epipolar line F * x = 0.
    """
    
    H,W = im_1.shape[:2]
    lines_1 = epipolar_line(F_10, x_0)

    vis_1 = to_color(im_1)
    vis_0 = to_color(im_0) if im_0 is not None else None
    
    col = (0,255,0)
    for l1 in lines_1:
        try: 
            x0, y0 = map(int, [0, -l1[2] / l1[1] ])
            x1, y1 = map(int, [W, -(l1[2] + l1[0] * W) / l1[1] ])
            cv2.line(vis_1, (x0,y0), (x1,y1), col, 1)
        except: 
            pass
            # raise RuntimeWarning('Failed to estimate epipolar line {:s}'.format(l1))

    if vis_0 is not None: 
        for x in x_0: 
            cv2.circle(vis_0, tuple(x), 5, col, -1)
        return np.hstack([vis_0, vis_1])
    
    return vis_1


class Frustum(object): 
    def __init__(self, pose, zmin=0.0, zmax=0.1, fov=np.deg2rad(60)): 
    
        self.p0 = np.array([0,0,0])
        self.near, self.far = np.array([0,0,zmin]), np.array([0,0,zmax])
        self.near_off, self.far_off = np.tan(fov / 2) * zmin, np.tan(fov / 2) * zmax
        self.zmin = zmin
        self.zmax = zmax
        self.fov = fov

        self.pose = pose
        self.p0 = pose.tvec

    def get_vertices(self): 
        arr = [self.near + np.array([-1, -1, 0]) * self.near_off, 
               self.near + np.array([1, -1, 0]) * self.near_off, 
               self.near + np.array([1, 1, 0]) * self.near_off, 
               self.near + np.array([-1, 1, 0]) * self.near_off, 
               
               self.far + np.array([-1, -1, 0]) * self.far_off, 
               self.far + np.array([1, -1, 0]) * self.far_off, 
               self.far + np.array([1, 1, 0]) * self.far_off, 
               self.far + np.array([-1, 1, 0]) * self.far_off]
        
        nll, nlr, nur, nul, fll, flr, fur, ful = self.pose * np.vstack(arr)
        return nll, nlr, nur, nul, fll, flr, fur, ful


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
