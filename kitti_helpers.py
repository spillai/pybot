import os, cv2
import numpy as np
from collections import namedtuple
from bot_geometry.rigid_transform import RigidTransform

# Q = [ 1 0 0      -c_x
#       0 1 0      -c_y
#       0 0 0      f
#       0 0 -1/T_x (c_x - c_x')/T_x ]')]
CalibParams = namedtuple('CalibParams', ['R0', 'R1', 'P0', 'P1', 'Q', 'T0', 'T1', 'fx', 'fy', 'cx', 'cy', 'baseline'])
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

    R0, R1 = np.eye(3), np.eye(3)
    T0, T1 = np.zeros(3), np.float32([-baseline_px, 0, 0])

    Q = np.float32([[-1, 0, 0, cx],
                    [0, -1, 0, cy], 
                    [0, 0, 0, -fx], 
                    [0, 0, q43,0]])

    return CalibParams(R0, R1, P0, P1, Q, T0, T1, fx, fy, cx, cy, baseline)
    

def kitti_stereo_calib_params(scale=1.0): 
    f = 718.856*scale
    cx, cy = 607.192*scale, 185.2157*scale
    baseline_px = 386.1448 * scale

    return get_calib_params(f, f, cx, cy, baseline_px=baseline_px)

def kitti_load_poses(fn): 
    X = (np.fromfile(os.path.expanduser(fn), dtype=np.float64, sep=' ')).reshape(-1,12)
    return map(lambda p: RigidTransform.from_Rt(p[:3,:3], p[:3,3]), 
                map(lambda x: x.reshape(3,4), X))

def kitti_poses_to_str(poses): 
    return "\r\n".join(map(lambda x: " ".join(map(str, 
                                                  (x.to_homogeneous_matrix()[:3,:4]).flatten())), poses))

def kitti_poses_to_mat(poses): 
    return np.vstack(map(lambda x: (x.to_homogeneous_matrix()[:3,:4]).flatten(), poses)).astype(np.float64)

def bumblebee_stereo_calib_params_ming(scale=1.0): 
    fx, fy = 809.53*scale, 809.53*scale
    cx, cy = 321.819*scale, 244.555*scale
    baseline = 0.119909

    return get_calib_params(fx, fy, cx, cy, baseline=baseline)

def bumblebee_stereo_calib_params(scale=1.0): 
    fx, fy = 0.445057*640*scale, 0.59341*480*scale
    cx, cy = 0.496427*640*scale, 0.519434*480*scale
    baseline = 0.120018 

    return get_calib_params(fx, fy, cx, cy, baseline=baseline)

global ply_header
ply_header = '''ply
format ascii 1.0
element vertex %(vert_num)d
property float x
property float y
property float z
property uchar red
property uchar green
property uchar blue
end_header
'''

def write_ply(fn, verts, colors):
    global ply_header
    verts = verts.reshape(-1, 3)
    colors = colors.reshape(-1, 3)
    verts = np.hstack([verts, colors])
    with open(fn, 'w') as f:
        f.write(ply_header % dict(vert_num=len(verts)))
        np.savetxt(f, verts, '%f %f %f %d %d %d')


# Write ply for testing purposes
def write_ply_with_dispmask(fn, left_im, disp, X): 
    mask = disp > disp.min()
    colors = cv2.cvtColor(left_im, cv2.COLOR_GRAY2RGB)
    write_ply(fn, X[mask], colors[mask])
    print '%s saved' % fn


def stereo_camera(): 
    K = np.array([[718.856, 0, 607.1928], 
                  [0, 718.856, 185.2157], 
                  [0, 0, 1]], dtype=np.float64)
    D = np.zeros(4, dtype=np.float64)
