# Author: Sudeep Pillai <spillai@csail.mit.edu>
# License: MIT

import cv2
import numpy as np
from .image_utils import to_gray, to_color

def dense_optical_flow(im1, im2, pyr_scale=0.5, levels=3, winsize=5, 
                       iterations=3, poly_n=5, poly_sigma=1.2, fb_threshold=-1, 
                       mask1=None, mask2=None, 
                       flow1=None, flow2=None): 

    if flow1 is None: 
        fflow = cv2.calcOpticalFlowFarneback(to_gray(im1), to_gray(im2), pyr_scale, levels, winsize, 
                                             iterations, poly_n, poly_sigma, 0)
    else: 
        fflow = cv2.calcOpticalFlowFarneback(to_gray(im1), to_gray(im2), pyr_scale, levels, winsize, 
                                             iterations, poly_n, poly_sigma, 0, flow1.copy())

    if mask1 is not None: 
        fflow[~mask1.astype(np.bool)] = np.nan

    if fb_threshold > 0: 
        H, W = im1.shape[:2]
        xs, ys = np.meshgrid(np.arange(W), np.arange(H))
        xys1 = np.dstack([xs, ys])
        xys2 = xys1 + fflow
        rflow = dense_optical_flow(im2, im1, pyr_scale=pyr_scale, levels=levels, 
                                   winsize=winsize, iterations=iterations, poly_n=poly_n, 
                                   poly_sigma=poly_sigma, fb_threshold=-1)
        if mask2 is not None: 
            rflow[~mask2.astype(np.bool)] = np.nan

        xys1r = xys2 + rflow
        fb_bad = (np.fabs(xys1r - xys1) > fb_threshold).all(axis=2)
        fflow[fb_bad] = np.nan

    return fflow
 
def dense_optical_flow_sf(im1, im2, layers=3, averaging_block_size=2, max_flow=4): 
    flow = np.zeros((im1.shape[0], im1.shape[1], 2))
    cv2.calcOpticalFlowSF(im1, im2, flow, layers, averaging_block_size, max_flow)
    return flow

def sparse_optical_flow(im1, im2, pts, fb_threshold=-1, 
                        window_size=15, max_level=2, 
                        criteria=(cv2.TERM_CRITERIA_EPS | cv2.TERM_CRITERIA_COUNT, 10, 0.03)): 

    # Forward flow
    p1, st, err = cv2.calcOpticalFlowPyrLK(im1, im2, pts, None, 
                                           winSize=(window_size, window_size), 
                                           maxLevel=max_level, criteria=criteria )

    # Backward flow
    if fb_threshold > 0:     
        p0r, st0, err = cv2.calcOpticalFlowPyrLK(im2, im1, p1, None, 
                                           winSize=(window_size, window_size), 
                                           maxLevel=max_level, criteria=criteria)
        p0r[st0 == 0] = np.nan

        # Set only good
        fb_good = (np.fabs(p0r-p0) < fb_threshold).all(axis=1)

        p1[~fb_good] = np.nan
        st = np.bitwise_and(st, st0)
        err[~fb_good] = np.nan
        
    return p1, st, err

def draw_flow(img, flow, step=16):
    h, w = img.shape[:2]
    y, x = np.mgrid[step/2:h:step, step/2:w:step].reshape(2,-1)
    fx, fy = flow[y,x].T
    m = np.bitwise_and(np.isfinite(fx), np.isfinite(fy))
    lines = np.vstack([x[m], y[m], x[m]+fx[m], y[m]+fy[m]]).T.reshape(-1, 2, 2)
    lines = np.int32(lines + 0.5)
    vis = cv2.cvtColor(img, cv2.COLOR_GRAY2BGR)
    cv2.polylines(vis, lines, 0, (0, 255, 0))
    for (x1, y1), (x2, y2) in lines:
        cv2.circle(vis, (x1, y1), 1, (0, 255, 0), -1)
    return vis

def draw_hsv(flow, scale=2):
    h, w = flow.shape[:2]
    fx, fy = flow[:,:,0], flow[:,:,1]
    ang = np.arctan2(fy, fx) + np.pi
    v = np.sqrt(fx*fx+fy*fy)
    hsv = np.zeros((h, w, 3), np.uint8)
    hsv[...,0] = ang*(180/np.pi/2)
    hsv[...,1] = 255
    hsv[...,2] = np.minimum(v*4*scale, 255)
    bgr = cv2.cvtColor(hsv, cv2.COLOR_HSV2BGR)
    return bgr

def warp_flow(img, flow):
    h, w = flow.shape[:2]
    flow = -flow
    flow[:,:,0] += np.arange(w)
    flow[:,:,1] += np.arange(h)[:,np.newaxis]
    res = cv2.remap(img, flow, None, cv2.INTER_LINEAR)
    return res

def test_flow(img1, img2): 
    gray1 = cv2.cvtColor(img1, cv2.COLOR_BGR2GRAY)
    gray2 = cv2.cvtColor(img2, cv2.COLOR_BGR2GRAY)
    flow = cv2.calcOpticalFlowFarneback(gray1, gray2, 0.5, 3, 15, 3, 5, 1.2, 0)
    cv2.imshow('flow', draw_flow(gray1, flow))
    cv2.imshow('flow HSV', draw_hsv(flow))
    cv2.imshow('warp', warp_flow(cur_glitch, flow))


# def flow_pts(flow):
#     valid = (flow > 0.1).all(axis=2)
#     # valid = np.bitwise_and(valid_flow[:,:,0], valid_flow[:,:,1])
    
#     H,W = valid.shape[:2]
#     xs, ys = np.meshgrid(np.arange(W), np.arange(H))
#     flow_x, flow_y = flow[:,:,0] ,flow[:,:,1]

#     pts1 = np.vstack([xs[valid], ys[valid]]).T
#     pts2 = np.vstack([xs[valid] + flow_x[valid], ys[valid] + flow_y[valid]]).T
    
#     return pts1.astype(np.float32), pts2.astype(np.float32)


# class KeyframeFlow(object): 
#     def __init__(self, calib, flow_scale=1.0, maxlen=20): 
#         self.calib_ = calib.scaled(flow_scale)
#         self.flow_scale_ = flow_scale
#         self.kf_q_ = Accumulator(maxlen=maxlen)
#         self.cams_q_ = Accumulator(maxlen=maxlen)

#     def process_dense(self): 
#         # Only estimate over semi-dense pixels
#         mask1 = fast_proposals(to_gray(self.kf_q_[-2].img), 15) > 0
#         mask2 = fast_proposals(to_gray(self.kf_q_[-1].img), 15) > 0

#         # Setup cameras for kfs
#         cam1, cam2 = self.cams_q_[-2], self.cams_q_[-1]

#         # Estimate epipolar lines
#         ptsm = valid_pixels(mask1, mask1)[:,:2].astype(np.float32)
#         F_21 = cam1.F(cam2)
#         l1 = epipolar_line(F_21, ptsm)

#         # Visualize epipolar flow
#         H, W = mask1.shape[:2]
#         ref_flow = np.zeros((H,W,2), dtype=np.float32)
#         ref_flow[ptsm[:,1].astype(np.int32), ptsm[:,0].astype(np.int32)] = l1[:,:2]#  * 4        

#         # Compute dense optical flow
#         flow = dense_optical_flow(to_gray(self.kf_q_[-2].img), 
#                                   to_gray(self.kf_q_[-1].img), pyr_scale=0.5, fb_threshold=2, 
#                                   mask1=mask1, mask2=mask2, flow1=ref_flow)
#         flow[~np.isfinite(flow)] = 0
#         pts1, pts2 = flow_pts(flow)
#         if not len(pts1): return


#         # dv = pts2-pts1
#         # dv /= np.linalg.norm(dv, axis=1).reshape(-1,1)
#         # valid = np.fabs(np.sum(dv * l1[:,:2], axis=1)) < 0.2
        
#         # # Epipolar constraint inliers
#         # pts1, pts2 = pts1[valid], pts2[valid]
#         # if not len(pts1): return

#         # ref_flow = np.zeros_like(flow)
#         # ref_flow[pts1[:,1].astype(np.int32), pts1[:,0].astype(np.int32)] = pts2-pts1

#         # imshow_cv('pts1', draw_features(self.kf_q_[-2].img, pts1))
#         # imshow_cv('pts2', draw_features(self.kf_q_[-1].img, pts1))
#         # pts1 = self.scaled_calib_.undistort_points(pts1)
#         # pts2 = self.scaled_calib_.undistort_points(pts2)

#         X = triangulate_points(cam1, pts1, cam2, pts2)
#         valid = X[:,2] > 1
#         X = X[valid]
        
#         draw_utils.publish_cloud('vio-X', [X[:,:3]], 
#                                  frame_id='camera', element_id=self.kf_q_.index-1, reset=self.kf_q_.index == 0)
        
#         imshow_cv('flow', np.hstack([draw_hsv(ref_flow, scale=4), draw_hsv(flow, scale=4)]))

#     def process_sparse(self): 

#         if not hasattr(self, 'klt_'): 
#             from pybot.vision.trackers.base_klt import BaseKLT, OpenCVKLT
#             from pybot.vision.trackers.tracker_utils import OpticalFlowTracker
#             self.klt_ = OpenCVKLT()

#             # Triangulated points
#             self.X_ = dict()

#         # =====================
#         # Setup cameras for kfs
#         cam1, cam2 = self.cams_q_[-2], self.cams_q_[-1]

#         # =====================
#         # KLT tracking
#         vis = to_color(np.copy(self.kf_q_[-1].img))
#         self.klt_.process(to_gray(self.kf_q_[-1].img))
#         self.klt_.draw_tracks(vis, colored=True)
#         imshow_cv('klt', vis)

#         # vis2 = to_color(np.copy(self.kf_q_[-1].img))

#         # # Determine matches
#         # tids, pts1, pts2 = self.klt_.matches(index1=-10, index2=-1)
#         # cols = np.tile([255,0,0], (len(pts1), 1)).astype(np.int64)
#         # matches = draw_lines(vis2, pts1, pts2, colors=cols)
#         # if not len(pts1): return

#         # # Estimate epipolar lines
#         # F_21 = cam1.F(cam2)
#         # l1 = epipolar_line(F_21, pts1)

#         # # Determine inliers 
#         # dv = pts2-pts1
#         # dv /= np.linalg.norm(dv, axis=1).reshape(-1,1)
#         # valid = np.fabs(np.sum(dv * l1[:,:2], axis=1)) < 0.2

#         # # Epipolar constraint inliers
#         # pts1, pts2 = pts1[valid], pts2[valid]
#         # if not len(pts1): return

#         # cols = np.tile([0,255,0], (len(pts1), 1)).astype(np.int64)
#         # imshow_cv('matches', draw_lines(matches, pts1, pts2, colors=cols))

#         # # Triangulate points based on wide baseline
#         # X = triangulate_points(self.cams_q_[-10], pts1, self.cams_q_[-1], pts2)
#         # for tid, p in zip(tids, X): 
#         #     self.X_[tid] = p

#         # # valid = X[:,2] > 1
#         # # X = X[valid]
        
#         # draw_utils.publish_cloud('vio-X', [np.vstack(self.X_.values())[:,:3]], 
#         #                          frame_id='camera', element_id=self.kf_q_.index-1, reset=True) # self.kf_q_.index == 0)
#         # # print pts1, pts2

#     def process(self, im, pose): 
#         self.kf_q_.accumulate(Keyframe(self.kf_q_.index, 0, pose, img=im_resize(im, scale=self.flow_scale_)))
#         self.cams_q_.accumulate(
#             Camera.from_intrinsics_extrinsics(
#             self.calib_, 
#             CameraExtrinsic.from_rigid_transform(self.kf_q_.latest.pose.inverse()))
#         )

#         draw_utils.publish_pose_list('vio-rgb', [Pose.from_rigid_transform(self.kf_q_.index, self.kf_q_.latest.pose)], 
#                                          frame_id='camera', reset=self.kf_q_.index == 0)

#         if self.kf_q_.length < 2: 
#             return

#         # Process sparse klt
#         self.process_sparse()

# def test_keyframe_flow():         
#     # Keyframe flow
#     kf_flow = KeyframeFlow(dataset.calib, self.flow_scale_)
#     kf_flow.process(img, self.p_wc_.latest)
