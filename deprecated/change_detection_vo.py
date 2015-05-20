import cv, cv2, time
import numpy as np

from collections import deque

import matplotlib.pylab as plt
from matplotlib.colors import colorConverter

from pybot_externals import SparseStereoMatch, PyOctoMap, StereoELAS, StereoVISO2, stixel_world, depth_inpainting
from fs_pcl_utils import CorrespondenceRejectionSAC

from occ_det.utils.rigid_transform import RigidTransform
from occ_det.utils.kitti_helpers import kitti_stereo_calib_params, bumblebee_stereo_calib_params, bumblebee_stereo_calib_params_ming, write_ply
from occ_det.utils.imshow_utils import imshow_cv, imshow_plt
from occ_det.utils.dataset_readers import StereoDatasetReader, KITTIStereoDatasetReader, BumblebeeStereoDatasetReader
from occ_det.utils.optflow_utils import draw_hsv, draw_flow
from occ_det.utils.stereo_utils import StereoSGBM, StereoSGBM2
# from occ_det.utils.io_utils import VideoWriter
# from occ_det.utils.gl_viewer import GLViewer

# from occ_det.utils.dbg_utils import set_trac

class ChangeDetectionVO: 

    lk_params = dict( winSize  = (15, 15),
                      maxLevel = 2,
                      criteria = (cv2.TERM_CRITERIA_EPS | cv2.TERM_CRITERIA_COUNT, 10, 0.03))

    def __init__(self, directory, output='output.mp4'): 
        self.scale = 0.5
        self.output_filename = output

        # KITTI params
        self.calib_params = kitti_stereo_calib_params(self.scale)
        self.dataset = KITTIStereoDatasetReader(directory=directory)

        # Stereo block matcher
        # self.sgbm = StereoSGBM()
        self.elas = StereoELAS()

        # Scene flow
        self.im_grid = None

        # Stereo VO
        print 'Baseline: ', self.calib_params.baseline
        self.vo = StereoVISO2(f=self.calib_params.f / self.scale, 
                              cx=self.calib_params.cx / self.scale, cy=self.calib_params.cy / self.scale, 
                              baseline=self.calib_params.baseline) 

        # Discretize
        self._s = 4 # image/cloud subsample

    def normalize(self, vec): 
        return (vec.astype(np.float32) - np.min(vec)) / (np.max(vec) - np.min(vec))


    def build_regular_grid(self, shape, sample): 
        if self.im_grid is None: 
            H, W = shape[:2]
            xs, ys = np.meshgrid(np.arange(0,W,sample), np.arange(0,H,sample))
            self.im_grid = ((np.dstack([xs,ys])).reshape(-1,2)).astype(np.float32)
        return

    def compute_scene_flow(self, p21, X1, shape, sample=4): 
        # One time construction of meshed grid
        self.build_regular_grid(shape, sample)

        # Project scene points 
        rvec,_ = cv2.Rodrigues(self.calib_params.R0)
        proj,_ = cv2.projectPoints(p21 * X1.reshape(-1,3).astype(np.float32), rvec, self.calib_params.T0,
                                self.calib_params.P0[:3,:3], np.zeros(4))
        x_t_est = proj.astype(np.float32).reshape((-1, 2))

        # Compare against expected image points based on VO
        # print self.im_grid.shape, x_t_est.shape, shape
        flow = (x_t_est - self.im_grid).reshape(shape[0]/sample, shape[1]/sample, -1)
        flow_scaled = cv2.resize(flow, (shape[1],shape[0]), interpolation=cv2.INTER_LINEAR)

        return flow_scaled

    def compute_dense_optical_flow(self, sample=1): 
        flow = cv2.calcOpticalFlowFarneback(
            self.ims[-2], self.ims[-1], 0.5, 3, 15, 3, 5, 1.2, 0)
        return flow[::sample,::sample]

    def compute_sparse_optical_flow(self): 
        # One time construction of meshed grid
        self.build_regular_grid(shape, sample)

        p0 = self.im_grid.reshape(-1, 1, 2)
        p1, st, err = cv2.calcOpticalFlowPyrLK(self.ims[-2], self.ims[-1], p0, None, **lk_params)
        p0r, st, err = cv2.calcOpticalFlowPyrLK(self.ims[-1], self.ims[-2], p1, None, **lk_params)
        d = abs(p0-p0r).reshape(-1, 2).max(-1)
        good = d < 1

    def estimate_motion_from_flow(self, Xs, flow, flow_wts, sample=4): 
        H, W = flow.shape[:2]

        # Mesh grid
        xs, ys = np.meshgrid(np.arange(0,W,sample), np.arange(0,H,sample))

        # Pick top wts
        wts = flow_wts[::sample,::sample].flatten()
        winds = np.where(wts > 0)[0]
        top_inds = winds[np.argsort(wts[winds])[::-1][:4000]]

        # Ensure predictions are within image bounds
        xys = np.hstack([xs.reshape(-1,1), ys.reshape(-1,1)])[top_inds]
        xys2 = (xys + flow[xys[:,1], xys[:,0]]).astype(np.int32)
        inds = reduce(lambda x,y: np.bitwise_and(x,y), [xys[:,0] >= 0, xys[:,0] < W, 
                                                        xys2[:,0] >= 0, xys2[:,0] < W, 
                                                        xys[:,1] >= 0, xys[:,1] < H, 
                                                        xys2[:,1] >= 0, xys2[:,1] < H])

        # RANSAC tf estimation between 2 sets of 3d points
        src, tgt = Xs[-2][xys[inds,1], xys[inds,0]], Xs[-1][xys2[inds,1], xys2[inds,0]]
        vinds = np.isfinite(np.hstack([src,tgt])).all(axis=1)
        Tts, inliers = CorrespondenceRejectionSAC(source=tgt[vinds], target=src[vinds], 
                                                  source_dense=np.array([]), 
                                                  target_dense=np.array([]), 
                                                  inlier_threshold=0.3, max_iterations=300)

        # Plot inlier/outlier img
        inds_m = np.zeros_like(inds, dtype=bool)
        inds_m[vinds[inliers]] = True
        inlier_pts, outlier_pts = np.where(inds_m)[0], np.where(~inds_m)[0]

        # Visualize inliers/outliers 
        viz_im = np.zeros(shape=(H,W,3), dtype=np.uint8)
        viz_im[xys[inlier_pts,1],xys[inlier_pts,0],1] = 255
        viz_im[xys[outlier_pts,1],xys[outlier_pts,0],2] = 255

        imshow_cv('flow inliers', viz_im)

        print 'Inliers %i/%i points' % (len(inliers), len(vinds))
        return Tts


    def compute_epipole(self, p): 
        rvec,_ = cv2.Rodrigues(p.quat.to_homogeneous_matrix())
        print rvec, p.quat.to_homogeneous_matrix()
        proj,_ = cv2.projectPoints(np.array([[0.,0.,1000.]]), rvec, p.tvec,
                                self.calib_params.P0[:3,:3], np.zeros(4))
        print proj.flatten()
        return proj.flatten()

    def run_stereo(self): 

        self.ims, disps, poses, Xs = deque(maxlen=2), deque(maxlen=2), deque(maxlen=2), deque(maxlen=2)
        poses_est = deque(maxlen=2)
        for left_im, right_im in self.dataset.iter_stereo_frames(): 
            if left_im is None or right_im is None: break

            # ================================
            # VO on full-size image
            T = self.vo.process(left_im, right_im)
            poses.append(RigidTransform.from_homogenous_matrix(T))

            # ================================
            # Re-size stereo pair 
            left_im = cv2.resize(left_im, None, fx=self.scale, fy=self.scale, interpolation=cv2.INTER_LINEAR)
            right_im = cv2.resize(right_im, None, fx=self.scale, fy=self.scale, interpolation=cv2.INTER_LINEAR)

            # ================================
            # Compute dense disparity, and point cloud
            # sgbm.compute/elas.process
            st = time.time()
            # disp = self.sgbm.compute(left_im, right_im)
            disp = self.elas.process(left_im, right_im)
            print 'Time taken to stereo', time.time() - st

            # ================================
            # Add images, depth to queue
            self.ims.append(left_im)
            disps.append(disp)

            # ================================
            # Reproject to 3D with calib params
            X = cv2.reprojectImageTo3D(disp, self.calib_params.Q)
            # X_pub, im_pub = np.copy(X[::self._s,::self._s]).reshape(-1,3), \
            #                 np.copy(left_im[::self._s,::self._s]).reshape(-1,1)

            # X_pubmask = ~np.bitwise_or(X_pub[:,2] > 75.0, X_pub[:,2] < 1.0)
            # X_pub_m, im_pub_m = X_pub[X_pubmask], im_pub[X_pubmask]
            Xs.append(X)
            if len(self.ims) < 2: continue

            # ================================
            # Compute scene flow
            pose12 = (poses[-1].inverse()).oplus(poses[-2])
            scene_flow = self.compute_scene_flow(pose12, Xs[-2],
                                                 shape=Xs[-2].shape[:2], sample=1)

            # ================================
            # Compute dense optical flow (as prior)
            flow = self.compute_dense_optical_flow(sample=1)

            # # ================================
            # # Compute epipole
            # ep = np.int32(self.compute_epipole((poses[-2].inverse()).oplus(poses[-1])))

            # H, W = left_im.shape
            # cv2.line(left_im, (ep[0],0), (ep[0],H-1), (0,0,255), 1)

            # ================================
            # Viz flows
            mag = np.linalg.norm(scene_flow, axis=2)
            mag = (mag * 255 / 100.0).astype(np.uint8)
            mag_color = cv2.applyColorMap(mag, cv2.COLORMAP_JET)
            imshow_cv('flow mag', mag_color)
            imshow_cv('flow | scene_flow hsv', np.vstack([draw_hsv(flow), draw_hsv(scene_flow)]))
            imshow_cv('flow | scene_flow', np.vstack([draw_flow(left_im, flow), 
                                                      draw_flow(left_im, scene_flow)]))

            # # Convert to 1D flow
            # zero = np.zeros(shape=flow.shape[:2])
            # flow_x = np.dstack([flow[:,:,0], zero])
            # scene_flow_x = np.dstack([scene_flow[:,:,0], zero])
            # imshow_cv('flow_x | scene_flow hsv', np.vstack([draw_hsv(flow_x), draw_hsv(scene_flow_x)]))
            # imshow_cv('flow_x | scene_flow', np.vstack([draw_flow(left_im, flow_x), 
            #                                             draw_flow(left_im, scene_flow_x)]))
                
            # ================================
            # Compute discrepancy between scene and dense flow
            flow_norm, scene_flow_norm = np.linalg.norm(flow, axis=2), np.linalg.norm(scene_flow, axis=2)
            nflow, nscene_flow = flow / np.dstack([flow_norm, flow_norm]), \
                                 scene_flow / np.dstack([scene_flow_norm, scene_flow_norm])

            flow_dir_diff = np.sum(np.multiply(nflow, nscene_flow), axis=2)
            flow_mag_diff = np.exp( -(scene_flow_norm - flow_norm)**2 / 5 ) 

            # Prune out pts with low/nan disparities and remove low flow pts
            outlier_inds = reduce(lambda x,y: np.bitwise_or(x,y), [disps[-2] < 4, flow_norm < 1, flow_dir_diff < 0.7])

            flow_dir_diff[outlier_inds] = 0.
            flow_mag_diff[outlier_inds] = 0.
            matched_flow = np.multiply(flow_dir_diff, flow_mag_diff)

            imshow_cv('flow diff', np.hstack([flow_dir_diff, flow_mag_diff, matched_flow]))    

            # ================================
            # Estimation motion via dense OF
            flowT = self.estimate_motion_from_flow(Xs, scene_flow, matched_flow)
            # print (poses[-2].inverse()).oplus(poses[-1]), RigidTransform.from_homogenous_matrix(flowT)

            try: 
                poses_est.append(poses[-1].oplus(RigidTransform.from_homogenous_matrix(flowT)))
            except: 
                print 'Failed transformation estimation'

            # # # ================================
            # # # Diff on depths
            # # absdiff_disp = np.abs(disps[-1]-disps[-2])
            # # mask = (disps[-1] == 0) | (disps[-2] == 0)
            # # absdiff_disp[mask] = 0

            # disp8 = np.array((absdiff_disp)/(16) * 255, dtype=np.uint8)
            # disp_color = cv2.applyColorMap(disp8, cv2.COLORMAP_JET)
            # disp8_color = cv2.cvtColor(disp8, cv2.COLOR_GRAY2BGR)
            # out = np.hstack([disp8_color, disp_color])

            # imshow_cv('diff_disp', out)
            # # imshow_cv('diff', 0.5/255 * ims[-1] + 0.5/255 * ims[-2])
            
            # # Diff
            # absdiff_im = np.abs(ims[-1]-ims[-2])
            # # imshow_cv('absdiff', absdiff_im) # .astype(np.float32)/255.0)
            # imshow_cv('diff', 0.5/255 * ims[-1] + 0.5/255 * ims[-2])

            # # Scanline
            # H, W = left_im.shape
            # ys = self.normalize(absdiff_im[H*3/4,:])
            # p_ys = self.normalize(ims[-2][H*3/4,:])

            
            # # Plot scanline intensities
            # plt.clf()
            # plt.subplot(1,1,1)
            # plt.plot(np.arange(len(ys)), ys)
            # plt.tight_layout()
            # plt.pause(0.1)
            # # cv2.waitKey(0)

            # # Show stereo
            # left_im[H*3/4,:] = 255
            # # imshow_cv('stereo', np.hstack([left_im, right_im]))



            
