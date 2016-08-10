import cv2, time
import numpy as np

from collections import deque

import matplotlib.pylab as plt
from matplotlib.colors import colorConverter

from pybot_externals import SparseStereoMatch, PyOctoMap, PyHeightMap, StereoELAS, StereoVISO2, OrderedCostVolumeStereo
from fs_pcl_utils import compute_normals, ground_plane_estimation

from occ_det.utils.rigid_transform import RigidTransform
from occ_det.utils.imshow_utils import imshow_cv, imshow_plt

import occ_det.utils.edge_utils as edge_utils
import occ_det.utils.color_utils as color_utils

from occ_det.utils.stereo_utils import StereoSGBM, StereoBM
from occ_det.utils.io_utils import VideoWriter

# ROS draw utils
import occ_det.ros.draw_utils as draw_utils
np.set_printoptions(precision=8, suppress=True, threshold='nan', linewidth=160)

class FreeSpaceDetection: 
    def __init__(self, dataset, calib_params, scale=1.0, 
                 incremental_add=False, vo=False, velodyne=False, stereo=None, 
                 ground_plane=False): 

        # Dataset, scale and calib
        self.scale = scale
        self.calib_params = calib_params
        self.dataset = dataset 

        # Rolling window length
        self.qlen, self._s = 5, 2 # queue length, and image subsamplex
        self.poses = deque(maxlen=self.qlen)
        self.Xs, self.ims = deque(maxlen=self.qlen), \
                            deque(maxlen=self.qlen)

        # Incremental voxelization
        self.incremental_add = incremental_add

        # Setup problem
        if vo: self.setup_vo()
        if velodyne: self.setup_velodyne()
        if stereo is not None: self.setup_stereo(alg=stereo)
        if ground_plane : self.setup_ground_plane_estimation()

    # =========================================
    # Occupancy maps and voxelization
    def voxelize(self, X, poses=RigidTransform.identity()):
        st = time.time()

        # Update octomap
        self.octmap.clear()

        if isinstance(X, list) and isinstance(poses, list): 
            for pj, Xj in zip(poses, X): 
                self.octmap.updateNodesIncremental(pj.matrix, Xj)
        elif isinstance(X, np.ndarray) and isinstance(poses, RigidTransform): 
            # self.octmap.updateNodes(poses.matrix, X)
            self.octmap.updateNodes(X)
        else: 
            raise TypeError('Unknown X point cloud, and poses for voxelization')

        # Plot occupied octomap
        onode = self.octmap.getFreeAndOccupiedNodes()
        occ_cells, probs = onode.getOccCells(), onode.getProbs()
        try: 
            print 'Occ cells: ', occ_cells.shape, probs.shape
            print 'Octomap construction % s cloud %s'  % (occ_cells.shape, time.time() - st)
        except: 
            pass
        return occ_cells, probs

    def incr_voxelize(self, X, pose_mc=RigidTransform.identity()):
        st = time.time()

        # Update octomap
        self.octmap.updateNodesIncremental(pose_mc.matrix, pose_mc * X)
        # print 'Sensor to world ', pose_mc.matrix

        # Plot occupied octomap
        onode = self.octmap.getFreeAndOccupiedNodes()
        occ_cells, probs, resp = onode.getOccCells(), onode.getProbs(), onode.getResponses()
        # probs[(probs <= 0) | (probs > 1)] = 1.0

        try: 
            print 'Occ cells: ', occ_cells.shape, probs.shape
            print 'Octomap construction % s cloud %s'  % (occ_cells.shape, time.time() - st)
        except: 
            pass
        # occ_cells_ = pose_mc.inverse() * occ_cells
        return occ_cells, probs


    # =========================================
    # Velodyne heightmap
    def setup_velodyne(self): 
        # Setup height map computation
        self.hmap = PyHeightMap(grid_dim=300, height_diff_threshold=0.2, m_per_cell=0.3)

        # Setup occupancy map
        self.octmap = PyOctoMap(resolution=0.3, maxlen=self.qlen, decay=0.1)

    def construct_heightmap(self, X, ns='stereo_'): 
        self.hmap.build(X)
        occ, free = self.hmap.getOccupiedCells(), self.hmap.getFreeCells()
        occ_probs, free_probs = np.copy(occ[:,2]), np.copy(free[:,2])
        occ[:,2] = 0; free[:,2] = 0
        
        # Plot height map
        carr = plt.cm.hsv(occ_probs.astype(np.float32) / np.max(occ_probs))[:,:3] * 255
        draw_utils.publish_cloud(''.join([ns,'_obstacles']), occ, carr, frame_id='body')
        carr = plt.cm.hsv(free_probs.astype(np.float32) / np.max(free_probs))[:,:3] * 255
        draw_utils.publish_cloud(''.join([ns,'_free']), free, carr, frame_id='body')

    def run_velodyne(self):
        # Check hmap and octmap
        if not hasattr(self, 'hmap') or not hasattr(self, 'octmap'): 
            raise RuntimeError('Height Map/Octmap is not setup: consider running setup_velodyne()')

        # Velodyne pose
        pose_bv = RigidTransform.from_roll_pitch_yaw_x_y_z(0, 0, 0, -0.27, 0, 1.73)

        for vel_pc in self.dataset.iter_velodyne_frames(): 
            X = vel_pc[:,:3]

            # Collapsed scan
            self.construct_heightmap(pose_bv * X, ns='velodyne')

            # Plot height map
            draw_utils.publish_height_map('velodyne_cloud', pose_bv * X, frame_id='body', height_axis=2)

            yield None
            continue

            # Voxelize velodyne data
            occ_cells, probs = self.voxelize( [X], poses=[RigidTransform.identity()]) 

            # Voxel map plot (color by map z axis)
            carr = self.height_map(cells[:,2]) * 255
            self.publish_voxels('velodyne_octomap', pose_bv * occ_cells, carr, frame_id='body')

    # =========================================
    # Sparse Stereo matcher
    def run_sparse_stereo(self): 
        self.ssm = SparseStereoMatch(uniqueness=0.7, maxDisp=128, leftRightStep=2, 
                                     adaptivity=1.0, minThreshold=10)
        self.ssm.calib_set = False
        self.ssm.set_calibration = lambda H,W: \
                                   self.ssm.set_calib(self.calib_params.P0[:3,:3], 
                                                      self.calib_params.P1[:3,:3], # K0, K1
                                                      np.zeros(5), np.zeros(5), # D0, D1
                                                      self.calib_params.R0, self.calib_params.R1, 
                                                      self.calib_params.P0, self.calib_params.P1, 
                                                      self.calib_params.Q, self.calib_params.T1, 
                                                      W, H # round to closest multiple of 16
                                                  )

        # compute on stereo dataset
        for left_im, right_im in self.dataset.iter_stereo_frames(): 
           
            # One-time set calibration 
            sz = np.array(list(left_im.shape)) - np.array(list(left_im.shape)) % 16
            if not self.ssm.calib_set: 
                self.ssm.set_calibration(sz[0], sz[1])
                self.ssm.calib_set = True

            l, r = left_im[:sz[0],:sz[1]], right_im[:sz[0],:sz[1]]
            st = time.time()
            stereo_vis = self.ssm.process(l,r)
            print 'Time taken for sparse stereo', time.time() - st
            imshow_cv('sparse_stereo', stereo_vis)

    # =========================================
    # Stereo VO
    def setup_vo(self): 
        # Init libVISO2
        self.vo = StereoVISO2(f=self.calib_params.f / self.scale, 
                              cx=self.calib_params.cx / self.scale, cy=self.calib_params.cy / self.scale, 
                              baseline=self.calib_params.baseline) 

    def run_vo(self): 
        # Check vo
        if not hasattr(self, 'vo'): 
            raise RuntimeError('libVISO2 is not setup: consider running setup_vo()')

        # Store past 20 poses
        poses = deque(maxlen=20)

        # compute on stereo dataset
        for left_im, right_im in self.dataset.iter_stereo_frames(): 

            st = time.time()
            T = self.vo.process(left_im, right_im)
            print 'Time taken to compute stereo VO pose', time.time() - st
            poses.append(RigidTransform.from_homogenous_matrix(T))

            # Inv poses (latest/last pose is identity)
            rposes = [(poses[-1].inverse()).oplus(p) for p in poses] 
            # draw_utils.publish_pose_list('viso2_poses', rposes, frame_id='camera_link',size=0.2)

            draw_utils.publish_pose(rposes[0], frame_id='camera')

    # =========================================
    # Ground Plane estimation
    def setup_ground_plane_estimation(self): 
        # Median filtered ground plane
        self.gnd_filtered = deque(maxlen=10)

    def filtered_ground_plane(self, X): 
        gnd = ground_plane_estimation(X)
        self.gnd_filtered.append(gnd)

        gnd_filt = np.vstack(self.gnd_filtered)
        return np.median(gnd_filt, axis=0)

    def density_estimation(self, X, radius=0.1): 
        tree = BallTree(X, leaf_size=2)
        counts = tree.query_radius(X, radius, count_only=True)
        return counts.astype(np.float32) / 10

    def collapsed_scan(self, X): 
        # Plot cloud_votes
        X[:,2] = 0
        probs = self.density_estimation(X, radius=0.3)
        pinds = probs > 0.01
        carr = plt.cm.hsv(probs[pinds])[:,:3] * 255
        self.publish_cloud('cloud_votes', X[pinds], carr, frame_id='map')

    # =========================================
    # Stereo Reconstruction
    def setup_stereo(self, alg='sgbm'): 
        # Stereo block matcher
        stereo_params = StereoSGBM.params_
        stereo_params['minDisparity'] = 0
        stereo_params['numDisparities'] = 128
        if alg == 'sgbm': 
            self.stereo = StereoSGBM(params=stereo_params)
            self.stereo.process = lambda l,r: self.stereo.compute(l,r)
        elif alg == 'elas': 
            self.stereo = StereoELAS()
            # self.stereo.process = lambda l,r: self.stereo.process(l,r)
        elif alg == 'costvolume': 
            self.stereo = OrderedCostVolumeStereo(ordering_constraint=False, 
                                                  cost_volume_filtering='guided',
                                                  gamma=1.0, guide_r=5, guide_eps=0.0001,
                                                  SAD_window_size=3, 
                                                  median_post_processing=False, 
                                                  interpolate_disparities=True)

            self.stereo.process = lambda l,r,m: self.stereo.compute(l,r,m)
        else: 
            raise RuntimeError('Unknown stereo algorithm: %s. Use either sgbm, elas or ordered' % alg)

    def run_stereo(self): 
        # Check stereo
        if not hasattr(self, 'stereo'): 
            raise RuntimeError('Stereo matching is not setup: consider running setup_stereo()')

        # Check vo
        if not hasattr(self, 'vo'): 
            raise RuntimeError('libVISO2 is not setup: consider running setup_vo()')


        # Function helpers for stixel world
        sigmoid = lambda x, mu, s: 1.0/(1.0 + np.exp(-(x - mu)/s))

        # Cam-pose 
        cam_bc = RigidTransform.from_roll_pitch_yaw_x_y_z(0, 1.57, -1.57, 0, 0, 1.65)

        for left_im, right_im in self.dataset.iter_stereo_frames(): 

            # ================================
            # VO on full-size image
            st = time.time()
            T = self.vo.process(left_im, right_im)
            pose_odom  = RigidTransform.from_homogenous_matrix(T)
            _, _pitch, _, _x, _y, _z = pose_odom.to_roll_pitch_yaw_x_y_z()
            pose_odom_ = RigidTransform.from_roll_pitch_yaw_x_y_z(0,_pitch,0,_x,_y,_z)
            print 'Time taken to compute stereo VO pose', time.time() - st
            self.poses.append(pose_odom)
            
            # Camera wrt map, Base wrt map
            pose_mc = cam_bc.oplus(self.poses[-1])
            pose_mb = pose_mc.oplus(cam_bc.inverse())
            pose_bc = (pose_mb.inverse()).oplus(pose_mc)

            draw_utils.publish_tf(pose_mb, frame_id='body', child_frame_id='map')
            draw_utils.publish_tf(pose_mc, frame_id='camera_link', child_frame_id='map')

            # ================================
            # Re-size stereo pair 
            left_im = cv2.resize(left_im, None, fx=self.scale, fy=self.scale, interpolation=cv2.INTER_LINEAR)
            right_im = cv2.resize(right_im, None, fx=self.scale, fy=self.scale, interpolation=cv2.INTER_LINEAR)

            if left_im.ndim == 3: 
                left_img = np.copy(left_im)

                # Convert to grayscale
                left_im = cv2.cvtColor(left_im, cv2.COLOR_BGR2GRAY)
                right_im = cv2.cvtColor(right_im, cv2.COLOR_BGR2GRAY)
            else: 
                left_img = cv2.cvtColor(left_im, cv2.COLOR_GRAY2BGR)

            imshow_cv('stereo', np.hstack([left_im, right_im]))

            # ================================
            # Compute dense disparity, and point cloud
            # sgbm.compute/elas.process
            st = time.time()
            disp = self.stereo.process(left_im, right_im, np.array([]))
            print 'Time taken to stereo', time.time() - st

            # Mask out regions without edges
            mask = edge_utils.dilate(
                edge_utils.sobel_threshold(right_im, blur=5, dx=1, dy=1, threshold=10), 
                iterations=1)
            disp[mask == 0] = 0
            imshow_cv('sobel_threshold', mask)
            print '%% filled: ', np.sum(mask > 0) * 1.0/ (right_im.shape[0] * right_im.shape[1])

            # Plot disparity
            disp8 = np.array((disp * 255)/(64-np.min(disp)), dtype=np.uint8)
            disp_color = color_utils.colormap(disp / 64)
            
            imshow_cv('disparity', disp8)
            imshow_cv('disparity_cmap', disp_color)

            # ================================
            # Reproject to 3D with calib params
            X = cv2.reprojectImageTo3D(disp, self.calib_params.Q)
            X_pub, im_pub = np.copy(X[::self._s,::self._s]).reshape(-1,3), \
                            np.copy(left_img[::self._s,::self._s]).reshape(-1,3)

            # Ensure X_pub doesn't have points below ground plane
            X_pub = X_pub[X_pub[:,1] < 1.65]

            # ================================
            # Ground plane estimation (only the bottom half of the image)
            # gnd = ground_plane_estimation(np.copy(X_pub[::8]))
            # print gnd
            # # gnd = [0, -1, 0, -1.1]
            # X_pubmask = ~reduce(lambda x, y: np.bitwise_or(x, y), [X_pub[:,2] > 75.0, X_pub[:,2] < 1.0, X_pub[:,1] > 1.5])
            X_pubmask = ~reduce(lambda x, y: np.bitwise_or(x, y), [X_pub[:,2] > 75.0, X_pub[:,2] < 1.0])
            X_pub_m, im_pub_m = X_pub[X_pubmask], im_pub[X_pubmask]
            self.Xs.append(X_pub_m), self.ims.append(im_pub)

            draw_utils.publish_height_map('stereo_cloud', cam_bc * X_pub_m, frame_id='body', height_axis=2)

            # ================================
            # Height map representation
            # self.construct_heightmap(pose_bc * X_pub, ns='stereo')

            yield None

            # ================================
            # Register point clouds, and images
            # Voxelize and plot height map
            if self.incremental_add: 
                # Incremental voxelization
                # voxelize with pose_mc if mapping wrt to body frame
                occ_cells, probs = self.incr_voxelize( X_pub_m, pose_mc=pose_mc)
                if occ_cells is None: 
                    continue
                # occ_cells, probs = self.voxelize( [cam_bc * X_pub_m], poses=[RigidTransform.identity()]) 

                # Voxel map plot
                # carr = pose_mc * occ_cells
                # self.publish_voxels('stereo_octomap', occ_cells, frame_id='map', height_axis=1)

                # # Plot with height map/prob map
                # carr = self.height_map(occ_cells[:,2]) * 255
                carr = plt.cm.hsv(probs.flatten())[:,:3] * 255
                draw_utils.publish_voxels('registered_stereo_octomap', occ_cells, carr, frame_id='map')

                # Plot height map from body frame
                # draw_utils.publish_height_map('stereo_cloud', cam_bc * X_pub_m, frame_id='body', height_axis=2)
                draw_utils.publish_height_map('registered_stereo_cloud', pose_mc * X_pub_m, frame_id='map', height_axis=2)

            else: 
                # Inv poses (latest/last pose is identity)
                rposes = [(self.poses[-1].inverse()).oplus(p) for p in self.poses] 

                Xall = np.vstack([p_ * X_ for (p_, X_) in zip(rposes, self.Xs)])
                xinds = np.isfinite(Xall).all(axis=1)

                self.construct_heightmap(pose_bc * Xall, ns='stereo')

                # Xall = [p_ * X_ for (p_, X_) in zip(rposes, self.Xs)]
                occ_cells, probs = self.voxelize(Xall[xinds]) # , poses=rposes)
                if occ_cells is None: 
                    continue

                # Voxel map plot
                # Plot with height map/prob map
                # carr = self.height_map(cells[:,height_axis]) * 255
                carr = plt.cm.hsv(probs.flatten())[:,:3] * 255
                draw_utils.publish_voxels('registered_stereo_octomap', occ_cells, carr, frame_id='camera')

                # Height map coloring based on z-axis
                draw_utils.publish_height_map('registered_stereo_cloud', 
                                     np.vstack(Xall), frame_id='camera', height_axis=1)


        # OPTIONALLY median blur height map
        # Xh = cv2.medianBlur(Xh, 5)
        
        # TRUNCATE heightmap
        # _, Xh = cv2.threshold(Xh, 1.0, 1.0, cv2.THRESH_TRUNC)
        # Xh_cm = cv2.applyColorMap(np.array(Xh * 255 / 10.0, dtype=np.uint8), cv2.COLORMAP_JET)
        # imshow_cv('Xh_cm', Xh_cm)

        # STIXEL WORLD
        # tval = 0.2
        # stixel = stixel_world(cloud=Xh, max_height=0.2, blur=True)
        # _, stixel_trunc = cv2.threshold(stixel, tval, 1.0, cv2.THRESH_TRUNC)
        # imshow_cv('stixel_prob', stixel_trunc)

        # SEGMENTATION VIA DIFFERENCE IN NORMALS
        # # Compute Normals
        # N = compute_normals(X)
        # N[np.isnan(N)] = 0
        # left_img = cv2.cvtColor(left_im, cv2.COLOR_GRAY2RGB)
        # imshow_cv('normals', N)
        # # imshow_cv('normals', N * 0.25 + left_img.astype(np.float32)*0.75/255.0)
        # gamma = 0.2
        # imshow_cv('dot', 1-np.exp(-np.dot(N, np.array([0,1,0])) * gamma))

        # OCTMAP construction
        # # Insert data into octmap
        # # st = time.time()
        # self.octomap_stereo.updateNodes(X, True)
        # # print 'Time taken to update octree', time.time() - st

        # octomap = self.octomap_vel.getMap()
        # print 'Octomap keys: ', octomap.keys()
        # draw_utils.publish_octomap(octomap, frame_id='camera')

        # self.octomap_stereo.writeBinary('octmap_stereo.bt')
        # self.octomap_vel.writeBinary('octmap_vel.bt')

        
        # imshow_cv('disparity', (disp-np.min(disp))/128)
        # imshow_cv('elas_disparity', 
        # (disp2-self.sgbm.params['minDisparity'])/self.sgbm.params['numDisparities'])
        # imshow_cv('im', np.vstack([left_im, right_im]))

    # def project_pts(self): 
    #     # Project points back into image
    #     occ_X = X
    #     occ_color = carr
    #     rvec,_ = cv2.Rodrigues(self.calib_params.R0)
    #     proj,_ = cv2.projectPoints(occ_X.astype(np.float32), rvec, self.calib_params.T0, 
    #                                 self.calib_params.P0[:3,:3], np.zeros(4))
    #     occ_x = proj.astype(np.int32).reshape((-1, 2))
    #     inds, = np.where((occ_x[:,0] >= 5) & (occ_x[:,0] < self.calib_params.cx*2-5) & 
    #                      (occ_x[:,1] >= 5) & (occ_x[:,1] < self.calib_params.cy*2-5))

    #     return occ_x[inds], occ_color[inds] * 255

    # def sgbm_reduced(self, left_im, right_im): 
    #     disp_scale = 16
    #     left_im_blur = cv2.GaussianBlur(left_im, (3,3), 0)
    #     right_im_blur = cv2.GaussianBlur(right_im, (3,3), 0)

    #     # ================================
    #     # ELAS for prior disparity
    #     elas_disp = self.stereo.process(left_im, right_im)

    #     # Set maximum number of prior disparities
    #     if len(self.prior_disps) < 5: 
    #         self.prior_disps.append(np.copy(elas_disp))
    #     imshow_cv('prior_disps', reduce(lambda x, v: np.vstack([x,v]), self.prior_disps) * 1.0 / 255)

    #     # ================================
    #     # Compute median disp across past frames
    #     all_disps = reduce(lambda x, v: np.dstack([x,v]), self.prior_disps)
    #     prior_median_disp = np.median(all_disps, axis=2) if all_disps.ndim > 2 else np.copy(all_disps)

    #     # Ordered, and filtered median
    #     prior_median_disp[prior_median_disp < 5] = 0
    #     imshow_cv('median_disps', prior_median_disp * 1.0 / 128)
    #     prior_median_disp = ordered_row_disparity(prior_median_disp)
    #     imshow_cv('ordered_median_disps', prior_median_disp * 1.0 / 128)

    #     # Compute variance on past disparities
    #     prior_var_disp = np.std(all_disps, axis=2) if all_disps.ndim > 2 else all_disps
    #     imshow_cv('var_disps', prior_var_disp * 5.0 / 128)

    #     print 'Minmax: ', np.min(prior_median_disp), np.max(prior_median_disp)

    #     # ================================
    #     # Fast cost volume filtering
    #     st = time.time()
    #     disp = fast_cost_volume_filtering(left_im, right_im, prior_median_disp) # np.empty([]))
    #     disp = disp.astype(np.float32) / (disp_scale)
    #     print 'Volume Minmax: ', np.min(disp), np.max(disp)
    #     # disp = stereo_with_tree_filtering(left_im, right_im)
    #     print 'Time taken to sgbm_test', time.time() - st
    #     imshow_cv('disparity', disp.astype(np.float32) / 128)

    #     # Difference b/w median elas disp and current disparity
    #     imshow_cv('diff_disp', (np.abs(disp - prior_median_disp) > 2).astype(np.uint8) * 255)

    #     # ================================
    #     # Remove pts that have high var
    #     # inds = np.abs(disp - prior_median_disp) < 2
    #     # disp[inds] = prior_median_disp[inds]
    #     # imshow_cv('fused_disparity', disp.astype(np.float32) / 128)

    #     # dispm = cv2.bilateralFilter(disp, 5, 5*2, 5/2)
    #     dispm = cv2.medianBlur(disp, 5)
    #     imshow_cv('disp bil', dispm / 128)        

    #     return disp

    # def debug_sgbm_reduced(self, left_im, right_im): 
    #     disp = self.sgbm_reduced(left_im, right_im)

    #     # ================================
    #     # SGBM test 

    #     left_im = cv2.GaussianBlur(left_im, (3,3), 0)
    #     right_im = cv2.GaussianBlur(right_im, (3,3), 0)

    #     gx = cv2.Sobel(left_im, cv2.CV_32F, 1, 0)
    #     xinds = gx[left_im.shape[0]/2,:] > 200

    #     # gy = cv2.Sobel(left_im, cv2.CV_32F, 0, 1)
    #     # yinds = gy[:,left_im.shape[1]/2] > 200


    #     disp_color = cv2.cvtColor(disp, cv2.COLOR_GRAY2BGR)
    #     # disp_color[:,xinds,1] = 200
    #     # disp_color[yinds,:,1] = 200

    #     imshow_cv('disparity', disp_color)

    #     # # ================================
    #     # # CostVolume estimation on full-size image
    #     # st = time.time()
    #     # self.cost_vol.compute(left_im, right_im)
    #     # print 'Time taken to compute cost volume', time.time() - st

