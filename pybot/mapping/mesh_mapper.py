import cv2
import numpy as np
from collections import deque, namedtuple, defaultdict
from itertools import izip

from pybot.geometry.rigid_transform import Pose, RigidTransform
from pybot.utils.db_utils import AttrDict
from pybot.utils.misc import Counter, Accumulator, CounterWithPeriodicCallback 
from pybot.utils.timer import SimpleTimer, timeitmethod
from pybot.utils.misc import print_green, print_red, print_yellow
from pybot.vision.image_utils import to_color, to_gray
from pybot.vision.imshow_utils import imshow_cv, print_status
from pybot.utils.pose_utils import KeyframeSampler, Keyframe
 
from pybot.vision.camera_utils import Camera, CameraIntrinsic, CameraExtrinsic
from pybot.vision.camera_utils import triangulate_points, sampson_error, plot_epipolar_line

import pybot.externals.draw_utils as draw_utils

from pybot.mapping.slam import RobotVisualSLAM
from pybot.vision.draw_utils import draw_features, draw_matches
from pybot.vision.feature_detection import FeatureDetector
from pybot.vision.feature_detection import to_pts
from pybot.vision.trackers import OpenCVKLT, OpticalFlowTracker

# from pybot_vision import scaled_color_disp

# Testing
from pybot.mapping.gtsam import two_view_BA
from pybot.vision.camera_utils import triangulate_points, compute_essential, decompose_E

# MapPoint = namedtuple('MapPoint', ['id', 'pt', 'pt3', 'parallax'], verbose=False)


def filter_sampson_error(cam1, cam2, pts1, pts2, matched_ids, error=4): 
    """
    Filter feature matches via sampson error given the
    two camera extrinsics.
    """

    # Determine inliers (via sampson error)
    F = cam1.F(cam2)
    err = sampson_error(F, pts2, pts1)
    inliers, = np.where(np.fabs(err) < error)

    # Retain only inliers
    pts1, pts2, matched_ids = pts1[inliers], \
                              pts2[inliers], \
                              matched_ids[inliers]

    assert(len(pts1) == len(pts2) == len(matched_ids))
    return pts1, pts2, matched_ids

# def filter_parallax(self, cam, kf_pts, frame_pts, matched_ids): 

#     # Parallax check
#     min_parallax = np.float32([ self.kf_mpts_[tid].parallax for tid in matched_ids ])
#     kf_rays = self.cam_.ray(kf_pts, rotate=True)
#     frame_rays = cam.ray(frame_pts, rotate=True)

#     kf_rays_norm = np.linalg.norm(kf_rays, axis=1)
#     frame_rays_norm = np.linalg.norm(frame_rays, axis=1)
#     cosine_parallax = np.multiply(kf_rays, frame_rays).sum(axis=1) / (kf_rays_norm * frame_rays_norm)

#     # Retain only better parallax measures
#     good, = np.where(cosine_parallax < min_parallax)
#     for (cdist, tid) in izip(cosine_parallax[good], matched_ids[good]): 
#         self.kf_mpts_[tid].parallax = cdist

#     # Retain only the good matches from now onwards
#     kf_pts, frame_pts, matched_ids = kf_pts[good], \
#                                      frame_pts[good], \
#                                      matched_ids[good]

#     assert(len(kf_pts) == len(frame_pts) == len(matched_ids))
#     return kf_pts, frame_pts, matched_ids

class TrackReconstruction(object):
    """
    Reconstruct tracked features via Keyframe-Keyframe matches
    """
    def __init__(self, mapper, cam):
        self.cam_ = cam
        # self.mapper_ = mapper

        # KF-KF (has to be 3 to distinguish between
        # just added 2 frames, and added frames into
        # stream of deque)
        self.kf_items_q_ = Accumulator(maxlen=2)
        
        # Visual ISAM2 with KLT tracks
        self.vslam_ = RobotVisualSLAM(self.cam_.intrinsics, 
                                      min_landmark_obs=4, px_error_threshold=10,
                                      update_every_k_odom=1, 
                                      odom_noise=np.ones(6) * 0.5, 
                                      px_noise=np.ones(2) * 2.0,
                                      prior_point3d_noise=np.ones(3) * 0.01,
                                      verbose=False)
        
    @timeitmethod
    def on_frame(self, fidx, frame, kf_ids, kf_pts):
        """
        Keyframe-to-Keyframe matching
        """
        # Add KF items to queue
        self.kf_items_q_.accumulate(
            AttrDict(fidx=fidx, frame=frame, ids=kf_ids, pts=kf_pts, 
                     cam=Camera.from_intrinsics_extrinsics(
                         self.cam_.intrinsics, frame.pose.inverse())))

        # ---------------------------
        # 1. VSLAM ADD ODOM

        # Add odometry measurements incrementally
        self.vslam_.on_odom_absolute(fidx, frame.pose)

        # Add pose prior on second pose node (scale ambiguity)
        if fidx == 1:
            self.vslam_.add_pose_prior(fidx, frame.pose)
        
        # ---------------------------
        # 2. KF-KF matching

        # Here kf1 (older), kf2 (newer)
        kf2 = self.kf_items_q_.items[-1]
        fidx2, frame2, kf_ids2, kf_pts2, cam2 = kf2.fidx, kf2.frame, kf2.ids, kf2.pts, kf2.cam

        # Continue if only the first frame
        if len(self.kf_items_q_) < 2:
            return
        
        kf1 = self.kf_items_q_.items[-2]
        fidx1, frame1, kf_ids1, kf_pts1, cam1 = kf1.fidx, kf1.frame, kf1.ids, kf1.pts, kf1.cam

        kf_pts1_lut = {tid: pt for (tid,pt) in izip(kf_ids1,kf_pts1)}
        kf_pts2_lut = {tid: pt for (tid,pt) in izip(kf_ids2,kf_pts2)}

        # # Add features to the map
        # self.mapper_.add_points(fidx, kf_ids2, kf_pts2)

        # Find matches in the newer keyframe that are consistent from 
        # the previous frame
        matched_ids = np.intersect1d(kf_ids2, kf_ids1)
        if not len(matched_ids):
            return 

        # Keyframe and Frame points
        # print_yellow('{:}-{:} KF-KF {:} intersect1d {:} = {:}'
        #              .format(fidx1, fidx2, len(kf_ids1), len(kf_ids2), len(matched_ids)))
        kf_pts1 = np.vstack([ kf_pts1_lut[tid] for tid in matched_ids ])
        kf_pts2 = np.vstack([ kf_pts2_lut[tid] for tid in matched_ids ])

        fvis = draw_matches(frame2.img, kf_pts1, kf_pts2, colors=np.tile([0,0,255], [len(kf_pts1), 1]))
        npts1 = len(kf_pts1)

        # ---------------------------
        # 3. FILTERING VIA SAMPSON ERROR 

        # # Filter matched IDs based on epipolar constraint
        # # use sampson error (two-way pixel error)
        # kf_pts1, kf_pts2, matched_ids = filter_sampson_error(
        #     cam1, cam2, kf_pts1, kf_pts2, matched_ids, error=2
        # )
        # if not len(matched_ids): return         
        # npts2 = len(kf_pts1)

        # ---------------------------
        # 3. FILTERING VIA Fundamental matrix RANSAC

        # Fundamental matrix estimation
        method, px_dist, conf =  cv2.cv.CV_FM_RANSAC, 3, 0.99
        (F, inliers) = cv2.findFundamentalMat(kf_pts1, kf_pts2, method, px_dist, conf)
        inliers = inliers.ravel().astype(np.bool)
        kf_pts1, kf_pts2 = kf_pts1[inliers], kf_pts2[inliers]
        matched_ids = matched_ids[inliers]
        npts2 = len(kf_pts1)

        # Test BA
        E = compute_essential(F, cam1.K)
        R1, R2, t = decompose_E(E)
        print 'E', E
        print 'R1/R2/t', R1, R2, t
        X = triangulate_points(cam1, kf_pts1, cam2, kf_pts2)
        two_view_BA(cam1, kf_pts1, kf_pts2,
                    X, frame1.pose.inverse() * frame2.pose, scale_prior=True)
        
        # -----------------------------
        # Visualize
        fvis = draw_matches(fvis, kf_pts1, kf_pts2,
                            colors=np.tile([0,255,0], [len(kf_pts1), 1]))
        print_yellow('Matches {:}, Sampson Filtered {:}, Parallax filter'.format(npts1, npts2, npts2))
        imshow_cv('vis_matches', fvis)

        # -----------------------------
        # 4. VSLAM ADD LANDMARKS (INLIER MEASUREMENTS)
        # Add landmarks incrementally
        # Add measurements for the keyframes (as smartfactors)

        # When enough landmarks observed in the first 2 frames,
        # Add the previous frame's measurements
        if self.kf_items_q_.length == 2:
            self.vslam_.on_point_landmarks_smart(fidx-1, matched_ids,
                                                 kf_pts1, keep_tracked=True)

        # Add landmarks incrementally
        self.vslam_.on_point_landmarks_smart(fidx, matched_ids,
                                             kf_pts2, keep_tracked=True)

        
        # # =================================
        # # VSLAM UPDATE

        # if len(self.vslam_poses_) >= 2: 
        #     self.vslam_.update()
        #     ids, pts3 = self.vslam_.smart_update()

        #     # Update KF values
        #     self.mapper_.update_points3d(ids, pts3)

        #     draw_utils.publish_cloud('gtsam-pc', pts3, c='b', frame_id='camera', reset=fidx==0)

        # =================================
        # Prune KF_X: Ideally this should happen only when so many
        # keyframes are instantiated
        # self.mapper_.prune(fidx)

        # for tid in self.kf_X_.keys(): 
        #     if abs(fidx-self.kf_X_[tid].fidx) > 50: 
        #         del self.kf_X_[tid]

        # # Reset KF values 
        # for (tid, pt) in izip(kf_ids, kf_pts): 
        #     if tid not in self.kf_mpts_:
        #         self.kf_mpts_[tid] = AttrDict(id=tid, pt=pt, pt3=np.zeros(3)*np.nan, parallax=0.9998)

        # # Update KF values
        # self.kf_mpts_.update({ tid: AttrDict(id=tid, pt=pt, pt3=np.zeros(3) * np.nan, parallax=0.9998) 
        #                        for (tid, pt) in izip(kf_ids, kf_pts) if tid not in self.kf_mpts_ })

        # # Reset KF values 
        # self.kf_mpts_ = { tid: AttrDict(id=tid, pt=pt, pt3=np.zeros(3) * np.nan, parallax=0.98) 
        #                   for (tid, pt) in izip(kf_ids, kf_pts) }

    @timeitmethod
    def frame_visualization(self, fidx, frame):
        cam = Camera.from_intrinsics_extrinsics(
            self.cam_.intrinsics, frame.pose.inverse())
        pose = frame.pose
        vis = to_color(frame.img)
        vis = cam.undistort(vis)
        
        # Publish frame pose
        draw_utils.publish_pose_t('CAMERA_POSE', cam.w2o, frame_id='camera')

        # # Publish cloud along with keyframe
        # draw_utils.publish_cameras('vio-kf', [Pose.from_rigid_transform(fidx, pose)], frame_id='camera', 
        #                            size=2, zmax=0.1, reset=fidx==0, draw_nodes=False)
        
        # Visualize keyframe disparities
        # Color triangulated keypoints based on depth
        # Determine pt xy and depth for depth-based coloring 
        # draw_utils.publish_botviewer_image_t(vis.copy())
        # try:
        #     xy, X = self.mapper_.retrieve_map_points(fidx)
        #     # xy, depths = cam.project(X, check_bounds=True, return_depth=True)
        # except Exception as e:
        #     # print('frame_visualization exception {}'.format(e))
        #     imshow_cv('vis', vis)
        #     return


        # # Publish cloud
        # if len(X): 
        #     # Project all valid depths onto keyframe
        #     # depths = cam.depth_from_projection(X)

        #     in_front = np.bitwise_and(depths > 0, depths < 10)
        #     xyd = np.hstack([xy[in_front], depths[in_front].reshape(-1,1)])

        #     if len(xyd): 
        #         # Color depth and plot
        #         cols = scaled_color_disp(xyd[:,2], 16).reshape(-1,3).astype(np.int64)
        #         for (pt, col) in izip((xyd[:,:2]).astype(np.int64), cols): 
        #             tl, br = (pt-2).astype(int), (pt+2).astype(int)
        #             cv2.rectangle(vis, (tl[0], tl[1]), (br[0], br[1]), tuple(col), -1)
        #     # draw_utils.publish_cloud('vio-kf-pc', X, frame_id='camera', reset=fidx==0)

        # draw_utils.publish_cloud('vio-kf-pc', X, frame_id='vio-kf', element_id=fidx, reset=fidx==0)

        # imshow_cv('vis', vis)

    def on_track_delete(self, tracks):
        pass
        # for tid in tracks.keys(): 
        #     if tid in self.kf_mpts_: 
        #         del self.kf_mpts_[tid]

class MeshMapper(object):
    def __init__(self, lag=50): 
        self.lag_ = lag

        # Map points data
        self.X_ = defaultdict(lambda: AttrDict(pt3=np.zeros(3) * np.nan, pt=np.zeros(2)-1, fidx=-1))
        self.E_ = {}

    def add_points(self, fidx, ids, pts): 
        for (tid,pt) in izip(ids, pts):
            self.X_[tid].pt = pt
            self.X_[tid].fidx = fidx

    def update_points3d(self, ids, pts3): 
        for (tid, pt3) in izip(ids, pts3): 
            self.X_[tid].pt3 = pt3            

    def prune(self, fidx): 
        for tid in self.X_.keys(): 
            if abs(fidx-self.X_[tid].fidx) > self.lag_: 
                del self.X_[tid]

    def retrieve_map_points(self, fidx, prune=False): 

        # Retrieve valid 3D points
        # valid = map(lambda item: np.isfinite(item.pt3).all() and item.fidx == fidx, self.X_.itervalues())
        valid = map(lambda item: np.isfinite(item.pt3).all() and abs(item.fidx-fidx) < 10, self.X_.itervalues())

        X, xy = [], []
        for (v,item) in izip(valid, self.X_.itervalues()): 
            if v: 
                X.append(item.pt3)
                xy.append(item.pt)
        X = np.vstack(X)
        xy = np.vstack(xy)

        # # Optionally add line segments every so often
        # lines = []
        # for (v, (tid,item)) in izip(valid, self.X_.iteritems()): 
        #     # Only draw lines between inserted edges
        #     if not v or tid not in self.E_: continue 
            
        #     v1,v2 = tid, self.E_[tid]
        #     if v2 in self.X_: 
        #         lines.append(np.hstack([self.X_[v1].pt3, self.X_[v2].pt3]))
        # print 'Lines: ', len(self.X_), len(lines)
        
        # if len(lines): 
        #     edges = np.vstack(lines)
        #     v1, v2 = edges[:,:3], edges[:,3:6]
        #     valid = np.linalg.norm(v1-v2, axis=1) < 1
        #     draw_utils.publish_line_segments('mesh-lines', v1[valid], v2[valid], frame_id='camera', reset=fidx==0)


        # for (v1,v2) in self.E_.iteritems(): 
        #     if v1


        # if prune: 
        #     # Cleanup 
        #     for (v,tid) in izip(valid, self.X_.keys()): 
        #         if not v: del self.X_[tid]
        
        return xy, X
        
    def add_edges(self, ids, edges): 
        for (i1,i2) in edges: 
            v1,v2 = ids[i1], ids[i2] 
            self.E_[v1] = v2
            self.E_[v2] = v1
            # assert(v1 in self.X_ and v2 in self.X_)

    def add_faces(self, faces): 
        pass

class MeshReconstruction(object): 
    """
    Reconstruct tracked features and meshing
    """
    def __init__(self, calib): 
        self.cam_ = Camera.from_intrinsics(calib)

        # =================
        # Keyframe sampler
        self.kf_sampler_ = KeyframeSampler(theta=np.deg2rad(20), displacement=2, lookup_history=30, 
                                           on_sampled_cb=self.on_keyframe, verbose=False)

        # =================
        # Setup detector params
        detector_params = AttrDict(method='fast', grid=(12,9), max_corners=150, 
                                   max_levels=1, subpixel=True, params=FeatureDetector.fast_params)

        # Setup tracker params (either lk, or dense)
        tracker_params = AttrDict(method='lk', fb_check=True, 
                                  params=AttrDict(winSize=(21,21), maxLevel=3))

        # Create mesh klt
        from pybot.vision.trackers import MeshKLT
        self.mklt_ = MeshKLT.from_params(detector_params=detector_params, 
                                         tracker_params=tracker_params, 
                                         min_tracks=150, min_track_length=0, max_track_length=8)
        self.mklt_.register_on_track_delete_callback(self.on_track_delete)

                
        # =================
        # MeshMapper
        self.mapper_ = MeshMapper()
        self.f_counter_, self.kf_counter_ = Counter(), Counter()

        # =================
        # Keyframe-Frame tracker
        self.track_reconstruction_ = TrackReconstruction(self.mapper_, self.cam_)


    @timeitmethod
    def on_frame(self, t_pose, t_img, pose, img): 
        # Determine overlapping/tracked features
        # between keyframe and frame
        self.f_counter_.count()
        fidx = self.f_counter_.index

        # Visualize meshKLT
        mids, mpts = self.mklt_.process(img)
        # mesh_edges = self.mklt_.edges
        # self.mapper_.add_edges(mids, mesh_edges)

        self.mklt_.visualize(img, mids, mpts, colored=True)
        # print_yellow('MIDS {:}, MPTS {:}'.format(len(mids), len(mpts)))
                
        # Keyframe Sampler
        frame = Keyframe(pose=pose, img=img, index=fidx)
        self.kf_sampler_.append(frame)

        # # Frame-to-Keyframe tracking
        # # Provide track reconstruction with latest frame ids, and points
        # frame_ids, frame_pts = mids, mpts
        # if len(frame_pts): 
        #     frame_pts = self.cam_.undistort_points(frame_pts)
        # self.track_reconstruction_.on_frame(fidx, frame, frame_ids, frame_pts)

        # # Visualize keyframes
        # self.track_reconstruction_.frame_visualization(fidx, frame)

    def on_keyframe(self, fidx, frame): 
        self.kf_counter_.count()
        kfidx = self.kf_counter_.index

        # Keyframe-to-Keyframe tracking
        # Provide track reconstruction with latest KF ids, and points
        kf_ids, kf_pts = self.mklt_.latest_ids, self.mklt_.latest_pts
        if len(kf_pts): 
            kf_pts = self.cam_.undistort_points(kf_pts)
        self.track_reconstruction_.on_frame(kfidx, frame, kf_ids, kf_pts)

        # Visualize keyframes
        self.track_reconstruction_.frame_visualization(kfidx, frame)


    def on_track_delete(self, tracks): 
        self.track_reconstruction_.on_track_delete(tracks)

        # print('Removing {:} tracks'.format(len(tracks)))

        # for tid, track in tracks.iteritems(): 
        #     if track.length < 5: continue
        #     print tid, track.length
