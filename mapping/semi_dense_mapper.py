"""
=====================================================================
 LSD-SLAM Mapper 
=====================================================================

Map the environment in a semi-dense fashion from RGB image sequences

Run as: 

For all scenes: 
  >> python lsd_mapper.py

For specific target scenes: 
  >> python lsd_mapper.py -s scene_01,scene_02,scene_03

"""

# Author: Sudeep Pillai <spillai@csail.mit.edu>
# License: TODO

import numpy as np
import cv2, os, time
from collections import OrderedDict

from bot_vision.color_utils import get_color_by_label
from bot_vision.image_utils import median_blur
from bot_vision.imshow_utils import imshow_cv
from bot_utils.db_utils import AttrDict
from bot_geometry.rigid_transform import Pose, Quaternion, \
    RigidTransform, Sim3, normalize_vec

import bot_externals.draw_utils as draw_utils

class Mapper(object): 
    """ 
    
    SLAM interface 
    ==============

    Database to store map-related data. 
    
    Stored values: 
       poses, keyframes, points (for each keyframe), colors (for each point)


    
        Parameters
        ----------
        poses: List of RigidTransform(s)
           All tracked poses during mapping
        
        keyframes: List of RigidTransform(s)
           Keyframes used for pose graph optimization

        points: List of numpy array [N x 3] 
           Points reconstructed at each keyframe

        colors: List of numpy array [N x 3] 
           Colors corresponding to each point reconstructed

    """
    def __init__(self, poses=None, keyframes=None, update_every=20, update_cb=lambda: None): 
        self.poses = poses
        self.keyframes = keyframes
        self.update_every = update_every
        self.update_cb = update_cb

        if keyframes is None: 
            return

        # Keyframe lookup  (map frame index to keyframe)
        self.kf_idx = {kf.frame_id: kf.id for kf in self.keyframes.itervalues()}

        # Clean up keyframe points, colors
        for k_id in self.keyframes.keys(): 
            points = self.keyframes[k_id].points
            colors = self.keyframes[k_id].colors

            if points is None or colors is None: 
                continue

            colors = np.vstack(colors)
            points = np.vstack(points)
            inds = np.isfinite(points).all(axis=1)
            self.keyframes[k_id].points = points[inds]
            self.keyframes[k_id].colors = colors[inds]

    def add(self, pose): 
        self.poses.append(pose)

        # Update keyframes 
        if len(self.poses) % self.update_every == 0: 
            try: 
                self.update_cb()
                self.publish()
            except: 
                raise RuntimeError('Failed to update, check if update_cb is prepared')
            
    def fetch_keyframe(self, frame_id): 
        return self.keyframes[self.kf_idx[frame_id]]

    @classmethod
    def load(cls, path):
        db = AttrDict.load(path)
        keyframes = OrderedDict({kf.id:kf for kf in db.keyframes})
        return cls(db.poses, keyframes)

    def save(self, path): 
        """
        Save map, and relevant poses, keyframes, colors etc
        Not return class specific dictionary: 
        """
        print 'Saving keyframe ids: ', self.keyframes.keys()
        db = AttrDict(poses=self.poses, 
                      keyframes=self.keyframes.values())
        db.save(path)
        
    def publish(self): 
        """

        Publish map components: 
           poses: Intermediate poses, i.e. tracked poses (usually unoptimized)
           keyframes: Optimized keyframes (this will be updated every so often)
           points: Landmark features that are also optimized along with keyframes

        """
        draw_utils.publish_pose_list('slam_poses', self.poses, frame_id='camera')

        kf_ids = self.all_keyframe_ids
        kf_pts = self.all_points
        kf_cols = self.all_colors
        kf_poses = self.all_keyframes

        draw_utils.publish_cameras('slam_keyframes', kf_poses, frame_id='camera', draw_faces=False)
        draw_utils.publish_cloud('slam_keyframes_cloud', kf_pts, c=kf_cols, frame_id='camera')
        # draw_utils.publish_cloud('slam_keyframes_cloud', kf_pts, c=kf_cols, element_id=kf_ids, frame_id='slam_keyframes')
        
    @property
    def all_points(self): 
        return [kf.points for kf in self.keyframes.itervalues() if kf.points is not None]

    @property
    def all_colors(self): 
        return [kf.colors for kf in self.keyframes.itervalues() if kf.colors is not None ]

    @property
    def all_keyframe_ids(self): 
        return [kf.id for kf in self.keyframes.itervalues()]

    @property
    def all_keyframes(self): 
        return [kf.pose for kf in self.keyframes.itervalues()]

class LSDMapper(Mapper): 
    """ LSD-SLAM interface """
    default_params = AttrDict(
        calib = np.array([528.49404721/640, 528.49404721/480, 319.5/640, 239.5/480, 0,0,0], 
                         dtype=np.float64),
        in_width=640, in_height=480,
    )
    def __init__(self, params=default_params): 
        Mapper.__init__(self, poses=[], keyframes=OrderedDict(), update_cb=self._update_keyframes)
        
        from pybot_externals import LSDSLAM
        self.slam = LSDSLAM(**params)
        self.poses = []
        self.keyframes = OrderedDict()

    def _update_keyframes(self): 
        # Keyframe graph for updates
        kf_data = self.slam.getKeyframeGraph()

        # Only plot until penultimate
        for kf in kf_data: 
            k_id = kf.getId()
            pts, colors, pose = kf.getPoints(), kf.getColors(), kf.getPose()
            pose_id = Sim3.from_homogenous_matrix(pose)
            pose_id.id = k_id # HACK

            assert(len(pts) == len(colors))
            self.keyframes[k_id] = AttrDict(id=k_id, 
                pose=pose_id, points=kf.getPoints(), colors=kf.getColors()
            )

    def process(self, im): 
        # Process input frame
        self.slam.process(im)

        # Retain intermediate poses
        T = self.slam.getCurrentPoseEstimate()
        self.add(Sim3.from_homogenous_matrix(T))

        # self.poses.append(RigidTransform.from_homogenous_matrix(T))

        # # Show debug output
        # if len(self.keyframes): 
        #     vis = self.slam.getDebugOutput()
        #     draw_utils.publish_image_t('lsd_debug', vis)
        #     print 'vis', vis.shape
        #     # imshow_cv('vis', vis)


    def optimize(self): 
        # Final optimization on keyframes/map
        self.slam.optimize()

class ORBMapper(Mapper): 
    """ ORB-SLAM interface """
    default_params = AttrDict(
        K = np.array([[528.49404721, 0, 319.5], 
                      [0, 528.49404721, 239.5],
                      [0, 0, 1]], dtype=np.float64), 
        W = 640, H = 480
    )
    def __init__(self, params=default_params): 
        Mapper.__init__(self, poses=[], keyframes=OrderedDict(), update_cb=self._update_keyframes)
        from pybot_externals import ORBSLAM

        path = '/home/spillai/perceptual-learning/software/python/apps/config/orb_slam/'
        self.slam = ORBSLAM(settings=path + 'Settings_udub.yaml', vocab=path + 'ORBvoc.yml')


    def _update_keyframes(self): 
        # Keyframe graph for updates
        kf_data = self.slam.getKeyframeGraph()

        # Only plot until penultimate
        for kf in kf_data: 
            k_id = kf.getId()
            f_id = kf.getFrameId()
            pose_cw = RigidTransform.from_homogenous_matrix(kf.getPose())
            pose_wc = pose_cw.inverse()
            cloud_w = kf.getPoints()
            # cloud_c = pose_cw * cloud_w
            colors = (np.tile([0,0,1.0], [len(cloud_w),1])).astype(np.float32)

            # print k_id, len(cloud_w), len(cloud_c)

            # For now, add Pose with ID to pose, not necessary later on
            self.keyframes[k_id] = AttrDict(id=k_id, frame_id=f_id, 
                                            pose=Pose.from_rigid_transform(k_id, pose_wc), 
                                            points=cloud_w, colors=colors, img=kf.getImage())

        print 'Keyframes: ', len(self.keyframes), 'Poses: ', len(self.poses)

    def process(self, im): 
        # Process input frame
        self.slam.process(im)
        
        # Retain intermediate poses
        Tcw = self.slam.getCurrentPoseEstimate()

        try:
            pose_cw = RigidTransform.from_homogenous_matrix(Tcw)
            pose_wc = pose_cw.inverse()
        except: 
            return

        # Add pose to map
        self.add(pose_wc)

        imshow_cv('orb-slam', im)
    
class MultiViewMapper(Mapper): 
    """ 
    
    SVO-Depth-Filter based reconstruction
    
    Notes: 
      1. Only reconstruct semi-densely after first 5 keyframes
      2. Increased patch size in svo depth filter to 16 from 8
    
    """
    default_params = AttrDict(
        K = np.array([[528.49404721, 0, 319.5], 
                      [0, 528.49404721, 239.5],
                      [0, 0, 1]], dtype=np.float64), 
        W = 640, H = 480
        ,gridSize=30, nPyrLevels=1, max_n_kfs=4
    )
    def __init__(self, params=default_params): 
        from pybot_externals import SVO_DepthFilter
        self.depth_filter = SVO_DepthFilter(**params)
        self.kf_every = 5
        self.idx = 0

    def load(self, path):
        db = AttrDict.load(path)
        keyframes = OrderedDict({kf.id:kf for kf in db.keyframes})
        Mapper.__init__(self, db.poses, keyframes)

    def save(self, path): 
        print 'Saving keyframe ids: ', self.keyframes.keys()
        db = AttrDict(poses=self.poses, 
                      keyframes=self.keyframes.values())
        db.save(path)
        

    def process_img(self, img, pose_wc): 
        # Only process once for semi-dense depth estimation
        self.depth_filter.process(img, (pose_wc.to_homogeneous_matrix()).astype(np.float64), self.idx % self.kf_every == 0)
        imshow_cv('kf_img', img)

        try: 
            cloud = self.depth_filter.getPoints()
            colors = (self.depth_filter.getColors() * 1.0 / 255).astype(np.float32)
            # colors = (np.tile([1.0,0,0], [len(cloud),1])).astype(np.float32)
            draw_utils.publish_cloud('depth_keyframe_cloud', cloud, c=colors, frame_id='camera')
        except Exception as e: 
            print e

        self.idx += 1
        
    def process(self):
        # Show sparse map and keyframes
        self.publish()

        # Multi-view semi-dense reconstruction
        for kf_id in self.keyframes.keys(): 
            self.keyframes[kf_id].points = None
            self.keyframes[kf_id].colors = None
            if kf_id < 20: 
                continue
            self.process_img(self.keyframes[kf_id].img, self.keyframes[kf_id].pose)
            draw_utils.publish_cameras('current_keyframe', [self.keyframes[kf_id].pose], frame_id='camera', size=2)

        # HACK: save to first keyframe
        self.keyframes[0].points = self.depth_filter.getPoints()
        self.keyframes[0].colors = (self.depth_filter.getColors() * 1.0 / 255).astype(np.float32)


class SemiDenseMapper(object): 
    default_params = AttrDict(
        slam=LSDMapper.default_params, 
        cache=AttrDict(map_dir='map', overwrite=True)
    )
    def __init__(self, params=default_params): 
        self.params = params

    def run(self, key, scene, remap_sparse=True): 
        print 'Processing scene %s' % key
        if self.params.mapper == 'orb': 
            self._run_scene_orb(key, scene, remap_sparse=remap_sparse)
        elif self.params.mapper == 'lsd': 
            self._run_scene_lsd(key, scene)
        else: 
            raise RuntimeError('Mapper not available: %s' % self.params.mapper)
            
    def _run_scene_lsd(self, key, scene):
        # Process images in scene
        slam = LSDMapper(params=self.params.slam)
        for f in scene.iteritems(every_k_frames=1): 
            slam.process(f.img)

        # Save slam map temporarily
        slam.save(os.path.join(self.params.cache.map_dir, '%s.h5' % key))

        # Finalize
        slam.optimize()
        slam.save(os.path.join(self.params.cache.map_dir, '%s.h5' % key))

    def _run_scene_orb(self, key, scene, remap_sparse=True):
        
        map_file = os.path.join(self.params.cache.map_dir, '%s.h5' % key) 

        # Remap the point cloud
        if remap_sparse: 

            # Process images in scene
            slam = ORBMapper(params=self.params.slam)

            # Doesn't support skipping frames (frame_id inconsistent)
            for f in scene.iteritems(every_k_frames=1): 
                slam.process(f.img)

            # Save slam map temporarily
            slam.save(map_file)
        
        # Finalize (semi-dense depth filter)
        mv_mapper = MultiViewMapper()
        mv_mapper.load(map_file)
        mv_mapper.process()
        
        mv_mapper.save(os.path.join(self.params.cache.map_dir, '%s_dense.h5' % key))
