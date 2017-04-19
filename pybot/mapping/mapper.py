"""
Generic Keyframe-based Mapping / Data Layer
===========================================

Map the environment in a keyframe-based fashion. 

Persistent map storage. 
Update and publish in an incremental manner. 

"""

# Author: Sudeep Pillai <spillai@csail.mit.edu>
# License: MIT

import os
import numpy as np

from abc import ABCMeta, abstractmethod
from collections import OrderedDict

from pybot.geometry.rigid_transform import Pose, RigidTransform, Sim3
from pybot.utils.db_utils import AttrDict
from pybot.utils.misc import dequedict, CounterWithPeriodicCallback
from pybot.vision.color_utils import colormap
from pybot.vision.imshow_utils import imshow_cv
from pybot.vision.image_utils import to_color, im_mosaic_list
from pybot.vision.draw_utils import draw_features
from pybot.vision.camera_utils import Camera, \
    CameraIntrinsic, CameraExtrinsic
from pybot.externals.lcm import draw_utils

class Keyframe(object): 
    """
    Basic interface for keyframes

    Most variables are persistent except for {dirty_}.
    Only pose, points, colors should be allowed to
    be modified. 

        id, frame_id, pose, points, colors, img

    """
    def __init__(self, **kwargs):
        self.data_ = AttrDict(**kwargs)

    @property
    def id(self): 
        return self.data_.id

    @property
    def frame_id(self): 
        return self.data_.frame_id

    @property
    def pose(self): 
        return self.data_.pose

    @property
    def ids(self): 
        return self.data_.ids

    @property
    def points(self): 
        return self.data_.points

    @property
    def colors(self): 
        return self.data_.colors

    @property
    def img(self): 
        return self.data_.img

    @pose.setter
    def pose(self, pose): 
        self.data_.pose = pose

    @points.setter
    def points(self, points): 
        self.data_.points = points

    @colors.setter
    def colors(self, colors): 
        self.data_.colors = colors

    @classmethod
    def from_dict(self, d): 
        return cls(**d)

    def to_dict(self, d): 
        return self.data_

    @classmethod
    def from_KeyframeData(cls, kf, is_sim3=False):
        kf_id = kf.getId()
        pose = kf.getPose()
        kf_pose = Sim3.from_matrix(pose) if is_sim3 else \
                RigidTransform.from_matrix(pose)        
        kf_pose.id = kf_id

        points = kf.getPoints()
        try: 
            colors = kf.getColors()
        except: 
            colors = (np.tile([0,0,1.0], [len(points),1])).astype(np.float32)

        assert(len(points) == len(colors))
        return cls(id=kf_id, frame_id=kf.getFrameId(), 
                   pose=kf_pose, points=points, colors=colors, ids=kf.getPointIds(), 
                   img=kf.getImage(), points2d=kf.getPoints2d(), kf_ids=kf.getKFIds())

    def on_changed(self):
        """
        Define callbacks on new updates to keyframe 
        via (property setters)
        """
        raise NotImplementedError()

    def visualize(self, camera_intrinsic): 
        if self.img is None: 
            raise RuntimeError('Keyframe image is not available to visualize')

        vis = to_color(self.img)
        cam = Camera.from_intrinsics_extrinsics(camera_intrinsic, 
                                                CameraExtrinsic.identity())

        pts, depths = cam.project(self.points, check_bounds=True, return_depth=True)

        colors = np.int64(colormap(depths.astype(np.float32) / 32.0).reshape(-1,3))        
        return draw_features(vis, pts, colors=colors, size=4)

class Mapper(object): 
    """ 
    
    SLAM interface 
    ==============

    Database to store map-related data. 
    
    Stored values: 
       poses, keyframes, points (for each keyframe), colors (for each
       point)
    
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

        publish_rate: 
            Publish keyframe point clouds and poses every k frames

    Example: 

    mapper = Mapper()
    mapper.add_pose(p_wc)

    """
    def __init__(self, poses=[], keyframes=OrderedDict(), 
                 incremental=True, update_kf_rate=10, 
                 publish=True, publish_rate=10, name='SLAM'): 

        # Keyframe lookup  (map frame index to keyframe)
        self.keyframes_ = keyframes
        self.keyframes_dirty_ = { kf.id: True 
                                  for kf in self.keyframes_.itervalues() }
        self.keyframes_lut_ = { kf.frame_id: kf.id 
                                for kf in self.keyframes_.itervalues() }
        self.poses_ = poses
        self.reset_ = True
        self.name_ = name

        # Setup a counter that publishes the map every k frames
        self.publish_cb_ = CounterWithPeriodicCallback(
            every_k=publish_rate, process_cb=self.publish)

        # Setup a counter that updates the keyframe graph every k frames
        self.update_kf_cb_ = CounterWithPeriodicCallback(
            every_k=update_kf_rate, process_cb=self.update_keyframes)

        # Clean up keyframe points, colors
        for k_id in self.keyframes_.keys(): 
            points = self.keyframes_[k_id].points
            colors = self.keyframes_[k_id].colors

            if points is None or colors is None: 
                continue

            colors = np.vstack(colors)
            points = np.vstack(points)
            inds = np.isfinite(points).all(axis=1)
            self.keyframes_[k_id].points = points[inds]
            self.keyframes_[k_id].colors = colors[inds]

        print('''\nMapper =======>\n''' \
              '''KF updates (every={:})\n''' \
              '''Publish (every={:})\n''' \
              '''=============================\n''' \
                .format(update_kf_rate, publish_rate))
        
    @abstractmethod
    def update_keyframes(self):
        """ 
        Abstract method to update the map with keyframe data

        keyframes_ need to be
        updated based on new information, call
        update_kf on a per-keyframe basis        
        
        """
        raise NotImplementedError()
        
    def add_pose(self, pose): 
        """
        Add poses to the map
        Provide a unique id for the appended poses
        """

        # Add pose, and publish
        p = Pose.from_rigid_transform(len(self.poses_), pose)
        self.poses_.append(p)
        draw_utils.publish_pose_list(self.name_ + '_poses', [p], frame_id='camera', reset=False)
        
        # Counter increment for publishing, kf updates
        self.publish_cb_.poll()
        self.update_kf_cb_.poll()
        
    def fetch_keyframe(self, frame_id): 
        return self.keyframes_[self.keyframes_lut_[frame_id]]
        
    def publish(self, incremental=True): 
        """
        Publish map components: 
           poses: Intermediate poses, i.e. tracked poses (usually unoptimized)
           keyframes: Optimized keyframes (this will be updated every so often)
           points: Landmark features that are also optimized along with keyframes

        """
        if incremental:
            if len(self.poses_) and not hasattr(self.poses_[0], 'id'): 
                raise RuntimeError('''Mapper.poses is not of type RigidTransform '''
                                   ''' with an ID, incremental drawing will not work ''')

            # Get all dirty keyframes and their corresponding 
            # ids, pts, colors, and poses
            kf_ids = self.dirty_keyframe_ids
            kf_poses = self.dirty_keyframe_poses
            kf_pts = self.dirty_points
            kf_cols = self.dirty_colors
            # print 'kf_poses, kf_ids', len(kf_poses), len(kf_ids)

            # Draw all the dirty keyframes
            draw_utils.publish_cameras(self.name_ + '_keyframes_cams', kf_poses, draw_faces=False, 
                                       frame_id='camera', reset=False)
            draw_utils.publish_pose_list(self.name_ + '_keyframes', kf_poses, 
                                         frame_id='camera', reset=False)
            draw_utils.publish_cloud(self.name_ + '_keyframes_cloud', kf_pts, c=kf_cols, 
                                     frame_id=self.name_ + '_keyframes', element_id=kf_ids, reset=False)

            # Finish up and clean out all the keyframes (lut)
            for kf_id in self.keyframes_.iterkeys(): 
                self.keyframes_dirty_[kf_id] = False

        else: 
            # Draw all poses (should be unique frame ids)
            draw_utils.publish_pose_list(self.name_ + '_poses', self.poses, frame_id='camera')

            kf_ids = self.all_keyframe_ids
            kf_poses = self.all_keyframe_poses
            kf_pts = self.all_points
            kf_cols = self.all_colors


            draw_utils.publish_cameras(self.name_ + '_keyframes', kf_poses, draw_faces=False, 
                                       frame_id='camera', reset=self.reset)
            draw_utils.publish_cloud(self.name_ + '_keyframes_cloud', kf_pts, c=kf_cols, 
                                     frame_id='camera', reset=self.reset)
            # draw_utils.publish_cloud('slam_keyframes_cloud', kf_pts, c=kf_cols, element_id=kf_ids, 
            # frame_id='slam_keyframes')

        self.reset_ = False

    
    @property
    def poses(self): 
        return self.poses_

    @property
    def keyframes(self): 
        return self.keyframes_

    @property
    def all_points(self): 
        return [kf.points for kf in self.keyframes_.itervalues() if kf.points is not None]

    @property
    def all_colors(self): 
        return [kf.colors for kf in self.keyframes_.itervalues() if kf.colors is not None ]

    @property
    def all_keyframe_ids(self): 
        return [kf.id for kf in self.keyframes_.itervalues()]

    @property
    def all_keyframe_poses(self): 
        return [kf.pose for kf in self.keyframes_.itervalues()]

    @property
    def dirty_points(self): 
        return [kf.points for kf in self.keyframes_.itervalues() if kf.points is not None and self.keyframes_dirty_[kf.id]]

    @property
    def dirty_colors(self): 
        return [kf.colors for kf in self.keyframes_.itervalues() if kf.colors is not None and self.keyframes_dirty_[kf.id] ]

    @property
    def dirty_keyframe_ids(self): 
        return [kf.id for kf in self.keyframes_.itervalues() if self.keyframes_dirty_[kf.id]]

    @property
    def dirty_keyframe_poses(self): 
        return [kf.pose for kf in self.keyframes_.itervalues() if self.keyframes_dirty_[kf.id]]

    @property
    def publish_rate(self): 
        return self.publish_cb.every_k

    @property
    def update_kf_rate(self): 
        return self.update_kf_cb.every_k

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
        print 'Saving keyframe ids: ', self.keyframes_.keys()
        db = AttrDict(poses=self.poses_, 
                      keyframes=self.keyframes_.values())
        db.save(path)


    
class MultiViewMapper(Mapper): 
    """ 
    
    SVO-Depth-Filter based reconstruction
    
    Notes: 
      1. Only reconstruct semi-densely after first 5 keyframes
      2. Increased patch size in svo depth filter to 16 from 8

    Use as: 

    >> mv_mapper = MultiViewMapper(cam.K, cam.shape[1], cam.shape[0], gridSize=10, nPyrLevels=1, max_n_kfs=4,
                                   kf_displacement=0.25, kf_theta=np.deg2rad(15))

    # Reconstruct scene
    >> for fidx, f in enumerate(scene.iteritems(every_k_frames=1)):
           mv_mapper.process(f.img, f.pose)
               
    """
    def __init__(self, K, W, H, gridSize=30, nPyrLevels=1, max_n_kfs=4,
                 kf_theta=np.deg2rad(20), kf_displacement=0.25, detector='edge',
                 use_photometric_disparity_error=False, verbose=False): 
        Mapper.__init__(self)

        # SVO Depth Filter (Dependency)
        from pybot_externals import SVO_DepthFilter
        self.depth_filter = SVO_DepthFilter(np.float64(K), int(W), int(H), 
                                            gridSize=gridSize, nPyrLevels=nPyrLevels,
                                            max_n_kfs=max_n_kfs, detector=detector,
                                            use_photometric_disparity_error=use_photometric_disparity_error,
                                            verbose=verbose)
        self.cam_intrinsic_ = CameraIntrinsic(K, shape=(int(H), int(W)))

        self.mosaics_ = dequedict(maxlen=max_n_kfs)
        self.last_kf_inv = None
        self.kf_theta = kf_theta
        self.kf_displacement = kf_displacement

    def update_keyframes(self):
        """ Process/Update Keyframe data """

        # Add keyframe and set dirty (for publishing)
        kf_data = self.depth_filter.getKeyframeGraph()
        for kfj in kf_data:
            
            kf = Keyframe.from_KeyframeData(kfj)
            self.keyframes_[kf.id] = kf
            self.keyframes_dirty_[kf.id] = True
            self.mosaics_[kf.id] = kf.visualize(self.cam_intrinsic_)

        if len(self.mosaics_): 
            imshow_cv('kfs', im_mosaic_list(self.mosaics_.values(), scale=0.75, width=3))        

        # print 'Updated IDS: ', [kf.getId() for kf in kf_data]
        # print 'Dirty IDS: ', [(kf.id, kf.dirty) for kf in self.keyframes.itervalues()]

    def process(self, img, pose_wc): 
        """
        Incremental reconstruction of the map, given keyframe
        poses and the corresponding image
        """

        # Check if new frame has rotated at-least by theta or moved by displacement
        if self.last_kf_inv is not None:
            newp = self.last_kf_inv * pose_wc
            d, r = np.linalg.norm(newp.tvec), np.fabs(newp.to_rpyxyz()[:3])
            is_new_kf = not bool(d < self.kf_displacement and (r < self.kf_theta).all())
        else:
            is_new_kf = True

        # Set new pose inverse
        if is_new_kf:
            self.last_kf_inv = pose_wc.inverse()

        # Only process once for semi-dense depth estimation
        self.depth_filter.process(img, (pose_wc.matrix).astype(np.float64), is_new_kf)
        imshow_cv('kf_img', img)

        # Add pose (polls publish_cb and update_kf_cb) 
        self.add_pose(pose_wc)
        
    def run(self):
        """
        Batch reconstruction of the map, given already established
        KeyframeGraph (keyframes, poses, and images)
        
        This should be run only after a SLAM-solution has been
        optimized over the keyframe graph.
        """
    
        # Show sparse map and keyframes
        self.publish()

        # Multi-view semi-dense reconstruction
        for kf_id in self.keyframes.keys(): 
            self.keyframes[kf_id].points = None
            self.keyframes[kf_id].colors = None
            if kf_id < 20: 
                continue
            self.process(self.keyframes[kf_id].img, self.keyframes[kf_id].pose)
            draw_utils.publish_cameras('current_keyframe', [self.keyframes[kf_id].pose], frame_id='camera', size=2)

        # HACK: save to first keyframe
        self.keyframes[0].points = self.depth_filter.getPoints()
        self.keyframes[0].colors = (self.depth_filter.getColors() * 1.0 / 255).astype(np.float32)

    def load(self, path):
        db = AttrDict.load(path)
        keyframes = OrderedDict({kf.id:kf for kf in db.keyframes})
        Mapper.__init__(self, db.poses, keyframes)

    def save(self, path): 
        print 'Saving keyframe ids: ', self.keyframes.keys()
        db = AttrDict(poses=self.poses, 
                      keyframes=self.keyframes.values())
        db.save(path)
