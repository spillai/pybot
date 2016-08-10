# Author: Sudeep Pillai <spillai@csail.mit.edu>
# License: MIT

import os
import numpy as np

from .mapper import Mapper, MultiViewMapper 


# class LSDMapper(Mapper): 
#     """ LSD-SLAM interface """
#     default_params = AttrDict(
#         calib = np.array([528.49404721/640, 528.49404721/480, 319.5/640, 239.5/480, 0,0,0], 
#                          dtype=np.float64),
#         in_width=640, in_height=480,
#     )
#     def __init__(self, params=default_params): 
#         Mapper.__init__(self, poses=[], keyframes=OrderedDict())
        
#         from pybot_externals import LSDSLAM
#         self.slam = LSDSLAM(**params)
        
#         self.poses_ = []
#         self.keyframes_ = OrderedDict()

#     def update_keyframes(self): 
#         # Keyframe graph for updates
#         kf_data = self.slam.getKeyframeGraph()

#         # Only plot until penultimate
#         for kf in kf_data: 
#             k_id = kf.getId()
#             pts, colors, pose = kf.getPoints(), kf.getColors(), kf.getPose()
#             pose_id = Sim3.from_homogenous_matrix(pose)
#             pose_id.id = k_id # HACK

#             assert(len(pts) == len(colors))
#             self.update_keyframe(
#                 KeyFrame(k_id, k_id, pose=pose_id, points=kf.getPoints(), colors=kf.getColors())
#             )

#     def process(self, im): 
#         # Process input frame
#         self.slam.process(im)

#         # Retain intermediate poses
#         T = self.slam.getCurrentPoseEstimate()
#         self.add_pose(Sim3.from_homogenous_matrix(T))

#         # self.poses.append(RigidTransform.from_homogenous_matrix(T))

#         # # Show debug output
#         # if len(self.keyframes): 
#         #     vis = self.slam.getDebugOutput()
#         #     draw_utils.publish_image_t('lsd_debug', vis)
#         #     print 'vis', vis.shape
#         #     # imshow_cv('vis', vis)

#         return T

#     def optimize(self): 
#         # Final optimization on keyframes/map
#         self.slam.optimize()

# class ORBMapper(Mapper): 
#     """ ORB-SLAM interface """
#     # default_params = AttrDict(
#     #     K = np.array([[528.49404721, 0, 319.5], 
#     #                   [0, 528.49404721, 239.5],
#     #                   [0, 0, 1]], dtype=np.float64), 
#     #     W = 640, H = 480
#     # )
#     def __init__(self, settings='Settings_udub.yaml'): 
#         Mapper.__init__(self, poses=[], keyframes=OrderedDict())
#         from pybot_externals import ORBSLAM

#         path = '/home/spillai/perceptual-learning/software/python/apps/config/orb_slam/'
#         self.slam = ORBSLAM(settings=path + settings, vocab=path + 'ORBvoc.yml')
#         print('ORBSLAM: Settings File %s' % (path + settings))

#     def update_keyframes(self): 
#         # Keyframe graph for updates
#         kf_data = self.slam.getKeyframeGraph()

#         # Only plot until penultimate
#         for kf in kf_data: 
#             k_id = kf.getId()
#             f_id = kf.getFrameId()
#             pose_cw = RigidTransform.from_homogenous_matrix(kf.getPose())
#             pose_wc = pose_cw.inverse()
#             cloud_w = kf.getPoints()
#             # cloud_c = pose_cw * cloud_w
#             colors = (np.tile([0,0,1.0], [len(cloud_w),1])).astype(np.float32)

#             # print k_id, len(cloud_w), len(cloud_c)

#             # For now, add Pose with ID to pose, not necessary later on
#             self.update_keyframe(
#                 KeyFrame(k_id, frame_id, pose=Pose.from_rigid_transform(k_id, pose_wc), 
#                      points=cloud_w, colors=colors, img=kf.getImage())
#             )
            
#         print 'Keyframes: ', len(self.keyframes), 'Poses: ', len(self.poses)

#     def initialize_baseline(self, Tcw): 
#         self.slam.initialize_baseline(Tcw)

#     def process(self, im, right=None): 
#         # Process input frame
#         if right is None: 
#             self.slam.process(im)
#         else: 
#             self.slam.process_stereo(im, right)
        
#         # Retain intermediate poses
#         Tcw = self.slam.getCurrentPoseEstimate()

#         try:
#             pose_cw = RigidTransform.from_homogenous_matrix(Tcw)
#             pose_wc = pose_cw.inverse()
#         except: 
#             return 

#         # Add pose to map
#         self.add_pose(pose_wc)

#         imshow_cv('orb-slam', im)

#         return Tcw


# class SemiDenseMapper(object): 
#     default_params = AttrDict(
#         slam=LSDMapper.default_params, 
#         cache=AttrDict(map_dir='map', overwrite=True)
#     )
#     def __init__(self, params=default_params): 
#         self.params = params

#     def run(self, key, scene, remap_sparse=True): 
#         print 'Processing scene %s' % key
#         if self.params.mapper == 'orb': 
#             self._run_scene_orb(key, scene, remap_sparse=remap_sparse)
#         elif self.params.mapper == 'lsd': 
#             self._run_scene_lsd(key, scene)
#         else: 
#             raise RuntimeError('Mapper not available: %s' % self.params.mapper)
            
#     def _run_scene_lsd(self, key, scene):
#         # Process images in scene
#         slam = LSDMapper(params=self.params.slam)
#         for f in scene.iteritems(every_k_frames=1): 
#             slam.process(f.img)

#         # Save slam map temporarily
#         slam.save(os.path.join(self.params.cache.map_dir, '%s.h5' % key))

#         # Finalize
#         slam.optimize()
#         slam.save(os.path.join(self.params.cache.map_dir, '%s.h5' % key))

#     def _run_scene_orb(self, key, scene, remap_sparse=True):
        
#         map_file = os.path.join(self.params.cache.map_dir, '%s.h5' % key) 

#         # Remap the point cloud
#         if remap_sparse: 

#             # Process images in scene
#             slam = ORBMapper(params=self.params.slam)

#             # Doesn't support skipping frames (frame_id inconsistent)
#             for f in scene.iteritems(every_k_frames=1): 
#                 slam.process(f.img)

#             # Save slam map temporarily
#             slam.save(map_file)
        
#         # Finalize (semi-dense depth filter)
#         mv_mapper = MultiViewMapper()
#         mv_mapper.load(map_file)
#         mv_mapper.run()
        
#         mv_mapper.save(os.path.join(self.params.cache.map_dir, '%s_dense.h5' % key))
