# Author: Sudeep Pillai <spillai@csail.mit.edu>
# License: MIT

import os
import numpy as np
from collections import OrderedDict

from pybot import user_data
from pybot.utils.db_utils import AttrDict
from pybot.geometry import RigidTransform, Pose, Sim3
from pybot.mapping import Keyframe, Mapper, MultiViewMapper 

class LSDMapper(Mapper): 
    """ LSD-SLAM interface """
    default_params = AttrDict(
        calib = np.array([528.49404721/640, 528.49404721/480, 319.5/640, 239.5/480, 0,0,0], 
                         dtype=np.float64),
        in_width=640, in_height=480,
    )
    def __init__(self, params=default_params): 
        Mapper.__init__(self, poses=[], keyframes=OrderedDict())
        
        from pybot_externals import LSDSLAM
        self.slam = LSDSLAM(**params)
        
        self.poses_ = []
        self.keyframes_ = OrderedDict()

    def update_keyframes(self): 
        # Keyframe graph for updates
        kf_data = self.slam.getKeyframeGraph()

        # Only plot until penultimate
        for kf in kf_data: 
            k_id = kf.getId()
            pts, colors, pose = kf.getPoints(), kf.getColors(), kf.getPose()
            pose_id = Sim3.from_matrix(pose)
            pose_id.id = k_id # HACK

            assert(len(pts) == len(colors))
            self.keyframes_[k_id] = Keyframe(id=k_id, frame_id=k_id,
                                             pose=pose_id,
                                             points=kf.getPoints(),
                                             colors=kf.getColors())
                

    def process(self, im): 
        # Process input frame
        self.slam.process(im)

        # Retain intermediate poses
        T = self.slam.getCurrentPoseEstimate()
        self.add_pose(Sim3.from_matrix(T))

        # self.poses.append(RigidTransform.from_matrix(T))

        # # Show debug output
        # if len(self.keyframes): 
        #     vis = self.slam.getDebugOutput()
        #     draw_utils.publish_image_t('lsd_debug', vis)
        #     print 'vis', vis.shape
        #     # imshow_cv('vis', vis)

        return T

    def optimize(self): 
        # Final optimization on keyframes/map
        self.slam.optimize()

class ORBMapper(Mapper): 
    """ ORB-SLAM interface """
    default_params = AttrDict(
        settings='Settings_udub.yaml', vocab='ORBvoc.txt'
    )
    def __init__(self, params=default_params): 
        Mapper.__init__(self, poses=[], keyframes=OrderedDict())

        from pybot_externals import ORBSLAM2

        # Only monocular supported for now
        mode = ORBSLAM2.eSensor.MONOCULAR
        self.slam = ORBSLAM2(settings=params.settings, vocab=params.vocab, mode=mode)
        self.slam.process = self.slam.process_monocular        
        print('ORBSLAM: Settings {}'.format(params.pprint()))

    def update_keyframes(self): 
        # Keyframe graph for updates
        kf_data = self.slam.getKeyframeGraph()

        # Only plot until penultimate
        for kfj in kf_data: 
            k_id = kfj.getId()
            f_id = kfj.getFrameId()
            pose_wc = RigidTransform.from_matrix(kfj.getPose())
            pose_cw = pose_wc.inverse()
            cloud_w = kfj.getPoints()
            cloud_c = pose_cw * cloud_w
            colors = (np.tile([0,0,1.0], [len(cloud_w),1])).astype(np.float32)
            
            # For now, add Pose with ID to pose, not necessary later on
            kf = Keyframe(id=k_id, frame_id=f_id,
                          pose=Pose.from_rigid_transform(k_id, pose_wc), 
                          points=cloud_c, colors=colors,
                          img=kfj.getImage())

            # kf = Keyframe.from_KeyframeData(kfj)
                        
            self.keyframes_[kf.id] = kf
            self.keyframes_dirty_[kf.id] = True
            # self.mosaics_[kf.id] = kf.visualize(self.calib_)
            
        print 'Keyframes: ', len(self.keyframes_), 'Poses: ', len(self.poses)

    def initialize_baseline(self, Tcw): 
        self.slam.initialize_baseline(Tcw)

    def process(self, im, right=None): 
        # Process input frame
        if right is None: 
            self.slam.process(im)
        else: 
            self.slam.process_stereo(im, right)
        
        # Retain intermediate poses
        Tcw = self.slam.getCurrentPoseEstimate()
        try:
            pose_cw = RigidTransform.from_matrix(Tcw)
            pose_wc = pose_cw.inverse()
        except Exception, e:
            print('Exception, failed to initialize {}, returning'.format(e))            
            return 
        
        # Add pose to map
        self.add_pose(pose_wc)
        return Tcw


class SemiDenseMapper(object): 
    default_params = AttrDict(
        slam_method='orb',
        slam_params=ORBMapper.default_params, 
        cache=AttrDict(map_dir='map', overwrite=True)
    )
    def __init__(self, params=default_params): 
        self.params = params

    def run(self, key, scene, remap_sparse=True): 
        print 'Processing scene %s' % key
        if self.params.slam_method == 'orb': 
            self._run_scene_orb(key, scene, remap_sparse=remap_sparse)
        elif self.params.slam_method == 'lsd': 
            self._run_scene_lsd(key, scene)
        else: 
            raise RuntimeError('slam_method not available: %s' % self.params.slam_method)
            
    def _run_scene_lsd(self, key, scene):
        # Process images in scene
        slam = LSDMapper(params=self.params.slam_params)
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
            slam = ORBMapper(self.params.slam_params)

            # Doesn't support skipping frames (frame_id inconsistent)
            for fidx, f in enumerate(scene.iteritems(every_k_frames=1)): 
                slam.process(f.img)
                
            # Save slam map temporarily
            slam.save(map_file)

        # Load calib from settings
        from pybot_externals import yaml_calib_settings
        calib = AttrDict(yaml_calib_settings(self.params.slam_params.settings))
        calib.shape = (480,640)
            
        # Finalize (semi-dense depth filter)
        mv_mapper = MultiViewMapper(calib.K, calib.shape[1], calib.shape[0], gridSize=4, nPyrLevels=1, max_n_kfs=10,
                                    use_photometric_disparity_error=True, 
                                    kf_displacement=0.4, kf_theta=np.deg2rad(20), detector='edge')
        mv_mapper.load(map_file)
        import IPython; IPython.embed()

        
        mv_mapper.run()
        
        mv_mapper.save(os.path.join(self.params.cache.map_dir, '%s_dense.h5' % key))
