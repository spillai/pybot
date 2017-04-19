# Author: Sudeep Pillai <spillai@csail.mit.edu>
# License: MIT

import os
import numpy as np
from itertools import izip

from pybot.utils.db_utils import AttrDict
from pybot.utils.dataset_readers import natural_sort, \
    FileReader, NoneReader, DatasetReader, ImageDatasetReader, \
    StereoDatasetReader, VelodyneDatasetReader

from pybot.geometry.rigid_transform import RigidTransform, Quaternion
from pybot.vision.camera_utils import StereoCamera

def body2camera(height=1):
    return RigidTransform.from_rpyxyz(-np.pi/2, 0, -np.pi/2, 
                                      0, 0, height, axes='sxyz')

def kitti_stereo_calib(sequence, scale=1.0): 
    seq = int(sequence)
    print('KITTI Dataset Reader: Sequence ({:}) @ Scale ({:})'.format(sequence, scale))
    if seq >= 0 and seq <= 2: 
        return KITTIDatasetReader.kitti_00_02.scaled(scale)
    elif seq == 3: 
        return KITTIDatasetReader.kitti_03.scaled(scale)
    elif seq >= 4 and seq <= 12: 
        return KITTIDatasetReader.kitti_04_12.scaled(scale)
    else:
        return None

def kitti_load_poses(fn): 
    X = (np.fromfile(fn, dtype=np.float64, sep=' ')).reshape(-1,12)
    return map(lambda p: RigidTransform.from_Rt(p[:3,:3], p[:3,3]), 
                map(lambda x: x.reshape(3,4), X))

def kitti_poses_to_str(poses): 
    return "\r\n".join(map(lambda x: " ".join(map(str, 
                                                  (x.matrix[:3,:4]).flatten())), poses))

def kitti_poses_to_mat(poses): 
    return np.vstack(map(lambda x: (x.matrix[:3,:4]).flatten(), poses)).astype(np.float64)
    
class OXTSReader(DatasetReader):
    def __init__(self, dataformat, template='oxts/data/%010i.txt', start_idx=0, max_files=100000): 
        super(OXTSReader, self).__init__(template=template, process_cb=self.oxts_process_cb,
                                         start_idx=start_idx, max_files=max_files)
        self.oxts_format_fn_ = dataformat
        self.oxts_formats_ = [line.split(':')[0] for line in open(self.oxts_format_fn_)]

        self.scale_ = None
        self.p_init_ = None
        
    @property
    def oxts_formats(self):
        return self.oxts_formats_
        
    def oxts_process_cb(self, fn):
        X = np.fromfile(fn, dtype=np.float64, sep=' ')
        packet = AttrDict({fmt: x for (fmt, x) in zip(self.oxts_formats, X)})
        er = 6378137.  # earth radius (approx.) in meters

        # compute scale from first lat value
        if self.scale_ is None: 
            self.scale_ = np.cos(packet.lat * np.pi / 180.)
        
        # Use a Mercator projection to get the translation vector
        tx = self.scale_ * packet.lon * np.pi * er / 180.
        ty = self.scale_ * er * \
             np.log(np.tan((90. + packet.lat) * np.pi / 360.))
        tz = packet.alt
        t = np.array([tx, ty, tz])
        
        # We want the initial position to be the origin, but keep the ENU
        # coordinate system
        rx, ry, rz = packet.roll, packet.pitch, packet.yaw
        Rx = np.float32([1,0,0, 0, np.cos(rx), -np.sin(rx), 0, np.sin(rx), np.cos(rx)]).reshape(3,3)
        Ry = np.float32([np.cos(ry),0,np.sin(ry), 0, 1, 0, -np.sin(ry), 0, np.cos(ry)]).reshape(3,3)
        Rz = np.float32([np.cos(rz), -np.sin(rz), 0, np.sin(rz), np.cos(rz), 0, 0, 0, 1]).reshape(3,3)
        R = np.dot(Rz, Ry.dot(Rx))
        pose = RigidTransform.from_Rt(R, t)

        if self.p_init_ is None:
            self.p_init_ = pose.inverse()
            
        # Use the Euler angles to get the rotation matrix
        return AttrDict(packet=packet, pose=self.p_init_ * pose)

class KITTIDatasetReader(object): 
    """
    KITTIDatasetReader: ImageDatasetReader + VelodyneDatasetReader + Calib
    http://www.cvlibs.net/datasets/kitti/setup.php
    """
    kitti_00_02 = StereoCamera.from_calib_params(718.86, 718.86, 607.19, 185.22, 
                                                 baseline_px=386.1448, shape=np.int32([376, 1241]))
    kitti_03 = StereoCamera.from_calib_params(721.5377, 721.5377, 609.5593, 172.854, 
                                                 baseline_px=387.5744, shape=np.int32([376, 1241]))
    kitti_04_12 = StereoCamera.from_calib_params(707.0912, 707.0912, 601.8873, 183.1104, 
                                                    baseline_px=379.8145, shape=np.int32([370, 1226]))

    wheel_baseline = 1.6
    baseline = 0.5371 # baseline_px / fx
    velo2cam = 0.27 # Velodyne is 27 cm behind cam_0 (x-forward, y-left, z-up)
    velo_height = 1.73
    cam_height = 1.65
    
    p_bc = RigidTransform.from_rpyxyz(-np.pi/2, 0, -np.pi/2, 0, 0, 0, axes='sxyz').inverse()
    p_bv = RigidTransform.from_rpyxyz(0, 0, 0, -0.27, 0, 0, axes='sxyz')
    
    velodyne2body = p_bv
    camera2body = p_bc
    
    def __init__(self, directory='', 
                 sequence='', 
                 left_template='image_0/%06i.png', 
                 right_template='image_1/%06i.png', 
                 velodyne_template='velodyne/%06i.bin',
                 start_idx=0, max_files=100000, scale=1.0): 

        # Set args
        self.sequence = sequence
        self.scale = scale

        # Get calib
        try:
            self.calib = kitti_stereo_calib(sequence, scale=scale)
        except Exception as e:
            self.calib = None
        
        # Read stereo images
        seq_directory = os.path.join(os.path.expanduser(directory), 'sequences', sequence)
        self.stereo = StereoDatasetReader(directory=seq_directory, 
                                          left_template=os.path.join(seq_directory,left_template), 
                                          right_template=os.path.join(seq_directory,right_template), 
                                          start_idx=start_idx, max_files=max_files, scale=scale)
        
        # Read poses
        try: 
            pose_fn = os.path.join(os.path.expanduser(directory), 'poses', ''.join([sequence, '.txt']))
            self.poses_ = FileReader(pose_fn, process_cb=kitti_load_poses)
        except Exception as e:
            self.poses_ = NoneReader()


        # Read velodyne
        try:
            self.velodyne_ = VelodyneDatasetReader(
                template=os.path.join(seq_directory,velodyne_template), 
                start_idx=start_idx, max_files=max_files
            )
        except Exception as e: 
            self.velodyne_ = NoneReader()

        print 'Initialized stereo dataset reader with %f scale' % scale

    @property
    def velodyne(self):
        return self.velodyne_

    @property
    def poses(self):
        return self.poses_.items
        
    def iteritems(self, *args, **kwargs): 
        return self.stereo.left.iteritems(*args, **kwargs)

    def iter_stereo_frames(self, *args, **kwargs): 
        return self.stereo.iteritems(*args, **kwargs)

    def iter_velodyne_frames(self, *args, **kwargs):         
        """
        for pc in dataset.iter_velodyne_frames(): 
          X = pc[:,:3]
        """
        return self.velodyne.iteritems(*args, **kwargs)

    def iter_stereo_velodyne_frames(self, *args, **kwargs):         
        return izip(self.left.iteritems(*args, **kwargs), 
                    self.right.iteritems(*args, **kwargs), 
                    self.velodyne.iteritems(*args, **kwargs))

    def iterframes(self, *args, **kwargs): 
        for (left, right), pose, velodyne in izip(self.iter_stereo_frames(*args, **kwargs),
                                                  self.poses_.iteritems(*args, **kwargs),
                                                  self.velodyne.iteritems(*args, **kwargs)): 
            yield AttrDict(left=left, right=right, velodyne=velodyne, pose=pose)

    def iter_gt_frames(self, *args, **kwargs): 
        for (left, right), pose in izip(self.iter_stereo_frames(*args, **kwargs), self.poses_.iteritems(*args, **kwargs)): 
            yield AttrDict(left=left, right=right, velodyne=None, pose=pose)
            
    @property
    def stereo_frames(self): 
        return self.iter_stereo_frames()

    @property
    def velodyne_frames(self): 
        return self.iter_velodyne_frames()

    
    # @classmethod
    # def stereo_test_dataset(cls, directory, subdir, scale=1.0):
    #     """
    #     Ground truth dataset iterator
    #     """

    #     left_directory = os.path.join(os.path.expanduser(directory), '%s_0' % subdir)
    #     right_directory = os.path.join(os.path.expanduser(directory), '%s_1' % subdir)
    #     noc_directory = os.path.join(os.path.expanduser(directory), 'disp_noc')
    #     occ_directory = os.path.join(os.path.expanduser(directory), 'disp_occ')

    #     c = cls(sequence='00')
    #     c.scale = scale
    #     c.calib = kitti_stereo_calib(1, scale=scale)

    #     # Stereo is only evaluated on the _10.png images
    #     c.stereo = StereoDatasetReader.from_directory(left_directory, right_directory, pattern='*_10.png')
    #     c.noc = ImageDatasetReader.from_directory(noc_directory)
    #     c.occ = ImageDatasetReader.from_directory(occ_directory)
    #     c.poses = [None] * c.stereo.length

    #     return c

    @classmethod
    def iterscenes(cls, sequences, directory='', 
                   left_template='image_0/%06i.png', right_template='image_1/%06i.png', 
                   velodyne_template='velodyne/%06i.bin', start_idx=0, max_files=100000, 
                   scale=1.0, verbose=False): 
        
        for seq in progressbar(sequences, size=len(sequences), verbose=verbose): 
            yield seq, cls(
                directory=directory, sequence=seq, left_template=left_template, 
                right_template=right_template, velodyne_template=velodyne_template, 
                start_idx=start_idx, max_files=max_files)
            
class KITTIStereoGroundTruthDatasetReader(object): 
    def __init__(self, directory, is_2015=False, scale=1.0):
        """
        Ground truth dataset iterator
        """
        if is_2015: 
            left_dir, right_dir = 'image_2', 'image_3'
            noc_dir, occ_dir = 'disp_noc_0', 'disp_occ_0'
            calib_left, calib_right = 'P2', 'P3'
        else: 
            left_dir, right_dir = 'image_0', 'image_1'
            noc_dir, occ_dir = 'disp_noc', 'disp_occ'
            calib_left, calib_right = 'P0', 'P1'

        self.scale = scale

        # Stereo is only evaluated on the _10.png images
        self.stereo = StereoDatasetReader(os.path.expanduser(directory), 
                                          left_template=''.join([left_dir, '/%06i_10.png']), 
                                          right_template=''.join([right_dir, '/%06i_10.png']), scale=scale, grayscale=True)
        self.noc = ImageDatasetReader(template=os.path.join(os.path.expanduser(directory), noc_dir, '%06i_10.png'))
        self.occ = ImageDatasetReader(template=os.path.join(os.path.expanduser(directory), occ_dir, '%06i_10.png'))

        def calib_read(fn, scale): 
            db = AttrDict.load_yaml(fn)
            P0 = np.float32(db[calib_left].split(' '))
            P1 = np.float32(db[calib_right].split(' '))
            fx, cx, cy = P0[0], P0[2], P0[6]
            baseline_px = np.fabs(P1[3])
            return StereoCamera.from_calib_params(fx, fx, cx, cy, baseline_px=baseline_px)

        self.calib = DatasetReader(template=os.path.join(os.path.expanduser(directory), 'calib/%06i.txt'), 
                                   process_cb=lambda fn: calib_read(fn, scale))

        self.poses_ = NoneReader()

    def iter_gt_frames(self, *args, **kwargs):
        """
        Iterate over all the ground-truth data
           - For noc, occ disparity conversion, see devkit_stereo_flow/matlab/disp_read.m
        """
        for (left, right), noc, occ, calib in izip(self.iter_stereo_frames(*args, **kwargs), 
                                                         self.noc.iteritems(*args, **kwargs), 
                                                         self.occ.iteritems(*args, **kwargs), 
                                                         self.calib.iteritems(*args, **kwargs)):
            yield AttrDict(left=left, right=right, 
                           depth=(occ/256).astype(np.float32),
                           noc=(noc/256).astype(np.float32), 
                           occ=(occ/256).astype(np.float32), 
                           calib=calib, pose=None)
                
    def iteritems(self, *args, **kwargs): 
        return self.stereo.left.iteritems(*args, **kwargs)

    def iter_stereo_frames(self, *args, **kwargs): 
        return self.stereo.iteritems(*args, **kwargs)

    def iterframes(self, *args, **kwargs): 
        for (left, right), pose in izip(self.iter_stereo_frames(*args, **kwargs), self.poses.iteritems(*args, **kwargs)): 
            yield AttrDict(left=left, right=right, velodyne=None, pose=pose)

    @property
    def stereo_frames(self): 
        return self.iter_stereo_frames()

    @property
    def poses(self):
        return self.poses_

class KITTIRawDatasetReader(object): 
    """
    KITTIRawDatasetReader: KITTIDatasetReader + OXTS reader
    """
    def __init__(self, directory, 
                 left_template='image_00/data/%010i.png', 
                 right_template='image_01/data/%010i.png', 
                 velodyne_template='velodyne_points/data/%010i.bin', 
                 oxts_template='oxts/data/%010i.txt',
                 start_idx=0, max_files=100000, scale=1.0): 

        self.scale = scale
        
        # Read stereo images
        try: 
            self.stereo_ = StereoDatasetReader(directory=directory, 
                                               left_template=left_template, 
                                               right_template=right_template, 
                                               start_idx=start_idx, max_files=max_files, scale=scale)
        except Exception, e:
            print('Failed to read stereo data: {}'.format(e))
            self.stereo_ = NoneReader()
        
        # Read velodyne
        try: 
            self.velodyne_ = VelodyneDatasetReader(
                template=os.path.join(directory,velodyne_template), 
                start_idx=start_idx, max_files=max_files
            )
        except Exception, e:
            print('Failed to read velodyne data: {}'.format(e))
            self.velodyne_ = NoneReader()
            
        # Read oxts
        try: 
            oxts_format_fn = os.path.join(os.path.expanduser(directory), 'oxts/dataformat.txt')
            oxts_fn = os.path.join(os.path.expanduser(directory), oxts_template)
            self.oxts_ = OXTSReader(oxts_format_fn, template=oxts_fn, start_idx=start_idx, max_files=max_files)
        except Exception as e:
            self.oxts_ = NoneReader()

    def iter_oxts_frames(self, *args, **kwargs): 
        return self.oxts_.iteritems(*args, **kwargs)
            
    def iterframes(self, *args, **kwargs): 
        for (left, right), oxts, velodyne in izip(self.iter_stereo_frames(*args, **kwargs), 
                                        self.iter_oxts_frames(*args, **kwargs),
                                        self.iter_velodyne_frames(*args, **kwargs)): 
            yield AttrDict(left=left, right=right, velodyne=velodyne, pose=oxts.pose, oxts=oxts.packet)
    
    def iter_stereo_frames(self, *args, **kwargs): 
        return self.stereo_.iteritems(*args, **kwargs)

    def iter_velodyne_frames(self, *args, **kwargs):         
        return self.velodyne_.iteritems(*args, **kwargs)

    def iter_stereo_velodyne_frames(self, *args, **kwargs):         
        return izip(self.left.iteritems(*args, **kwargs), 
                    self.right.iteritems(*args, **kwargs), 
                    self.velodyne.iteritems(*args, **kwargs))

    @property
    def stereo_frames(self): 
        return self.iter_stereo_frames()

    @property
    def oxts_fieldnames(self): 
        return self.oxts_.oxts_formats

    @property
    def oxts_data(self):
        return map(lambda oxts: oxts.packet, self.oxts_.iteritems())

    @property
    def poses(self):
        return map(lambda oxts: oxts.pose, self.oxts_.iteritems())

class OmnicamDatasetReader(object): 
    """
    OmnicamDatasetReader: ImageDatasetReader + VelodyneDatasetReader + Calib

    shape: H x W
    """
    shape = (1400,1400)
    def __init__(self, directory='', 
                 left_template='image_02/data/%010i.png', 
                 right_template='image_03/data/%010i.png', 
                 velodyne_template='velodyne_points/data/%010i.bin',
                 oxts_template='oxts/data/%010i.txt',
                 start_idx=0, max_files=100000, scale=1.0): 

        # Set args
        self.scale = scale

        # Read stereo images
        try: 
            seq_directory = os.path.expanduser(directory)
            self.stereo_ = StereoDatasetReader(directory=seq_directory,
                                              left_template=os.path.join(seq_directory,left_template), 
                                              right_template=os.path.join(seq_directory,right_template), 
                                              start_idx=start_idx, max_files=max_files, scale=scale)
        except:
            print('Failed to read stereo data: {}'.format(e))
            self.stereo_ = NoneReader()
            
        # Read velodyne
        try: 
            self.velodyne_ = VelodyneDatasetReader(
                template=os.path.join(seq_directory,velodyne_template), 
                start_idx=start_idx, max_files=max_files
            )
        except Exception, e:
            print('Failed to read velodyne data: {}'.format(e))
            self.velodyne_ = NoneReader()

        # Read oxts
        try: 
            oxts_format_fn = os.path.join(seq_directory, 'oxts/dataformat.txt')
            oxts_fn = os.path.join(seq_directory, oxts_template)
            self.oxts_ = OXTSReader(oxts_format_fn, template=oxts_fn, start_idx=start_idx, max_files=max_files)
        except Exception as e:
            self.oxts_ = NoneReader()

    @property
    def calib(self):
        raise NotImplementedError('Catadioptric camera calibration not yet implemented')
        
    @property
    def velodyne(self):
        return self.velodyne_

    @property
    def oxts(self):
        return self.oxts_

    @property
    def stereo(self):
        return self.stereo_

    def iterframes(self, *args, **kwargs): 
        for (left, right), oxts in izip(self.iter_stereo_frames(*args, **kwargs), 
                                        self.iter_oxts_frames(*args, **kwargs)): 
            yield AttrDict(left=left, right=right, velodyne=None, pose=oxts.pose, oxts=oxts.packet)
    
    def iter_stereo_frames(self, *args, **kwargs): 
        return self.stereo.iteritems(*args, **kwargs)

    def iter_velodyne_frames(self, *args, **kwargs):
        return self.velodyne.iteritems(*args, **kwargs)

    def iter_stereo_velodyne_frames(self, *args, **kwargs):         
        return izip(self.left.iteritems(*args, **kwargs), 
                    self.right.iteritems(*args, **kwargs), 
                    self.velodyne.iteritems(*args, **kwargs))

    def iter_oxts_frames(self, *args, **kwargs): 
        return self.oxts_.iteritems(*args, **kwargs)

    @property
    def stereo_frames(self): 
        return self.iter_stereo_frames()

    @property
    def velodyne_frames(self): 
        return self.iter_velodyne_frames()

    @property
    def oxts_fieldnames(self): 
        return self.oxts_.oxts_formats

    @property
    def oxts_data(self):
        return map(lambda oxts: oxts.packet, self.oxts_.iteritems())

    @property
    def poses(self):
        return map(lambda oxts: oxts.pose, self.oxts_.iteritems())

# def test_omnicam(dire):
#     return OmnicamDatasetReader(directory='/media/spillai/MRG-HD1/data/omnidirectional/')


