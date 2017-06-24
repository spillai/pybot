import os.path
import numpy as np
from itertools import izip, islice
from scipy.io import loadmat

from pybot.geometry.rigid_transform import RigidTransform
from pybot.utils.io_utils import VideoCapture
from pybot.utils.db_utils import AttrDict
from pybot.mapping.nav_utils import metric_from_gps, bearing_from_metric_gps

class StLuciaReader(object):
    def __init__(self, directory): 

        # Read dataset
        self.video_path_ = os.path.join(
            os.path.expanduser(directory), 'webcam_video.avi')

        # Metric coords from GPS
        gps = loadmat(os.path.join(
            os.path.expanduser(directory), 'fGPS.mat'))['fGPS']
        mgps = metric_from_gps(gps)
        assert(np.isfinite(mgps).all())
        mgps -= mgps.mean(axis=0)

        # Determine bearing from Sequential GPS
        theta = bearing_from_metric_gps(mgps)
        assert(np.isfinite(theta).all())
        self.poses_ = [ RigidTransform.from_rpyxyz(0,0,th,gps[1],gps[0],0)
                        for gps,th in zip(mgps, theta) ]
        
    def iterframes(self):
        cap = VideoCapture(self.video_path_)
        for (img,pose) in izip(cap.iteritems(), self.poses_):
            yield AttrDict(img=img, pose=pose)

    @property
    def poses(self):
        return self.poses_

    @property
    def length(self):
        return self.len(self.poses_)

        
# from pybot.externals.ros.bag_utils import ROSBagReader, ImageDecoder, NavMsgDecoder
# channel = '/stlucia/camera/image/compressed'
# dataset = ROSBagReader(
#     filename=os.path.expanduser(args.filename), 
#     decoder=[ImageDecoder(channel=channel, scale=args.scale, compressed=compressed)],
#     every_k_frames=1, start_idx=0, index=False)
# dataset_iterframes = imap(lambda (t, ch, im): (f.img, f.pose), dataset.iterframes())
        
