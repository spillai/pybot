from pybot.utils.test_utils import test_video
from pybot.utils.db_utils import AttrDict
from pybot.vision.feature_detection import FeatureDetector
from pybot.vision.trackers.base_klt import OpenCVKLT, MeshKLT
    
if __name__ == "__main__":
    
    # Setup detector params
    detector_params = AttrDict(method='fast', grid=(16,10), max_corners=1200, 
                               max_levels=3, subpixel=False, params=FeatureDetector.fast_params)

    # Setup tracker params (either lk, or dense)
    lk_params = AttrDict(winSize=(5,5), maxLevel=3)
    tracker_params = AttrDict(method='lk', fb_check=True, params=lk_params)
    
    # farneback_params = AttrDict(pyr_scale=0.5, levels=3, winsize=15, 
    #                             iterations=3, poly_n=7, poly_sigma=1.5, flags=0)
    # tracker_params = AttrDict(method='dense', fb_check=True, params=farneback_params)


    # Create detector from params
    det = FeatureDetector(**detector_params)
    
    # Create KLT from detector params only
    klt = OpenCVKLT.from_params(detector_params=detector_params, 
                                tracker_params=tracker_params, 
                                min_tracks=1000)
    
    # Create mesh klt
    klt = MeshKLT.from_params(detector_params=detector_params, 
                              tracker_params=tracker_params, 
                              min_tracks=1000)

    for im in test_video(color=False):
        pts = det.process(im)
        ids, tracked = klt.process(im)
    
