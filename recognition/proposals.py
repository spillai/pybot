
import cv2
import numpy as np

from bot_utils.plot_utils import colormap
from bot_vision.image_utils import im_resize

import os.path
import warnings
with warnings.catch_warnings():
    warnings.simplefilter("ignore")
    import gop
    from bot_vision.recognition.gop_util import setupLearned as gop_setuplearned

class ObjectProposal(object): 
    """
    Usage: 

        prop = ObjectProposal.create('GOP', params)
        bboxes = prop.process(im)
    """
    def __init__(self, proposer, scale=1): 
        self.proposer_ = proposer
        self.scale_ = scale
        
        if not hasattr(self.proposer_, 'process'): 
            raise NotImplementedError('Proposer does not have process implemented')

    def process(self, im): 
        boxes = self.proposer_.process(im_resize(im, scale=self.scale_))
        return (boxes * 1 / self.scale_).astype(np.int32)

    @staticmethod
    def visualize(vis, bboxes, ellipse=False, colored=True): 
        if not colored: 
            cols = np.tile([240,240,240], [len(bboxes), 1])
        else: 
            N = 20
            cwheel = colormap(np.linspace(0, 1, N))
            cols = np.vstack([cwheel[idx % N] for idx, _ in enumerate(bboxes)])            

        for col, b in zip(cols, bboxes): 
            if ellipse: 
                cv2.ellipse(vis, ((b[0]+b[2])/2, (b[1]+b[3])/2), ((b[2]-b[0])/2, (b[3]-b[1])/2), 0, 0, 360, tuple(col), 1)
            else: 
                cv2.rectangle(vis, (b[0], b[1]), (b[2], b[3]), tuple(col), 2)
        return vis

    @classmethod
    def create(cls, method='GOP', scale=1, num_proposals=1000, params=None): 
        if method == 'GOP': 
            params = dict(detector='sf', num_proposals=num_proposals) \
                     if params is None else params
            return cls(GOPObjectProposal(**params), scale=scale)
        elif method == 'BING': 
            return cls(BINGObjectProposal(**params), scale=scale)
        else: 
            raise RuntimeError('Unknown proposals method: %s' % method)

class GOPObjectProposal(object): 
    def __init__(self, detector='sobel', num_proposals=300): 
        gop_data_dir = """/home/spillai/perceptual-learning/software/externals/recognition-pod/proposals/gop/data/"""
        self.prop_ = gop.proposals.Proposal( gop_setuplearned( 140, 4, 0.8, gop_data_dir=gop_data_dir ) )

        if detector == 'sf': 
            self.segmenter_ = gop.contour.MultiScaleStructuredForest()
            self.segmenter_.load( os.path.join(gop_data_dir, 'sf.dat'))
        elif detector == 'sobel': 
            self.segmenter_ = gop.contour.DirectedSobel()
        else: 
            raise RuntimeError('Unknown detector %s' % detector)

        self.num_proposals_ = num_proposals

    def process(self, im): 
        # To get the 0th proposal segment: b[0, s.s]
        # segmented = np.zeros(shape=s.s.shape[:2], dtype=np.uint8)

        im_ = gop.imgproc.npread(np.copy(im))
        s = gop.segmentation.geodesicKMeans( im_, self.segmenter_, self.num_proposals_ )
        b = self.prop_.propose( s )
        return s.maskToBox( b )

        

def gop_propose(im, detector='sf', num_proposals=1000, scale=1):
    """
    Simpler-function for gop proposals.
    
        Don't use this function if you expect to run this 
        at high frame rates
    
    """
    p = ObjectProposal.create(params=dict(detector=detector, num_proposals=num_proposals), scale=scale)
    return p.process(im)

# class BINGObjectProposal(object): 
#     def __init__(self, voc_dir, num_proposals=10): 
#         self.proposer_ = BINGObjectness()
#         self.num_proposals_ = num_proposals
#         try: 
#             self.proposer_.build(voc_dir)
#             print voc_dir
#         except: 
#             raise RuntimeError('Error with selective search voc dir %s' % voc_dir)

#     def process(self, im): 
#         return self.proposer_.process(im, self.num_proposals_)
            


