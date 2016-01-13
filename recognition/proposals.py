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

    def process(self, im): 
        boxes = self.proposer_.process(im_resize(im, scale=self.scale_))
        return (boxes * 1 / self.scale_).astype(np.int32)

    @classmethod
    def create(cls, method='GOP', scale=1, params=None): 
        if method == 'GOP': 
            return cls(GOPObjectProposal(**params), scale=scale)
        elif method == 'BING': 
            return cls(BINGObjectProposal(**params), scale=scale)
        else: 
            raise RuntimeError('Unknown proposals method: %s' % method)

class GOPObjectProposal(object): 
    def __init__(self, detector='sf', num_proposals=1000): 
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
        im_ = gop.imgproc.npread(np.copy(im))
        st = time.time()
        s = gop.segmentation.geodesicKMeans( im_, self.segmenter_, self.num_proposals_ )
        b = self.prop_.propose( s )
        return s.maskToBox( b )

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
            


