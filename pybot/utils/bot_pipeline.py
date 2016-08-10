""" 
A stripped-down implementation of hartmann pipelines. This module is very
similar to sklearn.pipeline in its implementation.

Author: Sudeep Pillai <spillai@csail.mit.edu>
Licence: BSD
"""

import numpy as np

# __all__ = ['Pipeline', 'FeatureUnion']

def no_op(*args, **kwargs): 
    pass

class BotFilter(object): 
    def __init__(self, run=no_op, setup=no_op, pre_run=no_op, post_run=no_op, finalize=no_op): 
        # setattr(self, 'run', run)
        setattr(self, 'setup', setup)
        setattr(self, 'pre_run', pre_run)
        setattr(self, 'post_run', post_run)
        setattr(self, 'finalize', finalize)

class BotPipeline(object): 
    def __init__(self, steps):
        names = map(lambda step: step['name'], steps)
        if len(set(names)) != len(names):
            raise ValueError("Names provided are not unique: %s" % (names,))
            
        # Shallow copy
        self.steps_ = steps
        def raise_error_if_not_defined(t, name):
            if not (hasattr(t, name)): 
                raise TypeError("Filter %s.%s() has not been implemented" % (t, name))

        # Raise error of attributes unavailable, and 
        # setup individual components
        for step in self.steps_:
            step_process = step['process']
            
            # raise_error_if_not_defined(step_process, 'run')
            # raise_error_if_not_defined(step_process, 'setup')
            # raise_error_if_not_defined(step_process, 'pre_run')
            # raise_error_if_not_defined(step_process, 'post_run')
            # raise_error_if_not_defined(step_process, 'finalize')

            # step_filter.setup(step['params'])
            
    def run(self): 
        
        pout = None
        
        last_pass = False
        
        for item in self.steps_[0]['process'].run(): 
            pout = item
            for step in self.steps_[1:]:
                pout = step['process'].run(*pout)                
                
    def finalize(self): 
        for step in self.steps_: 
            step.finalize()

from pybot.utils.test_utils import test_dataset
from pybot.vision.imshow_utils import imshow_cv
from pybot_externals import StereoELAS
from pybot.vision.color_utils import colormap

class BotSource(object): 
    def __init__(self): 
        pass

    # def run(self): 
    #     pass

    def set_source(self, source): 
        self.source_ = source

    def iteritems(self, *args, **kwargs): 
        return self.source_.iteritems(*args, **kwargs)


from pybot.utils.io_utils import VideoCapture
from pybot.externals.lcm.log_utils import KinectLCMLogReader, KinectDecoder
                        
class MonoSource(BotSource): 
    """
    Formats:

      'cam://0'
      'file:///home/spillai/data/test.avi'
      'lcm:///home/spillai/data/lcmlog-2015-07-07.00?=KINECT_DATA'
    
    """
    def __init__(self, source_file, source_params):
        BotSource.__init__(self)
        source_type, source_name = source_file.split('://')
        if source_type == 'cam': 
            source = VideoCapture(filename=int(source_name), size=(640,480), fps=60) # replace with source_params
        elif source_type == 'file': 
            source = VideoCapture(filename=source_name) # replace with source_params
        elif source_type == 'lcm': 
            splits = source_name.split('?=')
            filename, channel = splits if len(splits) == 2 else (splits[0], 'KINECT_DATA') 
            source = KinectLCMLogReader(filename=source_name, 
                                         channel=channel, extract_depth=True)
        else: 
            raise RuntimeError("Unknown Source type %s" % source_type)

        self.set_source(source)

class StereoSource(BotSource): 
    def __init__(self, source_file, source_params):
        BotSource.__init__(self)
        source_type, source_name = source_file.split('://')
        # if source_type == 'cam': 
        #     try: 
        #         idx0, idx1 = source_name.split(',')
        #     except ValueError as e:  
        #         print(e.args)
        #     source = MonoSource(
        #         VideoCapture(filename=int(source_name), size=(640,480), fps=60) # replace with source_params
        
        if source_type == 'firewire': 
            from pybot_vision import DC1394Device
            from pybot.utils.db_utils import AttrDict

            dataset = DC1394Device()
            dataset.init()
        elif source_type == 'dir':
            StereoImageDatasetReader(directory=directory)
        else: 
            raise RuntimeError("Unknown Source type %s" % source_type)

    
    # def run(self): 
    #     pass

# def app_factory(source_type='mono', source_file='cam://0'):
    
#     if source_type == 'mono':
#         source = MonoSource
#     elif source_type == 'stereo': 
#         source = StereoSource
#     elif source_type == 'rgbd': 
#         source = RGBDSource
#     else: 
#         raise RuntimeError('Unknown source_type: %s for factory production' % source_type)

#     class SourcedApp(source): 
#         def __init__(self): 
#             source.__init__(self, source_file)

#     return SourcedApp


class TestDatasetIterator(BotFilter): 
    def __init__(self, *args, **kwargs):
        # BotFilter.__init__(self)
        self.obj_ = test_dataset(*args, **kwargs)

    def run(self): 
        for l,r in self.obj_.iter_stereo_frames(): 
            imshow_cv('lr', np.vstack([l,r]))
            yield l,r

class TestFilter(BotFilter): 
    def __init__(self, *args, **kwargs):
        # BotFilter.__init__(self)
        # self.obj_ = test_dataset() # *args, **kwargs)
        self.obj_ = StereoELAS()
        setattr(self, 'post_run', TestFilter.visualize)

    def run(self, l, r):
        disp = self.obj_.process(l, r)
        # disp_color = colormap(disp / 64)
        # imshow_cv('stereo', disp_color)
        return disp
        
    # @staticmethod
    # def visualize(disp): 
    #     disp_color = colormap(disp / 64)
    #     imshow_cv('stereo', disp_color)
    #     return None

if __name__ == "__main__": 
    # pipeline = BotPipeline([dict(name='dataset', process=TestDatasetIterator(scale=1)), 
    #                         dict(name='tfilter', process=TestFilter()), 
    #                     ])
    # pipeline.run()

    # StereoSource()
    
