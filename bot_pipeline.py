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

from bot_utils.test_utils import test_dataset
from bot_vision.imshow_utils import imshow_cv
from pybot_externals import StereoELAS
from bot_vision.color_utils import colormap

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
    pipeline = BotPipeline([dict(name='dataset', process=TestDatasetIterator(scale=1)), 
                            dict(name='tfilter', process=TestFilter()), 
                        ])
    pipeline.run()
