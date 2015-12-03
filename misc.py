from collections import deque
import progressbar as pb

def setup_pbar(maxval): 
    widgets = ['Progress: ', pb.Percentage(), ' ', pb.Bar(), ' ', pb.ETA()]
    pbar = pb.ProgressBar(widgets=widgets, maxval=maxval)
    pbar.start()
    pbar.increment = lambda : pbar.update(pbar.currval + 1)
    return pbar

class Counter(object): 
    def __init__(self): 
        self.idx_ = 0

    def count(self): 
        self.idx_ += 1

    def reset(self): 
        self.idx_ = 0

    def check_divisibility(self, every_k): 
        return self.idx_ % every_k == 0 #  and self.idx_ > 0 

    @property
    def index(self): 
        return self.idx_

class Accumulator(Counter): 
    def __init__(self, maxlen=100): 
        Counter.__init__(self)
        self.items_ = deque(maxlen=maxlen)

    def accumulate(self, item): 
        self.items_.append(item)
        self.count()

    def accumulate_list(self, items): 
        for item in items: 
            self.accumulate(item)

    @property
    def latest(self): 
        return self.items_[-1]

    @property
    def first(self): 
        return self.items_[0]

    @property
    def items(self): 
        return self.items_
        
    @property
    def length(self): 
        return len(self.items_)


class PoseAccumulator(Accumulator): 
    def __init__(self, maxlen=100, relative=False): 
        Accumulator.__init__(self, maxlen=maxlen)

        self.relative_ = relative
        self.init_ = None

    def accumulate(self, pose):
        if self.relative_: 
            if self.init_ is None: 
                self.init_ = pose
            p = self.relative_to_init(pose)
        else: 
            p = pose

        # Call accumulate on base class
        super(PoseAccumulator, self).accumulate(p)
        
    def relative_to_init(self, pose_wt): 
        """ pose of [t] wrt [0]:  p_0t = p_w0.inverse() * p_wt """  
        return (self.init_.inverse()).oplus(pose_wt)

        
class CounterWithPeriodicCallback(Counter): 
    """
    robot_poses = PoseAccumulator(maxlen=1000, relative=True)
    robot_poses_counter = CounterWithPeriodicCallback(
        every_k=10, 
        process_cb=lambda: draw_utils.publish_pose_list('ROBOT_POSES', robot_poses.items, 
                                                        frame_id=ref_frame_id, reset=reset_required())   
    )
    robot_poses_counter.register_callback(robot_poses, 'accumulate')
    """
    def __init__(self, every_k=2, process_cb=lambda: None): 
        Counter.__init__(self)
        self.every_k_ = every_k
        self.process_cb_ = process_cb

    def poll(self): 
        self.count()
        if self.check_divisibility(self.every_k_):
            self.process_cb_()
            self.reset()

    def register_callback(self, cls_instance, function_name): 
        """ Register a wrapped function that polls the counter """

        def polled_function_cb(func):
            def polled_function(*args, **kwargs): 
                self.poll()
                return func(*args, **kwargs)
            return polled_function

        try:
            orig_func = getattr(cls_instance, function_name)
            function_cb = setattr(cls_instance, function_name, polled_function_cb(orig_func))
        except: 
            raise AttributeError('function %s has not been defined in instance' % function_name)
        
        print('Setting new polled callback for %s.%s' % (type(cls_instance).__name__, function_name))
        
class SkippedCounter(Counter): 
    def __init__(self, skip=10, **kwargs): 
        Counter.__init__(self)
        self.skip_ = skip
        self.skipped_ = False

    @property
    def skipped(self): 
        return self.skipped_

    def poll(self): 
        self.skipped_ = True
        if self.check_divisibility(self.skip_):
            self.reset()
            self.skipped_ = False
        self.count()
        return self.skipped_

class SkippedPoseAccumulator(PoseAccumulator): 
    def __init__(self, skip=10, **kwargs): 
        Counter.__init__(self)
        PoseAccumulator.__init__(self, **kwargs)
        self.skip_ = skip
        self.skipped_ = False

    @property
    def skipped(self): 
        return self.skipped_

    def accumulate(self, pose): 
        self.skipped_ = True
        if self.check_divisibility(self.skip_):
            PoseAccumulator.accumulate(self, pose)
            self.reset()
            self.skipped_ = False
        self.count()
    


# class IndexCounter(object): 
#     def __init__(self, start=0): 
#         self._idx = start

#     def increment(self): 
#         idx = np.copy(self._idx)
#         self._idx += 1 
#         return idx
