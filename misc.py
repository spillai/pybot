from collections import deque

def print_red(prt): print("\033[91m {}\033[00m" .format(prt))
def print_green(prt): print("\033[92m {}\033[00m" .format(prt))
def print_yellow(prt): print("\033[93m {}\033[00m" .format(prt))
def print_lightpurple(prt): print("\033[94m {}\033[00m" .format(prt))
def print_purple(prt): print("\033[95m {}\033[00m" .format(prt))
def print_cyan(prt): print("\033[96m {}\033[00m" .format(prt))
def print_lightgray(prt): print("\033[97m {}\033[00m" .format(prt))
def print_black(prt): print("\033[98m {}\033[00m" .format(prt))

cols_lut = { 'red': print_red, 'green': print_green, 'yellow': print_yellow, 
             'lightpurple': print_lightpurple, 'purple': print_purple, 
             'cyan': print_cyan, 'lightgray': print_lightgray, 'black': print_black }
def print_color(prt, color='green'): 
    try: 
        cols_lut[color](prt)
    except: 
        raise KeyError('print_color lookup failed, use from {:}'.format(cols_lut.keys()))

def setup_pbar(maxval): 
    import progressbar as pb
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
        return self.idx_-1

    @property
    def length(self): 
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

    def append(self, item): 
        self.accumulate(item)

    def extend(self, items): 
        self.accumulate_list(items)

    def __len__(self): 
        return len(self.items_)

    def __getitem__(self, index):
        return self.items_[index]

    def __setitem__(self, index, value):
        self.items_[index] = value

    @property
    def latest(self): 
        return self.items_[-1]

    @property
    def first(self): 
        return self.items_[0]

    @property
    def items(self): 
        return self.items_
        
    # @property
    # def length(self): 
    #     return len(self.items_)


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

class PoseInterpolator(PoseAccumulator): 
    def __init__(self, maxlen=100, relative=False): 
        PoseAccumulator.__init__(self, maxlen=maxlen, relative=relative)

        self.relative_ = relative
        self.init_ = None
        
    def add(self, pose): 
        super(PoseAccumulator, self).accumulate(pose)

    def query(self, pose): 
        raise NotImplementedError()

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

    @property
    def every_k(self): 
        return self.every_k_

    def poll(self): 
        self.count()
        if self.check_divisibility(self.every_k_):
            self.process_cb_()
            self.reset()
            return True
        return False

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
        except Exception, e:
            raise AttributeError('function %s has not been defined in instance {:}'.format(function_name, e))
        
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
