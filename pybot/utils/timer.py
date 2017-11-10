# Author: Sudeep Pillai <spillai@csail.mit.edu>
# License: MIT
from __future__ import print_function

import time
from collections import OrderedDict
from functools import wraps

def print_green(prt): print("\033[92m {}\033[00m" .format(prt))

global g_timers
g_timers = OrderedDict()

def named_timer(name): 
    global g_timers
    header = '\n' if len(g_timers) == 0 else ''
    if name not in g_timers: 
        g_timers[name] = SimpleTimer(name, header=header)
    try: 
        return g_timers[name] 
    except KeyError as e: 
        raise RuntimeError('Failed to retrieve timer {:}'.format(e))

def timeitmethod(func):
    @wraps(func)
    def wrapper(*args, **kwargs):
        try: 
            name = ''.join([args[0].__class__.__name__, '::', func.__name__])
        except: 
            raise RuntimeError('timeitmethod requires first argument to be self')
        named_timer(name).start()
        r = func(*args, **kwargs)
        named_timer(name).stop()
        return r
    return wrapper


def timeit(func):
    @wraps(func)
    def wrapper(*args, **kwargs):
        try: 
            name = ''.join([func.__name__])
        except: 
            raise RuntimeError('timeitmethod requires first argument to be self')
        named_timer(name).start()
        r = func(*args, **kwargs)
        named_timer(name).stop()
        return r
    return wrapper

class SimpleTimer: 
    def __init__(self, name='', hz=0.5, header=''): 
        self.name_ = name
        self.hz_ = hz
        self.header_ = header

        self.counter_ = 0
        self.last_ = time.time()
        self.period_ = 0
        self.calls_ = 0

        self.last_print_ = time.time()        
        self.last_fps_ = 0

    def __enter__(self):
        self.start()
        return self
    
    def __exit__(self, *args):
        self.stop(force_print=True)
        return self
        
    def poll(self): 
        self.counter_ += 1
        now = time.time()
        dt = (now - self.last_)

        if (now-self.last_print_) > 1.0 / self.hz_: 
            T = dt / self.counter_
            fps = 1.0 / T
            self.calls_ += self.counter_
            print_green('{:s}\t[{:5.1f} ms, {:5.1f} Hz, {:d} ]\t{:s}'
                        .format(self.header_, T * 1e3, fps, int(self.calls_), self.name_))
            self.last_ = now
            self.last_print_ = now
            self.counter_ = 0

    def poll_piecemeal(self, force_print=False): 
        self.counter_ += 1
        now = time.time()
        dt = (now - self.last_)
        self.period_ += dt

        if (now-self.last_print_) > 1.0 / self.hz_ or force_print:
            T = self.period_ / self.counter_
            fps = 1.0 / T
            self.calls_ += self.counter_
            print_green('{:s}\t[{:5.1f} ms, {:5.1f} Hz, {:d} ]\t{:s}'
                        .format(self.header_, T * 1e3, fps, int(self.calls_), self.name_))
            self.last_ = now
            self.last_print_ = now
            self.last_fps_ = fps
            self.counter_ = 0
            self.period_ = 0
            
    def start(self): 
        self.last_ = time.time()

    def stop(self, force_print=False): 
        self.poll_piecemeal(force_print=force_print)

    @property
    def fps(self): 
        return self.last_fps_
        
