import time

class SimpleTimer: 
    def __init__(self, name='', iterations=10): 
        self.counter_ = 0
        self.last_ = time.time()
        self.name_ = name
        # self.iterations = iterations
        self.period_ = 0

    def poll(self): 
        self.counter_ += 1
        now = time.time()
        dt = (now - self.last_) * 1e3

        if dt > 1000: 
            self.period_ = dt / self.counter_
            print('%s: %4.3f ms (avg over %i iterations)' % (self.name, self.period_, self.counter_))
            self.last_ = now
            self.counter_ = 0

    def poll_piecemeal(self): 
        self.counter_ += 1
        now = time.time()
        dt = (now - self.last_) * 1e3
        self.period_ += dt

        if self.period_ > 100:
            print('%s: %4.3f ms (avg over %i iterations)' % (self.name_, self.period_ / self.counter_, self.counter_))
            self.last_ = now
            self.counter_ = 0
            self.period_ = 0
            
    def start(self): 
        self.last_ = time.time()

    def stop(self): 
        self.poll_piecemeal()

    @property
    def fps(self): 
        try: 
            return 1e3 / self.period_ 
        except: 
            return 0.0
