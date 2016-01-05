import time

class SimpleTimer: 
    def __init__(self, name='', iterations=10): 
        self._counter = 0
        self._last = time.time()
        self._name = name
        # self.iterations = iterations
        self.period_ = 0

    def poll(self): 
        self._counter += 1
        now = time.time()
        if now - self._last > 1: 
            self.period_ = (now - self._last) * 1e3 / self._counter
            print('%s: %4.3f ms (avg over %i iterations)' % (self._name, self.period_, self._counter))
            self._last = now
            self._counter = 0
            
    def start(self): 
        self._last = time.time()

    def stop(self): 
        self.poll()

    @property
    def fps(self): 
        try: 
            return 1e3 / self.period_ 
        except: 
            return 0.0
