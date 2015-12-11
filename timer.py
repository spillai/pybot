import time

class SimpleTimer: 
    def __init__(self, name='', iterations=10): 
        self._counter = 0
        self._last = time.time()
        self._name = name
        self.iterations = iterations
        self.period_ = 0

    def poll(self): 
        self._counter += 1
        if self._counter == self.iterations: 
            self._counter = 0
            now = time.time()
            self.period_ = (now - self._last) * 1e3 / self.iterations
            print('%s: %4.3f ms (avg over %i iterations)' % (self._name, self.period_, self.iterations))
            self._last = now

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
