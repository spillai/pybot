import time

class SimpleTimer: 
    def __init__(self, iterations=10): 
        self._counter = 0
        self._last = time.time()
        self.iterations = iterations
    
    def poll(self): 
        self._counter += 1
        if self._counter == self.iterations: 
            self._counter = 0
            now = time.time()
            print 'Profile: %4.3f ms' % ((now - self._last) * 1e3 / self.iterations)
            self._last = now

        
