import time
from collections import defaultdict

class Profiler(object): 
    def __init__(self): 
        self.elapsed_time = 0.0
        self._last, self._counts = 0, 0

    def start(self): 
        if self._last != 0: 
            raise RuntimeError('Called start() multiple times without stop')
        self._last = time.time()
        self._counts += 1

    def stop(self): 
        self.elapsed_time += (time.time() - self._last)
        self._last = 0

    @property
    def elapsed(self): 
        assert(self._counts > 0)
        return self.elapsed_time * 1.0 / self._counts
        
    def __repr__(self): 
        return '%4.3f s' % self.elapsed

class ProfilerReport(defaultdict): 
    def __init__(self): 
        defaultdict.__init__(self, Profiler)
        
    def __repr__(self): 
        report_str = 'Report: \n'
        for k,v in self.iteritems(): 
            report_str += '\t%s: \t%s\n' % (k,v)
        return report_str

if __name__ == "__main__": 
    r = ProfilerReport()
    r['test'].start()
    time.sleep(0.123)
    r['test'].stop()

    r['test'].start()
    time.sleep(0.125)
    r['test'].stop()

    r['test2'].start()
    time.sleep(0.125)
    r['test2'].stop()
    print r

    
