
class IndexCounter(object): 
    def __init__(self, start=0): 
        self._idx = start

    def increment(self): 
        idx = np.copy(self._idx)
        self._idx += 1 
        return idx
