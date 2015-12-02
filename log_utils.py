import os.path

class LogReader(object): 
    def __init__(self, filename, decoder=None, start_idx=0, every_k_frames=1, index=False):
        filename = os.path.expanduser(filename)
		
        if filename is None or not os.path.exists(os.path.expanduser(filename)):
            raise Exception('Invalid Filename: %s' % filename)

        # Store attributes
        self.filename = filename
        self.decoder = decoder
        self.every_k_frames = every_k_frames
        self.start_idx = start_idx
        
        # Load the log
        self._log = self.load_log(self.filename)

        # Build index
        self.idx = 0
        if index: 
            self._index()
        else: 
            self.index = None

    def load_log(self, filename): 
        raise NotImplementedError('load_log not implemented in LogReader')
