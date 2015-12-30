import os.path

class Decoder(object): 
    def __init__(self, channel='', every_k_frames=1, decode_cb=lambda data: None): 
        self.channel = channel
        self.every_k_frames = every_k_frames
        self.decode_cb = decode_cb
        self.idx = 0

    def decode(self, data): 
        try: 
            return self.decode_cb(data)
        except Exception as e:
            print e
            raise RuntimeError('Error decoding channel: %s by %s' % (self.channel, self))

    def can_decode(self, channel): 
        return self.channel == channel

    def should_decode(self): 
        self.idx += 1
        return self.idx % self.every_k_frames == 0 

class LogReader(object): 
    def __init__(self, filename, decoder=None, start_idx=0, every_k_frames=1, index=False):
        filename = os.path.expanduser(filename)
		
        if filename is None or not os.path.exists(os.path.expanduser(filename)):
            raise Exception('Invalid Filename: %s' % filename)

        # Store attributes
        self.filename = filename
        if isinstance(decoder, list): 
            self.decoder = { dec.channel: dec for dec in decoder } 
        else: 
            self.decoder = { decoder.channel: decoder }
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

        # Create Look-up table for subscriptions
        self.cb_ = {}


    def load_log(self, filename): 
        raise NotImplementedError('load_log not implemented in LogReader')

    def subscribe(self, channel, callback): 
        self.cb_[channel] = callback

    def check_tf_relations(self, relations): 
        raise NotImplementedError()

    def establish_tfs(self, relations): 
        raise NotImplementedError()

    def iteritems(self): 
        raise NotImplementedError()

    def iter_frames(self): 
        raise NotImplementedError()

    def run(self):
        if not len(self.cb_): 
            raise RuntimeError('No callbacks registered yet, subscribe to channels first!')

        for self.idx, (t, ch, data) in enumerate(self.iter_frames()): 
            try: 
                self.cb_[ch](t, data)
            except KeyError: 
                pass
            except Exception, e: 
                import traceback
                traceback.print_exc()
                raise RuntimeError()
