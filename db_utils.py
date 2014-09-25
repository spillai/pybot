#!/usr/bin/python
    
class AttrDict(dict):
    def __init__(self, *args, **kwargs):
        super(AttrDict, self).__init__(*args, **kwargs)
        self.__dict__ = self

    # __getattr__ = dict.__getitem__
    # __setattr__ = dict.__setitem__
