from collections import defaultdict

class MayBeCalled(object):
    def __call__(self, *args, **kwargs):
        return None

class nop(object):
    """
    Nop class to handle misc optional imports
    Shamelessly ripped off 
    from http://stackoverflow.com/questions/24946321/how-do-i-write-a-no-op-or-dummy-class-in-python
    """
    def __init__(self, name=''): 
        self.name_ = name

    def __get__(self, *args):
        return MayBeCalled()

    def __hasattr__(self, attr):
        if len(self.name_): print('{}::{}'.format(self.name_, attr))
        return True

    def __getattr__(self, attr):
        if len(self.name_): print('{}::{}'.format(self.name_, attr))
        return MayBeCalled()

