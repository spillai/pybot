# Simple image descriptor for just testing stuff
from descriptor import Descriptor

class TestDesc (Descriptor):

    def __init__ (self):
        pass

    def extract (self, image):
        return (image.mean(axis=0)).mean(axis=0)
