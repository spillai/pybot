import os.path
from pkgutil import extend_path

pathdir, _ = os.path.split(__path__[0])
newpath = os.path.join(pathdir, 'vision_private')
__path__ = extend_path([newpath], __name__)
