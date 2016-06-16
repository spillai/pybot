import numpy as np
from distutils.core import setup
from distutils.extension import Extension
from Cython.Distutils import build_ext

cmdclass = {}
ext_modules = [
    Extension(
        "recognition.cython_bbox",
        ["recognition/bbox.pyx"],
        extra_compile_args=["-Wno-cpp", "-Wno-unused-function"],
    ),
    Extension(
        "recognition.cython_nms",
        ["recognition/nms.pyx"],
        extra_compile_args=["-Wno-cpp", "-Wno-unused-function"],
    )
]
cmdclass.update({'build_ext': build_ext})

setup(
    name='bot_vision',
    cmdclass=cmdclass,
    ext_modules=ext_modules,
    include_dirs=[np.get_include()]
)
