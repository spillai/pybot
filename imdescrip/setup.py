#!/usr/bin/env python
""" Setup utility for the imdescrip package. """


from distutils.core import setup

setup(
    name='imdescrip',
    version='1.0',
    description='Image descriptor extraction routines.',
    author='Daniel Steinberg',
    author_email='d.steinberg@acfr.usyd.edu.au',
    url='https://github.com/dsteinberg/imdescrip',
    packages=[
        'imdescrip',
        'imdescrip.descriptors', 
        'imdescrip.utils'
        ],
    install_requires=[
        "scipy >= 0.9.0", 
        "numpy >= 1.6.1",
        "spams >= 2.3",
        "pyvlfeat >= 0.1.1a3"
        ]
)
