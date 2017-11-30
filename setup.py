#!/usr/bin/env python

import sys
import os
import errno
import subprocess

from distutils.core import setup
from setuptools import find_packages
from setuptools import Command

def mkdir_p(path):
    try:
        os.makedirs(path)
    except os.error as exc:
        if exc.errno != errno.EEXIST or \
           not os.path.isdir(path):
            raise

print('Configuring...')
mkdir_p('cmake_build')
subprocess.Popen(['cmake','../pybot/src'],
                 cwd='cmake_build').wait()

print('Compiling extension...')
subprocess.Popen(['make','-j4'], cwd='cmake_build').wait()

print('Building package')
GITHUB_URL = 'https://github.com/spillai/pybot'
DOWNLOAD_URL = GITHUB_URL + '/archive/pybot-v0.1.tar.gz'
setup(
    name='pybot',
    version='0.2',
    description='Research tools for mobile robots',
    author='Sudeep Pillai',
    license='MIT',    
    author_email='spillai@csail.mit.edu',
    url=GITHUB_URL,
    download_url='', 
    packages=find_packages(),
    scripts=[],
    package_data={
        'pybot': ['pybot_types.so']
    },
)
