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

print('Found: {}'.format(find_packages()))
def copy_dir(dir_path):
    base_dir = os.path.join('pybot', dir_path)
    for (dirpath, dirnames, files) in os.walk(base_dir):
        for f in files:
            if '.pyc' in f: continue
            yield os.path.join(dirpath.split('/', 1)[1], f)

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
    package_dir={'pybot': 'pybot'},
    package_data={
        'pybot': ['pybot_types.so'],
        '': [f for f in copy_dir('externals/viewer')]
    },
    zip_safe=False
)
