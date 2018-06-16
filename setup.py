#!/usr/bin/env python
import errno
import io
import platform
import os
import re
import sys
import sysconfig
import subprocess

from distutils.version import LooseVersion
from setuptools import setup, Extension, find_packages, Command
from setuptools.command.build_ext import build_ext
from setuptools.command.test import test as TestCommand
from shutil import copyfile, copymode

GITHUB_URL = 'https://github.com/spillai/pybot'
README = open('README.md').read()

def mkdir_p(path):
    try:
        os.makedirs(path)
    except os.error as exc:
        if exc.errno != errno.EEXIST or \
           not os.path.isdir(path):
            raise

def read(*names, **kwargs):
    with io.open(
        os.path.join(os.path.dirname(__file__), *names),
        encoding=kwargs.get("encoding", "utf8")
    ) as fp:
        return fp.read()

def find_version(*file_paths):
    version_file = read(*file_paths)
    version_match = re.search(r"^__version__ = ['\"]([^'\"]*)['\"]",
                              version_file, re.M)
    if version_match:
        return version_match.group(1)
    raise RuntimeError("Unable to find version string.")


print('Configuring...')
mkdir_p('cmake_build')
subprocess.Popen(['cmake','../pybot/src'],
                 cwd='cmake_build').wait()

print('Compiling extension...')
subprocess.Popen(['make','-j4'], cwd='cmake_build').wait()

print('Building package')
print('Found: {}'.format(find_packages()))
def copy_dir(dir_path):
    base_dir = os.path.join('pybot', dir_path)
    for (dirpath, dirnames, files) in os.walk(base_dir):
        for f in files:
            if '.pyc' in f: continue
            yield os.path.join(dirpath.split('/', 1)[1], f)

VERSION = find_version('pybot', '__init__.py')
setup(
    name='pybot',
    version=VERSION,
    description='Research tools for autonomous systems in Python',
    long_description=README,
    author='Sudeep Pillai',
    author_email='spillai@csail.mit.edu',
    license='MIT',
    url=GITHUB_URL,
    packages=find_packages(exclude=('tests',)),
    package_dir={'pybot': 'pybot'},
    package_data={
        'pybot': ['pybot_types.so'],
        'externals': [f for f in copy_dir('externals/viewer')]
    },
    zip_safe=False,
    include_package_data=True
)
