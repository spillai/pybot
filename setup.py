#!/usr/bin/env python

import errno
import platform
import os
import re
import sys
import sysconfig
import subprocess

# from distutils.core import setup
# from setuptools import find_packages
# from setuptools import Command

from distutils.version import LooseVersion
from setuptools import setup, Extension, find_packages, Command
from setuptools.command.build_ext import build_ext
from setuptools.command.test import test as TestCommand
from shutil import copyfile, copymode

# def mkdir_p(path):
#     try:
#         os.makedirs(path)
#     except os.error as exc:
#         if exc.errno != errno.EEXIST or \
#            not os.path.isdir(path):
#             raise

# print('Configuring...')
# mkdir_p('cmake_build')
# subprocess.Popen(['cmake','../pybot/src'],
#                  cwd='cmake_build').wait()

# print('Compiling extension...')
# subprocess.Popen(['make','-j4'], cwd='cmake_build').wait()

# class CMakeExtension(Extension):
#     def __init__(self, name, sourcedir=''):
#         Extension.__init__(self, name, sources=[])
#         self.sourcedir = os.path.abspath(sourcedir)

# class CMakeBuild(build_ext):
#     def run(self):
#         try:
#             out = subprocess.check_output(['cmake', '--version'])
#         except OSError:
#             raise RuntimeError(
#                 "CMake must be installed to build the following extensions: " +
#                 ", ".join(e.name for e in self.extensions))

#         if platform.system() == "Windows":
#             cmake_version = LooseVersion(re.search(r'version\s*([\d.]+)',
#                                          out.decode()).group(1))
#             if cmake_version < '3.1.0':
#                 raise RuntimeError("CMake >= 3.1.0 is required on Windows")

#         for ext in self.extensions:
#             self.build_extension(ext)

#     def build_extension(self, ext):
#         extdir = os.path.abspath(
#             os.path.dirname(self.get_ext_fullpath(ext.name)))
#         cmake_args = ['-DCMAKE_LIBRARY_OUTPUT_DIRECTORY=' + extdir,
#                       '-DPYTHON_EXECUTABLE=' + sys.executable]

#         cfg = 'Debug' if self.debug else 'Release'
#         build_args = ['--config', cfg]

#         if platform.system() == "Windows":
#             cmake_args += ['-DCMAKE_LIBRARY_OUTPUT_DIRECTORY_{}={}'.format(
#                 cfg.upper(),
#                 extdir)]
#             if sys.maxsize > 2**32:
#                 cmake_args += ['-A', 'x64']
#             build_args += ['--', '/m']
#         else:
#             cmake_args += ['-DCMAKE_BUILD_TYPE=' + cfg]
#             build_args += ['--', '-j2']

#         env = os.environ.copy()
#         env['CXXFLAGS'] = '{} -DVERSION_INFO=\\"{}\\"'.format(
#             env.get('CXXFLAGS', ''),
#             self.distribution.get_version())
#         if not os.path.exists(self.build_temp):
#             os.makedirs(self.build_temp)
#         subprocess.check_call(['cmake', ext.sourcedir] + cmake_args,
#                               cwd=self.build_temp, env=env)
#         subprocess.check_call(['cmake', '--build', '.'] + build_args,
#                               cwd=self.build_temp)
#         # Copy *_test file to tests directory
#         test_bin = os.path.join(self.build_temp, 'python_cpp_example_test')
#         self.copy_test_file(test_bin)
#         print()  # Add an empty line for cleaner output

#     def copy_test_file(self, src_file):
#         '''
#         Copy ``src_file`` to ``dest_file`` ensuring parent directory exists.
#         By default, message like `creating directory /path/to/package` and
#         `copying directory /src/path/to/package -> path/to/package` are displayed on standard output. Adapted from scikit-build.
#         '''
#         # Create directory if needed
#         dest_dir = os.path.join(os.path.dirname(
#             os.path.abspath(__file__)), 'tests', 'bin')
#         if dest_dir != "" and not os.path.exists(dest_dir):
#             print("creating directory {}".format(dest_dir))
#             os.makedirs(dest_dir)

#         # Copy file
#         dest_file = os.path.join(dest_dir, os.path.basename(src_file))
#         print("copying {} -> {}".format(src_file, dest_file))
#         copyfile(src_file, dest_file)
#         copymode(src_file, dest_file)


print('Building package')
GITHUB_URL = 'https://github.com/spillai/pybot'
# DOWNLOAD_URL = GITHUB_URL + '/archive/pybot-v0.1.tar.gz'

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
    description='Research tools for autonomous systems in Python',
    author='Sudeep Pillai',
    license='MIT',    
    author_email='spillai@csail.mit.edu',
    url=GITHUB_URL,
    download_url='', 
    packages=find_packages(),
    # ext_modules=[CMakeExtension('src/')],
    # cmdclass=dict(build_ext=CMakeBuild),
    scripts=[],
    package_dir={'pybot': 'pybot'},
    package_data={
        'pybot': ['pybot_types.so'],
        '': [f for f in copy_dir('externals/viewer')]
    },
    zip_safe=False
)
