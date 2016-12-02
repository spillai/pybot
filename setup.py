from setuptools import setup
from setuptools import find_packages


setup(name='pybot',
      version='0.1',
      description='Research tools for autonomous systems using Python',
      author='Sudeep Pillai',
      author_email='spillai@csail.mit.com',
      url='https://github.com/spillai/pybot',
      download_url='https://github.com/pybot/tarball/0.0.1',
      license='MIT',
      # install_requires=['tables', 'six'],
      # extras_require={
      #               'h5py': ['h5py'],
      # },
      packages=find_packages())
      
