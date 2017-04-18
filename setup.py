from setuptools import setup
from setuptools import find_packages

setup(name='pybot',
      version='0.1',
      description='Research tools for autonomous systems in Python',
      author='Sudeep Pillai',
      author_email='spillai@csail.mit.com',
      url='https://github.com/spillai/pybot',
      download_url='https://github.com/spillai/pybot/archive/pybot-v0.1.tar.gz',
      license='MIT',
      # install_requires=['tables', 'six'],
      # extras_require={
      #               'h5py': ['h5py'],
      # },
      packages=find_packages())
      
