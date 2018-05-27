pybot
=============

Research tools for autonomous systems using Python<br>
Author: [Sudeep Pillai](http://people.csail.mit.edu/spillai) [(spillai@csail.mit.edu)](mailto:spillai@csail.mit.edu)  
License: MIT

[![Build Status](https://travis-ci.org/spillai/pybot.svg?branch=py35)](https://travis-ci.org/spillai/pybot)

## Modules

**geometry:** General-purpose tools for computing rigid-body
transformations. This is a preliminary version that currently deals
mostly with **SE(3)** or 6-DOF (3-DoF Rotational + 3-DoF translation)
and some support for **Sim(3)** motions.

**vision:** Computer vision package with several tools including
  camera, tracking, 2d features, 3d features, optical flow,
  recognition, object proposals, caffe, classifier training,
  bag-of-words training, geometry, stereo, drawing etc.

**utils:** Basic tooling that includes attribute dictionaries,
database-utils including incremental hdf5 tables, dataset readers
[ImageDatasets, StereoDatasets, VelodyneDatasets etc], dataset helpers
[[KITTI](http://www.cvlibs.net/datasets/kitti/), itertools
recipes, timing/profiling tools, io utils
[video/image writing, mkdirs, find_files, config parsers, joblib utils, stdout tee-ing, JSON],
other misc tools including pretty prints, progressbars, colored
prints, counters, accumulators (indexed deques), accumulators with
periodic callbacks etc.

**externals:** ROS/LCM drawing tools, ROS/LCM log readers log reader

## Installation

See [INSTALL](INSTALL.md)

### Quick install 

Install pybot and its dependencies into a new conda environment.
```sh
conda config --add channels menpo
conda create --name pybot-py35 python=3.5
conda install --name pybot-py35 -c s_pillai pybot
```
## Contributing
We appreciate all contributions. If you would like to contribute new
features or fix existing bugs, please open an issue and discuss them
with us first. 
