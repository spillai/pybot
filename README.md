pybot
---

Research tools for autonomous systems using Python
Author: [Sudeep Pillai](http://people.csail.mit.edu/spillai) [(spillai@csail.mit.edu)](mailto:spillai@csail.mit.edu)  
License: MIT

[![Build Status](https://travis-ci.org/spillai/pybot.svg?branch=py35)](https://travis-ci.org/spillai/pybot)

Modules
---
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
[[KITTI](http://www.cvlibs.net/datasets/kitti/),
[NYU-RGBD](http://cs.nyu.edu/~silberman/datasets/nyu_depth_v2.html),
[SUN3D](http://sun3d.cs.princeton.edu/),
[Tsukuba Stereo](http://cvlab-home.blogspot.com/2012/05/h2fecha-2581457116665894170-displaynone.html),
,
[VaFRIC](https://www.doc.ic.ac.uk/~ahanda/VaFRIC/test_datasets.html),
[UW-RGBD](https://rgbd-dataset.cs.washington.edu/)], itertools
recipes, timing/profiling tools, io utils
[video/image writing, mkdirs, find_files, config parsers, joblib utils, stdout tee-ing, JSON],
other misc tools including pretty prints, progressbars, colored
prints, counters, accumulators (indexed deques), accumulators with
periodic callbacks etc.

**externals:** ROS/LCM drawing tools, ROS/LCM log readers, [Google
  Project Tango](https://get.google.com/tango/) log reader

**New (upcoming) modules**: *mapping*, *nnet* with support for visual
odometry, vSLAM, semi-dense reconstruction, and more recent
developments in CNN-based object recognition, localization (ROI
pooling), and SLAM-aware recognition. See [references](REFERENCES.md) for more details
about specific implementations. 

Installation
---
See [INSTALL](INSTALL.md)
