pybot
---

Research tools for autonomous systems using Python<br>
**Author: [Sudeep Pillai](http://people.csail.mit.edu/spillai)** ([spillai@csail.mit.edu](mailto:spillai@csail.mit.edu))<br>
**License: MIT**<br>

Modules
---
**geometry** is Python package that provides general-purpose tools for fast
prototyping of robotics applications (such as Visual Odometry, SLAM) that
require computing rigid-body transformations. This is a preliminary version that
currently deals mostly with **SE(3)** or 6-DOF (3-DoF Rotational + 3-DoF
translation) and some support for **Sim(3)** motions.

**externals** Bot Viewer capabilities in Python<br>

Dependencies
---
[bot_geometry](https://github.com/spillai/pybot_geometry), [NumPy](https://github.com/numpy/numpy), [lcm](https://github.com/lcm-proj/lcm), [collections viewer](https://github.mit.edu/mrg/visualization-pod), [libbot](https://github.com/RobotLocomotion/libbot)

Install miniconda and OpenCV from menpo
```sh
wget --no-check-certificate https://repo.continuum.io/miniconda/Miniconda2-latest-Linux-x86_64.sh
bash Miniconda2-latest-Linux-x86_64.sh -b -p $HOME/anaconda
conda install -c https://conda.anaconda.org/menpo opencv
```
Menpo packages: opencv 2.4.11
[ [OSX](https://anaconda.org/menpo/opencv/2.4.11/download/osx-64/opencv-2.4.11-py27_1.tar.bz2),  [x64](https://anaconda.org/menpo/opencv/2.4.11/download/linux-64/opencv-2.4.11-nppy27_0.tar.bz2) ]


All the dependencies need to be available in the `PYTHONPATH`. 
With ROS (reading bag files etc): `pip install catkin_pkg rospkg`

Examples
---
All the 3D visualization demos for the works below were created using the above set of tools. <br>
[Monocular SLAM-Supported Object Recognition](https://www.youtube.com/watch?v=m6sStUk3UVk), 
[High-speed Stereo Reconstruction](http://people.csail.mit.edu/spillai/projects/fast-stereo-reconstruction/pillai_fast_stereo16.mp4)


Tensorflow (Linux 64-bit)
---

Ubuntu/Linux 64-bit, CPU only, Python 2.7
```sh
$ export TF_BINARY_URL=https://storage.googleapis.com/tensorflow/linux/cpu/tensorflow-0.10.0-cp27-none-linux_x86_64.whl
```

Ubuntu/Linux 64-bit, GPU enabled, Python 2.7
Requires CUDA toolkit 7.5 and CuDNN v5. For other versions, see
"Install from sources" below.
```sh
$ export
TF_BINARY_URL=https://storage.googleapis.com/tensorflow/linux/gpu/tensorflow-0.10.0-cp27-none-linux_x86_64.whl
```

Tensorflow (Mac 64-bit)
---

Mac OS X, CPU only, Python 2.7:
```sh
$ export
TF_BINARY_URL=https://storage.googleapis.com/tensorflow/mac/cpu/tensorflow-0.10.0-py2-none-any.whl
```

# Mac OS X, GPU enabled, Python 2.7:
```sh
$ export
TF_BINARY_URL=https://storage.googleapis.com/tensorflow/mac/gpu/tensorflow-0.10.0-py2-none-any.whl
```
