# pybot
===

Research tools for autonomous systems using Python<br>
**Author: [Sudeep Pillai](http://people.csail.mit.edu/spillai)** (spillai@csail.mit.edu)<br>
**License: MIT**<br>

# pybot_geometry
===

**pybot_geometry** is Python package that provides general-purpose tools for fast
prototyping of robotics applications (such as Visual Odometry, SLAM) that
require computing rigid-body transformations. This is a preliminary version that
currently deals mostly with **SE(3)** or 6-DOF (3-DoF Rotational + 3-DoF
translation) and some support for **Sim(3)** motions.

pybot_externals
===

Bot Viewer capabilities in Python<br>


Dependencies
---
[bot_geometry](https://github.com/spillai/pybot_geometry), [NumPy](https://github.com/numpy/numpy), [lcm](https://github.com/lcm-proj/lcm), [collections viewer](https://github.mit.edu/mrg/visualization-pod), [libbot](https://github.com/RobotLocomotion/libbot)

All the dependencies need to be available in the `PYTHONPATH`. 

With ROS (reading bag files etc): `pip install catkin_pkg rospkg`

Examples
---
All the 3D visualization demos for the works below were created using the above set of tools. <br>
[Monocular SLAM-Supported Object Recognition](https://www.youtube.com/watch?v=m6sStUk3UVk), 
[High-speed Stereo Reconstruction](http://people.csail.mit.edu/spillai/projects/fast-stereo-reconstruction/pillai_fast_stereo16.mp4)
