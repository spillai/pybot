Installation
---
Most of the installation assumes you have a clean python virtual
environment to work with. I prefer conda for my development
environment, but you're free to use any alternative (as long as you do
not globally install, in which case I can not provide much support).

1) Install miniconda and setup path in `~/.bashrc`
```sh
wget --no-check-certificate https://repo.continuum.io/miniconda/Miniconda2-latest-Linux-x86_64.sh
bash Miniconda2-latest-Linux-x86_64.sh -b -p $HOME/anaconda
```

2) Install dependencies into a new conda environment
```sh
conda config --add channels menpo
conda create --name pybot --file conda_requirements.txt
```
Alternatively, if you'd like to add **pybot** to an already existing
environment,
```sh
conda config --add channels menpo
conda install --name pybot --file conda_requirements.txt
```

3) (Optional) ROS dependencies (reading bag files etc) into the same
environment. First ensure that you are within the **pybot** virtual
environment.
```sh
source activate pybot
pip install catkin_pkg rospkg
```

Dependencies
---
1) OpenCV

**pybot** heavily relies on OpenCV 2.4.11 for most of the computationally
expensive computer vision procedures. We rely on the
[menpo](https://anaconda.org/menpo/opencv) conda channel for the
specific version of OpenCV (that is pre-compiled for your platform).
```sh
conda install -c https://conda.anaconda.org/menpo opencv=2.4.11 -y
```

2) Tensorflow [Setup](https://www.tensorflow.org/versions/r0.11/get_started/os_setup.html#pip-installation)  
 Conda handles all the dependencies for **pybot** based on your
 architecture. **pybot** has limited test coverage at the moment, so
 please use at your own risk.  
 
 **Use at your own risk**
