Installation
---
Most of the installation assumes you have a clean python virtual
environment to work with. I prefer conda for my development
environment, but you're free to use any alternative (as long as you do
not globally install, in which case I can not provide much support).

1) Install miniconda and setup path in `~/.bashrc`
```sh
wget --no-check-certificate https://repo.continuum.io/miniconda/Miniconda3-latest-Linux-x86_64.sh
bash Miniconda3-latest-Linux-x86_64.sh -b -p $HOME/anaconda
```

2) Install pybot and its dependencies into a new conda environment
```sh
conda config --add channels menpo
conda create --name pybot python=3.5
source activate pybot
conda install -c s_pillai pybot
```
Alternatively, if you'd like to add **pybot** to an already existing
environment,
```sh
conda config --add channels menpo
conda install --name <existing_environment> -c s_pillai pybot
```

3) (Optional) ROS dependencies (reading bag files etc) into the same
environment. First ensure that you are within the **pybot** virtual
environment.
```sh
source activate pybot
pip install catkin_pkg rospkg
```
 
 **Use at your own risk**
