## Prabuddhi Wariyapperuma (Student ID: 26619055)

# human

This repository contains a ROS package for real-time human detection, tracking individuals within bounding boxes, and estimating thier body postures using Ultralytics YOLOv8 and integrating features for multi-camera support and 2D to 3D pose estimation.  

## How to arrange the setup before running the system

PREREQUISITES:
1. Install ROS Noetic through the link in [here](http://wiki.ros.org/noetic/Installation/Ubuntu) in Ubuntu 20.04.
2. Create a catkin workspace called `catkin_ws` and create an `src` directory by issuing the command `mkdir -p ~/catkin_ws/src`
3. In the source (src) directory, clone the [human](https://github.com/Prabuddhi-05/human.git) repository. 
4. Follow the following commands.

```bash
$ python3 -m pip install -r ultralytics_ros/requirements.txt
$ cd ~/catkin_ws
$ rosdep install -r -y -i --from-paths .
$ catkin_make
```


6. ROS package for real-time object detection using the Ultralytics YOLO, enabling flexible integration with various robotics applications.
7. 
8.





1. Create a catkin workspace called `catkin_ws` and create an `src` directory by issuing the command `mkdir -p ~/catkin_ws/src`
2. In the source directory `catkin_ws/src`, update the system through `sudo apt-get update && sudo apt-get upgrade`
3. In `catkin_ws/src`, install all the dependencies `rosdep install --from-paths . -i -y`
4. Installl scikit-learn thorugh `sudo apt-get install python3-sklearn python3-sklearn-lib`
5. Fork the workshop repository to `catkin_ws/src` through the command `git clone https://github.com/LCAS/CMP9767M.git`
6. Copy the contents of the zip file `Assignment` (after extracted) to `catkin_ws/src`
7. OR instead of Step 6, fork the contents of the `Assignment` repository in GitHub to `catkin_ws/src` through `git clone https://github.com/Prabuddhi-05/Assignment.git`
9. Move to the `catkin_ws` and issue the command `catkin_make` 
10. Source the workspace through the command `source devel/setup.bash`

# How to use the code:
PREREQUISITES:

The robocup_human_sensing package requires the following packages/dependencies in order to be used. Make sure that all packages are cloned into the directory `~/<workspace name>/src` where `<workspace name>` is your workspace name (the default name used is `catkin_ws`).

1. Install ROS Noetic following the steps shown [here](http://wiki.ros.org/noetic/Installation/Ubuntu). 
2. Clone the [openpose](https://github.com/CMU-Perceptual-Computing-Lab/openpose.git) repository. This repository contains the Openpose framework used to extract human skeleton features based on RGB images. Make sure to download and install the OpenPose prerequisites for your particular operating system (e.g. cuda, cuDNN, OpenCV, Caffe, Python). Follow the instructions shown [here](https://github.com/CMU-Perceptual-Computing-Lab/openpose/blob/master/doc/installation/0_index.md).
3. Clone the robocup_human_sensing repository.
4. Run the script `~/<workspace name>/src/robocup_human_sensing/config/install_deps.sh` to install the rest of dependencies and necessary ROS packages.
5. Finally, make sure to build and source the workspace.

HOW TO USE IT:
* The config files into the (`~/robocup_human_sensing/tmule/` directory) have several parameters that can be modified in case some features of are not required for certain tests, e.g. not running the robot camera by using only data from Bag files. Moreover, directories of important files can be modified using these configuration parameters, e.g. the bag files directory, the workspace directory, etc. Make sure to modify the directories according to your system.
* To use the human gesture recognition feature for first time, it is necessary to uncompress the file which contains the trained model. This file is located in the `~/robocup_human_sensing/config/` directory. In the `config` folder, you will also find a global config file named `global_config.yaml` which contains important parameters, directories, and dictionaries used for the gesture and posture recognition. Make sure to modify the directories according to your system. Especially the directory where OpenPose was installed.

To launch any config file into the `~/robocup_human_sensing/tmule/` directory, it is necessary to execute the following commands in terminal:
```
roscd robocup_human_sensing/tmule
tmule -c <config_file_name>.yaml launch
```
To terminate the execution of a specific tmule session:
```
tmule -c <config_file_name>.yaml terminate
```
To monitor the state of every panel launched for the current active tmule sessions:
```
tmux a
```


