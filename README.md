## Prabuddhi Wariyapperuma (Student ID: 26619055)

# human

This repository contains a ROS package for real-time human detection, tracking humans within bounding boxes, and estimating their body postures using Ultralytics YOLOv8 and integrating features for multi-camera support, 2D to 3D pose estimation and visualization of 3D pose using human stick figures.  

## How to arrange the setup before running the system

PREREQUISITES:
1. Install ROS Noetic through the link in [here](http://wiki.ros.org/noetic/Installation/Ubuntu) in Ubuntu 20.04.
2. Create a catkin workspace called `catkin_ws` and create an `src` directory by issuing the command `mkdir -p ~/catkin_ws/src`
3. In the source (`src` ) directory, clone the [human](https://github.com/Prabuddhi-05/human.git) repository. 
4. Execute the following commands in the terminal.

```bash
$ python3 -m pip install -r ultralytics_ros/requirements.txt
$ cd ~/catkin_ws
$ rosdep install -r -y -i --from-paths .
```
5. In `src`, clone the [realsense_ros_gazebo](https://github.com/nilseuropa/realsense_ros_gazebo.git) repository. This package extends RealSense camera support to the Gazebo simulation environment.
6. In `src`, clone the [realsense-ros](https://github.com/IntelRealSense/realsense-ros.git) repository. It consists of the package to integrate RealSense cameras into ROS.
7. Move to the `catkin_ws` and build and source the workspace by issuing the commands `catkin_make` and `source devel/setup.bash` respectively.
8. In the home directory, clone the [husky](https://github.com/husky/husky.git) repository. This provides ROS support to the Husky robot model.
9. Copy/replace the following files located in the `~/ultralytics_ros/resources` to the relevant new locations.
*  `husky.urdf.xacro` file to location `~/opt/ros/noetic/share/husky_description/urdf`
*  `robot_husky.rviz` and `robot_husky_left.rviz` files to the location `~/opt/ros/noetic/share/husky_viz/rviz`
*  `view_people.launch` and `view_people_left.launch` files to the location `~/opt/ros/noetic/share/husky_viz/launch`

HOW TO USE IT:

The table below gives the list of launch files in the repository. 

| No. | Launch file name | Purpose of the launch file | How to use | 
|-----------------|-----------------|-----------------|-----------------|
|1| tracker_real.launch|  For human detection on raw color images from different image sources, from a RealSense camera or a web camera, by modifying the "image_topic" parameter accordingly| |
|2| tracker_bag.launch | For human detection on color images obtained by playing back the content of a ROS bag file| |
|3| husky.launch|To launch Gazebo in a simulation environment represented by a cafe world model, and to spawn a Husky robot model into this cafe world| |
|4| tracker_gaz_one.launch|For human detection on color images obtained from a single camera mounted on the Husky robot| Run with 3|
|5| tracker_gaz.launch|For human detection on color images obtained from all four cameras mounted on the Husky robot with the "multiple-camera single-detector" method| Run with 3| 
|6| tracker_gaz_multiple.launch|For human detection on color images obtained from all four cameras mounted on the Husky robot with the "multiple-camera multiple-detector" method| Run with 3| 
|7|tracker_left.launch|For 2D to 3D pose estimation, visualization of 3D pose as human stick figures, reconstruction of stick figures with missing joints and construction of stick figures with sensible joint lengths for the images captured from a single camera mounted on the Husky robot|Run with 3 and 9| 
|8| tracker_all.launch|For 2D to 3D pose estimation, visualization of 3D pose as human stick figures, reconstruction of stick figures with missing joints and construction of stick figures with sensible joint lengths for the images captured from a multiple cameras mounted on the Husky robot|Run with 3 and 10|
|9| view_people_left.launch|To launch RViz for visualizing the Husky robot's data from a single camera mounted on the Husky robot||  
|10|view_people.launch|To launch RViz for visualizing the Husky robot's data from all four cameras mounted on the Husky robot|| 


References : [ultralytics_ros](https://github.com/Alpaca-zip/ultralytics_ros.git)
