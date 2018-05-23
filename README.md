# new_detect
a ROS,ZED SSD,Tensorflow package

- A ros package for autonomous driving object detection.
- This performance is better than zed_ros_kerasssd_pkg

## Function
It contains modules of:
1. A tensorflow implemented SSD
2. Subscribe image info from ZED Camera ROS node, obtained the PointCloud information of Object bounding box center. calculate distance of object. 
3. Publish a ROS message about the object information(object type/confidence/distance/)
4. ROS node name: new_detect_pkg
5. ROS message name: new_obj_info

## Environment
- Ubuntu16.04
- CUDA 9.0
- CUDNN 7.0.5
- ROS Kinetic
- ZED Camera SDK 2.2.1
- zed-ros-wrapper-2.2.x
- Python version 3.5
- tensorflow-gpu (1.5.0rc1)
- cv-bridge (1.12.7)


## Pre-install
### Environment
### Others:
- cv_bridge for python3

## Usage
clone and unziped into your ROS workspace/src(suppose workspace names catkin_ws)


```
# for the first time
cd ~/catkin_ws 
catkin_make -DCATKIN_WHITELIST_PACKAGES=new_detect
cd src/detect_pkg/
chmod u+x bin/hello
cd ~/catkin_ws
. devel/setup.bash

# run it!!
roslaunch zed_wrapper zed.launch
# open a new terminal
rosrun new_detect hello

# check ROS Node statues
# open a new terminal
rostopic echo -p /new_obj_info
```


## Performance
GTX GFORCE 1070: 7FPS


