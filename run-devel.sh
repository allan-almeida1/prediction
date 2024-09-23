#!/bin/bash
docker run -it --rm --gpus all -v ~/catkin_ws:/home/ros/catkin_ws \
    --env DISPLAY --volume /tmp/.X11-unix:/tmp/.X11-unix:rw --name ros-container \
    --network host ros-noetic-container
