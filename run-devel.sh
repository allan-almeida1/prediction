#!/bin/bash
docker run -it --rm -v ~/catkin_ws:/home/ros/catkin_ws \
    --env DISPLAY=$DISPLAY --volume /tmp/.X11-unix:/tmp/.X11-unix:rw --name ros-container \
    --network host ros-noetic-container
