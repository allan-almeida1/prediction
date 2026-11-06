#!/bin/bash
docker run -it --rm --gpus all -v ros-cuda-cache:/home/ros/.nv -v "$(cd "$(dirname "$0")" && pwd)":/home/ros/catkin_ws/src/prediction \
    --env DISPLAY=$DISPLAY --env XAUTHORITY=/tmp/.Xauthority --volume "${XAUTHORITY:-$HOME/.Xauthority}":/tmp/.Xauthority:ro --volume /tmp/.X11-unix:/tmp/.X11-unix:rw --name ros-container \
    --network host ros-noetic-container
