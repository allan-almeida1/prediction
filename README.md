# Transfer Learning-based Lane Line Detection System for Visual Path Following Control

[![License: MIT](https://img.shields.io/badge/License-MIT-yellow.svg)](https://opensource.org/licenses/MIT)

## Introduction

This repository contains the code for the ROS package `prediction` which is used to detect a path from a camera image.

## Dependencies

[![ROS](https://img.shields.io/badge/ROS-Noetic-blue)](http://wiki.ros.org/noetic/Installation)
[![TensorFlow](https://img.shields.io/badge/TensorFlow-2.13.1-orange)](https://www.tensorflow.org/)
[![OpenCV](https://img.shields.io/badge/OpenCV-4.8.1-green)](https://opencv.org/)
[![NumPy](https://img.shields.io/badge/NumPy-1.23.5-yellowgreen)](https://numpy.org/)

The package depends on the following libraries:

- [ROS](http://wiki.ros.org/ROS/Installation)
- [TensorFlow](https://www.tensorflow.org/install)
- [OpenCV](https://opencv.org/)
- [NumPy](https://numpy.org/)

## Installation

To install the package, clone the repository into your catkin workspace and build it:

```bash
cd ~/catkin_ws/src
git clone
cd ..
catkin_make # or catkin build
```

## Usage

To run the package on a video file, use the following command:

```bash
roslaunch prediction video.launch
```

To run the package on a ROS topic in the simulation environment, use the following command:

```bash
roslaunch prediction unity.launch
```

## GPU Support

To enable GPU support, you need to install the CUDA Toolkit and cuDNN. Then, you need to install the GPU version of TensorFlow. A script was created to automate this process and create a virtual conda environment with everything set up. To use it, run the following command:

```bash
./scripts/create_env [env_name]
```

where `[env_name]` is the name of the environment you want to create. The script will create a new conda environment with the name `[env_name]` and install all the required packages. To activate the environment, run the following command:

To run with GPU support, you need to activate the conda environment first:

```bash
conda activate [env_name]
```

Then, you can run the package as usual.

**Note:** The script is set up to install CUDA Toolkit 11.8, cuDNN 8.6.0.163 and Tensorflow 2.13. If you want to use different versions, you need to edit the script accordingly.