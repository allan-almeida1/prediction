<h1 align="center">
<img src="image.png"/><br/><br/>
Transfer Learning-based Lane Line Detection System for Visual Path Following Control
</h1>

<h4 align="center">
    <a href="#introduction">Introduction</a> |
    <a href="#dependencies">Dependencies</a> |
    <a href="#installation">Installation</a> |
    <a href="#nodes">Nodes</a> |
    <a href="#launch-files">Launch Files</a> |
    <a href="#gpu-support">GPU Support</a>
</h4>

## Introduction

This repository contains the code for the ROS package `prediction` which is used to detect a path from a camera image.

## Dependencies

[![ROS](https://img.shields.io/badge/ROS-Noetic-blue)](http://wiki.ros.org/noetic/Installation)
[![TensorFlow](https://img.shields.io/badge/TensorFlow-2.13.1-orange)](https://www.tensorflow.org/)
[![OpenCV](https://img.shields.io/badge/OpenCV-4.8.1-green)](https://opencv.org/)
[![NumPy](https://img.shields.io/badge/NumPy-1.23.5-yellowgreen)](https://numpy.org/)
[![MLPack](https://img.shields.io/badge/MLPack-4.3.0-darkblue)](https://www.mlpack.org/)
[![Eigen](https://img.shields.io/badge/Eigen-3.3.9-blue)](http://eigen.tuxfamily.org/index.php?title=Main_Page)

The package depends on the following libraries:

- [ROS](http://wiki.ros.org/ROS/Installation)
- [TensorFlow](https://www.tensorflow.org/install)
- [OpenCV](https://opencv.org/)
- [NumPy](https://numpy.org/)
- [MLPack](https://www.mlpack.org/)
- [Eigen](http://eigen.tuxfamily.org/index.php?title=Main_Page)

TensorFlow, OpenCV (python) and NumPy will be installed automatically when you run the script to create the conda environment. The other dependencies need to be installed manually.

To install Eigen, run the following command:

```bash
sudo apt install libeigen3-dev
```

To install MLPack, run the following command:

```bash
sudo apt install libmlpack-dev
```

## Installation

To install the package, clone the repository into your catkin workspace and build it:

```bash
cd ~/catkin_ws/src
git clone
cd ..
catkin_make # or catkin build
```

## Nodes

The package contains the following nodes:

- **prediction.py**: This node is used to detect a path from a camera image. It subscribes to the topic defined by param `/prediction_node/topic_name` and runs semantic segmentation on the image using an adapted version of the ERFNet model. The node then publishes the binary image to the topic `/image_raw_bin`.

- **processing**: This node is used to process the binary image published by the `prediction` node. It subscribes to the topic `/image_raw_bin` and runs processing steps like normalization, clustering and polynomial fitting on the image.

## Launch Files

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