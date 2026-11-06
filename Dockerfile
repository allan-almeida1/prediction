FROM osrf/ros:noetic-desktop-full

# Create a new user
ARG USER_NAME=ros
ARG USER_ID
ARG GROUP_ID
RUN groupadd -g $GROUP_ID $USER_NAME && \
    useradd -l -m -u $USER_ID -g $GROUP_ID $USER_NAME && \
    chown -R $USER_NAME:$USER_NAME /home/$USER_NAME

# Create a ROS workspace
RUN mkdir -p /home/ros/catkin_ws/src && \
    chown -R ros:ros /home/ros/catkin_ws
USER ros
WORKDIR /home/ros/catkin_ws
RUN /bin/bash -c '. /opt/ros/noetic/setup.bash; catkin_make'

# Copy the source code
COPY --chown=ros:ros . /home/ros/catkin_ws/src/prediction

# Install latest version of cmake
WORKDIR /home
USER root
RUN apt-get update && \
    apt-get install -y wget --no-install-recommends && \
    apt-get clean && \
    rm -rf /var/lib/apt/lists/* && \
    wget -q https://github.com/Kitware/CMake/releases/download/v3.30.3/cmake-3.30.3-linux-x86_64.sh && \
    mkdir -p /usr/local/cmake-3.30.3 && \
    chmod -R 777 /usr/local/cmake-3.30.3 && \
    chmod +x cmake-3.30.3-linux-x86_64.sh && \
    ./cmake-3.30.3-linux-x86_64.sh --prefix=/usr/local/cmake-3.30.3 --skip-license

# Update the PATH to prioritize the new CMake version
ENV PATH="/usr/local/cmake-3.30.3/bin:$PATH"

# Install and configure the Python environment
USER root
RUN apt-get update && \
    apt-get install -y python3-pip --no-install-recommends && \
    apt-get clean && \
    rm -rf /var/lib/apt/lists/* && \
    pip3 install --upgrade --no-cache-dir pip==24.2 && \
    pip3 install --upgrade --no-cache-dir setuptools==75.1.0

# Create the python environment and install the required packages
WORKDIR /home/ros/
RUN apt-get update && \
    apt-get install python3.8-venv --no-install-recommends -y && \
    apt-get clean && \
    rm -rf /var/lib/apt/lists/* && \
    python3 -m venv tf_env && \
    /home/ros/tf_env/bin/pip install --upgrade --no-cache-dir pip==24.2 && \
    /home/ros/tf_env/bin/pip install --no-cache-dir tensorflow==2.13.* opencv-python==4.9.* matplotlib==3.1.2 rospkg==1.5.1 PyYAML==5.3.1 && \
    /home/ros/tf_env/bin/pip install --no-cache-dir \
        nvidia-cuda-runtime-cu11==11.8.89 nvidia-cublas-cu11==11.11.3.6 nvidia-cufft-cu11==10.9.0.58 \
        nvidia-curand-cu11==10.3.0.86 nvidia-cusolver-cu11==11.4.1.48 nvidia-cusparse-cu11==11.7.5.86 \
        nvidia-cuda-cupti-cu11==11.8.87 nvidia-cudnn-cu11==8.6.0.163

# CUDA 11.8 + cuDNN 8.6 runtime libraries for TensorFlow 2.13 GPU support (pip wheels, no system CUDA needed)
ENV NV_LIBS=/home/ros/tf_env/lib/python3.8/site-packages/nvidia
ENV LD_LIBRARY_PATH="${NV_LIBS}/cuda_runtime/lib:${NV_LIBS}/cublas/lib:${NV_LIBS}/cufft/lib:${NV_LIBS}/curand/lib:${NV_LIBS}/cusolver/lib:${NV_LIBS}/cusparse/lib:${NV_LIBS}/cuda_cupti/lib:${NV_LIBS}/cudnn/lib:${LD_LIBRARY_PATH}"
# Newer GPUs (e.g. RTX 50-series) need TF's kernels JIT-compiled from PTX; cache them (persisted via a volume in run-devel-gpu.sh)
ENV CUDA_CACHE_MAXSIZE=4294967296
RUN mkdir -p /home/ros/.nv && chown -R ros:ros /home/ros/.nv

# Build the workspace
USER ros
WORKDIR /home/ros/catkin_ws
RUN rosdep update && \
    rosdep install --from-paths src --ignore-src -r -y
RUN /bin/bash -c '. /opt/ros/noetic/setup.bash; catkin_make'
