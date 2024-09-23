#!/bin/bash
eval "$(conda shell.bash hook)"
conda activate $1 || { echo "Failed to activate conda environment $1"; exit 1; }
conda install -c conda-forge cudatoolkit=11.8.0 -y
python3 -m pip install nvidia-cudnn-cu11==8.6.0.163 tensorflow==2.16.* opencv-python matplotlib rospkg PyYAML
mkdir -p $CONDA_PREFIX/etc/conda/activate.d
echo 'CUDNN_PATH=$(dirname $(python -c "import nvidia.cudnn;print(nvidia.cudnn.__file__)"))' >> $CONDA_PREFIX/etc/conda/activate.d/env_vars.sh
echo 'export LD_LIBRARY_PATH=$CUDNN_PATH/lib:$CONDA_PREFIX/lib/:$LD_LIBRARY_PATH' >> $CONDA_PREFIX/etc/conda/activate.d/env_vars.sh
source $CONDA_PREFIX/etc/conda/activate.d/env_vars.sh
