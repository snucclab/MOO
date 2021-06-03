#!/bin/bash

echo ------------------------------------------------------------------
echo CPU information
lscpu
echo
echo
echo ------------------------------------------------------------------
echo Memory information
lsmem
echo
echo
echo ------------------------------------------------------------------
echo Linux distribution information
lsb_release -ds
uname -a
echo
echo
echo ------------------------------------------------------------------
echo Graphic device information
lspci | grep NVIDIA
echo
/usr/local/cuda/bin/nvcc --version
echo
nvidia-smi
echo
echo
echo ------------------------------------------------------------------
echo Python information
python --version
echo "import torch; print('Torch ', torch.__version__, ' with CUDA ', torch.version.cuda);" | python -
pip list