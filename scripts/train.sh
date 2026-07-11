#!/bin/bash

export LD_LIBRARY_PATH=$CONDA_PREFIX/lib:$LD_LIBRARY_PATH
echo "Current LD_LIBRARY_PATH: $LD_LIBRARY_PATH"

python ./src/train_aerovla.py "$@"
