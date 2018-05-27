#!/bin/bash
# Notes:
# Env vars: https://conda.io/docs/user-guide/tasks/build-packages/environment-variables.html

# Install dependencies
# conda install --file environment.yml -y -n $CONDA_DEFAULT_ENV

# Install
$CONDA_PYTHON_EXE $SRC_DIR/setup.py install
$CONDA_PYTHON_EXE $SRC_DIR/setup.py bdist bdist_wheel

