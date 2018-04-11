#!/bin/bash

# Install dependencies
# conda install --file environment.yml -y -n $CONDA_DEFAULT_ENV

# Install 
$PYTHON setup.py install
$PYTHON setup.py bdist_wheel sdist

