#!/usr/bin/env bash
set -ex

# Environment variables needed during the build
export CMAKE_LIBRARY_PATH=$PREFIX/lib:$PREFIX/include:$CMAKE_LIBRARY_PATH
export CMAKE_PREFIX_PATH=$PREFIX

cmake_args=()
cmake_args+=("-DCMAKE_INSTALL_PREFIX=$PREFIX")

# Build pybot/src
# mkdir -p cmake_build && pushd cmake_build
# cmake "${cmake_args[@]}" $CMAKE_ARGS ..
# make install
# popd

# Build pybot
if [[ "$OSTYPE" == "darwin"* ]]; then
  MACOSX_DEPLOYMENT_TARGET=10.9 python setup.py install
  exit 0
fi
pushd $RECIPE_DIR/../../
python setup.py install
popd

