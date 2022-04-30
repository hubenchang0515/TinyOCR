#! /usr/bin/bash
PROJECT_DIR="$(pwd)/$(dirname $0)/.."
TEMP_DIR="_temp1"
ARCH="$(arch)"
INSTALL_DIR="$PROJECT_DIR/usr/"

rm -rf $TEMP_DIR && mkdir -p $TEMP_DIR && 
cd $TEMP_DIR
git clone git@github.com:Tencent/ncnn.git
cd ncnn 
# git submodule update --init
mkdir -p build && cd build
cmake -DCMAKE_INSTALL_PREFIX=$INSTALL_DIR \
      -DCMAKE_BUILD_TYPE=Release \
      -DNCNN_VULKAN=OFF \
      -DNCNN_AVX512=OFF \
      -DNCNN_BUILD_BENCHMARK=OFF \
      -DNCNN_BUILD_EXAMPLES=OFF \
      .. 

make -j4 && make install