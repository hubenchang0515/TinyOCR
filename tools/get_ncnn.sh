#! /usr/bin/bash
PROJECT_DIR="$(pwd)/$(dirname $0)/.."
TEMP_DIR="_temp1"
ARCH="$(arch)"

SEARCH_GLSLANG_TARGET_DIRS=(
      "/usr/lib/cmake"
      "/usr/lib/cmake/glslang"
      "/usr/lib/$ARCH-linux-gnu/cmake"
      "/usr/lib/$ARCH-linux-gnu/cmake/glslang"
)

INSTALL_DIR="$PROJECT_DIR/usr/$ARCH-cpu/"
NCNN_SYSTEM_GLSLANG=OFF
for DIR in ${SEARCH_GLSLANG_TARGET_DIRS[@]}
do
      if [ -f $DIR/glslangTargets.cmake ]
      then
            INSTALL_DIR="$PROJECT_DIR/usr/$ARCH-gpu/"
            NCNN_SYSTEM_GLSLANG=ON
            GLSLANG_TARGET_DIR=$DIR
            break
      fi
done

rm -rf $TEMP_DIR && mkdir -p $TEMP_DIR && cd $TEMP_DIR
git clone git@github.com:Tencent/ncnn.git
cd ncnn 
git submodule update --init
mkdir -p build && cd build
cmake -DCMAKE_INSTALL_PREFIX=$INSTALL_DIR \
      -DCMAKE_BUILD_TYPE=Release \
      -DGLSLANG_TARGET_DIR=$GLSLANG_TARGET_DIR \
      -DNCNN_VULKAN=ON \
      -DNCNN_SYSTEM_GLSLANG=$NCNN_SYSTEM_GLSLANG \
      -DNCNN_AVX512=OFF \
      -DNCNN_BUILD_BENCHMARK=OFF \
      -DNCNN_BUILD_EXAMPLES=OFF \
      .. 

make && make install