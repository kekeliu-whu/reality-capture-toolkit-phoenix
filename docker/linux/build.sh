set -ex

CODE_WS=/buildspace/reality-capture-toolkit
CODE_BIN=/buildspace/bin

# clean up existing directory
rm -rf ${CODE_BIN}
mkdir -p ${CODE_BIN}

# build pgo
cd ${CODE_WS}/pgo
rm -rf build && mkdir build && cd build
cmake .. -DCMAKE_BUILD_TYPE=Release -DCMAKE_TOOLCHAIN_FILE=/buildspace/vcpkg/scripts/buildsystems/vcpkg.cmake
if [ "$(nproc --all)" -lt 8 ]; then
  make -j1;
else
  make -j$(nproc);
fi

# build xcolor
cd ${CODE_WS}/xcolor
rm -rf build && mkdir build && cd build
cmake .. -DCMAKE_BUILD_TYPE=Release -DCMAKE_TOOLCHAIN_FILE=/buildspace/vcpkg/scripts/buildsystems/vcpkg.cmake
if [ "$(nproc --all)" -lt 8 ]; then
  make -j1;
else
  make -j$(nproc);
fi
strip xsfm xsfm_reset_s20_cameras process_point_cloud_s20
cp xsfm xsfm_reset_s20_cameras process_point_cloud_s20 ${CODE_BIN}

cp ${CODE_WS}/docker/linux/run.sh ${CODE_BIN}

wget https://github.com/colmap/colmap/releases/download/3.11.1/vocab_tree_flickr100K_words32K.bin -O ${CODE_BIN}/vocab_tree_flickr100K_words32K.bin
