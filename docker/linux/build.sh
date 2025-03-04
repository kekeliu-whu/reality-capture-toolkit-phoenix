set -ex

CODE_WS=/buildspace/reality-capture-toolkit
CODE_BIN=/buildspace/bin


rm -rf ${CODE_BIN}
mkdir -p ${CODE_BIN}

cd ${CODE_WS}/pgo
rm -rf build && mkdir build && cd build
cmake .. -DCMAKE_BUILD_TYPE=Release -DCMAKE_TOOLCHAIN_FILE=/buildspace/vcpkg/scripts/buildsystems/vcpkg.cmake && make
cp process_point_cloud_s10 ${CODE_BIN}

cd ${CODE_WS}/xcolor
rm -rf build && mkdir build && cd build
cmake .. -DCMAKE_BUILD_TYPE=Release -DCMAKE_TOOLCHAIN_FILE=/buildspace/vcpkg/scripts/buildsystems/vcpkg.cmake && make
cp sfm ${CODE_BIN}

cp ${CODE_WS}/docker/linux/run.sh ${CODE_BIN}
