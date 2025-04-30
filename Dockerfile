# Stage 1: Build the code (temporary layer, not included in the final image)
FROM registry.cn-hangzhou.aliyuncs.com/c137/reality-capture-toolkit:base-cuda-build-vcpkg-manual AS builder
COPY . /buildspace/reality-capture-toolkit
RUN bash /buildspace/reality-capture-toolkit/docker/linux/build.sh

# Stage 2: Create the final image
FROM registry.cn-hangzhou.aliyuncs.com/c137/reality-capture-toolkit:base-cuda-build-vcpkg-manual
COPY --from=builder /buildspace/bin /buildspace/bin
# Set LD_LIBRARY_PATH and run ldconfig
ENV LD_LIBRARY_PATH=/buildspace/vcpkg/installed/x64-linux/lib/manual-link/:${LD_LIBRARY_PATH}
RUN ldconfig

# setup entrypoint
CMD ["bash"]
