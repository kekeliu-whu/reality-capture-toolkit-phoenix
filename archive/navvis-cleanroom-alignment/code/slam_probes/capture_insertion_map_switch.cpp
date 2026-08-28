#include <dlfcn.h>

#include <cstdint>
#include <cstdio>
#include <cstdlib>
#include <mutex>

namespace {

using GompParallel = void (*)(void (*)(void*), void*, unsigned, unsigned);

std::mutex output_mutex;
std::uint64_t insertion_call = 0;

GompParallel ResolveGompParallel() {
  static auto function = reinterpret_cast<GompParallel>(
      dlsym(RTLD_NEXT, "GOMP_parallel"));
  return function;
}

void CaptureInsertionLauncher(void (*function)(void*), void* data,
                              unsigned num_threads, unsigned flags) {
  Dl_info information{};
  if (dladdr(reinterpret_cast<void*>(function), &information) == 0 ||
      information.dli_fbase == nullptr || data == nullptr) {
    return;
  }
  const auto image_offset =
      reinterpret_cast<std::uintptr_t>(function) -
      reinterpret_cast<std::uintptr_t>(information.dli_fbase);
  if (image_offset != 0x48ed50) {
    return;
  }

  // The launcher at image offset 0x48f200 places the ray-vector pointer at
  // closure +0 and the selected MultiResolutionSurfelGrid at closure +8.
  const auto* closure = static_cast<const std::uintptr_t*>(data);
  const auto* ray_vector =
      reinterpret_cast<const std::uintptr_t*>(closure[0]);
  const auto insertion_map = closure[1];
  if (ray_vector == nullptr) {
    return;
  }
  const auto* begin = reinterpret_cast<const float*>(ray_vector[0]);
  const auto* end = reinterpret_cast<const float*>(ray_vector[1]);
  const std::size_t count =
      (reinterpret_cast<const char*>(end) -
       reinterpret_cast<const char*>(begin)) /
      (6 * sizeof(float));

  const char* output = std::getenv("NAVVIS_INSERTION_MAP_CAPTURE");
  if (output == nullptr || *output == '\0') {
    output = "/tmp/navvis_vendor_insertion_map_switch.csv";
  }
  std::lock_guard<std::mutex> lock(output_mutex);
  FILE* stream = std::fopen(output, insertion_call == 0 ? "w" : "a");
  if (stream == nullptr) {
    return;
  }
  if (insertion_call == 0) {
    std::fprintf(stream,
                 "call,map,ray_vector,count,first_endpoint_x,"
                 "first_endpoint_y,first_endpoint_z,num_threads,flags\n");
  }
  const float x = count == 0 ? 0.0F : begin[3];
  const float y = count == 0 ? 0.0F : begin[4];
  const float z = count == 0 ? 0.0F : begin[5];
  std::fprintf(stream, "%llu,0x%llx,0x%llx,%zu,%.9g,%.9g,%.9g,%u,%u\n",
               static_cast<unsigned long long>(insertion_call),
               static_cast<unsigned long long>(insertion_map),
               static_cast<unsigned long long>(closure[0]), count, x, y, z,
               num_threads, flags);
  std::fclose(stream);
  ++insertion_call;
}

}  // namespace

extern "C" void GOMP_parallel(void (*function)(void*), void* data,
                              unsigned num_threads, unsigned flags) {
  CaptureInsertionLauncher(function, data, num_threads, flags);
  ResolveGompParallel()(function, data, num_threads, flags);
}
