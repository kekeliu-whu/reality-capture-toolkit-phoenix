
#ifdef __linux__
#include <malloc.h>
#include <sys/resource.h>
#include <fstream>
#else
// clang-format off
#include <windows.h>
#include <psapi.h>
// clang-format on
#endif

#include <spdlog/spdlog.h>

#include "utils.h"

namespace xcolor {

void PrintMemoryUsage() {
#ifdef __linux__
  std::ifstream file("/proc/self/status");
  std::string line;
  while (std::getline(file, line)) {
    // VmRSS is Resident Set Size, the physical memory used by the process
    if (line.find("VmRSS:") != std::string::npos) {
      spdlog::info(line);
      break;
    }
  }
#else
  PROCESS_MEMORY_COUNTERS pmc;
  if (GetProcessMemoryInfo(GetCurrentProcess(), &pmc, sizeof(pmc))) {
    spdlog::info("Memory usage: {} KB", pmc.WorkingSetSize / 1024);
  }
#endif
}

void MallocTrim() {
#ifdef __linux__
  malloc_trim(0);
#endif
}

}  // namespace xcolor
