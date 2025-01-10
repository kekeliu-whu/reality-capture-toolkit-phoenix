
#ifdef __linux__
#include <fstream>
#include <malloc.h>
#include <sys/resource.h>
#else
// clang-format off
#include <windows.h>
#include <psapi.h>
// clang-format on
#endif

#include <glog/logging.h>

#include "migration/utils.h"

void PrintMemoryUsage() {
#ifdef __linux__
  std::ifstream file("/proc/self/status");
  std::string line;
  while (std::getline(file, line)) {
    // VmRSS is Resident Set Size, the physical memory used by the process
    if (line.find("VmRSS:") != std::string::npos) {
      LOG(INFO) << line;
      break;
    }
  }
#else
  PROCESS_MEMORY_COUNTERS pmc;
  if (GetProcessMemoryInfo(GetCurrentProcess(), &pmc, sizeof(pmc))) {
    LOG(INFO) << "Memory usage: " << pmc.WorkingSetSize / 1024 << " KB";
  }
#endif
}

void MallocTrim() {
#ifdef __linux__
  malloc_trim(0);
#endif
}
