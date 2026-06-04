#ifndef PLATFORM_COMPAT_HPP
#define PLATFORM_COMPAT_HPP

// ============================================================
// Windows/POSIX compatibility layer for VoxelSLAM
//
// Replaces:
//   malloc_trim() -> _heapmin() on MSVC, no-op otherwise
//   sleep(s)      -> Sleep(ms) on Windows
//   access()      -> _access() on Windows
//   unistd.h      -> Windows equivalents
//   /proc/self/status -> GetProcessMemoryInfo on Windows
// ============================================================

#include <thread>
#include <chrono>

// ---- sleep (seconds) — portable C++11, works on both Windows and Linux ----
inline void platform_sleep(double seconds) {
  std::this_thread::sleep_for(std::chrono::duration<double>(seconds));
}

#ifdef _WIN32

#include <windows.h>
#include <psapi.h>
#include <io.h>        // _access
#include <direct.h>    // _mkdir
#include <string>
#include <cerrno>

#pragma comment(lib, "psapi.lib")

// ---- malloc_trim ----
inline int platform_malloc_trim(size_t /*pad*/) {
  _heapmin();
  return 1;
}

// ---- access(path, F_OK) -> check existence ----
inline bool platform_dir_exists(const char* path) {
  return _access(path, 0) == 0;  // 0 = F_OK (existence only)
}

// ---- mkdir (recursive, creates intermediate directories) ----
inline int platform_mkdir(const char* path) {
  // Use Win32 CreateDirectoryA which handles intermediate dirs via
  // SHCreateDirectoryEx or manual recursion
  std::string p(path);
  // Normalize to backslashes for Win32 API
  for (char &c : p) if (c == '/') c = '\\';
  // Remove trailing backslash
  while (!p.empty() && p.back() == '\\') p.pop_back();

  // Try creating the full path
  if (CreateDirectoryA(p.c_str(), NULL)) return 0;
  DWORD err = GetLastError();
  if (err == ERROR_ALREADY_EXISTS) return 0;

  // If parent missing, recurse up
  if (err == ERROR_PATH_NOT_FOUND) {
    size_t pos = p.find_last_of('\\');
    if (pos != std::string::npos && pos > 0) {
      std::string parent = p.substr(0, pos);
      if (platform_mkdir(parent.c_str()) != 0) return -1;
      if (CreateDirectoryA(p.c_str(), NULL)) return 0;
      if (GetLastError() == ERROR_ALREADY_EXISTS) return 0;
    }
  }
  return -1;
}

// ---- get_memory (replaces /proc/self/status) ----
inline double platform_get_memory_gb() {
  PROCESS_MEMORY_COUNTERS_EX pmc;
  pmc.cb = sizeof(PROCESS_MEMORY_COUNTERS_EX);
  if (GetProcessMemoryInfo(GetCurrentProcess(),
                           reinterpret_cast<PROCESS_MEMORY_COUNTERS*>(&pmc),
                           sizeof(pmc))) {
    return pmc.WorkingSetSize / (1024.0 * 1024.0 * 1024.0);
  }
  return 0.0;
}

#else  // Linux / POSIX

#include <unistd.h>
#include <sys/stat.h>
#include <malloc.h>
#include <fstream>
#include <sstream>
#include <string>

inline int platform_malloc_trim(size_t pad) {
  return malloc_trim(pad);
}

inline bool platform_dir_exists(const char* path) {
  return access(path, F_OK) == 0;
}

inline int platform_mkdir(const char* path) {
  return mkdir(path, S_IRWXU | S_IRGRP | S_IXGRP | S_IROTH | S_IXOTH);
}

// Reads /proc/self/status for VmRSS, returns value in GB
inline double platform_get_memory_gb() {
  std::ifstream infile("/proc/self/status");
  double mem = 0.0;
  std::string line;
  while (std::getline(infile, line)) {
    std::stringstream ss(line);
    std::string key;
    ss >> key;
    if (key == "VmRSS:") {
      std::string val;
      ss >> val;
      mem = std::stod(val);
      break;
    }
  }
  return mem / (1024 * 1024);  // kB -> GB  (1048576)
}

#endif  // _WIN32

#endif  // PLATFORM_COMPAT_HPP
