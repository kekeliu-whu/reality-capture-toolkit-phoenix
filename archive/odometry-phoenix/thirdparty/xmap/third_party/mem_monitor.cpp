#include "mem_monitor.h"

#include <linux/kernel.h>
#include <linux/unistd.h>
#include <memory.h>
#include <unistd.h>

using namespace xmap;

static int get_pid(const char* process_name, const char* user = nullptr) {
  if (user == nullptr) {
    user = getlogin();
  }

  char cmd[512];
  if (user) {
    sprintf(cmd, "pgrep %s -u %s", process_name, user);
  }

  FILE* pstr = popen(cmd, "r");

  if (pstr == nullptr) {
    return 0;
  }

  char buff[512];
  ::memset(buff, 0, sizeof(buff));
  if (NULL == fgets(buff, 512, pstr)) {
    return 0;
  }

  return atoi(buff);
}

static std::string _read_stat_file(const char* file_name) {
  std::string stat_info;
  FILE* fd;
  if ((fd = fopen(file_name, "r"))) {
    stat_info.resize(1024);
    fgets(const_cast<char*>(stat_info.c_str()), 1024, fd);
    stat_info.resize(strlen(stat_info.c_str()));
    fclose(fd);
  }
  return stat_info;
}

static uint64_t get_proc_mem(unsigned int pid) {
  std::string pid_meminfo = _read_stat_file("/proc/self/statm");
  uint64_t virt, res;
  sscanf(pid_meminfo.c_str(), "%lu%lu", &virt, &res);
  return res * (getpagesize());  /// / 1024
}

void Mem_monitor::set_proc_name(std::string name) {
  m_proc_name = name;
  m_pid = get_pid(m_proc_name.c_str());
}

uint64_t Mem_monitor::get_proc_mem_usage() const { return get_proc_mem(m_pid); }

uint64_t Mem_monitor::get_sys_avail_mem() const {
  uint64_t total, avail;
  FILE* fd;
  if ((fd = fopen("/proc/meminfo", "r"))) {
    char buff[512];
    ::memset(buff, 0, sizeof(buff));
    while (NULL != fgets(buff, 512, fd)) {
      if (strncmp(buff, "MemTotal:", 9) == 0) {
        sscanf(buff, "MemTotal: %lu kB", &total);
      }
      if (strncmp(buff, "MemAvailable:", 13) == 0) {
        sscanf(buff, "MemAvailable: %lu kB", &avail);
      }
      ::memset(buff, 0, sizeof(buff));
    }
    fclose(fd);
  }
  return avail * 1024;
}

double Mem_monitor::get_proc_mem_usage_in_gb() const {
  return get_proc_mem_usage() / 1024.0 / 1024.0 / 1024.0;
}

double Mem_monitor::get_sys_avail_mem_in_gb() const {
  return get_sys_avail_mem() / 1024.0 / 1024.0 / 1024.0;
}
