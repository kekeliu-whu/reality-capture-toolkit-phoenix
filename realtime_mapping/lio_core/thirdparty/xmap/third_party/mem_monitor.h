#pragma once

#include <stdint.h>
#include <string>

namespace xmap {
class Mem_monitor {
  // singleton
 private:
  std::string m_proc_name;
  uint64_t m_pid;

  Mem_monitor() {}
  Mem_monitor(const Mem_monitor&) = delete;
  Mem_monitor& operator=(const Mem_monitor&) = delete;

 public:
  static Mem_monitor& get_instance() {
    static Mem_monitor instance;
    return instance;
  }

  void set_proc_name(std::string name);
  uint64_t get_proc_mem_usage() const;
  uint64_t get_sys_avail_mem() const;

  double get_proc_mem_usage_in_gb() const;
  double get_sys_avail_mem_in_gb() const;
};
}  // namespace xmap
