// Offline acceptance probe: log scalar Lua values consumed by the installed
// reference binary. This shared object is never linked into the clean-room
// implementation.
#include <dlfcn.h>

#include <cstdio>
#include <cstdlib>
#include <mutex>
#include <string>
#include <type_traits>

namespace {

std::mutex log_mutex;

template <typename Value>
void Log(const char* type, const std::string& key, const Value& value) {
  std::lock_guard<std::mutex> lock(log_mutex);
  const char* path = std::getenv("NAVVIS_LUA_PARAMETER_LOG");
  FILE* output = path == nullptr ? stderr : std::fopen(path, "a");
  if (output == nullptr) return;
  if constexpr (std::is_same_v<Value, double>) {
    std::fprintf(output, "%s\t%s\t%.17g\n", type, key.c_str(), value);
  } else {
    std::fprintf(output, "%s\t%s\t%lld\n", type, key.c_str(),
                 static_cast<long long>(value));
  }
  std::fflush(output);
  if (output != stderr) std::fclose(output);
}

template <typename Function>
Function Next(const char* symbol) {
  dlerror();
  void* address = dlsym(RTLD_NEXT, symbol);
  const char* error = dlerror();
  if (error != nullptr || address == nullptr) {
    std::fprintf(stderr, "log_lua_parameters: dlsym(%s): %s\n", symbol,
                 error == nullptr ? "not found" : error);
    std::abort();
  }
  return reinterpret_cast<Function>(address);
}

}  // namespace

namespace navvis {

class LuaParameterDictionary {
 public:
  double getDouble(const std::string& key);
  bool getBool(const std::string& key);
  int getInt(const std::string& key);
  int getNonNegativeInt(const std::string& key);
  bool hasKey(const std::string& key) const;
};

double LuaParameterDictionary::getDouble(const std::string& key) {
  using Function = double (*)(LuaParameterDictionary*, const std::string&);
  static const Function next = Next<Function>(
      "_ZN6navvis22LuaParameterDictionary9getDoubleERKNSt7__cxx1112basic_stringIcSt11char_traitsIcESaIcEEE");
  const double value = next(this, key);
  Log("double", key, value);
  return value;
}

bool LuaParameterDictionary::getBool(const std::string& key) {
  using Function = bool (*)(LuaParameterDictionary*, const std::string&);
  static const Function next = Next<Function>(
      "_ZN6navvis22LuaParameterDictionary7getBoolERKNSt7__cxx1112basic_stringIcSt11char_traitsIcESaIcEEE");
  const bool value = next(this, key);
  Log("bool", key, value);
  return value;
}

int LuaParameterDictionary::getInt(const std::string& key) {
  using Function = int (*)(LuaParameterDictionary*, const std::string&);
  static const Function next = Next<Function>(
      "_ZN6navvis22LuaParameterDictionary6getIntERKNSt7__cxx1112basic_stringIcSt11char_traitsIcESaIcEEE");
  const int value = next(this, key);
  Log("int", key, value);
  return value;
}

int LuaParameterDictionary::getNonNegativeInt(const std::string& key) {
  using Function = int (*)(LuaParameterDictionary*, const std::string&);
  static const Function next = Next<Function>(
      "_ZN6navvis22LuaParameterDictionary17getNonNegativeIntERKNSt7__cxx1112basic_stringIcSt11char_traitsIcESaIcEEE");
  const int value = next(this, key);
  Log("non_negative_int", key, value);
  return value;
}

bool LuaParameterDictionary::hasKey(const std::string& key) const {
  using Function = bool (*)(const LuaParameterDictionary*, const std::string&);
  static const Function next = Next<Function>(
      "_ZNK6navvis22LuaParameterDictionary6hasKeyERKNSt7__cxx1112basic_stringIcSt11char_traitsIcESaIcEEE");
  const bool value = next(this, key);
  Log("has_key", key, value);
  return value;
}

}  // namespace navvis
