// Offline acceptance probe: persist the decrypted Lua chunks submitted by the
// installed configuration loader. This is not part of the clean-room runtime.
#include <dlfcn.h>

#include <cstdio>
#include <cstdlib>
#include <mutex>

struct lua_State;

namespace {

std::mutex output_mutex;

void Dump(const char* kind, const char* name, const char* buffer, size_t size) {
  std::lock_guard<std::mutex> lock(output_mutex);
  const char* path = std::getenv("NAVVIS_LUA_BUFFER_LOG");
  if (path == nullptr) return;
  FILE* output = std::fopen(path, "ab");
  if (output == nullptr) return;
  std::fprintf(output, "\n-- BEGIN %s %s (%zu bytes)\n", kind,
               name == nullptr ? "<unnamed>" : name, size);
  std::fwrite(buffer, 1, size, output);
  std::fprintf(output, "\n-- END %s %s\n", kind,
               name == nullptr ? "<unnamed>" : name);
  std::fclose(output);
}

template <typename Function>
Function Next(const char* symbol) {
  return reinterpret_cast<Function>(dlsym(RTLD_NEXT, symbol));
}

}  // namespace

extern "C" int luaL_loadbufferx(lua_State* state, const char* buffer,
                                size_t size, const char* name,
                                const char* mode) {
  using Function = int (*)(lua_State*, const char*, size_t, const char*,
                           const char*);
  static const Function next = Next<Function>("luaL_loadbufferx");
  Dump("luaL_loadbufferx", name, buffer, size);
  return next(state, buffer, size, name, mode);
}

extern "C" int luaL_loadstring(lua_State* state, const char* source) {
  using Function = int (*)(lua_State*, const char*);
  static const Function next = Next<Function>("luaL_loadstring");
  const size_t size = source == nullptr ? 0 : __builtin_strlen(source);
  Dump("luaL_loadstring", "<string>", source, size);
  return next(state, source);
}
