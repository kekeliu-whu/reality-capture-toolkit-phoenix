// Read-only probe for the installed configuration checker.
//
// The NavVis checker resolves and decrypts all included Lua files before it
// calls LuaParameterDictionary::toString().  Interposing that single method
// lets the clean-room regression observe the resolved parameter tree without
// modifying either the installed files or the vendor process.

#include <dlfcn.h>
#include <fcntl.h>
#include <unistd.h>

#include <cstdlib>
#include <string>

namespace navvis {

class LuaParameterDictionary {
 public:
  std::string toString() const;
};

std::string LuaParameterDictionary::toString() const {
  using ToString = std::string (*)(const LuaParameterDictionary*);
  static const auto original = reinterpret_cast<ToString>(
      dlsym(RTLD_NEXT,
            "_ZNK6navvis22LuaParameterDictionary8toStringB5cxx11Ev"));
  if (original == nullptr) {
    _exit(127);
  }
  std::string resolved = original(this);
  const char* output_path = std::getenv("NAVVIS_CONFIG_CAPTURE");
  if (output_path != nullptr && output_path[0] != '\0') {
    const int fd = open(output_path, O_WRONLY | O_CREAT | O_TRUNC, 0600);
    if (fd >= 0) {
      const char* cursor = resolved.data();
      std::size_t remaining = resolved.size();
      while (remaining > 0) {
        const ssize_t written = write(fd, cursor, remaining);
        if (written <= 0) {
          break;
        }
        cursor += written;
        remaining -= static_cast<std::size_t>(written);
      }
      close(fd);
    }
  }
  return resolved;
}

}  // namespace navvis
