#include "xmap_util.h"

namespace xmap {

/**
 * 递归创建文件夹
 * @param path 路径
 */
void createFolders(const std::string& path) {
  std::string currentPath = "";
  for (const auto& folder : std::filesystem::path(path)) {
    currentPath += folder.string();
    if (!std::filesystem::exists(currentPath)) {
      if (std::filesystem::create_directory(currentPath)) {
        std::cout << "Created folder: " << currentPath << std::endl;
      } else {
        std::cout << "Failed to create folder: " << currentPath << std::endl;
        return;
      }
    }
    currentPath += "/";
  }
}

/**
 * 删除文件夹内的所有文件
 * @param folderPath 路径
 */
void deleteFolderContents(const std::string& folderPath) {
  for (const auto& entry : std::filesystem::directory_iterator(folderPath)) {
    if (entry.is_directory()) {
      deleteFolderContents(entry.path().string());
      std::filesystem::remove(entry.path());
    } else {
      std::filesystem::remove(entry.path());
    }
  }
}

#ifdef __linux__
std::string strprintf(const char* fmt, ...) {
  constexpr size_t BUFF_SZ = 10240;
  static __thread char buf[BUFF_SZ];
  va_list ap;
  va_start(ap, fmt);

  vsnprintf(&buf[0], BUFF_SZ, fmt, ap);
  std::string ret = std::string(buf);
  va_end(ap);

  return ret;
}
#endif

}  // namespace xmap