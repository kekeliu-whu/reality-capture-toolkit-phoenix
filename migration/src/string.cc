#include <Windows.h>
#include <string>

#include "migration/string.h"

namespace {

#ifdef _WIN32

std::string CodePageToUTF8Win(const std::string& str, unsigned int code_page) {
  int wide_len = MultiByteToWideChar(code_page, 0, str.c_str(), -1, nullptr, 0);
  if (wide_len <= 0) return "";

  std::wstring wide_str(wide_len, L'\0');
  MultiByteToWideChar(code_page, 0, str.c_str(), -1, &wide_str[0], wide_len);

  int utf8_len = WideCharToMultiByte(CP_UTF8, 0, wide_str.c_str(), -1, nullptr, 0, nullptr, nullptr);
  if (utf8_len <= 0) return "";

  std::string utf8_str(utf8_len, '\0');
  WideCharToMultiByte(CP_UTF8, 0, wide_str.c_str(), -1, &utf8_str[0], utf8_len, nullptr, nullptr);

  if (!utf8_str.empty() && utf8_str.back() == '\0') {
    utf8_str.pop_back();
  }

  return utf8_str;
};

std::string UTF8ToCodePageWin(const std::string& str, unsigned int code_page) {
  int wide_len = MultiByteToWideChar(CP_UTF8, 0, str.c_str(), -1, nullptr, 0);
  if (wide_len <= 0) return "";

  std::wstring wide_str(wide_len, L'\0');
  MultiByteToWideChar(CP_UTF8, 0, str.c_str(), -1, &wide_str[0], wide_len);

  int local_len = WideCharToMultiByte(code_page, 0, wide_str.c_str(), -1, nullptr, 0, nullptr, nullptr);
  if (local_len <= 0) return "";

  std::string local_str(local_len, '\0');
  WideCharToMultiByte(code_page, 0, wide_str.c_str(), -1, &local_str[0], local_len, nullptr, nullptr);

  if (!local_str.empty() && local_str.back() == '\0') {
    local_str.pop_back();
  }

  return local_str;
}

#endif

}  // namespace

std::string PlatformToUTF8(const std::string& str) {
#ifdef _WIN32
  return CodePageToUTF8Win(str, GetACP());
#else
  // Assume UTF-8 on POSIX systems.
  return str;
#endif
}

std::string UTF8ToPlatform(const std::string& str) {
#ifdef _WIN32
  return UTF8ToCodePageWin(str, GetACP());
#else
  // On POSIX, assume UTF-8 is the system encoding.
  return str;
#endif
}