#define _GNU_SOURCE

#include <cstdarg>
#include <cstdio>
#include <cstring>
#include <dlfcn.h>

namespace {

using VfprintfFunction = int (*)(FILE*, const char*, va_list);
using FprintfFunction = int (*)(FILE*, const char*, ...);
using CheckedVfprintfFunction = int (*)(FILE*, int, const char*, va_list);
using CheckedFprintfFunction = int (*)(FILE*, int, const char*, ...);

VfprintfFunction realVfprintf() {
  static const auto function = reinterpret_cast<VfprintfFunction>(
      dlsym(RTLD_NEXT, "vfprintf"));
  return function;
}

FprintfFunction realFprintf() {
  static const auto function = reinterpret_cast<FprintfFunction>(
      dlsym(RTLD_NEXT, "fprintf"));
  return function;
}

CheckedVfprintfFunction realCheckedVfprintf() {
  static const auto function = reinterpret_cast<CheckedVfprintfFunction>(
      dlsym(RTLD_NEXT, "__vfprintf_chk"));
  return function;
}

CheckedFprintfFunction realCheckedFprintf() {
  static const auto function = reinterpret_cast<CheckedFprintfFunction>(
      dlsym(RTLD_NEXT, "__fprintf_chk"));
  return function;
}

}  // namespace

extern "C" int fprintf(FILE* stream, const char* format, ...) {
  va_list arguments;
  va_start(arguments, format);

  int result = 0;
  if (std::strcmp(format, "%17f\n") == 0) {
    const double value = va_arg(arguments, double);
    result = realFprintf()(stream, "%a\n", value);
  } else if (std::strcmp(format, "% 10d % 10d %17f\n") == 0) {
    const int row = va_arg(arguments, int);
    const int column = va_arg(arguments, int);
    const double value = va_arg(arguments, double);
    result = realFprintf()(stream, "%d %d %a\n", row, column, value);
  } else {
    result = realVfprintf()(stream, format, arguments);
  }

  va_end(arguments);
  return result;
}

extern "C" int __fprintf_chk(FILE* stream, int flag, const char* format, ...) {
  va_list arguments;
  va_start(arguments, format);

  int result = 0;
  if (std::strcmp(format, "%17f\n") == 0) {
    const double value = va_arg(arguments, double);
    result = realCheckedFprintf()(stream, flag, "%a\n", value);
  } else if (std::strcmp(format, "% 10d % 10d %17f\n") == 0) {
    const int row = va_arg(arguments, int);
    const int column = va_arg(arguments, int);
    const double value = va_arg(arguments, double);
    result = realCheckedFprintf()(stream, flag, "%d %d %a\n", row, column, value);
  } else {
    result = realCheckedVfprintf()(stream, flag, format, arguments);
  }

  va_end(arguments);
  return result;
}
