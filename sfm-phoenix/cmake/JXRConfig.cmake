include(FindPackageHandleStandardArgs)
include(SelectLibraryConfigurations)

set(_jxr_prefix "")
set(_jxr_triplet "x64-windows")
if(DEFINED VCPKG_TARGET_TRIPLET AND NOT VCPKG_TARGET_TRIPLET STREQUAL "")
  set(_jxr_triplet "${VCPKG_TARGET_TRIPLET}")
endif()

set(_jxr_candidate_prefixes "")
if(DEFINED VCPKG_INSTALLED_DIR AND NOT VCPKG_INSTALLED_DIR STREQUAL "")
  list(APPEND _jxr_candidate_prefixes "${VCPKG_INSTALLED_DIR}/${_jxr_triplet}")
endif()

if(DEFINED CMAKE_TOOLCHAIN_FILE AND NOT CMAKE_TOOLCHAIN_FILE STREQUAL "")
  get_filename_component(_jxr_toolchain_dir "${CMAKE_TOOLCHAIN_FILE}" DIRECTORY)
  get_filename_component(_jxr_vcpkg_root "${_jxr_toolchain_dir}/../.." ABSOLUTE)
  list(APPEND _jxr_candidate_prefixes "${_jxr_vcpkg_root}/installed/${_jxr_triplet}")
endif()

if(DEFINED ENV{VCPKG_ROOT})
  file(TO_CMAKE_PATH "$ENV{VCPKG_ROOT}" _jxr_env_vcpkg_root)
  list(APPEND _jxr_candidate_prefixes "${_jxr_env_vcpkg_root}/installed/${_jxr_triplet}")
endif()

list(APPEND _jxr_candidate_prefixes "D:/vcpkg/installed/${_jxr_triplet}")
list(REMOVE_DUPLICATES _jxr_candidate_prefixes)

foreach(_jxr_candidate IN LISTS _jxr_candidate_prefixes)
  if(EXISTS "${_jxr_candidate}/include/jxrlib/JXRGlue.h")
    set(_jxr_prefix "${_jxr_candidate}")
    break()
  endif()
endforeach()

if(_jxr_prefix)
  message(STATUS "JXR prefix: ${_jxr_prefix}")
endif()

if(_jxr_prefix)
  find_path(JXR_INCLUDE_DIRS
    NAMES JXRGlue.h
    PATHS "${_jxr_prefix}/include"
    PATH_SUFFIXES jxrlib
    NO_DEFAULT_PATH)

  find_library(JPEGXR_LIBRARY_RELEASE
    NAMES jpegxr
    PATHS "${_jxr_prefix}/lib"
    NO_DEFAULT_PATH)
  find_library(JPEGXR_LIBRARY_DEBUG
    NAMES jpegxrd
    PATHS "${_jxr_prefix}/debug/lib"
    NO_DEFAULT_PATH)
  select_library_configurations(JPEGXR)

  find_library(JXRGLUE_LIBRARY_RELEASE
    NAMES jxrglue
    PATHS "${_jxr_prefix}/lib"
    NO_DEFAULT_PATH)
  find_library(JXRGLUE_LIBRARY_DEBUG
    NAMES jxrglued
    PATHS "${_jxr_prefix}/debug/lib"
    NO_DEFAULT_PATH)
  select_library_configurations(JXRGLUE)
endif()

set(JXR_LIBRARIES ${JXRGLUE_LIBRARY} ${JPEGXR_LIBRARY})

find_package_handle_standard_args(
  JXR
  REQUIRED_VARS JXR_INCLUDE_DIRS JXR_LIBRARIES)