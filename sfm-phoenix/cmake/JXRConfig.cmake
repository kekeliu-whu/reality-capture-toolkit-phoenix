include(FindPackageHandleStandardArgs)
include(SelectLibraryConfigurations)

set(_jxr_prefix "")
if(DEFINED VCPKG_INSTALLED_DIR AND DEFINED VCPKG_TARGET_TRIPLET)
  set(_jxr_prefix "${VCPKG_INSTALLED_DIR}/${VCPKG_TARGET_TRIPLET}")
elseif(DEFINED ENV{VCPKG_ROOT} AND DEFINED VCPKG_TARGET_TRIPLET)
  file(TO_CMAKE_PATH "$ENV{VCPKG_ROOT}" _jxr_vcpkg_root)
  set(_jxr_prefix "${_jxr_vcpkg_root}/installed/${VCPKG_TARGET_TRIPLET}")
elseif(EXISTS "D:/vcpkg/installed/x64-windows")
  set(_jxr_prefix "D:/vcpkg/installed/x64-windows")
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