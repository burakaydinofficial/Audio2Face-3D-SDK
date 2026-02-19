# FindA2X_SDK.cmake
#
# Finds the pre-built NVIDIA Audio2Face-3D SDK (audio2x shared library).
#
# Set A2X_SDK_ROOT to the directory containing:
#   include/audio2face/audio2face.h
#   lib/audio2x.lib | lib/libaudio2x.so
#
# Produces the imported target:
#   A2X_SDK::audio2x
#
# Variables set:
#   A2X_SDK_FOUND
#   A2X_SDK_INCLUDE_DIR
#   A2X_SDK_LIBRARY

# Allow user override via environment or CMake variable
if(NOT A2X_SDK_ROOT)
  set(A2X_SDK_ROOT "$ENV{A2X_SDK_ROOT}" CACHE PATH "Root of pre-built Audio2Face SDK")
endif()

# --- Find the header ---
find_path(A2X_SDK_INCLUDE_DIR
  NAMES audio2face/audio2face.h
  PATHS
    ${A2X_SDK_ROOT}/include
    ${A2X_SDK_ROOT}
  PATH_SUFFIXES include
  NO_DEFAULT_PATH
)

# --- Find the library ---
find_library(A2X_SDK_LIBRARY
  NAMES audio2x
  PATHS
    ${A2X_SDK_ROOT}/lib
    ${A2X_SDK_ROOT}/bin
    ${A2X_SDK_ROOT}
  PATH_SUFFIXES lib lib64 bin
  NO_DEFAULT_PATH
)

# On Windows, find the DLL (runtime) separately from the import library
if(WIN32)
  find_file(A2X_SDK_DLL
    NAMES audio2x.dll
    PATHS
      ${A2X_SDK_ROOT}/bin
      ${A2X_SDK_ROOT}/lib
      ${A2X_SDK_ROOT}
    PATH_SUFFIXES bin lib
    NO_DEFAULT_PATH
  )
endif()

# --- Standard find_package handling ---
include(FindPackageHandleStandardArgs)
find_package_handle_standard_args(A2X_SDK
  REQUIRED_VARS A2X_SDK_LIBRARY A2X_SDK_INCLUDE_DIR
)

# --- Create imported target ---
if(A2X_SDK_FOUND AND NOT TARGET A2X_SDK::audio2x)
  add_library(A2X_SDK::audio2x SHARED IMPORTED)
  set_target_properties(A2X_SDK::audio2x PROPERTIES
    INTERFACE_INCLUDE_DIRECTORIES "${A2X_SDK_INCLUDE_DIR}"
  )

  if(WIN32)
    set_target_properties(A2X_SDK::audio2x PROPERTIES
      IMPORTED_LOCATION "${A2X_SDK_DLL}"      # .dll (runtime)
      IMPORTED_IMPLIB   "${A2X_SDK_LIBRARY}"  # .lib (import library)
    )
  else()
    set_target_properties(A2X_SDK::audio2x PROPERTIES
      IMPORTED_LOCATION "${A2X_SDK_LIBRARY}"
    )
  endif()
endif()

mark_as_advanced(A2X_SDK_INCLUDE_DIR A2X_SDK_LIBRARY A2X_SDK_DLL)
