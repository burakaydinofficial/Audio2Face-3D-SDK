cmake_minimum_required(VERSION 3.24)

# When built as a standalone project (not via the root CMakeLists), declare a project.
if(NOT TARGET audio2x-sdk::audio2x)
  project(a2f_bridge LANGUAGES CXX)

  set(CMAKE_CXX_STANDARD 17)
  set(CMAKE_CXX_STANDARD_REQUIRED ON)
  set(CMAKE_POSITION_INDEPENDENT_CODE ON)

  # Find the pre-built Audio2Face SDK via our custom find module.
  list(APPEND CMAKE_MODULE_PATH "${CMAKE_CURRENT_SOURCE_DIR}/cmake")
  find_package(A2X_SDK REQUIRED)

  # CUDA is needed for the cudaStreamSynchronize call in the bridge.
  find_package(CUDAToolkit 12.8 REQUIRED)

  set(A2F_BRIDGE_LINK_TARGET A2X_SDK::audio2x)
else()
  # In-tree build: link against the alias target from audio2x-sdk/CMakeLists.txt
  set(A2F_BRIDGE_LINK_TARGET audio2x-sdk::audio2x)
  # Ensure CUDA::cudart is available (root CMakeLists normally finds it,
  # but guard against different parent projects).
  if(NOT TARGET CUDA::cudart)
    find_package(CUDAToolkit 12.8 REQUIRED)
  endif()
endif()

# ---------------------------------------------------------------------------
# Bridge shared library
# ---------------------------------------------------------------------------
add_library(a2f_bridge SHARED
  src/a2f_bridge.cpp
)

target_include_directories(a2f_bridge
  PUBLIC
    $<BUILD_INTERFACE:${CMAKE_CURRENT_SOURCE_DIR}/include>
    $<INSTALL_INTERFACE:include>
)

target_compile_definitions(a2f_bridge
  PRIVATE
    A2F_BRIDGE_EXPORTS
    _GLIBCXX_USE_CXX11_ABI=0
)

target_link_libraries(a2f_bridge
  PRIVATE
    ${A2F_BRIDGE_LINK_TARGET}
    CUDA::cudart
)

set_target_properties(a2f_bridge PROPERTIES
  OUTPUT_NAME "a2f_bridge"
  ARCHIVE_OUTPUT_DIRECTORY ${CMAKE_CURRENT_BINARY_DIR}/lib
  LIBRARY_OUTPUT_DIRECTORY ${CMAKE_CURRENT_BINARY_DIR}/lib
  RUNTIME_OUTPUT_DIRECTORY ${CMAKE_CURRENT_BINARY_DIR}/bin
)

if(UNIX)
  target_link_options(a2f_bridge PRIVATE
    "LINKER:-Bsymbolic"
    "LINKER:--exclude-libs,ALL"
  )
  # Only export our extern "C" symbols
  set_target_properties(a2f_bridge PROPERTIES
    CXX_VISIBILITY_PRESET hidden
    C_VISIBILITY_PRESET hidden
  )
endif()

if(MSVC)
  target_compile_options(a2f_bridge PRIVATE /wd4267 /EHsc /nologo)
endif()

# Alias for downstream use
add_library(a2f::bridge ALIAS a2f_bridge)
