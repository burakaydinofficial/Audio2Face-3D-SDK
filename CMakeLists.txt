cmake_minimum_required(VERSION 3.24)

project(audio2x-sdk LANGUAGES CXX)

set(CMAKE_POSITION_INDEPENDENT_CODE ON)

set(CMAKE_MODULE_PATH
    ${CMAKE_CURRENT_SOURCE_DIR}/cmake/Modules
    ${CMAKE_MODULE_PATH}
)

set(CMAKE_CXX_STANDARD 17)
set(CMAKE_CXX_STANDARD_REQUIRED ON)

# Normalize build type for dependency paths

# Note: CMake caches package paths found via find_package().
# Changing CMAKE_BUILD_TYPE later won't re-find dependencies.
# Use a clean build directory to switch between Debug/Release.
if(CMAKE_BUILD_TYPE MATCHES "^[Rr]el.*") # Match Release and RelWithDebInfo.
    set(DEPS_BUILD_TYPE "release")
else()
    set(DEPS_BUILD_TYPE "debug")
endif()

# Run fetch_deps script to pull dependencies before configuring the project
if(WIN32)
    set(SCRIPT_EXT "bat")
else()
    set(SCRIPT_EXT "sh")
endif()

set(PACKMAN_PYTHON ${PROJECT_SOURCE_DIR}/tools/packman/python.${SCRIPT_EXT})

if(NOT EXISTS "${PROJECT_SOURCE_DIR}/_deps/target-deps")
    set(FETCH_DEPS_CMD "${PROJECT_SOURCE_DIR}/fetch_deps.${SCRIPT_EXT}")
    message(STATUS "Fetching dependencies with ${FETCH_DEPS_CMD}.")
    if(CMAKE_CONFIGURATION_TYPES)
        # We can't differentiate between debug and release at generation time in multi-config generators, so we fetch both.
        execute_process(
            COMMAND ${CMAKE_COMMAND} -E env "PYTHON=${PACKMAN_PYTHON}" ${FETCH_DEPS_CMD} "debug"
            COMMAND ${CMAKE_COMMAND} -E env "PYTHON=${PACKMAN_PYTHON}" ${FETCH_DEPS_CMD} "release"
            WORKING_DIRECTORY ${PROJECT_SOURCE_DIR}
            RESULT_VARIABLE FETCH_DEPS_RESULT
        )
    else()
        execute_process(
            COMMAND ${CMAKE_COMMAND} -E env "PYTHON=${PACKMAN_PYTHON}" ${FETCH_DEPS_CMD} ${DEPS_BUILD_TYPE}
            WORKING_DIRECTORY ${PROJECT_SOURCE_DIR}
            RESULT_VARIABLE FETCH_DEPS_RESULT
        )
    endif()
    
    message(STATUS "Fetch deps result: ${FETCH_DEPS_RESULT}")
endif()


# Set where to find dependencies
list(APPEND CMAKE_PREFIX_PATH
    ${PROJECT_SOURCE_DIR}/_deps/target-deps/audiofile
    ${PROJECT_SOURCE_DIR}/_deps/target-deps/cxxopts
    ${PROJECT_SOURCE_DIR}/_deps/target-deps/eigen
    ${PROJECT_SOURCE_DIR}/_deps/target-deps/magic_enum
    ${PROJECT_SOURCE_DIR}/_deps/target-deps/nlohmann_json
    ${PROJECT_SOURCE_DIR}/_deps/target-deps/refl-cpp
    ${PROJECT_SOURCE_DIR}/_deps/target-deps/tbtsvd
)

# This allows us to find a CUDA installation specified by ENV{CUDA_PATH} or CMAKE_CUDA_COMPILER_TOOLKIT_ROOT.
# NOTE: If CUDA has already been enabled in a parent project, this may not modify the location. However, this
# check ensures the version is sufficient.
find_package(CUDAToolkit 12.8 REQUIRED)
if(DEFINED CUDAToolkit_BIN_DIR)
    if(WIN32)
        set(CMAKE_CUDA_COMPILER ${CUDAToolkit_BIN_DIR}/nvcc.exe)
    else()
        set(CMAKE_CUDA_COMPILER ${CUDAToolkit_BIN_DIR}/nvcc)
    endif()
endif()

set(CMAKE_CUDA_ARCHITECTURES "75;80;86;87;89;90;100;120" CACHE STRING "CUDA architectures")
set(CMAKE_CUDA_RUNTIME_LIBRARY "Shared" CACHE STRING "Use shared CUDA runtime")
enable_language(CUDA)

# Find dependencies
find_package(AudioFile REQUIRED)
find_package(Benchmark REQUIRED)
find_package(Cnpy REQUIRED)
find_package(CXXOPTS REQUIRED)
find_package(Eigen3 REQUIRED)
find_package(GTest REQUIRED)
find_package(MagicEnum REQUIRED)
find_package(NlohmannJson REQUIRED)
find_package(ReflCpp REQUIRED)
find_package(TbtSVD REQUIRED)
find_package(TensorRT REQUIRED)
find_package(ZLIB 1.3.1 REQUIRED)

# Install pip dependencies from deps/requirements.txt
add_custom_target(install-pip-deps
    COMMAND ${PACKMAN_PYTHON} -m pip install -r ${PROJECT_SOURCE_DIR}/deps/requirements.txt --target ${PROJECT_SOURCE_DIR}/_deps/target-deps/pip_prebundle
    BYPRODUCTS ${PROJECT_SOURCE_DIR}/_deps/target-deps/pip_prebundle
    WORKING_DIRECTORY ${PROJECT_SOURCE_DIR}
    COMMENT "Installing pip dependencies from: ${PROJECT_SOURCE_DIR}/deps/requirements.txt"
    VERBATIM
    USES_TERMINAL # Print output in real time.
)


# Add preprocessor definitions
add_compile_definitions(
    AUDIO2X_LOG_LEVEL=1
    AUDIO2FACE_LOG_LEVEL=1
    AUDIO2EMOTION_LOG_LEVEL=1
    _GLIBCXX_USE_CXX11_ABI=0
    EIGEN_MPL2_ONLY
    USE_NVTX
)

if(MSVC)
    set(CMAKE_CXX_FLAGS "${CMAKE_CXX_FLAGS} /wd4267 /EHsc /nologo")
    set(CMAKE_CXX_FLAGS_RELEASE "${CMAKE_CXX_FLAGS_RELEASE} /MD")
    set(CMAKE_CXX_FLAGS_DEBUG "${CMAKE_CXX_FLAGS_DEBUG} /MDd")

    set(CMAKE_EXE_LINKER_FLAGS "${CMAKE_EXE_LINKER_FLAGS} /NODEFAULTLIB:LIBCMT")
    set(CMAKE_SHARED_LINKER_FLAGS "${CMAKE_SHARED_LINKER_FLAGS} /NODEFAULTLIB:LIBCMT")
elseif(UNIX)
    set(CMAKE_CXX_FLAGS "${CMAKE_CXX_FLAGS} -fPIC -Wno-undef -Wno-deprecated-declarations")
    set(CMAKE_CXX_FLAGS_DEBUG "${CMAKE_CXX_FLAGS_DEBUG} -ggdb3")
endif()


if (NOT DEFINED TEST_DATA_DIR)
    set(TEST_DATA_DIR "")
else()
    # Normalize to forward slashes and add a trailing slash
    file(TO_CMAKE_PATH "${TEST_DATA_DIR}" TEST_DATA_DIR_NORMALIZED)
    set(TEST_DATA_DIR "${TEST_DATA_DIR_NORMALIZED}/")
endif()

if (NOT DEFINED AUDIO2X_SDK_ROOT)
    set(AUDIO2X_SDK_ROOT ${CMAKE_CURRENT_SOURCE_DIR})
endif()

enable_testing()

add_subdirectory(audio2x-common)
add_subdirectory(audio2emotion-sdk)
add_subdirectory(audio2face-sdk)
add_subdirectory(audio2x-sdk)

option(BUILD_BRIDGE "Build the a2f_bridge C API shared library" OFF)
if(BUILD_BRIDGE)
  add_subdirectory(bridge)
endif()
