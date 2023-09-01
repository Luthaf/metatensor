# Torch seems to require features added in CMake 3.15, but does not
# declare it in its own `cmake_minimum_required`
cmake_minimum_required(VERSION 3.15)

if(NOT "${LAST_CMAKE_VERSION}" VERSION_EQUAL ${CMAKE_VERSION})
    set(LAST_CMAKE_VERSION ${CMAKE_VERSION} CACHE INTERNAL "Last version of cmake used to configure")
    if (${CMAKE_CURRENT_SOURCE_DIR} STREQUAL ${CMAKE_SOURCE_DIR})
        message(STATUS "Running CMake version ${CMAKE_VERSION}")
    endif()
endif()


file(READ ${CMAKE_CURRENT_SOURCE_DIR}/VERSION METATENSOR_TORCH_VERSION)
string(STRIP ${METATENSOR_TORCH_VERSION} METATENSOR_TORCH_VERSION)

project(metatensor_torch
    VERSION ${METATENSOR_TORCH_VERSION}
    LANGUAGES CXX
)

option(METATENSOR_TORCH_TESTS "Build metatensor-torch C++ tests" OFF)
set(BIN_INSTALL_DIR "bin" CACHE PATH "Path relative to CMAKE_INSTALL_PREFIX where to install binaries/DLL")
set(LIB_INSTALL_DIR "lib" CACHE PATH "Path relative to CMAKE_INSTALL_PREFIX where to install libraries")
set(INCLUDE_INSTALL_DIR "include" CACHE PATH "Path relative to CMAKE_INSTALL_PREFIX where to install headers")

# Set a default build type if none was specified
if (${CMAKE_CURRENT_SOURCE_DIR} STREQUAL ${CMAKE_SOURCE_DIR})
    if("${CMAKE_BUILD_TYPE}" STREQUAL "" AND "${CMAKE_CONFIGURATION_TYPES}" STREQUAL "")
        message(STATUS "Setting build type to 'release' as none was specified.")
        set(
            CMAKE_BUILD_TYPE "release"
            CACHE STRING
            "Choose the type of build, options are: none(CMAKE_CXX_FLAGS or CMAKE_C_FLAGS used) debug release relwithdebinfo minsizerel."
            FORCE
        )
        set_property(CACHE CMAKE_BUILD_TYPE PROPERTY STRINGS release debug relwithdebinfo minsizerel none)
    endif()
endif()

set(REQUIRED_METATENSOR_VERSION "0.1")
find_package(metatensor ${REQUIRED_METATENSOR_VERSION} CONFIG REQUIRED)

# FindCUDNN.cmake distributed with PyTorch is a bit broken, so we have a
# fixed version in `cmake/FindCUDNN.cmake`
set(CMAKE_MODULE_PATH "${CMAKE_CURRENT_SOURCE_DIR}/cmake;${CMAKE_MODULE_PATH}")

find_package(Torch 1.11 REQUIRED)

set(METATENSOR_TORCH_HEADERS
    "include/metatensor/torch/array.hpp"
    "include/metatensor/torch/labels.hpp"
    "include/metatensor/torch/block.hpp"
    "include/metatensor/torch/tensor.hpp"
    "include/metatensor/torch.hpp"
)

set(METATENSOR_TORCH_SOURCE
    "src/array.cpp"
    "src/labels.cpp"
    "src/block.cpp"
    "src/tensor.cpp"
    "src/misc.cpp"
    "src/register.cpp"
)

add_library(metatensor_torch SHARED
    ${METATENSOR_TORCH_HEADERS}
    ${METATENSOR_TORCH_SOURCE}
)

target_link_libraries(metatensor_torch PUBLIC torch metatensor::shared)
target_compile_features(metatensor_torch PUBLIC cxx_std_14)
target_include_directories(metatensor_torch PUBLIC
    $<BUILD_INTERFACE:${CMAKE_CURRENT_SOURCE_DIR}/include>
    $<BUILD_INTERFACE:${CMAKE_CURRENT_BINARY_DIR}/include>
    $<INSTALL_INTERFACE:${INCLUDE_INSTALL_DIR}>
)

# Create a header defining METATENSOR_TORCH_EXPORT for to export classes/functions
# in DLL on Windows.
set_target_properties(metatensor_torch PROPERTIES
    # hide non-exported symbols by default, this mimics Windows behavior on Unix
    CXX_VISIBILITY_PRESET hidden
)

include(GenerateExportHeader)
generate_export_header(metatensor_torch
    BASE_NAME METATENSOR_TORCH
    EXPORT_FILE_NAME ${CMAKE_CURRENT_BINARY_DIR}/include/metatensor/torch/exports.h
)
target_compile_definitions(metatensor_torch PRIVATE metatensor_torch_EXPORTS)


if (METATENSOR_TORCH_TESTS)
    enable_testing()
    add_subdirectory(tests)
endif()

#------------------------------------------------------------------------------#
# Installation configuration
#------------------------------------------------------------------------------#
include(CMakePackageConfigHelpers)
write_basic_package_version_file(
    metatensor_torch-config-version.cmake
    VERSION ${METATENSOR_TORCH_VERSION}
    COMPATIBILITY SameMinorVersion
)

install(TARGETS metatensor_torch
    EXPORT metatensor_torch-targets
    ARCHIVE DESTINATION ${LIB_INSTALL_DIR}
    LIBRARY DESTINATION ${LIB_INSTALL_DIR}
    RUNTIME DESTINATION ${BIN_INSTALL_DIR}
)
install(EXPORT metatensor_torch-targets
    DESTINATION ${LIB_INSTALL_DIR}/cmake/metatensor_torch
)

configure_file(
    ${CMAKE_CURRENT_SOURCE_DIR}/cmake/metatensor_torch-config.in.cmake
    ${CMAKE_CURRENT_BINARY_DIR}/metatensor_torch-config.cmake
    @ONLY
)
install(FILES
    ${CMAKE_CURRENT_BINARY_DIR}/metatensor_torch-config-version.cmake
    ${CMAKE_CURRENT_BINARY_DIR}/metatensor_torch-config.cmake
    DESTINATION ${LIB_INSTALL_DIR}/cmake/metatensor_torch
)

install(DIRECTORY "include/metatensor" DESTINATION ${INCLUDE_INSTALL_DIR})
install(DIRECTORY ${CMAKE_CURRENT_BINARY_DIR}/include/metatensor DESTINATION ${INCLUDE_INSTALL_DIR})