cmake_minimum_required(VERSION 3.18)
project(curvelets_torch LANGUAGES CXX)

set(_CURVELETS_IS_TOP_LEVEL OFF)
if(CMAKE_SOURCE_DIR STREQUAL CMAKE_CURRENT_SOURCE_DIR)
    set(_CURVELETS_IS_TOP_LEVEL ON)
endif()

if(_CURVELETS_IS_TOP_LEVEL AND NOT CMAKE_CONFIGURATION_TYPES AND NOT CMAKE_BUILD_TYPE)
    set(CMAKE_BUILD_TYPE Release CACHE STRING "Build type" FORCE)
endif()

if(_CURVELETS_IS_TOP_LEVEL)
    set(CMAKE_CXX_STANDARD 20)
    set(CMAKE_CXX_STANDARD_REQUIRED ON)
    set(CMAKE_CXX_EXTENSIONS OFF)
endif()

set(CURVELETS_TORCH_CXX11_ABI "1" CACHE STRING
    "GLIBCXX CXX11 ABI flag when linking directly against libtorch")

set(_TORCH_ROOT "")
if(DEFINED TORCH_INSTALL_PREFIX AND EXISTS "${TORCH_INSTALL_PREFIX}/include/torch/csrc/api/include/torch/torch.h")
    set(_TORCH_ROOT "${TORCH_INSTALL_PREFIX}")
else()
    foreach(_prefix ${CMAKE_PREFIX_PATH})
        foreach(_cand "${_prefix}" "${_prefix}/../..")
            get_filename_component(_abs_cand "${_cand}" ABSOLUTE)
            if(EXISTS "${_abs_cand}/include/torch/csrc/api/include/torch/torch.h")
                set(_TORCH_ROOT "${_abs_cand}")
                break()
            endif()
        endforeach()
        if(_TORCH_ROOT)
            break()
        endif()
    endforeach()
endif()

set(_NEEDS_CUDA_TOOLKIT OFF)
if(_TORCH_ROOT AND EXISTS "${_TORCH_ROOT}/share/cmake/Caffe2/Caffe2Config.cmake")
    file(READ "${_TORCH_ROOT}/share/cmake/Caffe2/Caffe2Config.cmake" _CAFFE2_CFG)
    if(_CAFFE2_CFG MATCHES "set\\(CAFFE2_USE_CUDA 1\\)")
        set(_NEEDS_CUDA_TOOLKIT ON)
    endif()
endif()

find_program(_CURVELETS_NVCC nvcc
    HINTS ENV CUDA_HOME ENV CUDA_PATH
    PATH_SUFFIXES bin
)

set(_APPLY_FALLBACK_TORCH_FLAGS OFF)
if(NOT _NEEDS_CUDA_TOOLKIT OR _CURVELETS_NVCC OR NOT _TORCH_ROOT)
    find_package(Torch REQUIRED)
    set(_TORCH_RPATH "")
    if(_TORCH_ROOT)
        set(_TORCH_RPATH "${_TORCH_ROOT}/lib")
    endif()
else()
    message(STATUS "curvelets_torch: CUDA toolkit not found; linking directly against libtorch at ${_TORCH_ROOT}")
    find_package(Threads REQUIRED)
    find_library(_TORCH_LIB torch PATHS "${_TORCH_ROOT}/lib" REQUIRED NO_DEFAULT_PATH)
    find_library(_TORCH_CPU_LIB torch_cpu PATHS "${_TORCH_ROOT}/lib" REQUIRED NO_DEFAULT_PATH)
    find_library(_C10_LIB c10 PATHS "${_TORCH_ROOT}/lib" REQUIRED NO_DEFAULT_PATH)
    set(TORCH_INCLUDE_DIRS
        "${_TORCH_ROOT}/include"
        "${_TORCH_ROOT}/include/torch/csrc/api/include"
    )
    set(TORCH_LIBRARIES
        "${_TORCH_LIB}"
        "${_TORCH_CPU_LIB}"
        "${_C10_LIB}"
        Threads::Threads
        ${CMAKE_DL_LIBS}
    )
    if(NOT MSVC)
        set(TORCH_CXX_FLAGS "-D_GLIBCXX_USE_CXX11_ABI=${CURVELETS_TORCH_CXX11_ABI}")
        set(_APPLY_FALLBACK_TORCH_FLAGS ON)
    endif()
    set(_TORCH_RPATH "${_TORCH_ROOT}/lib")
endif()

option(CURVELETS_BUILD_TESTS "Build standalone C++ tests" ${_CURVELETS_IS_TOP_LEVEL})
option(CURVELETS_BUILD_EXAMPLES "Build C++ examples" ${_CURVELETS_IS_TOP_LEVEL})
option(CURVELETS_INSTALL "Generate install rules" ${_CURVELETS_IS_TOP_LEVEL})

add_library(curvelets_torch SHARED
    src/udct_ops.cpp
    src/udct_module.cpp
)
add_library(curvelets::torch ALIAS curvelets_torch)

set_target_properties(curvelets_torch PROPERTIES
    POSITION_INDEPENDENT_CODE ON
    CXX_EXTENSIONS OFF
)
target_compile_features(curvelets_torch PUBLIC cxx_std_20)
target_compile_definitions(curvelets_torch PRIVATE CURVELETS_BUILD_DLL)
if(MSVC)
    target_compile_definitions(curvelets_torch PUBLIC NOMINMAX)
endif()

if(NOT WIN32 AND _TORCH_RPATH)
    set_target_properties(curvelets_torch PROPERTIES
        BUILD_RPATH "${_TORCH_RPATH}"
        INSTALL_RPATH "${_TORCH_RPATH}"
    )
endif()

target_include_directories(curvelets_torch
    PUBLIC
        $<BUILD_INTERFACE:${CMAKE_CURRENT_SOURCE_DIR}/include>
        $<INSTALL_INTERFACE:include>
        ${TORCH_INCLUDE_DIRS}
)

target_link_libraries(curvelets_torch PUBLIC ${TORCH_LIBRARIES})

if(_APPLY_FALLBACK_TORCH_FLAGS AND TORCH_CXX_FLAGS)
    separate_arguments(TORCH_CXX_FLAGS_LIST NATIVE_COMMAND "${TORCH_CXX_FLAGS}")
    target_compile_options(curvelets_torch PUBLIC ${TORCH_CXX_FLAGS_LIST})
endif()

if(CURVELETS_BUILD_TESTS)
    add_executable(test_udct_cpp tests/test_udct_cpp.cpp)
    target_link_libraries(test_udct_cpp PRIVATE curvelets::torch)
    if(NOT WIN32)
        set_target_properties(test_udct_cpp PROPERTIES
            BUILD_RPATH "$<TARGET_FILE_DIR:curvelets_torch>;${_TORCH_RPATH}"
        )
    endif()
endif()

if(CURVELETS_BUILD_EXAMPLES)
    add_executable(train_model examples/train_model.cpp)
    target_link_libraries(train_model PRIVATE curvelets::torch)
    if(NOT WIN32)
        set_target_properties(train_model PROPERTIES
            BUILD_RPATH "$<TARGET_FILE_DIR:curvelets_torch>;${_TORCH_RPATH}"
        )
    endif()
endif()

if(CURVELETS_INSTALL)
    include(GNUInstallDirs)
    install(TARGETS curvelets_torch
        LIBRARY DESTINATION ${CMAKE_INSTALL_LIBDIR}
        ARCHIVE DESTINATION ${CMAKE_INSTALL_LIBDIR}
        RUNTIME DESTINATION ${CMAKE_INSTALL_BINDIR}
    )
    install(DIRECTORY include/curvelets
        DESTINATION ${CMAKE_INSTALL_INCLUDEDIR}
    )
endif()
