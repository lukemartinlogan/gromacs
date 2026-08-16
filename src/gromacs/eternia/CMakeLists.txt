# ---------------------------------------------------------------------------
# src/gromacs/eternia -- PME grid paging over the Clio CTE gpu_vector.
#
# STANDALONE, and configured separately from GROMACS. Two reasons:
#
#  1. The paging kernels are C++20 device coroutines, which only clang++-22
#     driving -x cuda compiles; GROMACS builds its CUDA with nvcc, which
#     rejects co_await in device code outright.
#  2. The kernel is being proved against a CPU reference before any GROMACS
#     file is touched. Debugging a device coroutine inside GROMACS's build
#     would be the same mistake the LAMMPS integration made.
#
#   cmake -S src/gromacs/eternia -B build-eternia \
#     -DCMAKE_CUDA_COMPILER=clang++-22 \
#     -DCMAKE_CUDA_ARCHITECTURES=89 \
#     -DCMAKE_PREFIX_PATH=<clio-inst> \
#     -Diowarp-core_DIR=<clio-inst>/lib/cmake/iowarp-core
# ---------------------------------------------------------------------------
cmake_minimum_required(VERSION 3.30)
project(gromacs_eternia LANGUAGES CXX)

set(CMAKE_CXX_STANDARD 20)
set(CMAKE_CXX_STANDARD_REQUIRED ON)
set(CMAKE_POSITION_INDEPENDENT_CODE ON)

enable_language(CUDA)
if(NOT CMAKE_CUDA_COMPILER MATCHES "clang")
  message(FATAL_ERROR
    "needs clang as the CUDA compiler (-DCMAKE_CUDA_COMPILER=clang++-22): "
    "nvcc cannot compile the device coroutines the paging kernel uses.")
endif()

# iowarp-core names these in its exported link interfaces without
# find_package-ing them, so they must be imported first or the generate step
# fails on a missing target. Same workaround as the LAMMPS lib/eternia build.
find_package(nvcomp QUIET CONFIG)
find_package(yaml-cpp QUIET CONFIG)
find_package(Threads QUIET)
find_package(iowarp-core REQUIRED)
find_package(CUDAToolkit REQUIRED)

set(CMAKE_CUDA_FLAGS
    "${CMAKE_CUDA_FLAGS} -Wno-unknown-cuda-version -fno-strict-aliasing")
if(EXISTS "${CUDAToolkit_TARGET_DIR}/include/cccl")
  set(CMAKE_CUDA_FLAGS
      "${CMAKE_CUDA_FLAGS} -isystem ${CUDAToolkit_TARGET_DIR}/include/cccl")
endif()

set_source_files_properties(eternia_pme_bench.cc PROPERTIES LANGUAGE CUDA)
add_executable(gmx_eternia_pme_bench eternia_pme_bench.cc)
target_compile_definitions(gmx_eternia_pme_bench PRIVATE CLIO_YIELD_CORO)
target_include_directories(gmx_eternia_pme_bench PRIVATE ${IOWARP_CORE_INCLUDE_DIR})

# ctp::cudacxx and clio::run::clio_run_cxx_gpu carry CTP_ENABLE_CUDA=1; the
# host-only targets deliberately do not, and without it every Clio header
# takes its host branch and fails on the first __threadfence in device code.
target_link_libraries(gmx_eternia_pme_bench PRIVATE
    clio::run::cxx
    clio::run::clio_run_cxx_gpu
    clio::run::admin_client
    clio::run::bdev_client
    clio::cte::core_client
    ctp::cudacxx
    CUDA::cudart)

# Separable compilation OFF: it leaves relocatable device code needing a
# device-link step, and under clang-CUDA CMake drives fatbinary with
# pre-CUDA-13 options. Every device callee here lives in a header, so this
# translation unit's fatbin is self-contained. iowarp-core makes the same
# choice for the same reason.
set_target_properties(gmx_eternia_pme_bench PROPERTIES
    CUDA_SEPARABLE_COMPILATION OFF)
