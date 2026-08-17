# ---------------------------------------------------------------------------
# The Eternia paged nonbonded library, built as an ExternalProject.
#
# Separate toolchain because the paging kernel suspends on a page fault using
# C++20 device coroutines, which only clang++-22 -x cuda compiles, while
# GROMACS builds its CUDA with nvcc -- and nvcc rejects co_await in device
# code. The two halves meet at eternia_nb.h, which names neither CUDA nor
# Clio.
# ---------------------------------------------------------------------------
include(ExternalProject)

set(GMX_ETERNIA_SRC_DIR ${CMAKE_CURRENT_LIST_DIR})
set(GMX_ETERNIA_BIN_DIR ${CMAKE_BINARY_DIR}/eternia-build)
set(GMX_ETERNIA_INS_DIR ${CMAKE_BINARY_DIR}/eternia-install)

set(GMX_ETERNIA_CUDA_COMPILER "clang++-22" CACHE STRING
  "clang used for the Eternia device coroutines")

get_filename_component(GMX_ETERNIA_CLIO_PREFIX
  "${iowarp-core_DIR}/../../.." ABSOLUTE)
set(_etn_prefix "${GMX_ETERNIA_CLIO_PREFIX}")
foreach(_p IN LISTS CMAKE_PREFIX_PATH)
  string(APPEND _etn_prefix "|${_p}")
endforeach()

ExternalProject_Add(eternia_nb_build
  SOURCE_DIR ${GMX_ETERNIA_SRC_DIR}
  BINARY_DIR ${GMX_ETERNIA_BIN_DIR}
  INSTALL_DIR ${GMX_ETERNIA_INS_DIR}
  CMAKE_ARGS
    -DCMAKE_INSTALL_PREFIX=${GMX_ETERNIA_INS_DIR}
    -DCMAKE_BUILD_TYPE=${CMAKE_BUILD_TYPE}
    -DCMAKE_CUDA_COMPILER=${GMX_ETERNIA_CUDA_COMPILER}
    -DCMAKE_CUDA_ARCHITECTURES=${CMAKE_CUDA_ARCHITECTURES}
    -Diowarp-core_DIR=${iowarp-core_DIR}
    "-DCMAKE_PREFIX_PATH=${_etn_prefix}"
    -DETERNIA_LIB_ONLY=ON
  LIST_SEPARATOR |
  BUILD_BYPRODUCTS ${GMX_ETERNIA_INS_DIR}/lib/libgmx_eternia_nb.a
  # Without this the sub-build is stamped done once and never rerun, so
  # editing the kernel relinks against the previous archive and reports
  # success.
  BUILD_ALWAYS ON
)

add_library(GMX::eternia STATIC IMPORTED)
set_target_properties(GMX::eternia PROPERTIES
  IMPORTED_LOCATION ${GMX_ETERNIA_INS_DIR}/lib/libgmx_eternia_nb.a
  INTERFACE_INCLUDE_DIRECTORIES ${GMX_ETERNIA_SRC_DIR})
add_dependencies(GMX::eternia eternia_nb_build)

# Link the Clio libraries BY PATH, not through their CMake targets: those
# carry a cxx_std_20 usage requirement that CMake applies to every
# translation unit of anything linking them, which conflicts with GROMACS's
# own standard. A path carries no usage requirements. The kernel lives
# entirely inside the archive; GROMACS needs the symbols, not the headers.
set(GMX_ETERNIA_LIBS)
foreach(_l clio_run_cxx clio_run_cxx_gpu clio_admin_client clio_bdev_client
           clio_cte_core_client clio_ctp_cuda clio_ctp_host)
  find_library(_etn_lib_${_l} NAMES ${_l}
               HINTS ${GMX_ETERNIA_CLIO_PREFIX}/lib REQUIRED)
  list(APPEND GMX_ETERNIA_LIBS ${_etn_lib_${_l}})
endforeach()
find_library(_etn_zmq NAMES zmq REQUIRED)
list(APPEND GMX_ETERNIA_LIBS ${_etn_zmq})
