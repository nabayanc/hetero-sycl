cmake_minimum_required(VERSION 3.23)
project(spmv-hetero-sycl LANGUAGES CXX)

# -------------------------------------------------------------------- #
#  Tool‑chain discovery                                                #
# -------------------------------------------------------------------- #
if (NOT CMAKE_CXX_COMPILER)
  if (DEFINED ENV{DPCPP})
    message(STATUS "Using DPCPP from $ENV{DPCPP}")
    set(CMAKE_CXX_COMPILER $ENV{DPCPP} CACHE STRING "" FORCE)
  endif()
endif()

# -------------------------------------------------------------------- #
#  Language & build‑type                                               #
# -------------------------------------------------------------------- #
set(CMAKE_CXX_STANDARD 17)
set(CMAKE_CXX_STANDARD_REQUIRED ON)

if (NOT CMAKE_BUILD_TYPE)
  set(CMAKE_BUILD_TYPE Release)
endif()

# -------------------------------------------------------------------- #
#  Common SYCL flags (compile + link)                                  #
# -------------------------------------------------------------------- #
set(SYCL_COMMON_FLAGS
    -fsycl
    -fsycl-device-code-split=per_kernel
    -fsycl-targets=native_cpu,nvptx64-nvidia-cuda
    -Xsycl-target-backend=nvptx64-nvidia-cuda --cuda-gpu-arch=sm_86
)

add_compile_options(${SYCL_COMMON_FLAGS})
add_link_options(${SYCL_COMMON_FLAGS})

# Optional: CUDA toolkit path (only needed if not in default location)
if (DEFINED ENV{CUDA_PATH})
  add_compile_options(--cuda-path=$ENV{CUDA_PATH})
  add_link_options   (--cuda-path=$ENV{CUDA_PATH})
endif()

# Enable per‑kernel profiling info (sets a macro)
add_compile_definitions(SPMV_ENABLE_PROFILING)


# (adjust the paths below to match your system)
set(OpenMP_CXX_FLAGS       "-fopenmp -I/usr/lib/gcc/x86_64-linux-gnu/11/include")
set(OpenMP_CXX_LIB_NAMES    gomp)
set(OpenMP_gomp_LIBRARY     "/usr/lib/gcc/x86_64-linux-gnu/11/libgomp.so")

find_package(OpenMP REQUIRED)
add_executable(createcsr apps/createcsr/createcsr.cpp)
target_link_libraries(createcsr PRIVATE OpenMP::OpenMP_CXX)


# -------------------------------------------------------------------- #
#  Sub‑directories                                                     #
# -------------------------------------------------------------------- #
add_subdirectory(src)
add_subdirectory(apps/spmv_cli)
add_subdirectory(apps/spmv_bicgstab)
add_subdirectory(apps/exp01_baseline_spmv)
add_subdirectory(apps/exp03_multithreaded_dispatch)
enable_testing()
add_subdirectory(tests)
