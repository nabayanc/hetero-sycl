# apps/exp01_baseline_spmv/CMakeLists.txt
add_executable(exp03_multithreaded_dispatch_main exp03_multithreaded_dispatch_main.cpp)

target_link_libraries(exp03_multithreaded_dispatch_main PRIVATE
    spmv_obj # From the main src directory
    sycl
)