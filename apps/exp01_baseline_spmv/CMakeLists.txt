# apps/exp01_baseline_spmv/CMakeLists.txt
add_executable(exp01_baseline_spmv_main exp01_baseline_spmv_main.cpp)

target_link_libraries(exp01_baseline_spmv_main PRIVATE
    spmv_obj # From the main src directory
    sycl
)