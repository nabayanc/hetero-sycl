# apps/spmv_cli/CMakeLists.txt  (final version)

add_executable(bicgstab bicgstab_cli.cpp)

#  • spmv_obj supplies the .o files *and* propagates its include‑dir usage
#  • sycl links the runtime
target_link_libraries(bicgstab PRIVATE
    spmv_obj     # ← only once; do NOT also add $<TARGET_OBJECTS:…>
    sycl
)
