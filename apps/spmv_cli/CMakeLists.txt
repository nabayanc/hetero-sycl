# apps/spmv_cli/CMakeLists.txt  (final version)

add_executable(spmv_cli main.cpp)

#  • spmv_obj supplies the .o files *and* propagates its include‑dir usage
#  • sycl links the runtime
target_link_libraries(spmv_cli PRIVATE
    spmv_obj     # ← only once; do NOT also add $<TARGET_OBJECTS:…>
    sycl
)
