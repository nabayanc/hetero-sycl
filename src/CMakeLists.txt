# Gather sources
file(GLOB LIB_HEADERS "${PROJECT_SOURCE_DIR}/include/spmv/*.hpp")
file(GLOB LIB_SOURCES
     "${CMAKE_CURRENT_SOURCE_DIR}/*.cpp"
     "${CMAKE_CURRENT_SOURCE_DIR}/schedulers/*.cpp"
     "${CMAKE_CURRENT_SOURCE_DIR}/schedulers/adaptive_load_balancing.cpp"
     "${CMAKE_CURRENT_SOURCE_DIR}/spmv_csr.cpp")

# ①  Build an OBJECT library ⇒ compiles each .cpp once, no archive stage
add_library(spmv_obj OBJECT ${LIB_HEADERS} ${LIB_SOURCES})

target_include_directories(spmv_obj
    PUBLIC
        $<BUILD_INTERFACE:${PROJECT_SOURCE_DIR}/include>
)

target_link_libraries(spmv_obj
    PUBLIC
        sycl                           # propagate include dirs / flags
)

set_target_properties(spmv_obj PROPERTIES
    POSITION_INDEPENDENT_CODE ON)      # good practice
