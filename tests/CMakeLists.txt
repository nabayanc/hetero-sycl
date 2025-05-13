add_executable(test_devices test_devices.cpp)
target_link_libraries(test_devices PRIVATE sycl)
add_test(NAME list_devices COMMAND test_devices)
