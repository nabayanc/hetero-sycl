#include <sycl/sycl.hpp>
#include <iostream>
int main() {
  for(auto& p : sycl::platform::get_platforms()) {
    std::cout << p.get_info<sycl::info::platform::name>() << '\n';
    for(auto& d : p.get_devices())
      std::cout << "  - "
                << d.get_info<sycl::info::device::name>() << '\n';
  }
  return 0;
}
