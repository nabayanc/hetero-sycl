#include "spmv/scheduler.hpp"
#include "spmv/scheduler_registry.hpp"
#include <stdexcept>
#include <unordered_map>

namespace spmv::detail {

/* ------------------------------------------------------------------ */
/*  Internal registry: string → factory function                       */
/* ------------------------------------------------------------------ */
static std::unordered_map<std::string, maker_t>& registry()
{
  static std::unordered_map<std::string, maker_t> R;
  return R;
}

/*  Registrar ctor inserts maker into the table                       */
Registrar::Registrar(const char* name, maker_t maker)
{
  registry().emplace(name, maker);
}

} // namespace spmv::detail

/* -------------------------------------------------------------------- */
/*  Public factory: look up scheduler by name                           */
/* -------------------------------------------------------------------- */
namespace spmv {

std::unique_ptr<IScheduler> make_scheduler(const std::string& kind)
{
  using namespace detail;
  auto it = registry().find(kind);
  if (it == registry().end())
    throw std::runtime_error("unknown scheduler: " + kind);
  return it->second();                 // call maker()
}

} // namespace spmv
