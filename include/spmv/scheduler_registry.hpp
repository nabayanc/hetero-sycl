#pragma once
#include "spmv/scheduler.hpp"
#include <memory>

namespace spmv::detail {

/* ------------------------------------------------------------------ */
/*  Factory‑function alias that returns a new scheduler instance       */
/* ------------------------------------------------------------------ */
using maker_t = std::unique_ptr<IScheduler>(*)();

/* ------------------------------------------------------------------ */
/*  Registrar: static object whose ctor inserts <name, maker> into     */
/*  the global map (definition lives in scheduler_factory.cpp).        */
/* ------------------------------------------------------------------ */
struct Registrar {
  Registrar(const char* name, maker_t maker);
};

} // namespace spmv::detail

/* ------------------------------------------------------------------ */
/*  Macro every concrete scheduler uses to self‑register               */
/* ------------------------------------------------------------------ */
#define REGISTER_SCHEDULER(NAME, TYPE)                                      \
  static ::spmv::detail::Registrar _reg_##TYPE{                             \
      NAME, []() -> std::unique_ptr<::spmv::IScheduler> {                   \
        return std::make_unique<TYPE>();                                    \
      }}
