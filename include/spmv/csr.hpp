#pragma once

#include <vector>
#include <string>
#include <stdexcept>
#include <fstream>
#include <sstream>
#include <algorithm>
#include <tuple>

namespace spmv {

/**
 * @brief Compressed Sparse Row matrix on the host.
 */
struct CSR {
  int                 nrows   = 0;   ///< number of rows
  int                 ncols   = 0;   ///< number of columns
  int                 nnz     = 0;   ///< number of nonzeros
  std::vector<float>  vals;          ///< nonzero values, length nnz
  std::vector<int>    col_idx;       ///< column indices, length nnz
  std::vector<int>    row_ptr;       ///< row pointers, length nrows+1

  /** 
   * @return true if the matrix is empty (no nonzeros)
   */
  [[nodiscard]] 
  bool empty() const noexcept { 
    return nnz == 0; 
  }

  /**
   * @brief Load a Matrix-Market coordinate (.mtx) file into CSR format.
   * @param path  Path to the .mtx file
   * @throws std::runtime_error on I/O or parse errors
   */
  static CSR load_mm(const std::string& path);
};

} // namespace spmv
