#include "spmv/csr.hpp"

#include <fstream>
#include <sstream>
#include <tuple>
#include <vector>
#include <algorithm>
#include <stdexcept>

namespace spmv {

CSR CSR::load_mm(const std::string& path) {
  std::ifstream in(path);
  if (!in)
    throw std::runtime_error("cannot open " + path);

  std::string line;

  // 1) Read header
  if (!std::getline(in, line) || line.rfind("%%MatrixMarket", 0) != 0)
    throw std::runtime_error("invalid MatrixMarket header");

  // 2) Skip comments to size line
  do {
    if (!std::getline(in, line))
      throw std::runtime_error("unexpected EOF before size line");
  } while (line.empty() || line[0] == '%');

  // 3) Parse M, N, (ignore NNZ)
  std::istringstream header(line);
  int M, N, dummy_nnz;
  if (!(header >> M >> N >> dummy_nnz))
    throw std::runtime_error("invalid size line: " + line);

  CSR A;
  A.nrows = M;
  A.ncols = N;

  // 4) Read *all* available triples until EOF
  std::vector<std::tuple<int,int,float>> triples;
  while (std::getline(in, line)) {
    if (line.empty() || line[0] == '%') 
      continue;
    std::istringstream iss(line);
    int r, c;
    float v;
    if (!(iss >> r >> c >> v))
      throw std::runtime_error("error parsing line: " + line);
    triples.emplace_back(r-1, c-1, v);
  }
  A.nnz = int(triples.size());

  // 5) Sort by row then column
  std::sort(triples.begin(), triples.end(),
    [](auto &a, auto &b) {
      if (std::get<0>(a) != std::get<0>(b))
        return std::get<0>(a) < std::get<0>(b);
      return std::get<1>(a) < std::get<1>(b);
    });

  // 6) Build CSR arrays
  A.row_ptr.resize(M+1);
  A.col_idx.resize(A.nnz);
  A.vals   .resize(A.nnz);

  int cur_row = 0;
  A.row_ptr[0] = 0;
  int idx = 0;
  for (auto &t : triples) {
    int r = std::get<0>(t);
    int c = std::get<1>(t);
    float v = std::get<2>(t);

    while (cur_row < r) {
      A.row_ptr[++cur_row] = idx;
    }

    A.col_idx[idx] = c;
    A.vals   [idx] = v;
    ++idx;
  }
  while (cur_row < M) {
    A.row_ptr[++cur_row] = idx;
  }

  return A;
}

} // namespace spmv
