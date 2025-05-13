// apps/createcsr/createcsr.cpp
#include <bits/stdc++.h>
#include <getopt.h>

int main(int argc, char** argv) {
  int rows=0, cols=0;
  double density=0;
  std::string outfile;
  static struct option opts[] = {
    {"rows",    required_argument, 0, 'r'},
    {"cols",    required_argument, 0, 'c'},
    {"density", required_argument, 0, 'd'},
    {"output",  required_argument, 0, 'o'},
    {0,0,0,0}
  };
  int opt;
  while((opt = getopt_long(argc, argv, "r:c:d:o:", opts, nullptr)) != -1){
    switch(opt){
      case 'r': rows    = std::stoi(optarg); break;
      case 'c': cols    = std::stoi(optarg); break;
      case 'd': density = std::stod(optarg); break;
      case 'o': outfile = optarg;           break;
      default: std::cerr<<"Usage: createcsr --rows R --cols C --density D --output F\n"; return 1;
    }
  }
  if(rows<=0||cols<=0||density<=0||outfile.empty()){
    std::cerr<<"Invalid args\n"; return 1;
  }

  std::mt19937_64 rng(std::random_device{}());
  std::uniform_real_distribution<double> ud(0.0,1.0);

  std::ofstream out(outfile);
  out<<"%%MatrixMarket matrix coordinate real general\n";
  out<<rows<<" "<<cols<<" ";
  // estimate nnz
  size_t nnz_est = size_t(rows)*size_t(cols)*density;
  out<<nnz_est<<"\n";

  for(int i=0;i<rows;++i){
    for(int j=0;j<cols;++j){
      if(ud(rng) < density){
        double v = ud(rng);
        out<< (i+1) <<" "<< (j+1) <<" "<< v <<"\n";
      }
    }
  }
  return 0;
}
