#include <libint2.hpp>
#include <iostream>
using namespace libint2;
int main(int argc,char**argv){
  initialize();
  int lmax=atoi(argv[1]); int d=atoi(argv[2]); int la=argc>3?atoi(argv[3]):lmax; int lb=argc>4?atoi(argv[4]):lmax;
  try{
    Engine e(Operator::coulomb, 3, lmax, d);
    e.set(BraKet::xs_xx);
    Shell site{{1.3},{{0,false,{1.0}}},{{0,0,0}}};
    Shell a{{0.9},{{la,true,{1.0}}},{{0.0,0.0,1.0}}};
    Shell b{{0.7},{{lb,true,{1.0}}},{{0.3,0.2,-0.5}}};
    e.compute(site,a,b);
    std::cout<<"OK l="<<lmax<<" deriv="<<d<<" nresults="<<e.results().size()<<" first="<<(e.results()[0]?e.results()[0][0]:0)<<"\n";
  }catch(std::exception&x){std::cout<<"EXC "<<x.what()<<"\n";}
  finalize();
}
