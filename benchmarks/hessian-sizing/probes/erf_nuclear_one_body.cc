#include <libint2.hpp>
#include <iostream>
#include <cmath>
using namespace libint2;
// erf_nuclear (one-body, finite-width Gaussian charge) vs eri3 with a Gaussian site, and derivs
static double one(Engine&e, Shell&a, Shell&b, int comp, int idx){ e.compute(a,b); return e.results()[comp]?e.results()[comp][idx]:0.0; }
int main(int argc,char**argv){
  initialize();
  int la=atoi(argv[1]), lb=atoi(argv[2]); double zeta=0.8;
  std::vector<std::pair<double,std::array<double,3>>> q{{1.0,{0.4,-0.3,0.9}}};
  auto mk=[&](int deriv,Operator op){
    Engine e(op,1,std::max(la,lb),deriv);
    if(op==Operator::erf_nuclear) e.set_params(std::make_tuple(std::sqrt(zeta)*(getenv("WS")?atof(getenv("WS")):1.0),q)); else e.set_params(q);
    return e; };
  Shell b{{0.7},{{lb,true,{1.0}}},{{0.3,0.2,-0.5}}};
  auto A=[&](double dx){ return Shell{{0.9},{{la,true,{1.0}}},{{0.0+dx,0.0,1.0}}}; };
  // value check: erf_nuclear vs eri3(site gaussian)
  {
    Shell a=A(0);
    Engine en=mk(0,Operator::erf_nuclear);
    en.compute(a,b);
    double v1=en.results()[0][0];
    Engine e3(Operator::coulomb,3,std::max(la,lb),0); e3.set(BraKet::xs_xx);
    Shell site{{zeta},{{0,false,{1.0}}},{{0.4,-0.3,0.9}}};
    e3.compute(site,a,b);
    double N=std::pow(2*zeta/M_PI,0.75);
    double v3=-e3.results()[0][0]*std::pow(zeta/M_PI,1.5)/N;
    std::cout<<"value erf_nuclear="<<v1<<" eri3-site="<<v3<<" diff="<<v1-v3<<"\n";
  }
  // deriv1 vs FD of value; deriv2(Ax,Ax) vs FD of deriv1 Ax
  double h=1e-4;
  Engine e0=mk(0,Operator::erf_nuclear), e1=mk(1,Operator::erf_nuclear), e2=mk(2,Operator::erf_nuclear);
  Shell ap=A(h), am=A(-h), a0=A(0);
  double fd1=(one(e0,ap,b,0,0)-one(e0,am,b,0,0))/(2*h);
  double an1=one(e1,a0,b,0,0);
  std::cout<<"d1 analytic="<<an1<<" fd="<<fd1<<" diff="<<an1-fd1<<"\n";
  double fd2=(one(e1,ap,b,0,0)-one(e1,am,b,0,0))/(2*h);
  double an2=one(e2,a0,b,0,0);
  std::cout<<"d2(Ax,Ax) analytic="<<an2<<" fd="<<fd2<<" diff="<<an2-fd2<<"\n";
  finalize();
}
