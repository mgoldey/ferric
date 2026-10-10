#include <libint2.hpp>
#include <iostream>
#include <cmath>
using namespace libint2;
// 4th-order central stencil of eri3 first-derivative blocks vs libint's analytic 2nd derivatives.
int main(int argc,char**argv){
  initialize();
  int la=atoi(argv[1]), lb=atoi(argv[2]); double h=atof(argv[3]); double zeta=atof(argv[4]);
  int L=std::max(la,lb);
  Engine e1(Operator::coulomb,3,L,1); e1.set(BraKet::xs_xx);
  Engine e2(Operator::coulomb,3,L,2); e2.set(BraKet::xs_xx);
  Shell site{{zeta},{{0,false,{1.0}}},{{0.4,-0.3,0.9}}};
  auto mkA=[&](int ax,double d){ std::array<double,3> c{0.0,0.0,1.0}; c[ax]+=d; return Shell{{0.9,0.3},{{la,true,{1.0,0.5}}},c}; };
  auto mkB=[&](int ax,double d){ std::array<double,3> c{0.3,0.2,-0.5}; c[ax]+=d; return Shell{{0.7},{{lb,true,{1.0}}},c}; };
  int na=2*la+1, nb=2*lb+1, n=na*nb;
  auto idx2=[&](int i,int j){ return i*9 - i*(i-1)/2 + (j-i); };
  struct T{int ci,cj;const char*name;};
  // first-deriv comp ci differentiated by shifting centre of cj (3..5 -> A axis, 6..8 -> B axis)
  T tests[]={{3,4,"Ax,Ay"},{3,7,"Ax,By"},{3,3,"Ax,Ax"},{6,7,"Bx,By"},{4,6,"Ay,Bx"}};
  for(auto&t:tests){
    // analytic
    e2.compute(site,mkA(0,0),mkB(0,0));
    std::vector<double> an(n);
    int k=idx2(std::min(t.ci,t.cj),std::max(t.ci,t.cj));
    for(int i=0;i<n;i++) an[i]=e2.results()[k]?e2.results()[k][i]:0.0;
    // stencil: shift the centre carrying cj
    double w[4]={1,-8,8,-1}; double off[4]={-2,-1,1,2};
    std::vector<double> fd(n,0.0);
    for(int p=0;p<4;p++){
      double d=off[p]*h;
      Shell a = t.cj<6 ? mkA(t.cj-3,d) : mkA(0,0);
      Shell b = t.cj>=6 ? mkB(t.cj-6,d) : mkB(0,0);
      e1.compute(site,a,b);
      for(int i=0;i<n;i++) fd[i]+=w[p]*(e1.results()[t.ci]?e1.results()[t.ci][i]:0.0)/(12*h);
    }
    double md=0,mx=0; for(int i=0;i<n;i++){md=std::max(md,std::fabs(an[i]-fd[i]));mx=std::max(mx,std::fabs(an[i]));}
    printf("l=(%d,%d) h=%.0e zeta=%g %-6s block=%d  max|an|=%.3e max|an-fd|=%.3e\n",la,lb,h,zeta,t.name,n,mx,md);
  }
  finalize();
}
