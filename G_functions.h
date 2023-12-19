#ifndef GFUNCTIONS_H
#define GFUNCTIONS_H
#include "F2_functions.h"
#include "K2_functions.h"

typedef std::complex<double> comp;


comp G_ij(  comp En, 
            comp p,
            comp k,
            double mi,
            double mj,
            double mk,
            comp total_P,
            double L,
            double epsilon_h       )
{
    comp oneby2omegapLcube = 1.0/(2.0*omega_func(p,mi)*L*L*L);
    comp oneby2omegakLcube = 1.0/(2.0*omega_func(k,mj)*L*L*L);

    comp sig_i = sigma(En,p,mi,total_P);
    comp sig_j = sigma(En,k,mj,total_P);
    comp cutoff1 = cutoff_function_1(sig_i,mj,mk,epsilon_h);
    comp cutoff2 = cutoff_function_1(sig_j,mi,mk,epsilon_h);

    comp denom = (En - omega_func(p,mi) - omega_func(k,mj))*(En - omega_func(p,mi) - omega_func(k,mj))
                -((comp)total_P - p - k)*(total_P - p - k) - mk*mk; 

    comp onebydenom = 1.0/denom; 

    //std::cout<<"sigi="<<sig_i<<'\t'<<"sigj="<<sig_j<<'\t'<<"cutoff1="<<cutoff1<<'\t'<<"cutoff2="<<cutoff2<<"denom="<<denom<<std::endl;

    return oneby2omegapLcube*cutoff1*cutoff2*onebydenom*oneby2omegakLcube;

}






#endif