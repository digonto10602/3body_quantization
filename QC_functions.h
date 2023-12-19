#ifndef QCFUNCTIONS_H
#define QCFUNCTIONS_H
#include "F2_functions.h"
#include "K2_functions.h"
#include "G_functions.h"

typedef std::complex<double> comp;

//first the QC3 for identical particles

comp QC3_ID(   comp En,
            std::vector<comp> p,
            std::vector<comp> k,
            double eta_i,
            double scattering_length,  
            double mi,
            double mj,
            double mk,
            comp total_P,
            double alpha,
            double epsilon_h,
            double L    )
{
    comp px = p[0];
    comp py = p[1];
    comp pz = p[2];
    comp kx = k[0];
    comp ky = k[1];
    comp kz = k[2];

    comp spec_p = std::sqrt(px*px + py*py + pz*pz);
    comp spec_k = std::sqrt(kx*kx + ky*ky + kz*kz);
    comp sig_p = sigma(En, spec_p, mi, total_P);
    //std::cout<<"Here"<<std::endl;

    comp F = F2_i1(En, k, p, total_P, L, mi, mj, mk, alpha, epsilon_h);
    comp K2 = tilde_K2_00(eta_i, scattering_length, spec_p, spec_k, sig_p, mi, mj, mk, epsilon_h, L );
    comp G = G_ij(En, spec_p, spec_k, mi, mj, mk, total_P, L, epsilon_h);

    //std::cout<<"E="<<En<<'\t'<<"F="<<F<<'\t'<<"K2="<<K2<<'\t'<<"G="<<G<<std::endl;

    comp F3 = F/3.0 - F*(1.0/(1.0/K2 + F + G))*F;

    return F3; 


}










#endif