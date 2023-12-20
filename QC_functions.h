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
            std::vector<comp> total_P,
            double eta_i,
            double scattering_length,  
            double mi,
            double mj,
            double mk,
            double alpha,
            double epsilon_h,
            double L,
            int max_shell_num    )
{
    comp px = p[0];
    comp py = p[1];
    comp pz = p[2];
    comp kx = k[0];
    comp ky = k[1];
    comp kz = k[2];

    comp spec_p = std::sqrt(px*px + py*py + pz*pz);
    comp spec_k = std::sqrt(kx*kx + ky*ky + kz*kz);

    comp Px = total_P[0];
    comp Py = total_P[1];
    comp Pz = total_P[2];

    comp total_P_val = std::sqrt(Px*Px + Py*Py + Pz*Pz);

    comp sig_p = sigma(En, spec_p, mi, total_P_val);
    //std::cout<<"Here"<<std::endl;

    comp F = F2_i1(En, k, p, total_P, L, mi, mj, mk, alpha, epsilon_h, max_shell_num);
    comp K2 = tilde_K2_00(eta_i, scattering_length, p, k, sig_p, mi, mj, mk, epsilon_h, L );
    comp G = G_ij(En, p, k, total_P, mi, mj, mk, L, epsilon_h);

    //std::cout<<"E="<<En<<'\t'<<"F="<<F<<'\t'<<"K2="<<K2<<'\t'<<"G="<<G<<std::endl;

    comp F3 = F/3.0 - F*(1.0/(1.0/K2 + F + G))*F;

    return F3; 


}










#endif