#ifndef K2FUNCTIONS_H
#define K2FUNCTIONS_H

#include "F2_functions.h"
#include<Eigen/Dense>

/* We code the K2 functions here, K2inv denotes the first definition that goes in to the
tilde_K2 function defined in 2.15 and 2.14 equations of the paper https://arxiv.org/pdf/2111.12734.pdf
*/

typedef std::complex<double> comp;

comp K2_inv_00( double eta_i, //this is the symmetry factor, eta=1 for piK (i=1), and eta=1/2 for KK (i=2)
                double scattering_length,
                comp sigma_i,
                double mj, 
                double mk,
                double epsilon_h    )
{
    double pi = std::acos(-1.0);
    comp A = eta_i/(8.0*pi*sigma_i);
    comp B = -1.0/scattering_length; 
    comp C = std::abs(q2psq_star(sigma_i, mj, mk));
    comp D = 1.0 - cutoff_function_1(sigma_i, mj, mk, epsilon_h);

    return A*(B + C*D);
}

comp tilde_K2_00(   double eta_i, 
                    double scattering_length,
                    std::vector<comp> p,
                    std::vector<comp> k,
                    comp sigma_i,
                    double mi,
                    double mj,
                    double mk,
                    double epsilon_h, 
                    double L    )
{
    double tolerance = 1.0e-10;
    comp px = p[0];
    comp py = p[1];
    comp pz = p[2];

    comp spec_p = std::sqrt(px*px + py*py + pz*pz);
    
    comp kx = k[0];
    comp ky = k[1];
    comp kz = k[2];

    comp spec_k = std::sqrt(kx*kx + ky*ky + kz*kz);

    double p1 = real(spec_p);
    double p2 = imag(spec_p);
    double k1 = real(spec_k);
    double k2 = imag(spec_k);

    if(abs(p1-k1)<tolerance && abs(p2-k2)<tolerance)
    {
        comp omega1 = omega_func(spec_k,mi);
        comp K2 = 1.0/K2_inv_00(eta_i, scattering_length, sigma_i, mj, mk, epsilon_h);
        return 2.0*omega1*L*L*L*K2;
    }
    else 
    {
        return 0.0;
    }
}


#endif