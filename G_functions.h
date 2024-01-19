#ifndef GFUNCTIONS_H
#define GFUNCTIONS_H
#include "F2_functions.h"
#include "K2_functions.h"
#include<Eigen/Dense>

typedef std::complex<double> comp;


comp G_ij(  comp En, 
            std::vector<comp> p,
            std::vector<comp> k,
            std::vector<comp> total_P,
            double mi,
            double mj,
            double mk,
            double L,
            double epsilon_h       )
{
    comp px = p[0];
    comp py = p[1];
    comp pz = p[2];

    comp spec_p = std::sqrt(px*px + py*py + pz*pz);

    comp kx = k[0];
    comp ky = k[1];
    comp kz = k[2];

    comp spec_k = std::sqrt(kx*kx + ky*ky + kz*kz);

    comp Px = total_P[0];
    comp Py = total_P[1];
    comp Pz = total_P[2];

    comp total_P_val = std::sqrt(Px*Px + Py*Py + Pz*Pz);

    comp oneby2omegapLcube = 1.0/(2.0*omega_func(spec_p,mi)*L*L*L);
    comp oneby2omegakLcube = 1.0/(2.0*omega_func(spec_k,mj)*L*L*L);

    comp sig_i = sigma(En,spec_p,mi,total_P_val);
    comp sig_j = sigma(En,spec_k,mj,total_P_val);
    comp cutoff1 = cutoff_function_1(sig_i,mj,mk,epsilon_h);
    comp cutoff2 = cutoff_function_1(sig_j,mi,mk,epsilon_h);

    comp mom_P_p_k_x = Px - px - kx;
    comp mom_P_p_k_y = Py - py - ky;
    comp mom_P_p_k_z = Pz - pz - kz; 

    comp mom_P_p_k = std::sqrt(mom_P_p_k_x*mom_P_p_k_x + mom_P_p_k_y*mom_P_p_k_y + mom_P_p_k_z*mom_P_p_k_z);


    comp denom = (En - omega_func(spec_p,mi) - omega_func(spec_k,mj))*(En - omega_func(spec_p,mi) - omega_func(spec_k,mj))
                -mom_P_p_k*mom_P_p_k - mk*mk; 

    comp onebydenom = 1.0/denom; 

    //std::cout<<"sigi="<<sig_i<<'\t'<<"sigj="<<sig_j<<'\t'<<"cutoff1="<<cutoff1<<'\t'<<"cutoff2="<<cutoff2<<"denom="<<denom<<std::endl;

    return oneby2omegapLcube*cutoff1*cutoff2*onebydenom*oneby2omegakLcube;

}

void G_ij_mat(  Eigen::MatrixXcd &Gmat,
                comp En, 
                std::vector< std::vector<comp> > &p_config, 
                std::vector< std::vector<comp> > &k_config, 
                std::vector<comp> total_P, 
                double mi, 
                double mj, 
                double mk, 
                double L, 
                double epsilon_h    )
{
    int size1 = p_config[0].size(); 
    int size2 = k_config[0].size();
    std::cout<<"size1 for Gmat = "<<size1<<std::endl; 
    std::cout<<"size2 for Gmat = "<<size2<<std::endl; 

    for(int i=0; i<size1; ++i)
    {

        for(int j=0; j<size2; ++j)
        {
            comp px = p_config[0][i];
            comp py = p_config[1][i];
            comp pz = p_config[2][i];

            comp spec_p = std::sqrt(px*px + py*py + pz*pz);

            std::vector<comp> p(3);
            p[0] = px; 
            p[1] = py; 
            p[2] = pz; 

            comp kx = k_config[0][j];
            comp ky = k_config[1][j];
            comp kz = k_config[2][j]; 

            comp spec_k = std::sqrt(kx*kx + ky*ky + kz*kz);

            std::vector<comp> k(3);
            k[0] = kx; 
            k[1] = ky; 
            k[2] = kz; 

            comp Px = total_P[0];
            comp Py = total_P[1];
            comp Pz = total_P[2];

            comp Gij_val = G_ij(En, p, k, total_P, mi, mj, mk, L, epsilon_h);

            Gmat(i,j) = Gij_val; 
            std::cout<<i<<'\t'<<j<<'\t'<<Gmat(i,j)<<std::endl; 
        }
    }
}
 





#endif