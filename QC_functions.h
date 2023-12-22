#ifndef QCFUNCTIONS_H
#define QCFUNCTIONS_H
#include "F2_functions.h"
#include "K2_functions.h"
#include "G_functions.h"

typedef std::complex<double> comp;

//Basic Solvers that we will need in this program
void LinearSolver_3(	Eigen::MatrixXcd &A,
					Eigen::MatrixXcd &X,
					Eigen::MatrixXcd &B,
					double &relerr 			)
{
	X = A.partialPivLu().solve(B);

	relerr = (A*X - B).norm()/B.norm();
}

void LinearSolver_4(	Eigen::MatrixXcd &A,
					Eigen::MatrixXcd &X,
					Eigen::MatrixXcd &B,
					double &relerr 			)
{
	X = A.colPivHouseholderQr().solve(B);

	relerr = (A*X - B).norm()/B.norm();
}

//first the F3 for identical particles

comp F3_ID(   comp En,
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

/* Here we write down the function that creates the F3 matrix for identical 
particles, this definition follows strictly the formulation of https://arxiv.org/pdf/2111.12734.pdf
but will be tested to reproduce the results from fig 1 of https://arxiv.org/pdf/1803.04169.pdf   */
void F3_ID_mat( Eigen::MatrixXcd &F3mat,
                comp En, 
                std::vector< std::vector<comp> > p_config,
                std::vector< std::vector<comp> > k_config, 
                std::vector<comp> total_P, 
                double eta_i, 
                double scattering_length, 
                double mi,
                double mj, 
                double mk, 
                double alpha, 
                double epsilon_h, 
                double L, 
                int max_shell_num   )
{
    int size = p_config[0].size();

    Eigen::MatrixXcd F2_mat(size,size);
    Eigen::MatrixXcd K2inv_mat(size,size);
    Eigen::MatrixXcd G_mat(size,size);

    F2_i_mat( F2_mat, En, p_config, k_config, total_P, mi, mj, mk, L, alpha, epsilon_h, max_shell_num );
    K2inv_i_mat( K2inv_mat, eta_i, scattering_length, En, p_config, k_config, total_P, mi, mj, mk, epsilon_h, L );
    G_ij_mat( G_mat, En, p_config, k_config, total_P, mi, mj, mk, L, epsilon_h ); 

    Eigen::MatrixXcd temp_identity_mat(size,size);
    temp_identity_mat.setIdentity();

    Eigen::MatrixXcd H_mat = K2inv_mat + F2_mat + G_mat; 
    //H_mat = H_mat*10000;
    Eigen::MatrixXcd H_mat_inv(size,size);
    double relerror = 0.0;

    //LinearSolver_3(H_mat, H_mat_inv, temp_identity_mat, relerror);
    LinearSolver_4(H_mat, H_mat_inv, temp_identity_mat, relerror);
    std::cout << "Identity = " << temp_identity_mat << std::endl;
    //H_mat_inv = H_mat_inv/10000;

    //F3mat = F2_mat/3.0 - F2_mat*H_mat_inv*F2_mat;
    Eigen::MatrixXcd temp_F3_mat(size,size);
    Eigen::MatrixXcd temp_mat_A(size,size);
    temp_mat_A = H_mat_inv*F2_mat;//H_mat.inverse()*F2_mat;
    temp_F3_mat = F2_mat*temp_mat_A;
    F3mat = F2_mat/3.0 - temp_F3_mat; 

    char debug = 'y';
    if(debug=='y')
    {
        std::cout << "F2 mat = " << std::endl;
        std::cout << F2_mat*0.5*L*L*L << std::endl; 
        std::cout << "========================" << std::endl;
        std::cout << "G mat = " << std::endl; 
        std::cout << L*L*L*G_mat << std::endl;
        std::cout << "========================" << std::endl; 
        std::cout << "K2inv mat = " << std::endl; 
        std::cout << K2inv_mat << std::endl; 
        std::cout << "========================" << std::endl; 
        std::cout << "FHinvF mat = " << std::endl; 
        std::cout << temp_F3_mat << std::endl; 
        std::cout << "========================" << std::endl;
    }
}










#endif