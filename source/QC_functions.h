#ifndef QCFUNCTIONS_H
#define QCFUNCTIONS_H
#include "functions.h"
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
but will be tested to reproduce the results from fig 1 of https://arxiv.org/pdf/1803.04169.pdf
p.s. Here the F2 function is multiplied with 0.5 to match that of Raul's paper. This
0.5 comes from the usage of identical particles.   */
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

    //Here we change the matrices to the definition of 
    //Raul's paper

    F2_mat = 0.5*F2_mat*L*L*L;
    G_mat = L*L*L*G_mat;
    K2inv_mat = K2inv_mat*L*L*L; 

    Eigen::MatrixXcd temp_identity_mat(size,size);
    temp_identity_mat.setIdentity();

    Eigen::MatrixXcd H_mat = K2inv_mat + F2_mat + G_mat; 
    //H_mat = H_mat*10000;
    Eigen::MatrixXcd H_mat_inv(size,size);
    double relerror = 0.0;

    //LinearSolver_3(H_mat, H_mat_inv, temp_identity_mat, relerror);
    //LinearSolver_4(H_mat, H_mat_inv, temp_identity_mat, relerror);

    //std::cout << "Identity = " << temp_identity_mat << std::endl;
    //H_mat_inv = H_mat_inv/10000;

    //F3mat = F2_mat/3.0 - F2_mat*H_mat_inv*F2_mat;
    Eigen::MatrixXcd temp_F3_mat(size,size);
    Eigen::MatrixXcd temp_mat_A(size,size);
    //temp_mat_A = H_mat_inv*F2_mat;//H_mat.inverse()*F2_mat;
    //temp_F3_mat = F2_mat*temp_mat_A;
    
    F3mat = (1.0/(L*L*L))*(F2_mat/3.0 - F2_mat*H_mat.inverse()*F2_mat);//temp_F3_mat; 

    char debug = 'n';
    if(debug=='y')
    {
        std::cout << "F2 mat = " << std::endl;
        std::cout << F2_mat<< std::endl; 
        std::cout << "========================" << std::endl;
        std::cout << "G mat = " << std::endl; 
        std::cout << G_mat << std::endl;
        std::cout << "========================" << std::endl; 
        std::cout << "K2inv mat = " << std::endl; 
        std::cout << K2inv_mat/(L*L*L) << std::endl; 
        std::cout << "========================" << std::endl; 
        std::cout << "F3 mat = " << std::endl; 
        std::cout << F3mat << std::endl; 
        std::cout << "========================" << std::endl;
    }
}


/* Here we code the F3 for 2+1 system, how it is defined in 
the paper = https://arxiv.org/pdf/2105.12094.pdf    
The energies atEn, masses atmpi, atmK are all in lattice units,
the 1's put mK in i=1 and the 2body scattering happens between pi-K, 
so scattering length_1 = pi-K scattering length, eta_1 = 1
and scattering_length_2 = K-K scattering length with eta_2 = 1/2  */

void F2_mat_builder(    Eigen::MatrixXcd &Fmat,
                        Eigen::MatrixXcd &Fmat1,
                        Eigen::MatrixXcd &Fmat2,
                        int size1, 
                        int size2   )
{
    for(int i=0;i<size1+size2;++i)
    {
        for(int j=0;j<size1+size2;++j)
        {
            if(i<size1 && j<size1)
            {
                Fmat(i,j) = Fmat1(i,j);
            }
            else if(i<size1 && j>=size1)
            {
                Fmat(i,j) = 0.0;
            }
            else if(i>=size1 && j<size1)
            {
                Fmat(i,j) = 0.0;
            }
            else 
            {
                Fmat(i,j) = Fmat2(i,j);
            }
        }
    }
}
                                            

void F3_ND_2plus1_mat(  Eigen::MatrixXcd &F3mat,
                        comp En, 
                        std::vector< std::vector<comp> > p_config,
                        std::vector< std::vector<comp> > k_config, 
                        std::vector<comp> total_P, 
                        double eta_i_1,
                        double eta_i_2, 
                        double scattering_length_1,
                        double scattering_length_2,  
                        double m_pi,
                        double m_K,  
                        double alpha, 
                        double epsilon_h, 
                        double L,
                        double xi,  
                        int max_shell_num   )
{
    int size = p_config[0].size();

    int size1, size2;
    
    
    Eigen::MatrixXcd K2_mat_1(size,size);
    Eigen::MatrixXcd K2_mat_2(size,size);
    Eigen::MatrixXcd K2_mat(2*size,2*size);
    
    

    //for i = 1 
    double mi = m_K; 
    double mj = m_K; 
    double mk = m_pi; 

    std::vector< std::vector<comp> > p_config1(3,std::vector<comp> ());
    double config_tolerance = 1.0e-5;
    config_maker_1(p_config1, En, total_P, mi, mj, mk, L, xi, epsilon_h, config_tolerance );
    std::vector< std::vector<comp> > k_config1 = p_config1;

    size1 = p_config1[0].size(); 
    Eigen::MatrixXcd F2_mat_1(size1,size1);
    Eigen::MatrixXcd K2inv_mat_1(size1,size1);
    F2_i_mat( F2_mat_1, En, p_config1, k_config1, total_P, mi, mj, mk, L, alpha, epsilon_h, max_shell_num );
    K2inv_i_mat( K2inv_mat_1, eta_i_1, scattering_length_1, En, p_config1, k_config1, total_P, mi, mj, mk, epsilon_h, L );
    //K2_mat_1 = K2inv_mat_1.inverse();
    
    //for i = 2 
    mi = m_pi; 
    mj = m_K; 
    mk = m_K; 

    std::vector< std::vector<comp> > p_config2(3,std::vector<comp> ());
    config_maker_1(p_config2, En, total_P, mi, mj, mk, L, xi, epsilon_h, config_tolerance );
    std::vector< std::vector<comp> > k_config2 = p_config2;

     

    size2 = p_config2[0].size(); 
    //std::cout<<size2<<std::endl; 
    Eigen::MatrixXcd F2_mat_2(size2,size2);
    Eigen::MatrixXcd K2inv_mat_2(size2,size2);
    
    
    F2_i_mat( F2_mat_2, En, p_config2, k_config2, total_P, mi, mj, mk, L, alpha, epsilon_h, max_shell_num );
    K2inv_i_mat( K2inv_mat_2, eta_i_2, scattering_length_2, En, p_config2, k_config2, total_P, mi, mj, mk, epsilon_h, L );
    //K2_mat_2 = K2inv_mat_2.inverse();
    char debug1 = 'n';
    if(debug1=='y')
    {
        std::cout << "spec 1 size = " << std::endl;
        std::cout << p_config1[0].size()<< std::endl; 
        std::cout << "========================" << std::endl;
        std::cout << "spec 2 size = " << std::endl; 
        std::cout << p_config2[0].size() << std::endl;
        std::cout << "========================" << std::endl; 
        std::cout << "F2 mat 1 = " << std::endl; 
        std::cout << F2_mat_1 << std::endl; 
        std::cout << "========================" << std::endl; 
        std::cout << "F2 mat 2 = " << std::endl; 
        std::cout << F2_mat_2 << std::endl; 
        std::cout << "========================" << std::endl;
    }
    //for (i,j) = 1 1 and k = 2 
    mi = m_K; 
    mj = m_K; 
    mk = m_pi; 
    Eigen::MatrixXcd G_mat_11(size1,size1);
    G_ij_mat( G_mat_11, En, p_config1, k_config1, total_P, mi, mj, mk, L, epsilon_h ); 

    //for (i,j) = 1 2 and k = 1 
    mi = m_K; 
    mj = m_pi; 
    mk = m_K; 

    Eigen::MatrixXcd G_mat_12(size1,size2);
    G_ij_mat( G_mat_12, En, p_config1, k_config2, total_P, mi, mj, mk, L, epsilon_h ); 

    //for (i,j) = 2 1 and k = 1 
    mi = m_pi; 
    mj = m_K; 
    mk = m_K; 

    Eigen::MatrixXcd G_mat_21(size2,size1);
    G_ij_mat( G_mat_21, En, p_config2, k_config1, total_P, mi, mj, mk, L, epsilon_h ); 
    

    Eigen::MatrixXcd F2_mat(size1 + size2,size1 + size2);

    Eigen::MatrixXcd Filler0_12(size1,size2);
    Eigen::MatrixXcd Filler0_21(size2,size1);
    Eigen::MatrixXcd Filler0_22(size2,size2);

    Filler0_12 = Eigen::MatrixXcd::Zero(size1,size2);
    Filler0_21 = Eigen::MatrixXcd::Zero(size2,size1);
    Filler0_22 = Eigen::MatrixXcd::Zero(size2,size2);
    
    F2_mat <<   F2_mat_1,   Filler0_12,
                Filler0_21, F2_mat_2; 

    //F2_mat.topLeftCorner(size1,size1) = F2_mat_1;
    //F2_mat.topRightCorner(size1,size2) = Eigen::MatrixXcd::Zero(size2,size1);
    //F2_mat.bottomLeftCorner(size2,size1) = Eigen::MatrixXcd::Zero(size1,size2);
    //F2_mat.bottomRightCorner(size2,size2) = F2_mat_2;
    //K2_mat <<   K2_mat_1,                   Eigen::MatrixXcd::Zero(size,size),
    //            Eigen::MatrixXcd::Zero(size,size), 0.5*K2_mat_2;

    //F2_mat_builder(F2_mat,F2_mat_1, F2_mat_2, size1, size2);
    
    //std::cout<<"F2mat = "<<'\n'<<F2_mat<<std::endl;

    Eigen::MatrixXcd K2inv_mat(size1 + size2,size1 + size2);

    K2inv_mat   <<  K2inv_mat_1, Filler0_12,
                    Filler0_21,  2.0*K2inv_mat_2;

    Eigen::MatrixXcd G_mat(size1 + size2,size1 + size2);

    G_mat_12 = std::sqrt(2.0)*G_mat_12;
    G_mat_21 = std::sqrt(2.0)*G_mat_21;

    G_mat  <<   G_mat_11, G_mat_12,
                G_mat_21, Filler0_22;


    Eigen::MatrixXcd temp_identity_mat(size1 + size2,size1 + size2);
    temp_identity_mat.setIdentity();

    Eigen::MatrixXcd H_mat = K2inv_mat + F2_mat + G_mat; 
    //H_mat = H_mat*10000;
    Eigen::MatrixXcd H_mat_inv(size1 + size2,size1 + size2);
    double relerror = 0.0;

    //LinearSolver_3(H_mat, H_mat_inv, temp_identity_mat, relerror);
    LinearSolver_4(H_mat, H_mat_inv, temp_identity_mat, relerror);

    //std::cout << "Identity = " << temp_identity_mat << std::endl;
    //H_mat_inv = H_mat_inv/10000;

    //F3mat = F2_mat/3.0 - F2_mat*H_mat_inv*F2_mat;
    //Eigen::MatrixXcd temp_F3_mat(size,size);
    //Eigen::MatrixXcd temp_mat_A(size,size);
    //temp_mat_A = H_mat_inv*F2_mat;//H_mat.inverse()*F2_mat;
    //temp_F3_mat = F2_mat*temp_mat_A;
    
    //F3mat = (F2_mat/3.0 - F2_mat*H_mat.inverse()*F2_mat);//temp_F3_mat;

    F3mat = Eigen::MatrixXcd(size1+size2, size1+size2); 
    F3mat = (F2_mat/3.0 - F2_mat*H_mat_inv*F2_mat);//temp_F3_mat; 

    char debug = 'n';
    if(debug=='y')
    {
        std::cout << "F2 mat = " << std::endl;
        std::cout << F2_mat<< std::endl; 
        std::cout << "========================" << std::endl;
        std::cout << "G mat = " << std::endl; 
        std::cout << G_mat << std::endl;
        std::cout << "========================" << std::endl; 
        std::cout << "K2inv mat = " << std::endl; 
        std::cout << K2inv_mat << std::endl; 
        std::cout << "========================" << std::endl; 
        std::cout << "F3 mat = " << std::endl; 
        std::cout << F3mat << std::endl; 
        std::cout << "========================" << std::endl;
    }
}


/* Test F3 for 2+1 system*/
void test_F3_ND_2plus1_mat( Eigen::MatrixXcd &F3mat,
                            Eigen::MatrixXcd &F2mat,
                            Eigen::MatrixXcd &K2imat,
                            Eigen::MatrixXcd &Gmat, 
                            comp En, 
                            std::vector< std::vector<comp> > p_config,
                            std::vector< std::vector<comp> > k_config, 
                            std::vector<comp> total_P, 
                            double eta_i_1,
                            double eta_i_2, 
                            double scattering_length_1,
                            double scattering_length_2,  
                            double m_pi,
                            double m_K,  
                            double alpha, 
                            double epsilon_h, 
                            double L, 
                            double xi, 
                            int max_shell_num   )
{
    int size = p_config[0].size();

    int size1=0, size2=0;
    
    
    //Eigen::MatrixXcd K2_mat_1(size,size);
    //Eigen::MatrixXcd K2_mat_2(size,size);
    //Eigen::MatrixXcd K2_mat(2*size,2*size);
    
    

    //for i = 1 
    double mi = m_K; 
    double mj = m_K; 
    double mk = m_pi; 

    std::vector< std::vector<comp> > p_config1(3,std::vector<comp> ());
    double config_tolerance = 1.0e-5;

    config_maker_1(p_config1, En, total_P, mi, mj, mk, L, xi, epsilon_h, config_tolerance );
    //std::cout<<"size1 in QC = "<<size1<<std::endl;
    
    std::vector< std::vector<comp> > k_config1 = p_config1;

    size1 = p_config1[0].size(); 
    Eigen::MatrixXcd F2_mat_1(size1,size1);
    Eigen::MatrixXcd K2inv_mat_1(size1,size1);
    //F2_i_mat( F2_mat_1, En, p_config1, k_config1, total_P, mi, mj, mk, L, alpha, epsilon_h, max_shell_num );
    F2_i_mat_1( F2_mat_1, En, p_config1, k_config1, total_P, mi, mj, mk, L, xi, alpha, epsilon_h, max_shell_num );
    
    K2inv_i_mat( K2inv_mat_1, eta_i_1, scattering_length_1, En, p_config1, k_config1, total_P, mi, mj, mk, epsilon_h, L );
    //K2_mat_1 = K2inv_mat_1.inverse();
    
    //for i = 2 
    mi = m_pi; 
    mj = m_K; 
    mk = m_K; 

    std::vector< std::vector<comp> > p_config2(3,std::vector<comp> ());
    config_maker_1(p_config2, En, total_P, mi, mj, mk, L, xi, epsilon_h, config_tolerance );
    std::vector< std::vector<comp> > k_config2 = p_config2;

     

    size2 = p_config2[0].size(); 
    //std::cout<<size2<<std::endl; 
    Eigen::MatrixXcd F2_mat_2(size2,size2);
    Eigen::MatrixXcd K2inv_mat_2(size2,size2);

    //std::cout << "spec 1 size = " << size1 << '\t' << "spec 2 size = " << size2 << std::endl; 
    
    
    //F2_i_mat( F2_mat_2, En, p_config2, k_config2, total_P, mi, mj, mk, L, alpha, epsilon_h, max_shell_num );
    F2_i_mat_1( F2_mat_2, En, p_config2, k_config2, total_P, mi, mj, mk, L, xi, alpha, epsilon_h, max_shell_num );
    
    K2inv_i_mat( K2inv_mat_2, eta_i_2, scattering_length_2, En, p_config2, k_config2, total_P, mi, mj, mk, epsilon_h, L );
    //K2_mat_2 = K2inv_mat_2.inverse();
    char debug1 = 'n';
    if(debug1=='y')
    {
        std::cout << "spec 1 size = " << std::endl;
        std::cout << p_config1[0].size()<< std::endl; 
        std::cout << "========================" << std::endl;
        std::cout << "spec 2 size = " << std::endl; 
        std::cout << p_config2[0].size() << std::endl;
        std::cout << "========================" << std::endl; 
        std::cout << "F2 mat 1 = " << std::endl; 
        std::cout << F2_mat_1 << std::endl; 
        std::cout << "========================" << std::endl; 
        std::cout << "F2 mat 2 = " << std::endl; 
        std::cout << F2_mat_2 << std::endl; 
        std::cout << "========================" << std::endl;
    }
    //for (i,j) = 1 1 and k = 2 
    mi = m_K; 
    mj = m_K; 
    mk = m_pi; 
    Eigen::MatrixXcd G_mat_11(size1,size1);
    G_ij_mat( G_mat_11, En, p_config1, k_config1, total_P, mi, mj, mk, L, epsilon_h ); 

    //for (i,j) = 1 2 and k = 1 
    mi = m_K; 
    mj = m_pi; 
    mk = m_K; 

    Eigen::MatrixXcd G_mat_12(size1,size2);
    G_ij_mat( G_mat_12, En, p_config1, k_config2, total_P, mi, mj, mk, L, epsilon_h ); 

    //for (i,j) = 2 1 and k = 1 
    mi = m_pi; 
    mj = m_K; 
    mk = m_K; 

    Eigen::MatrixXcd G_mat_21(size2,size1);
    G_ij_mat( G_mat_21, En, p_config2, k_config1, total_P, mi, mj, mk, L, epsilon_h ); 
    

    Eigen::MatrixXcd F2_mat(size1 + size2,size1 + size2);

    Eigen::MatrixXcd Filler0_12(size1,size2);
    Eigen::MatrixXcd Filler0_21(size2,size1);
    Eigen::MatrixXcd Filler0_22(size2,size2);

    Filler0_12 = Eigen::MatrixXcd::Zero(size1,size2);
    Filler0_21 = Eigen::MatrixXcd::Zero(size2,size1);
    Filler0_22 = Eigen::MatrixXcd::Zero(size2,size2);
    
    F2_mat <<   F2_mat_1,   Filler0_12,
                Filler0_21, F2_mat_2; 

    //F2_mat.topLeftCorner(size1,size1) = F2_mat_1;
    //F2_mat.topRightCorner(size1,size2) = Eigen::MatrixXcd::Zero(size2,size1);
    //F2_mat.bottomLeftCorner(size2,size1) = Eigen::MatrixXcd::Zero(size1,size2);
    //F2_mat.bottomRightCorner(size2,size2) = F2_mat_2;
    //K2_mat <<   K2_mat_1,                   Eigen::MatrixXcd::Zero(size,size),
    //            Eigen::MatrixXcd::Zero(size,size), 0.5*K2_mat_2;

    //F2_mat_builder(F2_mat,F2_mat_1, F2_mat_2, size1, size2);
    
    //std::cout<<"F2mat = "<<'\n'<<F2_mat<<std::endl;

    Eigen::MatrixXcd K2inv_mat(size1 + size2,size1 + size2);

    K2inv_mat   <<  K2inv_mat_1, Filler0_12,
                    Filler0_21,  2.0*K2inv_mat_2;

    Eigen::MatrixXcd G_mat(size1 + size2,size1 + size2);

    G_mat_12 = std::sqrt(2.0)*G_mat_12;
    G_mat_21 = std::sqrt(2.0)*G_mat_21;

    G_mat  <<   G_mat_11, G_mat_12,
                G_mat_21, Filler0_22;


    Eigen::MatrixXcd temp_identity_mat(size1 + size2,size1 + size2);
    temp_identity_mat.setIdentity();

    Eigen::MatrixXcd H_mat = K2inv_mat + F2_mat + G_mat; 
    //H_mat = H_mat*10000;
    Eigen::MatrixXcd H_mat_inv(size1 + size2,size1 + size2);
    double relerror = 0.0;

    //LinearSolver_3(H_mat, H_mat_inv, temp_identity_mat, relerror);
    LinearSolver_4(H_mat, H_mat_inv, temp_identity_mat, relerror);

    //std::cout << "Identity = " << temp_identity_mat << std::endl;
    //H_mat_inv = H_mat_inv/10000;

    //F3mat = F2_mat/3.0 - F2_mat*H_mat_inv*F2_mat;
    //Eigen::MatrixXcd temp_F3_mat(size,size);
    //Eigen::MatrixXcd temp_mat_A(size,size);
    //temp_mat_A = H_mat_inv*F2_mat;//H_mat.inverse()*F2_mat;
    //temp_F3_mat = F2_mat*temp_mat_A;
    
    //F3mat = (F2_mat/3.0 - F2_mat*H_mat.inverse()*F2_mat);//temp_F3_mat;

    F3mat = Eigen::MatrixXcd(size1+size2, size1+size2); 
    F3mat = (F2_mat/3.0 - F2_mat*H_mat_inv*F2_mat);//temp_F3_mat; 

    F2mat = Eigen::MatrixXcd(size1+size2, size1+size2); 
    K2imat = Eigen::MatrixXcd(size1+size2, size1+size2); 
    Gmat = Eigen::MatrixXcd(size1+size2, size1+size2); 

    Gmat = G_mat; 
    F2mat = F2_mat; 
    K2imat = K2inv_mat; 
    

    char debug = 'n';
    if(debug=='y')
    {
        std::cout << "F2 mat = " << std::endl;
        std::cout << F2_mat<< std::endl; 
        std::cout << "========================" << std::endl;
        std::cout << "G mat = " << std::endl; 
        std::cout << G_mat << std::endl;
        std::cout << "========================" << std::endl; 
        std::cout << "K2inv mat = " << std::endl; 
        std::cout << K2inv_mat << std::endl; 
        std::cout << "========================" << std::endl; 
        std::cout << "F3 mat = " << std::endl; 
        std::cout << F3mat << std::endl; 
        std::cout << "========================" << std::endl;
    }
}


//This sends out the correct F3iso by taking 
//the appropriate eigen vector |1, 1/Sqrt(2)> 
void test_F3iso_ND_2plus1_mat( Eigen::MatrixXcd &F3mat,
                            comp &F3iso, 
                            Eigen::MatrixXcd &F2mat,
                            Eigen::MatrixXcd &K2imat,
                            Eigen::MatrixXcd &Gmat, 
                            comp En, 
                            std::vector< std::vector<comp> > p_config,
                            std::vector< std::vector<comp> > k_config, 
                            std::vector<comp> total_P, 
                            double eta_i_1,
                            double eta_i_2, 
                            double scattering_length_1,
                            double scattering_length_2,  
                            double m_pi,
                            double m_K,  
                            double alpha, 
                            double epsilon_h, 
                            double L, 
                            double xi, 
                            int max_shell_num   )
{
    int size = p_config[0].size();

    int size1=0, size2=0;
    
    
    //Eigen::MatrixXcd K2_mat_1(size,size);
    //Eigen::MatrixXcd K2_mat_2(size,size);
    //Eigen::MatrixXcd K2_mat(2*size,2*size);
    
    

    //for i = 1 
    double mi = m_K; 
    double mj = m_K; 
    double mk = m_pi; 

    std::vector< std::vector<comp> > p_config1(3,std::vector<comp> ());
    double config_tolerance = 1.0e-5;

    config_maker_1(p_config1, En, total_P, mi, mj, mk, L, xi, epsilon_h, config_tolerance );
    //std::cout<<"size1 in QC = "<<size1<<std::endl;
    
    std::vector< std::vector<comp> > k_config1 = p_config1;

    size1 = p_config1[0].size(); 
    Eigen::MatrixXcd F2_mat_1(size1,size1);
    Eigen::MatrixXcd K2inv_mat_1(size1,size1);
    //F2_i_mat( F2_mat_1, En, p_config1, k_config1, total_P, mi, mj, mk, L, alpha, epsilon_h, max_shell_num );
    F2_i_mat_1( F2_mat_1, En, p_config1, k_config1, total_P, mi, mj, mk, L, xi, alpha, epsilon_h, max_shell_num );
    
    K2inv_i_mat( K2inv_mat_1, eta_i_1, scattering_length_1, En, p_config1, k_config1, total_P, mi, mj, mk, epsilon_h, L );
    //K2_mat_1 = K2inv_mat_1.inverse();
    
    //for i = 2 
    mi = m_pi; 
    mj = m_K; 
    mk = m_K; 

    std::vector< std::vector<comp> > p_config2(3,std::vector<comp> ());
    config_maker_1(p_config2, En, total_P, mi, mj, mk, L, xi, epsilon_h, config_tolerance );
    std::vector< std::vector<comp> > k_config2 = p_config2;

     

    size2 = p_config2[0].size(); 
    //std::cout<<size2<<std::endl; 
    Eigen::MatrixXcd F2_mat_2(size2,size2);
    Eigen::MatrixXcd K2inv_mat_2(size2,size2);

    //std::cout << "spec 1 size = " << size1 << '\t' << "spec 2 size = " << size2 << std::endl; 
    
    
    //F2_i_mat( F2_mat_2, En, p_config2, k_config2, total_P, mi, mj, mk, L, alpha, epsilon_h, max_shell_num );
    F2_i_mat_1( F2_mat_2, En, p_config2, k_config2, total_P, mi, mj, mk, L, xi, alpha, epsilon_h, max_shell_num );
    
    K2inv_i_mat( K2inv_mat_2, eta_i_2, scattering_length_2, En, p_config2, k_config2, total_P, mi, mj, mk, epsilon_h, L );
    //K2_mat_2 = K2inv_mat_2.inverse();
    char debug1 = 'n';
    if(debug1=='y')
    {
        std::cout << "spec 1 size = " << std::endl;
        std::cout << p_config1[0].size()<< std::endl; 
        std::cout << "========================" << std::endl;
        std::cout << "spec 2 size = " << std::endl; 
        std::cout << p_config2[0].size() << std::endl;
        std::cout << "========================" << std::endl; 
        std::cout << "F2 mat 1 = " << std::endl; 
        std::cout << F2_mat_1 << std::endl; 
        std::cout << "========================" << std::endl; 
        std::cout << "F2 mat 2 = " << std::endl; 
        std::cout << F2_mat_2 << std::endl; 
        std::cout << "========================" << std::endl;
    }
    //for (i,j) = 1 1 and k = 2 
    mi = m_K; 
    mj = m_K; 
    mk = m_pi; 
    Eigen::MatrixXcd G_mat_11(size1,size1);
    G_ij_mat( G_mat_11, En, p_config1, k_config1, total_P, mi, mj, mk, L, epsilon_h ); 

    //for (i,j) = 1 2 and k = 1 
    mi = m_K; 
    mj = m_pi; 
    mk = m_K; 

    Eigen::MatrixXcd G_mat_12(size1,size2);
    G_ij_mat( G_mat_12, En, p_config1, k_config2, total_P, mi, mj, mk, L, epsilon_h ); 

    //for (i,j) = 2 1 and k = 1 
    mi = m_pi; 
    mj = m_K; 
    mk = m_K; 

    Eigen::MatrixXcd G_mat_21(size2,size1);
    G_ij_mat( G_mat_21, En, p_config2, k_config1, total_P, mi, mj, mk, L, epsilon_h ); 
    

    Eigen::MatrixXcd F2_mat(size1 + size2,size1 + size2);

    Eigen::MatrixXcd Filler0_12(size1,size2);
    Eigen::MatrixXcd Filler0_21(size2,size1);
    Eigen::MatrixXcd Filler0_22(size2,size2);

    Filler0_12 = Eigen::MatrixXcd::Zero(size1,size2);
    Filler0_21 = Eigen::MatrixXcd::Zero(size2,size1);
    Filler0_22 = Eigen::MatrixXcd::Zero(size2,size2);
    
    F2_mat <<   F2_mat_1,   Filler0_12,
                Filler0_21, F2_mat_2; 

    //F2_mat.topLeftCorner(size1,size1) = F2_mat_1;
    //F2_mat.topRightCorner(size1,size2) = Eigen::MatrixXcd::Zero(size2,size1);
    //F2_mat.bottomLeftCorner(size2,size1) = Eigen::MatrixXcd::Zero(size1,size2);
    //F2_mat.bottomRightCorner(size2,size2) = F2_mat_2;
    //K2_mat <<   K2_mat_1,                   Eigen::MatrixXcd::Zero(size,size),
    //            Eigen::MatrixXcd::Zero(size,size), 0.5*K2_mat_2;

    //F2_mat_builder(F2_mat,F2_mat_1, F2_mat_2, size1, size2);
    
    //std::cout<<"F2mat = "<<'\n'<<F2_mat<<std::endl;

    Eigen::MatrixXcd K2inv_mat(size1 + size2,size1 + size2);

    K2inv_mat   <<  K2inv_mat_1, Filler0_12,
                    Filler0_21,  2.0*K2inv_mat_2;

    Eigen::MatrixXcd G_mat(size1 + size2,size1 + size2);

    G_mat_12 = std::sqrt(2.0)*G_mat_12;
    G_mat_21 = std::sqrt(2.0)*G_mat_21;

    G_mat  <<   G_mat_11, G_mat_12,
                G_mat_21, Filler0_22;


    Eigen::MatrixXcd temp_identity_mat(size1 + size2,size1 + size2);
    temp_identity_mat.setIdentity();

    Eigen::MatrixXcd H_mat = K2inv_mat + F2_mat + G_mat; 
    //H_mat = H_mat*10000;
    Eigen::MatrixXcd H_mat_inv(size1 + size2,size1 + size2);
    double relerror = 0.0;

    //LinearSolver_3(H_mat, H_mat_inv, temp_identity_mat, relerror);
    LinearSolver_4(H_mat, H_mat_inv, temp_identity_mat, relerror);

    //std::cout << "Identity = " << temp_identity_mat << std::endl;
    //H_mat_inv = H_mat_inv/10000;

    //F3mat = F2_mat/3.0 - F2_mat*H_mat_inv*F2_mat;
    //Eigen::MatrixXcd temp_F3_mat(size,size);
    //Eigen::MatrixXcd temp_mat_A(size,size);
    //temp_mat_A = H_mat_inv*F2_mat;//H_mat.inverse()*F2_mat;
    //temp_F3_mat = F2_mat*temp_mat_A;
    
    //F3mat = (F2_mat/3.0 - F2_mat*H_mat.inverse()*F2_mat);//temp_F3_mat;

    F3mat = Eigen::MatrixXcd(size1+size2, size1+size2); 
    F3mat = (F2_mat/3.0 - F2_mat*H_mat_inv*F2_mat);//temp_F3_mat; 

    F2mat = Eigen::MatrixXcd(size1+size2, size1+size2); 
    K2imat = Eigen::MatrixXcd(size1+size2, size1+size2); 
    Gmat = Eigen::MatrixXcd(size1+size2, size1+size2); 

    Gmat = G_mat; 
    F2mat = F2_mat; 
    K2imat = K2inv_mat; 

    int eigvec_counter = 0; 
    Eigen::VectorXcd EigVec(size1 + size2); 
    for(int i=0;i<size1;++i)
    {
        EigVec(i) = 1.0;
        eigvec_counter += 1; 
    }
    for(int i=eigvec_counter ;i<(size1+size2); ++i)
    {
        EigVec(i) = 1.0/std::sqrt(2.0); 
    }
    
    F3iso = EigVec.transpose()*F3mat*EigVec; 

    char debug = 'n';
    if(debug=='y')
    {
        std::cout << "F2 mat = " << std::endl;
        std::cout << F2_mat<< std::endl; 
        std::cout << "========================" << std::endl;
        std::cout << "G mat = " << std::endl; 
        std::cout << G_mat << std::endl;
        std::cout << "========================" << std::endl; 
        std::cout << "K2inv mat = " << std::endl; 
        std::cout << K2inv_mat << std::endl; 
        std::cout << "========================" << std::endl; 
        std::cout << "F3 mat = " << std::endl; 
        std::cout << F3mat << std::endl; 
        std::cout << "========================" << std::endl;
    }
}

//This sends out the appropriate eigen vector |1, 1/Sqrt(2)> 
void test_F3iso_ND_2plus1_mat_with_normalization( Eigen::MatrixXcd &F3mat,
                            Eigen::VectorXcd &state_vec, 
                            Eigen::MatrixXcd &F2mat,
                            Eigen::MatrixXcd &K2imat,
                            Eigen::MatrixXcd &Gmat, 
                            Eigen::MatrixXcd &Hmatinv, 
                            comp En, 
                            std::vector< std::vector<comp> > p_config,
                            std::vector< std::vector<comp> > k_config, 
                            std::vector<comp> total_P, 
                            double eta_i_1,
                            double eta_i_2, 
                            double scattering_length_1,
                            double scattering_length_2,  
                            double m_pi,
                            double m_K,  
                            double alpha, 
                            double epsilon_h, 
                            double L, 
                            double xi, 
                            int max_shell_num   )
{
    int size = p_config[0].size();

    int size1=0, size2=0;
    
    
    //Eigen::MatrixXcd K2_mat_1(size,size);
    //Eigen::MatrixXcd K2_mat_2(size,size);
    //Eigen::MatrixXcd K2_mat(2*size,2*size);
    
    

    //for i = 1 
    double mi = m_K; 
    double mj = m_K; 
    double mk = m_pi; 

    std::vector< std::vector<comp> > p_config1(3,std::vector<comp> ());
    double config_tolerance = 1.0e-5;

    config_maker_1(p_config1, En, total_P, mi, mj, mk, L, xi, epsilon_h, config_tolerance );
    //std::cout<<"size1 in QC = "<<size1<<std::endl;
    
    std::vector< std::vector<comp> > k_config1 = p_config1;

    size1 = p_config1[0].size(); 
    Eigen::MatrixXcd F2_mat_1(size1,size1);
    Eigen::MatrixXcd K2inv_mat_1(size1,size1);
    //F2_i_mat( F2_mat_1, En, p_config1, k_config1, total_P, mi, mj, mk, L, alpha, epsilon_h, max_shell_num );
    F2_i_mat_1( F2_mat_1, En, p_config1, k_config1, total_P, mi, mj, mk, L, xi, alpha, epsilon_h, max_shell_num );
    
    K2inv_i_mat( K2inv_mat_1, eta_i_1, scattering_length_1, En, p_config1, k_config1, total_P, mi, mj, mk, epsilon_h, L );
    //K2_mat_1 = K2inv_mat_1.inverse();
    
    //for i = 2 
    mi = m_pi; 
    mj = m_K; 
    mk = m_K; 

    std::vector< std::vector<comp> > p_config2(3,std::vector<comp> ());
    config_maker_1(p_config2, En, total_P, mi, mj, mk, L, xi, epsilon_h, config_tolerance );
    std::vector< std::vector<comp> > k_config2 = p_config2;

     

    size2 = p_config2[0].size(); 
    //std::cout<<size2<<std::endl; 
    Eigen::MatrixXcd F2_mat_2(size2,size2);
    Eigen::MatrixXcd K2inv_mat_2(size2,size2);

    //std::cout << "spec 1 size = " << size1 << '\t' << "spec 2 size = " << size2 << std::endl; 
    
    
    //F2_i_mat( F2_mat_2, En, p_config2, k_config2, total_P, mi, mj, mk, L, alpha, epsilon_h, max_shell_num );
    F2_i_mat_1( F2_mat_2, En, p_config2, k_config2, total_P, mi, mj, mk, L, xi, alpha, epsilon_h, max_shell_num );
    
    K2inv_i_mat( K2inv_mat_2, eta_i_2, scattering_length_2, En, p_config2, k_config2, total_P, mi, mj, mk, epsilon_h, L );
    //K2_mat_2 = K2inv_mat_2.inverse();
    char debug1 = 'n';
    if(debug1=='y')
    {
        std::cout << "spec 1 size = " << std::endl;
        std::cout << p_config1[0].size()<< std::endl; 
        std::cout << "========================" << std::endl;
        std::cout << "spec 2 size = " << std::endl; 
        std::cout << p_config2[0].size() << std::endl;
        std::cout << "========================" << std::endl; 
        std::cout << "F2 mat 1 = " << std::endl; 
        std::cout << F2_mat_1 << std::endl; 
        std::cout << "========================" << std::endl; 
        std::cout << "F2 mat 2 = " << std::endl; 
        std::cout << F2_mat_2 << std::endl; 
        std::cout << "========================" << std::endl;
    }
    //for (i,j) = 1 1 and k = 2 
    mi = m_K; 
    mj = m_K; 
    mk = m_pi; 
    Eigen::MatrixXcd G_mat_11(size1,size1);
    G_ij_mat( G_mat_11, En, p_config1, k_config1, total_P, mi, mj, mk, L, epsilon_h ); 

    //for (i,j) = 1 2 and k = 1 
    mi = m_K; 
    mj = m_pi; 
    mk = m_K; 

    Eigen::MatrixXcd G_mat_12(size1,size2);
    G_ij_mat( G_mat_12, En, p_config1, k_config2, total_P, mi, mj, mk, L, epsilon_h ); 

    //for (i,j) = 2 1 and k = 1 
    mi = m_pi; 
    mj = m_K; 
    mk = m_K; 

    Eigen::MatrixXcd G_mat_21(size2,size1);
    G_ij_mat( G_mat_21, En, p_config2, k_config1, total_P, mi, mj, mk, L, epsilon_h ); 
    

    Eigen::MatrixXcd F2_mat(size1 + size2,size1 + size2);

    Eigen::MatrixXcd Filler0_12(size1,size2);
    Eigen::MatrixXcd Filler0_21(size2,size1);
    Eigen::MatrixXcd Filler0_22(size2,size2);

    Filler0_12 = Eigen::MatrixXcd::Zero(size1,size2);
    Filler0_21 = Eigen::MatrixXcd::Zero(size2,size1);
    Filler0_22 = Eigen::MatrixXcd::Zero(size2,size2);
    
    F2_mat <<   F2_mat_1,   Filler0_12,
                Filler0_21, F2_mat_2; 

    //F2_mat.topLeftCorner(size1,size1) = F2_mat_1;
    //F2_mat.topRightCorner(size1,size2) = Eigen::MatrixXcd::Zero(size2,size1);
    //F2_mat.bottomLeftCorner(size2,size1) = Eigen::MatrixXcd::Zero(size1,size2);
    //F2_mat.bottomRightCorner(size2,size2) = F2_mat_2;
    //K2_mat <<   K2_mat_1,                   Eigen::MatrixXcd::Zero(size,size),
    //            Eigen::MatrixXcd::Zero(size,size), 0.5*K2_mat_2;

    //F2_mat_builder(F2_mat,F2_mat_1, F2_mat_2, size1, size2);
    
    //std::cout<<"F2mat = "<<'\n'<<F2_mat<<std::endl;

    Eigen::MatrixXcd K2inv_mat(size1 + size2,size1 + size2);

    K2inv_mat   <<  K2inv_mat_1, Filler0_12,
                    Filler0_21,  2.0*K2inv_mat_2;

    Eigen::MatrixXcd G_mat(size1 + size2,size1 + size2);

    G_mat_12 = std::sqrt(2.0)*G_mat_12;
    G_mat_21 = std::sqrt(2.0)*G_mat_21;

    G_mat  <<   G_mat_11, G_mat_12,
                G_mat_21, Filler0_22;


    Eigen::MatrixXcd temp_identity_mat(size1 + size2,size1 + size2);
    temp_identity_mat.setIdentity();

    Eigen::MatrixXcd H_mat = K2inv_mat + F2_mat + G_mat; 
    //H_mat = H_mat*10000;
    Eigen::MatrixXcd H_mat_inv(size1 + size2,size1 + size2);
    double relerror = 0.0;

    //LinearSolver_3(H_mat, H_mat_inv, temp_identity_mat, relerror);
    LinearSolver_4(H_mat, H_mat_inv, temp_identity_mat, relerror);

    //std::cout << "Identity = " << temp_identity_mat << std::endl;
    //H_mat_inv = H_mat_inv/10000;

    //F3mat = F2_mat/3.0 - F2_mat*H_mat_inv*F2_mat;
    //Eigen::MatrixXcd temp_F3_mat(size,size);
    //Eigen::MatrixXcd temp_mat_A(size,size);
    //temp_mat_A = H_mat_inv*F2_mat;//H_mat.inverse()*F2_mat;
    //temp_F3_mat = F2_mat*temp_mat_A;
    
    //F3mat = (F2_mat/3.0 - F2_mat*H_mat.inverse()*F2_mat);//temp_F3_mat;

    F3mat = Eigen::MatrixXcd(size1+size2, size1+size2); 
    F3mat = (F2_mat/3.0 - F2_mat*H_mat_inv*F2_mat);//temp_F3_mat; 

    F2mat = Eigen::MatrixXcd(size1+size2, size1+size2); 
    K2imat = Eigen::MatrixXcd(size1+size2, size1+size2); 
    Gmat = Eigen::MatrixXcd(size1+size2, size1+size2); 
    Hmatinv = Eigen::MatrixXcd(size1+size2, size1+size2); 

    Gmat = G_mat; 
    F2mat = F2_mat; 
    K2imat = K2inv_mat; 
    Hmatinv = H_mat_inv; 

    int eigvec_counter = 0; 
    Eigen::VectorXcd EigVec(size1 + size2); 
    for(int i=0;i<size1;++i)
    {
        EigVec(i) = 1.0;
        eigvec_counter += 1; 
    }
    for(int i=eigvec_counter ;i<(size1+size2); ++i)
    {
        EigVec(i) = 1.0/std::sqrt(2.0); 
    }
    
    //F3iso = EigVec.transpose()*F3mat*EigVec; 

    state_vec = Eigen::VectorXcd(size1+size2); 

    state_vec = EigVec; 
    
    char debug = 'n';
    if(debug=='y')
    {
        std::cout << "F2 mat = " << std::endl;
        std::cout << F2_mat<< std::endl; 
        std::cout << "========================" << std::endl;
        std::cout << "G mat = " << std::endl; 
        std::cout << G_mat << std::endl;
        std::cout << "========================" << std::endl; 
        std::cout << "K2inv mat = " << std::endl; 
        std::cout << K2inv_mat << std::endl; 
        std::cout << "========================" << std::endl; 
        std::cout << "F3 mat = " << std::endl; 
        std::cout << F3mat << std::endl; 
        std::cout << "========================" << std::endl;
    }
}


//This one has a variable strength alpha multiplied to 
//K^{-1} term in F3, if twobody_alpha is large means weak 
//two body strength and small means larger twobody strength 

void test_F3iso_ND_2plus1_mat_with_normalization_twobody_var_strength_alpha( 
                            Eigen::MatrixXcd &F3mat,
                            Eigen::VectorXcd &state_vec, 
                            Eigen::MatrixXcd &F2mat,
                            Eigen::MatrixXcd &K2imat,
                            Eigen::MatrixXcd &Gmat, 
                            Eigen::MatrixXcd &Hmatinv, 
                            comp En, 
                            std::vector< std::vector<comp> > p_config,
                            std::vector< std::vector<comp> > k_config, 
                            std::vector<comp> total_P, 
                            double eta_i_1,
                            double eta_i_2, 
                            double scattering_length_1,
                            double scattering_length_2,  
                            double m_pi,
                            double m_K, 
                            double twobody_alpha,  
                            double alpha, 
                            double epsilon_h, 
                            double L, 
                            double xi, 
                            int max_shell_num   )
{
    int size = p_config[0].size();

    int size1=0, size2=0;
    
    
    //Eigen::MatrixXcd K2_mat_1(size,size);
    //Eigen::MatrixXcd K2_mat_2(size,size);
    //Eigen::MatrixXcd K2_mat(2*size,2*size);
    
    

    //for i = 1 
    double mi = m_K; 
    double mj = m_K; 
    double mk = m_pi; 

    std::vector< std::vector<comp> > p_config1(3,std::vector<comp> ());
    double config_tolerance = 1.0e-5;

    config_maker_1(p_config1, En, total_P, mi, mj, mk, L, xi, epsilon_h, config_tolerance );
    //std::cout<<"size1 in QC = "<<size1<<std::endl;
    
    std::vector< std::vector<comp> > k_config1 = p_config1;

    size1 = p_config1[0].size(); 
    Eigen::MatrixXcd F2_mat_1(size1,size1);
    Eigen::MatrixXcd K2inv_mat_1(size1,size1);
    //F2_i_mat( F2_mat_1, En, p_config1, k_config1, total_P, mi, mj, mk, L, alpha, epsilon_h, max_shell_num );
    F2_i_mat_1( F2_mat_1, En, p_config1, k_config1, total_P, mi, mj, mk, L, xi, alpha, epsilon_h, max_shell_num );
    
    K2inv_i_mat( K2inv_mat_1, eta_i_1, scattering_length_1, En, p_config1, k_config1, total_P, mi, mj, mk, epsilon_h, L );
    //K2_mat_1 = K2inv_mat_1.inverse();
    
    //for i = 2 
    mi = m_pi; 
    mj = m_K; 
    mk = m_K; 

    std::vector< std::vector<comp> > p_config2(3,std::vector<comp> ());
    config_maker_1(p_config2, En, total_P, mi, mj, mk, L, xi, epsilon_h, config_tolerance );
    std::vector< std::vector<comp> > k_config2 = p_config2;

     

    size2 = p_config2[0].size(); 
    //std::cout<<size2<<std::endl; 
    Eigen::MatrixXcd F2_mat_2(size2,size2);
    Eigen::MatrixXcd K2inv_mat_2(size2,size2);

    //std::cout << "spec 1 size = " << size1 << '\t' << "spec 2 size = " << size2 << std::endl; 
    
    
    //F2_i_mat( F2_mat_2, En, p_config2, k_config2, total_P, mi, mj, mk, L, alpha, epsilon_h, max_shell_num );
    F2_i_mat_1( F2_mat_2, En, p_config2, k_config2, total_P, mi, mj, mk, L, xi, alpha, epsilon_h, max_shell_num );
    
    K2inv_i_mat( K2inv_mat_2, eta_i_2, scattering_length_2, En, p_config2, k_config2, total_P, mi, mj, mk, epsilon_h, L );
    //K2_mat_2 = K2inv_mat_2.inverse();
    char debug1 = 'n';
    if(debug1=='y')
    {
        std::cout << "spec 1 size = " << std::endl;
        std::cout << p_config1[0].size()<< std::endl; 
        std::cout << "========================" << std::endl;
        std::cout << "spec 2 size = " << std::endl; 
        std::cout << p_config2[0].size() << std::endl;
        std::cout << "========================" << std::endl; 
        std::cout << "F2 mat 1 = " << std::endl; 
        std::cout << F2_mat_1 << std::endl; 
        std::cout << "========================" << std::endl; 
        std::cout << "F2 mat 2 = " << std::endl; 
        std::cout << F2_mat_2 << std::endl; 
        std::cout << "========================" << std::endl;
    }
    //for (i,j) = 1 1 and k = 2 
    mi = m_K; 
    mj = m_K; 
    mk = m_pi; 
    Eigen::MatrixXcd G_mat_11(size1,size1);
    G_ij_mat( G_mat_11, En, p_config1, k_config1, total_P, mi, mj, mk, L, epsilon_h ); 

    //for (i,j) = 1 2 and k = 1 
    mi = m_K; 
    mj = m_pi; 
    mk = m_K; 

    Eigen::MatrixXcd G_mat_12(size1,size2);
    G_ij_mat( G_mat_12, En, p_config1, k_config2, total_P, mi, mj, mk, L, epsilon_h ); 

    //for (i,j) = 2 1 and k = 1 
    mi = m_pi; 
    mj = m_K; 
    mk = m_K; 

    Eigen::MatrixXcd G_mat_21(size2,size1);
    G_ij_mat( G_mat_21, En, p_config2, k_config1, total_P, mi, mj, mk, L, epsilon_h ); 
    

    Eigen::MatrixXcd F2_mat(size1 + size2,size1 + size2);

    Eigen::MatrixXcd Filler0_12(size1,size2);
    Eigen::MatrixXcd Filler0_21(size2,size1);
    Eigen::MatrixXcd Filler0_22(size2,size2);

    Filler0_12 = Eigen::MatrixXcd::Zero(size1,size2);
    Filler0_21 = Eigen::MatrixXcd::Zero(size2,size1);
    Filler0_22 = Eigen::MatrixXcd::Zero(size2,size2);
    
    F2_mat <<   F2_mat_1,   Filler0_12,
                Filler0_21, F2_mat_2; 

    //F2_mat.topLeftCorner(size1,size1) = F2_mat_1;
    //F2_mat.topRightCorner(size1,size2) = Eigen::MatrixXcd::Zero(size2,size1);
    //F2_mat.bottomLeftCorner(size2,size1) = Eigen::MatrixXcd::Zero(size1,size2);
    //F2_mat.bottomRightCorner(size2,size2) = F2_mat_2;
    //K2_mat <<   K2_mat_1,                   Eigen::MatrixXcd::Zero(size,size),
    //            Eigen::MatrixXcd::Zero(size,size), 0.5*K2_mat_2;

    //F2_mat_builder(F2_mat,F2_mat_1, F2_mat_2, size1, size2);
    
    //std::cout<<"F2mat = "<<'\n'<<F2_mat<<std::endl;

    Eigen::MatrixXcd K2inv_mat(size1 + size2,size1 + size2);

    K2inv_mat   <<  K2inv_mat_1, Filler0_12,
                    Filler0_21,  2.0*K2inv_mat_2;

    K2inv_mat = twobody_alpha*K2inv_mat; 
    
    Eigen::MatrixXcd G_mat(size1 + size2,size1 + size2);

    G_mat_12 = std::sqrt(2.0)*G_mat_12;
    G_mat_21 = std::sqrt(2.0)*G_mat_21;

    G_mat  <<   G_mat_11, G_mat_12,
                G_mat_21, Filler0_22;


    Eigen::MatrixXcd temp_identity_mat(size1 + size2,size1 + size2);
    temp_identity_mat.setIdentity();

    Eigen::MatrixXcd H_mat = K2inv_mat + F2_mat + G_mat; 
    //H_mat = H_mat*10000;
    Eigen::MatrixXcd H_mat_inv(size1 + size2,size1 + size2);
    double relerror = 0.0;

    //LinearSolver_3(H_mat, H_mat_inv, temp_identity_mat, relerror);
    LinearSolver_4(H_mat, H_mat_inv, temp_identity_mat, relerror);

    //std::cout << "Identity = " << temp_identity_mat << std::endl;
    //H_mat_inv = H_mat_inv/10000;

    //F3mat = F2_mat/3.0 - F2_mat*H_mat_inv*F2_mat;
    //Eigen::MatrixXcd temp_F3_mat(size,size);
    //Eigen::MatrixXcd temp_mat_A(size,size);
    //temp_mat_A = H_mat_inv*F2_mat;//H_mat.inverse()*F2_mat;
    //temp_F3_mat = F2_mat*temp_mat_A;
    
    //F3mat = (F2_mat/3.0 - F2_mat*H_mat.inverse()*F2_mat);//temp_F3_mat;

    F3mat = Eigen::MatrixXcd(size1+size2, size1+size2); 
    F3mat = (F2_mat/3.0 - F2_mat*H_mat_inv*F2_mat);//temp_F3_mat; 

    F2mat = Eigen::MatrixXcd(size1+size2, size1+size2); 
    K2imat = Eigen::MatrixXcd(size1+size2, size1+size2); 
    Gmat = Eigen::MatrixXcd(size1+size2, size1+size2); 
    Hmatinv = Eigen::MatrixXcd(size1+size2, size1+size2); 

    Gmat = G_mat; 
    F2mat = F2_mat; 
    K2imat = K2inv_mat; 
    Hmatinv = H_mat_inv; 

    int eigvec_counter = 0; 
    Eigen::VectorXcd EigVec(size1 + size2); 
    for(int i=0;i<size1;++i)
    {
        EigVec(i) = 1.0;
        eigvec_counter += 1; 
    }
    for(int i=eigvec_counter ;i<(size1+size2); ++i)
    {
        EigVec(i) = 1.0/std::sqrt(2.0); 
    }
    
    //F3iso = EigVec.transpose()*F3mat*EigVec; 

    state_vec = Eigen::VectorXcd(size1+size2); 

    state_vec = EigVec; 
    
    char debug = 'n';
    if(debug=='y')
    {
        std::cout << "F2 mat = " << std::endl;
        std::cout << F2_mat<< std::endl; 
        std::cout << "========================" << std::endl;
        std::cout << "G mat = " << std::endl; 
        std::cout << G_mat << std::endl;
        std::cout << "========================" << std::endl; 
        std::cout << "K2inv mat = " << std::endl; 
        std::cout << K2inv_mat << std::endl; 
        std::cout << "========================" << std::endl; 
        std::cout << "F3 mat = " << std::endl; 
        std::cout << F3mat << std::endl; 
        std::cout << "========================" << std::endl;
    }
}



/* Here is the determinant of F3_inv, this if for testing 
the free levels vs L to check if they match with our previous 
results of free spectrum, a good check for consistency of the F3 matrix
it is a 'double' data type since we are only testing the real part of F3
and since we don't input any imaginary part anywhere, F3 is purely real */

double det_F3_ND_2plus1_mat(    comp En, 
                                std::vector< std::vector<comp> > p_config,
                                std::vector< std::vector<comp> > k_config, 
                                std::vector<comp> total_P, 
                                double eta_i_1,
                                double eta_i_2, 
                                double scattering_length_1,
                                double scattering_length_2,  
                                double m_pi,
                                double m_K,  
                                double alpha, 
                                double epsilon_h, 
                                double L, 
                                double xi, 
                                int max_shell_num   )
{
    int size = p_config[0].size();

    Eigen::MatrixXcd F2_mat_1(size,size);
    Eigen::MatrixXcd F2_mat_2(size,size);
    Eigen::MatrixXcd F2_mat(2*size,2*size);
    
    Eigen::MatrixXcd K2inv_mat_1(size,size);
    Eigen::MatrixXcd K2inv_mat_2(size,size);
    Eigen::MatrixXcd K2_mat_1(size,size);
    Eigen::MatrixXcd K2_mat_2(size,size);
    
    Eigen::MatrixXcd K2_mat(2*size,2*size);
    Eigen::MatrixXcd K2inv_mat(2*size,2*size);
    
    Eigen::MatrixXcd G_mat_11(size,size);
    Eigen::MatrixXcd G_mat_12(size,size);
    Eigen::MatrixXcd G_mat_21(size,size);
    Eigen::MatrixXcd G_mat(2*size,2*size); 

    Eigen::MatrixXcd F3mat(2*size,2*size); 

    //for i = 1 
    double mi = m_K; 
    double mj = m_K; 
    double mk = m_pi; 

    F2_i_mat( F2_mat_1, En, p_config, k_config, total_P, mi, mj, mk, L, alpha, epsilon_h, max_shell_num );
    K2inv_i_mat( K2inv_mat_1, eta_i_1, scattering_length_1, En, p_config, k_config, total_P, mi, mj, mk, epsilon_h, L );
    K2_mat_1 = K2inv_mat_1.inverse();

    //for i = 2 
    mi = m_pi; 
    mj = m_K; 
    mk = m_K; 

    F2_i_mat( F2_mat_2, En, p_config, k_config, total_P, mi, mj, mk, L, alpha, epsilon_h, max_shell_num );
    K2inv_i_mat( K2inv_mat_2, eta_i_2, scattering_length_2, En, p_config, k_config, total_P, mi, mj, mk, epsilon_h, L );
    K2_mat_2 = K2inv_mat_2.inverse();

    //for (i,j) = 1 1 and k = 2 
    mi = m_K; 
    mj = m_K; 
    mk = m_pi; 

    G_ij_mat( G_mat_11, En, p_config, k_config, total_P, mi, mj, mk, L, epsilon_h ); 

    //for (i,j) = 1 2 and k = 1 
    mi = m_K; 
    mj = m_pi; 
    mk = m_K; 

    G_ij_mat( G_mat_12, En, p_config, k_config, total_P, mi, mj, mk, L, epsilon_h ); 

    //for (i,j) = 2 1 and k = 1 
    mi = m_pi; 
    mj = m_K; 
    mk = m_K; 

    G_ij_mat( G_mat_21, En, p_config, k_config, total_P, mi, mj, mk, L, epsilon_h ); 

    F2_mat <<   F2_mat_1,                   Eigen::MatrixXcd::Zero(size,size),
                Eigen::MatrixXcd::Zero(size,size), F2_mat_2; 
    
    K2_mat <<   K2_mat_1,                   Eigen::MatrixXcd::Zero(size,size),
                Eigen::MatrixXcd::Zero(size,size), 0.5*K2_mat_2;

    K2inv_mat = K2_mat.inverse();

    G_mat  <<   G_mat_11,                  std::sqrt(2.0)*G_mat_12,
                std::sqrt(2.0)*G_mat_21,   Eigen::MatrixXcd::Zero(size,size);


    Eigen::MatrixXcd temp_identity_mat(size,size);
    temp_identity_mat.setIdentity();

    Eigen::MatrixXcd H_mat = K2inv_mat + F2_mat + G_mat; 
    //H_mat = H_mat*10000;
    Eigen::MatrixXcd H_mat_inv(2*size,2*size);
    double relerror = 0.0;

    //LinearSolver_3(H_mat, H_mat_inv, temp_identity_mat, relerror);
    //LinearSolver_4(H_mat, H_mat_inv, temp_identity_mat, relerror);

    //std::cout << "Identity = " << temp_identity_mat << std::endl;
    //H_mat_inv = H_mat_inv/10000;

    //F3mat = F2_mat/3.0 - F2_mat*H_mat_inv*F2_mat;
    //Eigen::MatrixXcd temp_F3_mat(size,size);
    //Eigen::MatrixXcd temp_mat_A(size,size);
    //temp_mat_A = H_mat_inv*F2_mat;//H_mat.inverse()*F2_mat;
    //temp_F3_mat = F2_mat*temp_mat_A;
    
    F3mat = (F2_mat/3.0 - F2_mat*H_mat.inverse()*F2_mat);//temp_F3_mat; 

    char debug = 'n';
    if(debug=='y')
    {
        std::cout << "F2 mat = " << std::endl;
        std::cout << F2_mat<< std::endl; 
        std::cout << "========================" << std::endl;
        std::cout << "G mat = " << std::endl; 
        std::cout << G_mat << std::endl;
        std::cout << "========================" << std::endl; 
        std::cout << "K2 mat = " << std::endl; 
        std::cout << K2_mat << std::endl; 
        std::cout << "========================" << std::endl; 
        std::cout << "F3 mat = " << std::endl; 
        std::cout << F3mat << std::endl; 
        std::cout << "========================" << std::endl;
    }

    char debug1 = 'n';
    if(debug1=='y')
    {
        int F2_flag = std::isnan(abs(F2_mat.determinant()));
        if(F2_flag==1) std::cout<<"F2 is nan"<<std::endl;
        int G_flag = std::isnan(abs(G_mat.determinant()));
        if(G_flag==1) std::cout<<"G is nan"<<std::endl;
        int K2_flag = std::isnan(abs(K2_mat.determinant()));
        if(K2_flag==1) std::cout<<"K2 is nan"<<std::endl;
        int F3_flag = std::isnan(abs(F3mat.determinant()));
        if(F3_flag==1)
        {
            std::cout<<"F3 is nan"<<std::endl;
        
    
            std::cout << "F2 mat = " << std::endl;
            std::cout << F2_mat<< std::endl; 
            std::cout << "========================" << std::endl;
            std::cout << "G mat = " << std::endl; 
            std::cout << G_mat << std::endl;
            std::cout << "========================" << std::endl; 
            std::cout << "K2 mat = " << std::endl; 
            std::cout << K2_mat << std::endl; 
            std::cout << "========================" << std::endl; 
            std::cout << "F3 mat = " << std::endl; 
            std::cout << F3mat << std::endl; 
            std::cout << "========================" << std::endl;

        } 
    }
    
    
    

    Eigen::MatrixXcd F3matinv = F3mat.inverse();
    double result = abs(F3matinv.determinant());
    return result; 
}

/* This function has become redundant and will be deleted*/
/* Here we test for additional poles 
that can come from the denominator of the F3 functions going 
to zero. We change the definition and only include the denominator
part */
void testF3_additionalpoles_1(  Eigen::MatrixXcd &F3matdenom,
                                Eigen::MatrixXcd &F2mat,
                                Eigen::MatrixXcd &K2imat,
                                Eigen::MatrixXcd &Gmat, 
                                comp En, 
                                std::vector< std::vector<comp> > p_config,
                                std::vector< std::vector<comp> > k_config, 
                                std::vector<comp> total_P, 
                                double eta_i_1,
                                double eta_i_2, 
                                double scattering_length_1,
                                double scattering_length_2,  
                                double m_pi,
                                double m_K,  
                                double alpha, 
                                double epsilon_h, 
                                double L, 
                                double xi, 
                                int max_shell_num   )
{
    int size = p_config[0].size();

    int size1=0, size2=0;
    
    
    //Eigen::MatrixXcd K2_mat_1(size,size);
    //Eigen::MatrixXcd K2_mat_2(size,size);
    //Eigen::MatrixXcd K2_mat(2*size,2*size);
    
    

    //for i = 1 
    double mi = m_K; 
    double mj = m_K; 
    double mk = m_pi; 

    std::vector< std::vector<comp> > p_config1(3,std::vector<comp> ());
    double config_tolerance = 1.0e-5;

    config_maker_1(p_config1, En, total_P, mi, mj, mk, L, xi, epsilon_h, config_tolerance );
    //std::cout<<"size1 in QC = "<<size1<<std::endl;
    
    std::vector< std::vector<comp> > k_config1 = p_config1;

    size1 = p_config1[0].size(); 
    Eigen::MatrixXcd F2_mat_1(size1,size1);
    Eigen::MatrixXcd K2inv_mat_1(size1,size1);
    //F2_i_mat( F2_mat_1, En, p_config1, k_config1, total_P, mi, mj, mk, L, alpha, epsilon_h, max_shell_num );
    F2_i_mat_1( F2_mat_1, En, p_config1, k_config1, total_P, mi, mj, mk, L, xi, alpha, epsilon_h, max_shell_num );
    
    K2inv_i_mat( K2inv_mat_1, eta_i_1, scattering_length_1, En, p_config1, k_config1, total_P, mi, mj, mk, epsilon_h, L );
    //K2_mat_1 = K2inv_mat_1.inverse();
    
    //for i = 2 
    mi = m_pi; 
    mj = m_K; 
    mk = m_K; 

    std::vector< std::vector<comp> > p_config2(3,std::vector<comp> ());
    config_maker_1(p_config2, En, total_P, mi, mj, mk, L, xi, epsilon_h, config_tolerance );
    std::vector< std::vector<comp> > k_config2 = p_config2;

     

    size2 = p_config2[0].size(); 
    //std::cout<<size2<<std::endl; 
    Eigen::MatrixXcd F2_mat_2(size2,size2);
    Eigen::MatrixXcd K2inv_mat_2(size2,size2);

    //std::cout << "spec 1 size = " << size1 << '\t' << "spec 2 size = " << size2 << std::endl; 
    
    
    //F2_i_mat( F2_mat_2, En, p_config2, k_config2, total_P, mi, mj, mk, L, alpha, epsilon_h, max_shell_num );
    F2_i_mat_1( F2_mat_2, En, p_config2, k_config2, total_P, mi, mj, mk, L, xi, alpha, epsilon_h, max_shell_num );
    
    K2inv_i_mat( K2inv_mat_2, eta_i_2, scattering_length_2, En, p_config2, k_config2, total_P, mi, mj, mk, epsilon_h, L );
    //K2_mat_2 = K2inv_mat_2.inverse();
    char debug1 = 'n';
    if(debug1=='y')
    {
        std::cout << "spec 1 size = " << std::endl;
        std::cout << p_config1[0].size()<< std::endl; 
        std::cout << "========================" << std::endl;
        std::cout << "spec 2 size = " << std::endl; 
        std::cout << p_config2[0].size() << std::endl;
        std::cout << "========================" << std::endl; 
        std::cout << "F2 mat 1 = " << std::endl; 
        std::cout << F2_mat_1 << std::endl; 
        std::cout << "========================" << std::endl; 
        std::cout << "F2 mat 2 = " << std::endl; 
        std::cout << F2_mat_2 << std::endl; 
        std::cout << "========================" << std::endl;
    }
    //for (i,j) = 1 1 and k = 2 
    mi = m_K; 
    mj = m_K; 
    mk = m_pi; 
    Eigen::MatrixXcd G_mat_11(size1,size1);
    G_ij_mat( G_mat_11, En, p_config1, k_config1, total_P, mi, mj, mk, L, epsilon_h ); 

    //for (i,j) = 1 2 and k = 1 
    mi = m_K; 
    mj = m_pi; 
    mk = m_K; 

    Eigen::MatrixXcd G_mat_12(size1,size2);
    G_ij_mat( G_mat_12, En, p_config1, k_config2, total_P, mi, mj, mk, L, epsilon_h ); 

    //for (i,j) = 2 1 and k = 1 
    mi = m_pi; 
    mj = m_K; 
    mk = m_K; 

    Eigen::MatrixXcd G_mat_21(size2,size1);
    G_ij_mat( G_mat_21, En, p_config2, k_config1, total_P, mi, mj, mk, L, epsilon_h ); 
    

    Eigen::MatrixXcd F2_mat(size1 + size2,size1 + size2);

    Eigen::MatrixXcd Filler0_12(size1,size2);
    Eigen::MatrixXcd Filler0_21(size2,size1);
    Eigen::MatrixXcd Filler0_22(size2,size2);

    Filler0_12 = Eigen::MatrixXcd::Zero(size1,size2);
    Filler0_21 = Eigen::MatrixXcd::Zero(size2,size1);
    Filler0_22 = Eigen::MatrixXcd::Zero(size2,size2);
    
    F2_mat <<   F2_mat_1,   Filler0_12,
                Filler0_21, F2_mat_2; 

    Eigen::MatrixXcd K2inv_mat(size1 + size2,size1 + size2);

    K2inv_mat   <<  K2inv_mat_1, Filler0_12,
                    Filler0_21,  2.0*K2inv_mat_2;

    Eigen::MatrixXcd G_mat(size1 + size2,size1 + size2);

    G_mat_12 = std::sqrt(2.0)*G_mat_12;
    G_mat_21 = std::sqrt(2.0)*G_mat_21;

    G_mat  <<   G_mat_11, G_mat_12,
                G_mat_21, Filler0_22;


    Eigen::MatrixXcd temp_identity_mat(size1 + size2,size1 + size2);
    temp_identity_mat.setIdentity();

    Eigen::MatrixXcd H_mat = K2inv_mat + F2_mat + G_mat; 
    Eigen::MatrixXcd H_mat_inv(size1 + size2,size1 + size2);
    double relerror = 0.0;

    //LinearSolver_3(H_mat, H_mat_inv, temp_identity_mat, relerror);
    //LinearSolver_4(H_mat, H_mat_inv, temp_identity_mat, relerror);

    
    //F3mat = (F2_mat/3.0 - F2_mat*H_mat.inverse()*F2_mat);//temp_F3_mat;

    F3matdenom = Eigen::MatrixXcd(size1+size2, size1+size2); 
    F3matdenom = temp_identity_mat + (K2inv_mat + G_mat)*F2_mat.inverse();

    F2mat = Eigen::MatrixXcd(size1+size2, size1+size2); 
    K2imat = Eigen::MatrixXcd(size1+size2, size1+size2); 
    Gmat = Eigen::MatrixXcd(size1+size2, size1+size2); 

    Gmat = G_mat; 
    F2mat = F2_mat; 
    K2imat = K2inv_mat; 
    

    char debug = 'n';
    if(debug=='y')
    {
        std::cout << "F2 mat = " << std::endl;
        std::cout << F2_mat<< std::endl; 
        std::cout << "========================" << std::endl;
        std::cout << "G mat = " << std::endl; 
        std::cout << G_mat << std::endl;
        std::cout << "========================" << std::endl; 
        std::cout << "K2inv mat = " << std::endl; 
        std::cout << K2inv_mat << std::endl; 
        std::cout << "========================" << std::endl; 
        std::cout << "F3 mat = " << std::endl; 
        std::cout << F3matdenom << std::endl; 
        std::cout << "========================" << std::endl;
    }
}

/* This function has become redundant and will be deleted*/

void testF3_additionalpoles_2(  Eigen::MatrixXcd &F3matdenom,
                                Eigen::MatrixXcd &F2mat,
                                Eigen::MatrixXcd &K2imat,
                                Eigen::MatrixXcd &Gmat, 
                                comp En, 
                                std::vector< std::vector<comp> > p_config,
                                std::vector< std::vector<comp> > k_config, 
                                std::vector<comp> total_P, 
                                double eta_i_1,
                                double eta_i_2, 
                                double scattering_length_1,
                                double scattering_length_2,  
                                double m_pi,
                                double m_K,  
                                double alpha, 
                                double epsilon_h, 
                                double L, 
                                double xi, 
                                int max_shell_num   )
{
    int size = p_config[0].size();

    int size1=0, size2=0;
    
    
    //Eigen::MatrixXcd K2_mat_1(size,size);
    //Eigen::MatrixXcd K2_mat_2(size,size);
    //Eigen::MatrixXcd K2_mat(2*size,2*size);
    
    

    //for i = 1 
    double mi = m_K; 
    double mj = m_K; 
    double mk = m_pi; 

    std::vector< std::vector<comp> > p_config1(3,std::vector<comp> ());
    double config_tolerance = 1.0e-5;

    config_maker_1(p_config1, En, total_P, mi, mj, mk, L, xi, epsilon_h, config_tolerance );
    //std::cout<<"size1 in QC = "<<size1<<std::endl;
    
    std::vector< std::vector<comp> > k_config1 = p_config1;

    size1 = p_config1[0].size(); 
    Eigen::MatrixXcd F2_mat_1(size1,size1);
    Eigen::MatrixXcd K2inv_mat_1(size1,size1);
    //F2_i_mat( F2_mat_1, En, p_config1, k_config1, total_P, mi, mj, mk, L, alpha, epsilon_h, max_shell_num );
    F2_i_mat_1( F2_mat_1, En, p_config1, k_config1, total_P, mi, mj, mk, L, xi, alpha, epsilon_h, max_shell_num );
    
    K2inv_i_mat( K2inv_mat_1, eta_i_1, scattering_length_1, En, p_config1, k_config1, total_P, mi, mj, mk, epsilon_h, L );
    //K2_mat_1 = K2inv_mat_1.inverse();
    
    //for i = 2 
    mi = m_pi; 
    mj = m_K; 
    mk = m_K; 

    std::vector< std::vector<comp> > p_config2(3,std::vector<comp> ());
    config_maker_1(p_config2, En, total_P, mi, mj, mk, L, xi, epsilon_h, config_tolerance );
    std::vector< std::vector<comp> > k_config2 = p_config2;

     

    size2 = p_config2[0].size(); 
    //std::cout<<size2<<std::endl; 
    Eigen::MatrixXcd F2_mat_2(size2,size2);
    Eigen::MatrixXcd K2inv_mat_2(size2,size2);

    //std::cout << "spec 1 size = " << size1 << '\t' << "spec 2 size = " << size2 << std::endl; 
    
    
    //F2_i_mat( F2_mat_2, En, p_config2, k_config2, total_P, mi, mj, mk, L, alpha, epsilon_h, max_shell_num );
    F2_i_mat_1( F2_mat_2, En, p_config2, k_config2, total_P, mi, mj, mk, L, xi, alpha, epsilon_h, max_shell_num );
    
    K2inv_i_mat( K2inv_mat_2, eta_i_2, scattering_length_2, En, p_config2, k_config2, total_P, mi, mj, mk, epsilon_h, L );
    //K2_mat_2 = K2inv_mat_2.inverse();
    char debug1 = 'n';
    if(debug1=='y')
    {
        std::cout << "spec 1 size = " << std::endl;
        std::cout << p_config1[0].size()<< std::endl; 
        std::cout << "========================" << std::endl;
        std::cout << "spec 2 size = " << std::endl; 
        std::cout << p_config2[0].size() << std::endl;
        std::cout << "========================" << std::endl; 
        std::cout << "F2 mat 1 = " << std::endl; 
        std::cout << F2_mat_1 << std::endl; 
        std::cout << "========================" << std::endl; 
        std::cout << "F2 mat 2 = " << std::endl; 
        std::cout << F2_mat_2 << std::endl; 
        std::cout << "========================" << std::endl;
    }
    //for (i,j) = 1 1 and k = 2 
    mi = m_K; 
    mj = m_K; 
    mk = m_pi; 
    Eigen::MatrixXcd G_mat_11(size1,size1);
    G_ij_mat( G_mat_11, En, p_config1, k_config1, total_P, mi, mj, mk, L, epsilon_h ); 

    //for (i,j) = 1 2 and k = 1 
    mi = m_K; 
    mj = m_pi; 
    mk = m_K; 

    Eigen::MatrixXcd G_mat_12(size1,size2);
    G_ij_mat( G_mat_12, En, p_config1, k_config2, total_P, mi, mj, mk, L, epsilon_h ); 

    //for (i,j) = 2 1 and k = 1 
    mi = m_pi; 
    mj = m_K; 
    mk = m_K; 

    Eigen::MatrixXcd G_mat_21(size2,size1);
    G_ij_mat( G_mat_21, En, p_config2, k_config1, total_P, mi, mj, mk, L, epsilon_h ); 
    

    Eigen::MatrixXcd F2_mat(size1 + size2,size1 + size2);

    Eigen::MatrixXcd Filler0_12(size1,size2);
    Eigen::MatrixXcd Filler0_21(size2,size1);
    Eigen::MatrixXcd Filler0_22(size2,size2);

    Filler0_12 = Eigen::MatrixXcd::Zero(size1,size2);
    Filler0_21 = Eigen::MatrixXcd::Zero(size2,size1);
    Filler0_22 = Eigen::MatrixXcd::Zero(size2,size2);
    
    F2_mat <<   F2_mat_1,   Filler0_12,
                Filler0_21, F2_mat_2; 

    Eigen::MatrixXcd K2inv_mat(size1 + size2,size1 + size2);

    K2inv_mat   <<  K2inv_mat_1, Filler0_12,
                    Filler0_21,  2.0*K2inv_mat_2;

    Eigen::MatrixXcd G_mat(size1 + size2,size1 + size2);

    G_mat_12 = std::sqrt(2.0)*G_mat_12;
    G_mat_21 = std::sqrt(2.0)*G_mat_21;

    G_mat  <<   G_mat_11, G_mat_12,
                G_mat_21, Filler0_22;


    Eigen::MatrixXcd temp_identity_mat(size1 + size2,size1 + size2);
    temp_identity_mat.setIdentity();

    Eigen::MatrixXcd H_mat = K2inv_mat + F2_mat + G_mat; 
    Eigen::MatrixXcd H_mat_inv(size1 + size2,size1 + size2);
    double relerror = 0.0;

    //LinearSolver_3(H_mat, H_mat_inv, temp_identity_mat, relerror);
    //LinearSolver_4(H_mat, H_mat_inv, temp_identity_mat, relerror);

    
    //F3mat = (F2_mat/3.0 - F2_mat*H_mat.inverse()*F2_mat);//temp_F3_mat;

    F3matdenom = Eigen::MatrixXcd(size1+size2, size1+size2); 
    F3matdenom = temp_identity_mat + K2inv_mat.inverse()*(G_mat + F2_mat);

    F2mat = Eigen::MatrixXcd(size1+size2, size1+size2); 
    K2imat = Eigen::MatrixXcd(size1+size2, size1+size2); 
    Gmat = Eigen::MatrixXcd(size1+size2, size1+size2); 

    Gmat = G_mat; 
    F2mat = F2_mat; 
    K2imat = K2inv_mat; 
    

    char debug = 'n';
    if(debug=='y')
    {
        std::cout << "F2 mat = " << std::endl;
        std::cout << F2_mat<< std::endl; 
        std::cout << "========================" << std::endl;
        std::cout << "G mat = " << std::endl; 
        std::cout << G_mat << std::endl;
        std::cout << "========================" << std::endl; 
        std::cout << "K2inv mat = " << std::endl; 
        std::cout << K2inv_mat << std::endl; 
        std::cout << "========================" << std::endl; 
        std::cout << "F3 mat = " << std::endl; 
        std::cout << F3matdenom << std::endl; 
        std::cout << "========================" << std::endl;
    }
}

/* This function has become redundant and will be deleted*/

void testF3_additionalpoles_3(  Eigen::MatrixXcd &F3matdenom,
                                Eigen::MatrixXcd &F2mat,
                                Eigen::MatrixXcd &K2imat,
                                Eigen::MatrixXcd &Gmat, 
                                comp En, 
                                std::vector< std::vector<comp> > p_config,
                                std::vector< std::vector<comp> > k_config, 
                                std::vector<comp> total_P, 
                                double eta_i_1,
                                double eta_i_2, 
                                double scattering_length_1,
                                double scattering_length_2,  
                                double m_pi,
                                double m_K,  
                                double alpha, 
                                double epsilon_h, 
                                double L, 
                                double xi, 
                                int max_shell_num   )
{
    int size = p_config[0].size();

    int size1=0, size2=0;
    
    
    //Eigen::MatrixXcd K2_mat_1(size,size);
    //Eigen::MatrixXcd K2_mat_2(size,size);
    //Eigen::MatrixXcd K2_mat(2*size,2*size);
    
    

    //for i = 1 
    double mi = m_K; 
    double mj = m_K; 
    double mk = m_pi; 

    std::vector< std::vector<comp> > p_config1(3,std::vector<comp> ());
    double config_tolerance = 1.0e-5;

    config_maker_1(p_config1, En, total_P, mi, mj, mk, L, xi, epsilon_h, config_tolerance );
    //std::cout<<"size1 in QC = "<<size1<<std::endl;
    
    std::vector< std::vector<comp> > k_config1 = p_config1;

    size1 = p_config1[0].size(); 
    Eigen::MatrixXcd F2_mat_1(size1,size1);
    Eigen::MatrixXcd K2inv_mat_1(size1,size1);
    //F2_i_mat( F2_mat_1, En, p_config1, k_config1, total_P, mi, mj, mk, L, alpha, epsilon_h, max_shell_num );
    F2_i_mat_1( F2_mat_1, En, p_config1, k_config1, total_P, mi, mj, mk, L, xi, alpha, epsilon_h, max_shell_num );
    
    K2inv_i_mat( K2inv_mat_1, eta_i_1, scattering_length_1, En, p_config1, k_config1, total_P, mi, mj, mk, epsilon_h, L );
    //K2_mat_1 = K2inv_mat_1.inverse();
    
    //for i = 2 
    mi = m_pi; 
    mj = m_K; 
    mk = m_K; 

    std::vector< std::vector<comp> > p_config2(3,std::vector<comp> ());
    config_maker_1(p_config2, En, total_P, mi, mj, mk, L, xi, epsilon_h, config_tolerance );
    std::vector< std::vector<comp> > k_config2 = p_config2;

     

    size2 = p_config2[0].size(); 
    //std::cout<<size2<<std::endl; 
    Eigen::MatrixXcd F2_mat_2(size2,size2);
    Eigen::MatrixXcd K2inv_mat_2(size2,size2);

    //std::cout << "spec 1 size = " << size1 << '\t' << "spec 2 size = " << size2 << std::endl; 
    
    
    //F2_i_mat( F2_mat_2, En, p_config2, k_config2, total_P, mi, mj, mk, L, alpha, epsilon_h, max_shell_num );
    F2_i_mat_1( F2_mat_2, En, p_config2, k_config2, total_P, mi, mj, mk, L, xi, alpha, epsilon_h, max_shell_num );
    
    K2inv_i_mat( K2inv_mat_2, eta_i_2, scattering_length_2, En, p_config2, k_config2, total_P, mi, mj, mk, epsilon_h, L );
    //K2_mat_2 = K2inv_mat_2.inverse();
    char debug1 = 'n';
    if(debug1=='y')
    {
        std::cout << "spec 1 size = " << std::endl;
        std::cout << p_config1[0].size()<< std::endl; 
        std::cout << "========================" << std::endl;
        std::cout << "spec 2 size = " << std::endl; 
        std::cout << p_config2[0].size() << std::endl;
        std::cout << "========================" << std::endl; 
        std::cout << "F2 mat 1 = " << std::endl; 
        std::cout << F2_mat_1 << std::endl; 
        std::cout << "========================" << std::endl; 
        std::cout << "F2 mat 2 = " << std::endl; 
        std::cout << F2_mat_2 << std::endl; 
        std::cout << "========================" << std::endl;
    }
    //for (i,j) = 1 1 and k = 2 
    mi = m_K; 
    mj = m_K; 
    mk = m_pi; 
    Eigen::MatrixXcd G_mat_11(size1,size1);
    G_ij_mat( G_mat_11, En, p_config1, k_config1, total_P, mi, mj, mk, L, epsilon_h ); 

    //for (i,j) = 1 2 and k = 1 
    mi = m_K; 
    mj = m_pi; 
    mk = m_K; 

    Eigen::MatrixXcd G_mat_12(size1,size2);
    G_ij_mat( G_mat_12, En, p_config1, k_config2, total_P, mi, mj, mk, L, epsilon_h ); 

    //for (i,j) = 2 1 and k = 1 
    mi = m_pi; 
    mj = m_K; 
    mk = m_K; 

    Eigen::MatrixXcd G_mat_21(size2,size1);
    G_ij_mat( G_mat_21, En, p_config2, k_config1, total_P, mi, mj, mk, L, epsilon_h ); 
    

    Eigen::MatrixXcd F2_mat(size1 + size2,size1 + size2);

    Eigen::MatrixXcd Filler0_12(size1,size2);
    Eigen::MatrixXcd Filler0_21(size2,size1);
    Eigen::MatrixXcd Filler0_22(size2,size2);

    Filler0_12 = Eigen::MatrixXcd::Zero(size1,size2);
    Filler0_21 = Eigen::MatrixXcd::Zero(size2,size1);
    Filler0_22 = Eigen::MatrixXcd::Zero(size2,size2);
    
    F2_mat <<   F2_mat_1,   Filler0_12,
                Filler0_21, F2_mat_2; 

    Eigen::MatrixXcd K2inv_mat(size1 + size2,size1 + size2);

    K2inv_mat   <<  K2inv_mat_1, Filler0_12,
                    Filler0_21,  2.0*K2inv_mat_2;

    Eigen::MatrixXcd G_mat(size1 + size2,size1 + size2);

    G_mat_12 = std::sqrt(2.0)*G_mat_12;
    G_mat_21 = std::sqrt(2.0)*G_mat_21;

    G_mat  <<   G_mat_11, G_mat_12,
                G_mat_21, Filler0_22;


    Eigen::MatrixXcd temp_identity_mat(size1 + size2,size1 + size2);
    temp_identity_mat.setIdentity();

    Eigen::MatrixXcd H_mat = K2inv_mat + F2_mat + G_mat; 
    Eigen::MatrixXcd H_mat_inv(size1 + size2,size1 + size2);
    double relerror = 0.0;

    //LinearSolver_3(H_mat, H_mat_inv, temp_identity_mat, relerror);
    //LinearSolver_4(H_mat, H_mat_inv, temp_identity_mat, relerror);

    
    //F3mat = (F2_mat/3.0 - F2_mat*H_mat.inverse()*F2_mat);//temp_F3_mat;

    F3matdenom = Eigen::MatrixXcd(size1+size2, size1+size2); 
    F3matdenom =  - H_mat.inverse();//temp_F3_mat;


    F2mat = Eigen::MatrixXcd(size1+size2, size1+size2); 
    K2imat = Eigen::MatrixXcd(size1+size2, size1+size2); 
    Gmat = Eigen::MatrixXcd(size1+size2, size1+size2); 

    Gmat = G_mat; 
    F2mat = F2_mat; 
    K2imat = K2inv_mat; 
    

    char debug = 'n';
    if(debug=='y')
    {
        std::cout << "F2 mat = " << std::endl;
        std::cout << F2_mat<< std::endl; 
        std::cout << "========================" << std::endl;
        std::cout << "G mat = " << std::endl; 
        std::cout << G_mat << std::endl;
        std::cout << "========================" << std::endl; 
        std::cout << "K2inv mat = " << std::endl; 
        std::cout << K2inv_mat << std::endl; 
        std::cout << "========================" << std::endl; 
        std::cout << "F3 mat = " << std::endl; 
        std::cout << F3matdenom << std::endl; 
        std::cout << "========================" << std::endl;
    }
}


/* This function return the F3inv_sum which is used to 
get an estimate of K3iso from lattice data. This returns only the 
real part of the F3inv_sum, this function is only used in the cpp 
called generate_K3iso.cpp, the executable created is used by the
python script named K3iso_maker.py  */

comp function_F3inv_ND_2plus1_mat( Eigen::MatrixXcd &F3mat,
                                Eigen::MatrixXcd &F2mat,
                                Eigen::MatrixXcd &K2imat,
                                Eigen::MatrixXcd &Gmat, 
                                comp En, 
                                std::vector< std::vector<comp> > p_config,
                                std::vector< std::vector<comp> > k_config, 
                                std::vector<comp> total_P, 
                                double eta_i_1,
                                double eta_i_2, 
                                double scattering_length_1,
                                double scattering_length_2,  
                                double m_pi,
                                double m_K,  
                                double alpha, 
                                double epsilon_h, 
                                double L, 
                                double xi, 
                                int max_shell_num   )
{
    int size = p_config[0].size();

    int size1=0, size2=0;
    
    
    //Eigen::MatrixXcd K2_mat_1(size,size);
    //Eigen::MatrixXcd K2_mat_2(size,size);
    //Eigen::MatrixXcd K2_mat(2*size,2*size);
    
    

    //for i = 1 
    double mi = m_K; 
    double mj = m_K; 
    double mk = m_pi; 

    std::vector< std::vector<comp> > p_config1(3,std::vector<comp> ());
    double config_tolerance = 1.0e-5;

    config_maker_1(p_config1, En, total_P, mi, mj, mk, L, xi, epsilon_h, config_tolerance );
    //std::cout<<"size1 in QC = "<<size1<<std::endl;
    
    std::vector< std::vector<comp> > k_config1 = p_config1;

    size1 = p_config1[0].size(); 
    Eigen::MatrixXcd F2_mat_1(size1,size1);
    Eigen::MatrixXcd K2inv_mat_1(size1,size1);
    //F2_i_mat( F2_mat_1, En, p_config1, k_config1, total_P, mi, mj, mk, L, alpha, epsilon_h, max_shell_num );
    F2_i_mat_1( F2_mat_1, En, p_config1, k_config1, total_P, mi, mj, mk, L, xi, alpha, epsilon_h, max_shell_num );
    
    K2inv_i_mat( K2inv_mat_1, eta_i_1, scattering_length_1, En, p_config1, k_config1, total_P, mi, mj, mk, epsilon_h, L );
    //K2_mat_1 = K2inv_mat_1.inverse();
    
    //for i = 2 
    mi = m_pi; 
    mj = m_K; 
    mk = m_K; 

    std::vector< std::vector<comp> > p_config2(3,std::vector<comp> ());
    config_maker_1(p_config2, En, total_P, mi, mj, mk, L, xi, epsilon_h, config_tolerance );
    std::vector< std::vector<comp> > k_config2 = p_config2;

     

    size2 = p_config2[0].size(); 
    //std::cout<<size2<<std::endl; 
    Eigen::MatrixXcd F2_mat_2(size2,size2);
    Eigen::MatrixXcd K2inv_mat_2(size2,size2);

    //std::cout << "spec 1 size = " << size1 << '\t' << "spec 2 size = " << size2 << std::endl; 
    
    
    //F2_i_mat( F2_mat_2, En, p_config2, k_config2, total_P, mi, mj, mk, L, alpha, epsilon_h, max_shell_num );
    F2_i_mat_1( F2_mat_2, En, p_config2, k_config2, total_P, mi, mj, mk, L, xi, alpha, epsilon_h, max_shell_num );
    
    K2inv_i_mat( K2inv_mat_2, eta_i_2, scattering_length_2, En, p_config2, k_config2, total_P, mi, mj, mk, epsilon_h, L );
    //K2_mat_2 = K2inv_mat_2.inverse();
    char debug1 = 'n';
    if(debug1=='y')
    {
        std::cout << "spec 1 size = " << std::endl;
        std::cout << p_config1[0].size()<< std::endl; 
        std::cout << "========================" << std::endl;
        std::cout << "spec 2 size = " << std::endl; 
        std::cout << p_config2[0].size() << std::endl;
        std::cout << "========================" << std::endl; 
        std::cout << "F2 mat 1 = " << std::endl; 
        std::cout << F2_mat_1 << std::endl; 
        std::cout << "========================" << std::endl; 
        std::cout << "F2 mat 2 = " << std::endl; 
        std::cout << F2_mat_2 << std::endl; 
        std::cout << "========================" << std::endl;
    }
    //for (i,j) = 1 1 and k = 2 
    mi = m_K; 
    mj = m_K; 
    mk = m_pi; 
    Eigen::MatrixXcd G_mat_11(size1,size1);
    G_ij_mat( G_mat_11, En, p_config1, k_config1, total_P, mi, mj, mk, L, epsilon_h ); 

    //for (i,j) = 1 2 and k = 1 
    mi = m_K; 
    mj = m_pi; 
    mk = m_K; 

    Eigen::MatrixXcd G_mat_12(size1,size2);
    G_ij_mat( G_mat_12, En, p_config1, k_config2, total_P, mi, mj, mk, L, epsilon_h ); 

    //for (i,j) = 2 1 and k = 1 
    mi = m_pi; 
    mj = m_K; 
    mk = m_K; 

    Eigen::MatrixXcd G_mat_21(size2,size1);
    G_ij_mat( G_mat_21, En, p_config2, k_config1, total_P, mi, mj, mk, L, epsilon_h ); 
    

    Eigen::MatrixXcd F2_mat(size1 + size2,size1 + size2);

    Eigen::MatrixXcd Filler0_12(size1,size2);
    Eigen::MatrixXcd Filler0_21(size2,size1);
    Eigen::MatrixXcd Filler0_22(size2,size2);

    Filler0_12 = Eigen::MatrixXcd::Zero(size1,size2);
    Filler0_21 = Eigen::MatrixXcd::Zero(size2,size1);
    Filler0_22 = Eigen::MatrixXcd::Zero(size2,size2);
    
    F2_mat <<   F2_mat_1,   Filler0_12,
                Filler0_21, F2_mat_2; 

    //F2_mat.topLeftCorner(size1,size1) = F2_mat_1;
    //F2_mat.topRightCorner(size1,size2) = Eigen::MatrixXcd::Zero(size2,size1);
    //F2_mat.bottomLeftCorner(size2,size1) = Eigen::MatrixXcd::Zero(size1,size2);
    //F2_mat.bottomRightCorner(size2,size2) = F2_mat_2;
    //K2_mat <<   K2_mat_1,                   Eigen::MatrixXcd::Zero(size,size),
    //            Eigen::MatrixXcd::Zero(size,size), 0.5*K2_mat_2;

    //F2_mat_builder(F2_mat,F2_mat_1, F2_mat_2, size1, size2);
    
    //std::cout<<"F2mat = "<<'\n'<<F2_mat<<std::endl;

    Eigen::MatrixXcd K2inv_mat(size1 + size2,size1 + size2);

    K2inv_mat   <<  K2inv_mat_1, Filler0_12,
                    Filler0_21,  2.0*K2inv_mat_2;

    Eigen::MatrixXcd G_mat(size1 + size2,size1 + size2);

    G_mat_12 = std::sqrt(2.0)*G_mat_12;
    G_mat_21 = std::sqrt(2.0)*G_mat_21;

    G_mat  <<   G_mat_11, G_mat_12,
                G_mat_21, Filler0_22;


    Eigen::MatrixXcd temp_identity_mat(size1 + size2,size1 + size2);
    temp_identity_mat.setIdentity();

    Eigen::MatrixXcd H_mat = K2inv_mat + F2_mat + G_mat; 
    //H_mat = H_mat*10000;
    Eigen::MatrixXcd H_mat_inv(size1 + size2,size1 + size2);
    double relerror = 0.0;

    //LinearSolver_3(H_mat, H_mat_inv, temp_identity_mat, relerror);
    LinearSolver_4(H_mat, H_mat_inv, temp_identity_mat, relerror);

    //std::cout << "Identity = " << temp_identity_mat << std::endl;
    //H_mat_inv = H_mat_inv/10000;

    //F3mat = F2_mat/3.0 - F2_mat*H_mat_inv*F2_mat;
    //Eigen::MatrixXcd temp_F3_mat(size,size);
    //Eigen::MatrixXcd temp_mat_A(size,size);
    //temp_mat_A = H_mat_inv*F2_mat;//H_mat.inverse()*F2_mat;
    //temp_F3_mat = F2_mat*temp_mat_A;
    
    //F3mat = (F2_mat/3.0 - F2_mat*H_mat.inverse()*F2_mat);//temp_F3_mat;

    F3mat = Eigen::MatrixXcd(size1+size2, size1+size2); 
    F3mat = (F2_mat/3.0 - F2_mat*H_mat_inv*F2_mat);//temp_F3_mat; 

    F2mat = Eigen::MatrixXcd(size1+size2, size1+size2); 
    K2imat = Eigen::MatrixXcd(size1+size2, size1+size2); 
    Gmat = Eigen::MatrixXcd(size1+size2, size1+size2); 

    Gmat = G_mat; 
    F2mat = F2_mat; 
    K2imat = K2inv_mat; 

    Eigen::MatrixXcd F3matinv = F3mat.inverse(); 
    

    char debug = 'n';
    if(debug=='y')
    {
        std::cout << "F2 mat = " << std::endl;
        std::cout << F2_mat<< std::endl; 
        std::cout << "========================" << std::endl;
        std::cout << "G mat = " << std::endl; 
        std::cout << G_mat << std::endl;
        std::cout << "========================" << std::endl; 
        std::cout << "K2inv mat = " << std::endl; 
        std::cout << K2inv_mat << std::endl; 
        std::cout << "========================" << std::endl; 
        std::cout << "F3 mat = " << std::endl; 
        std::cout << F3mat << std::endl; 
        std::cout << "========================" << std::endl;
    }
    

    return real(F3matinv.sum()); 
}

comp function_F3_ND_2plus1_mat( Eigen::MatrixXcd &F3mat,
                                Eigen::MatrixXcd &F2mat,
                                Eigen::MatrixXcd &K2imat,
                                Eigen::MatrixXcd &Gmat, 
                                comp En, 
                                std::vector< std::vector<comp> > p_config,
                                std::vector< std::vector<comp> > k_config, 
                                std::vector<comp> total_P, 
                                double eta_i_1,
                                double eta_i_2, 
                                double scattering_length_1,
                                double scattering_length_2,  
                                double m_pi,
                                double m_K,  
                                double alpha, 
                                double epsilon_h, 
                                double L, 
                                double xi, 
                                int max_shell_num   )
{
    int size = p_config[0].size();

    int size1=0, size2=0;
    
    
    //Eigen::MatrixXcd K2_mat_1(size,size);
    //Eigen::MatrixXcd K2_mat_2(size,size);
    //Eigen::MatrixXcd K2_mat(2*size,2*size);
    
    

    //for i = 1 
    double mi = m_K; 
    double mj = m_K; 
    double mk = m_pi; 

    std::vector< std::vector<comp> > p_config1(3,std::vector<comp> ());
    double config_tolerance = 1.0e-5;

    config_maker_1(p_config1, En, total_P, mi, mj, mk, L, xi, epsilon_h, config_tolerance );
    //std::cout<<"size1 in QC = "<<size1<<std::endl;
    
    std::vector< std::vector<comp> > k_config1 = p_config1;

    size1 = p_config1[0].size(); 
    Eigen::MatrixXcd F2_mat_1(size1,size1);
    Eigen::MatrixXcd K2inv_mat_1(size1,size1);
    //F2_i_mat( F2_mat_1, En, p_config1, k_config1, total_P, mi, mj, mk, L, alpha, epsilon_h, max_shell_num );
    F2_i_mat_1( F2_mat_1, En, p_config1, k_config1, total_P, mi, mj, mk, L, xi, alpha, epsilon_h, max_shell_num );
    
    K2inv_i_mat( K2inv_mat_1, eta_i_1, scattering_length_1, En, p_config1, k_config1, total_P, mi, mj, mk, epsilon_h, L );
    //K2_mat_1 = K2inv_mat_1.inverse();
    
    //for i = 2 
    mi = m_pi; 
    mj = m_K; 
    mk = m_K; 

    std::vector< std::vector<comp> > p_config2(3,std::vector<comp> ());
    config_maker_1(p_config2, En, total_P, mi, mj, mk, L, xi, epsilon_h, config_tolerance );
    std::vector< std::vector<comp> > k_config2 = p_config2;

     

    size2 = p_config2[0].size(); 
    //std::cout<<size2<<std::endl; 
    Eigen::MatrixXcd F2_mat_2(size2,size2);
    Eigen::MatrixXcd K2inv_mat_2(size2,size2);

    //std::cout << "spec 1 size = " << size1 << '\t' << "spec 2 size = " << size2 << std::endl; 
    
    
    //F2_i_mat( F2_mat_2, En, p_config2, k_config2, total_P, mi, mj, mk, L, alpha, epsilon_h, max_shell_num );
    F2_i_mat_1( F2_mat_2, En, p_config2, k_config2, total_P, mi, mj, mk, L, xi, alpha, epsilon_h, max_shell_num );
    
    K2inv_i_mat( K2inv_mat_2, eta_i_2, scattering_length_2, En, p_config2, k_config2, total_P, mi, mj, mk, epsilon_h, L );
    //K2_mat_2 = K2inv_mat_2.inverse();
    char debug1 = 'n';
    if(debug1=='y')
    {
        std::cout << "spec 1 size = " << std::endl;
        std::cout << p_config1[0].size()<< std::endl; 
        std::cout << "========================" << std::endl;
        std::cout << "spec 2 size = " << std::endl; 
        std::cout << p_config2[0].size() << std::endl;
        std::cout << "========================" << std::endl; 
        std::cout << "F2 mat 1 = " << std::endl; 
        std::cout << F2_mat_1 << std::endl; 
        std::cout << "========================" << std::endl; 
        std::cout << "F2 mat 2 = " << std::endl; 
        std::cout << F2_mat_2 << std::endl; 
        std::cout << "========================" << std::endl;
    }
    //for (i,j) = 1 1 and k = 2 
    mi = m_K; 
    mj = m_K; 
    mk = m_pi; 
    Eigen::MatrixXcd G_mat_11(size1,size1);
    G_ij_mat( G_mat_11, En, p_config1, k_config1, total_P, mi, mj, mk, L, epsilon_h ); 

    //for (i,j) = 1 2 and k = 1 
    mi = m_K; 
    mj = m_pi; 
    mk = m_K; 

    Eigen::MatrixXcd G_mat_12(size1,size2);
    G_ij_mat( G_mat_12, En, p_config1, k_config2, total_P, mi, mj, mk, L, epsilon_h ); 

    //for (i,j) = 2 1 and k = 1 
    mi = m_pi; 
    mj = m_K; 
    mk = m_K; 

    Eigen::MatrixXcd G_mat_21(size2,size1);
    G_ij_mat( G_mat_21, En, p_config2, k_config1, total_P, mi, mj, mk, L, epsilon_h ); 
    

    Eigen::MatrixXcd F2_mat(size1 + size2,size1 + size2);

    Eigen::MatrixXcd Filler0_12(size1,size2);
    Eigen::MatrixXcd Filler0_21(size2,size1);
    Eigen::MatrixXcd Filler0_22(size2,size2);

    Filler0_12 = Eigen::MatrixXcd::Zero(size1,size2);
    Filler0_21 = Eigen::MatrixXcd::Zero(size2,size1);
    Filler0_22 = Eigen::MatrixXcd::Zero(size2,size2);
    
    F2_mat <<   F2_mat_1,   Filler0_12,
                Filler0_21, F2_mat_2; 

    //F2_mat.topLeftCorner(size1,size1) = F2_mat_1;
    //F2_mat.topRightCorner(size1,size2) = Eigen::MatrixXcd::Zero(size2,size1);
    //F2_mat.bottomLeftCorner(size2,size1) = Eigen::MatrixXcd::Zero(size1,size2);
    //F2_mat.bottomRightCorner(size2,size2) = F2_mat_2;
    //K2_mat <<   K2_mat_1,                   Eigen::MatrixXcd::Zero(size,size),
    //            Eigen::MatrixXcd::Zero(size,size), 0.5*K2_mat_2;

    //F2_mat_builder(F2_mat,F2_mat_1, F2_mat_2, size1, size2);
    
    //std::cout<<"F2mat = "<<'\n'<<F2_mat<<std::endl;

    Eigen::MatrixXcd K2inv_mat(size1 + size2,size1 + size2);

    K2inv_mat   <<  K2inv_mat_1, Filler0_12,
                    Filler0_21,  2.0*K2inv_mat_2;

    Eigen::MatrixXcd G_mat(size1 + size2,size1 + size2);

    G_mat_12 = std::sqrt(2.0)*G_mat_12;
    G_mat_21 = std::sqrt(2.0)*G_mat_21;

    G_mat  <<   G_mat_11, G_mat_12,
                G_mat_21, Filler0_22;


    Eigen::MatrixXcd temp_identity_mat(size1 + size2,size1 + size2);
    temp_identity_mat.setIdentity();

    Eigen::MatrixXcd H_mat = K2inv_mat + F2_mat + G_mat; 
    //H_mat = H_mat*10000;
    Eigen::MatrixXcd H_mat_inv(size1 + size2,size1 + size2);
    double relerror = 0.0;

    //LinearSolver_3(H_mat, H_mat_inv, temp_identity_mat, relerror);
    LinearSolver_4(H_mat, H_mat_inv, temp_identity_mat, relerror);

    //std::cout << "Identity = " << temp_identity_mat << std::endl;
    //H_mat_inv = H_mat_inv/10000;

    //F3mat = F2_mat/3.0 - F2_mat*H_mat_inv*F2_mat;
    //Eigen::MatrixXcd temp_F3_mat(size,size);
    //Eigen::MatrixXcd temp_mat_A(size,size);
    //temp_mat_A = H_mat_inv*F2_mat;//H_mat.inverse()*F2_mat;
    //temp_F3_mat = F2_mat*temp_mat_A;
    
    //F3mat = (F2_mat/3.0 - F2_mat*H_mat.inverse()*F2_mat);//temp_F3_mat;

    F3mat = Eigen::MatrixXcd(size1+size2, size1+size2); 
    F3mat = (F2_mat/3.0 - F2_mat*H_mat_inv*F2_mat);//temp_F3_mat; 

    F2mat = Eigen::MatrixXcd(size1+size2, size1+size2); 
    K2imat = Eigen::MatrixXcd(size1+size2, size1+size2); 
    Gmat = Eigen::MatrixXcd(size1+size2, size1+size2); 

    Gmat = G_mat; 
    F2mat = F2_mat; 
    K2imat = K2inv_mat; 

    //Eigen::MatrixXcd F3matinv = F3mat.inverse(); 
    

    char debug = 'n';
    if(debug=='y')
    {
        std::cout << "F2 mat = " << std::endl;
        std::cout << F2_mat<< std::endl; 
        std::cout << "========================" << std::endl;
        std::cout << "G mat = " << std::endl; 
        std::cout << G_mat << std::endl;
        std::cout << "========================" << std::endl; 
        std::cout << "K2inv mat = " << std::endl; 
        std::cout << K2inv_mat << std::endl; 
        std::cout << "========================" << std::endl; 
        std::cout << "F3 mat = " << std::endl; 
        std::cout << F3mat << std::endl; 
        std::cout << "========================" << std::endl;
    }
    

    return real(F3mat.sum()); 
}


/*  This function will be used solely for the pole finder 
    and will be used with generate_pole.cpp and pole_maker.py */
comp function_for_pole_F3inv_ND_2plus1_mat( comp En, 
                                            std::vector< std::vector<comp> > p_config,
                                            std::vector< std::vector<comp> > k_config, 
                                            std::vector<comp> total_P, 
                                            double eta_i_1,
                                            double eta_i_2, 
                                            double scattering_length_1,
                                            double scattering_length_2,  
                                            double m_pi,
                                            double m_K,  
                                            double alpha, 
                                            double epsilon_h, 
                                            double L, 
                                            double xi, 
                                            int max_shell_num   )
{
    Eigen::MatrixXcd F3mat;
    Eigen::MatrixXcd F2mat;
    Eigen::MatrixXcd K2imat;
    Eigen::MatrixXcd Gmat; 
                                            
    int size = p_config[0].size();

    int size1=0, size2=0;
    
    
    //Eigen::MatrixXcd K2_mat_1(size,size);
    //Eigen::MatrixXcd K2_mat_2(size,size);
    //Eigen::MatrixXcd K2_mat(2*size,2*size);
    

    //for i = 1 
    double mi = m_K; 
    double mj = m_K; 
    double mk = m_pi; 

    std::vector< std::vector<comp> > p_config1(3,std::vector<comp> ());
    double config_tolerance = 1.0e-5;

    config_maker_1(p_config1, En, total_P, mi, mj, mk, L, xi, epsilon_h, config_tolerance );
    //std::cout<<"size1 in QC = "<<size1<<std::endl;
    
    std::vector< std::vector<comp> > k_config1 = p_config1;

    size1 = p_config1[0].size(); 
    Eigen::MatrixXcd F2_mat_1(size1,size1);
    Eigen::MatrixXcd K2inv_mat_1(size1,size1);
    //F2_i_mat( F2_mat_1, En, p_config1, k_config1, total_P, mi, mj, mk, L, alpha, epsilon_h, max_shell_num );
    F2_i_mat_1( F2_mat_1, En, p_config1, k_config1, total_P, mi, mj, mk, L, xi, alpha, epsilon_h, max_shell_num );
    
    K2inv_i_mat( K2inv_mat_1, eta_i_1, scattering_length_1, En, p_config1, k_config1, total_P, mi, mj, mk, epsilon_h, L );
    //K2_mat_1 = K2inv_mat_1.inverse();
    
    //for i = 2 
    mi = m_pi; 
    mj = m_K; 
    mk = m_K; 

    std::vector< std::vector<comp> > p_config2(3,std::vector<comp> ());
    config_maker_1(p_config2, En, total_P, mi, mj, mk, L, xi, epsilon_h, config_tolerance );
    std::vector< std::vector<comp> > k_config2 = p_config2;

     

    size2 = p_config2[0].size(); 
    //std::cout<<size2<<std::endl; 
    Eigen::MatrixXcd F2_mat_2(size2,size2);
    Eigen::MatrixXcd K2inv_mat_2(size2,size2);

    //std::cout << "spec 1 size = " << size1 << '\t' << "spec 2 size = " << size2 << std::endl; 
    
    
    //F2_i_mat( F2_mat_2, En, p_config2, k_config2, total_P, mi, mj, mk, L, alpha, epsilon_h, max_shell_num );
    F2_i_mat_1( F2_mat_2, En, p_config2, k_config2, total_P, mi, mj, mk, L, xi, alpha, epsilon_h, max_shell_num );
    
    K2inv_i_mat( K2inv_mat_2, eta_i_2, scattering_length_2, En, p_config2, k_config2, total_P, mi, mj, mk, epsilon_h, L );
    //K2_mat_2 = K2inv_mat_2.inverse();
    char debug1 = 'n';
    if(debug1=='y')
    {
        std::cout << "spec 1 size = " << std::endl;
        std::cout << p_config1[0].size()<< std::endl; 
        std::cout << "========================" << std::endl;
        std::cout << "spec 2 size = " << std::endl; 
        std::cout << p_config2[0].size() << std::endl;
        std::cout << "========================" << std::endl; 
        std::cout << "F2 mat 1 = " << std::endl; 
        std::cout << F2_mat_1 << std::endl; 
        std::cout << "========================" << std::endl; 
        std::cout << "F2 mat 2 = " << std::endl; 
        std::cout << F2_mat_2 << std::endl; 
        std::cout << "========================" << std::endl;
    }
    //for (i,j) = 1 1 and k = 2 
    mi = m_K; 
    mj = m_K; 
    mk = m_pi; 
    Eigen::MatrixXcd G_mat_11(size1,size1);
    G_ij_mat( G_mat_11, En, p_config1, k_config1, total_P, mi, mj, mk, L, epsilon_h ); 

    //for (i,j) = 1 2 and k = 1 
    mi = m_K; 
    mj = m_pi; 
    mk = m_K; 

    Eigen::MatrixXcd G_mat_12(size1,size2);
    G_ij_mat( G_mat_12, En, p_config1, k_config2, total_P, mi, mj, mk, L, epsilon_h ); 

    //for (i,j) = 2 1 and k = 1 
    mi = m_pi; 
    mj = m_K; 
    mk = m_K; 

    Eigen::MatrixXcd G_mat_21(size2,size1);
    G_ij_mat( G_mat_21, En, p_config2, k_config1, total_P, mi, mj, mk, L, epsilon_h ); 
    

    Eigen::MatrixXcd F2_mat(size1 + size2,size1 + size2);

    Eigen::MatrixXcd Filler0_12(size1,size2);
    Eigen::MatrixXcd Filler0_21(size2,size1);
    Eigen::MatrixXcd Filler0_22(size2,size2);

    Filler0_12 = Eigen::MatrixXcd::Zero(size1,size2);
    Filler0_21 = Eigen::MatrixXcd::Zero(size2,size1);
    Filler0_22 = Eigen::MatrixXcd::Zero(size2,size2);
    
    F2_mat <<   F2_mat_1,   Filler0_12,
                Filler0_21, F2_mat_2; 

    //F2_mat.topLeftCorner(size1,size1) = F2_mat_1;
    //F2_mat.topRightCorner(size1,size2) = Eigen::MatrixXcd::Zero(size2,size1);
    //F2_mat.bottomLeftCorner(size2,size1) = Eigen::MatrixXcd::Zero(size1,size2);
    //F2_mat.bottomRightCorner(size2,size2) = F2_mat_2;
    //K2_mat <<   K2_mat_1,                   Eigen::MatrixXcd::Zero(size,size),
    //            Eigen::MatrixXcd::Zero(size,size), 0.5*K2_mat_2;

    //F2_mat_builder(F2_mat,F2_mat_1, F2_mat_2, size1, size2);
    
    //std::cout<<"F2mat = "<<'\n'<<F2_mat<<std::endl;

    Eigen::MatrixXcd K2inv_mat(size1 + size2,size1 + size2);

    K2inv_mat   <<  K2inv_mat_1, Filler0_12,
                    Filler0_21,  2.0*K2inv_mat_2;

    Eigen::MatrixXcd G_mat(size1 + size2,size1 + size2);

    G_mat_12 = std::sqrt(2.0)*G_mat_12;
    G_mat_21 = std::sqrt(2.0)*G_mat_21;

    G_mat  <<   G_mat_11, G_mat_12,
                G_mat_21, Filler0_22;


    Eigen::MatrixXcd temp_identity_mat(size1 + size2,size1 + size2);
    temp_identity_mat.setIdentity();

    Eigen::MatrixXcd H_mat = K2inv_mat + F2_mat + G_mat; 
    //H_mat = H_mat*10000;
    Eigen::MatrixXcd H_mat_inv(size1 + size2,size1 + size2);
    double relerror = 0.0;

    //LinearSolver_3(H_mat, H_mat_inv, temp_identity_mat, relerror);
    LinearSolver_4(H_mat, H_mat_inv, temp_identity_mat, relerror);

    //std::cout << "Identity = " << temp_identity_mat << std::endl;
    //H_mat_inv = H_mat_inv/10000;

    //F3mat = F2_mat/3.0 - F2_mat*H_mat_inv*F2_mat;
    //Eigen::MatrixXcd temp_F3_mat(size,size);
    //Eigen::MatrixXcd temp_mat_A(size,size);
    //temp_mat_A = H_mat_inv*F2_mat;//H_mat.inverse()*F2_mat;
    //temp_F3_mat = F2_mat*temp_mat_A;
    
    //F3mat = (F2_mat/3.0 - F2_mat*H_mat.inverse()*F2_mat);//temp_F3_mat;

    F3mat = Eigen::MatrixXcd(size1+size2, size1+size2); 
    F3mat = (F2_mat/3.0 - F2_mat*H_mat_inv*F2_mat);//temp_F3_mat; 

    F2mat = Eigen::MatrixXcd(size1+size2, size1+size2); 
    K2imat = Eigen::MatrixXcd(size1+size2, size1+size2); 
    Gmat = Eigen::MatrixXcd(size1+size2, size1+size2); 

    Gmat = G_mat; 
    F2mat = F2_mat; 
    K2imat = K2inv_mat; 

    Eigen::MatrixXcd F3matinv = F3mat.inverse(); 
    

    char debug = 'n';
    if(debug=='y')
    {
        std::cout << "F2 mat = " << std::endl;
        std::cout << F2_mat<< std::endl; 
        std::cout << "========================" << std::endl;
        std::cout << "G mat = " << std::endl; 
        std::cout << G_mat << std::endl;
        std::cout << "========================" << std::endl; 
        std::cout << "K2inv mat = " << std::endl; 
        std::cout << K2inv_mat << std::endl; 
        std::cout << "========================" << std::endl; 
        std::cout << "F3 mat = " << std::endl; 
        std::cout << F3mat << std::endl; 
        std::cout << "========================" << std::endl;
    }
    

    return real(F3matinv.sum()); 
}



/* Test F3 for 3 ID system, we K2inv to zero and 
then change it gradually to see how the spectrum shifts */
void test_F3_ID_zeroK2( Eigen::MatrixXcd &F3mat,
                            Eigen::MatrixXcd &F2mat,
                            Eigen::MatrixXcd &K2imat,
                            Eigen::MatrixXcd &Gmat, 
                            comp En, 
                            std::vector<comp> total_P, 
                            double eta_i_1,
                            double eta_i_2, 
                            double scattering_length_1,
                            double scattering_length_2,  
                            double m_pi,
                            double m_K,  
                            double alpha, 
                            double epsilon_h, 
                            double L, 
                            double xi, 
                            int max_shell_num   )
{
    int size1=0, size2=0;
    
    
    //Eigen::MatrixXcd K2_mat_1(size,size);
    //Eigen::MatrixXcd K2_mat_2(size,size);
    //Eigen::MatrixXcd K2_mat(2*size,2*size);
    
    

    //for i = 1 
    double mi = m_K; 
    double mj = m_K; 
    double mk = m_pi; //here m_pi = m_K

    std::vector< std::vector<comp> > p_config1(3,std::vector<comp> ());
    std::vector< std::vector<int> > n_config1(3,std::vector<int> ());
    double config_tolerance = 1.0e-5;

    config_maker_2(p_config1, n_config1, En, total_P, mi, mj, mk, L, xi, epsilon_h, config_tolerance );
    //std::cout<<"size1 in QC = "<<size1<<std::endl;
    
    std::vector< std::vector<comp> > k_config1 = p_config1;

    size1 = p_config1[0].size(); 
    Eigen::MatrixXcd F2_mat_1(size1,size1);
    Eigen::MatrixXcd K2inv_mat_1(size1,size1);
    //F2_i_mat( F2_mat_1, En, p_config1, k_config1, total_P, mi, mj, mk, L, alpha, epsilon_h, max_shell_num );
    F2_i_mat_1( F2_mat_1, En, p_config1, k_config1, total_P, mi, mj, mk, L, xi, alpha, epsilon_h, max_shell_num );
    
    K2inv_i_mat( K2inv_mat_1, eta_i_1, scattering_length_1, En, p_config1, k_config1, total_P, mi, mj, mk, epsilon_h, L );
    //K2_mat_1 = K2inv_mat_1.inverse();
    
    
    char debug1 = 'n';
    if(debug1=='y')
    {
        std::cout << "spec 1 size = " << std::endl;
        std::cout << p_config1[0].size()<< std::endl; 
        std::cout << "========================" << std::endl;
        std::cout << "F2 mat 1 = " << std::endl; 
        std::cout << F2_mat_1 << std::endl; 
        std::cout << "========================" << std::endl; 
    }
    //for (i,j) = 1 1 and k = 2 
    mi = m_K; 
    mj = m_K; 
    mk = m_pi; 
    Eigen::MatrixXcd G_mat_11(size1,size1);
    G_ij_mat( G_mat_11, En, p_config1, k_config1, total_P, mi, mj, mk, L, epsilon_h ); 


    Eigen::MatrixXcd temp_identity_mat(size1,size1);
    temp_identity_mat.setIdentity();
    double relerror = 0; 

    Eigen::MatrixXcd H_mat =  K2inv_mat_1 + 0.5*F2_mat_1 + G_mat_11;
    
    Eigen::MatrixXcd H_mat_inv(size1, size1); 

    LinearSolver_4(H_mat, H_mat_inv, temp_identity_mat, relerror);

    Eigen::MatrixXcd NewF2 = 0.5*F2_mat_1;
    Eigen::MatrixXcd temp1 = H_mat_inv*NewF2;
    Eigen::MatrixXcd temp2 = NewF2*temp1; 
    Eigen::MatrixXcd temp3 = NewF2/3.0; 

    F3mat = Eigen::MatrixXcd(size1, size1); 
    //F3mat = (F2_mat_1/3.0 - F2_mat_1*H_mat.inverse()*F2_mat_1);//temp_F3_mat; 

    F3mat = temp3 - temp2;//temp_F3_mat; 

    F2mat = Eigen::MatrixXcd(size1, size1); 
    K2imat = Eigen::MatrixXcd(size1, size1); 
    Gmat = Eigen::MatrixXcd(size1, size1); 

    Gmat = G_mat_11; 
    F2mat = F2_mat_1; 
    K2imat = K2inv_mat_1; 
    

    char debug = 'n';
    if(debug=='y')
    {
        for(int i=0;i<n_config1[0].size();++i)
        {
            double pi = std::acos(-1.0); 
            double twopibyxiL = 2.0*pi/(xi*L);
            int nx = n_config1[0][i];
            int ny = n_config1[1][i];
            int nz = n_config1[2][i];

            comp px = ((comp)nx)*twopibyxiL;
            comp py = ((comp)ny)*twopibyxiL;
            comp pz = ((comp)nz)*twopibyxiL;

            std::vector<comp> check_p(3); 
            check_p[0] = px; 
            check_p[1] = py; 
            check_p[2] = pz; 

            comp sig_check_p = sigma_pvec_based(En, check_p, mi, total_P);
            comp cutoff_check_p = cutoff_function_1(sig_check_p, mj, mk, epsilon_h); 

            std::cout<<"n = ["<<nx<<'\t'<<ny<<'\t'<<nz<<"]"<<'\t'
                     <<"cutoff = "<<cutoff_check_p<<std::endl; 
        }

        std::cout << "F2 mat = " << std::endl;
        std::cout << F2_mat_1<< std::endl; 
        std::cout << "========================" << std::endl;
        std::cout << "G mat = " << std::endl; 
        std::cout << G_mat_11 << std::endl;
        std::cout << "========================" << std::endl; 
        std::cout << "K2inv mat = " << std::endl; 
        std::cout << K2inv_mat_1 << std::endl; 
        std::cout << "========================" << std::endl; 
        std::cout << "F3 mat = " << std::endl; 
        std::cout << F3mat << std::endl; 
        std::cout << "========================" << std::endl;
    }
}



#endif