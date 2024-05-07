/* This code takes in the frame momentum shell number and the lab energy
and returns the real(F3inv) to fit K3iso 
What it does :
Based on the nPx, nPy, nPz values, it generates the matrix and solves the F3 function 
to give out the value of F3inv at energy = energy_CM 

HERE Energy_A_CM, Energy_B_CM and EnergyCM all are in CM FRAME energies 

This uses  the following input */
/* ./eigen_F3inv nPx nPy nPz energy_CM */

#include "F2_functions.h"
#include "K2_functions.h"
#include "G_functions.h"
#include "QC_functions.h"
#include "pole_searching.h"
#include "splines.h"

void generate_eigen_based_F3inv_L20(    int nPx, 
                                        int nPy, 
                                        int nPz,
                                        double energy_CM )
{
    /*  Inputs  */
    
    double L = 20;
    double Lbyas = L;
    double xi = 1.0; 
    double xi1 = 3.444;/* found from lattice */
    L = L*xi1; // This is done to set the spatial 
                // unit in terms of a_t everywhere 
    Lbyas = L; 
    double scattering_length_1_piK = 4.04;// - 0.2; //total uncertainty 0.05 stat 0.15 systematic 
    double scattering_length_2_KK = 4.07;// - 0.07; //total uncertainty 0.07 stat 
    double eta_1 = 1.0;
    double eta_2 = 0.5; 
    double atmpi = 0.06906;
    double atmK = 0.09698;

    //atmpi = atmpi/atmK; 
    //atmK = 1.0;
    

    double alpha = 0.5;
    double epsilon_h = 0.0;
    int max_shell_num = 20;

    double pi = std::acos(-1.0); 
    comp twopibyL = 2.0*pi/L;
    comp twopibyxiLbyas = 2.0*pi/(xi*Lbyas);

    comp Px = ((comp)nPx)*twopibyxiLbyas;//twopibyL;
    comp Py = ((comp)nPy)*twopibyxiLbyas;//twopibyL;
    comp Pz = ((comp)nPz)*twopibyxiLbyas;//twopibyL;
    std::vector<comp> total_P(3);
    total_P[0] = Px; 
    total_P[1] = Py; 
    total_P[2] = Pz; 
    comp total_P_val = std::sqrt(Px*Px + Py*Py + Pz*Pz);

    
    /*----------------------------------------------*/

    std::vector< std::vector<comp> > p_config(3,std::vector<comp> ());
    double config_tolerance = 1.0e-5;

    std::vector< std::vector<comp> > k_config = p_config; 


    int size = p_config[0].size();
            
    Eigen::VectorXcd state_vec;   
    Eigen::MatrixXcd F3_mat;//(Eigen::Dynamic,Eigen::Dynamic);
    Eigen::MatrixXcd F2_mat;
    Eigen::MatrixXcd K2i_mat; 
    Eigen::MatrixXcd G_mat; 
    Eigen::MatrixXcd Hmatinv; 

    comp En = Ecm_to_E(energy_CM,total_P);
    test_F3iso_ND_2plus1_mat_with_normalization(  F3_mat, state_vec, F2_mat, K2i_mat, G_mat, Hmatinv, En, p_config, k_config, total_P, eta_1, eta_2, scattering_length_1_piK, scattering_length_2_KK, atmpi, atmK, alpha, epsilon_h, L, xi, max_shell_num); 

    comp Ecm_calculated = E_to_Ecm(En, total_P);
    comp F3iso = state_vec.transpose()*F3_mat*state_vec; 
    //comp F2iso = state_vec.transpose()*F2_mat*state_vec;
    //comp K2inv_iso = state_vec.transpose()*K2i_mat*state_vec;
    //comp Giso = state_vec.transpose()*G_mat*state_vec;
    //comp Hmatinv_iso = state_vec.transpose()*Hmatinv*state_vec;

    //comp norm1 = state_vec.transpose()*state_vec; 
    //double norm2 = real(norm1); 
    
    std::cout<<std::setprecision(30);
    std::cout<<real(1.0/F3iso)<<std::endl; 
}

int main(int argc, char *argv[])
{
    //./eigen_F3inv nPx nPy nPz energy_CM
    int nPx = std::stoi(argv[1]);
    int nPy = std::stoi(argv[2]);
    int nPz = std::stoi(argv[3]);
    double energy_CM = std::stod(argv[4]);

    generate_eigen_based_F3inv_L20( nPx, nPy, nPz, energy_CM ); 

    return 0; 
}