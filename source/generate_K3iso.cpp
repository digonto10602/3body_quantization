/* This code takes in the frame momentum shell number and the lab energy
and returns the real(F3inv) to constrain K3iso
This uses  the following input */
/* ./K3iso nPx nPy nPz energy */

#include "F2_functions.h"
#include "K2_functions.h"
#include "G_functions.h"
#include "QC_functions.h"
#include "pole_searching.h"

void generate_K3iso_L20(    int nPx, 
                            int nPy, 
                            int nPz, 
                            comp En )
{

    /*  Inputs  */
    
    double L = 20;
    double Lbyas = L;
    double xi = 3.444; /* found from lattice */
    

    double scattering_length_1_piK = 4.04;
    double scattering_length_2_KK = 4.07;
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

    //comp lab_energy = En; 

    Eigen::MatrixXcd F3_mat;
    Eigen::MatrixXcd F2_mat;
    Eigen::MatrixXcd K2i_mat; 
    Eigen::MatrixXcd G_mat; 

    //these are dummy p_config and k_config 
    std::vector<std::vector<comp> > p_config(3, std::vector<comp>());
    std::vector<std::vector<comp> > k_config(3, std::vector<comp>());

    comp K3iso = -function_F3inv_ND_2plus1_mat(  F3_mat, F2_mat, K2i_mat, G_mat, En, p_config, k_config, total_P, eta_1, eta_2, scattering_length_1_piK, scattering_length_2_KK, atmpi, atmK, alpha, epsilon_h, L, xi, max_shell_num); 
            
    std::cout<<std::setprecision(20); 
    std::cout<<real(K3iso)<<std::endl; 
}

int main(int argc, char *argv[])
{
    int nPx = std::stoi(argv[1]);
    int nPy = std::stoi(argv[2]);
    int nPz = std::stoi(argv[3]);

    double energy = std::stod(argv[4]);

    generate_K3iso_L20(nPx, nPy, nPz, energy);

    return 0; 
}