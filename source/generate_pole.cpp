/* This code takes in the frame momentum shell number and the lab energy
along with two guesses of energy in lab frame and returns the pole in 
F3inv, should be used with pole_maker.py file 
This uses  the following input */
/* ./generate_pole nPx nPy nPz en_guess1 en_guess2 max_iteration */

#include "F2_functions.h"
#include "K2_functions.h"
#include "G_functions.h"
#include "QC_functions.h"
#include "pole_searching.h"

void generate_pole_L20(     int nPx, 
                            int nPy, 
                            int nPz, 
                            double En_guess1,
                            double En_guess2,
                            int max_iter )
{

    /*  Inputs  */
    
    double L = 20;
    double Lbyas = L;
    double xi = 1; 
    double xi1 = 3.444; /* found from lattice */
    L = L*xi1; 
    Lbyas = L; 

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

    /*
    F3_inv_mat_poles_secant_method( double init_guess1,
                                    double init_guess2, 
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
                                    int max_shell_num,
                                    int max_iter,
                                    double &pole   )
    */

    double pole = 0.0;
    F3_inv_mat_poles_secant_method(En_guess1, En_guess2, total_P, eta_1, eta_2, scattering_length_1_piK, scattering_length_2_KK, atmpi, atmK, alpha, epsilon_h, L, xi, max_shell_num, max_iter, pole );
            
    std::cout<<std::setprecision(20); 
    std::cout<<pole<<std::endl; 

    /* Remember that this returns the poles in lab frame, 
       the python script converts this energy to cm frame energy */
}

int main(int argc, char *argv[])
{
    int nPx = std::stoi(argv[1]);
    int nPy = std::stoi(argv[2]);
    int nPz = std::stoi(argv[3]);

    double energy_guess1 = std::stod(argv[4]);
    double energy_guess2 = std::stod(argv[5]);

    int max_iteration = std::stoi(argv[6]);

    generate_pole_L20(nPx, nPy, nPz, energy_guess1, energy_guess2, max_iteration);

    return 0; 
}