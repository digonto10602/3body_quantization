/* This code takes in the frame momentum shell number and the lab energy
and returns the real(F3inv) to fit K3iso 
What it does :
Based on the nPx, nPy, nPz values, it loads the appropriate file of ultraHQ
F3 files and builds the splines between energy_A_CM and energy_B_CM, it then gives 
out the value of F3inv at energy = energy_CM 

HERE Energy_A_CM, Energy_B_CM and EnergyCM all are in CM FRAME energies 

[ There are chances of speeding this up, for instance, if we create and save the 
the splines for each interval, then they dont need to be generated each time when 
we are performing the K3iso fitting. But that is for later. ]

This uses  the following input */
/* ./spline_F3inv nPx nPy nPz spline_size energy_A_CM energy_B_CM energy_CM */

#include "F2_functions.h"
#include "K2_functions.h"
#include "G_functions.h"
#include "QC_functions.h"
#include "pole_searching.h"
#include "splines.h"

void generate_spline_based_F3inv_L20(   int nPx, 
                                        int nPy, 
                                        int nPz,
                                        int spline_size, 
                                        double energy_A_CM,
                                        double energy_B_CM,
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

    // Reading F3 files for data
    std::ifstream fin; 
    std::string drive = "./test_files/F3_for_pole_KKpi_L20/";
    std::string filename = drive + "ultraHQ_F3_for_pole_KKpi_L20_nP_"
                            + std::to_string(nPx)
                            + std::to_string(nPy) 
                            + std::to_string(nPz) 
                            + ".dat";

    fin.open(filename.c_str());
    
    double f1_En, f1_Ecm, f1_norm, f1_F3, f1_F2,
            f1_G, f1_K2inv, f1_Hinv;
    
    std::vector<double> f1_Ecm_vec;
    std::vector<double> f1_F3_vec; 
    std::vector<double> f1_F3inv_vec; 
    
    while(fin>>f1_En>>f1_Ecm>>f1_norm>>f1_F3>>f1_F2>>f1_G>>f1_K2inv>>f1_Hinv)
    {
        f1_Ecm_vec.push_back(f1_Ecm);
        f1_F3_vec.push_back(f1_F3);
        f1_F3inv_vec.push_back(1.0/f1_F3);
    }

    fin.close(); 
    /*----------------------------------------------*/

    std::vector<double> selected_Ecm; 
    std::vector<double> selected_F3inv; 
    for(int i=0;i<f1_F3inv_vec.size();++i)
    {
        if(f1_Ecm_vec[i]>energy_A_CM && f1_Ecm_vec[i]<energy_B_CM)
        {
            selected_Ecm.push_back(f1_Ecm_vec[i]);
            selected_F3inv.push_back(f1_F3inv_vec[i]);
        }
    }

    int F3inv_vec_size = selected_F3inv.size(); 
    if(F3inv_vec_size<spline_size) spline_size = F3inv_vec_size; 

    int initial_point = 0;
    int final_point = F3inv_vec_size; 
    int del_point = abs(final_point - initial_point)/spline_size; 

    std::vector<int> index_vec; 
    for(int i=0;i<spline_size; ++i)
    {
        int ind = initial_point + i*del_point; 
        index_vec.push_back(ind); 
    }

    //These two are selected for spline code 
    std::vector<double> xj_vec;
    std::vector<double> fj_vec; 
    for(int i=0;i<index_vec.size();++i)
    {
        xj_vec.push_back(selected_Ecm[index_vec[i]]);
        fj_vec.push_back(selected_F3inv[index_vec[i]]);

    }

    int size = xj_vec.size(); 
    std::vector<std::vector<double> > Sij_vec1(size, std::vector<double> (size)); 
    Sij_builder(energy_CM, xj_vec, Sij_vec1);

    double xval = energy_CM; 
    int j_val = 0; 
    for(int j=0; j<xj_vec.size()-1; ++j)
    {
        double xj_check1 = xj_vec[j];
        double xj_check2 = xj_vec[j+1]; 

        if(xval>=xj_check1 && xval<=xj_check2)
        {
            j_val = j; 
        }
    }

    double f_spline_val = 0.0; 

    for(int k=0; k<xj_vec.size(); ++k)
    {
        f_spline_val = f_spline_val + Sij_vec1[k][j_val]*fj_vec[k];
    }
    
    std::cout<<std::setprecision(30);
    std::cout<<f_spline_val<<std::endl; 
}

int main(int argc, char *argv[])
{
    //./spline_F3inv nPx nPy nPz spline_size energy_A_CM energy_B_CM energy_CM
    int nPx = std::stoi(argv[1]);
    int nPy = std::stoi(argv[2]);
    int nPz = std::stoi(argv[3]);
    int spline_size = std::stoi(argv[4]); 
    double energy_A_CM = std::stod(argv[5]); 
    double energy_B_CM = std::stod(argv[6]);
    double energy_CM = std::stod(argv[7]);

    generate_spline_based_F3inv_L20( nPx, nPy, nPz, spline_size, energy_A_CM, energy_B_CM, energy_CM ); 

    return 0; 
}