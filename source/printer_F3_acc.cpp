//THIS DOES NOT WORK

#include "functions.h"
#include "F2_functions.h"
#include "K2_functions.h"
#include "G_functions.h"
#include "QC_functions.h"
#include "pole_searching.h"


void test_detF3inv_vs_En_KKpi_acc()
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

    /*---------------------------------------------------*/

    /*---------------------P config----------------------*/
    int nPmax = 20;
    std::vector<std::vector<int> > nP_config(3,std::vector<int>());

    for(int i=0;i<nPmax+1;++i)
    {
        for(int j=0;j<nPmax+1;++j)
        {
            for(int k=0;k<nPmax+1;++k)
            {
                int nsq = i*i + j*j + k*k;
                if(nsq<=4)
                {

                    if(i>=j && j>=k)
                    {
                        std::cout<<"P config:"<<std::endl;
                        std::cout<<i<<'\t'<<j<<'\t'<<k<<std::endl; 

                        nP_config[0].push_back(i);
                        nP_config[1].push_back(j);
                        nP_config[2].push_back(k);
            
                    }
                }
            }
        }
    } 


    int P_config_size = nP_config[0].size();

    /*-----------------------------------------------------*/

    //for(int i=0;i<P_config_size;++i)
    {
        int nPx = 0;//nP_config[0][i];
        int nPy = 0;//nP_config[1][i];
        int nPz = 0;//nP_config[2][i];
    
        std::string filename =    "F3_for_pole_KKpi_L20_nP_"//"F3_for_pole_KKpi_scatlength_--_L20_nP_"//
                                + std::to_string((int)nPx)
                                + std::to_string((int)nPy)
                                + std::to_string((int)nPz)
                                + ".dat";

        //std::string filename = "temp";
        comp Px = ((comp)nPx)*twopibyxiLbyas;//twopibyL;
        comp Py = ((comp)nPy)*twopibyxiLbyas;//twopibyL;
        comp Pz = ((comp)nPz)*twopibyxiLbyas;//twopibyL;
        std::vector<comp> total_P(3);
        total_P[0] = Px; 
        total_P[1] = Py; 
        total_P[2] = Pz; 
        comp total_P_val = std::sqrt(Px*Px + Py*Py + Pz*Pz);


        double mi = atmK;
        double mj = atmK;
        double mk = atmpi; 
        //for nP 100 the first run starts 0.4184939100000000245
        double KKpi_threshold = atmK + atmK + atmpi; 
        double KKpipi_threshold = 2.0*atmK + 2.0*atmpi; 
        double KKKK_threshold = 5.0*atmK; 

        double En_initial = std::sqrt(KKpi_threshold*KKpi_threshold + 0.0000001 + abs(total_P_val*total_P_val));//.27;//0.4184939100000000245;//0.26302;
        double En_final = std::sqrt(KKKK_threshold*KKKK_threshold + abs(total_P_val*total_P_val));;
        double En_points = 2500;

        double delE = abs(En_initial - En_final)/En_points;

        std::ofstream fout; 
        fout.open(filename.c_str());

        double* result_F3 = NULL;
	    result_F3 = new double[(int)En_points+1];

        double* Ecm_vec = NULL;
	    Ecm_vec = new double[(int)En_points+1];
	
	    for(int i=0;i<En_points+1;++i)
        {
            Ecm_vec[i] = 0;
            result_F3[i] = 0;
        } 

        #pragma acc data copy(Ecm_vec[0:En_points],result_F3[0:En_points]) 
	    #pragma acc parallel loop independent
        for(int i=0; i<En_points; ++i)
        {
            double En = En_initial + i*delE; 

            std::vector< std::vector<comp> > p_config(3,std::vector<comp> ());
            double config_tolerance = 1.0e-5;
            //config_maker_1(p_config, En, total_P, mi, mj, mk, L, xi, epsilon_h, config_tolerance );

            std::vector< std::vector<comp> > k_config = p_config; 


            int size = p_config[0].size();
            //std::cout<<"size = "<<size<<std::endl;  
            Eigen::MatrixXcd F3_mat;//(Eigen::Dynamic,Eigen::Dynamic);
            Eigen::MatrixXcd F2_mat;
            Eigen::MatrixXcd K2i_mat; 
            Eigen::MatrixXcd G_mat; 

            comp F3iso; 

            test_F3iso_ND_2plus1_mat(  F3_mat, F3iso, F2_mat, K2i_mat, G_mat, En, p_config, k_config, total_P, eta_1, eta_2, scattering_length_1_piK, scattering_length_2_KK, atmpi, atmK, alpha, epsilon_h, L, xi, max_shell_num); 
            //std::cout<<"ran until here"<<std::endl;
            //Eigen::MatrixXcd F3_mat_inv = F3_mat.inverse();
            //std::cout<<std::setprecision(3)<<"F3mat=\n"<<F3_mat<<std::endl; 
            //double res = det_F3_ND_2plus1_mat( En, p_config, k_config, total_P, eta_1, eta_2, scattering_length_1_piK, scattering_length_2_KK, atmpi, atmK, alpha, epsilon_h, L, xi, max_shell_num); 
            comp Ecm_calculated = E_to_Ecm(En, total_P);

            comp detF3 = 0.0;//F3_mat.determinant();  // We don't need these values for now 
            comp sumF3 = F3iso;//F3_mat.sum(); 
            comp detF3inv = 0.0;//F3_mat_inv.determinant(); 
            comp sumF3inv = 1.0/F3iso; //F3_mat_inv.sum(); 

            Ecm_vec[i] = real(Ecm_calculated); 
            result_F3[i] = real(F3iso);
            std::cout<<"running = "<<i<<std::endl; 
            
        }
        //fout.close();

        for(int i=0;i<En_points;++i)
        {
            std::cout<<std::setprecision(20)<<i<<'\t'<<Ecm_vec[i]<<'\t'<<result_F3[i]<<std::endl; 
        }

        delete [] Ecm_vec;
        Ecm_vec = NULL;
        delete [] result_F3; 
        result_F3 = NULL; 
    }               
}

int main()
{
    test_detF3inv_vs_En_KKpi_acc();
    return 0; 
}

