
#include "functions.h"
#include "F2_functions.h"
#include "K2_functions.h"
#include "G_functions.h"
#include "QC_functions.h"
#include "pole_searching.h"
#include "omp.h"

void test_detF3inv_vs_En_KKpi_omp()
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

    for(int ind1=0;ind1<P_config_size;++ind1)
    {
        int nPx = nP_config[0][ind1];
        int nPy = nP_config[1][ind1];
        int nPz = nP_config[2][ind1];
    
        std::string filename =    "ultraHQ_F3_for_pole_KKpi_L20_nP_"//"F3_for_pole_KKpi_scatlength_--_L20_nP_"//
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
        double En_points = 50000;

        double delE = abs(En_initial - En_final)/En_points;

        std::ofstream fout; 
        fout.open(filename.c_str());

        double* norm_vec = NULL; 
        norm_vec = new double[(int)En_points+1];

        double* En_vec = NULL; 
        En_vec = new double[(int)En_points+1];

        double* Ecm_vec = NULL;
	    Ecm_vec = new double[(int)En_points+1];

        comp* result_F3 = NULL;
	    result_F3 = new comp[(int)En_points+1];

        comp* result_F2 = NULL; 
        result_F2 = new comp[(int)En_points+1]; 

        comp* result_K2inv = NULL; 
        result_K2inv = new comp[(int)En_points+1];

        comp* result_G = NULL; 
        result_G = new comp[(int)En_points+1];

        comp* result_Hinv = NULL; 
        result_Hinv = new comp[(int)En_points+1];

	    for(int i=0;i<En_points+1;++i)
        {
            norm_vec[i] = 0.0; 
            En_vec[i] = 0.0;
            Ecm_vec[i] = 0.0;
            result_F3[i] = 0.0;
            result_F2[i] = 0.0;
            result_K2inv[i] = 0.0;
            result_G[i] = 0.0;
            result_Hinv[i] = 0.0; 
        } 

        //#pragma acc data copy(Ecm_vec[0:En_points],result_F3[0:En_points]) 
	    //#pragma acc parallel loop independent
        int loopcounter = 0; 
        int i=0; 
        #pragma omp parallel for 
        for(i=0; i<(int)En_points + 1; ++i)
        {
            double En = En_initial + i*delE; 

            std::vector< std::vector<comp> > p_config(3,std::vector<comp> ());
            double config_tolerance = 1.0e-5;
            //config_maker_1(p_config, En, total_P, mi, mj, mk, L, xi, epsilon_h, config_tolerance );

            std::vector< std::vector<comp> > k_config = p_config; 


            int size = p_config[0].size();
            //std::cout<<"size = "<<size<<std::endl;
            Eigen::VectorXcd state_vec;   
            Eigen::MatrixXcd F3_mat;//(Eigen::Dynamic,Eigen::Dynamic);
            Eigen::MatrixXcd F2_mat;
            Eigen::MatrixXcd K2i_mat; 
            Eigen::MatrixXcd G_mat; 
            Eigen::MatrixXcd Hmatinv; 

            test_F3iso_ND_2plus1_mat_with_normalization(  F3_mat, state_vec, F2_mat, K2i_mat, G_mat, Hmatinv, En, p_config, k_config, total_P, eta_1, eta_2, scattering_length_1_piK, scattering_length_2_KK, atmpi, atmK, alpha, epsilon_h, L, xi, max_shell_num); 

            comp Ecm_calculated = E_to_Ecm(En, total_P);

            comp F3iso = state_vec.transpose()*F3_mat*state_vec; 
            comp F2iso = state_vec.transpose()*F2_mat*state_vec;
            comp K2inv_iso = state_vec.transpose()*K2i_mat*state_vec;
            comp Giso = state_vec.transpose()*G_mat*state_vec;
            comp Hmatinv_iso = state_vec.transpose()*Hmatinv*state_vec;

            comp norm1 = state_vec.transpose()*state_vec; 
            double norm2 = real(norm1); 
            norm_vec[i] = 1.0/std::sqrt(norm2); 
            En_vec[i] = En; 
            Ecm_vec[i] = real(Ecm_calculated); 
            result_F3[i] = F3iso;
            result_F2[i] = F2iso; 
            result_G[i] = Giso; 
            result_K2inv[i] = K2inv_iso; 
            result_Hinv[i] = Hmatinv_iso; 

            //std::cout<<"running = "<<i<<std::endl; 
            double looppercent = ((loopcounter+1)/(En_points))*100.0;

            loopcounter = loopcounter + 1; 
            int divisor = (En_points)/10; 
            if(loopcounter%divisor==0)
            {
                //if((int)looppercent==templooppercent)
                //{
                //    continue; 
                //}
                //else 
                {
                    std::cout<<"P="<<nPx<<nPy<<nPz<<" run completion: "<<looppercent<<"%"<<std::endl; 
                    //templooppercent = (int) looppercent; 
                }
            }
            
        }

        for(int i=0;i<En_points;++i)
        {
            //std::cout<<std::setprecision(20)<<i<<'\t'<<Ecm_vec[i]<<'\t'<<result_F3[i]<<std::endl; 
            fout<<std::setprecision(20)
                <<En_vec[i]<<'\t'
                <<Ecm_vec[i]<<'\t'
                <<norm_vec[i]<<'\t'
                <<real(result_F3[i])<<'\t'
                <<real(result_F2[i])<<'\t'
                <<real(result_G[i])<<'\t'
                <<real(result_K2inv[i])<<'\t'
                <<real(result_Hinv[i])<<std::endl; 
        }
        fout.close();

        std::cout<<"P = "<<nPx<<nPy<<nPz<<" file generated!"<<std::endl; 

        delete [] norm_vec;
        norm_vec = NULL; 
        delete [] En_vec; 
        En_vec = NULL; 
        delete [] Ecm_vec;
        Ecm_vec = NULL;
        delete [] result_F3; 
        result_F3 = NULL;
        delete [] result_F2; 
        result_F2 = NULL; 
        delete [] result_K2inv;
        result_K2inv = NULL; 
        delete [] result_G; 
        result_G = NULL; 
        delete [] result_Hinv; 
        result_Hinv = NULL; 

    }               
}


/* Here we create 6 sets of data files based on two sets of 
scattering lengths with their uncertainties */
void test_detF3inv_vs_En_KKpi_6_diff_ma_omp()
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

    std::vector<double> scat1(2);
    std::vector<double> scat2(2); 
    scat1[0] = 4.04 - 0.2; 
    scat1[1] = 4.04 + 0.2; 
    //scat1[2] = 4.04 + 0.2; 
    scat2[0] = 4.07 - 0.07; 
    scat2[1] = 4.07 + 0.07; 
    //scat2[2] = 4.07 + 0.07; 

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
    for(int ma1=0;ma1<scat1.size();++ma1)
    {
    
    for(int ma2=0;ma2<scat2.size();++ma2)
    {
        scattering_length_1_piK = scat1[ma1];
        scattering_length_2_KK = scat2[ma2]; 
        
    for(int ind1=0;ind1<P_config_size;++ind1)
    {
        int nPx = nP_config[0][ind1];
        int nPy = nP_config[1][ind1];
        int nPz = nP_config[2][ind1];
    
        std::string filename =    "ultraHQ_F3_for_pole_KKpi_L20_nP_"//"F3_for_pole_KKpi_scatlength_--_L20_nP_"//
                                + std::to_string((int)nPx)
                                + std::to_string((int)nPy)
                                + std::to_string((int)nPz)
                                + "_mapiK_" + std::to_string(scattering_length_1_piK)
                                + "_maKK_" + std::to_string(scattering_length_2_KK)
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
        double En_points = 25000;

        double delE = abs(En_initial - En_final)/En_points;

        std::ofstream fout; 
        fout.open(filename.c_str());

        double* norm_vec = NULL; 
        norm_vec = new double[(int)En_points+1];

        double* En_vec = NULL; 
        En_vec = new double[(int)En_points+1];

        double* Ecm_vec = NULL;
	    Ecm_vec = new double[(int)En_points+1];

        comp* result_F3 = NULL;
	    result_F3 = new comp[(int)En_points+1];

        comp* result_F2 = NULL; 
        result_F2 = new comp[(int)En_points+1]; 

        comp* result_K2inv = NULL; 
        result_K2inv = new comp[(int)En_points+1];

        comp* result_G = NULL; 
        result_G = new comp[(int)En_points+1];

        comp* result_Hinv = NULL; 
        result_Hinv = new comp[(int)En_points+1];

	    for(int i=0;i<En_points+1;++i)
        {
            norm_vec[i] = 0.0; 
            En_vec[i] = 0.0;
            Ecm_vec[i] = 0.0;
            result_F3[i] = 0.0;
            result_F2[i] = 0.0;
            result_K2inv[i] = 0.0;
            result_G[i] = 0.0;
            result_Hinv[i] = 0.0; 
        } 

        //#pragma acc data copy(Ecm_vec[0:En_points],result_F3[0:En_points]) 
	    //#pragma acc parallel loop independent
        int loopcounter = 0; 
        int i=0; 
        #pragma omp parallel for schedule(guided)
        for(i=0; i<(int)En_points + 1; ++i)
        {
            double En = En_initial + i*delE; 

            std::vector< std::vector<comp> > p_config(3,std::vector<comp> ());
            double config_tolerance = 1.0e-5;
            //config_maker_1(p_config, En, total_P, mi, mj, mk, L, xi, epsilon_h, config_tolerance );

            std::vector< std::vector<comp> > k_config = p_config; 


            int size = p_config[0].size();
            //std::cout<<"size = "<<size<<std::endl;
            Eigen::VectorXcd state_vec;   
            Eigen::MatrixXcd F3_mat;//(Eigen::Dynamic,Eigen::Dynamic);
            Eigen::MatrixXcd F2_mat;
            Eigen::MatrixXcd K2i_mat; 
            Eigen::MatrixXcd G_mat; 
            Eigen::MatrixXcd Hmatinv; 

            test_F3iso_ND_2plus1_mat_with_normalization(  F3_mat, state_vec, F2_mat, K2i_mat, G_mat, Hmatinv, En, p_config, k_config, total_P, eta_1, eta_2, scattering_length_1_piK, scattering_length_2_KK, atmpi, atmK, alpha, epsilon_h, L, xi, max_shell_num); 

            comp Ecm_calculated = E_to_Ecm(En, total_P);

            comp F3iso = state_vec.transpose()*F3_mat*state_vec; 
            comp F2iso = state_vec.transpose()*F2_mat*state_vec;
            comp K2inv_iso = state_vec.transpose()*K2i_mat*state_vec;
            comp Giso = state_vec.transpose()*G_mat*state_vec;
            comp Hmatinv_iso = state_vec.transpose()*Hmatinv*state_vec;

            comp norm1 = state_vec.transpose()*state_vec; 
            double norm2 = real(norm1); 
            norm_vec[i] = 1.0/std::sqrt(norm2); 
            En_vec[i] = En; 
            Ecm_vec[i] = real(Ecm_calculated); 
            result_F3[i] = F3iso;
            result_F2[i] = F2iso; 
            result_G[i] = Giso; 
            result_K2inv[i] = K2inv_iso; 
            result_Hinv[i] = Hmatinv_iso; 

            //std::cout<<"running = "<<i<<std::endl; 
            double looppercent = ((loopcounter+1)/(En_points))*100.0;

            loopcounter = loopcounter + 1; 
            int divisor = (En_points)/10; 
            if(loopcounter%divisor==0)
            {
                //if((int)looppercent==templooppercent)
                //{
                //    continue; 
                //}
                //else 
                {
                    std::cout<<"P="<<nPx<<nPy<<nPz<<" "
                             <<"ma_piK="<<scattering_length_1_piK<<" "
                             <<"ma_KK="<<scattering_length_2_KK<<" "
                             <<"run completion: "<<looppercent<<"%"<<std::endl; 
                    //templooppercent = (int) looppercent; 
                }
            }
            
        }

        for(int i=0;i<En_points;++i)
        {
            //std::cout<<std::setprecision(20)<<i<<'\t'<<Ecm_vec[i]<<'\t'<<result_F3[i]<<std::endl; 
            fout<<std::setprecision(20)
                <<En_vec[i]<<'\t'
                <<Ecm_vec[i]<<'\t'
                <<norm_vec[i]<<'\t'
                <<real(result_F3[i])<<'\t'
                <<real(result_F2[i])<<'\t'
                <<real(result_G[i])<<'\t'
                <<real(result_K2inv[i])<<'\t'
                <<real(result_Hinv[i])<<std::endl; 
        }
        fout.close();

        std::cout<<"P = "<<nPx<<nPy<<nPz<<" file generated!"<<std::endl; 

        delete [] norm_vec;
        norm_vec = NULL; 
        delete [] En_vec; 
        En_vec = NULL; 
        delete [] Ecm_vec;
        Ecm_vec = NULL;
        delete [] result_F3; 
        result_F3 = NULL;
        delete [] result_F2; 
        result_F2 = NULL; 
        delete [] result_K2inv;
        result_K2inv = NULL; 
        delete [] result_G; 
        result_G = NULL; 
        delete [] result_Hinv; 
        result_Hinv = NULL; 

    }
    }
    }               
}


void test_F2_for_missing_poles()
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

    std::vector<double> scat1(2);
    std::vector<double> scat2(2); 
    scat1[0] = 4.04 - 0.2; 
    scat1[1] = 4.04 + 0.2; 
    //scat1[2] = 4.04 + 0.2; 
    scat2[0] = 4.07 - 0.07; 
    scat2[1] = 4.07 + 0.07; 
    //scat2[2] = 4.07 + 0.07; 

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
    std::vector<comp> total_P(3);

    int nPx = 0; 
    int nPy = 0; 
    int nPz = 0; 
    total_P[0] = ((comp)nPx)*twopibyxiLbyas;
    total_P[1] = ((comp)nPy)*twopibyxiLbyas;
    total_P[2] = ((comp)nPz)*twopibyxiLbyas;

    double En = 3.9; 
    double mi = atmK; 
    double mj = atmK; 
    double mk = atmpi; 

    std::vector< std::vector<comp> > p_config1(3,std::vector<comp> ());
    double config_tolerance = 1.0e-5;

    config_maker_1(p_config1, En, total_P, mi, mj, mk, L, xi, epsilon_h, config_tolerance );
    std::vector< std::vector<comp> > k_config1 = p_config1;
    int size1 = p_config1[0].size(); 
    Eigen::MatrixXcd F2_mat_1(size1,size1);
    Eigen::MatrixXcd K2inv_mat_1(size1,size1);
    //F2_i_mat( F2_mat_1, En, p_config1, k_config1, total_P, mi, mj, mk, L, alpha, epsilon_h, max_shell_num );
    F2_i_mat_1( F2_mat_1, En, p_config1, k_config1, total_P, mi, mj, mk, L, xi, alpha, epsilon_h, max_shell_num );
    
}


void test_3body_non_int()
{
    /*  Inputs  */
    
    double L = 20;
    double Lbyas = L;
    double xi = 3.444; /* found from lattice */
    int nmax = 20; 
    int nsq_max = 20;
    

    double scattering_length_1_piK = -4.04;
    double scattering_length_2_KK = -4.07;
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

    double m1 = atmK; 
    double m2 = atmK; 
    double m3 = atmpi; 

    int nPx = 2;
    int nPy = 0; 
    int nPz = 0; 
    
    std::string filename = "3body_non_int_points_using_c_code_L20_P" 
                            + std::to_string(nPx)
                            + std::to_string(nPy)
                            + std::to_string(nPz)
                            + ".dat";

    
    std::vector<comp> total_P(3); 
    total_P[0] = ((comp)nPx)*twopibyxiLbyas; 
    total_P[1] = ((comp)nPy)*twopibyxiLbyas; 
    total_P[2] = ((comp)nPz)*twopibyxiLbyas; 
    threebody_non_int_spectrum(filename, m1, m2, m3, total_P, xi, Lbyas, nmax, nsq_max);
}

int main()
{
    //test_detF3inv_vs_En_KKpi_omp();
    //test_detF3inv_vs_En_KKpi_6_diff_ma_omp();
    test_3body_non_int();
    return 0; 
}

