#include "F2_functions.h"
#include "K2_functions.h"
#include "G_functions.h"
#include "QC_functions.h"
#include "pole_searching.h"



void test_F2_i1_mombased_vs_En()
{
    double mi = 1.0;
    double mj = 1.0;
    double mk = 1.0;
    double epsilon_h = 0.0; 

    double En_initial = 0.1;
    double En_final = 5.5;
    double En_points = 3000.0;
    double del_En = abs(En_initial - En_final)/En_points;

    double alpha = 0.750;
    double L = 6.0;
    //double En = 3.2;
    int max_shell_num = 50;

    std::ofstream fout; 
    
    std::string filename = "F2_i1_test.dat";

    fout.open(filename.c_str());

    comp total_P = 0.0;
    comp spec_p = 0.0;
    
    for(int i=0;i<En_points + 1;++i)
    {
        double En = En_initial + i*del_En; 
        std::vector<comp> k(3);
        k[0] = 0.0;
        k[1] = 0.0;
        k[2] = 0.0;
        std::vector<comp> p = k;
        std::vector<comp> total_P = k;

        comp total_P_val = std::sqrt(total_P[0]*total_P[0] + total_P[1]*total_P[1] + total_P[2]*total_P[2]);

        comp sigma_p = sigma(En, spec_p, mi, total_P_val);

        comp F2 = F2_i1(En, k, p, total_P, L, mi, mj, mk, alpha, epsilon_h, max_shell_num );

        //comp sigk = sigma(En,spec_p,mi,total_P);

        std::cout<<std::setprecision(20)<<En<<'\t'<<real(sigma_p)<<'\t'<<real(F2)<<'\t'<<imag(F2)<<std::endl; 

        fout<<std::setprecision(20);
        //fout<<real(sigma_p)<<'\t'<<imag(sigma_p)<<'\t'<<real(F2)<<'\t'<<imag(F2)<<std::endl; 
        fout<<En-mi<<'\t'<<real(F2)<<'\t'<<imag(F2)<<std::endl; 
    }
    fout.close();


}

void K2printer()
{
    double m = 1;
    double a = -10.0;
    double L = 6.0;

    double En_initial = 0.1;
    double En_final = 5.5;
    double En_points = 3000.0;
    double del_En = abs(En_initial - En_final)/En_points;

    std::string filename = "K2file.dat";
    std::ofstream fout;
    fout.open(filename.c_str());

    std::vector<comp> p(3);
    p[0] = 0.0;
    p[1] = 0.0;
    p[2] = 0.0;
    std::vector<comp> k = p;
    std::vector<comp> total_P = p;

    for(int i=0;i<En_points+1;++i)
    {
        double En = En_initial + i*del_En;
        double spec_p = 0.0;
        double total_P = 0.0; 
        comp sigp = sigma(En, spec_p, m, total_P);
        comp K2 = tilde_K2_00(0.5,a, p, k, sigp, m, m, m, 0.0, L);

        fout<<En-m<<'\t'<<-real(K2)<<'\t'<<-imag(K2)<<std::endl;
    }
    fout.close();

}

void test_config_maker()
{
    double En = 3.1;
    double L = 6;
    double mi = 1.0;

    std::vector< std::vector<comp> > p_config(3,std::vector<comp> ());

    config_maker(p_config, En, mi, L);

    for(int i=0; i<p_config.size(); ++i)
    {
        for(int j=0; j<p_config[0].size(); ++j)
        {
            std::cout << "p" << i << "," <<j<<" = " << p_config[i][j] << std::endl;
        }

    }
}

void test_F2_i_mat()
{
    double En = 3.1;
    double L = 6;
    double mi = 1.0;
    double mj = 1.0;
    double mk = 1.0;

    double alpha = 0.5;
    double epsilon_h = 0.0;
    int max_shell_num = 50;

    std::vector< std::vector<comp> > p_config(3,std::vector<comp> ());

    config_maker(p_config, En, mi, L);

    std::vector< std::vector<comp> > k_config = p_config; 

    comp Px = 0.0;
    comp Py = 0.0;
    comp Pz = 0.0;
    std::vector<comp> total_P(3);
    total_P[0] = Px; 
    total_P[1] = Py; 
    total_P[2] = Pz; 

    int size = p_config[0].size(); 
    Eigen::MatrixXcd F_mat(size,size);

    F2_i_mat( F_mat, En, p_config, k_config, total_P, mi, mj, mk, L, alpha, epsilon_h, max_shell_num );

    std::cout << F_mat << std::endl; 
}

void test_K2_i_mat()
{
    double En = 3.1;
    double L = 6;
    double mi = 1.0;
    double mj = 1.0;
    double mk = 1.0;

    double alpha = 0.5;
    double epsilon_h = 0.0;
    int max_shell_num = 50;
    double eta_i = 0.5;
    double scattering_length = -10.0;

    std::vector< std::vector<comp> > p_config(3,std::vector<comp> ());

    config_maker(p_config, En, mi, L);

    std::vector< std::vector<comp> > k_config = p_config; 

    comp Px = 0.0;
    comp Py = 0.0;
    comp Pz = 0.0;
    std::vector<comp> total_P(3);
    total_P[0] = Px; 
    total_P[1] = Py; 
    total_P[2] = Pz; 

    int size = p_config[0].size(); 
    Eigen::MatrixXcd K2inv_mat(size,size);

    K2inv_i_mat(K2inv_mat, eta_i, scattering_length, En, p_config, k_config, total_P, mi, mj, mk, epsilon_h, L );
                    

    std::cout << K2inv_mat << std::endl; 
}

void test_G_ij_mat()
{
    double En = 3.1;
    double L = 6;
    double mi = 1.0;
    double mj = 1.0;
    double mk = 1.0;

    double alpha = 0.5;
    double epsilon_h = 0.0;
    int max_shell_num = 50;
    double eta_i = 0.5;
    double scattering_length = -10.0;

    std::vector< std::vector<comp> > p_config(3,std::vector<comp> ());

    config_maker(p_config, En, mi, L);

    std::vector< std::vector<comp> > k_config = p_config; 

    comp Px = 0.0;
    comp Py = 0.0;
    comp Pz = 0.0;
    std::vector<comp> total_P(3);
    total_P[0] = Px; 
    total_P[1] = Py; 
    total_P[2] = Pz; 

    int size = p_config[0].size(); 
    Eigen::MatrixXcd G_mat(size,size);

    G_ij_mat(G_mat, En, p_config, k_config, total_P, mi, mj, mk, L, epsilon_h); 
                    

    std::cout << G_mat << std::endl; 
}

void test_F3_mat()
{
    double En = 3.1;
    double L = 6;
    double mi = 1.0;
    double mj = 1.0;
    double mk = 1.0;

    double alpha = 0.5;
    double epsilon_h = 0.0;
    int max_shell_num = 18;
    double eta_i = 0.5;
    double scattering_length = -10.0;

    comp Px = 0.0;
    comp Py = 0.0;
    comp Pz = 0.0;
    std::vector<comp> total_P(3);
    total_P[0] = Px; 
    total_P[1] = Py; 
    total_P[2] = Pz; 

    std::vector< std::vector<comp> > p_config(3,std::vector<comp> ());

    //config_maker(p_config, En, mi, L);
    double config_tolerance = 1.0e-5;
    config_maker_1(p_config, En, total_P, mi, mj, mk, L, epsilon_h, config_tolerance );

    std::vector< std::vector<comp> > k_config = p_config; 

    

    for(int i=0;i<p_config[0].size();++i)
    {
        comp px = p_config[0][i];
        comp py = p_config[1][i];
        comp pz = p_config[2][i];
        comp spec_p = std::sqrt(px*px + py*py + pz*pz);

        std::cout << "p = " << spec_p << '\t'
                  << "px= " << px     << '\t'
                  << "py= " << py     << '\t'
                  << "pz= " << pz     << std::endl; 
    }

    int size = p_config[0].size(); 
    Eigen::MatrixXcd F3_mat(size,size);

    F3_ID_mat(F3_mat, En, p_config, k_config, total_P, eta_i, scattering_length, mi, mj, mk, alpha, epsilon_h, L, max_shell_num );

    std::cout << "F3 mat = " << std::endl;                
    std::cout << F3_mat << std::endl; 
    std::cout << F3_mat.sum() << std::endl;
}

void test_F3_mat_vs_En()
{
    //double En = 3.1;
    double L = 6;
    double mi = 1.0;
    double mj = 1.0;
    double mk = 1.0;

    double alpha = 0.5;
    double epsilon_h = 0.0;
    int max_shell_num = 20;
    double eta_i = 0.5;
    double scattering_length = -10.0;
    
    double En_initial = 2.5;
    double En_final = 4.5;
    double En_points = 2999.0;
    double del_En = abs(En_initial - En_final)/En_points; 

    std::ofstream fout; 
    std::string filename = "F3_ID_test_1.dat";
    fout.open(filename.c_str());

    comp Px = 0.0;
    comp Py = 0.0;
    comp Pz = 0.0;
    std::vector<comp> total_P(3);
    total_P[0] = Px; 
    total_P[1] = Py; 
    total_P[2] = Pz; 

    for(int i=0; i<En_points+1; ++i)
    {
        double En = En_initial + i*del_En; 
        std::vector< std::vector<comp> > p_config(3,std::vector<comp> ());

        //config_maker(p_config, En, mi, L);
        double config_tolerance = 1.0e-5;
        config_maker_1(p_config, En, total_P, mi, mj, mk, L, epsilon_h, config_tolerance );

        std::vector< std::vector<comp> > k_config = p_config; 

        

        int size = p_config[0].size(); 
        Eigen::MatrixXcd F3_mat(size,size);

        F3_ID_mat(F3_mat, En, p_config, k_config, total_P, eta_i, scattering_length, mi, mj, mk, alpha, epsilon_h, L, max_shell_num );

        //std::cout << F3_mat << std::endl;            
        comp res = F3_mat.sum();
        //std::cout << F3_mat << std::endl; 
        std::cout << "En = " << En << " F3 = " << res << " matrix size = " << size << std::endl;
        fout << std::setprecision(20) << En << '\t' << real(res) << '\t' << imag(res) << std::endl; 
    }

    fout.close();
    
}

void test_F3_nd_2plus1()
{
    double En = 3.2;
    double L = 6;

    double mpi = 1.01;
    double mK = 1.02;

    double eta_1 = 1.0;
    double eta_2 = 0.5;
    double scattering_length_1_piK = -4.04;
    double scattering_length_2_KK = -4.07;

    double mi = mK;
    double mj = mK; 
    double mk = mpi; 

    double alpha = 0.5;
    double epsilon_h = 0.0;
    int max_shell_num = 20;

    comp Px = 0.0;
    comp Py = 0.0;
    comp Pz = 0.0;
    std::vector<comp> total_P(3);
    total_P[0] = Px; 
    total_P[1] = Py; 
    total_P[2] = Pz; 

    std::vector< std::vector<comp> > p_config(3,std::vector<comp> ());
    double config_tolerance = 1.0e-5;
    config_maker_1(p_config, En, total_P, mi, mj, mk, L, epsilon_h, config_tolerance );

    std::vector< std::vector<comp> > k_config = p_config; 

        

    int size = p_config[0].size();
    std::cout<<"size = "<<size<<std::endl;  
    Eigen::MatrixXcd F3_mat(2*size,2*size);

    F3_ND_2plus1_mat(  F3_mat, En, p_config, k_config, total_P, eta_1, eta_2, scattering_length_1_piK, scattering_length_2_KK, mpi, mK, alpha, epsilon_h, L, max_shell_num); 
    
    double res = det_F3_ND_2plus1_mat( En, p_config, k_config, total_P, 1.0, 0.5, -10, -20, mpi, mK, alpha, epsilon_h, L, max_shell_num); 
    
    std::cout<<std::setprecision(10)<<"det of F3 = "<<F3_mat.determinant()<<std::endl; 
}

void test_detF3inv_vs_En()
{

    /*  Inputs  */
    
    double L = 5;

    double scattering_length_1_piK = 0.15;//-4.04;
    double scattering_length_2_KK = 0.1;//-4.07;
    double eta_1 = 1.0;
    double eta_2 = 0.5; 
    double atmpi = 50;//0.06906;
    double atmK = 100;//0.09698;

    atmpi = atmpi/atmK; 
    atmK = 1.0;
    

    double alpha = 0.5;
    double epsilon_h = 0.0;
    int max_shell_num = 20;

    comp Px = 0.0;
    comp Py = 0.0;
    comp Pz = 0.0;
    std::vector<comp> total_P(3);
    total_P[0] = Px; 
    total_P[1] = Py; 
    total_P[2] = Pz; 

    double mi = atmK;
    double mj = atmK;
    double mk = atmpi; 

    double En_initial = 3.1;//0.26302;
    double En_final = 3.5;//0.31;
    double En_points = 10;

    double delE = abs(En_initial - En_final)/En_points; 

    std::ofstream fout; 
    std::string filename = "det_F3inv_test_L5.dat";
    fout.open(filename.c_str());

    for(int i=0; i<En_points+1; ++i)
    {
        double En = En_initial + i*delE; 

        std::vector< std::vector<comp> > p_config(3,std::vector<comp> ());
        double config_tolerance = 1.0e-5;
        config_maker_1(p_config, En, total_P, mi, mj, mk, L, epsilon_h, config_tolerance );

        std::vector< std::vector<comp> > k_config = p_config; 

        

        int size = p_config[0].size();
        std::cout<<"size = "<<size<<std::endl;  
        Eigen::MatrixXcd F3_mat(2*size,2*size);

        F3_ND_2plus1_mat(  F3_mat, En, p_config, k_config, total_P, eta_1, eta_2, scattering_length_1_piK, scattering_length_2_KK, atmpi, atmK, alpha, epsilon_h, L, max_shell_num); 
    
        Eigen::MatrixXcd F3_mat_inv = F3_mat.inverse();
        //double res = det_F3_ND_2plus1_mat( En, p_config, k_config, total_P, eta_1, eta_2, scattering_length_1_piK, scattering_length_2_KK, atmpi, atmK, alpha, epsilon_h, L, max_shell_num); 

        fout << std::setprecision(20) << En << '\t' << real(F3_mat_inv.determinant()) << std::endl;
        
        std::cout<<std::setprecision(20);
        std::cout<< "En = " << En << '\t' << "det of F3inv = "<< real(F3_mat_inv.determinant()) << std::endl; 
    }
}

void test_detF3_vs_En()
{

    /*  Inputs  */
    
    double L = 20;

    double scattering_length_1_piK = -4.04;
    double scattering_length_2_KK = -4.07;
    double eta_1 = 1.0;
    double eta_2 = 0.5; 
    double atmpi = 0.06906;
    double atmK = 0.09698;

    double alpha = 0.5;
    double epsilon_h = 0.0;
    int max_shell_num = 20;

    comp Px = 0.0;
    comp Py = 0.0;
    comp Pz = 0.0;
    std::vector<comp> total_P(3);
    total_P[0] = Px; 
    total_P[1] = Py; 
    total_P[2] = Pz; 

    double mi = atmK;
    double mj = atmK;
    double mk = atmpi; 

    double En_initial = 0.26302;
    double En_final = 0.31;
    double En_points = 10;

    double delE = abs(En_initial - En_final)/En_points; 

    std::ofstream fout; 
    std::string filename = "det_F3_test_L6.dat";
    fout.open(filename.c_str());

    for(int i=0; i<En_points+1; ++i)
    {
        double En = En_initial + i*delE; 

        std::vector< std::vector<comp> > p_config(3,std::vector<comp> ());
        double config_tolerance = 1.0e-5;
        config_maker_1(p_config, En, total_P, mi, mj, mk, L, epsilon_h, config_tolerance );

        std::vector< std::vector<comp> > k_config = p_config; 

        

        int size = p_config[0].size();
        std::cout<<"size = "<<size<<std::endl;  
        Eigen::MatrixXcd F3_mat(2*size,2*size);

        F3_ND_2plus1_mat(  F3_mat, En, p_config, k_config, total_P, eta_1, eta_2, scattering_length_1_piK, scattering_length_2_KK, atmpi, atmK, alpha, epsilon_h, L, max_shell_num); 
    
        //double res = det_F3_ND_2plus1_mat( En, p_config, k_config, total_P, eta_1, eta_2, scattering_length_1_piK, scattering_length_2_KK, atmpi, atmK, alpha, epsilon_h, L, max_shell_num); 

        fout << std::setprecision(20) << En << '\t' << abs(F3_mat.determinant()) << std::endl;
        
        std::cout<<std::setprecision(20);
        std::cout<< "En = " << En << '\t' << "det of F3inv = "<< abs(F3_mat.determinant()) << std::endl; 
    }
}


int main()
{
    //This was a test for the identical case for 3particles
    //test_F2_i1_mombased();
    //K2printer();
    //test_F2_i1_mombased_vs_En();
    //test_QC3_vs_En();
    //I00_sum_F_test();
    //test_config_maker();
    //test_F2_i_mat();
    //test_K2_i_mat();
    //test_G_ij_mat();
    //test_F3_mat();
    //test_F3_mat_vs_En();
    //-----------------------------------------------------

    //From here we test for 2+1 system
    //test_F3_nd_2plus1();

    //test_detF3_vs_En();

    //test_F3inv_pole_searching();

    test_detF3inv_vs_En();
    
    return 0;
}