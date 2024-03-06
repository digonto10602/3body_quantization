/*
        Compilation command:
        g++ printer_function.cpp -o printer -O3 -std=c++14 -I/usr/include/eigen3/ -fopenmp

*/


#include "functions.h"
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
    double xi = 3.444;// added new
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
    config_maker_1(p_config, En, total_P, mi, mj, mk, L, xi, epsilon_h, config_tolerance );

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
    double xi = 3.444; //added new 
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
        config_maker_1(p_config, En, total_P, mi, mj, mk, L, xi, epsilon_h, config_tolerance );

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
    double xi = 3.444;//added new 

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
    config_maker_1(p_config, En, total_P, mi, mj, mk, L, xi, epsilon_h, config_tolerance );

    std::vector< std::vector<comp> > k_config = p_config; 

        

    int size = p_config[0].size();
    std::cout<<"size = "<<size<<std::endl;  
    Eigen::MatrixXcd F3_mat(2*size,2*size);

    F3_ND_2plus1_mat(  F3_mat, En, p_config, k_config, total_P, eta_1, eta_2, scattering_length_1_piK, scattering_length_2_KK, mpi, mK, alpha, epsilon_h, L, xi, max_shell_num); 
    
    double res = det_F3_ND_2plus1_mat( En, p_config, k_config, total_P, 1.0, 0.5, -10, -20, mpi, mK, alpha, epsilon_h, L, xi, max_shell_num); 
    
    std::cout<<std::setprecision(10)<<"det of F3 = "<<F3_mat.determinant()<<std::endl; 
}

/* This function was used to final check and 
the results between this code and the FRL code */
void test_detF3inv_vs_En()
{

    /*  Inputs  */
    
    double L = 5;
    double xi = 1.0;//3.444;

    double scattering_length_1_piK = -2.0;//0.15;//-4.04;
    double scattering_length_2_KK = -2.0;//0.1;//-4.07;
    double eta_1 = 1.0;
    double eta_2 = 0.5; 
    double atmpi = 0.9;//0.06906;
    double atmK = 1;//0.09698;

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
    std::string filename = "F3_2plus1_test_poles_L5.dat";
    fout.open(filename.c_str());

    for(int i=0; i<En_points+1; ++i)
    {
        double En = En_initial + i*delE; 

        std::vector< std::vector<comp> > p_config(3,std::vector<comp> ());
        double config_tolerance = 1.0e-5;
        config_maker_1(p_config, En, total_P, mi, mj, mk, L, xi, epsilon_h, config_tolerance );

        std::vector< std::vector<comp> > k_config = p_config; 

        

        int size = p_config[0].size();
        //std::cout<<"size = "<<size<<std::endl;  
        Eigen::MatrixXcd F3_mat;//(Eigen::Dynamic,Eigen::Dynamic);
        Eigen::MatrixXcd F2_mat;
        Eigen::MatrixXcd K2i_mat; 
        Eigen::MatrixXcd G_mat; 

        test_F3_ND_2plus1_mat(  F3_mat, F2_mat, K2i_mat, G_mat, En, p_config, k_config, total_P, eta_1, eta_2, scattering_length_1_piK, scattering_length_2_KK, atmpi, atmK, alpha, epsilon_h, L, xi, max_shell_num); 
    
        //std::cout<<std::setprecision(3)<<"F3mat=\n"<<F3_mat<<std::endl; 
        Eigen::MatrixXcd F3_mat_inv = F3_mat.inverse();
        //double res = det_F3_ND_2plus1_mat( En, p_config, k_config, total_P, eta_1, eta_2, scattering_length_1_piK, scattering_length_2_KK, atmpi, atmK, alpha, epsilon_h, L, max_shell_num); 

        fout    << std::setprecision(20) 
                << En << '\t' 
                << real(F2_mat.determinant()) << '\t'
                << real(G_mat.determinant()) << '\t'
                << real(K2i_mat.determinant()) << '\t'
                << real(F3_mat_inv.determinant()) << std::endl;
        
        std::cout<<std::setprecision(20);
        std::cout<< "En = " << En << '\t' << "det of F3inv = "<< real(F3_mat_inv.determinant()) << std::endl; 
    }
}

void test_detF3_vs_En()
{

    /*  Inputs  */
    
    double L = 20;
    double xi = 3.444; //added new

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
        config_maker_1(p_config, En, total_P, mi, mj, mk, L, xi, epsilon_h, config_tolerance );

        std::vector< std::vector<comp> > k_config = p_config; 

        

        int size = p_config[0].size();
        std::cout<<"size = "<<size<<std::endl;  
        Eigen::MatrixXcd F3_mat(2*size,2*size);

        F3_ND_2plus1_mat(  F3_mat, En, p_config, k_config, total_P, eta_1, eta_2, scattering_length_1_piK, scattering_length_2_KK, atmpi, atmK, alpha, epsilon_h, L, xi, max_shell_num); 
    
        //double res = det_F3_ND_2plus1_mat( En, p_config, k_config, total_P, eta_1, eta_2, scattering_length_1_piK, scattering_length_2_KK, atmpi, atmK, alpha, epsilon_h, L, xi, max_shell_num); 

        fout << std::setprecision(20) << En << '\t' << abs(F3_mat.determinant()) << std::endl;
        
        std::cout<<std::setprecision(20);
        std::cout<< "En = " << En << '\t' << "det of F3inv = "<< abs(F3_mat.determinant()) << std::endl; 
    }
}


void test_uneven_matrix()
{
    int size1 = 4;
    int size2 = 2;

    Eigen::MatrixXcd A(size1,size1);
    A = Eigen::MatrixXcd::Random(size1,size1);

    Eigen::MatrixXcd B(size2,size2);
    B = Eigen::MatrixXcd::Identity(size2,size2);

    std::cout<<A<<std::endl; 
    std::cout<<B<<std::endl; 

    Eigen::MatrixXcd C(size1+size2, size1+size2);
    std::cout<<C<<std::endl; 

    Eigen::MatrixXcd Filler1(size1, size2);
    Filler1 = Eigen::MatrixXcd::Zero(size1,size2);
    std::cout<<Filler1<<std::endl; 

    Eigen::MatrixXcd A12(size1,size2);
    A12 = Eigen::MatrixXcd::Random(size1,size2);

    Eigen::MatrixXcd Filler2(size2,size1);
    Filler2 = Eigen::MatrixXcd::Zero(size2,size1);
    std::cout<<Filler2<<std::endl; 
    

    //C << A, Filler1,
    //     Filler2, B; 
    C << A, A12,
         Filler2, B; 
    std::cout<<C<<std::endl;
}


void test_individual_functions()
{
    /*  Inputs  */
    double pi = std::acos(-1.0);
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

    double En = 3.1; 

    comp twopibyL = 2.0*pi/L; 
    comp px = 0.0*twopibyL;
    comp py = 0.0*twopibyL;
    comp pz = 0.0*twopibyL;

    comp spec_p = std::sqrt(px*px + py*py + pz*pz); 
    std::vector<comp> p(3);
    p[0] = px; 
    p[1] = py; 
    p[2] = pz; 
    
    comp kx = 0.0*twopibyL;
    comp ky = 0.0*twopibyL;
    comp kz = 0.0*twopibyL;

    comp spec_k = std::sqrt(kx*kx + ky*ky + kz*kz); 
    std::vector<comp> k(3);
    k[0] = kx; 
    k[1] = ky; 
    k[2] = kz; 

    comp Pminusk_x = Px - kx; 
    comp Pminusk_y = Py - ky; 
    comp Pminusk_z = Pz - kz; 
    comp Pminusk = std::sqrt(Pminusk_x*Pminusk_x + Pminusk_y*Pminusk_y + Pminusk_z*Pminusk_z);

    comp sig_k = (En - omega_func(spec_k,mi))*(En - omega_func(spec_k,mi)) - Pminusk*Pminusk; 
    double chosen_scattering_length = scattering_length_2_KK;
    comp K2_inv_val = K2_inv_00(eta_2, chosen_scattering_length, sig_k, mj, mk, epsilon_h);
    comp K2_inv_val_temp = K2_inv_00_test_FRL(eta_2, chosen_scattering_length, spec_k, sig_k, mi, mj, mk, epsilon_h); 
    
    comp sigma_vecbased = sigma_pvec_based(En,p,mi,total_P);
    comp qsq_vec_based = q2psq_star(sigma_vecbased,mj,mk);

    std::cout<<std::setprecision(25);

    std::cout<<"pi = "<<pi<<std::endl; 
    std::cout<<"h = "<<cutoff_function_1(sig_k, mj, mk, epsilon_h)<<std::endl; 
    std::cout<<"k = "<<spec_k<<std::endl; 
    std::cout<<"omega_k = "<<omega_func(spec_k, mi)<<std::endl; 
    std::cout<<"sig_i = "<<sig_k<<std::endl;
    std::cout<<"E2kstar = "<<std::sqrt(sig_k)<<std::endl; 
    std::cout<<"sig_i_vecbased = "<<sigma_vecbased<<std::endl;
    std::cout<<"qsq vecbased = "<<qsq_vec_based<<std::endl; 

    std::cout<<"q2 = "<<q2psq_star(sig_k, mj, mk)<<std::endl; 
    std::cout<<"q_abs = "<<std::abs(std::sqrt(q2psq_star(sig_k, mj, mk)))<<std::endl; 
    std::cout<<"K2_inv = "<<K2_inv_val/(2.0*omega_func(spec_k,mi))<<std::endl; 
    std::cout<<"K2_inv temp = "<<K2_inv_val_temp<<std::endl; 
    //std::cout<<"K2_inv = "<<K2_inv_val/std::pow(L,3)<<std::endl; 

    comp F2_val = F2_i1( En, k, p, total_P, L, mi, mj, mk, alpha, epsilon_h, max_shell_num);

    std::cout<<"F2_val = "<<F2_val<<std::endl; 

    std::cout<<"erfi(-0.5) = "<<ERFI_func(-0.5)<<std::endl; 

    double fadeeva_erfi = Faddeeva::erfi(-0.5);
    std::cout<<"faddeeva erfi(-0.5) = "<<fadeeva_erfi<<std::endl; 

    comp Gij_val = G_ij(En, p, k, total_P, mi, mj, mk, L, epsilon_h);

    std::cout<<"Gij val = "<<Gij_val<<std::endl; 



}

void test_individual_functions_KKpi()
{
    /*  Inputs  */
    double pi = std::acos(-1.0);
    double L = 20;
    double xi = 3.444; //added new 

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

    comp twopibyL = 2.0*pi/L; 

    std::string test_file1 = "qsq_test_KKpi_2body_KK_P200"; //this is the test file for qsq, sigma_p, H(p) data 
    comp Px = 2.0*twopibyL;
    comp Py = 0.0*twopibyL;
    comp Pz = 0.0*twopibyL;
    std::vector<comp> total_P(3);
    total_P[0] = Px; 
    total_P[1] = Py; 
    total_P[2] = Pz; 

    comp spec_P = std::sqrt(Px*Px + Py*Py + Pz*Pz); 
    comp total_P_val = spec_P; 

    double mi = atmpi;//atmK;
    double mj = atmK;
    double mk = atmK;//atmpi; 

    //double En_initial = 0.26302;
    //double En_final = 0.31;
    //double En_points = 10;

    //double delE = abs(En_initial - En_final)/En_points; 

    double En = 0.3184939100000000245;//0.26303; 

    
    comp px = 0.0*twopibyL;
    comp py = 0.0*twopibyL;
    comp pz = 0.0*twopibyL;

    comp spec_p = std::sqrt(px*px + py*py + pz*pz); 
    std::vector<comp> p(3);
    p[0] = px; 
    p[1] = py; 
    p[2] = pz; 
    std::vector<std::vector<comp> > p_config(3,std::vector<comp>());
    p_config[0].push_back(px);
    p_config[1].push_back(py);
    p_config[2].push_back(pz);
    
    comp kx = 0.0*twopibyL;
    comp ky = 0.0*twopibyL;
    comp kz = 0.0*twopibyL;

    comp spec_k = std::sqrt(kx*kx + ky*ky + kz*kz); 
    std::vector<comp> k(3);
    k[0] = kx; 
    k[1] = ky; 
    k[2] = kz; 
    std::vector<std::vector<comp> > k_config(3,std::vector<comp>());
    k_config[0].push_back(kx);
    k_config[1].push_back(ky);
    k_config[2].push_back(kz);
    
    comp Pminusk_x = Px - kx; 
    comp Pminusk_y = Py - ky; 
    comp Pminusk_z = Pz - kz; 
    comp Pminusk = std::sqrt(Pminusk_x*Pminusk_x + Pminusk_y*Pminusk_y + Pminusk_z*Pminusk_z);

    comp sig_k = (En - omega_func(spec_k,mi))*(En - omega_func(spec_k,mi)) - Pminusk*Pminusk; 
    double chosen_scattering_length = scattering_length_2_KK;
    comp K2_inv_val = K2_inv_00(eta_2, chosen_scattering_length, sig_k, mj, mk, epsilon_h);
    comp K2_inv_val_temp = K2_inv_00_test_FRL(eta_2, chosen_scattering_length, spec_k, sig_k, mi, mj, mk, epsilon_h); 
    
    comp sigma_vecbased = sigma_pvec_based(En,p,mi,total_P);
    comp qsq_vec_based = q2psq_star(sigma_vecbased,mj,mk);

    std::cout<<std::setprecision(25);

    std::cout<<"pi = "<<pi<<std::endl; 
    std::cout<<"h = "<<cutoff_function_1(sig_k, mj, mk, epsilon_h)<<std::endl; 
    std::cout<<"k = "<<spec_k<<std::endl; 
    std::cout<<"omega_k = "<<omega_func(spec_k, mi)<<std::endl; 
    std::cout<<"sig_i = "<<sig_k<<std::endl;
    std::cout<<"E2kstar = "<<std::sqrt(sig_k)<<std::endl; 
    std::cout<<"sig_i_vecbased = "<<sigma_vecbased<<std::endl;
    std::cout<<"qsq vecbased = "<<qsq_vec_based<<std::endl; 

    std::cout<<"q2 = "<<q2psq_star(sig_k, mj, mk)<<std::endl; 
    std::cout<<"q_abs = "<<std::abs(std::sqrt(q2psq_star(sig_k, mj, mk)))<<std::endl; 
    std::cout<<"K2_inv = "<<K2_inv_val/(2.0*omega_func(spec_k,mi))<<std::endl; 
    std::cout<<"K2_inv temp = "<<K2_inv_val_temp<<std::endl; 
    //std::cout<<"K2_inv = "<<K2_inv_val/std::pow(L,3)<<std::endl; 

    comp F2_val = F2_i1( En, k, p, total_P, L, mi, mj, mk, alpha, epsilon_h, max_shell_num);

    std::cout<<"F2_val = "<<F2_val<<std::endl; 

    std::cout<<"erfi(-0.5) = "<<ERFI_func(-0.5)<<std::endl; 

    double fadeeva_erfi = Faddeeva::erfi(-0.5);
    std::cout<<"faddeeva erfi(-0.5) = "<<fadeeva_erfi<<std::endl; 

    comp Gij_val = G_ij(En, p, k, total_P, mi, mj, mk, L, epsilon_h);

    std::cout<<"Gij val = "<<Gij_val<<std::endl; 

    // New testing //
    std::cout<<"========================================="<<std::endl; 

    std::cout<<"h = "<<cutoff_function_1(sig_k, mj, mk, epsilon_h)<<std::endl; 
    std::cout<<"En = "<<En<<std::endl;
    std::cout<<"Ecm calculated = "<<std::sqrt(En*En - spec_P*spec_P)<<std::endl; 
    std::cout<<"total_P = "<<spec_P<<std::endl; 

    //sum of 2+1 particle energy = 
    comp sum_energy = std::sqrt(sig_k + Pminusk*Pminusk) + omega_func(spec_k, mi);
    //comp sum_energy = std::sqrt(sig_k) + omega_func(spec_k, mi);
    
    std::cout<<"sum energy = "<< sum_energy <<std::endl; 

    //t cut comes in s when sig_i < |mj^2 - mk^2| 
    double t_cut_threshold_1 = abs(mj*mj - mk*mk);
    //double t_cut_threshold_2 = abs(mi*mi - m)
    std::cout<<"t_cut_threshold = "<< t_cut_threshold_1 <<std::endl;

    //Test 2//
    char turn_this_on = 'n';
    std::cout<<"=========================================="<<std::endl; 
    if(turn_this_on=='y')
    {
        Eigen::MatrixXcd F3_mat;//(Eigen::Dynamic,Eigen::Dynamic);
        Eigen::MatrixXcd F2_mat;
        Eigen::MatrixXcd K2i_mat; 
        Eigen::MatrixXcd G_mat; 

        test_F3_ND_2plus1_mat(  F3_mat, F2_mat, K2i_mat, G_mat, En, p_config, k_config, total_P, eta_1, eta_2, scattering_length_1_piK, scattering_length_2_KK, atmpi, atmK, alpha, epsilon_h, L, xi, max_shell_num); 
    
        //std::cout<<std::setprecision(3)<<"F3mat=\n"<<F3_mat<<std::endl; 
        Eigen::MatrixXcd F3_mat_inv = F3_mat.inverse();
        
        std::cout<<"F3inv det = "<<F3_mat_inv.determinant()<<std::endl; 
        std::cout<<"F3inv sum = "<<F3_mat_inv.sum()<<std::endl;
        std::cout<<"F2 det = "<<F2_mat.determinant()<<std::endl; 
        std::cout<<"K2i det = "<<K2i_mat.determinant()<<std::endl; 
        std::cout<<"G det = "<<G_mat.determinant()<<std::endl; 
    }
    //Test the q^2, x^2, r^2 etc.. for F2 function 
    std::cout<<"Test for q^2, r^2 =>"<<std::endl; 

    double KKpi_threshold = atmK + atmK + atmpi; 
    double KKpipi_threshold = 2.0*atmK + 2.0*atmpi; 

    double Elab1 = 0.01;//std::sqrt(KKpi_threshold*KKpi_threshold + abs(total_P_val*total_P_val));
    //double Elab2 = std::sqrt(0.38*0.38 + abs(total_P_val*total_P_val));
    double Elab2 = std::sqrt(KKpipi_threshold*KKpipi_threshold + abs(total_P_val*total_P_val));

    double En_initial = Elab1;//.27;//0.4184939100000000245;//0.26302;
    double En_final = Elab2;
    double En_points = 4000;

    double delE = abs(En_initial - En_final)/En_points;
    std::ofstream fout1;
    
    fout1.open(test_file1.c_str());
    for(int i=0;i<En_points+1;++i)
    {
        double En = En_initial + i*delE; 
        comp sig_p = sigma(En,spec_p,mi,total_P_val);
        std::cout<<"ran here"<<std::endl;
        comp sigma_vecbased1 = sigma_pvec_based(En,p,mi,total_P);
        comp qsq = q2psq_star(sig_p,mj,mk);
        comp qsq_vec_based1 = q2psq_star(sigma_vecbased1,mj,mk);
        comp cutoff1 = cutoff_function_1(sig_p,mj,mk,epsilon_h);
        comp cutoff2 = cutoff_function_1(sigma_vecbased1,mj,mk,epsilon_h);

        comp Ecm = E_to_Ecm(En,total_P);

        comp newsigma = (En - omega_func(spec_p,mi))*(En - omega_func(spec_p,mi)) - Pminusk*Pminusk; 
        comp check_sig_zero1 = abs(total_P_val - spec_p) + omega_func(spec_p,mi);
        comp check_sig_zero2 = -abs(total_P_val - spec_p) + omega_func(spec_p,mi);
        std::cout<<"E="<<En<<'\t'
                 <<"Ecm="<<Ecm<<'\t'
                 <<"sigp="<<sig_p<<'\t'
                 <<"qsq="<<qsq<<'\t'
                 <<"sigpV="<<sigma_vecbased1<<'\t'
                 //<<"newsigma="<<newsigma<<'\t'
                 <<"qsqV="<<qsq_vec_based1<<'\t'
                 <<"cut1="<<cutoff1<<'\t'
                 <<"cut2="<<cutoff2<<'\t'
                 <<"mjsq-mksq"<<std::abs(mj*mj - mk*mk)<<'\t'
                 <<"checksigzero1="<<E_to_Ecm(check_sig_zero1,total_P)<<'\t'
                 <<"checksigzero2="<<E_to_Ecm(check_sig_zero2,total_P)<<std::endl; 

        fout1<<En<<'\t'
            <<real(Ecm)<<'\t'
            <<imag(Ecm)<<'\t'
            <<real(sig_p)<<'\t'
            <<imag(sig_p)<<'\t'
            <<real(qsq)<<'\t'
            <<imag(qsq)<<'\t'
            <<real(cutoff1)<<'\t'
            <<imag(cutoff1)<<'\t'
            <<real(E_to_Ecm(check_sig_zero1,total_P))<<'\t'
            <<imag(E_to_Ecm(check_sig_zero1,total_P))<<'\t'
            <<real(E_to_Ecm(check_sig_zero2,total_P))<<'\t'
            <<imag(E_to_Ecm(check_sig_zero2,total_P))<<'\t'
            <<KKpi_threshold<<std::endl; 
        
    }
    fout1.close();

    /*===================================================*/
    
    
}


/* This code is the modified version of the checking code above
this code is to print the -F3inv to get the K3df_iso for different 
boosted P frames  */

void test_detF3inv_vs_En_KKpi()
{

    /*  Inputs  */
    
    double L = 20;
    double Lbyas = L;
    double xi = 1.0; 
    double xi1 = 3.444;/* found from lattice */
    L = L*xi1; // This is done to set the spatial 
                // unit in terms of a_t everywhere 
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

    for(int i=0;i<P_config_size;++i)
    {
        int nPx = nP_config[0][i];
        int nPy = nP_config[1][i];
        int nPz = nP_config[2][i];
    
        std::string filename =    "F3_for_pole_KKpi_L20_nP_"
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
        double KKKK_threshold = 4.2*atmK; 

        double En_initial = std::sqrt(KKpi_threshold*KKpi_threshold + 0.0000001 + abs(total_P_val*total_P_val));//.27;//0.4184939100000000245;//0.26302;
        double En_final = std::sqrt(KKKK_threshold*KKKK_threshold + abs(total_P_val*total_P_val));;
        double En_points = 25000;

        double delE = abs(En_initial - En_final)/En_points;

        std::ofstream fout; 
        fout.open(filename.c_str());

        for(int i=0; i<En_points; ++i)
        {
            double En = En_initial + i*delE; 

            std::vector< std::vector<comp> > p_config(3,std::vector<comp> ());
            double config_tolerance = 1.0e-5;
            config_maker_1(p_config, En, total_P, mi, mj, mk, L, xi, epsilon_h, config_tolerance );

            std::vector< std::vector<comp> > k_config = p_config; 


            int size = p_config[0].size();
            //std::cout<<"size = "<<size<<std::endl;  
            Eigen::MatrixXcd F3_mat;//(Eigen::Dynamic,Eigen::Dynamic);
            Eigen::MatrixXcd F2_mat;
            Eigen::MatrixXcd K2i_mat; 
            Eigen::MatrixXcd G_mat; 


            test_F3_ND_2plus1_mat(  F3_mat, F2_mat, K2i_mat, G_mat, En, p_config, k_config, total_P, eta_1, eta_2, scattering_length_1_piK, scattering_length_2_KK, atmpi, atmK, alpha, epsilon_h, L, xi, max_shell_num); 
            //std::cout<<"ran until here"<<std::endl;
    
            Eigen::MatrixXcd F3_mat_inv = F3_mat.inverse();
            //std::cout<<std::setprecision(3)<<"F3mat=\n"<<F3_mat<<std::endl; 
            //Eigen::MatrixXcd F3_mat_inv = F3_mat.inverse();
            //double res = det_F3_ND_2plus1_mat( En, p_config, k_config, total_P, eta_1, eta_2, scattering_length_1_piK, scattering_length_2_KK, atmpi, atmK, alpha, epsilon_h, L, xi, max_shell_num); 
            comp Ecm_calculated = E_to_Ecm(En, total_P);

            comp detF3 = F3_mat.determinant(); 
            comp sumF3 = F3_mat.sum(); 
            comp detF3inv = F3_mat_inv.determinant(); 
            comp sumF3inv = F3_mat_inv.sum(); 

            fout    << std::setprecision(20) 
                    << En << '\t' 
                    << real(Ecm_calculated) << '\t'
                    //<< real(F2_mat.sum()) << '\t'
                    //<< real(G_mat.sum()) << '\t'
                    //<< real(K2i_mat.sum()) << '\t'
                    //this is for F3 determinant
                    << real(detF3) << '\t'
                    << imag(detF3) << '\t'
                    //this is for F3inv determinant
                    //<< real(F3_mat_inv.determinant()) << '\t'
                    //this is for K3iso
                    //<< -real(F3_mat_inv.sum()) << std::endl;
                    //for F3iso 
                    << real(sumF3) << '\t'
                    << imag(sumF3) << '\t'
    
                    << real(detF3inv) << '\t'
                    << imag(detF3inv) << '\t'
                    << real(sumF3inv) << '\t'
                    << imag(sumF3inv) << std::endl;

            std::cout<<std::setprecision(20);
            std::cout<< "i = " << i << '\t'
                     << "En = " << En << '\t'
                     << "P = " << nPx << nPy << nPz << '\t' 
                     << "Ecm = " << Ecm_calculated << '\t' 
                     << "detF3 = " << F3_mat.determinant() << '\t'
                     << "F3iso = " << F3_mat.sum() << '\t'
                     //<< "det of F3inv = "<< real(F3_mat_inv.determinant()) << '\t' 
                     //<< "K3df_iso = "<< -real(F3_mat_inv.sum()) << std::endl;
                     << std::endl; 
        }
        fout.close();
    }               
}

void test_mass_dependences_F3_2plus1_vs_En()
{

    /*  Inputs  */
    
    double L = 5;
    double Lbyas = L;
    double xi = 1;//3.444; /* found from lattice */
    

    double scattering_length_1_piK = -2.0;//4.04;
    double scattering_length_2_KK = -2.0;//4.07;
    double eta_1 = 1.0;
    double eta_2 = 0.5; 
    double M1 = 1.0;
    double M2 = 0.99995;

    /*Since this code is copied from the function
    above, we make minimal changes, thats why intro
    duced M1 and M2 */
    double atmpi = M2;//0.06906;
    double atmK = M1;//0.09698;

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

    for(int i=0;i<P_config_size;++i)
    {
        int nPx = nP_config[0][i];
        int nPy = nP_config[1][i];
        int nPz = nP_config[2][i];
    
        std::string filename =    "mass_dependence_test_M1="
                                + std::to_string(M1)
                                + "_M2="
                                + std::to_string(M2)
                                + "_F3_L="
                                + std::to_string(L)
                                + "_nP_"
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
        double En_points = 4000;

        double delE = abs(En_initial - En_final)/En_points;

        std::ofstream fout; 
        fout.open(filename.c_str());

        for(int i=0; i<En_points+1; ++i)
        {
            double En = En_initial + i*delE; 

            std::vector< std::vector<comp> > p_config(3,std::vector<comp> ());
            double config_tolerance = 1.0e-5;
            config_maker_1(p_config, En, total_P, mi, mj, mk, L, xi, epsilon_h, config_tolerance );

            std::vector< std::vector<comp> > k_config = p_config; 


            int size = p_config[0].size();
            //std::cout<<"size = "<<size<<std::endl;  
            Eigen::MatrixXcd F3_mat;//(Eigen::Dynamic,Eigen::Dynamic);
            Eigen::MatrixXcd F2_mat;
            Eigen::MatrixXcd K2i_mat; 
            Eigen::MatrixXcd G_mat; 


            test_F3_ND_2plus1_mat(  F3_mat, F2_mat, K2i_mat, G_mat, En, p_config, k_config, total_P, eta_1, eta_2, scattering_length_1_piK, scattering_length_2_KK, atmpi, atmK, alpha, epsilon_h, L, xi, max_shell_num); 
            //std::cout<<"ran until here"<<std::endl;
    
            //std::cout<<std::setprecision(3)<<"F3mat=\n"<<F3_mat<<std::endl; 
            //Eigen::MatrixXcd F3_mat_inv = F3_mat.inverse();
            //double res = det_F3_ND_2plus1_mat( En, p_config, k_config, total_P, eta_1, eta_2, scattering_length_1_piK, scattering_length_2_KK, atmpi, atmK, alpha, epsilon_h, L, xi, max_shell_num); 
            comp Ecm_calculated = E_to_Ecm(En, total_P);
            fout    << std::setprecision(20) 
                    << En << '\t' 
                    << real(Ecm_calculated) << '\t'
                    << real(F2_mat.sum()) << '\t'
                    << real(G_mat.sum()) << '\t'
                    << real(K2i_mat.sum()) << '\t'
                    //this is for F3 determinant
                    << real(F3_mat.determinant()) << '\t'
                    //this is for F3inv determinant
                    //<< real(F3_mat_inv.determinant()) << '\t'
                    //this is for K3iso
                    //<< -real(F3_mat_inv.sum()) << std::endl;
                    //for F3iso 
                    << real(F3_mat.sum()) << std::endl;
            std::cout<<std::setprecision(20);
            std::cout<< "En = " << En << '\t'
                     << "P = " << nPx << nPy << nPz << '\t' 
                     << "Ecm = " << Ecm_calculated << '\t' 
                     << "detF3 = " << F3_mat.determinant() << '\t'
                     << "F3iso = " << F3_mat.sum() << '\t'
                     //<< "det of F3inv = "<< real(F3_mat_inv.determinant()) << '\t' 
                     //<< "K3df_iso = "<< -real(F3_mat_inv.sum()) << std::endl;
                     << std::endl; 
        }
        fout.close();
    }               
}


void test_F2_vs_sigp()
{
    /*  Inputs  */
    double pi = std::acos(-1.0);
    double L = 20;
    double Lbyas = L;

    double scattering_length_1_piK = -4.04;
    double scattering_length_2_KK = -4.07;
    double eta_1 = 1.0;
    double eta_2 = 0.5; 
    double atmpi = 0.06906;
    double atmK = 0.09698;

    //atmpi = atmpi/atmK; 
    //atmK = 1.0;
    double xi = 3.444;
    
    double mi = atmpi;//atmK;
    double mj = atmK;
    double mk = atmK;//atmpi; 


    double alpha = 0.5;
    double epsilon_h = 0.0;
    int max_shell_num = 20;

    comp twopibyL = 2.0*pi/L; 
    comp twopibyxiLbyas = 2.0*pi/(xi*Lbyas);

    double Ecm = atmpi + 2.0*atmK + 0.075;//2.9;//0.3184939100000000245;//0.26303;

    
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
    for(int i=0;i<P_config_size;++i)
    {
        int nPx = nP_config[0][i];
        int nPy = nP_config[1][i];
        int nPz = nP_config[2][i]; 

        std::string test_file1 =  "F2_vs_sigp_KKpi_2body_KK_P"
                                + std::to_string((int)nPx) 
                                + std::to_string((int)nPy)
                                + std::to_string((int)nPz)
                                + "_Ecm_" 
                                + std::to_string(Ecm) 
                                + ".dat"; //this is the test file for F2(p) data 

        std::ofstream fout1;
        fout1.open(test_file1.c_str());

        comp Px = ((comp) nPx)*twopibyxiLbyas;//twopibyL;
        comp Py = ((comp) nPy)*twopibyxiLbyas;//twopibyL;
        comp Pz = ((comp) nPz)*twopibyxiLbyas;//twopibyL;
        std::vector<comp> total_P(3);
        total_P[0] = Px; 
        total_P[1] = Py; 
        total_P[2] = Pz; 

        comp spec_P = std::sqrt(Px*Px + Py*Py + Pz*Pz); 
        comp total_P_val = spec_P; 

        double En = std::sqrt(Ecm*Ecm + abs(spec_P*spec_P)); 
        
        //double En_initial = 0.26302;
        //double En_final = 0.31;
        //double En_points = 10;

        //double delE = abs(En_initial - En_final)/En_points; 


        /*
        comp px = 0.0*twopibyL;
        comp py = 0.0*twopibyL;
        comp pz = 0.0*twopibyL;

        comp spec_p = std::sqrt(px*px + py*py + pz*pz); 
        std::vector<comp> p(3);
        p[0] = px; 
        p[1] = py; 
        p[2] = pz; 
        std::vector<std::vector<comp> > p_config(3,std::vector<comp>());
        p_config[0].push_back(px);
        p_config[1].push_back(py);
        p_config[2].push_back(pz);
    
        comp kx = 0.0*twopibyL;
        comp ky = 0.0*twopibyL;
        comp kz = 0.0*twopibyL;

        comp spec_k = std::sqrt(kx*kx + ky*ky + kz*kz); 
        std::vector<comp> k(3);
        k[0] = kx; 
        k[1] = ky; 
        k[2] = kz; 
        std::vector<std::vector<comp> > k_config(3,std::vector<comp>());
        k_config[0].push_back(kx);
        k_config[1].push_back(ky);
        k_config[2].push_back(kz);
    
        comp Pminusk_x = Px - kx; 
        comp Pminusk_y = Py - ky; 
        comp Pminusk_z = Pz - kz; 
        comp Pminusk = std::sqrt(Pminusk_x*Pminusk_x + Pminusk_y*Pminusk_y + Pminusk_z*Pminusk_z);

        std::cout<<std::setprecision(25);
        */


        /*
        int nmax = 20;
        std::vector<std::vector<int> > n_config(3,std::vector<int>());

        for(int i=-nmax;i<nmax+1;++i)
        {
            for(int j=-nmax;j<nmax+1;++j)
            {
                for(int k=-nmax;k<nmax+1;++k)
                {
                    int nsq = i*i + j*j + k*k;
                    if(nsq<=20)
                    {
                        n_config[0].push_back(i);
                        n_config[1].push_back(j);
                        n_config[2].push_back(k);
            
                    }
                }
            }
        }

        int config_size = n_config[0].size(); 

        */

        /*for(int i=0;i<config_size;++i)
        {
            int nx = n_config[0][i];
            int ny = n_config[1][i];
            int nz = n_config[2][i];

            int nsq = nx*nx + ny*ny + nz*nz; 
            comp px1 = twopibyL*((comp)nx);
            comp py1 = twopibyL*((comp)ny);
            comp pz1 = twopibyL*((comp)nz);

            std::vector<comp> p1(3);
            p1[0] = px1;
            p1[1] = py1;
            p1[2] = pz1; 

            comp spec_p = std::sqrt(px1*px1 + py1*py1 + pz1*pz1);
            comp sig_p = sigma_pvec_based(En, p1, mi, total_P);

            comp F2 = F2_i1(En, p1, p1, total_P, L, mi, mj, mk, alpha, epsilon_h, max_shell_num);

            fout1 << En <<'\t' << i << '\t' << nsq << '\t' << real(spec_p) << '\t' << imag(spec_p) << '\t'
                  << real(sig_p) << '\t' << imag(sig_p) << '\t' << real(F2) << '\t' << imag(F2) << std::endl; 

            std::cout << "i = " << i << '\t' 
                  << "nsq = " << nsq << '\t' 
                  << "spec_p = " << spec_p << '\t'
                  << "sig_p = " << sig_p << '\t' 
                  << "F2 = " << F2 << std::endl; 

        }*/

        double sigp_min = 0.0;
        double sigp_max = 6.0;
        double sigp_points = 100;
        double del_sigp = abs(sigp_max - sigp_min)/sigp_points; 

        double p_min = 0.0;
        double p_max = 0.5;
        double p_points = 1000;
        double del_p = abs(p_min - p_max)/p_points; 

        for(int i=0;i<p_points+1;++i)
        {
            double p_val = p_min + i*del_p; 

            int nx = p_val*L/(2.0*pi);//n_config[0][i];
            int ny = 0;//n_config[1][i];
            int nz = 0;//n_config[2][i];

            int nsq = nx*nx + ny*ny + nz*nz; 
            comp px1 = p_val;//twopibyL*((comp)nx);
            comp py1 = 0.0;//twopibyL*((comp)ny);
            comp pz1 = 0.0;//twopibyL*((comp)nz);

            std::vector<comp> p1(3);
            p1[0] = px1;
            p1[1] = py1;
            p1[2] = pz1; 

            comp spec_p = std::sqrt(px1*px1 + py1*py1 + pz1*pz1);
            comp sig_p = sigma_pvec_based(En, p1, mi, total_P);

            comp F2 = F2_i1(En, p1, p1, total_P, L, mi, mj, mk, alpha, epsilon_h, max_shell_num);
            double twobody_threshold = mj + mk ;
            fout1 << En <<'\t' << Ecm << '\t' << i << '\t' << nsq << '\t' 
                  << abs(mj*mj - mk*mk) << '\t' << twobody_threshold << '\t'
                  << real(spec_p) << '\t' << imag(spec_p) << '\t'
                  << real(sig_p) << '\t' << imag(sig_p) << '\t' << real(F2) << '\t' << imag(F2) << std::endl; 

            std::cout << "i = " << i << '\t' 
                      << "En = " << En << '\t'
                      << "Ecm = " << Ecm << '\t'
                      << "P = " << spec_P << '\t'
                      << "nsq = " << nsq << '\t' 
                      << "spec_p = " << spec_p << '\t'
                      << "sig_p = " << sig_p << '\t' 
                      << "F2 = " << F2 << std::endl; 

        }
        fout1.close();
    }
}

void test_F2_sum_func()
{
    /*  Inputs  */
    double pi = std::acos(-1.0);
    double L = 20;
    double Lbyas = L;

    double scattering_length_1_piK = -4.04;
    double scattering_length_2_KK = -4.07;
    double eta_1 = 1.0;
    double eta_2 = 0.5; 
    double atmpi = 0.06906;
    double atmK = 0.09698;

    //atmpi = atmpi/atmK; 
    //atmK = 1.0;
    double xi = 3.444;
    
    double mi = atmK;
    double mj = atmK;
    double mk = atmpi; 


    double alpha = 0.5;
    double epsilon_h = 0.0;
    int max_shell_num = 20;

    comp twopibyL = 2.0*pi/L; 
    comp twopibyxiLbyas = 2.0*pi/(xi*Lbyas);

    double Ecm = atmpi + 2.0*atmK + 0.075;//2.9;//0.3184939100000000245;//0.26303;
    
    
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

    char debug = 'n';
    double tolerance = 1.0e-11;

    comp px = 1.0*twopibyxiLbyas;
    comp py = 0.0*twopibyxiLbyas;
    comp pz = 0.0*twopibyxiLbyas;

    std::vector<comp> p(3); 
    p[0] = px; 
    p[1] = py; 
    p[2] = pz; 

    comp spec_p = std::sqrt(px*px + py*py + pz*pz);

    int nPx = 1;
    int nPy = 0;
    int nPz = 0; 

    comp Px = ((comp) nPx)*twopibyxiLbyas;
    comp Py = ((comp) nPy)*twopibyxiLbyas;
    comp Pz = ((comp) nPz)*twopibyxiLbyas;
    std::vector<comp> total_P(3);
    total_P[0] = Px; 
    total_P[1] = Py; 
    total_P[2] = Pz; 

    comp total_P_val = std::sqrt(Px*Px + Py*Py + Pz*Pz);

    double En = std::sqrt(Ecm*Ecm + abs(total_P_val*total_P_val));

    comp sigma_p = sigma_pvec_based(En, p, mi, total_P);


    comp gamma = (En - omega_func(spec_p,mi))/std::sqrt(sigma_p);
    //std::cout<<"gamma = "<<gamma<<std::endl;
    comp x = std::sqrt(q2psq_star(sigma_p,mj,mk))*(xi*L)/(2.0*pi);
    comp xii = 0.5*(1.0 + (mj*mj - mk*mk)/sigma_p); //this is another xi which is used inside the F sum function

    if(debug=='y')
    {
        std::cout<<"x = "<<x<<'\t'<<"sig_p = "<<sigma_p<<std::endl;

        //std::cout<<"x = "<<x<<'\t'<<"sig_p = "<<sigma_p<<std::endl;
    }

    comp npPx = (px - Px)*(xi*L)/(2.0*pi); 
    comp npPy = (py - Py)*(xi*L)/(2.0*pi);
    comp npPz = (pz - Pz)*(xi*L)/(2.0*pi);

    comp npP = std::sqrt(npPx*npPx + npPy*npPy + npPz*npPz);

    int c1 = 0;
    int c2 = 0; //these two are for checking if p and P are zero or not

    if(abs(spec_p)<1.0e-10 || abs(spec_p)==0.0 ) c1 = 1;
    if(abs(total_P_val)<1.0e-10 || abs(total_P_val)==0.0) c2 = 1;

    comp xibygamma = xii/gamma; 
    
    //int max_shell_num = 50;
    int na_x_initial = -max_shell_num;
    int na_x_final = +max_shell_num;
    int na_y_initial = -max_shell_num;
    int na_y_final = +max_shell_num;
    int na_z_initial = -max_shell_num;
    int na_z_final = +max_shell_num;

    comp summ = {0.0,0.0};
    comp temp_summ = {0.0,0.0};
    int counter = 0;

    std::string filename = "F2_sum_test_P" + std::to_string(nPx) + std::to_string(nPy) + std::to_string(nPz) + ".dat";
    std::ofstream fout;
    fout.open(filename.c_str());

    for(int i=na_x_initial;i<na_x_final+1;++i)
    {
        for(int j=na_y_initial;j<na_y_final+1;++j)
        {
            for(int k=na_z_initial;k<na_z_final+1;++k)
            {
                comp na = (comp) std::sqrt(i*i + j*j + k*k);

                comp nax = (comp) i;
                comp nay = (comp) j;
                comp naz = (comp) k;

                comp nax_npPx = nax*npPx;
                comp nay_npPy = nay*npPy;
                comp naz_npPz = naz*npPz; 

                comp na_dot_npP = nax_npPx + nay_npPy + naz_npPz;
                comp npPsq = npP*npP; 

                comp prod1 = ( (na_dot_npP/npPsq)*(1.0/gamma - 1.0) + xibygamma );
                
                comp rx = 0.0;
                comp ry = 0.0;
                comp rz = 0.0;
                
                if(c1==1 && c2==1)
                {
                    rx = nax;
                    ry = nay;
                    rz = naz; 
                }
                else if(abs(npPsq)==0)
                {
                    rx = nax;
                    ry = nay;
                    rz = naz;
                }
                else 
                {
                    rx = nax + npPx*prod1;
                    ry = nay + npPy*prod1;
                    rz = naz + npPz*prod1;
                }

                comp r = std::sqrt(rx*rx + ry*ry + rz*rz);

                
                
                summ = summ + std::exp(alpha*(x*x - r*r))/(x*x - r*r);

                if(debug=='y')
                {
                    //std::cout<<i<<'\t'<<j<<'\t'<<k<<'\t'<<x*x - r*r<<'\t'<<prod1<<'\t'<<summ<<std::endl;
                    if(!std::isnan(abs(prod1)))
                    {
                        //std::cout<<"npP = "<<npP<<'\t'
                        //     <<"prod1 = "<<prod1<<'\t'
                        //     <<"r = "<<r<<'\t'<<"rx = "<<rx<<'\t'<<"ry = "<<ry<<'\t'<<"rz = "<<rz<<std::endl;
                    }
                }
                //if(abs(summ - temp_summ)<tolerance)
                //{
                    //std::cout<<"sum broken at: i="<<i<<'\t'<<"j="<<j<<'\t'<<"k="<<k<<std::endl;
                    //break;
                //}
                //temp_summ = summ;

                fout << counter << '\t' << real(na*na) << '\t'
                     << real(x*x) << '\t' << imag(x*x) << '\t' 
                     << real(r*r) << '\t' << imag(r*r) << '\t'
                     << real(x*x - r*r) << '\t' << imag(x*x - r*r) << std::endl; 

                std::cout << counter << '\t' << na*na << '\t'
                          << real(x*x) << '\t' << imag(x*x) << '\t' 
                          << real(r*r) << '\t' << imag(r*r) << '\t'
                          << real(x*x - r*r) << '\t' << imag(x*x - r*r) << std::endl; 
                std::cout << prod1 << '\t' << gamma << '\t' << xibygamma << std::endl; 
                counter = counter + 1 ;


            }
        }
    }
    fout.close(); 
}

/* Here we test the modified denominator of the F3 function, we write 
the denominator as 1 + (K2inv + G)Finv or 1 + K2(F2+G) or (K2inv + F2 + G)^-1
to check for additional poles coming in F3 */

void test_additionalpoles_in_F3_vs_En_KKpi()
{

    /*  Inputs  */
    
    double L = 20;
    double Lbyas = L;
    double xi = 3.444; /* found from lattice */
    

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

    for(int i=0;i<P_config_size;++i)
    {
        int nPx = nP_config[0][i];
        int nPy = nP_config[1][i];
        int nPz = nP_config[2][i];
    
        std::string filename =   "additional_poles3_F3_KKpi_L20_nP_" //"F2_check_poles_L20_nP"
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
        double KKKK_threshold = 4.0*atmK; 

        double En_initial = std::sqrt(KKpi_threshold*KKpi_threshold + 0.001 + abs(total_P_val*total_P_val));//.27;//0.4184939100000000245;//0.26302;
        double En_final = std::sqrt(KKKK_threshold*KKKK_threshold + abs(total_P_val*total_P_val));;
        double En_points = 4000;

        double delE = abs(En_initial - En_final)/En_points;

        std::ofstream fout; 
        fout.open(filename.c_str());

        for(int i=1; i<En_points; ++i)
        {
            double En = En_initial + i*delE; 

            std::vector< std::vector<comp> > p_config(3,std::vector<comp> ());
            double config_tolerance = 1.0e-5;
            config_maker_1(p_config, En, total_P, mi, mj, mk, L, xi, epsilon_h, config_tolerance );

            std::vector< std::vector<comp> > k_config = p_config; 


            int size = p_config[0].size();
            //std::cout<<"size = "<<size<<std::endl;  
            Eigen::MatrixXcd F3_mat;//(Eigen::Dynamic,Eigen::Dynamic);
            Eigen::MatrixXcd F2_mat;
            Eigen::MatrixXcd K2i_mat; 
            Eigen::MatrixXcd G_mat; 


            //testF3_additionalpoles_1(  F3_mat, F2_mat, K2i_mat, G_mat, En, p_config, k_config, total_P, eta_1, eta_2, scattering_length_1_piK, scattering_length_2_KK, atmpi, atmK, alpha, epsilon_h, L, xi, max_shell_num); 
            //testF3_additionalpoles_2(  F3_mat, F2_mat, K2i_mat, G_mat, En, p_config, k_config, total_P, eta_1, eta_2, scattering_length_1_piK, scattering_length_2_KK, atmpi, atmK, alpha, epsilon_h, L, xi, max_shell_num); 
            testF3_additionalpoles_3(  F3_mat, F2_mat, K2i_mat, G_mat, En, p_config, k_config, total_P, eta_1, eta_2, scattering_length_1_piK, scattering_length_2_KK, atmpi, atmK, alpha, epsilon_h, L, xi, max_shell_num); 
            
            //std::cout<<"ran until here"<<std::endl;
    
            //std::cout<<std::setprecision(3)<<"F3mat=\n"<<F3_mat<<std::endl; 
            //Eigen::MatrixXcd F3_mat_inv = F3_mat.inverse();
            //double res = det_F3_ND_2plus1_mat( En, p_config, k_config, total_P, eta_1, eta_2, scattering_length_1_piK, scattering_length_2_KK, atmpi, atmK, alpha, epsilon_h, L, xi, max_shell_num); 
            comp Ecm_calculated = E_to_Ecm(En, total_P);
            fout    << std::setprecision(20) 
                    << En << '\t' 
                    << real(Ecm_calculated) << '\t'
                    << imag(Ecm_calculated) << '\t'
                    << real(F2_mat.determinant()) << '\t'
                    << real(F2_mat.sum()) << '\t'
                    << real(G_mat.determinant()) << '\t'
                    << real(G_mat.sum()) << '\t'
                    << real(K2i_mat.determinant()) << '\t'
                    << real(K2i_mat.sum()) << '\t'
                    //this is for F3 determinant
                    << real(F3_mat.determinant()) << '\t'
                    //this is for F3inv determinant
                    //<< real(F3_mat_inv.determinant()) << '\t'
                    //this is for K3iso
                    //<< -real(F3_mat_inv.sum()) << std::endl;
                    //for F3iso 
                    << real(F3_mat.sum()) << std::endl;
            std::cout<<std::setprecision(20);
            std::cout<< "En = " << En << '\t'
                     << "P = " << nPx << nPy << nPz << '\t' 
                     << "Ecm = " << Ecm_calculated << '\t' 
                     << "detF3 = " << F3_mat.determinant() << '\t'
                     << "F3iso = " << F3_mat.sum() << '\t'
                     //<< "det of F3inv = "<< real(F3_mat_inv.determinant()) << '\t' 
                     //<< "K3df_iso = "<< -real(F3_mat_inv.sum()) << std::endl;
                     << std::endl; 
        }
        fout.close();
    }               
}


/* Here we test the F3 functions for identical 3 particles with ma set to 
very small value 1.01, we then dial ma and see the gradual change of the spectrum, all the 
change will be done in QC function = test_F3_ID */
void test_F3_ID_printer()
{

    /*  Inputs  */
    
    double L = 5;
    double Lbyas = L;
    double xi = 1;//3.444; /* found from lattice */
    

    double scattering_length_1_piK = 1.01;//-4.04;
    double scattering_length_2_KK =  1.01;//-4.07;
    double eta_1 = 0.5;
    double eta_2 = 0.5;//0.5; 
    double atmpi = 1.0;//0.06906;
    double atmK = 1.0;//0.09698;

    //atmpi = atmpi/atmK; 
    //atmK = 1.0;
    

    double alpha = 0.5;
    double epsilon_h = 0.0;
    int max_shell_num = 20;
    int nmax = 4;
    int nsq_max = 4; 

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

                    if(i<=j && j<=k)
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

    //abort(); 
    int P_config_size = nP_config[0].size();

    /*-----------------------------------------------------*/

    for(int i=0;i<P_config_size;++i)
    {
        int nPx = nP_config[0][i];
        int nPy = nP_config[1][i];
        int nPz = nP_config[2][i];
    
        std::string filename =   "F3_ma=" 
                                + std::to_string((int)scattering_length_1_piK)
                                + "_L20_nP_" //"F2_check_poles_L20_nP"
                                + std::to_string((int)nPx)
                                + std::to_string((int)nPy)
                                + std::to_string((int)nPz)
                                + ".dat";

        std::string nonint_file = "non_int_3body_nP"
                                + std::to_string((int)nPx)
                                + std::to_string((int)nPy)
                                + std::to_string((int)nPz)
                                + ".dat";

        std::string gpole_file = "Gpole_3body_nP"
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

        double En_initial = std::sqrt(KKpi_threshold*KKpi_threshold + 0.00000001 + abs(total_P_val*total_P_val));//.27;//0.4184939100000000245;//0.26302;
        double En_final = std::sqrt(KKKK_threshold*KKKK_threshold + abs(total_P_val*total_P_val));;
        double En_points = 4000;

        double delE = abs(En_initial - En_final)/En_points;

        //threebody_non_int_spectrum(nonint_file, mi, mj, mk, total_P, xi, Lbyas, nmax, nsq_max);
        //threebody_Gpoles(gpole_file, mi, mj, mk, total_P, xi, Lbyas, nmax, nsq_max);

        std::ofstream fout; 
        fout.open(filename.c_str());

        for(int i=0; i<En_points+1; ++i)
        {
            double En = En_initial + i*delE; 

            Eigen::MatrixXcd F3_mat;//(Eigen::Dynamic,Eigen::Dynamic);
            Eigen::MatrixXcd F2_mat;
            Eigen::MatrixXcd K2i_mat; 
            Eigen::MatrixXcd G_mat; 


            test_F3_ID_zeroK2(  F3_mat, F2_mat, K2i_mat, G_mat, En, total_P, eta_1, eta_2, scattering_length_1_piK, scattering_length_2_KK, atmpi, atmK, alpha, epsilon_h, L, xi, max_shell_num); 
            
            Eigen::MatrixXcd K2iplusFplusG = K2i_mat + 0.5*F2_mat + G_mat; 

            F2_mat = 0.5*F2_mat; 
            
            comp Ecm_calculated = E_to_Ecm(En, total_P);
            fout    << std::setprecision(20) 
                    << En << '\t' 
                    << real(Ecm_calculated) << '\t'
                    << imag(Ecm_calculated) << '\t'
                    << real(F2_mat.determinant()) << '\t'
                    << real(F2_mat.sum()) << '\t'
                    << real(G_mat.determinant()) << '\t'
                    << real(G_mat.sum()) << '\t'
                    << real(K2iplusFplusG.sum()) << '\t'
                    << real(K2i_mat.sum()) << '\t'
                    //this is for F3 determinant
                    << real(F3_mat.determinant()) << '\t'
                    //this is for F3inv determinant
                    //<< real(F3_mat_inv.determinant()) << '\t'
                    //this is for K3iso
                    //<< -real(F3_mat_inv.sum()) << std::endl;
                    //for F3iso 
                    << real(F3_mat.sum()) << std::endl;
            std::cout<<std::setprecision(20);
            std::cout<< "En = " << En << '\t'
                     << "P = " << nPx << nPy << nPz << '\t' 
                     << "Ecm = " << Ecm_calculated << '\t' 
                     << "detF3 = " << F3_mat.determinant() << '\t'
                     << "F3iso = " << F3_mat.sum() << '\t'
                     //<< "det of F3inv = "<< real(F3_mat_inv.determinant()) << '\t' 
                     //<< "K3df_iso = "<< -real(F3_mat_inv.sum()) << std::endl;
                     << std::endl; 
        }
        fout.close();
    }               
}

void test_3body_non_int()
{
    /*  Inputs  */
    
    double L = 20;
    double Lbyas = L;
    double xi = 3.449;//3.444; /* found from lattice */
    int nmax = 20; 
    int nsq_max = 4;
    

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

    std::string filename = "3body_non_int_test.dat";

    int nPx = 1;
    int nPy = 1; 
    int nPz = 0; 
    std::vector<comp> total_P(3); 
    total_P[0] = ((comp)nPx)*twopibyxiLbyas; 
    total_P[1] = ((comp)nPy)*twopibyxiLbyas; 
    total_P[2] = ((comp)nPz)*twopibyxiLbyas; 
    threebody_non_int_spectrum(filename, m1, m2, m3, total_P, xi, Lbyas, nmax, nsq_max);
}

void poles_of_G_in_Ecm()
{
    /*  Inputs  */
    
    double L = 5;
    double Lbyas = L;
    double xi = 1;//3.444; /* found from lattice */
    

    double scattering_length_1_piK = -1000000000;//-4.04;
    double scattering_length_2_KK =  -1000000000;//-4.07;
    double eta_1 = 1.0;
    double eta_2 = 1.0;//0.5; 
    double atmpi = 1.0;//0.06906;
    double atmK = 1.0;//0.09698;

    //atmpi = atmpi/atmK; 
    //atmK = 1.0;
    

    double alpha = 0.5;
    double epsilon_h = 0.0;
    int max_shell_num = 20;
    int nmax = 4;
    int nsq_max = 4; 

    double pi = std::acos(-1.0); 
    comp twopibyL = 2.0*pi/L;
    comp twopibyxiLbyas = 2.0*pi/(xi*Lbyas);

    /*---------------------------------------------------*/


}

void test_Gmat_vs_sigp()
{
    /*  Inputs  */
    
    double L = 5;
    double Lbyas = L;
    double xi = 1;//3.444; /* found from lattice */
    

    double scattering_length_1_piK = -1000000000;//-4.04;
    double scattering_length_2_KK =  -1000000000;//-4.07;
    double eta_1 = 1.0;
    double eta_2 = 1.0;//0.5; 
    double atmpi = 1.0;//0.06906;
    double atmK = 1.0;//0.09698;

    //atmpi = atmpi/atmK; 
    //atmK = 1.0;
    
    double mi = atmK; 
    double mj = atmK; 
    double mk = atmpi; 

    double alpha = 0.5;
    double epsilon_h = 0.0;
    int max_shell_num = 20;
    int nmax = 4;
    int nsq_max = 4; 

    double pi = std::acos(-1.0); 
    comp twopibyL = 2.0*pi/L;
    comp twopibyxiLbyas = 2.0*pi/(xi*Lbyas);

    /*---------------------------------------------------*/

    int nPx = 1;
    int nPy = 0;
    int nPz = 0; 

    comp Px = ((comp)nPx)*twopibyxiLbyas;
    comp Py = ((comp)nPy)*twopibyxiLbyas;
    comp Pz = ((comp)nPz)*twopibyxiLbyas; 

    std::vector<comp> total_P(3); 
    total_P[0] = Px; 
    total_P[1] = Py; 
    total_P[2] = Pz; 

    comp spec_P = std::sqrt(Px*Px + Py*Py + Pz*Pz); 

    int npx = 1;
    int npy = 1; 
    int npz = 0; 

    comp px = ((comp)npx)*twopibyxiLbyas; 
    comp py = ((comp)npy)*twopibyxiLbyas; 
    comp pz = ((comp)npz)*twopibyxiLbyas;

    comp spec_p = std::sqrt(px*px + py*py + pz*pz); 
    std::vector<comp> p(3);
    p[0] = px; 
    p[1] = py; 
    p[2] = pz; 

    double Eninitial = 3.05;
    double Enfinal = 3.2; 
    double Enpoints = 5; 
    double delEn = std::abs(Eninitial - Enfinal)/Enpoints; 

    double kxinitial = 0.0;
    double kxfinal = 2.0; 
    double kxpoints = 500;
    double delkx = std::abs(kxinitial - kxfinal)/kxpoints; 

    for(int i=0; i<Enpoints+1; ++i)
    {
        double En = Eninitial + i*delEn; 

        std::string filename = "Gmat_vs_sigk_En_"
                                + std::to_string(En)
                                + "_nP_" 
                                + std::to_string(nPx) 
                                + std::to_string(nPy)
                                + std::to_string(nPz)
                                + "_np_"
                                + std::to_string(npx) 
                                + std::to_string(npy)
                                + std::to_string(npz)
                                + ".dat" ;
        std::ofstream fout; 
        fout.open(filename.c_str());

        for(int j=0; j<kxpoints; ++j)
        {
            double kx = kxinitial + j*delkx; 
            double ky = 0.0;
            double kz = 0.0; 

            std::vector<comp> k(3); 
            k[0] = kx; 
            k[1] = ky; 
            k[2] = kz; 

            comp sigk = sigma_pvec_based(En, k, mi, total_P); 

            comp cutoff = cutoff_function_1(sigk, mj, mk, epsilon_h);
            comp Gval = G_ij(En, p, k, total_P, mi, mj, mk, L, epsilon_h);

            std::cout<<En<<'\t'
                     <<kx<<'\t'
                     <<real(cutoff)<<'\t'
                     <<imag(cutoff)<<'\t'
                     <<real(Gval)<<'\t'
                     <<imag(Gval)<<std::endl;

            fout     <<En<<'\t'
                     <<kx<<'\t'
                     <<real(cutoff)<<'\t'
                     <<imag(cutoff)<<'\t'
                     <<real(Gval)<<'\t'
                     <<imag(Gval)<<std::endl;


        }
        fout.close(); 
    }
}

//Basically this prints a list of p values 
//in terms of 2pi/(xi L) times n, created to point
//out on p-axis where the spectator momentum would be

void p_in_lattice_units()
{
    std::ofstream fout; 
    std::string filename = "p_in_lattice_unit.dat";
    
    double pi = std::acos(-1.0); 
    double L = 5.0;
    double xi = 1.0;
    double twopibyxiL = 2.0*pi/(xi*L);

    int nmax = 20; 

    fout.open(filename.c_str()); 

    for(int i=-nmax; i<nmax+1; ++i)
    {
        for(int j=-nmax; j<nmax+1; ++j)
        {
            for(int k=-nmax; k<nmax+1; ++k)
            {
                int nsq = i*i + j*j + k*k; 

                comp px = ((comp)i)*twopibyxiL; 
                comp py = ((comp)j)*twopibyxiL; 
                comp pz = ((comp)k)*twopibyxiL;

                comp psq = px*px + py*py + pz*pz; 
                comp p_val = std::sqrt(psq); 

                fout<<real(p_val)<<'\t'<<imag(p_val)<<std::endl; 
            }
        }
    }
    fout.close();
    std::cout<<"file = "<<filename<<" generated."<<std::endl; 

}


/* Here we check the number of shells that are activated per energy */
void activated_shell(   int nPx, int nPy, int nPz )
{
    /*  Inputs  */
    
    double L = 5;
    double Lbyas = L;
    double xi = 1;//3.444; /* found from lattice */
    

    double scattering_length_1_piK = -1000000000;//-4.04;
    double scattering_length_2_KK =  -1000000000;//-4.07;
    double eta_1 = 1.0;
    double eta_2 = 1.0;//0.5; 
    double atmpi = 1.0;//0.06906;
    double atmK = 1.0;//0.09698;

    //atmpi = atmpi/atmK; 
    //atmK = 1.0;
    
    double mi = atmK; 
    double mj = atmK; 
    double mk = atmpi; 

    double alpha = 0.5;
    double epsilon_h = 0.0;
    int max_shell_num = 20;
    int nmax = 4;
    int nsq_max = 4; 

    double pi = std::acos(-1.0); 
    comp twopibyL = 2.0*pi/L;
    comp twopibyxiLbyas = 2.0*pi/(xi*Lbyas);

    /*---------------------------------------------------*/

    //int nPx = 1;
    //int nPy = 0;
    //int nPz = 0; 

    comp Px = ((comp)nPx)*twopibyxiLbyas;
    comp Py = ((comp)nPy)*twopibyxiLbyas;
    comp Pz = ((comp)nPz)*twopibyxiLbyas; 

    std::vector<comp> total_P(3); 
    total_P[0] = Px; 
    total_P[1] = Py; 
    total_P[2] = Pz; 

    comp spec_P = std::sqrt(Px*Px + Py*Py + Pz*Pz); 

    int npx = 0;
    int npy = 0; 
    int npz = 0; 

    comp px = ((comp)npx)*twopibyxiLbyas; 
    comp py = ((comp)npy)*twopibyxiLbyas; 
    comp pz = ((comp)npz)*twopibyxiLbyas;

    comp spec_p = std::sqrt(px*px + py*py + pz*pz); 
    std::vector<comp> p(3);
    p[0] = px; 
    p[1] = py; 
    p[2] = pz; 

    double Ecm_initial = 3.0;
    double Ecm_final = 4.0; 
    
    double En_initial = real(Ecm_to_E(Ecm_initial, total_P));
    double En_final = real(Ecm_to_E(Ecm_final, total_P)); 
    double En_points = 1000.0;
    double del_En = abs(En_initial - En_final)/En_points; 

    std::string filename = "activated_shell_P_" 
                            + std::to_string(nPx)
                            + std::to_string(nPy) 
                            + std::to_string(nPz)
                            + ".dat"; 

    std::ofstream fout; 
    fout.open(filename.c_str()); 


    for(int i=0; i<En_points; ++i)
    {
        double En = En_initial + i*del_En; 
        std::vector< std::vector<comp> > p_config1(3, std::vector<comp> ());
        std::vector< std::vector<int> > n_config(3, std::vector<int> ());
        double config_tolerance = 1.0e-5;

        config_maker_2(p_config1, n_config, En, total_P, mi, mj, mk, L, xi, epsilon_h, config_tolerance );

        int max_nsq = 0;
        int max_n = 0; 

        for(int j=0; j<n_config[0].size(); ++j)
        {
            int nx = n_config[0][j];
            int ny = n_config[1][j]; 
            int nz = n_config[2][j]; 

            int nsq = nx*nx + ny*ny + nz*nz; 
            int n = std::sqrt(nsq); 

            if(nsq>max_nsq) max_nsq = nsq; 
            if(n>max_n) max_n = n; 
        }

        std::cout<<En<<'\t'<<max_nsq<<'\t'<<max_n<<std::endl;

        comp Ecm = E_to_Ecm(En, total_P); 

        fout << real(Ecm) << '\t' << En << '\t' << max_nsq << '\t' << n_config[0].size() << std::endl; 
 
    }
    fout.close(); 
    
}


/* Here we test individual matrices of F2, G, K2i and F3 for a fixed 
energy with the FRL code base, we first check if the config maker is 
generating the same number of momentum configurations for the kinematic 
point, we then hardcode the spectator momentum as it is done in FRL and 
check if the generated matrices are the same for both cases  */

void pvec_by_hand(  std::vector< std::vector<comp> > &pvec,
                    double xi, 
                    double L, 
                    int npx, 
                    int npy, 
                    int npz )
{
    double pi = std::acos(-1.0); 

    comp px = 2.0*pi/(xi*L)*((comp) npx);
    comp py = 2.0*pi/(xi*L)*((comp) npy);
    comp pz = 2.0*pi/(xi*L)*((comp) npz);
    
    pvec[0].push_back(px); 
    pvec[1].push_back(py); 
    pvec[2].push_back(pz); 

}

void test_functions_with_FRL_codebase_ID()
{
    /*  Inputs  */
    
    double L = 5;
    double Lbyas = L;
    double xi = 1;//3.444; /* found from lattice */
    

    double scattering_length_1_piK = -2;//-4.04;
    double scattering_length_2_KK =  -2;//-4.07;
    double eta_1 = 0.5;
    double eta_2 = 1.0;//0.5; 
    double atmpi = 1.0;//0.06906;
    double atmK = 1.0;//0.09698;

    //atmpi = atmpi/atmK; 
    //atmK = 1.0;
    
    double mi = atmK; 
    double mj = atmK; 
    double mk = atmpi; 

    double alpha = 0.5;
    double epsilon_h = 0.0;
    int max_shell_num = 20;
    int nmax = 4;
    int nsq_max = 4; 

    double pi = std::acos(-1.0); 
    comp twopibyL = 2.0*pi/L;
    comp twopibyxiLbyas = 2.0*pi/(xi*Lbyas);

    /*---------------------------------------------------*/

    int nPx = 0;
    int nPy = 0;
    int nPz = 1; 

    comp Px = ((comp)nPx)*twopibyxiLbyas;
    comp Py = ((comp)nPy)*twopibyxiLbyas;
    comp Pz = ((comp)nPz)*twopibyxiLbyas; 

    std::vector<comp> total_P(3); 
    total_P[0] = Px; 
    total_P[1] = Py; 
    total_P[2] = Pz; 

    comp spec_P = std::sqrt(Px*Px + Py*Py + Pz*Pz); 

    int npx = 0;
    int npy = 0; 
    int npz = 0; 

    comp px = ((comp)npx)*twopibyxiLbyas; 
    comp py = ((comp)npy)*twopibyxiLbyas; 
    comp pz = ((comp)npz)*twopibyxiLbyas;

    comp spec_p = std::sqrt(px*px + py*py + pz*pz); 
    std::vector<comp> p(3);
    p[0] = px; 
    p[1] = py; 
    p[2] = pz; 

    double Ecm_initial = 3.0;
    double Ecm_final = 4.0; 
    
    double En_initial = real(Ecm_to_E(Ecm_initial, total_P));
    double En_final = real(Ecm_to_E(Ecm_final, total_P)); 
    double En_points = 1000.0;
    double del_En = abs(En_initial - En_final)/En_points; 

    double Ecm = 3.15; 
    comp En = Ecm_to_E(Ecm, total_P);

    std::vector<std::vector<comp> > p_config(3, std::vector<comp>()); 
    std::vector<std::vector<int> > n_config(3, std::vector<int>()); 
    double tolerance = 1.0e-10; 

    config_maker_2(p_config, n_config, En, total_P, mi, mj, mk, L, xi, epsilon_h, tolerance);

    for(int i=0; i<n_config[0].size(); ++i)
    {
        int nx = n_config[0][i]; 
        int ny = n_config[1][i]; 
        int nz = n_config[2][i]; 

        std::cout << "n config = [" << nx << '\t' << ny << '\t' << nz  << "]" << std::endl; 
    } 

    std::vector<std::vector<comp> > p_config_by_hand(3, std::vector<comp>()); 

    pvec_by_hand(p_config_by_hand, xi, L, 0, 0, 0); 
    pvec_by_hand(p_config_by_hand, xi, L, 0, 0, 1);
    pvec_by_hand(p_config_by_hand, xi, L, 1, 0, 1);
    pvec_by_hand(p_config_by_hand, xi, L,-1, 0, 1);
    pvec_by_hand(p_config_by_hand, xi, L, 0, 1, 1);
    pvec_by_hand(p_config_by_hand, xi, L, 0,-1, 1);

    int size1 = p_config_by_hand[0].size(); 
    Eigen::MatrixXcd F2_mat_1(size1,size1);
    F2_i_mat_1( F2_mat_1, En, p_config_by_hand, p_config_by_hand, total_P, mi, mj, mk, L, xi, alpha, epsilon_h, max_shell_num );
    
    std::cout<<"F2 = "<<std::endl; 
    std::cout<<0.5*F2_mat_1<<std::endl;
    std::cout<<"====================================="<<std::endl; 

    Eigen::MatrixXcd G_mat_11(size1,size1);
    G_ij_mat( G_mat_11, En, p_config_by_hand, p_config_by_hand, total_P, mi, mj, mk, L, epsilon_h ); 

    std::cout<<"G = "<<std::endl; 
    std::cout<<G_mat_11<<std::endl;
    std::cout<<"====================================="<<std::endl; 

    Eigen::MatrixXcd K2inv_mat_1(size1,size1);
    K2inv_i_mat( K2inv_mat_1, eta_1, scattering_length_1_piK, En, p_config_by_hand, p_config_by_hand, total_P, mi, mj, mk, epsilon_h, L );
    
    std::cout<<"K2i = "<<std::endl; 
    std::cout<<K2inv_mat_1<<std::endl;
    std::cout<<"====================================="<<std::endl; 

    Eigen::MatrixXcd F3mat(size1, size1);

    Eigen::MatrixXcd temp_identity_mat(size1,size1);
    temp_identity_mat.setIdentity();
    double relerror = 0; 

    Eigen::MatrixXcd H_mat =  K2inv_mat_1 + 0.5*F2_mat_1 + G_mat_11;
    
    Eigen::MatrixXcd H_mat_inv(size1, size1); 

    Eigen::MatrixXcd NewF2 = 0.5*F2_mat_1;

    LinearSolver_4(H_mat, H_mat_inv, temp_identity_mat, relerror);

    Eigen::MatrixXcd temp1 = H_mat_inv*NewF2;
    Eigen::MatrixXcd temp2 = NewF2*temp1; 
    Eigen::MatrixXcd temp3 = NewF2/3.0; 

    //F3mat = (F2_mat_1/3.0 - F2_mat_1*H_mat.inverse()*F2_mat_1);//temp_F3_mat; 

    F3mat = temp3 - temp2;//temp_F3_mat; 

    std::cout<<"F3 = "<<std::endl; 
    std::cout<<F3mat<<std::endl; 
    std::cout<<"====================================="<<std::endl; 


    std::cout<<"Hmat inv = "<<std::endl; 
    std::cout<<H_mat_inv<<std::endl; 
    std::cout<<"====================================="<<std::endl; 

    std::cout<<"check inv = "<<std::endl;
    std::cout<<H_mat*H_mat_inv<<std::endl; 
    std::cout<<"====================================="<<std::endl; 

}

void test_functions_with_FRL_codebase_2plus1()
{
    /*  Inputs  */
    
    double L = 5;
    double Lbyas = L;
    double xi = 1;//3.444; /* found from lattice */
    

    double scattering_length_1_piK = 4.04;
    double scattering_length_2_KK =  4.07;
    double eta_1 = 1.0;
    double eta_2 = 0.5;//0.5; 
    double atmpi = 0.5;//0.06906;
    double atmK = 1.0;//0.09698;

    //atmpi = atmpi/atmK; 
    //atmK = 1.0;
    
    double mi = atmK; 
    double mj = atmK; 
    double mk = atmpi; 

    double alpha = 0.5;
    double epsilon_h = 0.0;
    int max_shell_num = 20;
    int nmax = 4;
    int nsq_max = 4; 

    double pi = std::acos(-1.0); 
    comp twopibyL = 2.0*pi/L;
    comp twopibyxiLbyas = 2.0*pi/(xi*Lbyas);

    /*---------------------------------------------------*/

    int nPx = 1;
    int nPy = 1;
    int nPz = 1; 

    comp Px = ((comp)nPx)*twopibyxiLbyas;
    comp Py = ((comp)nPy)*twopibyxiLbyas;
    comp Pz = ((comp)nPz)*twopibyxiLbyas; 

    std::vector<comp> total_P(3); 
    total_P[0] = Px; 
    total_P[1] = Py; 
    total_P[2] = Pz; 

    comp spec_P = std::sqrt(Px*Px + Py*Py + Pz*Pz); 

    int npx = 0;
    int npy = 0; 
    int npz = 0; 

    comp px = ((comp)npx)*twopibyxiLbyas; 
    comp py = ((comp)npy)*twopibyxiLbyas; 
    comp pz = ((comp)npz)*twopibyxiLbyas;

    comp spec_p = std::sqrt(px*px + py*py + pz*pz); 
    std::vector<comp> p(3);
    p[0] = px; 
    p[1] = py; 
    p[2] = pz; 

    double Ecm_initial = 3.0;
    double Ecm_final = 4.0; 
    
    double En_initial = real(Ecm_to_E(Ecm_initial, total_P));
    double En_final = real(Ecm_to_E(Ecm_final, total_P)); 
    double En_points = 1000.0;
    double del_En = abs(En_initial - En_final)/En_points; 

    double Ecm = 2.51; 
    comp En = Ecm_to_E(Ecm, total_P);

    std::vector<std::vector<comp> > p_config(3, std::vector<comp>()); 
    std::vector<std::vector<comp> > k_config(3, std::vector<comp>());
    std::vector<std::vector<int> > n_config1(3, std::vector<int>()); 
    std::vector<std::vector<int> > n_config2(3, std::vector<int>()); 
    double tolerance = 1.0e-10; 

    config_maker_2(p_config, n_config1, En, total_P, mi, mj, mk, L, xi, epsilon_h, tolerance);
    config_maker_2(k_config, n_config2, En, total_P, mk, mi, mj, L, xi, epsilon_h, tolerance);

    for(int i=0; i<n_config1[0].size(); ++i)
    {
        int nx = n_config1[0][i]; 
        int ny = n_config1[1][i]; 
        int nz = n_config1[2][i]; 

        std::cout << "n config1 = [" << nx << '\t' << ny << '\t' << nz  << "]" << std::endl; 
    } 

    for(int i=0; i<n_config2[0].size(); ++i)
    {
        int nx = n_config2[0][i]; 
        int ny = n_config2[1][i]; 
        int nz = n_config2[2][i]; 

        std::cout << "n config2 = [" << nx << '\t' << ny << '\t' << nz  << "]" << std::endl; 
    } 

    //abort(); 
    
    std::vector<std::vector<comp> > p_config_by_hand(3, std::vector<comp>()); 
    std::vector<std::vector<comp> > k_config_by_hand(3, std::vector<comp>()); 

    p_config_by_hand = p_config; 
    k_config_by_hand = k_config; 

    /*
    pvec_by_hand(p_config_by_hand, xi, L, 0, 0, 0); 
    pvec_by_hand(p_config_by_hand, xi, L, 1, 1, 0);
    
    pvec_by_hand(k_config_by_hand, xi, L, 0, 0, 0);
    pvec_by_hand(k_config_by_hand, xi, L, 0, 1, 0);
    pvec_by_hand(k_config_by_hand, xi, L, 1, 0, 0);
    pvec_by_hand(k_config_by_hand, xi, L, 1, 1, 0);
    */
    /*pvec_by_hand(k_config_by_hand, xi, L,-1, 0, 0);
    pvec_by_hand(k_config_by_hand, xi, L, 0, 1, 0);
    pvec_by_hand(k_config_by_hand, xi, L, 0,-1, 0);
    pvec_by_hand(k_config_by_hand, xi, L, 0, 0, 1);
    pvec_by_hand(k_config_by_hand, xi, L, 1, 0, 1);
    pvec_by_hand(k_config_by_hand, xi, L,-1, 0, 1);
    pvec_by_hand(k_config_by_hand, xi, L, 0, 1, 1);
    pvec_by_hand(k_config_by_hand, xi, L, 0,-1, 1);
    */
    
    int size1 = p_config_by_hand[0].size(); 
    Eigen::MatrixXcd F2_mat_1(size1,size1);
    F2_i_mat_1( F2_mat_1, En, p_config_by_hand, p_config_by_hand, total_P, mi, mj, mk, L, xi, alpha, epsilon_h, max_shell_num );
    
    Eigen::MatrixXcd K2inv_mat_1(size1,size1);
    K2inv_i_mat( K2inv_mat_1, eta_1, scattering_length_1_piK, En, p_config_by_hand, p_config_by_hand, total_P, mi, mj, mk, epsilon_h, L );
    
    int size2 = k_config_by_hand[0].size(); 
    Eigen::MatrixXcd F2_mat_2(size2,size2);
    F2_i_mat_1( F2_mat_2, En, k_config_by_hand, k_config_by_hand, total_P, mk, mi, mj, L, xi, alpha, epsilon_h, max_shell_num );
    
    Eigen::MatrixXcd K2inv_mat_2(size2,size2);
    K2inv_i_mat( K2inv_mat_2, eta_2, scattering_length_2_KK, En, k_config_by_hand, k_config_by_hand, total_P, mk, mi, mj, epsilon_h, L );

    mi = atmK; 
    mj = atmK; 
    mk = atmpi; 
    
    Eigen::MatrixXcd G_mat_11(size1,size1);
    G_ij_mat( G_mat_11, En, p_config_by_hand, p_config_by_hand, total_P, mi, mj, mk, L, epsilon_h ); 

    mi = atmK; 
    mj = atmpi; 
    mk = atmK; 

    Eigen::MatrixXcd G_mat_12(size1,size2);
    G_ij_mat( G_mat_12, En, p_config_by_hand, k_config_by_hand, total_P, mi, mj, mk, L, epsilon_h ); 

    mi = atmpi; 
    mj = atmK; 
    mk = atmK; 

    Eigen::MatrixXcd G_mat_21(size2,size1);
    G_ij_mat( G_mat_21, En, k_config_by_hand, p_config_by_hand, total_P, mi, mj, mk, L, epsilon_h ); 

    Eigen::MatrixXcd F2_mat(size1 + size2,size1 + size2);

    Eigen::MatrixXcd Filler0_12(size1,size2);
    Eigen::MatrixXcd Filler0_21(size2,size1);
    Eigen::MatrixXcd Filler0_22(size2,size2);

    Filler0_12 = Eigen::MatrixXcd::Zero(size1,size2);
    Filler0_21 = Eigen::MatrixXcd::Zero(size2,size1);
    Filler0_22 = Eigen::MatrixXcd::Zero(size2,size2);
    
    F2_mat <<   F2_mat_1,   Filler0_12,
                Filler0_21, F2_mat_2; 

    //abort();
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

    Eigen::MatrixXcd F3mat(size1+size2, size1+size2); 
    F3mat = (F2_mat/3.0 - F2_mat*H_mat_inv*F2_mat);//temp_F3_mat; 
    
    
    std::cout<<"F2 = "<<std::endl; 
    std::cout<<F2_mat<<std::endl; 
    std::cout<<"====================================="<<std::endl; 

    std::cout<<"G = "<<std::endl; 
    std::cout<<G_mat<<std::endl; 
    std::cout<<"====================================="<<std::endl; 

    std::cout<<"K2i = "<<std::endl; 
    std::cout<<K2inv_mat<<std::endl; 
    std::cout<<"====================================="<<std::endl; 


    std::cout<<"F3 = "<<std::endl; 
    std::cout<<F3mat<<std::endl; 
    std::cout<<"====================================="<<std::endl; 

    std::cout<<std::setprecision(20);
    std::cout<<"F3iso = "<<F3mat.sum()<<std::endl;
    
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

    //test_detF3inv_vs_En();

    //This function is for F3 and F3inv both:
    test_detF3inv_vs_En_KKpi();

    //test_mass_dependences_F3_2plus1_vs_En();

    //test_uneven_matrix();

    //test_individual_functions();
    //test_individual_functions_KKpi();
    
    //test_F2_vs_sigp();
    //test_F2_sum_func();

    //test_additionalpoles_in_F3_vs_En_KKpi();

    //check the K2inv dependence of F3 for 3-ID
    //test_3body_non_int();
    //test_F3_ID_printer();

    //test_Gmat_vs_sigp(); 
    //p_in_lattice_units(); 

    //activated_shell(0,0,0);
    //activated_shell(0,0,1);
    //activated_shell(0,1,1);
    //activated_shell(1,1,1);
    //activated_shell(0,0,2);

    //test_functions_with_FRL_codebase_ID();

    //test_functions_with_FRL_codebase_2plus1();
    
    return 0;
}