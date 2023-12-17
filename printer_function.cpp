#include "F2_functions.h"

void test_cutoff_function_1()
{
    double mj = 1.0;
    double mi = 1.0;
    double mk = 1.0;
    double epsilon_h = 0.0; 

    double sigma_initial = 0.0;
    double sigma_final = 5.0;
    double sigma_points = 100.0;
    double del_sigma = abs(sigma_initial - sigma_final)/sigma_points;


    std::ofstream fout; 
    
    std::string filename = "cutoff_test.dat";

    fout.open(filename.c_str());



    for(int i=0;i<sigma_points+1;++i)
    {
        double sigk = sigma_initial + i*del_sigma; 

        comp cutoff = cutoff_function_1(sigk, mj, mk, epsilon_h);

        std::cout<<std::setprecision(20)<<sigk<<'\t'<<real(cutoff)<<'\t'<<imag(cutoff)<<std::endl; 

        fout<<std::setprecision(20);
        fout<<sigk<<'\t'<<real(cutoff)<<'\t'<<imag(cutoff)<<std::endl; 
    }
    fout.close();


}

void test_F2_i1_mombased()
{
    double mi = 1.0;
    double mj = 1.0;
    double mk = 1.0;
    double epsilon_h = 0.0; 

    double pz_initial = 0.00000000000001;
    double pz_final = 4.5;
    double pz_points = 100.0;
    double del_pz = abs(pz_initial - pz_final)/pz_points;

    double alpha = 0.5;
    double L = 6.0;
    double En = 3.2;

    std::ofstream fout; 
    
    std::string filename = "F2_i1_test.dat";

    fout.open(filename.c_str());

    comp total_P = 0.0000000000001;
    comp spec_p = 0.0;
    
    for(int i=0;i<pz_points + 1;++i)
    {
        double pz = pz_initial + i*del_pz; 
        std::vector<comp> k(3);
        k[0] = 0.0;
        k[1] = 0.0;
        k[2] = 0.00000000000001;
        std::vector<comp> p = k;
        p[2] = pz;  

        comp sigma_p = sigma(En, pz, mi, total_P);

        comp F2 = F2_i1(En, k, p, total_P, L, mi, mj, mk, alpha, epsilon_h );

        std::cout<<std::setprecision(20)<<pz<<'\t'<<real(F2)<<'\t'<<imag(F2)<<std::endl; 

        fout<<std::setprecision(20);
        fout<<real(sigma_p)<<'\t'<<imag(sigma_p)<<'\t'<<real(F2)<<'\t'<<imag(F2)<<std::endl; 
    }
    fout.close();


}

void I00_sum_F_test()
{
    double mi = 1.0;
    double mj = 1.0;
    double mk = 1.0;

    double sigk_initial = 1.5;
    double sigk_final = 3.5;
    double sigk_points = 1000.0;
    double del_sigk = abs(sigk_initial - sigk_final)/sigk_points;

    double En = 1.2;
    double total_P = 0.0;
    double alpha = 0.5;
    double L = 6;

    for(int i=0;i<sigk_points+1;++i)
    {
        double sigk = sigk_initial + i*del_sigk; 
        
        comp p = pmom(En,sigk,mi);

        comp I00 = I00_sum_F(En,sigk,p,total_P,alpha, mi, mj, mk, L);

        std::cout<<sigk<<'\t'<<real(p)<<'\t'<<real(I00)<<std::endl;
    }
}

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

    double alpha = 0.5;
    double L = 6.0;
    //double En = 3.2;

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
        p[2] = 0.0;  

        comp sigma_p = sigma(En, spec_p, mi, total_P);

        comp F2 = F2_i1(En, k, p, total_P, L, mi, mj, mk, alpha, epsilon_h );

        //comp sigk = sigma(En,spec_p,mi,total_P);

        std::cout<<std::setprecision(20)<<'\t'<<En<<'\t'<<real(sigma_p)<<'\t'<<real(F2)<<'\t'<<imag(F2)<<std::endl; 

        fout<<std::setprecision(20);
        //fout<<real(sigma_p)<<'\t'<<imag(sigma_p)<<'\t'<<real(F2)<<'\t'<<imag(F2)<<std::endl; 
        fout<<En-mi<<'\t'<<real(F2)<<'\t'<<imag(F2)<<std::endl; 
    }
    fout.close();


}

int main()
{
    //test_F2_i1_mombased();
    test_F2_i1_mombased_vs_En();
    //I00_sum_F_test();
    return 0;
}