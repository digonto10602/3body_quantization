#include "F2_functions.h"
#include "K2_functions.h"
#include "G_functions.h"
#include "QC_functions.h"


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

    double alpha = 0.750;
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

        std::cout<<std::setprecision(20)<<En<<'\t'<<real(sigma_p)<<'\t'<<real(F2)<<'\t'<<imag(F2)<<std::endl; 

        fout<<std::setprecision(20);
        //fout<<real(sigma_p)<<'\t'<<imag(sigma_p)<<'\t'<<real(F2)<<'\t'<<imag(F2)<<std::endl; 
        fout<<En-mi<<'\t'<<real(F2)<<'\t'<<imag(F2)<<std::endl; 
    }
    fout.close();


}

void test_QC3_vs_En()
{
    double pi = std::acos(-1.0);
    double mi = 1.0;
    double mj = 1.0;
    double mk = 1.0;
    double epsilon_h = 0.0; 

    double En_initial = 2.5;
    double En_final = 4.5;
    double En_points = 1000.0;
    double del_En = abs(En_initial - En_final)/En_points;

    double eta_i = 0.5;
    double alpha = 0.750;
    double L = 6.0;
    //double En = 3.2;

    std::ofstream fout; 
    
    std::string filename = "QC3_ID_test.dat";

    fout.open(filename.c_str());

    comp total_P = 0.0;
    //comp spec_p = 0.0;
    double scattering_length = -10.0;

    int nmax = 1;
    
    
    for(int i=0;i<En_points + 1;++i)
    {
        double En = En_initial + i*del_En;
        comp res = {0.0,0.0};
        for(int nx=-nmax;nx<nmax+1;++nx)
        {  
            for(int ny=-nmax;ny<nmax+1;++ny)
            {
                for(int nz=-nmax;nz<nmax+1;++nz)
                {
        //            int nx = 0;
        //            int ny = 0;
        //            int nz = 0;
                    //std::cout<<"here"<<std::endl;
                    std::cout<<"n ="<<nx<<ny<<nz<<std::endl;
                    std::vector<comp> k(3);
                    k[0] = (2.0*pi/L)*nx;
                    k[1] = (2.0*pi/L)*ny;
                    k[2] = (2.0*pi/L)*nz;
                    std::vector<comp> p = k; 
                    comp spec_p = std::sqrt(p[0]*p[0] + p[1]*p[1] + p[2]*p[2]);
                    comp spec_k = std::sqrt(k[0]*k[0] + k[1]*k[1] + k[2]*k[2]);
                    comp sigma_p = sigma(En, spec_p, mi, total_P); 

                    comp qc = QC3_ID(En, p, k, eta_i, scattering_length, mi, mj, mk, total_P, alpha, epsilon_h, L);
                    //comp G = G_ij(En,spec_p,spec_k, mi, mj, mk, total_P, L, epsilon_h);
                    //comp F = F2_i1(En, k, p, total_P, L, mi, mj, mk, alpha, epsilon_h);
                    if(real(qc)!=real(qc)) continue;
                    else 
                    res =  res + qc; 
                }
            }
        }
         
        



        //comp sigk = sigma(En,spec_p,mi,total_P);

        std::cout<<std::setprecision(20)<<En<<'\t'<<real(res)<<'\t'<<imag(res)<<std::endl; 

        fout<<std::setprecision(20);
        //fout<<real(sigma_p)<<'\t'<<imag(sigma_p)<<'\t'<<real(F2)<<'\t'<<imag(F2)<<std::endl; 
        fout<<En<<'\t'<<real(res)<<'\t'<<imag(res)<<std::endl; 
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

    for(int i=0;i<En_points+1;++i)
    {
        double En = En_initial + i*del_En;
        double spec_p = 0.0;
        double total_P = 0.0; 
        comp sigp = sigma(En, spec_p, m, total_P);
        comp K2 = tilde_K2_00(0.5,a, 0.0, 0.0, sigp, m, m, m, 0.0, L);

        fout<<En-m<<'\t'<<-real(K2)<<'\t'<<-imag(K2)<<std::endl;
    }
    fout.close();

}

int main()
{
    //test_F2_i1_mombased();
    //test_F2_i1_mombased_vs_En();
    K2printer();
    //test_QC3_vs_En();
    //I00_sum_F_test();
    return 0;
}