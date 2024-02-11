#ifndef POLESEARCHING_H
#define POLESEARCHING_H
#include "F2_functions.h"
#include "K2_functions.h"
#include "G_functions.h"
#include "QC_functions.h"

typedef std::complex<double> comp;

void det_F3mat_poles_secant_method( double init_guess1,
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
                                    double &pole   )
{
    double point1 = init_guess1;
    double point2 = init_guess2; 

    int max_iteration = 100;

    double epsilon = 1.0e-15;
    double tolerance = 1.0e-15;

    double mi = m_K;
    double mj = m_K; 
    double mk = m_pi; 

    comp ii = {0.0,1.0};
    double comp_en_additive = 1.0e-10;
    double point0; 

    for(int i=0; i<max_iteration; ++i)
    {
        //we make the momentum configs for each energies point1,2 and 0
        double En1 = point1; 
        std::vector< std::vector<comp> > p1_config(3,std::vector<comp> ());
        double config_tolerance = 1.0e-5;
        config_maker_1(p1_config, En1 , total_P, mi, mj, mk, L, xi, epsilon_h, config_tolerance );

        std::vector< std::vector<comp> > k1_config = p1_config;

        double En2 = point2; 
        std::vector< std::vector<comp> > p2_config(3,std::vector<comp> ());
        config_maker_1(p2_config, En2 + ii*comp_en_additive, total_P, mi, mj, mk, L, xi, epsilon_h, config_tolerance );

        std::vector< std::vector<comp> > k2_config = p2_config;

        double f1 = det_F3_ND_2plus1_mat( point1 + ii*comp_en_additive, p1_config, k1_config, total_P, eta_i_1, eta_i_2, scattering_length_1, scattering_length_2, m_pi, m_K, alpha, epsilon_h, L, xi, max_shell_num); 
        double f2 = det_F3_ND_2plus1_mat( point2 + ii*comp_en_additive, p2_config, k2_config, total_P, eta_i_1, eta_i_2, scattering_length_1, scattering_length_2, m_pi, m_K, alpha, epsilon_h, L, xi, max_shell_num); 

        point0 = point1 - f1*(point1 - point2)/(f1 - f2);

        double En0 = point0; 
        std::vector< std::vector<comp> > p0_config(3,std::vector<comp> ());
        config_maker_1(p0_config, En0 + ii*comp_en_additive, total_P, mi, mj, mk, L, xi, epsilon_h, config_tolerance );

        std::vector< std::vector<comp> > k0_config = p0_config;


        double f0 = det_F3_ND_2plus1_mat( point0 + ii*comp_en_additive, p0_config, k0_config, total_P, eta_i_1, eta_i_2, scattering_length_1, scattering_length_2, m_pi, m_K, alpha, epsilon_h, L, xi, max_shell_num); 

        char debug = 'y';

        if(debug=='y')
        {
            std::cout<<std::setprecision(20);
            std::cout<<"En1 = "<<En1<<'\t'
                 <<"En2 = "<<En2<<'\t'
                 <<"En0 = "<<En0<<'\n'
                 <<"f1 = " <<f1 <<'\t'
                 <<"f2 = " <<f2 <<'\t'
                 <<"f0 = " <<f0 <<std::endl;

        }
        

        if(abs(f0)<=tolerance)
        {
            pole = point0;
            //cout<<"pole found at, E = "<<pole<<endl;
            break;
        }
        else if(abs(std::real(point0-point1))<=epsilon && abs(std::imag(point0-point1))<=epsilon)
        {
            pole = point0; 
            //cout<<"pole found at, E = "<<pole<<endl;
            break; 
        }

        if(i==max_iteration-1)
        {
            std::cout<<"did not converge to a solution, choose a diff guess or increase iteration"<<std::endl;
        }

        point2 = point1; 
        point1 = point0;
    }
    
    
}

/* Here we test the F3 function we have written for 2+1 system. 
We set K3df=0, and get the energies for which det(F3inv)=0, we will 
compare the results with the free spectrum we previously calculated using 
the dispersion relation     */
void test_F3inv_pole_searching()
{
    double xi = 3.444; //added new
    double init_guess1 = 0.342;
    double init_guess2 = 0.34201;

    double Linitial = 19.0;
    double Lfinal = 25.0;
    double Lpoints = 10;

    double delL = abs(Linitial - Lfinal)/Lpoints;

    /*  Inputs  */
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

    
    double pole = 0.0;

    for(int i=0; i<Lpoints+1; ++i)
    {
        double L = Linitial + i*delL; 

        det_F3mat_poles_secant_method(init_guess1,init_guess2, total_P, eta_1, eta_2, scattering_length_1_piK,scattering_length_2_KK, atmK, atmpi, alpha, epsilon_h, L, xi, max_shell_num, pole);

        init_guess1 = pole; 
        init_guess2 = pole + 1.0e-5; 

        std::cout<<L<<'\t'<<pole<<std::endl; 

    }
}

#endif