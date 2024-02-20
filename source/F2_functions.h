#ifndef F2FUNCTIONS_H
#define F2FUNCTIONS_H



#include<bits/stdc++.h>
#include<cmath>
#include<Eigen/Dense>
#include "functions.h"
//#include "gsl/gsl_sf_dawson.h"
//#include<Eigen/Dense>

//#include "Faddeeva.cc"
#include "Faddeeva.hh"


/* First we code all the function needed for F2 functions, we start with a single F in S-wave
as needed to check. We follow the paper = https://arxiv.org/pdf/2111.12734.pdf */


typedef std::complex<double> comp;
//This is the result of the PV integration with proper
//regularization 

comp I0F(   comp En, 
            comp sigma_p,
            comp p,
            comp total_P, 
            double alpha,
            double mi,
            double mj,
            double mk, 
            double L )
{
    double pi = std::acos(-1.0);
    
    comp gamma = (En - omega_func(p,mi))/std::sqrt(sigma_p);
    comp x = std::sqrt(q2psq_star(sigma_p,mj,mk))*L/(2.0*pi);

    comp A = 4.0*pi*gamma;
    comp B = -std::sqrt(pi/alpha)*(1.0/2.0)*std::exp(alpha*x*x);
    //comp C = 0.5*(pi*x)*ERFI_func(std::sqrt(alpha*x*x));

    double relerr = 0.0;
    comp C = 0.5*(pi*x)*Faddeeva::erfi(std::sqrt(alpha*x*x),relerr);

    char debug = 'n';
    if(debug=='y')
    {
        std::cout<<std::setprecision(25);
        std::cout<<"constant = "<<A<<std::endl;
        std::cout<<"factor1 = "<<B<<std::endl; 
        std::cout<<"factor2 = "<<C<<std::endl; 
    }
    return A*(B + C);

}

//We consider that the spectator momentum p and 
//the total frame momentum P both are in the z-direction 
//this doesn't include the lattice anisotropy xi
comp I00_sum_F(     comp En, 
                    comp sigma_p,
                    std::vector<comp> p,
                    std::vector<comp> total_P, 
                    double alpha,
                    double mi,
                    double mj,
                    double mk, 
                    double L,
                    int max_shell_num    )
{
    char debug = 'n';
    double tolerance = 1.0e-11;
    double pi = std::acos(-1.0);

    comp px = p[0];
    comp py = p[1];
    comp pz = p[2];

    comp spec_p = std::sqrt(px*px + py*py + pz*pz);

    comp Px = total_P[0];
    comp Py = total_P[1];
    comp Pz = total_P[2];

    comp total_P_val = std::sqrt(Px*Px + Py*Py + Pz*Pz);

    comp gamma = (En - omega_func(spec_p,mi))/std::sqrt(sigma_p);
    //std::cout<<"gamma = "<<gamma<<std::endl;
    comp x = std::sqrt(q2psq_star(sigma_p,mj,mk))*L/(2.0*pi);
    comp xi = 0.5*(1.0 + (mj*mj - mk*mk)/sigma_p);

    if(debug=='y')
    {
        std::cout<<"x = "<<x<<'\t'<<"sig_p = "<<sigma_p<<std::endl;

        //std::cout<<"x = "<<x<<'\t'<<"sig_p = "<<sigma_p<<std::endl;
    }

    comp npPx = (px - Px)*L/(2.0*pi); 
    comp npPy = (py - Py)*L/(2.0*pi);
    comp npPz = (pz - Pz)*L/(2.0*pi);

    comp npP = std::sqrt(npPx*npPx + npPy*npPy + npPz*npPz);

    int c1 = 0;
    int c2 = 0; //these two are for checking if p and P are zero or not

    if(abs(spec_p)<1.0e-10 || abs(spec_p)==0.0 ) c1 = 1;
    if(abs(total_P_val)<1.0e-10 || abs(total_P_val)==0.0) c2 = 1;

    comp xibygamma = xi/gamma; 
    
    //int max_shell_num = 50;
    int na_x_initial = -max_shell_num;
    int na_x_final = +max_shell_num;
    int na_y_initial = -max_shell_num;
    int na_y_final = +max_shell_num;
    int na_z_initial = -max_shell_num;
    int na_z_final = +max_shell_num;

    comp summ = {0.0,0.0};
    comp temp_summ = {0.0,0.0};
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
            }
        }
    }

    return summ ;
}


//this includes the lattice anistropy xi
comp I00_sum_F_1(   comp En, 
                    comp sigma_p,
                    std::vector<comp> p,
                    std::vector<comp> total_P, 
                    double alpha,
                    double mi,
                    double mj,
                    double mk, 
                    double L,
                    double xi, 
                    int max_shell_num    )
{
    char debug = 'n';
    double tolerance = 1.0e-11;
    double pi = std::acos(-1.0);

    comp px = p[0];
    comp py = p[1];
    comp pz = p[2];

    comp spec_p = std::sqrt(px*px + py*py + pz*pz);

    comp Px = total_P[0];
    comp Py = total_P[1];
    comp Pz = total_P[2];

    comp total_P_val = std::sqrt(Px*Px + Py*Py + Pz*Pz);

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
                else 
                {
                    if(abs(npPsq)==0)
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
            }
        }
    }

    return summ ;
}



comp F2_i1( comp En, 
            std::vector<comp> k, //we assume that k,p,P are a 3-vector
            std::vector<comp> p,
            std::vector<comp> total_P,
            double L, 
            double mi,
            double mj, 
            double mk, 
            double alpha,
            double epsilon_h,
            int max_shell_num    )
{
    char debug = 'n';
    comp kx = k[0];
    comp ky = k[1];
    comp kz = k[2];

    comp px = p[0];
    comp py = p[1];
    comp pz = p[2];

    comp Px = total_P[0];
    comp Py = total_P[1];
    comp Pz = total_P[2];

    comp spec_k = std::sqrt(kx*kx + ky*ky + kz*kz);
    comp spec_p = std::sqrt(px*px + py*py + pz*pz);
    comp total_P_val = std::sqrt(Px*Px + Py*Py + Pz*Pz);

    comp sigp = sigma_pvec_based(En,p,mi,total_P);//sigma(En, spec_p, mi, total_P_val);

    comp cutoff = cutoff_function_1(sigp, mj, mk, epsilon_h);

    comp omega_p = omega_func(spec_p,mi);

    if(debug=='y')
    {
        std::cout << "p1x = " << px << '\t'
                  << "p1y = " << py << '\t'
                  << "p1z = " << pz << std::endl; 
        std::cout << "k1x = " << kx << '\t'
                  << "k1y = " << ky << '\t'
                  << "k1z = " << kz << std::endl;
        std::cout << "spec_k = " << spec_k << '\t'
                  << "spec_p = " << spec_p << std::endl; 
        std::cout << "total_P = " << total_P_val << std::endl; 
        std::cout << "sig_p = " << sigp << '\t'
                  << "cutoff = " << cutoff << '\t' 
                  << "omega_p = " << omega_p << std::endl; 
    }

    //condition for the delta function
    int condition_delta = 0;

    if(px==kx && py==ky && pz==kz)
    {
        condition_delta = 0;
    }
    else 
    {
        condition_delta = 1;
    }

    if(debug=='y')
    {
        std::cout << "condition delta = " << condition_delta << std::endl; 
    }
    
    if(condition_delta==1) return 0.0;
    else 
    {
        double pi = std::acos(-1.0);
        comp A = cutoff/(16.0*pi*pi*L*L*L*L*omega_p*(En - omega_p));

        comp B = I00_sum_F(En,sigp, p, total_P, alpha, mi, mj, mk, L, max_shell_num);

        comp C = I0F(En, sigp, spec_p, total_P_val, alpha, mi, mj, mk, L);
        //std::cout<<A<<'\t'<<B<<'\t'<<C<<std::endl; 

        if(debug=='y')
        {
            std::cout << "cutoff times constant = " << A << std::endl;  
            std::cout << "sum = " << B << std::endl; 
            std::cout << "analytical res = " << C << std::endl; 
            std::cout << std::endl; 
        }

        if(abs(A)==0)
        {
            return 0.0;
        }
        else
        return A*(B - C);
    }

}


//this F2_i1 function has lattice anisotropy xi
//added to it

comp F2_i1_1( comp En, 
            std::vector<comp> k, //we assume that k,p,P are a 3-vector
            std::vector<comp> p,
            std::vector<comp> total_P,
            double L,
            double xi,  
            double mi,
            double mj, 
            double mk, 
            double alpha,
            double epsilon_h,
            int max_shell_num    )
{
    char debug = 'n';
    comp kx = k[0];
    comp ky = k[1];
    comp kz = k[2];

    comp px = p[0];
    comp py = p[1];
    comp pz = p[2];

    comp Px = total_P[0];
    comp Py = total_P[1];
    comp Pz = total_P[2];

    comp spec_k = std::sqrt(kx*kx + ky*ky + kz*kz);
    comp spec_p = std::sqrt(px*px + py*py + pz*pz);
    comp total_P_val = std::sqrt(Px*Px + Py*Py + Pz*Pz);

    comp sigp = sigma_pvec_based(En,p,mi,total_P);//sigma(En, spec_p, mi, total_P_val);

    comp cutoff = cutoff_function_1(sigp, mj, mk, epsilon_h);

    comp omega_p = omega_func(spec_p,mi);

    if(debug=='y')
    {
        std::cout << "p1x = " << px << '\t'
                  << "p1y = " << py << '\t'
                  << "p1z = " << pz << std::endl; 
        std::cout << "k1x = " << kx << '\t'
                  << "k1y = " << ky << '\t'
                  << "k1z = " << kz << std::endl;
        std::cout << "spec_k = " << spec_k << '\t'
                  << "spec_p = " << spec_p << std::endl; 
        std::cout << "total_P = " << total_P_val << std::endl; 
        std::cout << "sig_p = " << sigp << '\t'
                  << "cutoff = " << cutoff << '\t' 
                  << "omega_p = " << omega_p << std::endl; 
    }

    //condition for the delta function
    int condition_delta = 0;

    if(px==kx && py==ky && pz==kz)
    {
        condition_delta = 0;
    }
    else 
    {
        condition_delta = 1;
    }

    if(debug=='y')
    {
        std::cout << "condition delta = " << condition_delta << std::endl; 
    }
    
    if(condition_delta==1) return 0.0;
    else 
    {
        double pi = std::acos(-1.0);
        comp A = cutoff/(16.0*pi*pi*L*L*L*L*omega_p*(En - omega_p));

        comp B = I00_sum_F_1(En,sigp, p, total_P, alpha, mi, mj, mk, L, xi, max_shell_num);

        comp C = I0F(En, sigp, spec_p, total_P_val, alpha, mi, mj, mk, L);
        //std::cout<<A<<'\t'<<B<<'\t'<<C<<std::endl; 

        if(debug=='y')
        {
            std::cout << "cutoff times constant = " << A << std::endl;  
            std::cout << "sum = " << B << std::endl; 
            std::cout << "analytical res = " << C << std::endl; 
            std::cout << std::endl; 
        }

        if(abs(A)==0)
        {
            return 0.0;
        }
        else
        return A*(B - C);
    }

}


/* This is the F2 matrix for a single block, this matrix could be
used to test QC3 for identical particles. Later on, we will use two 
of this matrices to build the F_hat matrix needed for the KKpi project */

void F2_i_mat(  Eigen::MatrixXcd &F2,
                comp En, 
                std::vector<std::vector<comp> > &p_config,
                std::vector<std::vector<comp> > &k_config,
                std::vector<comp> total_P,
                double mi,
                double mj, 
                double mk, 
                double L, 
                double alpha, 
                double epsilon_h,
                int max_shell_num  )
{
    char debug = 'n';

    if(debug=='y')
    {
        std::cout << "We will print out the components of F2 matrix" << std::endl;
        std::cout << "(F2_i_mat function from F2_functions.h)" << std::endl; 
    }
    for(int i=0; i<p_config[0].size(); ++i)
    {
        if(debug=='y')
        {
            std::cout << "-------------------------------------" << std::endl; 
        }
        for(int j=0; j<k_config[0].size(); ++j)
        {
            comp px = p_config[0][i];
            comp py = p_config[1][i];
            comp pz = p_config[2][i];

            comp spec_p = std::sqrt(px*px + py*py + pz*pz);
            std::vector<comp> p(3);
            p[0] = px;
            p[1] = py;
            p[2] = pz; 

            comp kx = k_config[0][j];
            comp ky = k_config[1][j];
            comp kz = k_config[2][j];

            comp spec_k = std::sqrt(kx*kx + ky*ky + kz*kz);
            std::vector<comp> k(3);
            k[0] = kx;
            k[1] = ky;
            k[2] = kz; 

            comp Px = total_P[0];
            comp Py = total_P[1];
            comp Pz = total_P[2];

            comp total_P_val = std::sqrt(Px*Px + Py*Py + Pz*Pz);
            
            comp F2_val = F2_i1(En, k, p, total_P, L, mi, mj, mk, alpha, epsilon_h, max_shell_num);

            F2(i,j) = F2_val;

            if(debug=='y')
            {
                std::cout << "i = " << i << '\t' << "j = " << j << std::endl; 
                std::cout << "px = " << px << '\t'
                          << "py = " << py << '\t'
                          << "pz = " << pz << std::endl; 
                std::cout << "kx = " << kx << '\t'
                          << "ky = " << ky << '\t'
                          << "kz = " << kz << std::endl;
                std::cout << "F2 val = " << 0.5*F2_val*L*L*L << std::endl;
                std::cout << "-------------------------------------" << std::endl;
            }

            


        }
        if(debug=='y')
        {
            std::cout << "-------------------------------------" << std::endl; 
        }
    }

    if(debug=='y')
    {
        std::cout << "=========================================" << std::endl; 
    }
}
            
//this F2_i_mat function has lattice anisotropy xi
void F2_i_mat_1(  Eigen::MatrixXcd &F2,
                comp En, 
                std::vector<std::vector<comp> > &p_config,
                std::vector<std::vector<comp> > &k_config,
                std::vector<comp> total_P,
                double mi,
                double mj, 
                double mk, 
                double L, 
                double xi, 
                double alpha, 
                double epsilon_h,
                int max_shell_num  )
{
    char debug = 'n';

    if(debug=='y')
    {
        std::cout << "We will print out the components of F2 matrix" << std::endl;
        std::cout << "(F2_i_mat function from F2_functions.h)" << std::endl; 
    }
    for(int i=0; i<p_config[0].size(); ++i)
    {
        if(debug=='y')
        {
            std::cout << "-------------------------------------" << std::endl; 
        }
        for(int j=0; j<k_config[0].size(); ++j)
        {
            comp px = p_config[0][i];
            comp py = p_config[1][i];
            comp pz = p_config[2][i];

            comp spec_p = std::sqrt(px*px + py*py + pz*pz);
            std::vector<comp> p(3);
            p[0] = px;
            p[1] = py;
            p[2] = pz; 

            comp kx = k_config[0][j];
            comp ky = k_config[1][j];
            comp kz = k_config[2][j];

            comp spec_k = std::sqrt(kx*kx + ky*ky + kz*kz);
            std::vector<comp> k(3);
            k[0] = kx;
            k[1] = ky;
            k[2] = kz; 

            comp Px = total_P[0];
            comp Py = total_P[1];
            comp Pz = total_P[2];

            comp total_P_val = std::sqrt(Px*Px + Py*Py + Pz*Pz);
            
            comp F2_val = F2_i1_1(En, k, p, total_P, L, xi, mi, mj, mk, alpha, epsilon_h, max_shell_num);

            F2(i,j) = F2_val;

            if(debug=='y')
            {
                std::cout << "i = " << i << '\t' << "j = " << j << std::endl; 
                std::cout << "px = " << px << '\t'
                          << "py = " << py << '\t'
                          << "pz = " << pz << std::endl; 
                std::cout << "kx = " << kx << '\t'
                          << "ky = " << ky << '\t'
                          << "kz = " << kz << std::endl;
                std::cout << "F2 val = " << 0.5*F2_val*L*L*L << std::endl;
                std::cout << "-------------------------------------" << std::endl;
            }

            


        }
        if(debug=='y')
        {
            std::cout << "-------------------------------------" << std::endl; 
        }
    }

    if(debug=='y')
    {
        std::cout << "=========================================" << std::endl; 
    }
}
            


#endif