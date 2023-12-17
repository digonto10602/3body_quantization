#ifndef F2FUNCTIONS_H
#define F2FUNCTIONS_H

#include<bits/stdc++.h>
#include<cmath>
//#include "gsl/gsl_sf_dawson.h"
//#include<Eigen/Dense>

/* First we code all the function needed for F2 functions, we start with a single F in S-wave
as needed to check. We follow the paper = https://arxiv.org/pdf/2111.12734.pdf */


typedef std::complex<double> comp;

comp omega_func(    comp p, 
                    double m    )
{
    return std::sqrt(p*p + m*m);
}

comp sigma( comp En,
            comp spec_p,
            double mi,
            comp total_P    )
{
    comp A = En - omega_func(spec_p,mi);
    comp B = total_P - spec_p;

    return A*A - B*B;
}

comp kallentriangle(    comp x, 
                        comp y, 
                        comp z  )
{
    return x*x + y*y + z*z - 2.0*x*y - 2.0*y*z - 2.0*z*x;
}

comp q2psq_star(    comp sigma_i,
                    double mj, 
                    double mk )
{
    return kallentriangle(sigma_i, mj*mj, mk*mk)/(4.0*sigma_i);
}

comp pmom(  comp En,
            comp sigk,
            double m    )
{
    return sqrt(kallentriangle(En,sigk,m*m))/(2.0*sqrt(En*En));
}

comp Jfunc( comp z  )
{
    if(std::real(z)<=0.0)
    {
        return 0.0;
    }
    else if(std::real(z)>0.0 && std::real(z)<1.0)
    {
        comp A = -1.0/z;
        comp B = std::exp(-1.0/(1.0-z));
        return std::exp(A*B);
    }
    else
    {
        return 1.0;
    }
    
    

}

comp cutoff_function_1( comp sigma_i,
                        double mj, 
                        double mk, 
                        double epsilon_h   )
{
    comp Z = (comp) (1.0 + epsilon_h)*( sigma_i - (comp) std::abs(mj*mj - mk*mk) )/( (mj + mk)*(mj + mk) - std::abs(mj*mj - mk*mk) );

    return Jfunc(Z);
}

/*

    S-wave F2 function for i = 1

*/

//This dawson function has imaginary argument for the purpose
//of generality of the code but the definition need to be changed 
//if we are using true complex x's, the definition would be changed
//to use fadeeva function w(z). For now since we are only using 
//real energies and real momentum we can use this defintion with 
//the argument being complex without creating any errors.

comp dawson_func(comp x)
{
    double steps = 1000000;
    comp summ = 0.0;

    comp y = 0.0;
    comp dely = x/steps;
    for(int i=0;i<(int)steps;++i)
    {
        summ = summ + std::exp(y*y)*dely;
        y=y+dely;
    }

    return std::exp(-x*x)*summ;
}

comp ERFI_func(comp x)
{
    double pi = std::acos(-1.0);
    return dawson_func(x)*2.0*std::exp(x*x)/std::sqrt(pi);
}


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
    comp C = (pi*x/2.0)*ERFI_func(std::sqrt(alpha*x*x));

    return A*(B + C);

}

//We consider that the spectator momentum p and 
//the total frame momentum P both are in the z-direction 

comp I00_sum_F(     comp En, 
                    comp sigma_p,
                    comp p,
                    comp total_P, 
                    double alpha,
                    double mi,
                    double mj,
                    double mk, 
                    double L    )
{
    double tolerance = 1.0e-11;
    double pi = std::acos(-1.0);

    comp gamma = (En - omega_func(p,mi))/std::sqrt(sigma_p);
    //std::cout<<"gamma = "<<gamma<<std::endl;
    comp x = std::sqrt(q2psq_star(sigma_p,mj,mk))*L/(2.0*pi);
    comp xi = 0.5*(1.0 + (mj*mj - mk*mk)/sigma_p);

    comp npP = (p - total_P)*L/(2.0*pi); //this is directed in the z-direction

    int c1 = 0;
    int c2 = 0; //these two are for checking if p and P are zero or not

    if(abs(p)<1.0e-10) c1 = 1;
    if(abs(total_P)<1.0e-10) c2 = 1;

    comp xibygamma = xi/gamma; 
    
    int max_shell_num = 20;
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

                comp nax_npPx = nax*0.0;
                comp nay_npPy = naz*0.0;
                comp naz_npPz = naz*npP; 

                comp na_dot_npP = nax_npPx + nay_npPy + naz_npPz;
                comp npPsq = npP*npP; 

                comp prod1 = ( (na_dot_npP/npPsq)*(1.0/gamma - 1.0) + xibygamma );
                
                comp rx = nax;
                comp ry = nay;
                comp rz = 0.0;
                
                if(c1==1 and c2==1)
                {
                    rz = naz;// + npP*prod1; 
                }
                else 
                {
                    rz = naz + npP*prod1;
                }

                comp r = std::sqrt(rx*rx + ry*ry + rz*rz);

                //std::cout<<"npP = "<<npP<<'\t'
                //         <<"prod1 = "<<prod1<<'\t'
                //         <<"r = "<<r<<'\t'<<"rx = "<<rx<<'\t'<<"ry = "<<ry<<'\t'<<"rz = "<<rz<<std::endl;

                summ = summ + std::exp(alpha*(x*x - r*r))/(x*x - r*r);
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
            std::vector<comp> k, //we assume that k,p is a 3-vector
            std::vector<comp> p,
            comp total_P,
            double L, 
            double mi,
            double mj, 
            double mk, 
            double alpha,
            double epsilon_h    )
{
    comp kx = k[0];
    comp ky = k[1];
    comp kz = k[2];

    comp px = p[0];
    comp py = p[1];
    comp pz = p[2];

    comp spec_k = std::sqrt(kx*kx + ky*ky + kz*kz);
    comp spec_p = std::sqrt(px*px + py*py + pz*pz);

    comp sigp = sigma(En, spec_p, mi, total_P);

    comp cutoff = cutoff_function_1(sigp, mj, mk, epsilon_h);

    comp omega_p = omega_func(spec_p,mi);

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
    
    //if(condition_delta==1) return 0.0;
    //else 
    {
        double pi = std::acos(-1.0);
        comp A = cutoff/(64.0*pi*pi*pi*L*L*L*L*omega_p*(En - omega_p));

        comp B = I00_sum_F(En,sigp, spec_p, total_P, alpha, mi, mj, mk, L);

        comp C = I0F(En, sigp, spec_p, total_P, alpha, mi, mj, mk, L);
        std::cout<<A<<'\t'<<B<<'\t'<<C<<std::endl; 

        return A*(B/(4.0*pi) - C/(4.0*pi));
    }

}


            


#endif