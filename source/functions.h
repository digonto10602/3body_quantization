#ifndef FUNCTIONS_H
#define FUNCTIONS_H 

#include<bits/stdc++.h>
#include<cmath>
#include<Eigen/Dense>

typedef std::complex<double> comp;

comp mysqrt(    comp x  )
{
    comp ii = {0.0,1.0};
    return ii*std::sqrt(-x);
}

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

/* This sigma takes the spectator momenta as a vector */
comp sigma_pvec_based(  comp En, 
                        std::vector<comp> p,
                        double mi, 
                        std::vector<comp> total_P   )
{
    comp px = p[0];
    comp py = p[1];
    comp pz = p[2];

    comp spec_p = std::sqrt(px*px + py*py + pz*pz);

    comp Px = total_P[0];
    comp Py = total_P[1];
    comp Pz = total_P[2];

    comp Pminusp_x = Px - px; 
    comp Pminusp_y = Py - py; 
    comp Pminusp_z = Pz - pz; 

    comp Pminusp_sq = Pminusp_x*Pminusp_x + Pminusp_y*Pminusp_y + Pminusp_z*Pminusp_z; 
    comp A = En - omega_func(spec_p,mi);

    return A*A - Pminusp_sq; 
}

comp kallentriangle(    comp x, 
                        comp y, 
                        comp z  )
{
    //return x*x + y*y + z*z - 2.0*x*y - 2.0*y*z - 2.0*z*x;
    return x*x + y*y + z*z - 2.0*(x*y + y*z + z*x);

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
    return std::sqrt(kallentriangle(En*En,sigk,m*m))/(2.0*sqrt(En*En));
}

comp kmax_for_P0(   comp En, 
                    double m )
{
    comp A = (En*En + m*m)/(2.0*En);

    return A*A - m*m;
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

    if(mj==mk && epsilon_h==0.0)
    {
        comp Z = sigma_i/(4.0*mj*mj);
        return Jfunc(Z); 
    }
    else 
    {
        //std::cout<<"went to else"<<std::endl; 
        comp Z = (comp) (1.0 + epsilon_h)*( sigma_i - (comp) std::abs(mj*mj - mk*mk) )/( (mj + mk)*(mj + mk) - std::abs(mj*mj - mk*mk) );
        return Jfunc(Z);
        
    }

    //return 1.0; 

}

comp E_to_Ecm(  comp En,
                std::vector<comp> total_P )
{
    comp Px = total_P[0];
    comp Py = total_P[1];
    comp Pz = total_P[2];

    comp tot_P = std::sqrt(Px*Px + Py*Py + Pz*Pz);

    return std::sqrt(En*En - tot_P*tot_P);
}

comp Ecm_to_E(  comp En_cm,
                std::vector<comp> total_P )
{
    comp Px = total_P[0];
    comp Py = total_P[1];
    comp Pz = total_P[2];

    comp tot_P = std::sqrt(Px*Px + Py*Py + Pz*Pz);

    return std::sqrt(En_cm*En_cm + tot_P*tot_P);
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

/* Configuration vectors for the spectator momentum, these are 
generated with only the integer numbers first, then multiplied with
2pi/L, and they go upto where H(k) becomes zero */

void config_maker(  std::vector< std::vector<comp> > &p_config,
                    comp En,
                    double mi,
                    double L    )
{
    double pi = std::acos(-1.0);

    comp kmax = pmom(En,0.0,mi);

    int nmax = (int)ceil((L/(2.0*pi))*abs(kmax));

    int nmaxsq = nmax*nmax; 

    for(int i=-nmax; i<nmax + 1; ++i)
    {
        for(int j=-nmax; j<nmax + 1; ++j)
        {
            for(int k=-nmax; k<nmax + 1; ++k)
            {
                int nsq = i*i + j*j + k*k; 
                if(nsq<=nmaxsq)
                {
                    comp px = (2.0*pi/L)*i;
                    comp py = (2.0*pi/L)*j;
                    comp pz = (2.0*pi/L)*k; 

                    comp p = std::sqrt(px*px + py*py + pz*pz);
                    
                    if(abs(p)<=abs(kmax))
                    {
                        p_config[0].push_back(px);
                        p_config[1].push_back(py);
                        p_config[2].push_back(pz);
                        //std::cout << "n = " << i << j << k << " nsq = " << nsq << " nmaxsq = " << nmaxsq << std::endl; 
                        //std::cout << "px = " << px << " py = " << py << " pz = " << pz << std::endl;
                        //std::cout << "p = " << p << " kmax = " << kmax << std::endl; 
                 
                    }
                    else 
                    {
                        continue; 
                    }
                }
            }
        }
    }
    //std::cout << "nmax = " << nmax << '\t' << "nval = " << (L/(2.0*pi))*abs(kmax) << '\t' 
    //          << "kmax = " << kmax << std::endl;
}



/* This config maker is based upon the cutoff function H(k) instead of kmax */
void config_maker_1(  std::vector< std::vector<comp> > &p_config,
                    comp En,
                    std::vector<comp> total_P,
                    double mi,
                    double mj, 
                    double mk,
                    double L,
                    double xi, 
                    double epsilon_h,
                    double tolerance    )
{
    char debug = 'n';
    double pi = std::acos(-1.0);

    //comp cutoff = cutoff_function_1(sig_k,mj, mk, epsilon_h);
    //comp kmax = pmom(En,0.0,mi);
    comp kmax = kmax_for_P0(En, mi);

    int nmax = 20;//(int)ceil((L/(2.0*pi))*abs(kmax));
    //std::cout<<"Nmax = "<<nmax<<std::endl;

    int nmaxsq = nmax*nmax; 

    for(int i=-nmax; i<nmax + 1; ++i)
    {
        for(int j=-nmax; j<nmax + 1; ++j)
        {
            for(int k=-nmax; k<nmax + 1; ++k)
            {
                int nsq = i*i + j*j + k*k; 
                //if(nsq<=nmaxsq)
                {
                    comp px = (2.0*pi/(xi*L))*i;
                    comp py = (2.0*pi/(xi*L))*j;
                    comp pz = (2.0*pi/(xi*L))*k; 

                    comp p = std::sqrt(px*px + py*py + pz*pz);

                    comp Px = total_P[0];
                    comp Py = total_P[1];
                    comp Pz = total_P[2];

                    comp Pminusp_x = Px - px;
                    comp Pminusp_y = Py - py;
                    comp Pminusp_z = Pz - pz; 
                    comp Pminusp = std::sqrt(Pminusp_x*Pminusp_x + Pminusp_y*Pminusp_y + Pminusp_z*Pminusp_z);


                    comp sig_k = (En - omega_func(p,mi))*(En - omega_func(p,mi)) - Pminusp*Pminusp; 

                    comp cutoff = cutoff_function_1(sig_k, mj, mk, epsilon_h);

                    //std::cout<<"kmax = "<<kmax<<'\t'<<"p = "<<p<<'\t'<<"sigi = "<<sig_k<<'\t'<<"cutoff = "<<cutoff<<std::endl; 
                    //std::cout<<"mi="<<mi<<'\t'<<"mj="<<mj<<'\t'<<"mk="<<mk<<std::endl;
                    double tmp = real(cutoff);
                    //std::cout<<"cutoff = "<<cutoff<<std::endl; 
                    if(tmp<tolerance) tmp = 0.0;
                    //if(abs(p)<=abs(kmax))

                    //this was set last time 
                    //if(tmp>0.0 && abs(p)<abs(kmax)) 
                    
                    if(tmp>0.0) 
                    {
                        p_config[0].push_back(px);
                        p_config[1].push_back(py);
                        p_config[2].push_back(pz);
                        if(debug=='y')
                        {
                            std::cout << "cutoff = " << cutoff << std::endl;
                            std::cout << "n = " << i << j << k << " nsq = " << nsq << " nmaxsq = " << nmaxsq << std::endl; 
                            std::cout << "px = " << px << " py = " << py << " pz = " << pz << std::endl;
                            std::cout << "p = " << p << " kmax = " << kmax << std::endl;
                        } 
                 
                    }
                    else 
                    {
                        continue; 
                    }
                }
            }
        }
    }
    int check_p_size = 0;
    int psize0 = p_config[0].size();
    int psize1 = p_config[1].size();
    int psize2 = p_config[2].size();

    if(debug=='y')
    {
        std::cout<<"size1 = "<<psize0<<'\t'<<"size2 = "<<psize1<<'\t'<<"size3 = "<<psize2<<std::endl; 
    }

    if(psize0==0 || psize1==0 || psize2==0)
    {
        p_config[0].push_back(0.0);
        p_config[1].push_back(0.0);
        p_config[2].push_back(0.0);
    }
    //std::cout << "nmax = " << nmax << '\t' << "nval = " << (L/(2.0*pi))*abs(kmax) << '\t' 
    //          << "kmax = " << kmax << std::endl;
}

/* This config maker is based upon the cutoff function H(k) instead of kmax
   This config maker returns the shells as well the p_config  */
void config_maker_2(  std::vector< std::vector<comp> > &p_config,
                    std::vector< std::vector<int> > &n_config, 
                    comp En,
                    std::vector<comp> total_P,
                    double mi,
                    double mj, 
                    double mk,
                    double L,
                    double xi, 
                    double epsilon_h,
                    double tolerance    )
{
    char debug = 'n';
    double pi = std::acos(-1.0);

    //comp cutoff = cutoff_function_1(sig_k,mj, mk, epsilon_h);
    //comp kmax = pmom(En,0.0,mi);
    comp kmax = kmax_for_P0(En, mi);

    int nmax = 20;//(int)ceil((L/(2.0*pi))*abs(kmax));
    //std::cout<<"Nmax = "<<nmax<<std::endl;

    int nmaxsq = nmax*nmax; 

    for(int i=-nmax; i<nmax + 1; ++i)
    {
        for(int j=-nmax; j<nmax + 1; ++j)
        {
            for(int k=-nmax; k<nmax + 1; ++k)
            {
                int nsq = i*i + j*j + k*k; 
                //if(nsq<=nmaxsq)
                {
                    comp px = (2.0*pi/(xi*L))*i;
                    comp py = (2.0*pi/(xi*L))*j;
                    comp pz = (2.0*pi/(xi*L))*k; 

                    comp p = std::sqrt(px*px + py*py + pz*pz);

                    comp Px = total_P[0];
                    comp Py = total_P[1];
                    comp Pz = total_P[2];

                    comp Pminusp_x = Px - px;
                    comp Pminusp_y = Py - py;
                    comp Pminusp_z = Pz - pz; 
                    comp Pminusp = std::sqrt(Pminusp_x*Pminusp_x + Pminusp_y*Pminusp_y + Pminusp_z*Pminusp_z);


                    comp sig_k = (En - omega_func(p,mi))*(En - omega_func(p,mi)) - Pminusp*Pminusp; 

                    comp cutoff = cutoff_function_1(sig_k, mj, mk, epsilon_h);

                    //std::cout<<"kmax = "<<kmax<<'\t'<<"p = "<<p<<'\t'<<"sigi = "<<sig_k<<'\t'<<"cutoff = "<<cutoff<<std::endl; 
                    //std::cout<<"mi="<<mi<<'\t'<<"mj="<<mj<<'\t'<<"mk="<<mk<<std::endl;
                    double tmp = real(cutoff);
                    //std::cout<<"cutoff = "<<cutoff<<std::endl; 
                    if(tmp<tolerance) tmp = 0.0;
                    //if(abs(p)<=abs(kmax))

                    //this was set last time 
                    //if(tmp>0.0 && abs(p)<abs(kmax)) 
                    
                    if(tmp>0.0) 
                    {
                        p_config[0].push_back(px);
                        p_config[1].push_back(py);
                        p_config[2].push_back(pz);
                        n_config[0].push_back(i);
                        n_config[1].push_back(j); 
                        n_config[2].push_back(k);
                        if(debug=='y')
                        {
                            std::cout << "cutoff = " << cutoff << std::endl;
                            std::cout << "n = " << i << j << k << " nsq = " << nsq << " nmaxsq = " << nmaxsq << std::endl; 
                            std::cout << "px = " << px << " py = " << py << " pz = " << pz << std::endl;
                            std::cout << "p = " << p << " kmax = " << kmax << std::endl;
                        } 
                 
                    }
                    else 
                    {
                        continue; 
                    }
                }
            }
        }
    }
    int check_p_size = 0;
    int psize0 = p_config[0].size();
    int psize1 = p_config[1].size();
    int psize2 = p_config[2].size();

    if(debug=='y')
    {
        std::cout<<"size1 = "<<psize0<<'\t'<<"size2 = "<<psize1<<'\t'<<"size3 = "<<psize2<<std::endl; 
    }

    if(psize0==0 || psize1==0 || psize2==0)
    {
        p_config[0].push_back(0.0);
        p_config[1].push_back(0.0);
        p_config[2].push_back(0.0);
        n_config[0].push_back(0);
        n_config[0].push_back(0); 
        n_config[0].push_back(0); 
    }
    //std::cout << "nmax = " << nmax << '\t' << "nval = " << (L/(2.0*pi))*abs(kmax) << '\t' 
    //          << "kmax = " << kmax << std::endl;
}


comp particle_energy(   comp spec_k,
                        double m    )
{
    comp omgk = omega_func(spec_k, m); 

    return omgk; 
}

void non_int_spectrum_config_maker( std::vector<std::vector<int> > &n_config,
                                    int nmax, //maximum shell that it is going to loop over
                                    int nsq_max ) //nsq_max is set as a cut off for the shell maker
{
    for(int i=-nmax; i<nmax+1; ++i)
    {
        for(int j=-nmax; j<nmax+1; ++j)
        {
            for(int k=-nmax; k<nmax+1; ++k)
            {
                int nx = i;
                int ny = j;
                int nz = k; 

                int nsq = i*i + j*j + k*k; 
                if(nsq<=nsq_max)
                {
                    n_config[0].push_back(i);
                    n_config[1].push_back(j);
                    n_config[2].push_back(k); 
                }
            }
        }
    }
}

comp threebody_non_int_energy_lab(  double m1, 
                                double m2, 
                                double m3, 
                                comp k1,
                                comp k2, 
                                comp k3 )
{
    comp particle_en1 = particle_energy(k1, m1);
    comp particle_en2 = particle_energy(k2, m2);
    comp particle_en3 = particle_energy(k3, m3);

    comp total_energy = particle_en1 + particle_en2 + particle_en3; 

    return total_energy; 

}

void threebody_non_int_spectrum(    std::string filename, 
                                    double m1, 
                                    double m2, 
                                    double m3, 
                                    std::vector<comp> total_P,
                                    double xi, 
                                    double L,
                                    int nmax, 
                                    int nsq_max )
{
    std::ofstream fout; 
    fout.open(filename.c_str());

    std::vector<std::vector<int> > n_config(3,std::vector<int> ());

    non_int_spectrum_config_maker(n_config, nmax, nsq_max); 

    int size = n_config[0].size(); 

    std::vector<double> energy; 

    double pi = std::acos(-1.0); 
    double twopibyxiL = 2.0*pi/(xi*L);
    
    /* This has been proven multiple times that this is the formulation 
    of non-int spectrum that REDSTAR uses to calculate the non-int spectrum and operators */
    for(int i=0; i<size; ++i)
    {
        for(int j=0; j<size; ++j)
        {
            comp Px = total_P[0];
            comp Py = total_P[1];
            comp Pz = total_P[2];
            comp spec_P = std::sqrt(Px*Px + Py*Py + Pz*Pz);

            comp px1 = twopibyxiL*n_config[0][i];
            comp py1 = twopibyxiL*n_config[1][i];
            comp pz1 = twopibyxiL*n_config[2][i];
            comp spec_p1 = std::sqrt(px1*px1 + py1*py1 + pz1*pz1);
            
            comp px2 = twopibyxiL*n_config[0][j];
            comp py2 = twopibyxiL*n_config[1][j];
            comp pz2 = twopibyxiL*n_config[2][j];
            comp spec_p2 = std::sqrt(px2*px2 + py2*py2 + pz2*pz2);

            comp px3 = Px - px1 - px2; 
            comp py3 = Py - py1 - py2; 
            comp pz3 = Pz - pz1 - pz2; 
            comp spec_p3 = std::sqrt(px3*px3 + py3*py3 + pz3*pz3);

            comp threebodyenergy = threebody_non_int_energy_lab(m1,m2,m3,spec_p1,spec_p2,spec_p3);

            comp Ecm = std::sqrt(threebodyenergy*threebodyenergy - spec_P*spec_P); 

            energy.push_back(real(Ecm)); 
        }
    }
    
   
    std::sort( energy.begin(), energy.end() );
    energy.erase( std::unique( energy.begin(), energy.end() ), energy.end() );

    for(int i=0;i<energy.size();++i)
    {
        //std::cout<<energy[i]<<std::endl; 
        fout<<std::setprecision(20)<<energy[i]<<std::endl;
    }
    fout.close(); 
    std::cout<<"non-int spectrum generated with filename = "<<filename<<std::endl; 


}

void threebody_Gpoles(    std::string filename, 
                                    double m1, 
                                    double m2, 
                                    double m3, 
                                    std::vector<comp> total_P,
                                    double xi, 
                                    double L,
                                    int nmax, 
                                    int nsq_max )
{
    std::ofstream fout; 
    fout.open(filename.c_str());

    std::vector<std::vector<int> > n_config(3,std::vector<int> ());

    non_int_spectrum_config_maker(n_config, nmax, nsq_max); 

    int size = n_config[0].size(); 

    std::vector<double> energy; 

    double pi = std::acos(-1.0); 
    double twopibyxiL = 2.0*pi/(xi*L);
    
    for(int i=0; i<size; ++i)
    {
        for(int j=0; j<size; ++j)
        {
            comp Px = total_P[0];
            comp Py = total_P[1];
            comp Pz = total_P[2];
            comp spec_P = std::sqrt(Px*Px + Py*Py + Pz*Pz);

            comp px1 = twopibyxiL*n_config[0][i];
            comp py1 = twopibyxiL*n_config[1][i];
            comp pz1 = twopibyxiL*n_config[2][i];
            comp spec_p1 = std::sqrt(px1*px1 + py1*py1 + pz1*pz1);
            
            comp px2 = twopibyxiL*n_config[0][j];
            comp py2 = twopibyxiL*n_config[1][j];
            comp pz2 = twopibyxiL*n_config[2][j];
            comp spec_p2 = std::sqrt(px2*px2 + py2*py2 + pz2*pz2);

            comp px3 = Px - px1 - px2; 
            comp py3 = Py - py1 - py2; 
            comp pz3 = Pz - pz1 - pz2; 
            comp spec_p3 = std::sqrt(px3*px3 + py3*py3 + pz3*pz3);

            comp particle_energy1 = particle_energy(spec_p1, m1);
            comp particle_energy2 = particle_energy(spec_p2, m2);
            comp particle_energy3 = particle_energy(spec_p3, m3);

            comp pole_energy1 = particle_energy1 + particle_energy2 + particle_energy3; 
            comp pole_energy2 = particle_energy1 + particle_energy2 - particle_energy3; 

            comp Ecm1 = std::sqrt(pole_energy1*pole_energy1 - spec_P*spec_P); 
            comp Ecm2 = std::sqrt(pole_energy2*pole_energy1 - spec_P*spec_P); 

            energy.push_back(real(Ecm1)); 
            energy.push_back(real(Ecm2)); 
        }
    }
    
   
    std::sort( energy.begin(), energy.end() );
    energy.erase( std::unique( energy.begin(), energy.end() ), energy.end() );

    for(int i=0;i<energy.size();++i)
    {
        //std::cout<<energy[i]<<std::endl; 
        fout<<std::setprecision(20)<<energy[i]<<std::endl;
    }
    fout.close(); 
    std::cout<<"poles of G generated with filename = "<<filename<<std::endl; 


}




#endif