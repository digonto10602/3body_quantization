#include<bits/stdc++.h>

typedef std::complex<double> comp; 

void xj(    std::vector<double> &xj_vec,
            double a, 
            double b,
            int N    )
{
    double h = std::abs(b-a)/((double)N); 
    for(int i=0; i<N+1; ++i)
    {
        double val = a + h*((double)i); 
        xj_vec.push_back(val); 
    }

}


void qj(    std::vector<double> &qj_vec,
            std::vector<double> mu_j_vec,
            std::vector<double> xj_vec  )
{
    int size = xj_vec.size(); 

    for(int j=0; j<size; ++j)
    {
        if(j==0)
        {
            qj_vec[j] = 0.0; 
        }
        else 
        {
            double hj = xj_vec[j] - xj_vec[j-1];
            double hj_plus_1 = xj_vec[j+1] - xj_vec[j]; 

            double lam = hj_plus_1/(hj + hj_plus_1);

            qj_vec[j] = -lam/(mu_j_vec[j]*qj_vec[j-1] + 2.0);
        }
    }
}

void pj(    std::vector<double> &pj_vec,
            std::vector<double> qj_vec,
            std::vector<double> mu_j_vec, 
            std::vector<double> xj_vec  )
{
    int size = xj_vec.size(); 

    for(int j=0; j<size; ++j)
    {
        if(j==0)
        {
            pj_vec[j] = 0.0;
        }
        else 
        {
            pj_vec[j] = mu_j_vec[j]*qj_vec[j-1] + 2.0; 
        }
    }
}

void mu_j(  std::vector<double> &mu_j_vec, 
            std::vector<double> xj_vec  )
{
    int size = xj_vec.size(); 

    for(int i=0; i<size; ++i)
    {
        double lam = 0.0; 
        if(i==0)
        {
            lam = 0.0;
        }
        else 
        {
            double hj = xj_vec[i] - xj_vec[i-1];
            double hj_plus_1 = xj_vec[i+1] - xj_vec[i]; 

            lam = hj_plus_1/(hj + hj_plus_1);
        }

        mu_j_vec[i] = 1.0 - lam; 
    }
}


void Bji(   std::vector<std::vector<double> > &Bji_vec, 
            std::vector<double> xj_vec  )
{
    int size = xj_vec.size(); 

    for(int j=0; j<size; ++j)
    {

        double hj = 0.0;
        double hj_plus_1 = 0.0; 
        if(j==0)
        {
            hj = 0.0;//xj_vec[0];
        }
        else 
        {
            hj = xj_vec[j] - xj_vec[j-1];
        }

        if(j==size-1)
        {
            hj_plus_1 = 0.0;//xj_vec[size-1];
        }
        else 
        {
            hj_plus_1 = xj_vec[j+1] - xj_vec[j]; 
        }


        for(int i=0; i<size; ++i)
        {
            double delta_ij = 0.0;
            double delta_i_jplus1 = 0.0; 
            double delta_i_jminus1 = 0.0; 

            if(i==j+1)
            {
                delta_i_jplus1 = 1.0;
            }
            else 
            {
                delta_i_jplus1 = 0.0;
            }
            
            if(i==j)
            {
                delta_ij = 1.0;
            }
            else 
            {
                delta_ij = 0.0; 
            }

            if(i==j-1)
            {
                delta_i_jminus1 = 1.0;
            }
            else 
            {
                delta_i_jminus1 = 0.0; 
            }

            Bji_vec[j][i] = + delta_i_jplus1*6.0/((hj + hj_plus_1)*hj_plus_1)
                            - delta_ij*6.0/(hj*hj_plus_1)
                            + delta_i_jminus1*6.0/((hj + hj_plus_1)*hj);

        }
    }
}

void Aji(   std::vector<std::vector<double> > &Aji_vec,
            std::vector<std::vector<double> > Bji_vec,
            std::vector<double> mu_j_vec,
            std::vector<double> pj_vec,
            std::vector<double> xj_vec  )
{
    int size = xj_vec.size();

    for(int j=0; j<size; ++j)
    {
        for(int i=0; i<size; ++i)
        {
            if(j==0)
            {
                Aji_vec[j][i] = 0.0; 
            }
            else 
            {
                Aji_vec[j][i] = Bji_vec[j][i]/pj_vec[j] - mu_j_vec[j]*Aji_vec[j-1][i]/pj_vec[j]; 
            }
        }
    }
}

void Cji(   std::vector<std::vector<double> > &Cji_vec, 
            std::vector<std::vector<double> > Aji_vec,
            std::vector<double> qj_vec,
            std::vector<double> xj_vec
        )
{
    int size = xj_vec.size(); 

    for(int j=size-1; j!=-1; --j)
    {
        for(int i=size-1; i!=-1; --i)
        {
            if(j==size-1)
            {
                Cji_vec[j][i] = 0.0;
            }
            else 
            {
                Cji_vec[j][i] = qj_vec[j]*Cji_vec[j+1][i] + Aji_vec[j][i];
            } 
        }
    }
}

void Sij(   std::vector<std::vector<double> > &Sij_vec, 
            double x, 
            std::vector<double> xj_vec, 
            std::vector<std::vector<double> > Cji_vec   )
{
    int size = xj_vec.size(); 
    
    for(int i=0; i<size; ++i)
    {
        for(int j=0; j<size-1; ++j)
        {   
            double delta_ij = 0.0; 
            double delta_iplus1_j = 0.0; 
            if(i==j)
            {
                delta_ij = 1.0;
            }
            else 
            {
                delta_ij = 0.0; 
            }
            
            if(i==j+1)
            {
                delta_iplus1_j = 1.0;
            }
            else 
            {
                delta_iplus1_j = 0.0; 
            }

            double hjplus1 = xj_vec[j+1] - xj_vec[j];
            
            double xminusxj = x - xj_vec[j]; 
            
            double term1 = (delta_iplus1_j - delta_ij)/hjplus1 ;
            double term2 = (hjplus1/6.0)*(2.0*Cji_vec[j][i] + Cji_vec[j+1][i]) ;
            double term3 = 0.5*Cji_vec[j][i] + xminusxj*(1.0/(6.0*hjplus1))*(Cji_vec[j+1][i] - Cji_vec[j][i]) ;
            
            double Sij_val =    delta_ij 
                                + xminusxj*(term1 - term2 + xminusxj*term3);

            Sij_vec[i][j] = Sij_val; 
        
        }

    }
}

void Sij_builder(   double x,
                    std::vector<double> xj_vec,
                    std::vector<std::vector<double> > &Sij_vec  )
{
    int size = xj_vec.size(); 

    std::vector<double> pj_vec(size);
    std::vector<double> qj_vec(size); 
    std::vector<double> mu_j_vec(size); 
    std::vector<std::vector<double> > Aji_vec(size,std::vector<double>(size)); 
    std::vector<std::vector<double> > Bji_vec(size,std::vector<double>(size)); 
    std::vector<std::vector<double> > Cji_vec(size,std::vector<double>(size)); 

    mu_j(mu_j_vec, xj_vec); 

    qj(qj_vec, mu_j_vec, xj_vec); 

    pj(pj_vec, qj_vec, mu_j_vec, xj_vec); 

    Bji(Bji_vec, xj_vec); 

    Aji(Aji_vec, Bji_vec, mu_j_vec, pj_vec, xj_vec); 

    Cji(Cji_vec, Aji_vec, qj_vec, xj_vec); 

    Sij(Sij_vec, x, xj_vec, Cji_vec); 
}


void print_vec_1D(  std::string cout_var, 
                    std::vector<double> vec )
{
    for(int i=0; i<vec.size(); ++i)
    {
        std::cout<<cout_var<<"["<<i<<"]="<<vec[i]<<std::endl; 
    }
}

void print_vec_2D(  std::string cout_var, 
                    std::vector<std::vector<double> > vec )
{
    int size1 = vec.size(); 
    int size2 = vec[0].size(); 

    for(int i=0; i<size1; ++i)
    {
        for(int j=0; j<size2; ++j)
        {
            std::cout<<cout_var<<"["<<i<<","<<j<<"]="<<vec[i][j]<<std::endl; 
        }
    }
}

double random_test_function_1( double x )
{
    //-10*x + 23*x^2  + 32*x^5/Sqrt[x]
    double A = -10*x;
    double B = 23*x*x;
    double C = 32*x*x*x*x*x/std::sqrt(x);
    double D = std::sqrt(x)*std::pow(x,5); 

    return A + B + C ; 
}

void test_spline_function_1()
{
    double a = 0.000001; 
    double b = 1.0; 
    int N = 50; 

    std::vector<double> xj_vec; 
    xj(xj_vec, a, b, N); 

    std::vector<double> fj_vec; 

    int size = xj_vec.size(); 

    for(int i=0; i<size; ++i)
    {
        fj_vec.push_back(random_test_function_1(xj_vec[i]));
    }

    std::string filename1 = "xj_fj.dat";
    std::ofstream fout; 
    fout.open(filename1.c_str()); 
    for(int i=0; i<size; ++i)
    {
        fout<<std::setprecision(20)<<xj_vec[i]<<'\t'<<fj_vec[i]<<std::endl; 
    }
    fout.close(); 

    std::string filename2 = "splines_fdata.dat";
    fout.open(filename2.c_str()); 
    double N1 = 100000; 

    for(int i=0; i<N1+1; ++i)
    {
        double h = std::abs(b-a)/((double)N1); 
        double xval = a + ((double)i)*h; 

        std::vector<std::vector<double> > Sij_vec1(size, std::vector<double> (size)); 
        Sij_builder(xval, xj_vec, Sij_vec1);

        int j_val = 0; 
        for(int j=0; j<xj_vec.size()-1; ++j)
        {
            double xj_check1 = xj_vec[j];
            double xj_check2 = xj_vec[j+1]; 

            if(xval>=xj_check1 && xval<=xj_check2)
            {
                j_val = j; 
            }
        }

        double f_spline_val = 0.0; 

        for(int k=0; k<xj_vec.size(); ++k)
        {
            f_spline_val = f_spline_val + Sij_vec1[k][j_val]*fj_vec[k];
        }

        fout<<std::setprecision(20)<<xval<<'\t'<<random_test_function_1(xval)<<'\t'<<f_spline_val<<std::endl; 
        std::cout<<xval<<'\t'<<random_test_function_1(xval)<<'\t'<<f_spline_val<<std::endl; 

    }

    fout.close(); 
}

void test_spline()
{
    //std::vector<std::vector<double> > Cji_vec(3,std::vector<double> (3)); 
    //Cji();

    double a = 0.0; 
    double b = 1.0; 
    int N = 10; 

    std::vector<double> xj_vec; 
    xj(xj_vec, a, b, N); 

    int size = xj_vec.size(); 
    double x = 0.51; 
    std::vector<std::vector<double> > Sij_vec(size, std::vector<double> (size)); 
    Sij_builder(x, xj_vec, Sij_vec);

    std::cout<<"size = "<<size<<std::endl; 
    
    print_vec_1D("xj",xj_vec);
    print_vec_2D("Sij",Sij_vec);

    std::ofstream fout; 
    std::string filename = "check_spline.dat"; 
    fout.open(filename.c_str()); 

    int N1 = 1000;

    int fixed_i = xj_vec.size()/2; 

    for(int i=0; i<N1+1; ++i)
    {
        double h = abs(b-a)/((double)N1); 
        double xval = a + i*h; 

        std::vector<std::vector<double> > Sij_vec1(size, std::vector<double> (size)); 
        Sij_builder(xval, xj_vec, Sij_vec1);

        int j_val = 0; 
        for(int j=0; j<xj_vec.size()-1; ++j)
        {
            double xj_check1 = xj_vec[j];
            double xj_check2 = xj_vec[j+1]; 

            if(xval>=xj_check1 && xval<=xj_check2)
            {
                j_val = j; 
            }
        }

        std::cout<<"xval = "<<xval<<'\t'<<"fixed i = "<<fixed_i<<'\t'
                 <<"selected j = "<<j_val<<'\t'
                 <<"Sij = "<<Sij_vec1[fixed_i][j_val]<<std::endl; 

        fout<<xval<<'\t'
            <<Sij_vec1[fixed_i][j_val]
            <<std::endl; 

        double summ = 0.0; 
        for(int k=0;k<size;++k)
        {
            summ = summ + Sij_vec1[k][j_val];
        }    
        std::cout<<"sum = "<<summ<<std::endl; 
    }

    fout.close(); 



}

int main()
{
    test_spline_function_1();

    return 0; 
}