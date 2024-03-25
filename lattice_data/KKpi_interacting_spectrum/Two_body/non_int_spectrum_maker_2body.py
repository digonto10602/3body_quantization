#This code makes the non-interacting spectrum for given particle with mass and
#lattice anisotropy xi and lattice volume L/a_s (Lbyas)

import numpy as np 
import matplotlib as mpl 
import matplotlib.pyplot as plt 
import scipy.interpolate
from mpl_toolkits.axes_grid1 import make_axes_locatable
import os.path
from scipy.optimize import curve_fit
import scipy.interpolate
import pandas as pd 

def config_maker( max_nsq ):
    nx = []
    ny = []
    nz = []

    count = 1
    for i in range(-max_nsq,max_nsq+1,1):
        for j in range(-max_nsq,max_nsq+1,1):
            for k in range(-max_nsq,max_nsq+1,1):
                config_sq = i*i + j*j + k*k 
                if(config_sq<=max_nsq):
                    print("config ",count,": ",i,j,k)
                    count = count + 1 
                    nx.append(i)
                    ny.append(j)
                    nz.append(k)
                else:
                    continue
        
    return nx,ny,nz 


def energy( atm, 
            xi, 
            Lbyas,
            nx,
            ny,
            nz  ):
    onebyxi = 1.0/xi 
    pi = np.pi
    twopibyLbyas = 2.0*pi/Lbyas 
    nsquare = nx*nx + ny*ny + nz*nz 

    return np.sqrt(atm*atm + onebyxi*onebyxi*twopibyLbyas*twopibyLbyas*nsquare)


def irrep_list_maker(energy_file):
    df = pd.read_csv(energy_file,delim_whitespace=True, header=None)
    dflist = df.values.tolist()
    irrep_list = []
    for i in range(len(dflist)):
        irrep = dflist[i][1]
    
        if len(irrep_list)==0:
            irrep_list.append(irrep)
        else:
            check_irrep_list_flag = 0
            for j in range(len(irrep_list)):
                temp_irrep = irrep_list[j]
            
                if(temp_irrep == irrep):
                    check_irrep_list_flag = 1
                    break 
            if(check_irrep_list_flag==0):
                irrep_list.append(irrep)

    return dflist, irrep_list 

def irrep_energy_list_maker(full_energy_list, fixed_irrep):
    Ecm_list = []
    Elat_list = []

    for i in range(len(full_energy_list)):
        if(full_energy_list[i][1]==fixed_irrep):
            Ecm_list.append(float(full_energy_list[i][2]))
            Elat_list.append(float(full_energy_list[i][3]))

    return Ecm_list, Elat_list 

def canonical_mom_maker(nx, ny, nz):
    temp_list = []
    temp_list.append(abs(nx))
    temp_list.append(abs(ny))
    temp_list.append(abs(nz))
    temp_list.sort(reverse=True)
    
    return temp_list

def nonint_spectrum_maker(atm, xi, max_nsq, irrep, Lmin, Lmax, Lpoints, energy_file):
    
    #tolerance for energy matching 
    tolerance = 1.0e-3
    #we get the full channel.energies files and the unique irrep list 
    full_energy_list, irrep_list = irrep_list_maker(energy_file)
    Lbyas = int(full_energy_list[0][0])
    print("L = ",Lbyas)
    #irrep = irrep_list[5]
    print("irrep = ",irrep)

    #we create the momentum configurations possible for max n^2
    nx, ny, nz = config_maker(max_nsq)

    #we take the frame momentum from the irrep string 
    frame_mom_str = irrep.split("_")
    frame_n = []
    for i in frame_mom_str[0]:
        frame_n.append(int(i))
    config_num = len(nx)
    frame_nx = frame_n[0]
    frame_ny = frame_n[1]
    frame_nz = frame_n[2]


    #we calculate the momentum square of the frame 
    onebyxi = 1.0/xi 
    pi = np.pi
    twopibyLbyas = 2.0*pi/Lbyas 
    frame_nsquare = frame_nx*frame_nx + frame_ny*frame_ny + frame_nz*frame_nz 
    print("f sq = ",frame_nsquare)
    frame_momsq = onebyxi*onebyxi*twopibyLbyas*twopibyLbyas*frame_nsquare

    #list for selected config numbers
    selected_config_list1 = []
    selected_config_list2 = []
    irrep_Ecm_list, irrep_Elat_list = irrep_energy_list_maker(full_energy_list,irrep)
    completed_ecm = []

    for i in range(config_num):
        for j in range(config_num):
            particle1_energy = energy(atm,xi,Lbyas,nx[i],ny[i],nz[i])
            particle2_energy = energy(atm,xi,Lbyas,nx[j],ny[j],nz[j])

            E_lat = particle1_energy + particle2_energy

            Ecm = np.sqrt(E_lat*E_lat - frame_momsq)
            #print(Ecm)
            

            for k in range(len(irrep_Ecm_list)):
                irrep_en = irrep_Ecm_list[k]
                irrep_en_diff = abs(irrep_en - Ecm)
                check_completed_ecm_flag = 0
                if(irrep_en_diff<=tolerance):
                    if(len(completed_ecm)==0):
                        selected_config_list1.append(i)
                        selected_config_list2.append(j)
                        completed_ecm.append(irrep_en)
                        continue 
                    else:
                        for l in range(len(completed_ecm)):
                            diff_completed_ecm = abs(irrep_en - completed_ecm[l])

                            if(diff_completed_ecm<=tolerance):
                                check_completed_ecm_flag = 1
                    
                    if(check_completed_ecm_flag==0):
                        selected_config_list1.append(i)
                        selected_config_list2.append(j)
                        completed_ecm.append(irrep_en)

    #print(irrep_Ecm_list)
    #print(completed_ecm)
    #print(selected_config_list1)
    #print(selected_config_list2)
    out_file_list = np.zeros((len(selected_config_list1)+1,Lpoints)) 
    
    for i in range(len(selected_config_list1)):
        ind1 = selected_config_list1[i]
        ind2 = selected_config_list2[i]

        canonical_n_list1 = canonical_mom_maker(nx[ind1],ny[ind1],nz[ind1])
        
        canonical_n_list2 = canonical_mom_maker(nx[ind2],ny[ind2],nz[ind2]) 

        L = np.linspace(Lmin,Lmax,Lpoints)

        

        for l_ind in range(Lpoints):

            onebyxi = 1.0/xi 
            pi = np.pi
            twopibyLbyas = 2.0*pi/L[l_ind] 
            frame_nsquare = frame_nx*frame_nx + frame_ny*frame_ny + frame_nz*frame_nz 
            #print("f sq = ",frame_nsquare)
            frame_momsq = onebyxi*onebyxi*twopibyLbyas*twopibyLbyas*frame_nsquare

            particle1_energy = energy(atm,xi,L[l_ind],nx[ind1],ny[ind1],nz[ind1])
            particle2_energy = energy(atm,xi,L[l_ind],nx[ind2],ny[ind2],nz[ind2])

            E_lat = particle1_energy + particle2_energy

            Ecm = np.sqrt(E_lat*E_lat - frame_momsq)

            out_file_list[0][l_ind] = L[l_ind] 
            out_file_list[i+1][l_ind] = Ecm 


    out_file = "non_int." + str(irrep) 
    f = open(out_file,'w')

    for i in range(len(out_file_list[0])):
        for j in range(len(out_file_list)):
            f.write(str(out_file_list[j][i]) + '\t')
            #print(out_file_list[i][j], sep=",")
        f.write('\n')
        #print("\n")        
        #print("p1:",canonical_n_list1[0],canonical_n_list1[1],canonical_n_list1[2],
        #      "p2:",canonical_n_list2[0],canonical_n_list2[1],canonical_n_list2[2])
    print(len(out_file_list))
    print(len(out_file_list[0]))
    f.close() 

def nonint_spectrum_maker_trivial_irrep(atm1, atm2, xi, max_nsq, irrep, Lmin, Lmax, Lpoints):
    
    #tolerance for energy matching 
    tolerance = 1.0e-3
    #we get the full channel.energies files and the unique irrep list 
    #full_energy_list, irrep_list = irrep_list_maker(energy_file)
    Lbyas = 20 #int(full_energy_list[0][0])
    print("L = ",Lbyas)
    #irrep = irrep_list[5]
    print("irrep = ",irrep)

    #we create the momentum configurations possible for max n^2
    nx, ny, nz = config_maker(max_nsq)

    #we take the frame momentum from the irrep string 
    frame_mom_str = irrep.split("_")
    frame_n = []
    for i in frame_mom_str[0]:
        frame_n.append(int(i))
    config_num = len(nx)
    frame_nx = frame_n[0]
    frame_ny = frame_n[1]
    frame_nz = frame_n[2]


    #we calculate the momentum square of the frame 
    onebyxi = 1.0/xi 
    pi = np.pi
    twopibyLbyas = 2.0*pi/Lbyas 
    frame_nsquare = frame_nx*frame_nx + frame_ny*frame_ny + frame_nz*frame_nz 
    print("f sq = ",frame_nsquare)
    frame_momsq = onebyxi*onebyxi*twopibyLbyas*twopibyLbyas*frame_nsquare

    #list for selected config numbers
    selected_config_list1 = []
    selected_config_list2 = []
    #irrep_Ecm_list, irrep_Elat_list = irrep_energy_list_maker(full_energy_list,irrep)
    completed_ecm = []

    Ecm_list = []
    Elat_list = []
    config1_list = []
    config2_list = []
    
    for i in range(config_num):
        nx1 = nx[i]
        ny1 = ny[i]
        nz1 = nz[i]

        nx2 = frame_nx - nx1 
        ny2 = frame_ny - ny1 
        nz2 = frame_nz - nz1 
        
        particle1_energy = energy(atm1,xi,Lbyas,nx1,ny1,nz1)
        particle2_energy = energy(atm2,xi,Lbyas,nx2,ny2,nz2)

        E_lat = particle1_energy + particle2_energy

        Ecm = np.sqrt(E_lat*E_lat - frame_momsq)


    
    for i in range(config_num):
        for j in range(config_num):
            particle1_energy = energy(atm,xi,Lbyas,nx[i],ny[i],nz[i])
            particle2_energy = energy(atm,xi,Lbyas,nx[j],ny[j],nz[j])

            E_lat = particle1_energy + particle2_energy

            Ecm = np.sqrt(E_lat*E_lat - frame_momsq)
            #print(Ecm)
            

            for k in range(len(irrep_Ecm_list)):
                irrep_en = irrep_Ecm_list[k]
                irrep_en_diff = abs(irrep_en - Ecm)
                check_completed_ecm_flag = 0
                if(irrep_en_diff<=tolerance):
                    if(len(completed_ecm)==0):
                        selected_config_list1.append(i)
                        selected_config_list2.append(j)
                        completed_ecm.append(irrep_en)
                        continue 
                    else:
                        for l in range(len(completed_ecm)):
                            diff_completed_ecm = abs(irrep_en - completed_ecm[l])

                            if(diff_completed_ecm<=tolerance):
                                check_completed_ecm_flag = 1
                    
                    if(check_completed_ecm_flag==0):
                        selected_config_list1.append(i)
                        selected_config_list2.append(j)
                        completed_ecm.append(irrep_en)

    #print(irrep_Ecm_list)
    #print(completed_ecm)
    #print(selected_config_list1)
    #print(selected_config_list2)
    out_file_list = np.zeros((len(selected_config_list1)+1,Lpoints)) 
    
    for i in range(len(selected_config_list1)):
        ind1 = selected_config_list1[i]
        ind2 = selected_config_list2[i]

        canonical_n_list1 = canonical_mom_maker(nx[ind1],ny[ind1],nz[ind1])
        
        canonical_n_list2 = canonical_mom_maker(nx[ind2],ny[ind2],nz[ind2]) 

        L = np.linspace(Lmin,Lmax,Lpoints)

        

        for l_ind in range(Lpoints):

            onebyxi = 1.0/xi 
            pi = np.pi
            twopibyLbyas = 2.0*pi/L[l_ind] 
            frame_nsquare = frame_nx*frame_nx + frame_ny*frame_ny + frame_nz*frame_nz 
            #print("f sq = ",frame_nsquare)
            frame_momsq = onebyxi*onebyxi*twopibyLbyas*twopibyLbyas*frame_nsquare

            particle1_energy = energy(atm,xi,L[l_ind],nx[ind1],ny[ind1],nz[ind1])
            particle2_energy = energy(atm,xi,L[l_ind],nx[ind2],ny[ind2],nz[ind2])

            E_lat = particle1_energy + particle2_energy

            Ecm = np.sqrt(E_lat*E_lat - frame_momsq)

            out_file_list[0][l_ind] = L[l_ind] 
            out_file_list[i+1][l_ind] = Ecm 


    out_file = "non_int." + str(irrep) 
    f = open(out_file,'w')

    for i in range(len(out_file_list[0])):
        for j in range(len(out_file_list)):
            f.write(str(out_file_list[j][i]) + '\t')
            #print(out_file_list[i][j], sep=",")
        f.write('\n')
        #print("\n")        
        #print("p1:",canonical_n_list1[0],canonical_n_list1[1],canonical_n_list1[2],
        #      "p2:",canonical_n_list2[0],canonical_n_list2[1],canonical_n_list2[2])
    print(len(out_file_list))
    print(len(out_file_list[0]))
    f.close() 


energy_file = "S2I1p_energies_L24"

full_list, irrep_list = irrep_list_maker(energy_file)
#nx,ny,nz = config_maker(4)
print(irrep_list)
atm = 0.09698
xi = 3.444
max_nsq = 4
Lmin = 19
Lmax = 25 
Lpoints = 2000 
full_energy_list, irrep_list = irrep_list_maker(energy_file)

#irrep = "200_A1"
#nonint_spectrum_maker(atm, xi, max_nsq, irrep, Lmin, Lmax, Lpoints, energy_file)
for i in range(len(irrep_list)):
    irrep = irrep_list[i]
    nonint_spectrum_maker(atm, xi, max_nsq, irrep, Lmin, Lmax, Lpoints, energy_file)

