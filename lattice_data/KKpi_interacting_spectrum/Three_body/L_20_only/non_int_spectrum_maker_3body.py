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
from IPython.display import display

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
                    #print("config ",count,": ",i,j,k)
                    count = count + 1 
                    nx.append(i)
                    ny.append(j)
                    nz.append(k)
                else:
                    continue
        
    return nx,ny,nz 

def config_maker_positive_only( max_nsq ):
    nx = []
    ny = []
    nz = []

    count = 1
    for i in range(0,max_nsq+1,1):
        for j in range(0,max_nsq+1,1):
            for k in range(0,max_nsq+1,1):
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

def nonint_spectrum_maker(atmk, atmpi, xi, max_nsq, irrep, Lmin, Lmax, Lpoints, energy_file):
    
    #tolerance for energy matching 
    tolerance = 1.0e-4
    tolerance2 = 1.0e-9
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
    frame_momsq = onebyxi*onebyxi*twopibyLbyas*twopibyLbyas*frame_nsquare

    #list for selected config numbers
    selected_config_list1 = []
    selected_config_list2 = []
    selected_config_list3 = []
    irrep_Ecm_list, irrep_Elat_list = irrep_energy_list_maker(full_energy_list,irrep)
    completed_ecm = []
    check_ecm_list = []
    check_config1_list = []
    check_config2_list = []
    check_config3_list = []
     
    for i in range(config_num):
        for j in range(config_num):
            for k in range(config_num):
                particle1_energy = energy(atmk,xi,Lbyas,nx[i],ny[i],nz[i])
                particle2_energy = energy(atmk,xi,Lbyas,nx[j],ny[j],nz[j])
                particle3_energy = energy(atmpi,xi,Lbyas,nx[k],ny[k],nz[k])

                E_lat = particle1_energy + particle2_energy + particle3_energy

                Ecm = np.sqrt(E_lat*E_lat - frame_momsq)
                #print(Ecm)
                #check_ecm_list.append(Ecm)
                #check_config1_list.append(i)
                #check_config2_list.append(j)
                #check_config3_list.append(k)


                for kl in range(len(irrep_Ecm_list)):
                    irrep_en = irrep_Ecm_list[kl]
                    irrep_en_diff = abs(irrep_en - Ecm)
                    
                    check_completed_ecm_flag = 0
                    if(irrep_en_diff<tolerance):
                        if(len(completed_ecm)==0):
                            selected_config_list1.append(i)
                            selected_config_list2.append(j)
                            selected_config_list3.append(k)
                            completed_ecm.append(irrep_en)
                            #print(irrep_en, Ecm, irrep_en_diff)
                            continue 
                        else:
                            for l in range(len(completed_ecm)):
                                diff_completed_ecm = abs(irrep_en - completed_ecm[l])

                                if(diff_completed_ecm<tolerance2):
                                    check_completed_ecm_flag = 1
                    
                        if(check_completed_ecm_flag==0):
                            selected_config_list1.append(i)
                            selected_config_list2.append(j)
                            selected_config_list3.append(k)
                            completed_ecm.append(irrep_en)
                            #print(irrep_en, Ecm, irrep_en_diff)


    #check_ecm_config = pd.DataFrame(list(zip(check_ecm_list,check_config1_list,check_config2_list,check_config3_list)))
    #final_check_ecm_config = check_ecm_config.sort_values(check_ecm_config.columns[0], ascending=True)
    #print(final_check_ecm_config)
    #print("irrep_Ecm_list = ",irrep_Ecm_list)
    #print("completed Ecm list = ",completed_ecm)
    #print("config 1 =",selected_config_list1)
    #print("config 2 =",selected_config_list2)
    #print("config 3 =",selected_config_list3)
    
    out_file_list = np.zeros((len(selected_config_list1)+1,Lpoints)) 
    #header_list = []
    #header_list.append("L")
    
    for i in range(len(selected_config_list1)):
        ind1 = selected_config_list1[i]
        ind2 = selected_config_list2[i]
        ind3 = selected_config_list3[i]


        canonical_n_list1 = canonical_mom_maker(nx[ind1],ny[ind1],nz[ind1])
        
        canonical_n_list2 = canonical_mom_maker(nx[ind2],ny[ind2],nz[ind2]) 

        canonical_n_list3 = canonical_mom_maker(nx[ind3],ny[ind3],nz[ind3])

        L = np.linspace(Lmin,Lmax,Lpoints)

        #temp_canonical = []
        #temp_canonical.append(",".join(canonical_n_list1))
        #temp_canonical.append(",".join(canonical_n_list2))
        #temp_canonical.append(",".join(canonical_n_list3))
        #temp_configs = ";".join(temp_canonical) 

        #header_list.append(temp_configs)
        

        for l_ind in range(Lpoints):

            onebyxi = 1.0/xi 
            pi = np.pi
            twopibyLbyas = 2.0*pi/L[l_ind] 
            frame_nsquare = frame_nx*frame_nx + frame_ny*frame_ny + frame_nz*frame_nz 
            frame_momsq = onebyxi*onebyxi*twopibyLbyas*twopibyLbyas*frame_nsquare

            particle1_energy = energy(atmk,xi,L[l_ind],nx[ind1],ny[ind1],nz[ind1])
            particle2_energy = energy(atmk,xi,L[l_ind],nx[ind2],ny[ind2],nz[ind2])
            particle3_energy = energy(atmpi,xi,L[l_ind],nx[ind3],ny[ind3],nz[ind3])

            E_lat = particle1_energy + particle2_energy + particle3_energy

            Ecm = np.sqrt(E_lat*E_lat - frame_momsq)
            #print(Ecm, E_lat, frame_momsq)

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
    #print(len(out_file_list))
    #print(len(out_file_list[0]))
    f.close() 
    print("file generated = ",out_file)


def full_nonint_spectrum_maker(atmk, atmpi, xi, max_nsq, irrep, Lmin, Lmax, Lpoints, energy_file):
    
    #tolerance for energy matching 
    tolerance = 1.0e-4
    tolerance2 = 1.0e-9
    #we get the full channel.energies files and the unique irrep list 
    full_energy_list, irrep_list = irrep_list_maker(energy_file)
    Lbyas = int(full_energy_list[0][0])
    print("L = ",Lbyas)
    #irrep = irrep_list[5]
    print("irrep = ",irrep)

    #we create the momentum configurations possible for max n^2
    nx, ny, nz = config_maker(max_nsq)

    print("config num = ",len(nx))

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

    print(nx,ny,nz)
    print("frame mom = ",frame_nx,frame_ny,frame_nz)
    #frame_nsquare = frame_nx*frame_nx + frame_ny*frame_ny + frame_nz*frame_nz 
    #frame_momsq = onebyxi*onebyxi*twopibyLbyas*twopibyLbyas*frame_nsquare


    out_file_list = np.zeros((config_num*config_num+1,Lpoints)) 
    L = np.linspace(Lmin,Lmax,Lpoints)
    #L = []
    #L.append(20)
    count = 0
    for i in range(config_num):
        for j in range(config_num):
 
            for l_ind in range(Lpoints):
                #l_ind=0
                onebyxi = 1.0/xi 
                pi = np.pi
                twopibyLbyas = 2.0*pi/L[l_ind] 
                frame_nsquare = frame_nx*frame_nx + frame_ny*frame_ny + frame_nz*frame_nz 
                frame_momsq = onebyxi*onebyxi*twopibyLbyas*twopibyLbyas*frame_nsquare
                

                particle1_energy = energy(atmk,xi,L[l_ind],nx[i],ny[i],nz[i])
                particle2_energy = energy(atmk,xi,L[l_ind],nx[j],ny[j],nz[j])
                #particle3_energy = energy(atmpi,xi,Lbyas,nx[k],ny[k],nz[k])
                particle3_energy = energy(atmpi,xi,L[l_ind],frame_nx - (nx[i] + nx[j]),
                                                            frame_ny - (ny[i] + ny[j]),
                                                            frame_nz - (nz[i] + nz[j]))
                #particle3_energy = energy(atmpi,xi,L[l_ind], - (nx[i] + nx[j]),
                #                                             - (ny[i] + ny[j]),
                #                                             - (nz[i] + nz[j]))

                E_lat = particle1_energy + particle2_energy + particle3_energy
                Ecm = np.sqrt(E_lat*E_lat - frame_momsq)
                #if(abs(Ecm - 0.3436)<=tolerance):
                #    print("Ecm = 0.3436 configs:")
                #    print("L=",L[l_ind])
                #    print(nx[i],ny[i],nz[i])
                #    print(nx[j],ny[j],nz[j])
                #    print(frame_nx - (nx[i] + nx[j]),
                #          frame_ny - (ny[i] + ny[j]),
                #          frame_nz - (nz[i] + nz[j]))
                #    print("actual Ecm = ",Ecm)
                #    print("==========================")
                #if(abs(Ecm - 0.3468)<=tolerance):
                #    print("Ecm = 0.3468 configs:")
                #    print(nx[i],ny[i],nz[i])
                #    print(nx[j],ny[j],nz[j])
                #    print(frame_nx - (nx[i] + nx[j]),
                #          frame_ny - (ny[i] + ny[j]),
                #          frame_nz - (nz[i] + nz[j]))
                #    print("actual Ecm = ",Ecm)
                #    print("==========================")

                out_file_list[0][l_ind] = L[l_ind] 
                out_file_list[count+1][l_ind] = Ecm 
            count = count + 1
                #print("nx1=",nx[i],"ny1=",ny[i],"nz1=",nz[i])
                #print("nx2=",nx[j],"ny2=",ny[j],"nz2=",nz[j])
                #print("nx3=",frame_nx - (nx[i] + nx[j]),"ny3=",frame_ny - (ny[i] + ny[j]),"nz3=",frame_nz - (nz[i] + nz[j]))
                #print("Ecm=",Ecm)
                



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
    #print(len(out_file_list))
    #print(len(out_file_list[0]))
    f.close() 
    print("file generated = ",out_file)


#This will be used from now on to create non-int spectrum 
#Works only for the trivial irreps 
def full_nonint_spectrum_maker_final(atmk, atmpi, xi, max_nsq, irrep, Lmin, Lmax, Lpoints, energy_file):
    
    #tolerance for energy matching 
    tolerance = 1.0e-4
    tolerance2 = 1.0e-9
    #we get the full channel.energies files and the unique irrep list 
    full_energy_list, irrep_list = irrep_list_maker(energy_file)
    Lbyas = int(full_energy_list[0][0])
    print("L = ",Lbyas)
    #irrep = irrep_list[5]
    print("irrep = ",irrep)

    #we create the momentum configurations possible for max n^2
    nx, ny, nz = config_maker(max_nsq)

    print("config num = ",len(nx))

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

    #print(nx,ny,nz)
    print("frame mom = ",frame_nx,frame_ny,frame_nz)
    #frame_nsquare = frame_nx*frame_nx + frame_ny*frame_ny + frame_nz*frame_nz 
    #frame_momsq = onebyxi*onebyxi*twopibyLbyas*twopibyLbyas*frame_nsquare

    #This dataframe is for L = 20 to check with redstar generated non-int spectrum 
    Ecm_list = []
    Elat_list = []
    particle1_config_list = []
    particle2_config_list = []
    particle3_config_list = []

    for i in range(config_num):
        for j in range(config_num):
            onebyxi = 1.0/xi 
            pi = np.pi
            twopibyLbyas = 2.0*pi/Lbyas 
            frame_nsquare = frame_nx*frame_nx + frame_ny*frame_ny + frame_nz*frame_nz 
            frame_momsq = onebyxi*onebyxi*twopibyLbyas*twopibyLbyas*frame_nsquare

            nx1 = nx[i]
            ny1 = ny[i]
            nz1 = nz[i]

            nx2 = nx[j]
            ny2 = ny[j]
            nz2 = nz[j]

            nx3 = frame_nx - (nx[i] + nx[j])    
            ny3 = frame_ny - (ny[i] + ny[j])
            nz3 = frame_nz - (nz[i] + nz[j])

            particle1_energy = energy(atmk,xi,Lbyas,nx[i],ny[i],nz[i])
            particle2_energy = energy(atmk,xi,Lbyas,nx[j],ny[j],nz[j])
            particle3_energy = energy(atmpi,xi,Lbyas,nx3,ny3,nz3)

            E_lat = particle1_energy + particle2_energy + particle3_energy
            Ecm = np.sqrt(E_lat*E_lat - frame_momsq)
            Ecm_list.append(Ecm)
            Elat_list.append(E_lat)
            particle1_config_list.append([nx1,ny1,nz1])
            particle2_config_list.append([nx2,ny2,nz2])
            particle3_config_list.append([nx3,ny3,nz3])


    Energy_df = pd.DataFrame(list(zip(Elat_list, Ecm_list, particle1_config_list, particle2_config_list, particle3_config_list)))

    #display(Energy_df[4])        

    sorted_energy_df = Energy_df.sort_values(1)
    sorted_energy_df = sorted_energy_df.reset_index(drop=True)
    #print(sorted_energy_df)

    
    unique_energy_df = sorted_energy_df[1].unique()

    non_int_L20_points_file = "non_int_L20_points_" + str(irrep) + ".dat"
    
    f = open(non_int_L20_points_file,"w")

    for i in range(len(unique_energy_df)):
        f.write(    str(Lbyas) + '\t' 
                +   str(unique_energy_df[i]) + '\n')
        
    f.close()
        
    selected_non_int_Ecm = []
    selected_non_int_Elat = []
    selected_particle1_config = []
    selected_particle2_config = []
    selected_particle3_config = []
    completed_ecm_list = []

    selection_flag = 0
    selection_counter = 0

    for i in range(len(sorted_energy_df[0])):
        val1 = sorted_energy_df[0][i]
        val2 = sorted_energy_df[1][i]
        val3 = sorted_energy_df[2][i]
        val4 = sorted_energy_df[3][i]
        val5 = sorted_energy_df[4][i]
        
        if(selection_counter>=len(unique_energy_df)):
            break 
        
        present_Ecm = unique_energy_df[selection_counter]
        selected = 1
        if(val2==present_Ecm):
            selection_counter = selection_counter + 1
            selected_non_int_Elat.append(val1) 
            selected_non_int_Ecm.append(val2) 
            selected_particle1_config.append(val3)
            selected_particle2_config.append(val4)
            selected_particle3_config.append(val5)
            selected = 0
        if(selected == 0):
            print(val1,val2,val3,val4,val5," selected")
        else:
            print(val1,val2,val3,val4,val5," not selected")
    
    print(unique_energy_df)
    
    out_file_list = np.zeros((config_num*config_num+1,Lpoints)) 
    L = np.linspace(Lmin,Lmax,Lpoints)

    count = 0
    for i in range(len(selected_particle1_config)):
        nx1 = selected_particle1_config[i][0]
        ny1 = selected_particle1_config[i][1]
        nz1 = selected_particle1_config[i][2]

        nx2 = selected_particle2_config[i][0]
        ny2 = selected_particle2_config[i][1]
        nz2 = selected_particle2_config[i][2]

        nx3 = selected_particle3_config[i][0]
        ny3 = selected_particle3_config[i][1]
        nz3 = selected_particle3_config[i][2]


        for l_ind in range(Lpoints):
            onebyxi = 1.0/xi 
            pi = np.pi
            twopibyLbyas = 2.0*pi/L[l_ind] 
            frame_nsquare = frame_nx*frame_nx + frame_ny*frame_ny + frame_nz*frame_nz 
            frame_momsq = onebyxi*onebyxi*twopibyLbyas*twopibyLbyas*frame_nsquare

            particle1_energy = energy(atmk,xi,L[l_ind],nx1,ny1,nz1)
            particle2_energy = energy(atmk,xi,L[l_ind],nx2,ny2,nz2)
            particle3_energy = energy(atmpi,xi,L[l_ind],nx3,ny3,nz3)

            E_lat = particle1_energy + particle2_energy + particle3_energy
            Ecm = np.sqrt(E_lat*E_lat - frame_momsq)
            out_file_list[0][l_ind] = L[l_ind] 
            out_file_list[count+1][l_ind] = Ecm 
        count = count + 1
    


    out_file = "non_int." + str(irrep) 
    f = open(out_file,'w')

    for i in range(len(out_file_list[0])):
        for j in range(len(out_file_list)):
            f.write(str(out_file_list[j][i]) + '\t')
            #print(out_file_list[i][j], sep=",")
        f.write('\n')
        
    f.close() 
    print("file generated = ",out_file)


energy_file = "S2I2_energies"

full_list, irrep_list = irrep_list_maker(energy_file)
#nx,ny,nz = config_maker(4)
print(irrep_list)
atmk = 0.09698
atmpi = 0.06906
xi = 3.444
max_nsq = 20
Lmin = 16
Lmax = 24
Lpoints = 2000 
full_energy_list, irrep_list = irrep_list_maker(energy_file)


irrep = "000_A1m"
#full_nonint_spectrum_maker_final(atmk,atmpi, xi, max_nsq, irrep, Lmin, Lmax, Lpoints, energy_file)
#nonint_spectrum_maker(atmk,atmpi, xi, max_nsq, irrep, Lmin, Lmax, Lpoints, energy_file)
#full_nonint_spectrum_maker(atmk,atmpi, xi, max_nsq, irrep, Lmin, Lmax, Lpoints, energy_file)
for i in range(len(irrep_list)):
    irrep = irrep_list[i]
    full_nonint_spectrum_maker_final(atmk,atmpi, xi, max_nsq, irrep, Lmin, Lmax, Lpoints, energy_file)
    #nonint_spectrum_maker(atmk,atmpi, xi, max_nsq, irrep, Lmin, Lmax, Lpoints, energy_file)

def check_ecm(L, atm1, atm2, xi, framenx, frameny, framenz,
                                 nx1, ny1, nz1,
                                 nx2, ny2, nz2,
                                 nx3, ny3, nz3  ):

    particle1_energy = energy(atm1,xi,L,nx1,ny1,nz1)
    particle2_energy = energy(atm1,xi,L,nx2,ny2,nz2)
                #particle3_energy = energy(atmpi,xi,Lbyas,nx[k],ny[k],nz[k])
    particle3_energy = energy(atm2,xi,L,nx3,ny3,nz3)


    frame_nsquare = framenx*framenx + frameny*frameny + framenz*framenz   
    onebyxi = 1.0/xi 
    pi = np.pi
    twopibyLbyas = 2.0*pi/L 

    frame_momsq = onebyxi*onebyxi*twopibyLbyas*twopibyLbyas*frame_nsquare
                


    elat = particle1_energy + particle2_energy + particle3_energy
    ecm = np.sqrt(elat*elat - frame_momsq)

    print(ecm)

check_ecm(20, atmk, atmpi, xi,  2, 0, 0,
                                0, 0, 0,
                                0, 0, 0,
                                2, 0, 0 )

check_ecm(20, atmk, atmpi, xi,  2, 0, 0,
                                1, 0, 1,
                                1, 0, -1,
                                0, 0, 0 )