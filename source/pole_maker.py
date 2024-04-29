# F3inv pole maker and fitter

import numpy as np 
import matplotlib as mpl 
import matplotlib.pyplot as plt 
from scipy.optimize import curve_fit
import scipy.interpolate
from mpl_toolkits.axes_grid1 import make_axes_locatable
from PyPDF2 import PdfMerger

import pandas as pd

import os 
import subprocess 

def E_to_Ecm(En, P):
    return np.sqrt(En**2 - P**2)

def Esq_to_Ecmsq(En, P):
    return En**2 - P**2

def Ecmsq_to_Esq(Ecm, P):
    return Ecm**2 + P**2

def Run_polefinder( nPx, nPy, nPz, P, En_guess1, En_guess2, max_iteration):
    F3inv_pole = subprocess.check_output(['./generate_pole',str(nPx),str(nPy),str(nPz),str(En_guess1),str(En_guess2),str(max_iteration)],shell=False)
    result_pole = F3inv_pole.decode('utf-8')
    finresult_pole = float(result_pole)
    pole_energy_lab = finresult_pole 
    pole_Ecm = np.sqrt(Esq_to_Ecmsq(pole_energy_lab,P))

    return pole_energy_lab, pole_Ecm 

def pole_P000():

    atmpi = 0.06906
    atmK = 0.09698

    KKpi_threshold = atmK + atmK + atmpi 

    KKKKbar_threshold = 4.0*atmK 

    xi = 3.444
    pi = np.arccos(-1.0)
    L = 20
    twopibyxiL = 2.0*pi/(xi*L)
    nPx = 0
    nPy = 0
    nPz = 0

    Px = nPx*twopibyxiL
    Py = nPy*twopibyxiL
    Pz = nPz*twopibyxiL

    P = np.sqrt(Px*Px + Py*Py + Pz*Pz)

    statecounter=0

    pole_vec = []

    En_guess1 = np.sqrt(Ecmsq_to_Esq(KKpi_threshold,P)) + 0.001
    En_guess2 = En_guess1 + 0.002

    En_cutoff = np.sqrt(Ecmsq_to_Esq(KKKKbar_threshold,P)) 

    delta_En = 0.01

    max_iteration = 500

    total_state = 3 # this is the number of poles it will be searching
                    # this is found from looking at the plot of K3iso or -F3inv_iso 

    infval = True 
    while(infval):
        

        pole_energy_lab, pole_Ecm = Run_polefinder( nPx, nPy, nPz, P, En_guess1, En_guess2, max_iteration)
        
        print("En1 = ",En_guess1," En2 = ",En_guess2," pole = ",pole_energy_lab," Ecm = ",pole_Ecm)
        same_pole_check = 0
        for i in range(len(pole_vec)):
            if(pole_Ecm==pole_vec[i]):
                same_pole_check = 1
                break 

        if(pole_energy_lab!=0.0):
            if(same_pole_check==0):
                pole_vec.append(pole_Ecm)
                En_guess1 = pole_energy_lab + delta_En/10 
                En_guess2 = pole_energy_lab + delta_En*2/10 
            else:
                En_guess1 = En_guess1 + delta_En
                En_guess2 = En_guess2 + delta_En
                continue  
                #pole_energy_lab, pole_Ecm = Run_polefinder( nPx, nPy, nPz, P, En_guess1, En_guess2, max_iteration)
        else: 
            En_guess1 = En_guess1 + delta_En
            En_guess2 = En_guess2 + delta_En
            continue  

        if(len(pole_vec)==total_state):
            print("found equal number of poles as of free states")
            break 
        if(En_guess2>En_cutoff):
            print("increase energy cutoff")   
            break 
        En_guess1 = En_guess1 + delta_En
        En_guess2 = En_guess2 + delta_En
    
    pole_file = "F3inv_poles_P" + str(nPx) + str(nPy) + str(nPz) + ".dat"
    fout = open(pole_file,"w")

    for i in range(len(pole_vec)):
        output = str(L) + '\t' + str(pole_vec[i]) + '\n' 
        fout.write(output)
    
    fout.close()


def P100():
    drive = "/home/digonto/Codes/Practical_Lattice_v2/KKpi_interacting_spectrum/Lattice_data/KKpi_L20/"
    filename1 = drive + "mass_100_A2_t012_MassJackFiles_mass_t0_12_reorder_state0.jack"
    filename2 = drive + "mass_100_A2_t012_MassJackFiles_mass_t0_12_reorder_state1.jack"
    filename3 = drive + "mass_100_A2_t012_MassJackFiles_mass_t0_12_reorder_state2.jack"
    filename4 = drive + "mass_100_A2_t012_MassJackFiles_mass_t0_12_reorder_state3.jack"
    filename5 = drive + "mass_100_A2_t012_MassJackFiles_mass_t0_12_reorder_state4.jack"
    filename6 = drive + "mass_100_A2_t012_MassJackFiles_mass_t0_12_reorder_state5.jack"

    filelist = [filename1, filename2, filename3, filename4, filename5, filename6]

    
    xi = 3.444
    pi = np.arccos(-1.0)
    L = 20
    twopibyxiL = 2.0*pi/(xi*L)
    nPx = 0
    nPy = 0
    nPz = 1

    Px = nPx*twopibyxiL
    Py = nPy*twopibyxiL
    Pz = nPz*twopibyxiL

    P = np.sqrt(Px*Px + Py*Py + Pz*Pz)

    statecounter=0

    for files in filelist:
        zerocol, energylab = np.genfromtxt(files,skip_header=1,unpack=True)

        outputfile = "K3iso_P" + str(nPx) + str(nPy) + str(nPz) + "_state" + str(statecounter) + ".dat"

        fout = open(outputfile,"w")



        for i in range(len(energylab)):
            K3iso = subprocess.check_output(['./generate_K3iso',str(nPx),str(nPy),str(nPz),str(energylab[i])],shell=False)
            result = K3iso.decode('utf-8')
            finresult = float(result)
            Ecm = E_to_Ecm(energylab[i],P)
            print(nPx,nPy,nPz,statecounter, energylab[i],E_to_Ecm(energylab[i],P),float(result))
            output = str(energylab[i]) + '\t' + str(Ecm) + '\t' + str(finresult) + '\n' 
            #output = output.rstrip('\n')
            fout.write(output)
    
        fout.close()
        statecounter = statecounter + 1

def P110():
    drive = "/home/digonto/Codes/Practical_Lattice_v2/KKpi_interacting_spectrum/Lattice_data/KKpi_L20/"
    filename1 = drive + "mass_110_A2_t013_MassJackFiles_mass_t0_13_reorder_state0.jack"
    filename2 = drive + "mass_110_A2_t013_MassJackFiles_mass_t0_13_reorder_state1.jack"
    filename3 = drive + "mass_110_A2_t013_MassJackFiles_mass_t0_13_reorder_state2.jack"
    filename4 = drive + "mass_110_A2_t013_MassJackFiles_mass_t0_13_reorder_state3.jack"
    filename5 = drive + "mass_110_A2_t013_MassJackFiles_mass_t0_13_reorder_state4.jack"
    filename6 = drive + "mass_110_A2_t013_MassJackFiles_mass_t0_13_reorder_state5.jack"

    filelist = [filename1, filename2, filename3, filename4, filename5, filename6]

    
    xi = 3.444
    pi = np.arccos(-1.0)
    L = 20
    twopibyxiL = 2.0*pi/(xi*L)
    nPx = 0
    nPy = 1
    nPz = 1

    Px = nPx*twopibyxiL
    Py = nPy*twopibyxiL
    Pz = nPz*twopibyxiL

    P = np.sqrt(Px*Px + Py*Py + Pz*Pz)

    statecounter=0

    for files in filelist:
        zerocol, energylab = np.genfromtxt(files,skip_header=1,unpack=True)

        outputfile = "K3iso_P" + str(nPx) + str(nPy) + str(nPz) + "_state" + str(statecounter) + ".dat"

        fout = open(outputfile,"w")



        for i in range(len(energylab)):
            K3iso = subprocess.check_output(['./generate_K3iso',str(nPx),str(nPy),str(nPz),str(energylab[i])],shell=False)
            result = K3iso.decode('utf-8')
            finresult = float(result)
            Ecm = E_to_Ecm(energylab[i],P)
            print(nPx,nPy,nPz,statecounter, energylab[i],E_to_Ecm(energylab[i],P),float(result))
            output = str(energylab[i]) + '\t' + str(Ecm) + '\t' + str(finresult) + '\n' 
            #output = output.rstrip('\n')
            fout.write(output)
    
        fout.close()
        statecounter = statecounter + 1

def P111():
    drive = "/home/digonto/Codes/Practical_Lattice_v2/KKpi_interacting_spectrum/Lattice_data/KKpi_L20/"
    filename1 = drive + "mass_111_A2_t012_MassJackFiles_mass_t0_12_reorder_state0.jack"
    filename2 = drive + "mass_111_A2_t012_MassJackFiles_mass_t0_12_reorder_state1.jack"
    filename3 = drive + "mass_111_A2_t012_MassJackFiles_mass_t0_12_reorder_state2.jack"
    filename4 = drive + "mass_111_A2_t012_MassJackFiles_mass_t0_12_reorder_state3.jack"
    filename5 = drive + "mass_111_A2_t012_MassJackFiles_mass_t0_12_reorder_state4.jack"
    filename6 = drive + "mass_111_A2_t012_MassJackFiles_mass_t0_12_reorder_state5.jack"

    filelist = [filename1, filename2, filename3, filename4, filename5, filename6]

    
    xi = 3.444
    pi = np.arccos(-1.0)
    L = 20
    twopibyxiL = 2.0*pi/(xi*L)
    nPx = 1
    nPy = 1
    nPz = 1

    Px = nPx*twopibyxiL
    Py = nPy*twopibyxiL
    Pz = nPz*twopibyxiL

    P = np.sqrt(Px*Px + Py*Py + Pz*Pz)

    statecounter=0

    for files in filelist:
        zerocol, energylab = np.genfromtxt(files,skip_header=1,unpack=True)

        outputfile = "K3iso_P" + str(nPx) + str(nPy) + str(nPz) + "_state" + str(statecounter) + ".dat"

        fout = open(outputfile,"w")



        for i in range(len(energylab)):
            Ecmsq = Esq_to_Ecmsq(energylab[i],P)
            
            if(Ecmsq<0.26**2 or Ecmsq>0.43**2):
                continue 
            else:
                Ecm = np.sqrt(Ecmsq)
                K3iso = subprocess.check_output(['./generate_K3iso',str(nPx),str(nPy),str(nPz),str(energylab[i])],shell=False)
            result = K3iso.decode('utf-8')
            finresult = float(result)
            
            print(nPx,nPy,nPz,statecounter, energylab[i],E_to_Ecm(energylab[i],P),float(result))
            output = str(energylab[i]) + '\t' + str(Ecm) + '\t' + str(finresult) + '\n' 
            #output = output.rstrip('\n')
            fout.write(output)

        fout.close()
        statecounter = statecounter + 1

def P200():
    drive = "/home/digonto/Codes/Practical_Lattice_v2/KKpi_interacting_spectrum/Lattice_data/KKpi_L20/"
    filename1 = drive + "mass_200_A2_t010_MassJackFiles_mass_t0_10_reorder_state0.jack"
    filename2 = drive + "mass_200_A2_t010_MassJackFiles_mass_t0_10_reorder_state1.jack"
    filename3 = drive + "mass_200_A2_t010_MassJackFiles_mass_t0_10_reorder_state2.jack"
    filename4 = drive + "mass_200_A2_t010_MassJackFiles_mass_t0_10_reorder_state3.jack"
    filename5 = drive + "mass_200_A2_t010_MassJackFiles_mass_t0_10_reorder_state4.jack"
    filename6 = drive + "mass_200_A2_t010_MassJackFiles_mass_t0_10_reorder_state5.jack"
    filename7 = drive + "mass_200_A2_t010_MassJackFiles_mass_t0_10_reorder_state6.jack"
    filename8 = drive + "mass_200_A2_t010_MassJackFiles_mass_t0_10_reorder_state7.jack"

    filelist = [filename1, filename2, filename3, filename4, filename5, filename6, filename7, filename8]

    
    xi = 3.444
    pi = np.arccos(-1.0)
    L = 20
    twopibyxiL = 2.0*pi/(xi*L)
    nPx = 0
    nPy = 0
    nPz = 2

    Px = nPx*twopibyxiL
    Py = nPy*twopibyxiL
    Pz = nPz*twopibyxiL

    P = np.sqrt(Px*Px + Py*Py + Pz*Pz)

    statecounter=0

    for files in filelist:
        zerocol, energylab = np.genfromtxt(files,skip_header=1,unpack=True)

        outputfile = "K3iso_P" + str(nPx) + str(nPy) + str(nPz) + "_state" + str(statecounter) + ".dat"

        fout = open(outputfile,'w')

        

        for i in range(len(energylab)):
            Ecmsq = Esq_to_Ecmsq(energylab[i],P)
            
            if(Ecmsq<0.26**2 or Ecmsq>0.43**2):
                continue 
            else:
                Ecm = np.sqrt(Ecmsq)
                K3iso = subprocess.check_output(['./generate_K3iso',str(nPx),str(nPy),str(nPz),str(energylab[i])],shell=False)
            result = K3iso.decode('utf-8')
            finresult = float(result)
            #Ecm = E_to_Ecm(energylab[i],P)
            print(nPx,nPy,nPz,statecounter, energylab[i],E_to_Ecm(energylab[i],P),float(result))
            output = str(energylab[i]) + '\t' + str(Ecm) + '\t' + str(finresult) + '\n' 
            #output = output.rstrip('\n')
            fout.write(output)
    
        fout.close()
        statecounter = statecounter + 1

#this one calculates the K3iso using the averaged data file
def jackknifeavg_lattice_data():
    #this drive path has been set from macbook
    drive = "/Users/digonto/GitHub/3body_quantization/lattice_data/KKpi_interacting_spectrum/Three_body/"
    #this drive path is for ubuntu
    drive = "/home/digonto/Codes/Practical_Lattice_v2/3body_quantization/lattice_data/KKpi_interacting_spectrum/Three_body/"
    
    filename1 = drive + "KKpi_spectrum.000_A1m"
    filename2 = drive + "KKpi_spectrum.100_A2"
    filename3 = drive + "KKpi_spectrum.110_A2"
    filename4 = drive + "KKpi_spectrum.111_A2"
    filename5 = drive + "KKpi_spectrum.200_A2"

    filelist = [filename1, filename2, filename3, filename4, filename5]

    
    xi = 3.444
    pi = np.arccos(-1.0)
    L = 20
    twopibyxiL = 2.0*pi/(xi*L)
    


    nconfig = [[0,0,0],[0,0,1],[0,1,1],[1,1,1],[0,0,2]]

    
    statecounter=0
    counter = 0
    
    for files in filelist:
        Ls, Ecm, err1, err2 = np.genfromtxt(files,unpack=True)

        nPx = nconfig[counter][0]
        nPy = nconfig[counter][1]
        nPz = nconfig[counter][2]

        Px = nPx*twopibyxiL
        Py = nPy*twopibyxiL
        Pz = nPz*twopibyxiL

        P = np.sqrt(Px*Px + Py*Py + Pz*Pz)


        

        for i in range(len(Ecm)):
            Ecm_val_ini = Ecm[i] - err2[i]
            Ecm_val_fin = Ecm[i] + err2[i]
            Elab_ini = np.sqrt(Ecmsq_to_Esq(Ecm_val_ini,P))
            Elab_fin = np.sqrt(Ecmsq_to_Esq(Ecm_val_fin,P))
            Elab = np.linspace(Elab_ini, Elab_fin, 200)

            outputfile = "K3iso_jackavg_P" + str(nPx) + str(nPy) + str(nPz) + "_state_" + str(i) + ".dat"

            fout = open(outputfile,"w")

            for j in range(len(Elab)):
                Elab_val = Elab[j]
                K3iso = subprocess.check_output(['./generate_K3iso',str(nPx),str(nPy),str(nPz),str(Elab_val)],shell=False)
                result = K3iso.decode('utf-8')
                finresult = float(result)
                calcEcmsq = Esq_to_Ecmsq(Elab_val,P)
                calcEcm = np.sqrt(calcEcmsq)
                print(nPx,nPy,nPz,i, Elab_val,calcEcm,float(result))
                output = str(Elab_val) + '\t' + str(calcEcm) + '\t' + str(finresult) + '\n' 
                #output = output.rstrip('\n')
                fout.write(output)
    
            fout.close()
        statecounter = statecounter + 1
        counter = counter + 1 

def jackknifeavg_centralvalue_lattice_data():
    #this drive path has been set from macbook
    drive = "/Users/digonto/GitHub/3body_quantization/lattice_data/KKpi_interacting_spectrum/Three_body/"
    #this drive path is for ubuntu
    drive = "/home/digonto/Codes/Practical_Lattice_v2/3body_quantization/lattice_data/KKpi_interacting_spectrum/Three_body/"
    
    filename1 = drive + "KKpi_spectrum.000_A1m"
    filename2 = drive + "KKpi_spectrum.100_A2"
    filename3 = drive + "KKpi_spectrum.110_A2"
    filename4 = drive + "KKpi_spectrum.111_A2"
    filename5 = drive + "KKpi_spectrum.200_A2"

    filelist = [filename1, filename2, filename3, filename4, filename5]

    
    xi = 3.444
    pi = np.arccos(-1.0)
    L = 20
    twopibyxiL = 2.0*pi/(xi*L)
    


    nconfig = [[0,0,0],[0,0,1],[0,1,1],[1,1,1],[0,0,2]]

    
    statecounter=0
    counter = 0
    
    for files in filelist:
        Ls, Ecm, err1, err2 = np.genfromtxt(files,unpack=True)

        nPx = nconfig[counter][0]
        nPy = nconfig[counter][1]
        nPz = nconfig[counter][2]

        Px = nPx*twopibyxiL
        Py = nPy*twopibyxiL
        Pz = nPz*twopibyxiL

        P = np.sqrt(Px*Px + Py*Py + Pz*Pz)

        outputfile = "K3iso_jackavg_centralval_P" + str(nPx) + str(nPy) + str(nPz) + ".dat"

        fout = open(outputfile,"w")

        

        for i in range(len(Ecm)):
            
            Ecm_val = Ecm[i]
            Elab_val = np.sqrt(Ecmsq_to_Esq(Ecm_val,P))
            K3iso = subprocess.check_output(['./generate_K3iso',str(nPx),str(nPy),str(nPz),str(Elab_val)],shell=False)
            result = K3iso.decode('utf-8')
            finresult = float(result)
            calcEcmsq = Esq_to_Ecmsq(Elab_val,P)
            calcEcm = np.sqrt(calcEcmsq)
            print(nPx,nPy,nPz,i, Elab_val,calcEcm,float(result))
            output = str(Elab_val) + '\t' + str(calcEcm) + '\t' + str(finresult) + '\n' 
            #output = output.rstrip('\n')
            fout.write(output)                  
            
            
    
        fout.close()
        statecounter = statecounter + 1
        counter = counter + 1 


#P000()
#P100()
#P110()
#P111()
#P200()

#jackknifeavg_lattice_data()
#jackknifeavg_centralvalue_lattice_data()

#pole_P000()

def pole_finding_by_reading_data_file_F3inv_P000():
    atmpi = 0.06906
    atmK = 0.09698

    KKpi_threshold = atmK + atmK + atmpi 

    KKKKbar_threshold = 4.0*atmK 

    xi = 3.444
    pi = np.arccos(-1.0)
    L = 20
    twopibyxiL = 2.0*pi/(xi*L)
    nPx = 0
    nPy = 0
    nPz = 0

    Px = nPx*twopibyxiL
    Py = nPy*twopibyxiL
    Pz = nPz*twopibyxiL

    P = np.sqrt(Px*Px + Py*Py + Pz*Pz)

    statecounter=0

    pole_vec = []

    total_state = 3

    filename1 = "F3_for_pole_KKpi_L20_nP_000.dat"

    (En, EcmR, ReF3det, ImF3det, 
     ReF3sum, ImF3sum, RedetInvF3, ImdetInvF3,
     ResumInvF3, ImsumInvF3) = np.genfromtxt(filename1, unpack=True)
    
    pole_vec = []
    gap_vec = []
    
    for i in range(len(EcmR)-1):
        Ecm_val1 = EcmR[i]
        E_val1 = En[i]
        E_val2 = En[i+1]
        Ecm_val2 = EcmR[i+1]
        avg_Ecm = (Ecm_val1 + Ecm_val2)/2.0

        F3inv_val1 = ResumInvF3[i]
        F3inv_val2 = ResumInvF3[i+1]

        #print(i,Ecm_val1,F3inv_val1)
        if(F3inv_val1<0.0 and F3inv_val2>0.0):
            print("initial pole found at = ",avg_Ecm," gap = ",abs(F3inv_val1 - F3inv_val2))
            pole_vec.append(avg_Ecm)
            gap_vec.append(abs(F3inv_val1 - F3inv_val2))
    
    df = pd.DataFrame(list(zip(pole_vec, gap_vec)))

    #print(df)

    sorted_df = df.sort_values(1)

    f2 = sorted_df.reset_index(drop=True)
    print("sorted frame of poles")
    print(f2)

    outfile = "Kdf0_spectrum_nP_000.dat"
    f = open(outfile,'w')

    for i in range(total_state):
        actual_pole = f2[0][i]
        print("poles = ",f2[0][i])
        f.write("20" + '\t' + str(actual_pole) + '\n')
    
    f.close() 

def pole_finding_by_reading_data_file_F3inv_P100():
    atmpi = 0.06906
    atmK = 0.09698

    KKpi_threshold = atmK + atmK + atmpi 

    KKKKbar_threshold = 4.0*atmK 

    xi = 3.444
    pi = np.arccos(-1.0)
    L = 20
    twopibyxiL = 2.0*pi/(xi*L)
    nPx = 0
    nPy = 0
    nPz = 0

    Px = nPx*twopibyxiL
    Py = nPy*twopibyxiL
    Pz = nPz*twopibyxiL

    P = np.sqrt(Px*Px + Py*Py + Pz*Pz)

    statecounter=0

    pole_vec = []

    total_state = 6

    filename1 = "F3_for_pole_KKpi_L20_nP_100.dat"

    (En, EcmR, ReF3det, ImF3det, 
     ReF3sum, ImF3sum, RedetInvF3, ImdetInvF3,
     ResumInvF3, ImsumInvF3) = np.genfromtxt(filename1, unpack=True)
    
    pole_vec = []
    gap_vec = []
    
    for i in range(len(EcmR)-1):
        Ecm_val1 = EcmR[i]
        E_val1 = En[i]
        E_val2 = En[i+1]
        Ecm_val2 = EcmR[i+1]
        avg_Ecm = (Ecm_val1 + Ecm_val2)/2.0

        F3inv_val1 = ResumInvF3[i]
        F3inv_val2 = ResumInvF3[i+1]

        #print(i,Ecm_val1,F3inv_val1)
        if(F3inv_val1<0.0 and F3inv_val2>0.0):
            print("initial pole found at = ",avg_Ecm," gap = ",abs(F3inv_val1 - F3inv_val2))
            pole_vec.append(avg_Ecm)
            gap_vec.append(abs(F3inv_val1 - F3inv_val2))
    
    df = pd.DataFrame(list(zip(pole_vec, gap_vec)))

    #print(df)

    sorted_df = df.sort_values(1)

    f2 = sorted_df.reset_index(drop=True)
    print("sorted frame of poles")
    print(f2)

    outfile = "Kdf0_spectrum_nP_100.dat"
    f = open(outfile,'w')

    for i in range(total_state):
        actual_pole = f2[0][i]
        print("poles = ",f2[0][i])
        f.write("20" + '\t' + str(actual_pole) + '\n')
    
    f.close() 

def pole_finding_by_reading_data_file_F3inv_P110():
    atmpi = 0.06906
    atmK = 0.09698

    KKpi_threshold = atmK + atmK + atmpi 

    KKKKbar_threshold = 4.0*atmK 

    xi = 3.444
    pi = np.arccos(-1.0)
    L = 20
    twopibyxiL = 2.0*pi/(xi*L)
    nPx = 0
    nPy = 0
    nPz = 0

    Px = nPx*twopibyxiL
    Py = nPy*twopibyxiL
    Pz = nPz*twopibyxiL

    P = np.sqrt(Px*Px + Py*Py + Pz*Pz)

    statecounter=0

    pole_vec = []

    total_state = 6

    filename1 = "F3_for_pole_KKpi_L20_nP_110.dat"

    (En, EcmR, ReF3det, ImF3det, 
     ReF3sum, ImF3sum, RedetInvF3, ImdetInvF3,
     ResumInvF3, ImsumInvF3) = np.genfromtxt(filename1, unpack=True)
    
    pole_vec = []
    gap_vec = []
    
    for i in range(len(EcmR)-1):
        Ecm_val1 = EcmR[i]
        E_val1 = En[i]
        E_val2 = En[i+1]
        Ecm_val2 = EcmR[i+1]
        avg_Ecm = (Ecm_val1 + Ecm_val2)/2.0

        F3inv_val1 = ResumInvF3[i]
        F3inv_val2 = ResumInvF3[i+1]

        #print(i,Ecm_val1,F3inv_val1)
        if(F3inv_val1<0.0 and F3inv_val2>0.0):
            print("initial pole found at = ",avg_Ecm," gap = ",abs(F3inv_val1 - F3inv_val2))
            pole_vec.append(avg_Ecm)
            gap_vec.append(abs(F3inv_val1 - F3inv_val2))
    
    df = pd.DataFrame(list(zip(pole_vec, gap_vec)))

    #print(df)

    sorted_df = df.sort_values(1)

    f2 = sorted_df.reset_index(drop=True)
    print("sorted frame of poles")
    print(f2)

    outfile = "Kdf0_spectrum_nP_110.dat"
    f = open(outfile,'w')

    for i in range(total_state):
        actual_pole = f2[0][i]
        print("poles = ",f2[0][i])
        f.write("20" + '\t' + str(actual_pole) + '\n')
    
    f.close() 

def pole_finding_by_reading_data_file_F3inv_P111():
    atmpi = 0.06906
    atmK = 0.09698

    KKpi_threshold = atmK + atmK + atmpi 

    KKKKbar_threshold = 4.0*atmK 

    xi = 3.444
    pi = np.arccos(-1.0)
    L = 20
    twopibyxiL = 2.0*pi/(xi*L)
    nPx = 0
    nPy = 0
    nPz = 0

    Px = nPx*twopibyxiL
    Py = nPy*twopibyxiL
    Pz = nPz*twopibyxiL

    P = np.sqrt(Px*Px + Py*Py + Pz*Pz)

    statecounter=0

    pole_vec = []

    total_state = 6

    filename1 = "F3_for_pole_KKpi_L20_nP_111.dat"

    (En, EcmR, ReF3det, ImF3det, 
     ReF3sum, ImF3sum, RedetInvF3, ImdetInvF3,
     ResumInvF3, ImsumInvF3) = np.genfromtxt(filename1, unpack=True)
    
    pole_vec = []
    gap_vec = []
    
    for i in range(len(EcmR)-1):
        Ecm_val1 = EcmR[i]
        E_val1 = En[i]
        E_val2 = En[i+1]
        Ecm_val2 = EcmR[i+1]
        avg_Ecm = (Ecm_val1 + Ecm_val2)/2.0

        F3inv_val1 = ResumInvF3[i]
        F3inv_val2 = ResumInvF3[i+1]

        #print(i,Ecm_val1,F3inv_val1)
        if(F3inv_val1<0.0 and F3inv_val2>0.0):
            print("initial pole found at = ",avg_Ecm," gap = ",abs(F3inv_val1 - F3inv_val2))
            pole_vec.append(avg_Ecm)
            gap_vec.append(abs(F3inv_val1 - F3inv_val2))
    
    df = pd.DataFrame(list(zip(pole_vec, gap_vec)))

    #print(df)

    sorted_df = df.sort_values(1)

    f2 = sorted_df.reset_index(drop=True)
    print("sorted frame of poles")
    print(f2)

    outfile = "Kdf0_spectrum_nP_111.dat"
    f = open(outfile,'w')

    for i in range(total_state):
        actual_pole = f2[0][i]
        print("poles = ",f2[0][i])
        f.write("20" + '\t' + str(actual_pole) + '\n')
    
    f.close() 

def pole_finding_by_reading_data_file_F3inv_P200():
    atmpi = 0.06906
    atmK = 0.09698

    KKpi_threshold = atmK + atmK + atmpi 

    KKKKbar_threshold = 4.0*atmK 

    xi = 3.444
    pi = np.arccos(-1.0)
    L = 20
    twopibyxiL = 2.0*pi/(xi*L)
    nPx = 0
    nPy = 0
    nPz = 0

    Px = nPx*twopibyxiL
    Py = nPy*twopibyxiL
    Pz = nPz*twopibyxiL

    P = np.sqrt(Px*Px + Py*Py + Pz*Pz)

    statecounter=0

    pole_vec = []

    total_state = 7

    filename1 = "F3_for_pole_KKpi_L20_nP_200.dat"

    (En, EcmR, ReF3det, ImF3det, 
     ReF3sum, ImF3sum, RedetInvF3, ImdetInvF3,
     ResumInvF3, ImsumInvF3) = np.genfromtxt(filename1, unpack=True)
    
    pole_vec = []
    gap_vec = []
    
    for i in range(len(EcmR)-1):
        Ecm_val1 = EcmR[i]
        E_val1 = En[i]
        E_val2 = En[i+1]
        Ecm_val2 = EcmR[i+1]
        avg_Ecm = (Ecm_val1 + Ecm_val2)/2.0

        F3inv_val1 = ResumInvF3[i]
        F3inv_val2 = ResumInvF3[i+1]

        #print(i,Ecm_val1,F3inv_val1)
        if(F3inv_val1<0.0 and F3inv_val2>0.0):
            print("initial pole found at = ",avg_Ecm," gap = ",abs(F3inv_val1 - F3inv_val2))
            pole_vec.append(avg_Ecm)
            gap_vec.append(abs(F3inv_val1 - F3inv_val2))
    
    df = pd.DataFrame(list(zip(pole_vec, gap_vec)))

    #print(df)

    sorted_df = df.sort_values(1)

    f2 = sorted_df.reset_index(drop=True)
    print("sorted frame of poles")
    print(f2)

    outfile = "Kdf0_spectrum_nP_200.dat"
    f = open(outfile,'w')

    for i in range(total_state):
        actual_pole = f2[0][i]
        print("poles = ",f2[0][i])
        f.write("20" + '\t' + str(actual_pole) + '\n')
    
    f.close() 

def pole_finding_by_reading_data_file_F3_P000():
    atmpi = 0.06906
    atmK = 0.09698

    KKpi_threshold = atmK + atmK + atmpi 

    KKKKbar_threshold = 4.0*atmK 

    xi = 3.444
    pi = np.arccos(-1.0)
    L = 20
    twopibyxiL = 2.0*pi/(xi*L)
    nPx = 0
    nPy = 0
    nPz = 0

    Px = nPx*twopibyxiL
    Py = nPy*twopibyxiL
    Pz = nPz*twopibyxiL

    P = np.sqrt(Px*Px + Py*Py + Pz*Pz)

    statecounter=0

    pole_vec = []

    total_state = 3

    filename1 = "F3_for_pole_KKpi_L20_nP_000_1.dat"

    (En, EcmR, ReF3det, ImF3det, 
     ReF3sum, ImF3sum, RedetInvF3, ImdetInvF3,
     ResumInvF3, ImsumInvF3) = np.genfromtxt(filename1, unpack=True)
    
    pole_vec = []
    gap_vec = []
    
    for i in range(len(EcmR)-1):
        Ecm_val1 = EcmR[i]
        E_val1 = En[i]
        E_val2 = En[i+1]
        Ecm_val2 = EcmR[i+1]
        avg_Ecm = (Ecm_val1 + Ecm_val2)/2.0

        F3_val1 = ReF3sum[i]
        F3_val2 = ReF3sum[i+1]

        #print(i,Ecm_val1,F3inv_val1)
        if(F3_val1<0.0 and F3_val2>0.0):
            print("initial pole found at = ",avg_Ecm," gap = ",abs(F3_val1 - F3_val2))
            pole_vec.append(avg_Ecm)
            gap_vec.append(abs(F3_val1 - F3_val2))
    
    #df = pd.DataFrame(list(zip(pole_vec, gap_vec)))

    #print(df)

    #sorted_df = df.sort_values(1, ascending=False)

    #f2 = sorted_df.reset_index(drop=True)
    #print("sorted frame of poles")
    #print(f2)

    outfile = "Kdf0_spectrum_nP_000_L20.dat"
    f = open(outfile,'w')

    #we are adding all the poles now
    #we will correct the spectrum by looking
    #at the F3iso data
    for i in range(len(pole_vec)):
        actual_pole = pole_vec[i]#f2[0][i]
        print("poles = ",pole_vec[i])#f2[0][i])
        f.write("20" + '\t' + str(actual_pole) + '\n')
    
    f.close() 


def pole_finding_by_reading_data_file_F3_P100():
    atmpi = 0.06906
    atmK = 0.09698

    KKpi_threshold = atmK + atmK + atmpi 

    KKKKbar_threshold = 4.0*atmK 

    xi = 3.444
    pi = np.arccos(-1.0)
    L = 20
    twopibyxiL = 2.0*pi/(xi*L)
    nPx = 0
    nPy = 0
    nPz = 0

    Px = nPx*twopibyxiL
    Py = nPy*twopibyxiL
    Pz = nPz*twopibyxiL

    P = np.sqrt(Px*Px + Py*Py + Pz*Pz)

    statecounter=0

    pole_vec = []

    total_state = 6

    filename1 = "F3_for_pole_KKpi_L20_nP_100_1.dat"

    (En, EcmR, ReF3det, ImF3det, 
     ReF3sum, ImF3sum, RedetInvF3, ImdetInvF3,
     ResumInvF3, ImsumInvF3) = np.genfromtxt(filename1, unpack=True)
    
    pole_vec = []
    gap_vec = []
    
    for i in range(len(EcmR)-1):
        Ecm_val1 = EcmR[i]
        E_val1 = En[i]
        E_val2 = En[i+1]
        Ecm_val2 = EcmR[i+1]
        avg_Ecm = (Ecm_val1 + Ecm_val2)/2.0

        F3_val1 = ReF3sum[i]
        F3_val2 = ReF3sum[i+1]

        #print(i,Ecm_val1,F3inv_val1)
        if(F3_val1<0.0 and F3_val2>0.0):
            print("initial pole found at = ",avg_Ecm," gap = ",abs(F3_val1 - F3_val2))
            pole_vec.append(avg_Ecm)
            gap_vec.append(abs(F3_val1 - F3_val2))
    
    #df = pd.DataFrame(list(zip(pole_vec, gap_vec)))

    #print(df)

    #sorted_df = df.sort_values(1, ascending=False)

    #f2 = sorted_df.reset_index(drop=True)
    #print("sorted frame of poles")
    #print(f2)

    outfile = "Kdf0_spectrum_nP_100_L20.dat"
    f = open(outfile,'w')

    #we are adding all the poles now
    #we will correct the spectrum by looking
    #at the F3iso data
    for i in range(len(pole_vec)):
        actual_pole = pole_vec[i]#f2[0][i]
        print("poles = ",pole_vec[i])#f2[0][i])
        f.write("20" + '\t' + str(actual_pole) + '\n')
    
    f.close() 

def pole_finding_by_reading_data_file_F3_P110():
    atmpi = 0.06906
    atmK = 0.09698

    KKpi_threshold = atmK + atmK + atmpi 

    KKKKbar_threshold = 4.0*atmK 

    xi = 3.444
    pi = np.arccos(-1.0)
    L = 20
    twopibyxiL = 2.0*pi/(xi*L)
    nPx = 0
    nPy = 0
    nPz = 0

    Px = nPx*twopibyxiL
    Py = nPy*twopibyxiL
    Pz = nPz*twopibyxiL

    P = np.sqrt(Px*Px + Py*Py + Pz*Pz)

    statecounter=0

    pole_vec = []

    total_state = 6

    filename1 = "F3_for_pole_KKpi_L20_nP_110_1.dat"

    (En, EcmR, ReF3det, ImF3det, 
     ReF3sum, ImF3sum, RedetInvF3, ImdetInvF3,
     ResumInvF3, ImsumInvF3) = np.genfromtxt(filename1, unpack=True)
    
    pole_vec = []
    gap_vec = []
    
    for i in range(len(EcmR)-1):
        Ecm_val1 = EcmR[i]
        E_val1 = En[i]
        E_val2 = En[i+1]
        Ecm_val2 = EcmR[i+1]
        avg_Ecm = (Ecm_val1 + Ecm_val2)/2.0

        F3_val1 = ReF3sum[i]
        F3_val2 = ReF3sum[i+1]

        #print(i,Ecm_val1,F3inv_val1)
        if(F3_val1<0.0 and F3_val2>0.0):
            print("initial pole found at = ",avg_Ecm," gap = ",abs(F3_val1 - F3_val2))
            pole_vec.append(avg_Ecm)
            gap_vec.append(abs(F3_val1 - F3_val2))
    
    #df = pd.DataFrame(list(zip(pole_vec, gap_vec)))

    #print(df)

    #sorted_df = df.sort_values(1, ascending=False)

    #f2 = sorted_df.reset_index(drop=True)
    #print("sorted frame of poles")
    #print(f2)

    outfile = "Kdf0_spectrum_nP_110_L20.dat"
    f = open(outfile,'w')

    #we are adding all the poles now
    #we will correct the spectrum by looking
    #at the F3iso data
    for i in range(len(pole_vec)):
        actual_pole = pole_vec[i]#f2[0][i]
        print("poles = ",pole_vec[i])#f2[0][i])
        f.write("20" + '\t' + str(actual_pole) + '\n')
    
    f.close() 

def pole_finding_by_reading_data_file_F3_P111():
    atmpi = 0.06906
    atmK = 0.09698

    KKpi_threshold = atmK + atmK + atmpi 

    KKKKbar_threshold = 4.0*atmK 

    xi = 3.444
    pi = np.arccos(-1.0)
    L = 20
    twopibyxiL = 2.0*pi/(xi*L)
    nPx = 0
    nPy = 0
    nPz = 0

    Px = nPx*twopibyxiL
    Py = nPy*twopibyxiL
    Pz = nPz*twopibyxiL

    P = np.sqrt(Px*Px + Py*Py + Pz*Pz)

    statecounter=0

    pole_vec = []

    total_state = 6

    filename1 = "F3_for_pole_KKpi_L20_nP_111_1.dat"

    (En, EcmR, ReF3det, ImF3det, 
     ReF3sum, ImF3sum, RedetInvF3, ImdetInvF3,
     ResumInvF3, ImsumInvF3) = np.genfromtxt(filename1, unpack=True)
    
    pole_vec = []
    gap_vec = []
    
    for i in range(len(EcmR)-1):
        Ecm_val1 = EcmR[i]
        E_val1 = En[i]
        E_val2 = En[i+1]
        Ecm_val2 = EcmR[i+1]
        avg_Ecm = (Ecm_val1 + Ecm_val2)/2.0

        F3_val1 = ReF3sum[i]
        F3_val2 = ReF3sum[i+1]

        #print(i,Ecm_val1,F3inv_val1)
        if(F3_val1<0.0 and F3_val2>0.0):
            print("initial pole found at = ",avg_Ecm," gap = ",abs(F3_val1 - F3_val2))
            pole_vec.append(avg_Ecm)
            gap_vec.append(abs(F3_val1 - F3_val2))
    
    #df = pd.DataFrame(list(zip(pole_vec, gap_vec)))

    #print(df)

    #sorted_df = df.sort_values(1, ascending=False)

    #f2 = sorted_df.reset_index(drop=True)
    #print("sorted frame of poles")
    #print(f2)

    outfile = "Kdf0_spectrum_nP_111_L20.dat"
    f = open(outfile,'w')

    #we are adding all the poles now
    #we will correct the spectrum by looking
    #at the F3iso data
    for i in range(len(pole_vec)):
        actual_pole = pole_vec[i]#f2[0][i]
        print("poles = ",pole_vec[i])#f2[0][i])
        f.write("20" + '\t' + str(actual_pole) + '\n')
    
    f.close() 

def pole_finding_by_reading_data_file_F3_P200():
    atmpi = 0.06906
    atmK = 0.09698

    KKpi_threshold = atmK + atmK + atmpi 

    KKKKbar_threshold = 4.0*atmK 

    xi = 3.444
    pi = np.arccos(-1.0)
    L = 20
    twopibyxiL = 2.0*pi/(xi*L)
    nPx = 0
    nPy = 0
    nPz = 0

    Px = nPx*twopibyxiL
    Py = nPy*twopibyxiL
    Pz = nPz*twopibyxiL

    P = np.sqrt(Px*Px + Py*Py + Pz*Pz)

    statecounter=0

    pole_vec = []

    total_state = 7

    filename1 = "F3_for_pole_KKpi_L20_nP_200_1.dat"

    (En, EcmR, ReF3det, ImF3det, 
     ReF3sum, ImF3sum, RedetInvF3, ImdetInvF3,
     ResumInvF3, ImsumInvF3) = np.genfromtxt(filename1, unpack=True)
    
    pole_vec = []
    gap_vec = []
    
    for i in range(len(EcmR)-1):
        Ecm_val1 = EcmR[i]
        E_val1 = En[i]
        E_val2 = En[i+1]
        Ecm_val2 = EcmR[i+1]
        avg_Ecm = (Ecm_val1 + Ecm_val2)/2.0

        F3_val1 = ReF3sum[i]
        F3_val2 = ReF3sum[i+1]

        #print(i,Ecm_val1,F3inv_val1)
        if(F3_val1<0.0 and F3_val2>0.0):
            print("initial pole found at = ",avg_Ecm," gap = ",abs(F3_val1 - F3_val2))
            pole_vec.append(avg_Ecm)
            gap_vec.append(abs(F3_val1 - F3_val2))
    
    #df = pd.DataFrame(list(zip(pole_vec, gap_vec)))

    #print(df)

    #sorted_df = df.sort_values(1, ascending=False)

    #f2 = sorted_df.reset_index(drop=True)
    #print("sorted frame of poles")
    #print(f2)

    outfile = "Kdf0_spectrum_nP_200_L20.dat"
    f = open(outfile,'w')

    #we are adding all the poles now
    #we will correct the spectrum by looking
    #at the F3iso data
    for i in range(len(pole_vec)):
        actual_pole = pole_vec[i]#f2[0][i]
        print("poles = ",pole_vec[i])#f2[0][i])
        f.write("20" + '\t' + str(actual_pole) + '\n')
    
    f.close() 

def pole_finding_by_reading_data_file_F3inv_all_boost():
    atmpi = 0.06906
    atmK = 0.09698

    KKpi_threshold = atmK + atmK + atmpi 

    KKKKbar_threshold = 4.0*atmK 

    xi = 3.444
    pi = np.arccos(-1.0)
    L = 20
    twopibyxiL = 2.0*pi/(xi*L)
    nPx = 0
    nPy = 0
    nPz = 0

    Px = nPx*twopibyxiL
    Py = nPy*twopibyxiL
    Pz = nPz*twopibyxiL

    P = np.sqrt(Px*Px + Py*Py + Pz*Pz)

    Pmomlist = ['000','100','110','111','200']

    for moms in Pmomlist:
        statecounter=0

        pole_vec = []

        total_state = 3

        drive = './test_files/F3_for_pole_KKpi_L20/'
        filename1 = drive + "ultraHQ_F3_for_pole_KKpi_L20_nP_" + moms + ".dat"

        (En, EcmR, norm, ReF3sum, F2, G, K2inv, H) = np.genfromtxt(filename1, unpack=True)
    

        ReF3inv = np.zeros((len(ReF3sum)))
        for i in range(len(ReF3sum)):
            ReF3inv[i] = 1.0/ReF3sum[i]

        pole_vec = []
        gap_vec = []
    
        for i in range(len(EcmR)-1):
            Ecm_val1 = EcmR[i]
            E_val1 = En[i]
            E_val2 = En[i+1]
            Ecm_val2 = EcmR[i+1]
            avg_Ecm = (Ecm_val1 + Ecm_val2)/2.0

            F3inv_val1 = ReF3inv[i]
            F3inv_val2 = ReF3inv[i+1]

            #print(i,Ecm_val1,F3inv_val1)
            if(F3inv_val1>0.0 and F3inv_val2<0.0):
                print("initial pole found at = ",avg_Ecm," gap = ",abs(F3inv_val1 - F3inv_val2))
                pole_vec.append(avg_Ecm)
                gap_vec.append(abs(F3inv_val1 - F3inv_val2))
    
        #df = pd.DataFrame(list(zip(pole_vec, gap_vec)))

        #print(df)

        #sorted_df = df.sort_values(1, ascending=False)

        #f2 = sorted_df.reset_index(drop=True)
        #print("sorted frame of poles")
        #print(f2)

        outfile = "F3inv_poles_nP_" + moms + "_L20.dat"
        f = open(outfile,'w')

        #we are adding all the poles now
        #we will correct the spectrum by looking
        #at the F3iso data
        for i in range(len(pole_vec)):
            actual_pole = pole_vec[i]#f2[0][i]
            print("poles = ",pole_vec[i])#f2[0][i])
            f.write("20" + '\t' + str(actual_pole) + '\n')
    
        f.close() 



#pole_finding_by_reading_data_file_F3_P000()
#pole_finding_by_reading_data_file_F3_P100()
#pole_finding_by_reading_data_file_F3_P110()
#pole_finding_by_reading_data_file_F3_P111()
#pole_finding_by_reading_data_file_F3_P200()
pole_finding_by_reading_data_file_F3inv_all_boost()
############################### TEST ############################################

#################################################################################
#file = open("temp.temp",'w')

#for i in range(10):
#    file.write(str(i) + '\n')

#file.close()
#command = ['./generate_K3iso','0','0','0','0.235373']    
#args = [float(arg) if arg.isdigit() else float(arg) for arg in command[1:]]
#checkres = subprocess.check_output(command,shell=False, stderr=subprocess.STDOUT)
#print(checkres.decode('ascii'))
##################################################################################

######################################################################################################################################
#hello = "hiaaa"
#output = subprocess.call(["echo " + str(hello) + " |awk '{print(substr($0,1,3)}'"]) #subprocess.check_output("echo ")

#output = subprocess.check_output(['/usr/bin/echo ', hello])

#output = subprocess.run(["echo",hello,"|awk '{print(substr($0,1,1)}'"],capture_output=True, shell=True)

#output1 = output.decode('ascii')

#print(output)
#print(output.stdout.decode())
#print(float(output1) + 2)
######################################################################################################################################
