# Here we plot the F3 function for each boost frame and compare them with 
# free spectrum generated from threebody_non_int_spectrum() code 


#This plots F3_iso for different total momentum P
#This is for KKpi system

#We also plot the KDF0 spectrum to check which pole are correct
#and make changes to the spectrum accordingly

import numpy as np 
import matplotlib as mpl 
import matplotlib.pyplot as plt 
import scipy.interpolate
from mpl_toolkits.axes_grid1 import make_axes_locatable
import os.path
from scipy.optimize import curve_fit
import scipy.interpolate
import pandas as pd
import plotly.validators.scatter.marker


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

def energy_list_maker_fixedmom(full_energy_list, fixed_mom):
    Ecm_list = []
    Elat_list = []


    for i in range(len(full_energy_list)):
        if(full_energy_list[i][1][0]==fixed_mom[0] and full_energy_list[i][1][1]==fixed_mom[1] and full_energy_list[i][1][2]==fixed_mom[2]):
            Ecm_list.append(float(full_energy_list[i][2]))
            Elat_list.append(float(full_energy_list[i][3]))

    return Ecm_list, Elat_list 



plt.rcParams.update({'font.size': 18})
plt.rc('font',**{'family':'serif','serif':['Computer Modern Roman']})
plt.rc('text', usetex=True)

drive = "/home/digonto/Codes/Practical_Lattice_v2/3body_quantization/test_files/F3inv_KKpi_pole_testing/"

testfile_drive = "/home/digonto/Codes/Practical_Lattice_v2/3body_quantization/test_files/"
filename1 = drive + "F3_for_pole_KKpi_L20_nP_000.dat"
filename2 = drive + "F3_for_pole_KKpi_L20_nP_100.dat"
filename3 = drive + "F3_for_pole_KKpi_L20_nP_110.dat"
filename4 = drive + "F3_for_pole_KKpi_L20_nP_111.dat"
filename5 = drive + "F3_for_pole_KKpi_L20_nP_200.dat"

spectrum1 = testfile_drive + "non_int_3body_nP000.dat"
spectrum2 = testfile_drive + "non_int_3body_nP100.dat"
spectrum3 = testfile_drive + "non_int_3body_nP110.dat"
spectrum4 = testfile_drive + "non_int_3body_nP111.dat"
spectrum5 = testfile_drive + "non_int_3body_nP200.dat"

Gpole1 = testfile_drive + "Gpole_3body_nP000.dat"
Gpole2 = testfile_drive + "Gpole_3body_nP100.dat"
Gpole3 = testfile_drive + "Gpole_3body_nP110.dat"
Gpole4 = testfile_drive + "Gpole_3body_nP111.dat"
Gpole5 = testfile_drive + "Gpole_3body_nP200.dat"

kdf0_spectrum1 = drive + "Kdf0_spectrum_nP_000_L20.dat"
kdf0_spectrum2 = drive + "Kdf0_spectrum_nP_100_L20.dat"
kdf0_spectrum3 = drive + "Kdf0_spectrum_nP_110_L20.dat"
kdf0_spectrum4 = drive + "Kdf0_spectrum_nP_111_L20.dat"
kdf0_spectrum5 = drive + "Kdf0_spectrum_nP_200_L20.dat"

atmpi = 1#0.06906

#raulF3file = "detF3inv_test_FRL_L5"
filelist = [filename1, filename2, filename3, filename4, filename5]
kdf_spec_filelist = [kdf0_spectrum1, kdf0_spectrum2, kdf0_spectrum3, kdf0_spectrum4, kdf0_spectrum5 ]
#additionalpole_filelist = [spectrum1, spectrum2, spectrum3, spectrum4, spectrum5]
#additionalpole_filelist1 = [Gpole1, Gpole2, Gpole3, Gpole4, Gpole5]
titlelist = ["000","100","110","111","200"]
fig,ax = plt.subplots(5,1,figsize=(16,25))
counter = 0
energy_file = testfile_drive + "S2I2.energies"
full_energy_list, irrep_list = irrep_list_maker(energy_file)
selected_irrep_list = ["000_A1m","100_A2","110_A2","111_A2","200_A2"]


for file in filelist:
    print("file loaded = ",file)
    filename = file 
    (En, EcmR, ReF3det, ImF3det, 
     ReF3sum, ImF3sum, RedetInvF3, ImdetInvF3,
     ResumInvF3, ImsumInvF3) = np.genfromtxt(filename, unpack=True)

    spec_file = kdf_spec_filelist[counter]
    (L_spec, kdf_spec) = np.genfromtxt(spec_file, unpack=True)
   #here F3det1 and F3iso1 are the denom part multiplied with 1 vector on both sides 

    non_int_energies_ecm, non_int_energies_elab = energy_list_maker_fixedmom(full_energy_list, titlelist[counter])
     
    zero_y_val = []
    for i in range(len(non_int_energies_ecm)):
        zero_y_val.append(0.0)
    
    np_zero_y_val = np.array(zero_y_val)

    
    ax[counter].set_xlim(0.262,0.42)
    #ax[counter].set_ylim(-1E-17,1E-17)
    ax[counter].set_ylim(-0.0000051,0.0000051)
    ax[counter].tick_params(axis='both', which='major', labelsize=25)
    ax[counter].tick_params(axis='both', which='minor', labelsize=25)
    #ax.set_yscale('log')
    #L = 6
    HFGtitle = "$(\mathcal{K}_2^{-1} + F_2 + G)^{-1}$"
    titlestring = "$P = $" + str(titlelist[counter])
    ax[counter].set_title(titlestring)
    ax[counter].plot(EcmR,ReF3sum, linewidth=2, zorder=4)
    #ax[counter].plot(EcmR,F2sum, linewidth=3, zorder=4,label="$F_2$")
    #ax[counter].plot(EcmR,Gsum, linewidth=3, zorder=4,label="$G$")
    #ax[counter].plot(EcmR,K2iplusFplusG, linewidth=3, zorder=4,label="$\\mathcal{K}_2^{-1} + F+G$")
    
    #ax[counter].plot(EcmR1,F21sum, linewidth=3, label=str("$F_2$, ") + titlestring)
    #ax[counter].plot(EcmR1,F3iso1/10000000000, linewidth=4, color='red', label=HFGtitle, zorder=3)
    ax[counter].scatter(non_int_energies_ecm,zero_y_val, marker='o', s=100, edgecolor="teal", facecolor='white',zorder=5)
    #ax[counter].scatter(gpole1,zero_y_val1, marker='o', s=100, edgecolor="red", linewidth=2, facecolor='white',zorder=5)
    
    #ax[counter].scatter(Ecm_int,np_zero_y_val2, marker='s', s=100, edgecolor="red", linewidth=2, facecolor='white',zorder=5)
    
    #ax[counter].plot(EcmR,Reqsq, linewidth=3, label="$q_{2,p}^2$")
    #ax[counter].plot(EcmR,Recutoff, linewidth=3, label="$H(p)$")
    #ax[counter].axvline(x=Recheckzero1[0],linestyle='--',color='red')
    #ax[counter].axvline(x=Recheckzero2[0],linestyle='--',color='green')
    #ax[counter].axvline(x=threshold[0],linestyle='--',color='black')
    ax[counter].axhline(y=0,linestyle='--',color='black',zorder=2)

    ax[counter].axvline(x=0.37,linestyle='--',color='black',zorder=2)
    
    for i in range(len(kdf_spec)):
        ax[counter].axvline(x=kdf_spec[i],linestyle='--',color='red',zorder=6)
    
    #ax.scatter(En, abs(detF3), marker='o', s=60, edgecolor="teal", facecolor='white',zorder=5, label="Digonto")
    #ax.scatter(En1, abs(detF31), marker='s', s=60,edgecolor="red", facecolor='white',zorder=3, label="FRL")
    
    #ax[counter].legend()
    #fig.tight_layout()
    ax[4].set_xlabel("$E_{cm}$", fontsize=25)
    ax[counter].set_ylabel("$F_{3,\\textrm{iso}}$",fontsize=25)
    #plt.draw()
    counter=counter+1


outputfile_str = "F3_KKpi_L20_v3.pdf"

plt.show()
#plt.savefig(outputfile_str)

#plt.close()

#print(full_energy_list[0][1][2])