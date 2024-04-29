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

drive = "../F3_for_pole_KKpi_L20/"

testfile_drive = "/home/digonto/Codes/Practical_Lattice_v2/3body_quantization/test_files/"
filename1 = drive + "ultraHQ_F3_for_pole_KKpi_L20_nP_000.dat"
filename2 = drive + "ultraHQ_F3_for_pole_KKpi_L20_nP_100.dat"
filename3 = drive + "ultraHQ_F3_for_pole_KKpi_L20_nP_110.dat"
filename4 = drive + "ultraHQ_F3_for_pole_KKpi_L20_nP_111.dat"
filename5 = drive + "ultraHQ_F3_for_pole_KKpi_L20_nP_200.dat"

poles1 = "F3inv_poles_nP_000_L20.dat"
poles2 = "F3inv_poles_nP_100_L20.dat"
poles3 = "F3inv_poles_nP_110_L20.dat"
poles4 = "F3inv_poles_nP_111_L20.dat"
poles5 = "F3inv_poles_nP_200_L20.dat"

atmpi = 1#0.06906

#raulF3file = "detF3inv_test_FRL_L5"
filelist = [filename1, filename2, filename3, filename4, filename5]
pole_filelist = [poles1, poles2, poles3, poles4, poles5]

titlelist = ["000","100","110","111","200"]
fig,ax = plt.subplots(5,1,figsize=(16,10))
counter = 0


for file in filelist:
    print("file loaded = ",file)
    filename = file 
    (En, EcmR, norm, F3, F2, G, K2inv, Hinv) = np.genfromtxt(filename, unpack=True)

    F3inv = 1.0/F3 

    spec_file = pole_filelist[counter]
    (L_spec, pole_spec) = np.genfromtxt(spec_file, unpack=True)
   #here F3det1 and F3iso1 are the denom part multiplied with 1 vector on both sides 

     
    zero_y_val = []
    for i in range(len(pole_spec)):
        zero_y_val.append(0.0)
    
    np_zero_y_val = np.array(zero_y_val)

    
    ax[counter].set_xlim(0.262,0.42)
    #ax[counter].set_ylim(-1E-17,1E-17)
    #ax[counter].set_ylim(-0.00051,0.00051)
    ax[counter].set_ylim(-1E7,1E7)
    ax[counter].tick_params(axis='both', which='major', labelsize=25)
    ax[counter].tick_params(axis='both', which='minor', labelsize=25)
    titlestring = "$P = $" + str(titlelist[counter])
    ax[counter].set_title(titlestring)
    ax[counter].plot(EcmR,F3inv, linewidth=2, zorder=4)
    
    ax[counter].scatter(pole_spec,np_zero_y_val, marker='o', s=100, edgecolor="teal", facecolor='white',zorder=5)
    ax[counter].axhline(y=0,linestyle='--',color='black',zorder=2)

    ax[counter].axvline(x=0.37,linestyle='--',color='black',zorder=2)
    
    fig.tight_layout()
    ax[4].set_xlabel("$E_{cm}$", fontsize=25)
    ax[counter].set_ylabel("$F_{3,\\textrm{iso}}$",fontsize=25)
    #plt.draw()
    counter=counter+1


outputfile_str = "F3_KKpi_L20_v3.pdf"

plt.show()
#plt.savefig(outputfile_str)

#plt.close()

#print(full_energy_list[0][1][2])