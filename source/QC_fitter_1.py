#This function fits the
import numpy as np 
import matplotlib as mpl 
import matplotlib.pyplot as plt 
from scipy.optimize import curve_fit
import scipy.interpolate
from mpl_toolkits.axes_grid1 import make_axes_locatable
from PyPDF2 import PdfMerger

import os 
import subprocess 
import math 
import sys 

from timeit import default_timer as timer

threebody_path_ubuntu = '/home/digonto/Codes/Practical_Lattice_v2/3body_quantization/'
threebody_path_macos = '/Users/digonto/GitHub/3body_quantization/'
macos_path2 = '/Users/digonto/GitHub/jackknife_codes/'
ubuntu_path2 = '/home/digonto/Codes/Practical_Lattice_v2/jackknife_codes/'

from sys import platform 
#print(platform)
if platform=="linux" or platform=="linux2":
    print("platform = ",platform)
    jackpath = ubuntu_path2
    threebody_path = threebody_path_ubuntu
elif platform=="darwin":
    print("platform = ",platform)
    jackpath = macos_path2
    threebody_path = threebody_path_macos

sys.path.insert(1, jackpath)

import jackknife 
from lattice_data_covariance import covariance_between_states_L20

def E_to_Ecm(En, P):
    return np.sqrt(En**2 - P**2)

def Esq_to_Ecmsq(En, P):
    return En**2 - P**2

def Ecmsq_to_Esq(Ecm, P):
    return Ecm**2 + P**2

def QC3(K3iso, F3inv):
    return F3inv + K3iso 

def sign_func(val):
    if(val>0.0):
        return 1.0; 
    else:
        return -1.0; 

def QC3_bissection_spline_based(pointA, pointB, K3iso1, K3iso2, nPx, nPy, nPz, nmax, tol, spline_size, energy_eps):
    A = pointA + energy_eps 
    B = pointB - energy_eps 
    
    F3inv_A = subprocess.check_output(['./spline_F3inv',str(nPx),str(nPy),str(nPz),str(spline_size),str(pointA),str(pointB),str(A)],shell=False)
    F3inv_result_A = F3inv_A.decode('utf-8')
    F3inv_fin_result_A = float(F3inv_result_A)
    K3iso_A = K3iso1 + K3iso2*(A*A)
    QC_A = QC3(K3iso_A, F3inv_fin_result_A)
    
    F3inv_B = subprocess.check_output(['./spline_F3inv',str(nPx),str(nPy),str(nPz),str(spline_size),str(pointA),str(pointB),str(B)],shell=False)
    F3inv_result_B = F3inv_B.decode('utf-8')
    F3inv_fin_result_B = float(F3inv_result_B)
    K3iso_B = K3iso1 + K3iso2*(B*B)
    QC_B = QC3(K3iso_B, F3inv_fin_result_B)

    print("QC_A = ",QC_A)
    print("QC_B = ",QC_B)
    
    if(QC_A==0.0):
        return A 
    elif(QC_B==0.0):
        return B 
    else:
        fin_result = 0.0
        for i in range(nmax):
            C = (A + B)/2.0 
            F3inv_C = subprocess.check_output(['./spline_F3inv',str(nPx),str(nPy),str(nPz),str(spline_size),str(pointA),str(pointB),str(C)],shell=False)
            F3inv_result_C = F3inv_C.decode('utf-8')
            F3inv_fin_result_C = float(F3inv_result_C)
            K3iso_C = K3iso1 + K3iso2*(C*C)
            QC_C = QC3(K3iso_C, F3inv_fin_result_C)
            print("QC_C = ",QC_C)
            if(QC_C==0.0 or (B-A)/2.0 < tol):
                fin_result = C 
                print("entered breaking condition for bissection with C = ",C)
                break 
            
            if(sign_func(QC_C)==sign_func(QC_A)):
                A = C 
                QC_A = QC_C 
            elif(sign_func(QC_C)==sign_func(QC_B)):
                B = C 
                QC_B = QC_C

            #fin_result = C  
        return fin_result 

def QC3_bissection_eigen_based(pointA, pointB, K3iso1, K3iso2, nPx, nPy, nPz, nmax, tol):
    A = pointA 
    B = pointB 
    
    F3inv_A = subprocess.check_output(['./eigen_F3inv',str(nPx),str(nPy),str(nPz),str(A)],shell=False)
    F3inv_result_A = F3inv_A.decode('utf-8')
    F3inv_fin_result_A = float(F3inv_result_A)
    K3iso_A = K3iso1 + K3iso2*(A*A)
    QC_A = QC3(K3iso_A, F3inv_fin_result_A)
    
    F3inv_B = subprocess.check_output(['./eigen_F3inv',str(nPx),str(nPy),str(nPz),str(B)],shell=False)
    F3inv_result_B = F3inv_B.decode('utf-8')
    F3inv_fin_result_B = float(F3inv_result_B)
    K3iso_B = K3iso1 + K3iso2*(B*B)
    QC_B = QC3(K3iso_B, F3inv_fin_result_B)

    print("QC_A = ",QC_A)
    print("QC_B = ",QC_B)
    
    if(QC_A==0.0):
        return A 
    elif(QC_B==0.0):
        return B 
    else:
        fin_result = 0.0
        for i in range(nmax):
            C = (A + B)/2.0 
            F3inv_C = subprocess.check_output(['./eigen_F3inv',str(nPx),str(nPy),str(nPz),str(C)],shell=False)
            F3inv_result_C = F3inv_C.decode('utf-8')
            F3inv_fin_result_C = float(F3inv_result_C)
            K3iso_C = K3iso1 + K3iso2*(C*C)
            QC_C = QC3(K3iso_C, F3inv_fin_result_C)
            print("QC_C = ",QC_C)
            if(QC_C==0.0 or abs(B-A)/2.0 < tol):
                fin_result = C 
                print("B-A/2 = ",abs(B-A)/2.0," tol = ", tol)
                print("entered breaking condition for bissection with C = ",C)
                break 
            
            if(sign_func(QC_C)==sign_func(QC_A)):
                A = C 
                QC_A = QC_C 
            elif(sign_func(QC_C)==sign_func(QC_B)):
                B = C 
                QC_B = QC_C

            #fin_result = C  
        return fin_result 


#nmax is the max iteration number 
#tol is the toleration for the bissection procedure 
#spline_size is the max number of splines we set
def K3iso_fitting_function(x0, nPx, nPy, nPz, nmax, tol, spline_size, corr_mat_inv):
    energy_eps = 1.0E-5 
    K3iso_1 = x0[0]
    K3iso_2 = x0[1]

    F3_drive = "/home/digonto/Codes/Practical_Lattice_v2/3body_quantization/test_files/F3_for_pole_KKpi_L20/"
    F3_file = F3_drive + "ultraHQ_F3_for_pole_KKpi_L20_nP_" + str(nPx) + str(nPy) + str(nPz) + ".dat"
    
    (En1, Ecm1, norm1, F3, F2, G, K2inv, Hinv) = np.genfromtxt(F3_file,unpack=True)
    F3inv = np.zeros((len(F3)))
    for i in range(len(F3)):
        F3inv[i] = 1.0/F3[i]

    F3inv_poles_drive = "/home/digonto/Codes/Practical_Lattice_v2/3body_quantization/test_files/F3inv_poles_L20/"    
    F3inv_poles_file = F3inv_poles_drive + "F3inv_poles_nP_" + str(nPx) + str(nPy) + str(nPz) + "_L20.dat"
    
    (L1, F3inv_poles) = np.genfromtxt(F3inv_poles_file, unpack=True)

    spectrum_drive = "/home/digonto/Codes/Practical_Lattice_v2/3body_quantization/lattice_data/KKpi_interacting_spectrum/Three_body/L_20_only/"
    spectrum_filename = spectrum_drive + "KKpi_spectrum.P_" + str(nPx) + str(nPy) + str(nPz) + "_usethisfile"

    (L2, Elatt_CM, Elatt_CM_stat, Elatt_CM_sys) = np.genfromtxt(spectrum_filename, unpack=True)

    energy_cutoff = 0.37

    Elatt_CM_selected = []
    Elatt_CM_SE_selected = []
    for i in range(len(Elatt_CM)):
        if(Elatt_CM[i]<energy_cutoff):
            Elatt_CM_selected.append(Elatt_CM[i])
            Elatt_CM_SE_selected.append(Elatt_CM_sys[i])

    np_Elatt_CM_selected = np.array(Elatt_CM_selected) 
    np_Elatt_CM_SE_selected = np.array(Elatt_CM_SE_selected)

    E_size = len(Elatt_CM_selected)
    print("E_size = ",E_size)

    E_QC_CM = [] 

    for i in range(E_size):
        Energy_A_CM = F3inv_poles[i] + energy_eps
        Energy_B_CM = F3inv_poles[i+1] - energy_eps 

        print("ECM A = ",Energy_A_CM)
        print("ECM B = ",Energy_B_CM)
        print("K3iso1 = ",K3iso_1)
        print("K3iso2 = ",K3iso_2)
        QC_spectrum = QC3_bissection_eigen_based(Energy_A_CM, Energy_B_CM, K3iso_1, K3iso_2, nPx, nPy, nPz, nmax, tol)
        print("bissection result = ",QC_spectrum)
        E_QC_CM.append(QC_spectrum)

    np_E_QC_CM = np.array(E_QC_CM)

    chisquare = 0.0 
    
    '''
    for i in range(E_size):
        iterm = (np_Elatt_CM_selected[i] - np_E_QC_CM[i])
        chisquare_val = iterm*iterm 
        chisquare = chisquare + chisquare_val

    '''
    
    for i in range(E_size):
        for j in range(E_size):
            iterm = (np_Elatt_CM_selected[i] - np_E_QC_CM[i])/np_Elatt_CM_SE_selected[i]
            jterm = (np_Elatt_CM_selected[j] - np_E_QC_CM[j])/np_Elatt_CM_SE_selected[j]
            chisquare_val = iterm*corr_mat_inv[i][j]*jterm
            chisquare = chisquare + chisquare_val  
     

    print("chisquare = ",chisquare)
    print("--------------------------")
    print("\n")
    return chisquare 


#this is based on the jackknife and lattice_data_covariance code in the 
#jackknife_codes repository, this will be supplied a list of spectrum 
#with frame momenta P as [nPx, nPy, nPz], state number [0,1,2,0,1,..] along with their corresponding 
#covariance matrices to perform the fitting, this is much more robust than 
#the previous fitting function which was built for checking a single frame spectrum 
def K3iso_fitting_function_all_moms_two_parameter(x0, nmax, states_avg, states_err, nP_list, state_no, covariance_matrix_inv, tol, spline_size):
    energy_eps = 1.0E-5 
    K3iso_1 = x0[0]
    K3iso_2 = x0[1]

    QC_states = []

    #for i in range(len(state_no)):
    #    print("state nums = ",i,state_no[i])
    
    for ind in range(0,len(states_avg),1):
        state_ecm = states_avg[ind]
        nPx = nP_list[ind][0]
        nPy = nP_list[ind][1]
        nPz = nP_list[ind][2]

        #print(state_no)
        #print("state num size = ",len(state_no))
        #print("i = ",ind)
        #print(state_no[ind+1])
        
        state_num_val = state_no[ind]

        

        if(state_num_val==0):    
            F3_drive = "/home/digonto/Codes/Practical_Lattice_v2/3body_quantization/test_files/F3_for_pole_KKpi_L20/"
            F3_file = F3_drive + "ultraHQ_F3_for_pole_KKpi_L20_nP_" + str(nPx) + str(nPy) + str(nPz) + ".dat"
    
            (En1, Ecm1, norm1, F3, F2, G, K2inv, Hinv) = np.genfromtxt(F3_file,unpack=True)
            F3inv = np.zeros((len(F3)))
            for i in range(0,len(F3),1):
                F3inv[i] = 1.0/F3[i]

            F3inv_poles_drive = "/home/digonto/Codes/Practical_Lattice_v2/3body_quantization/test_files/F3inv_poles_L20/"    
            F3inv_poles_file = F3inv_poles_drive + "F3inv_poles_nP_" + str(nPx) + str(nPy) + str(nPz) + "_L20.dat"
    
            (L1, F3inv_poles) = np.genfromtxt(F3inv_poles_file, unpack=True)


        Energy_A_CM = F3inv_poles[state_num_val] + energy_eps
        Energy_B_CM = F3inv_poles[state_num_val + 1] - energy_eps

        print("-----------------K3isoFit---------------------")
        print("P = ",nPx, nPy, nPz)
        print("state no = ",state_no[ind])
        print("ECM A = ",Energy_A_CM)
        print("ECM B = ",Energy_B_CM)
        print("K3iso1 = ",K3iso_1)
        print("K3iso2 = ",K3iso_2)
        QC_spectrum = QC3_bissection_eigen_based(Energy_A_CM, Energy_B_CM, K3iso_1, K3iso_2, nPx, nPy, nPz, nmax, tol)
        print("bissection result = ",QC_spectrum)
        Diff = abs((states_avg[ind] - QC_spectrum)/states_avg[ind])*100.0
        print("Ecm_latt = ",states_avg[ind]," Ecm_QC = ",QC_spectrum, " Diff = ",Diff,"%")
        QC_states.append(QC_spectrum)
        print("---------------------------------------------")

    #energy_cutoff = 0.37
    np_QC_states = np.array(QC_states)

    chisquare = 0.0 
    
    '''
    for i in range(E_size):
        iterm = (np_Elatt_CM_selected[i] - np_E_QC_CM[i])
        chisquare_val = iterm*iterm 
        chisquare = chisquare + chisquare_val

    '''
    E_size = len(states_avg)
    for i in range(0,E_size,1):
        for j in range(0,E_size,1):
            iterm = (states_avg[i] - np_QC_states[i])/states_err[i]
            jterm = (states_avg[j] - np_QC_states[j])/states_err[j]
            chisquare_val = iterm*covariance_matrix_inv[i][j]*jterm
            chisquare = chisquare + chisquare_val  
     

    print("chisquare = ",chisquare)
    print("--------------------------")
    print("\n")
    return chisquare 


#This is spline based 
def K3iso_fitting_function_all_moms_one_parameter(x0, nmax, states_avg, states_err, nP_list, state_no, covariance_matrix_inv, tol, spline_size):
    energy_eps = 1.0E-3
    K3iso_1 = x0[0]
    K3iso_2 = 0.0 #x0[1]

    QC_states = []

    #for i in range(len(state_no)):
    #    print("state nums = ",i,state_no[i])
    
    for ind in range(0,len(states_avg),1):
        state_ecm = states_avg[ind]
        nPx = nP_list[ind][0]
        nPy = nP_list[ind][1]
        nPz = nP_list[ind][2]

        #print(state_no)
        #print("state num size = ",len(state_no))
        #print("i = ",ind)
        #print(state_no[ind+1])
        
        state_num_val = state_no[ind]

        

        if(state_num_val==0):    
            F3_drive = "/home/digonto/Codes/Practical_Lattice_v2/3body_quantization/test_files/F3_for_pole_KKpi_L20/"
            F3_file = F3_drive + "ultraHQ_F3_for_pole_KKpi_L20_nP_" + str(nPx) + str(nPy) + str(nPz) + ".dat"
    
            (En1, Ecm1, norm1, F3, F2, G, K2inv, Hinv) = np.genfromtxt(F3_file,unpack=True)
            F3inv = np.zeros((len(F3)))
            for i in range(0,len(F3),1):
                F3inv[i] = 1.0/F3[i]

            F3inv_poles_drive = "/home/digonto/Codes/Practical_Lattice_v2/3body_quantization/test_files/F3inv_poles_L20/"    
            F3inv_poles_file = F3inv_poles_drive + "F3inv_poles_region_nP_" + str(nPx) + str(nPy) + str(nPz) + "_L20.dat"
    
            (L1, F3inv_poles, F3inv_poles_region_start, F3inv_poles_region_end) = np.genfromtxt(F3inv_poles_file, unpack=True)


        Energy_A_CM = F3inv_poles_region_start[state_num_val] #+ energy_eps #F3inv_poles[state_num_val] + energy_eps
        Energy_B_CM = F3inv_poles_region_end[state_num_val] #- energy_eps #F3inv_poles[state_num_val + 1] - energy_eps

        print("Energy_A_Cm = ",Energy_A_CM)
        print("Energy_B_CM = ",Energy_B_CM)
        for i in range(0,len(Ecm1)-1,1):
            if(Energy_A_CM>=Ecm1[i] and Energy_A_CM<=Ecm1[i+1]):
                ind1 = i
            if(Energy_B_CM>=Ecm1[i] and Energy_B_CM<=Ecm1[i+1]):
                ind2 = i
        print("ind1 = ",ind1)
        print("ind2 = ",ind2)
        Energy_A_CM = Ecm1[ind1 + 5]
        Energy_B_CM = Ecm1[ind2 - 5]

        print("-----------------K3isoFit---------------------")
        print("P = ",nPx, nPy, nPz)
        print("state no = ",state_no[ind])
        print("ECM A = ",Energy_A_CM)
        print("ECM B = ",Energy_B_CM)
        print("K3iso = ",K3iso_1)
        print("K3iso2 = ",K3iso_2)             
        QC_spectrum = QC3_bissection_eigen_based(Energy_A_CM, Energy_B_CM, K3iso_1, K3iso_2, nPx, nPy, nPz, nmax, tol)#QC3_bissection_spline_based(Energy_A_CM, Energy_B_CM, K3iso_1, K3iso_2, nPx, nPy, nPz, nmax, tol, spline_size)
        #QC_spectrum = QC3_bissection_spline_based(Energy_A_CM, Energy_B_CM, K3iso_1, K3iso_2, nPx, nPy, nPz, nmax, tol, spline_size, energy_eps)
        print("bissection result = ",QC_spectrum)
        Diff = abs((states_avg[ind] - QC_spectrum)/states_avg[ind])*100.0
        print("Ecm_latt = ",states_avg[ind]," Ecm_QC = ",QC_spectrum, " Diff = ",Diff,"%")
        QC_states.append(QC_spectrum)
        print("---------------------------------------------")

    #energy_cutoff = 0.37
    np_QC_states = np.array(QC_states)

    chisquare = 0.0 
    
    '''
    for i in range(E_size):
        iterm = (np_Elatt_CM_selected[i] - np_E_QC_CM[i])
        chisquare_val = iterm*iterm 
        chisquare = chisquare + chisquare_val

    '''
    E_size = len(states_avg)
    for i in range(0,E_size,1):
        for j in range(0,E_size,1):
            iterm = (states_avg[i] - np_QC_states[i])/states_err[i]
            jterm = (states_avg[j] - np_QC_states[j])/states_err[j]
            chisquare_val = iterm*covariance_matrix_inv[i][j]*jterm
            chisquare = chisquare + chisquare_val  
     

    print("chisquare = ",chisquare)
    print("--------------------------")
    print("\n")
    return chisquare 

def QC_spectrum_one_parameter(x0, nmax, states_avg, states_err, nP_list, state_no, tol):
    energy_eps = 1.0E-3
    K3iso_1 = x0[0]
    K3iso_2 = 0.0 #x0[1]

    QC_states = []

    #for i in range(len(state_no)):
    #    print("state nums = ",i,state_no[i])
    
    for ind in range(0,len(states_avg),1):
        state_ecm = states_avg[ind]
        nPx = nP_list[ind][0]
        nPy = nP_list[ind][1]
        nPz = nP_list[ind][2]

        #print(state_no)
        #print("state num size = ",len(state_no))
        #print("i = ",ind)
        #print(state_no[ind+1])
        
        state_num_val = state_no[ind]

        

        if(state_num_val==0):    
            F3_drive = "/home/digonto/Codes/Practical_Lattice_v2/3body_quantization/test_files/F3_for_pole_KKpi_L20/"
            F3_file = F3_drive + "ultraHQ_F3_for_pole_KKpi_L20_nP_" + str(nPx) + str(nPy) + str(nPz) + ".dat"
    
            (En1, Ecm1, norm1, F3, F2, G, K2inv, Hinv) = np.genfromtxt(F3_file,unpack=True)
            F3inv = np.zeros((len(F3)))
            for i in range(0,len(F3),1):
                F3inv[i] = 1.0/F3[i]

            F3inv_poles_drive = "/home/digonto/Codes/Practical_Lattice_v2/3body_quantization/test_files/F3inv_poles_L20/"    
            F3inv_poles_file = F3inv_poles_drive + "F3inv_poles_region_nP_" + str(nPx) + str(nPy) + str(nPz) + "_L20.dat"
    
            (L1, F3inv_poles, F3inv_poles_region_start, F3inv_poles_region_end) = np.genfromtxt(F3inv_poles_file, unpack=True)


        Energy_A_CM = F3inv_poles_region_start[state_num_val] #+ energy_eps #F3inv_poles[state_num_val] + energy_eps
        Energy_B_CM = F3inv_poles_region_end[state_num_val] #- energy_eps #F3inv_poles[state_num_val + 1] - energy_eps

        print("Energy_A_Cm = ",Energy_A_CM)
        print("Energy_B_CM = ",Energy_B_CM)
        for i in range(0,len(Ecm1)-1,1):
            if(Energy_A_CM>=Ecm1[i] and Energy_A_CM<=Ecm1[i+1]):
                ind1 = i
            if(Energy_B_CM>=Ecm1[i] and Energy_B_CM<=Ecm1[i+1]):
                ind2 = i
        print("ind1 = ",ind1)
        print("ind2 = ",ind2)
        Energy_A_CM = Ecm1[ind1 + 5]
        Energy_B_CM = Ecm1[ind2 - 5]

        print("-----------------K3isoFit---------------------")
        print("P = ",nPx, nPy, nPz)
        print("state no = ",state_no[ind])
        print("ECM A = ",Energy_A_CM)
        print("ECM B = ",Energy_B_CM)
        print("K3iso = ",K3iso_1)
        print("K3iso2 = ",K3iso_2)             
        QC_spectrum = QC3_bissection_eigen_based(Energy_A_CM, Energy_B_CM, K3iso_1, K3iso_2, nPx, nPy, nPz, nmax, tol)#QC3_bissection_spline_based(Energy_A_CM, Energy_B_CM, K3iso_1, K3iso_2, nPx, nPy, nPz, nmax, tol, spline_size)
        #QC_spectrum = QC3_bissection_spline_based(Energy_A_CM, Energy_B_CM, K3iso_1, K3iso_2, nPx, nPy, nPz, nmax, tol, spline_size, energy_eps)
        print("bissection result = ",QC_spectrum)
        Diff = abs((states_avg[ind] - QC_spectrum)/states_avg[ind])*100.0
        print("Ecm_latt = ",states_avg[ind]," Ecm_QC = ",QC_spectrum, " Diff = ",Diff,"%")
        QC_states.append(QC_spectrum)
        print("---------------------------------------------")

    #energy_cutoff = 0.37
    np_QC_states = np.array(QC_states)

    state_counter = 0
    filename = "QC_states_with_K3iso_" + str(K3iso_1) + "_nP_" + str(nPx) + str(nPy) + str(nPz) + ".dat"
    f = open(filename,'w' )
    for i in np_QC_states:
        print("QC state ",state_counter," = ",i)
        state_counter = state_counter + 1 
        f.write("20" + '\t' + str(i) + '\n')

    f.close()     
 


def test():
    nPx = 0
    nPy = 0
    nPz = 0 
    K3iso1 = 1000000.0
    K3iso2 = 10000000.0 

    x0 = [K3iso1, K3iso2]
    nmax = 100
    tol = 1E-10 
    spline_size = 50 

    rows, cols = (3, 3)
    arr = [[0 for i in range(cols)] for j in range(rows)]
    corr_mat = arr 

    corr_mat[0][0] = 1.0
    corr_mat[0][1] = 0.91
    corr_mat[0][2] = 0.90
    corr_mat[1][0] = corr_mat[0][1]
    corr_mat[1][1] = 1.0 
    corr_mat[2][2] = 1.0 
    corr_mat[2][0] = corr_mat[0][2] 
    corr_mat[1][2] = 0.85
    corr_mat[2][1] = corr_mat[1][2]

    print("we have started running")
    np_corr_mat = np.array(corr_mat)
    corr_mat_inv = np.linalg.inv(np_corr_mat)
    print("we inverted the corr mat")
    print(corr_mat_inv)


    res = scipy.optimize.minimize(K3iso_fitting_function,x0=x0,args=(nPx, nPy, nPz, nmax, tol, spline_size, corr_mat_inv),method='Nelder-Mead')
    
    print(res) 

'''    1.00  0.91    0.90  
             1.00    0.85  
                     1.00
'''


#This one is with multiple P frames 
def test1():
    nPx = 0
    nPy = 0
    nPz = 0 
    K3iso1 = 200000#0.000127
    K3iso2 = 0.0#10000000.0 

    x0 = [K3iso1] #[K3iso1, K3iso2]
    nmax = 500
    tol = 1E-16 
    spline_size = 500 

    #list_of_mom = ['000_A1m','100_A2','110_A2','111_A2','200_A2']
    list_of_mom = ['000_A1m']
    
    states_avg, states_err, nP_list, state_no, covariance_mat = covariance_between_states_L20(0.38, list_of_mom)

    for i in range(len(states_avg)):
        print(states_avg[i],states_err[i],nP_list[i],state_no[i])
    print("we have started running")
    np_cov_mat = np.array(covariance_mat)
    cov_mat_inv = np.linalg.inv(np_cov_mat)
    print("we inverted the corr mat")
    print(np_cov_mat)
    print("------------------------")
    print(cov_mat_inv)
    print("------------------------")

    start = timer()
    res = scipy.optimize.minimize(K3iso_fitting_function_all_moms_one_parameter,x0=x0,args=(nmax, states_avg, states_err, nP_list, state_no, cov_mat_inv, tol, spline_size),method='Nelder-Mead')
    end = timer()

    print(res) 
    print("time = ",end - start)


#We plot the chisquare based on K3iso1 
def test2():
    nPx = 0
    nPy = 0
    nPz = 0 
    #K3iso1 = 0.1#0.000127
    K3iso2 = 0.0#10000000.0 

    #x0 = [K3iso1] #[K3iso1, K3iso2]
    nmax = 100
    tol = 1E-10 
    spline_size = 500 

    #list_of_mom = ['000_A1m','100_A2','110_A2','111_A2','200_A2']
    list_of_mom = ['000_A1m']
    
    states_avg, states_err, nP_list, state_no, covariance_mat = covariance_between_states_L20(0.38, list_of_mom)

    for i in range(len(states_avg)):
        print(states_avg[i],states_err[i],nP_list[i],state_no[i])
    print("we have started running")
    np_cov_mat = np.array(covariance_mat)
    cov_mat_inv = np.linalg.inv(np_cov_mat)
    print("we inverted the corr mat")
    print(np_cov_mat)
    print("------------------------")
    print(cov_mat_inv)
    print("------------------------")

    start = timer()
    #res = scipy.optimize.minimize(K3iso_fitting_function_all_moms_one_parameter,x0=x0,args=(nmax, states_avg, states_err, nP_list, state_no, cov_mat_inv, tol, spline_size),method='Nelder-Mead')
    
    K3iso_space = np.linspace(2000,200000,5000)

    f = open("chisquare_vs_K3iso1_nP_000_1.dat",'w')
    for i in range(0,len(K3iso_space),1):
        K3iso1 = K3iso_space[i]
        x0 = [K3iso1]
        chisquare = K3iso_fitting_function_all_moms_one_parameter(x0, nmax, states_avg, states_err, nP_list, state_no, cov_mat_inv, tol, spline_size)
        f.write(str(K3iso1) + '\t' + str(chisquare) + '\n')
        print(K3iso1, chisquare)
    end = timer()
    f.close() 
    #print(res) 
    print("time = ",end - start)

#QC_spectrum with one parameter K3iso1 = constant 
def test3():
    nPx = 0
    nPy = 0
    nPz = 0 
    K3iso1 = -189765.8705129288#0.000127
    K3iso2 = 0.0#10000000.0 

    x0 = [K3iso1] #[K3iso1, K3iso2]
    nmax = 500
    tol = 1E-16 
    spline_size = 500 

    list_of_mom = ['000_A1m','100_A2','110_A2','111_A2','200_A2']
    #list_of_mom = ['200_A2']
    #list_of_mom = ['000_A1m']
    for irreps in list_of_mom:
        newlist_val = [irreps]

        states_avg, states_err, nP_list, state_no, covariance_mat = covariance_between_states_L20(0.40, newlist_val)

        print("irrep = ",irreps)
        for i in range(len(states_avg)):
            print(states_avg[i],states_err[i],nP_list[i],state_no[i])
        print("we have started running")
        np_cov_mat = np.array(covariance_mat)
        cov_mat_inv = np.linalg.inv(np_cov_mat)
        print("we inverted the corr mat")
        print(np_cov_mat)
        print("------------------------")
        print(cov_mat_inv)
        print("------------------------")

        start = timer()
        #res = scipy.optimize.minimize(K3iso_fitting_function_all_moms_one_parameter,x0=x0,args=(nmax, states_avg, states_err, nP_list, state_no, cov_mat_inv, tol, spline_size),method='Nelder-Mead')
        QC_spectrum_one_parameter(x0, nmax, states_avg, states_err, nP_list, state_no, tol)
    
        end = timer() 
        #print(res) 
        print("time = ",end - start)

def test5():
    list_of_mom = ['000_A1m','100_A2','110_A2','111_A2','200_A2']
    #list_of_mom = ['200_A2']
    #list_of_mom = ['000_A1m']
    for irreps in list_of_mom:
        newlist_val = [irreps]

        states_avg, states_err, nP_list, state_no, covariance_mat = covariance_between_states_L20(0.9, newlist_val)

        print("irrep = ",irreps)
        for i in range(len(states_avg)):
            print(states_avg[i],states_err[i],nP_list[i],state_no[i])

#This was done to test the spline based code
#Check spectrum for given K3df values 
def spectrum_checker_for_QC():
    plt.rcParams.update({'font.size': 22})
    plt.rc('font',**{'family':'serif','serif':['Computer Modern Roman']})
    plt.rc('text', usetex=True)
    
    nPx = 0
    nPy = 0
    nPz = 0 

    F3_drive = threebody_path + "test_files/F3_for_pole_KKpi_L20/"
    F3_file = F3_drive + "ultraHQ_F3_for_pole_KKpi_L20_nP_" + str(nPx) + str(nPy) + str(nPz) + ".dat"
    
    (En1, Ecm1, norm1, F3, F2, G, K2inv, Hinv) = np.genfromtxt(F3_file,unpack=True)
    F3inv = np.zeros((len(F3)))
    for i in range(len(F3)):
        F3inv[i] = 1.0/F3[i]

    F3inv_poles_drive = threebody_path + "test_files/F3inv_poles_L20/"    
    F3inv_poles_file = F3inv_poles_drive + "F3inv_poles_nP_" + str(nPx) + str(nPy) + str(nPz) + "_L20.dat"
    
    (L1, F3inv_poles) = np.genfromtxt(F3inv_poles_file, unpack=True)

    F3_poles_drive = threebody_path + "test_files/F3_for_pole_KKpi_L20/"
    F3_poles_file = F3_poles_drive + "F3_poles_nP_" + str(nPx) + str(nPy) + str(nPz) + "_L20.dat"
    
    (LF3, F3poles) = np.genfromtxt(F3_poles_file, unpack=True)
    
    spectrum_drive = threebody_path + "lattice_data/KKpi_interacting_spectrum/Three_body/L_20_only/"
    spectrum_filename = spectrum_drive + "KKpi_spectrum.P_" + str(nPx) + str(nPy) + str(nPz) + "_usethisfile"

    (L2, Elatt_CM, Elatt_CM_stat, Elatt_CM_sys) = np.genfromtxt(spectrum_filename, unpack=True)

    KDFnonzero_spectrum_drive = threebody_path + "test_files/QC_states_L20/"
    KDFnonzero_spectrum_file = KDFnonzero_spectrum_drive + "QC_states_with_K3iso_-189765.8705129288_nP_" + str(nPx) + str(nPy) + str(nPz) + ".dat"
    
    (Lkdf, Kdf_spec) = np.genfromtxt(KDFnonzero_spectrum_file, unpack=True) 
    
    K3iso1 = -189765.8705129288 #-4.45893908e+07  #1336082.36755021
    K3iso2 =  0.0 #4.94547308e+08 #-10866109.92757

    QC_val = []
    for i in range(len(F3inv)):
        K3iso = K3iso1 + K3iso2*Ecm1[i]**2
        QC_temp = QC3(K3iso,F3inv[i])
        QC_val.append(QC_temp)
    
    np_y_val = np.zeros((len(F3inv_poles)))
    
    np_QC_val = np.array(QC_val)


    spline_size = 500
    pointA = F3inv_poles[0] + 1.0E-5
    pointB = F3inv_poles[1] #- 1.0E-5

    Ecm_space = np.linspace(pointA, pointB, 100)
    QC_val_spline = []

    '''
    for i in range(len(Ecm_space)):
        F3inv_C = subprocess.check_output(['./spline_F3inv',str(nPx),str(nPy),str(nPz),str(spline_size),str(pointA),str(pointB),str(Ecm_space[i])],shell=False)
        F3inv_result_C = F3inv_C.decode('utf-8')
        F3inv_fin_result_C = float(F3inv_result_C)
        print("Ecm = ",Ecm_space[i],"F3inv = ",F3inv_fin_result_C)
        K3iso = K3iso1 + K3iso2*Ecm_space[i]**2
        QC_temp = QC3(K3iso,F3inv_fin_result_C)
        QC_val_spline.append(QC_temp)

    np_QC_val_spline = np.array(QC_val_spline)            
    '''

    fig, ax = plt.subplots(figsize=(12,5))

    #ax.set_title("K3iso = "+str(K3iso1))
    ax.set_ylim(-5E6,5E6)
    ax.set_xlim(0.26,0.37)
    ax.set_ylabel("$F_{3,iso}^{-1}$")
    ax.set_xlabel("$E_{cm}$")
    ax.plot(Ecm1,F3inv)
    #ax.plot(Ecm1,np_QC_val)
    ax.axhline(y=-K3iso1,color='red',label="$\\mathcal{K}_{3,iso} $= " + str(K3iso1))
    #ax.plot(Ecm1,np_QC_val)
    #ax.plot(Ecm_space,np_QC_val_spline)
    ax.axhline(y=0,color='black')
    #ax.scatter(F3inv_poles,np_y_val,s=100,facecolor='white',edgecolor='red')
    for i in range(0,len(Elatt_CM),1):
        ax.axvline(x=Elatt_CM[i],color='darkorange')

    for i in range(0, len(F3poles), 1):
        ax.axvline(x=F3poles[i],color='black')

    for i in range(0, len(Kdf_spec), 1):
        ax.axvline(x=Kdf_spec[i],color='red')

    ax.legend() 
    fig.tight_layout() 
    plt.draw() 
    outfile = "temp.png"
    plt.savefig(outfile)
    plt.show()
    plt.close()


def spectrum_checker_for_splines():
    plt.rcParams.update({'font.size': 22})
    plt.rc('font',**{'family':'serif','serif':['Computer Modern Roman']})
    plt.rc('text', usetex=True)
    
    nPx = 0
    nPy = 0
    nPz = 0 

    F3_drive = threebody_path + "test_files/F3_for_pole_KKpi_L20/"
    F3_file = F3_drive + "ultraHQ_F3_for_pole_KKpi_L20_nP_" + str(nPx) + str(nPy) + str(nPz) + ".dat"
    
    (En1, Ecm1, norm1, F3, F2, G, K2inv, Hinv) = np.genfromtxt(F3_file,unpack=True)
    F3inv = np.zeros((len(F3)))
    for i in range(len(F3)):
        F3inv[i] = 1.0/F3[i]

    F3inv_poles_drive = threebody_path + "test_files/F3inv_poles_L20/"    
    F3inv_poles_file = F3inv_poles_drive + "F3inv_poles_nP_" + str(nPx) + str(nPy) + str(nPz) + "_L20.dat"
    
    (L1, F3inv_poles) = np.genfromtxt(F3inv_poles_file, unpack=True)

    
    spectrum_drive = threebody_path + "lattice_data/KKpi_interacting_spectrum/Three_body/L_20_only/"
    spectrum_filename = spectrum_drive + "KKpi_spectrum.P_" + str(nPx) + str(nPy) + str(nPz) + "_usethisfile"

    (L2, Elatt_CM, Elatt_CM_stat, Elatt_CM_sys) = np.genfromtxt(spectrum_filename, unpack=True)


    K3iso1 =  1E-7#-4.45893908e+07  #1336082.36755021
    K3iso2 =  0.0#4.94547308e+08 #-10866109.92757

    QC_val = []
    for i in range(len(F3inv)):
        K3iso = K3iso1 + K3iso2*Ecm1[i]**2
        QC_temp = QC3(K3iso,F3inv[i])
        QC_val.append(QC_temp)
    
    np_y_val = np.zeros((len(F3inv_poles)))
    
    np_QC_val = np.array(QC_val)


    spline_size = 500
    
    QC_val_spline = []

    fig, ax = plt.subplots(figsize=(12,5))

    ax.set_ylim(-1E8,1E8)
    ax.set_xlim(0.26,0.37)

    start = timer()

    for i in range(0,len(F3inv_poles)-1,1):
        pointA = F3inv_poles[i] #+ 1.0E-7
        pointB = F3inv_poles[i+1] #- 1.0E-7
        pointC = (pointA + pointB)/2.0 

        Ecm_space1 = np.linspace(pointA, pointB, 100)
        Ecm_space2 = np.linspace(pointC, pointB, 100)

        Ecm_space = []
        F3inv_spline = []
        '''
        for j in range(0,len(Ecm_space1)-1,1):
            F3inv_C = subprocess.check_output(['./spline_F3inv',str(nPx),str(nPy),str(nPz),str(spline_size),str(pointA),str(pointB),str(Ecm_space1[j])],shell=False)
            F3inv_result_C = F3inv_C.decode('utf-8')
            F3inv_fin_result_C = float(F3inv_result_C)
            print("pointA = ",pointA,"pointB = ", pointB, "Ecm = ",Ecm_space1[j],"F3inv = ",F3inv_fin_result_C)
            #K3iso = K3iso1 + K3iso2*Ecm_space[i]**2
            #QC_temp = QC3(K3iso,F3inv_fin_result_C)
            #QC_val_spline.append(QC_temp)
            F3inv_spline.append(F3inv_fin_result_C) 
            Ecm_space.append(Ecm_space1[j])
        '''
        '''    
        for j in range(0,len(Ecm_space2)-1,1):
            F3inv_C = subprocess.check_output(['./spline_F3inv',str(nPx),str(nPy),str(nPz),str(spline_size),str(pointC),str(pointB),str(Ecm_space2[j])],shell=False)
            F3inv_result_C = F3inv_C.decode('utf-8')
            F3inv_fin_result_C = float(F3inv_result_C)
            print("pointC = ",pointC,"pointB = ", pointB, "Ecm = ",Ecm_space2[j],"F3inv = ",F3inv_fin_result_C)
            #K3iso = K3iso1 + K3iso2*Ecm_space[i]**2
            #QC_temp = QC3(K3iso,F3inv_fin_result_C)
            #QC_val_spline.append(QC_temp)
            F3inv_spline.append(F3inv_fin_result_C)
            Ecm_space.append(Ecm_space2[j])
        '''

        np_F3inv_spline = np.array(F3inv_spline)
        np_Ecm_space = np.array(Ecm_space)

        #ax.plot(np_Ecm_space, np_F3inv_spline, color='darkorange',zorder=3) 
        #ax.scatter(np_Ecm_space, np_F3inv_spline, s=100, color='darkorange',zorder=3) 

    np_QC_val_spline = np.array(QC_val_spline)            

    end = timer() 
    print("time = ",end - start)

    #ax.plot(Ecm1,F3inv, color='blue', zorder=4)
    ax.plot(Ecm1,np_QC_val, color='blue', zorder=4)
    
    #ax.plot(Ecm_space,np_QC_val_spline)
    ax.axhline(y=0,color='black')
    for i in range(0,len(F3inv_poles),1):
        ax.axvline(x=F3inv_poles[i],color='darkorange',zorder=6)
    #ax.scatter(F3inv_poles,np_y_val,s=100,facecolor='white',edgecolor='red')
    #for i in range(len(Elatt_CM)):
    #    ax.axvline(x=Elatt_CM[i],color='darkorange')
    plt.show()


#test() 
#test1()
spectrum_checker_for_QC()

#test2()

#test3()

#test5()

#spectrum_checker_for_splines()