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

def QC3_bissection_spline_based(pointA, pointB, K3iso1, K3iso2, nPx, nPy, nPz, nmax, tol, spline_size):
    A = pointA 
    B = pointB 
    
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


#This was done to test the spline based code
def spectrum_checker_for_QC():
    nPx = 0
    nPy = 0
    nPz = 0 

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


    K3iso1 =  1336082.36755021
    K3iso2 =  -10866109.92757

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

    ax.set_ylim(-1E7,1E7)
    ax.set_xlim(0.26,0.37)
    ax.plot(Ecm1,np_QC_val)
    #ax.plot(Ecm_space,np_QC_val_spline)
    ax.axhline(y=0,color='black')
    ax.scatter(F3inv_poles,np_y_val,s=100,facecolor='white',edgecolor='red')
    for i in range(len(Elatt_CM)):
        ax.axvline(x=Elatt_CM[i],color='darkorange')
    plt.show()

test() 
#spectrum_checker_for_QC()