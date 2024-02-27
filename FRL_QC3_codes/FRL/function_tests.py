################################################################################
# Import relevant python modules
################################################################################
# Standard python modules
import numpy as np, sys, os
np.set_printoptions(precision=3)
pi = np.pi; sqrt=np.sqrt; LA=np.linalg

# Set up paths for loading QC modules
from pathlib import Path
cwd = Path(os.getcwd())
sys.path.insert(0,str(cwd/'base_code'))
sys.path.insert(0,str(cwd/'base_code/F3'))
sys.path.insert(0,str(cwd/'base_code/Kdf3'))

from scipy.special import erfi,erfc
# QC modules
import defns, group_theory_defns as GT, projections as proj
from F3 import qcot_fits, F3_mat #, F_mat, G_mat, K2i_mat,
from Kdf3 import K3main
import defns, F_mat, G_mat, K2i_mat
################################################################################
def nothing_val(q2):
    return 0.0
################################################################################
# Define 2+1 system parameters --- *MUST ALWAYS RESCALE SO THAT M1=1*
################################################################################
#M1, M2 = M12
M1,M2 = [100.,50.]  # The 3-pt. system masses are [M1,M1,M2], e.g. in MeV
M1,M2 = [1.,M2/M1]  # We always rescale by M1 to make everything dimensionless
M12 = [M1,M2]
parity = -1         # Particle parity (-1 for pseudoscalars)
L = 5               # Box size (in units of 1/M1)
nnP = [0,0,0]       # 3-pt. FV spatial momentum (integer-valued)
Ecm = 3.1              # Total 3-pt. energy in CM frame (in units of M1)
E = defns.E_to_Ecm(Ecm,L,nnP,rev=True) # Total 3-pt. energy in moving frame (in units of M1)
################################################################################
# Define K2^{-1} parameters
################################################################################
waves = 's'  # Partial waves used for dimers with flavor-1 spectators
              # (flavor-2 spectators only use s-wave)
a_1s = 0.15   # s-wave scattering length for spectator-flavor-1 channel
r_1s = 0.0    # s-wave effective range for spectator-flavor-1 channel
a_1p = 0.2    # p-wave scattering length for spectator-flavor-1 channel
a_2s = 0.1    # s-wave scattering length for spectator-flavor-2 channel
################################################################################
# Define K2^{-1} phase shift functions
################################################################################
f_qcot_1s = lambda q2: qcot_fits.qcot_fit_s(q2,[a_1s,r_1s],ERE=True)
#f_qcot_1p = lambda q2: qcot_fits.qcot_fit_p(q2,a_1p)

f_qcot_1p = lambda q2: nothing_val(q2)

f_qcot_1sp = [f_qcot_1s, f_qcot_1p]
#f_qcot_1sp = [f_qcot_1s]
f_qcot_2s = [lambda q2: qcot_fits.qcot_fit_s(q2,[a_2s],ERE=True)]
################################################################################
# Define Kdf3 parameters
################################################################################
K3iso = [200, 400]      # Isotropic term is K3iso[0] + K3iso[1]*\Delta
K3B_par = 400          # Parameter for Kdf3^B1 term
K3E_par = 300          # Parameter for Kdf3^E1 term
################################################################################
# Determine relevant flavor-1 and flavor-2 spectator momenta
################################################################################
nnk_list_1 = defns.list_nnk_nnP(E,L,nnP, Mijk=[M1,M1,M2])
nnk_list_2 = defns.list_nnk_nnP(E,L,nnP, Mijk=[M2,M1,M1])
print('flavor 1 spectators:\n', nnk_list_1)
print('flavor 2 spectators:\n', nnk_list_2)
################################################################################
# Compute desired QC matrices
################################################################################

spec_const = 2.0*pi/L 
px = 0.0*spec_const 
py = 0.0*spec_const 
pz = 0.0*spec_const
p_spec = [px, py, pz] 

kx = 0.0*spec_const 
ky = 0.0*spec_const 
kz = 0.0*spec_const
k_spec = [kx, ky, kz]

Px = 0.0*spec_const 
Py = 0.0*spec_const 
Pz = 0.0*spec_const 
total_P = [Px, Py, Pz]
IPV = 0
eta_i = 0.5
kvec = k_spec
Pvec = total_P 
Mijk = [M1, M1, M2]

K2i = K2i_mat.K2i_inv_k(Ecm,kvec,Pvec, f_qcot_2s, Mijk=Mijk, waves=waves, eta_i=eta_i,IPV=IPV)

nnklist = [0,0,0]
nnPlist = [0,0,0]
nnk = np.array(nnklist)
nnP = np.array(nnPlist)
F2 = F_mat.F_i_nnk(E,nnP,L,nnk, Mijk=Mijk, waves='s')
print(nnk)
print(nnP)
print(nnk - nnP)
print(LA.norm(nnk - nnP))
Gval = G_mat.G_ij(E, L, nnk, nnk,0,0,0,0,nnP, Mijk=Mijk, return_all=False)
print("%.20f" %K2i)

print("%.25f" %F2)

print("%.25f" %Gval)

print("erfi(-0.5)","%.25f" %erfi(-0.5))