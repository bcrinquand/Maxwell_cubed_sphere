# Import modules
import numpy as N
import matplotlib.pyplot as P
import matplotlib
import time
import scipy.integrate as spi
from scipy import interpolate
from skimage.measure import find_contours
from math import *
import sys
import h5py
from tqdm import tqdm

# Import my figure routines
import sys
sys.path.append('../')
sys.path.append('../transformations/')

from figure_module import *

# Import metric functions and corrdinate transformations
from vec_transformations_flip import *
from form_transformations_flip import *

# from gr_metric import *
from gr_metric_stretched import *

outdir = '/home/bcrinqua/GitHub/Maxwell_cubed_sphere/data_3d_gr/'

########
# Topology of the patches
########

n_patches = 6
# Connect patch indices and names
sphere = {0: "A", 1: "B", 2: "C", 3: "D", 4: "N", 5: "S"}

class Sphere:
    A = 0
    B = 1
    C = 2
    D = 3
    N = 4
    S = 5

patches = range(n_patches)

topology = N.zeros((6, 6), dtype = object)
topology[Sphere.A, Sphere.B] = 'xx'
topology[Sphere.A, Sphere.N] = 'yx'
topology[Sphere.B, Sphere.C] = 'xy'
topology[Sphere.B, Sphere.N] = 'yy'
topology[Sphere.C, Sphere.S] = 'xy'
topology[Sphere.C, Sphere.D] = 'yy'
topology[Sphere.D, Sphere.S] = 'xx'
topology[Sphere.D, Sphere.A] = 'yx'
topology[Sphere.N, Sphere.C] = 'xx'
topology[Sphere.N, Sphere.D] = 'yx'
topology[Sphere.S, Sphere.B] = 'xy'
topology[Sphere.S, Sphere.A] = 'yy'

# Gets indices where topology is nonzero
index_row, index_col = N.nonzero(topology)[0], N.nonzero(topology)[1]
n_zeros = N.size(index_row) # Total number of interactions (12)

# Parameters
cfl = 0.4
Nl = 64
Nxi = 24
Neta = 24

# Spin parameter
a = 0.8
rh = 1.0 + N.sqrt(1.0 - a * a)

Nxi_int   = Nxi + 1  # Number of integer points
Nxi_half  = Nxi + 2  # Number of half-step points
Neta_int  = Neta + 1 # Number of integer points
Neta_half = Neta + 2 # NUmber of half-step points
Nl_int    = Nl + 1   # Number of integer points
Nl_half   = Nl + 2   # Number of half-step points

r_min, r_max     = 0.9 * rh, 2.0 * rh
l_min, l_max     = N.log(r_min), N.log(r_max)
xi_min, xi_max   = - N.pi / 4.0, N.pi / 4.0
eta_min, eta_max = - N.pi / 4.0, N.pi / 4.0

dl   = (l_max - l_min) / Nl
dxi  = (xi_max - xi_min) / Nxi
deta = (eta_max - eta_min) / Neta

# Define grids
l_int  = N.linspace(l_min, l_max, Nl_int)
l_half  = N.zeros(Nl_half)
l_half[0] = l_int[0]
l_half[-1] = l_int[-1]
l_half[1:-1] = l_int[:-1] + 0.5 * dl
r_int = N.exp(l_int)
r_half = N.exp(l_half)

xi_int  = N.linspace(xi_min, xi_max, Nxi_int)
xi_half  = N.zeros(Nxi_half)
xi_half[0] = xi_int[0]
xi_half[-1] = xi_int[-1]
xi_half[1:-1] = xi_int[:-1] + 0.5 * dxi

eta_int  = N.linspace(eta_min, eta_max, Neta_int)
eta_half  = N.zeros(Neta_half)
eta_half[0] = eta_int[0]
eta_half[-1] = eta_int[-1]
eta_half[1:-1] = eta_int[:-1] + 0.5 * deta

yBr_grid, xBr_grid = N.meshgrid(eta_half, xi_half)
yE1_grid, xE1_grid = N.meshgrid(eta_int, xi_half)
yE2_grid, xE2_grid = N.meshgrid(eta_half, xi_int)

# Physical fields
Bru = N.zeros((n_patches, Nl_int,  Nxi_half, Neta_half))
B1u = N.zeros((n_patches, Nl_half, Nxi_int,  Neta_half))
B2u = N.zeros((n_patches, Nl_half, Nxi_half, Neta_int))
Brd = N.zeros((n_patches, Nl_int,  Nxi_half, Neta_half))
B1d = N.zeros((n_patches, Nl_half, Nxi_int,  Neta_half))
B2d = N.zeros((n_patches, Nl_half, Nxi_half, Neta_int))

Dru = N.zeros((n_patches, Nl_half, Nxi_int, Neta_int))
D1u = N.zeros((n_patches, Nl_int,  Nxi_half, Neta_int))
D2u = N.zeros((n_patches, Nl_int,  Nxi_int,  Neta_half))
Drd = N.zeros((n_patches, Nl_half, Nxi_int, Neta_int))
D1d = N.zeros((n_patches, Nl_int,  Nxi_half, Neta_int))
D2d = N.zeros((n_patches, Nl_int,  Nxi_int,  Neta_half))

# Shifted by one time step
Bru0 = N.zeros((n_patches, Nl_int,  Nxi_half, Neta_half))
B1u0 = N.zeros((n_patches, Nl_half, Nxi_int, Neta_half))
B2u0 = N.zeros((n_patches, Nl_half, Nxi_half,  Neta_int))
Bru1 = N.zeros((n_patches, Nl_int,  Nxi_half, Neta_half))
B1u1 = N.zeros((n_patches, Nl_half, Nxi_int, Neta_half))
B2u1 = N.zeros((n_patches, Nl_half, Nxi_half,  Neta_int))

Dru0 = N.zeros((n_patches, Nl_half, Nxi_int, Neta_int))
D1u0 = N.zeros((n_patches, Nl_int,  Nxi_half, Neta_int))
D2u0 = N.zeros((n_patches, Nl_int,  Nxi_int,  Neta_half))
Dru1 = N.zeros((n_patches, Nl_half, Nxi_int, Neta_int))
D1u1 = N.zeros((n_patches, Nl_int,  Nxi_half, Neta_int))
D2u1 = N.zeros((n_patches, Nl_int,  Nxi_int,  Neta_half))

# Auxiliary fields and gradients
Erd = N.zeros((n_patches, Nl_half, Nxi_int, Neta_int))
E1d = N.zeros((n_patches, Nl_int,  Nxi_half, Neta_int))
E2d = N.zeros((n_patches, Nl_int,  Nxi_int,  Neta_half))
Eru = N.zeros((n_patches, Nl_half, Nxi_int, Neta_int))
E1u = N.zeros((n_patches, Nl_int,  Nxi_half, Neta_int))
E2u = N.zeros((n_patches, Nl_int,  Nxi_int,  Neta_half))

Hrd = N.zeros((n_patches, Nl_int,  Nxi_half, Neta_half))
H1d = N.zeros((n_patches, Nl_half, Nxi_int, Neta_half))
H2d = N.zeros((n_patches, Nl_half, Nxi_half,  Neta_int))
Hru = N.zeros((n_patches, Nl_int,  Nxi_half, Neta_half))
H1u = N.zeros((n_patches, Nl_half, Nxi_int, Neta_half))
H2u = N.zeros((n_patches, Nl_half, Nxi_half,  Neta_int))

dE1d2 = N.zeros((n_patches, Nl_int,  Nxi_half, Neta_half))
dE2d1 = N.zeros((n_patches, Nl_int,  Nxi_half, Neta_half))
dErd1 = N.zeros((n_patches, Nl_half, Nxi_half,  Neta_int))
dErd2 = N.zeros((n_patches, Nl_half, Nxi_int, Neta_half))
dE1dr = N.zeros((n_patches, Nl_half, Nxi_half, Neta_int))
dE2dr = N.zeros((n_patches, Nl_half, Nxi_int, Neta_half))

dHrd1 = N.zeros((n_patches, Nl_int,  Nxi_int,  Neta_half))
dHrd2 = N.zeros((n_patches, Nl_int,  Nxi_half, Neta_int))
dH1d2 = N.zeros((n_patches, Nl_half, Nxi_int, Neta_int))
dH2d1 = N.zeros((n_patches, Nl_half, Nxi_int, Neta_int))
dH1dr = N.zeros((n_patches, Nl_int,  Nxi_int, Neta_half))
dH2dr = N.zeros((n_patches, Nl_int,  Nxi_half, Neta_int))

# Interface terms
diff_Bru = N.zeros((n_patches, Nl_int,  Nxi_half, Neta_half))
diff_B1u = N.zeros((n_patches, Nl_half, Nxi_int, Neta_half))
diff_B2u = N.zeros((n_patches, Nl_half, Nxi_half, Neta_int))
diff_Dru = N.zeros((n_patches, Nl_half, Nxi_int, Neta_int))
diff_D1u = N.zeros((n_patches, Nl_int,  Nxi_half, Neta_int))
diff_D2u = N.zeros((n_patches, Nl_int,  Nxi_int, Neta_half))

# Initial magnetic field
INBr = N.zeros((n_patches, Nl_int,  Nxi_half, Neta_half))
INB1 = N.zeros((n_patches, Nl_half, Nxi_int, Neta_half))
INB2 = N.zeros((n_patches, Nl_half, Nxi_half,  Neta_int))

########
# Dump HDF5 output
########

def WriteFieldHDF5(it, field):

    outvec = (globals()[field])
    h5f = h5py.File(outdir + field + '_' + str(it).rjust(5, '0') + '.h5', 'w')

    for patch in range(n_patches):
        h5f.create_dataset(field + str(patch), data=outvec[patch, :, :, :])

    h5f.close()

def WriteAllFieldsHDF5(idump):

    WriteFieldHDF5(idump, "Bru")
    WriteFieldHDF5(idump, "B1u")
    WriteFieldHDF5(idump, "B2u")
    WriteFieldHDF5(idump, "Dru")
    WriteFieldHDF5(idump, "D1u")
    WriteFieldHDF5(idump, "D2u")

def WriteCoordsHDF5():

    h5f = h5py.File(outdir+'grid.h5', 'w')

    h5f.create_dataset('r_int', data = r_int)
    h5f.create_dataset('r_half', data = r_half)
    h5f.create_dataset('xi_int', data = xi_int)
    h5f.create_dataset('eta_int', data = eta_int)
    h5f.create_dataset('xi_half', data = xi_half)
    h5f.create_dataset('eta_half', data = eta_half)
    
    h5f.close()

########
# Define metric tensor
########

hrrd_Br       = N.empty((n_patches, Nl_int, Nxi_half, Neta_half))
hr1d_Br       = N.empty((n_patches, Nl_int, Nxi_half, Neta_half))
hr2d_Br       = N.empty((n_patches, Nl_int, Nxi_half, Neta_half))
h11d_Br       = N.empty((n_patches, Nl_int, Nxi_half, Neta_half))
h12d_Br       = N.empty((n_patches, Nl_int, Nxi_half, Neta_half))
h22d_Br       = N.empty((n_patches, Nl_int, Nxi_half, Neta_half))
alpha_Br      = N.empty((n_patches, Nl_int, Nxi_half, Neta_half))
beta_Br       = N.empty((n_patches, Nl_int, Nxi_half, Neta_half))
betard_Br     = N.empty((n_patches, Nl_int, Nxi_half, Neta_half))
beta2d_Br     = N.empty((n_patches, Nl_int, Nxi_half, Neta_half))
sqrt_det_h_Br = N.empty((n_patches, Nl_int, Nxi_half, Neta_half))
hrru_Br       = N.empty((n_patches, Nl_int, Nxi_half, Neta_half))
hr1u_Br       = N.empty((n_patches, Nl_int, Nxi_half, Neta_half))
hr2u_Br       = N.empty((n_patches, Nl_int, Nxi_half, Neta_half))
h11u_Br       = N.empty((n_patches, Nl_int, Nxi_half, Neta_half))
h12u_Br       = N.empty((n_patches, Nl_int, Nxi_half, Neta_half))
h22u_Br       = N.empty((n_patches, Nl_int, Nxi_half, Neta_half))

hrrd_B1       = N.empty((n_patches, Nl_half, Nxi_int, Neta_half))
hr1d_B1       = N.empty((n_patches, Nl_half, Nxi_int, Neta_half))
hr2d_B1       = N.empty((n_patches, Nl_half, Nxi_int, Neta_half))
h11d_B1       = N.empty((n_patches, Nl_half, Nxi_int, Neta_half))
h12d_B1       = N.empty((n_patches, Nl_half, Nxi_int, Neta_half))
h22d_B1       = N.empty((n_patches, Nl_half, Nxi_int, Neta_half))
alpha_B1      = N.empty((n_patches, Nl_half, Nxi_int, Neta_half))
beta_B1       = N.empty((n_patches, Nl_half, Nxi_int, Neta_half))
betard_B1     = N.empty((n_patches, Nl_half, Nxi_int, Neta_half))
beta2d_B1     = N.empty((n_patches, Nl_half, Nxi_int, Neta_half))
sqrt_det_h_B1 = N.empty((n_patches, Nl_half, Nxi_int, Neta_half))
hrru_B1       = N.empty((n_patches, Nl_half, Nxi_int, Neta_half))
hr1u_B1       = N.empty((n_patches, Nl_half, Nxi_int, Neta_half))
hr2u_B1       = N.empty((n_patches, Nl_half, Nxi_int, Neta_half))
h11u_B1       = N.empty((n_patches, Nl_half, Nxi_int, Neta_half))
h12u_B1       = N.empty((n_patches, Nl_half, Nxi_int, Neta_half))
h22u_B1       = N.empty((n_patches, Nl_half, Nxi_int, Neta_half))

hrrd_B2       = N.empty((n_patches, Nl_half, Nxi_half, Neta_int))
hr1d_B2       = N.empty((n_patches, Nl_half, Nxi_half, Neta_int))
hr2d_B2       = N.empty((n_patches, Nl_half, Nxi_half, Neta_int))
h11d_B2       = N.empty((n_patches, Nl_half, Nxi_half, Neta_int))
h12d_B2       = N.empty((n_patches, Nl_half, Nxi_half, Neta_int))
h22d_B2       = N.empty((n_patches, Nl_half, Nxi_half, Neta_int))
alpha_B2      = N.empty((n_patches, Nl_half, Nxi_half, Neta_int))
beta_B2       = N.empty((n_patches, Nl_half, Nxi_half, Neta_int))
betard_B2     = N.empty((n_patches, Nl_half, Nxi_half, Neta_int))
beta2d_B2     = N.empty((n_patches, Nl_half, Nxi_half, Neta_int))
sqrt_det_h_B2 = N.empty((n_patches, Nl_half, Nxi_half, Neta_int))
hrru_B2       = N.empty((n_patches, Nl_half, Nxi_half, Neta_int))
hr1u_B2       = N.empty((n_patches, Nl_half, Nxi_half, Neta_int))
hr2u_B2       = N.empty((n_patches, Nl_half, Nxi_half, Neta_int))
h11u_B2       = N.empty((n_patches, Nl_half, Nxi_half, Neta_int))
h12u_B2       = N.empty((n_patches, Nl_half, Nxi_half, Neta_int))
h22u_B2       = N.empty((n_patches, Nl_half, Nxi_half, Neta_int))

hrrd_Dr       = N.empty((n_patches, Nl_half, Nxi_int, Neta_int))
hr1d_Dr       = N.empty((n_patches, Nl_half, Nxi_int, Neta_int))
hr2d_Dr       = N.empty((n_patches, Nl_half, Nxi_int, Neta_int))
h11d_Dr       = N.empty((n_patches, Nl_half, Nxi_int, Neta_int))
h12d_Dr       = N.empty((n_patches, Nl_half, Nxi_int, Neta_int))
h22d_Dr       = N.empty((n_patches, Nl_half, Nxi_int, Neta_int))
alpha_Dr      = N.empty((n_patches, Nl_half, Nxi_int, Neta_int))
beta_Dr       = N.empty((n_patches, Nl_half, Nxi_int, Neta_int))
betard_Dr     = N.empty((n_patches, Nl_half, Nxi_int, Neta_int))
beta2d_Dr     = N.empty((n_patches, Nl_half, Nxi_int, Neta_int))
sqrt_det_h_Dr = N.empty((n_patches, Nl_half, Nxi_int, Neta_int))
hrru_Dr       = N.empty((n_patches, Nl_half, Nxi_int, Neta_int))
hr1u_Dr       = N.empty((n_patches, Nl_half, Nxi_int, Neta_int))
hr2u_Dr       = N.empty((n_patches, Nl_half, Nxi_int, Neta_int))
h11u_Dr       = N.empty((n_patches, Nl_half, Nxi_int, Neta_int))
h12u_Dr       = N.empty((n_patches, Nl_half, Nxi_int, Neta_int))
h22u_Dr       = N.empty((n_patches, Nl_half, Nxi_int, Neta_int))

hrrd_D1       = N.empty((n_patches, Nl_int, Nxi_half, Neta_int))
hr1d_D1       = N.empty((n_patches, Nl_int, Nxi_half, Neta_int))
hr2d_D1       = N.empty((n_patches, Nl_int, Nxi_half, Neta_int))
h11d_D1       = N.empty((n_patches, Nl_int, Nxi_half, Neta_int))
h12d_D1       = N.empty((n_patches, Nl_int, Nxi_half, Neta_int))
h22d_D1       = N.empty((n_patches, Nl_int, Nxi_half, Neta_int))
alpha_D1      = N.empty((n_patches, Nl_int, Nxi_half, Neta_int))
beta_D1       = N.empty((n_patches, Nl_int, Nxi_half, Neta_int))
betard_D1     = N.empty((n_patches, Nl_int, Nxi_half, Neta_int))
beta2d_D1     = N.empty((n_patches, Nl_int, Nxi_half, Neta_int))
sqrt_det_h_D1 = N.empty((n_patches, Nl_int, Nxi_half, Neta_int))
hrru_D1       = N.empty((n_patches, Nl_int, Nxi_half, Neta_int))
hr1u_D1       = N.empty((n_patches, Nl_int, Nxi_half, Neta_int))
hr2u_D1       = N.empty((n_patches, Nl_int, Nxi_half, Neta_int))
h11u_D1       = N.empty((n_patches, Nl_int, Nxi_half, Neta_int))
h12u_D1       = N.empty((n_patches, Nl_int, Nxi_half, Neta_int))
h22u_D1       = N.empty((n_patches, Nl_int, Nxi_half, Neta_int))

hrrd_D2       = N.empty((n_patches, Nl_int, Nxi_int, Neta_half))
hr1d_D2       = N.empty((n_patches, Nl_int, Nxi_int, Neta_half))
hr2d_D2       = N.empty((n_patches, Nl_int, Nxi_int, Neta_half))
h11d_D2       = N.empty((n_patches, Nl_int, Nxi_int, Neta_half))
h12d_D2       = N.empty((n_patches, Nl_int, Nxi_int, Neta_half))
h22d_D2       = N.empty((n_patches, Nl_int, Nxi_int, Neta_half))
alpha_D2      = N.empty((n_patches, Nl_int, Nxi_int, Neta_half))
beta_D2       = N.empty((n_patches, Nl_int, Nxi_int, Neta_half))
betard_D2     = N.empty((n_patches, Nl_int, Nxi_int, Neta_half))
beta2d_D2     = N.empty((n_patches, Nl_int, Nxi_int, Neta_half))
sqrt_det_h_D2 = N.empty((n_patches, Nl_int, Nxi_int, Neta_half))
hrru_D2       = N.empty((n_patches, Nl_int, Nxi_int, Neta_half))
hr1u_D2       = N.empty((n_patches, Nl_int, Nxi_int, Neta_half))
hr2u_D2       = N.empty((n_patches, Nl_int, Nxi_int, Neta_half))
h11u_D2       = N.empty((n_patches, Nl_int, Nxi_int, Neta_half))
h12u_D2       = N.empty((n_patches, Nl_int, Nxi_int, Neta_half))
h22u_D2       = N.empty((n_patches, Nl_int, Nxi_int, Neta_half))

for p in range(n_patches):
    print(p)
    for i in range(Nxi_int):
        for j in range(Neta_int):

            l0 = l_half[:]
            xi0 = xi_int[i]
            eta0 = eta_int[j]
            h11d_Dr[p, :, i, j] = g11d(p, l0, xi0, eta0, a)
            h22d_Dr[p, :, i, j] = g22d(p, l0, xi0, eta0, a)
            h12d_Dr[p, :, i, j] = g12d(p, l0, xi0, eta0, a)
            hrrd_Dr[p, :, i, j] = glld(p, l0, xi0, eta0, a)
            hr1d_Dr[p, :, i, j] = gl1d(p, l0, xi0, eta0, a)
            hr2d_Dr[p, :, i, j] = gl2d(p, l0, xi0, eta0, a)
            alpha_Dr[p, :, i, j]=  alphas(p, l0, xi0, eta0, a)
            beta_Dr[p, :, i, j] =  betalu(p, l0, xi0, eta0, a)
            betard_Dr[p, :, i, j]=  betald(p, l0, xi0, eta0, a)
            beta2d_Dr[p, :, i, j] =  beta2d(p, l0, xi0, eta0, a)
            
            metric = N.array([[hrrd_Dr[p, :, i, j], hr1d_Dr[p, :, i, j], hr2d_Dr[p, :, i, j]], \
                              [hr1d_Dr[p, :, i, j], h11d_Dr[p, :, i, j], h12d_Dr[p, :, i, j]], \
                              [hr2d_Dr[p, :, i, j], h12d_Dr[p, :, i, j], h22d_Dr[p, :, i, j]]])
            sqrt_det_h_Dr[p, :, i, j] = N.sqrt(N.linalg.det(N.moveaxis(metric, 2, 0)))

            inv_metric = N.linalg.inv(N.moveaxis(metric, 2, 0))
            hrru_Dr[p, :, i, j] = inv_metric[:, 0, 0]
            hr1u_Dr[p, :, i, j] = inv_metric[:, 0, 1]
            hr2u_Dr[p, :, i, j] = inv_metric[:, 0, 2]
            h11u_Dr[p, :, i, j] = inv_metric[:, 1, 1]
            h12u_Dr[p, :, i, j] = inv_metric[:, 1, 2]
            h22u_Dr[p, :, i, j] = inv_metric[:, 2, 2]

    for i in range(Nxi_half):
        for j in range(Neta_int):

            l0 = l_int[:]
            xi0 = xi_half[i]
            eta0 = eta_int[j]
            h11d_D1[p, :, i, j] = g11d(p, l0, xi0, eta0, a)
            h22d_D1[p, :, i, j] = g22d(p, l0, xi0, eta0, a)
            h12d_D1[p, :, i, j] = g12d(p, l0, xi0, eta0, a)
            hrrd_D1[p, :, i, j] = glld(p, l0, xi0, eta0, a)
            hr1d_D1[p, :, i, j] = gl1d(p, l0, xi0, eta0, a)
            hr2d_D1[p, :, i, j] = gl2d(p, l0, xi0, eta0, a)
            alpha_D1[p, :, i, j]=  alphas(p, l0, xi0, eta0, a)
            beta_D1[p, :, i, j] =  betalu(p, l0, xi0, eta0, a)
            betard_D1[p, :, i, j]=  betald(p, l0, xi0, eta0, a)
            beta2d_D1[p, :, i, j] =  beta2d(p, l0, xi0, eta0, a)

            metric = N.array([[hrrd_D1[p, :, i, j], hr1d_D1[p, :, i, j], hr2d_D1[p, :, i, j]], \
                              [hr1d_D1[p, :, i, j], h11d_D1[p, :, i, j], h12d_D1[p, :, i, j]], \
                              [hr2d_D1[p, :, i, j], h12d_D1[p, :, i, j], h22d_D1[p, :, i, j]]])
            sqrt_det_h_D1[p, :, i, j] = N.sqrt(N.linalg.det(N.moveaxis(metric, 2, 0)))

            inv_metric = N.linalg.inv(N.moveaxis(metric, 2, 0))
            hrru_D1[p, :, i, j] = inv_metric[:, 0, 0]
            hr1u_D1[p, :, i, j] = inv_metric[:, 0, 1]
            hr2u_D1[p, :, i, j] = inv_metric[:, 0, 2]
            h11u_D1[p, :, i, j] = inv_metric[:, 1, 1]
            h12u_D1[p, :, i, j] = inv_metric[:, 1, 2]
            h22u_D1[p, :, i, j] = inv_metric[:, 2, 2]

            l0 = l_half[:]
            xi0 = xi_half[i]
            eta0 = eta_int[j]
            h11d_B2[p, :, i, j] = g11d(p, l0, xi0, eta0, a)
            h22d_B2[p, :, i, j] = g22d(p, l0, xi0, eta0, a)
            h12d_B2[p, :, i, j] = g12d(p, l0, xi0, eta0, a)
            hrrd_B2[p, :, i, j] = glld(p, l0, xi0, eta0, a)
            hr1d_B2[p, :, i, j] = gl1d(p, l0, xi0, eta0, a)
            hr2d_B2[p, :, i, j] = gl2d(p, l0, xi0, eta0, a)
            alpha_B2[p, :, i, j]=  alphas(p, l0, xi0, eta0, a)
            beta_B2[p, :, i, j] =  betalu(p, l0, xi0, eta0, a)
            betard_B2[p, :, i, j]=  betald(p, l0, xi0, eta0, a)
            beta2d_B2[p, :, i, j] =  beta2d(p, l0, xi0, eta0, a)

            metric = N.array([[hrrd_B2[p, :, i, j], hr1d_B2[p, :, i, j], hr2d_B2[p, :, i, j]], \
                              [hr1d_B2[p, :, i, j], h11d_B2[p, :, i, j], h12d_B2[p, :, i, j]], \
                              [hr2d_B2[p, :, i, j], h12d_B2[p, :, i, j], h22d_B2[p, :, i, j]]])
            sqrt_det_h_B2[p, :, i, j] = N.sqrt(N.linalg.det(N.moveaxis(metric, 2, 0)))

            inv_metric = N.linalg.inv(N.moveaxis(metric, 2, 0))
            hrru_B2[p, :, i, j] = inv_metric[:, 0, 0]
            hr1u_B2[p, :, i, j] = inv_metric[:, 0, 1]
            hr2u_B2[p, :, i, j] = inv_metric[:, 0, 2]
            h11u_B2[p, :, i, j] = inv_metric[:, 1, 1]
            h12u_B2[p, :, i, j] = inv_metric[:, 1, 2]
            h22u_B2[p, :, i, j] = inv_metric[:, 2, 2]

    for i in range(Nxi_int):
        for j in range(Neta_half):

            l0 = l_int[:]
            xi0 = xi_int[i]
            eta0 = eta_half[j]
            h11d_D2[p, :, i, j] = g11d(p, l0, xi0, eta0, a)
            h22d_D2[p, :, i, j] = g22d(p, l0, xi0, eta0, a)
            h12d_D2[p, :, i, j] = g12d(p, l0, xi0, eta0, a)
            hrrd_D2[p, :, i, j] = glld(p, l0, xi0, eta0, a)
            hr1d_D2[p, :, i, j] = gl1d(p, l0, xi0, eta0, a)
            hr2d_D2[p, :, i, j] = gl2d(p, l0, xi0, eta0, a)
            alpha_D2[p, :, i, j]=  alphas(p, l0, xi0, eta0, a)
            beta_D2[p, :, i, j] =  betalu(p, l0, xi0, eta0, a)
            betard_D2[p, :, i, j]=  betald(p, l0, xi0, eta0, a)
            beta2d_D2[p, :, i, j] =  beta2d(p, l0, xi0, eta0, a)

            metric = N.array([[hrrd_D2[p, :, i, j], hr1d_D2[p, :, i, j], hr2d_D2[p, :, i, j]], \
                              [hr1d_D2[p, :, i, j], h11d_D2[p, :, i, j], h12d_D2[p, :, i, j]], \
                              [hr2d_D2[p, :, i, j], h12d_D2[p, :, i, j], h22d_D2[p, :, i, j]]])
            sqrt_det_h_D2[p, :, i, j] = N.sqrt(N.linalg.det(N.moveaxis(metric, 2, 0)))

            inv_metric = N.linalg.inv(N.moveaxis(metric, 2, 0))
            hrru_D2[p, :, i, j] = inv_metric[:, 0, 0]
            hr1u_D2[p, :, i, j] = inv_metric[:, 0, 1]
            hr2u_D2[p, :, i, j] = inv_metric[:, 0, 2]
            h11u_D2[p, :, i, j] = inv_metric[:, 1, 1]
            h12u_D2[p, :, i, j] = inv_metric[:, 1, 2]
            h22u_D2[p, :, i, j] = inv_metric[:, 2, 2]

            l0 = l_half[:]
            xi0 = xi_int[i]
            eta0 = eta_half[j]
            h11d_B1[p, :, i, j] = g11d(p, l0, xi0, eta0, a)
            h22d_B1[p, :, i, j] = g22d(p, l0, xi0, eta0, a)
            h12d_B1[p, :, i, j] = g12d(p, l0, xi0, eta0, a)
            hrrd_B1[p, :, i, j] = glld(p, l0, xi0, eta0, a)
            hr1d_B1[p, :, i, j] = gl1d(p, l0, xi0, eta0, a)
            hr2d_B1[p, :, i, j] = gl2d(p, l0, xi0, eta0, a)
            alpha_B1[p, :, i, j]=  alphas(p, l0, xi0, eta0, a)
            beta_B1[p, :, i, j] =  betalu(p, l0, xi0, eta0, a)
            betard_B1[p, :, i, j]=  betald(p, l0, xi0, eta0, a)
            beta2d_B1[p, :, i, j] =  beta2d(p, l0, xi0, eta0, a)

            metric = N.array([[hrrd_B1[p, :, i, j], hr1d_B1[p, :, i, j], hr2d_B1[p, :, i, j]], \
                              [hr1d_B1[p, :, i, j], h11d_B1[p, :, i, j], h12d_B1[p, :, i, j]], \
                              [hr2d_B1[p, :, i, j], h12d_B1[p, :, i, j], h22d_B1[p, :, i, j]]])
            sqrt_det_h_B1[p, :, i, j] = N.sqrt(N.linalg.det(N.moveaxis(metric, 2, 0)))

            inv_metric = N.linalg.inv(N.moveaxis(metric, 2, 0))
            hrru_B1[p, :, i, j] = inv_metric[:, 0, 0]
            hr1u_B1[p, :, i, j] = inv_metric[:, 0, 1]
            hr2u_B1[p, :, i, j] = inv_metric[:, 0, 2]
            h11u_B1[p, :, i, j] = inv_metric[:, 1, 1]
            h12u_B1[p, :, i, j] = inv_metric[:, 1, 2]
            h22u_B1[p, :, i, j] = inv_metric[:, 2, 2]

    for i in range(Nxi_half):
        for j in range(Neta_half):

            l0 = l_int[:]
            xi0 = xi_half[i]
            eta0 = eta_half[j]
            h11d_Br[p, :, i, j] = g11d(p, l0, xi0, eta0, a)
            h22d_Br[p, :, i, j] = g22d(p, l0, xi0, eta0, a)
            h12d_Br[p, :, i, j] = g12d(p, l0, xi0, eta0, a)
            hrrd_Br[p, :, i, j] = glld(p, l0, xi0, eta0, a)
            hr1d_Br[p, :, i, j] = gl1d(p, l0, xi0, eta0, a)
            hr2d_Br[p, :, i, j] = gl2d(p, l0, xi0, eta0, a)
            alpha_Br[p, :, i, j]=  alphas(p, l0, xi0, eta0, a)
            beta_Br[p, :, i, j] =  betalu(p, l0, xi0, eta0, a)
            betard_Br[p, :, i, j]=  betald(p, l0, xi0, eta0, a)
            beta2d_Br[p, :, i, j] =  beta2d(p, l0, xi0, eta0, a)

            metric = N.array([[hrrd_Br[p, :, i, j], hr1d_Br[p, :, i, j], hr2d_Br[p, :, i, j]], \
                              [hr1d_Br[p, :, i, j], h11d_Br[p, :, i, j], h12d_Br[p, :, i, j]], \
                              [hr2d_Br[p, :, i, j], h12d_Br[p, :, i, j], h22d_Br[p, :, i, j]]])
            sqrt_det_h_Br[p, :, i, j] = N.sqrt(N.linalg.det(N.moveaxis(metric, 2, 0)))

            inv_metric = N.linalg.inv(N.moveaxis(metric, 2, 0))
            hrru_Br[p, :, i, j] = inv_metric[:, 0, 0]
            hr1u_Br[p, :, i, j] = inv_metric[:, 0, 1]
            hr2u_Br[p, :, i, j] = inv_metric[:, 0, 2]
            h11u_Br[p, :, i, j] = inv_metric[:, 1, 1]
            h12u_Br[p, :, i, j] = inv_metric[:, 1, 2]
            h22u_Br[p, :, i, j] = inv_metric[:, 2, 2]

# Time step
# dt = cfl * N.min(1.0 / N.sqrt(1.0 / (dr * dr) + 1.0 / (r_max * r_max * dxi * dxi) + 1.0 / (r_max * r_max * deta * deta)))
# dt = cfl * N.min(1.0 / N.sqrt(hrru_Dr / (dl * dl) + h11u_Dr / (dxi * dxi) + h22u_Dr / (deta * deta))) # + 2.0 * h12u_Dr / (dxi * deta) + 2.0 * hr1u_Dr / (dr * dxi) + 2.0 * hr2u_Dr / (dr * deta)))

dt = cfl * N.min(1.0 / (beta_Dr / dl + alpha_Dr * N.sqrt(hrru_Dr / (dl * dl) + h11u_Dr / (dxi * dxi) + h22u_Dr / (deta * deta)))) # + 2.0 * h12u / (dxi * deta) + 2.0 * hl1u / (dr * dxi) + 2.0 * hl2u / (dr * deta)))

print("delta t = {}".format(dt))

########
# Generic coordinate transformation
########

from coord_transformations_flip import *

def transform_coords(patch0, patch1, xi0, eta0):
    fcoord0 = (globals()["coord_" + sphere[patch0] + "_to_sph"])
    fcoord1 = (globals()["coord_sph_to_" + sphere[patch1]])
    return fcoord1(*fcoord0(xi0, eta0))

########
# Generic vector transformation
########

from vec_transformations_flip import *

def transform_vect(patch0, patch1, xi0, eta0, vxi0, veta0):
    fcoord0 = (globals()["coord_" + sphere[patch0] + "_to_sph"])
    theta0, phi0 = fcoord0(xi0, eta0)
    fvec0 = (globals()["vec_" + sphere[patch0] + "_to_sph"])
    fvec1 = (globals()["vec_sph_to_" + sphere[patch1]])
    return fvec1(theta0, phi0, *fvec0(xi0, eta0, vxi0, veta0))

########
# Linear form transformations
########

from form_transformations_flip import *

def transform_form(patch0, patch1, xi0, eta0, vxi0, veta0):
    fcoord0 = (globals()["coord_" + sphere[patch0] + "_to_sph"])
    theta0, phi0 = fcoord0(xi0, eta0)
    fform0 = (globals()["form_" + sphere[patch0] + "_to_sph"])
    fform1 = (globals()["form_sph_to_" + sphere[patch1]])
    return fform1(theta0, phi0, *fform0(xi0, eta0, vxi0, veta0))

########
# Interpolation routine
########

def interp(arr_in, xA, xB, ax):
    f = interpolate.interp1d(xA, arr_in, axis = ax, kind='linear', fill_value=(0,0), bounds_error=False)
    return f(xB)

# def interp(arr_in, xA, xB):
#     return N.interp(xB, xA, arr_in)

def interp_int_to_half(tab_in):
    a, b = tab_in.shape
    tab_out = N.zeros((Nl_half, b))
    tab_out[0, :]    = tab_in[0, :]
    tab_out[-1, :]   = tab_in[-1, :]
    tab_out[1:-1, :] = 0.5 * (tab_in[1:, :] + N.roll(tab_in, 1, axis = 0)[1:, :])
    return tab_out

def interp_half_to_int(tab_in):
    a, b = tab_in.shape
    tab_out = N.zeros((Nl_int, b))
    tab_out[0, :]    = tab_in[0, :]
    tab_out[-1, :]   = tab_in[-1, :]
    tab_out[1:-1, :] = 0.5 * (tab_in[1:-2, :] + N.roll(tab_in, -1, axis = 0)[1:-2, :])
    return tab_out

########
# Pushers
########

P_int_2 = N.ones(Nxi_int)
P_int_2[0] = 0.5 
P_int_2[-1] = 0.5 

P_half_2 = N.ones(Nxi_half)
P_half_2[0] = 0.5 
P_half_2[1] = 0.25 
P_half_2[2] = 1.25 
P_half_2[-3] = 1.25 
P_half_2[-2] = 0.25 
P_half_2[-1] = 0.5 

def compute_diff_H(p):
    
    dHrd1[p, :, 0, :] = (- 0.5 * Hrd[p, :, 0, :] + 0.25 * Hrd[p, :, 1, :] + 0.25 * Hrd[p, :, 2, :]) / dxi / P_int_2[0]
    dHrd1[p, :, 1, :] = (- 0.5 * Hrd[p, :, 0, :] - 0.25 * Hrd[p, :, 1, :] + 0.75 * Hrd[p, :, 2, :]) / dxi / P_int_2[1]
    dHrd1[p, :, Nxi_int - 2, :] = (- 0.75 * Hrd[p, :, -3, :] + 0.25 * Hrd[p, :, -2, :] + 0.5 * Hrd[p, :, -1, :]) / dxi / P_int_2[Nxi_int - 2]
    dHrd1[p, :, Nxi_int - 1, :] = (- 0.25 * Hrd[p, :, -3, :] - 0.25 * Hrd[p, :, -2, :] + 0.5 * Hrd[p, :, -1, :]) / dxi / P_int_2[Nxi_int - 1]
    dHrd1[p, :, 2:(Nxi_int - 2), :] = (N.roll(Hrd, -1, axis = 2)[p, :, 2:(Nxi_int - 2), :] - Hrd[p, :, 2:(Nxi_int - 2), :]) / dxi

    dHrd2[p, :, :, 0] = (- 0.5 * Hrd[p, :, :, 0] + 0.25 * Hrd[p, :, :, 1] + 0.25 * Hrd[p, :, :, 2]) / deta / P_int_2[0]
    dHrd2[p, :, :, 1] = (- 0.5 * Hrd[p, :, :, 0] - 0.25 * Hrd[p, :, :, 1] + 0.75 * Hrd[p, :, :, 2]) / deta / P_int_2[1]
    dHrd2[p, :, :, Nxi_int - 2] = (- 0.75 * Hrd[p, :, :, -3] + 0.25 * Hrd[p, :, :, -2] + 0.5 * Hrd[p, :, :, -1]) / deta / P_int_2[Nxi_int - 2]
    dHrd2[p, :, :, Nxi_int - 1] = (- 0.25 * Hrd[p, :, :, -3] - 0.25 * Hrd[p, :, :, -2] + 0.5 * Hrd[p, :, :, -1]) / deta / P_int_2[Nxi_int - 1]
    dHrd2[p, :, :, 2:(Neta_int - 2)] = (N.roll(Hrd, -1, axis = 3)[p, :, :, 2:(Neta_int - 2)] - Hrd[p, :, :, 2:(Neta_int - 2)]) / deta

    dH1dr[p, 0, :, :] = (- 0.5 * H1d[p, 0, :, :] + 0.25 * H1d[p, 1, :, :] + 0.25 * H1d[p, 2, :, :]) / dl / P_int_2[0]
    dH1dr[p, 1, :, :] = (- 0.5 * H1d[p, 0, :, :] - 0.25 * H1d[p, 1, :, :] + 0.75 * H1d[p, 2, :, :]) / dl / P_int_2[1]
    dH1dr[p, Nl_int - 2, :, :] = (- 0.75 * H1d[p, -3, :, :] + 0.25 * H1d[p, -2, :, :] + 0.5 * H1d[p, -1, :, :]) / dl / P_int_2[Nxi_int - 2]
    dH1dr[p, Nl_int - 1, :, :] = (- 0.25 * H1d[p, -3, :, :] - 0.25 * H1d[p, -2, :, :] + 0.5 * H1d[p, -1, :, :]) / dl / P_int_2[Nxi_int - 1]
    dH1dr[p, 2:(Nl_int - 2), :, :] = (N.roll(H1d, -1, axis = 1)[p, 2:(Nl_int - 2), :, :] - H1d[p, 2:(Nl_int - 2), :, :]) / dl
    
    dH2dr[p, 0, :, :] = (- 0.5 * H2d[p, 0, :, :] + 0.25 * H2d[p, 1, :, :] + 0.25 * H2d[p, 2, :, :]) / dl / P_int_2[0]
    dH2dr[p, 1, :, :] = (- 0.5 * H2d[p, 0, :, :] - 0.25 * H2d[p, 1, :, :] + 0.75 * H2d[p, 2, :, :]) / dl / P_int_2[1]
    dH2dr[p, Nl_int - 2, :, :] = (- 0.75 * H2d[p, -3, :, :] + 0.25 * H2d[p, -2, :, :] + 0.5 * H2d[p, -1, :, :]) / dl / P_int_2[Nxi_int - 2]
    dH2dr[p, Nl_int - 1, :, :] = (- 0.25 * H2d[p, -3, :, :] - 0.25 * H2d[p, -2, :, :] + 0.5 * H2d[p, -1, :, :]) / dl / P_int_2[Nxi_int - 1]    
    dH2dr[p, 2:(Nl_int - 2), :, :] = (N.roll(H2d, -1, axis = 1)[p, 2:(Nl_int - 2), :, :] - H2d[p, 2:(Nl_int - 2), :, :]) / dl

    dH1d2[p, :, :, 0] = (- 0.5 * H1d[p, :, :, 0] + 0.25 * H1d[p, :, :, 1] + 0.25 * H1d[p, :, :, 2]) / deta / P_int_2[0]
    dH1d2[p, :, :, 1] = (- 0.5 * H1d[p, :, :, 0] - 0.25 * H1d[p, :, :, 1] + 0.75 * H1d[p, :, :, 2]) / deta / P_int_2[1]
    dH1d2[p, :, :, Nxi_int - 2] = (- 0.75 * H1d[p, :, :, -3] + 0.25 * H1d[p, :, :, -2] + 0.5 * H1d[p, :, :, -1]) / deta / P_int_2[Nxi_int - 2]
    dH1d2[p, :, :, Nxi_int - 1] = (- 0.25 * H1d[p, :, :, -3] - 0.25 * H1d[p, :, :, -2] + 0.5 * H1d[p, :, :, -1]) / deta / P_int_2[Nxi_int - 1]
    dH1d2[p, :, :, 2:(Neta_int - 2)] = (N.roll(H1d, -1, axis = 3)[p, :, :, 2:(Neta_int - 2)] - H1d[p, :, :, 2:(Neta_int - 2)]) / deta

    dH2d1[p, :, 0, :] = (- 0.5 * H2d[p, :, 0, :] + 0.25 * H2d[p, :, 1, :] + 0.25 * H2d[p, :, 2, :]) / dxi / P_int_2[0]
    dH2d1[p, :, 1, :] = (- 0.5 * H2d[p, :, 0, :] - 0.25 * H2d[p, :, 1, :] + 0.75 * H2d[p, :, 2, :]) / dxi / P_int_2[1]
    dH2d1[p, :, Nxi_int - 2, :] = (- 0.75 * H2d[p, :, -3, :] + 0.25 * H2d[p, :, -2, :] + 0.5 * H2d[p, :, -1, :]) / dxi / P_int_2[Nxi_int - 2]
    dH2d1[p, :, Nxi_int - 1, :] = (- 0.25 * H2d[p, :, -3, :] - 0.25 * H2d[p, :, -2, :] + 0.5 * H2d[p, :, -1, :]) / dxi / P_int_2[Nxi_int - 1]
    dH2d1[p, :, 2:(Nxi_int - 2), :] = (N.roll(H2d, -1, axis = 2)[p, :, 2:(Nxi_int - 2), :] - H2d[p, :, 2:(Nxi_int - 2), :]) / dxi

def compute_diff_E(p):

    dE2d1[p, :, 0, :] = (- 0.50 * E2d[p, :, 0, :] + 0.50 * E2d[p, :, 1, :]) / dxi / P_half_2[0]
    dE2d1[p, :, 1, :] = (- 0.25 * E2d[p, :, 0, :] + 0.25 * E2d[p, :, 1, :]) / dxi / P_half_2[1]
    dE2d1[p, :, 2, :] = (- 0.25 * E2d[p, :, 0, :] - 0.75 * E2d[p, :, 1, :] + E2d[p, :, 2, :]) / dxi / P_half_2[2]
    dE2d1[p, :, Nxi_half - 3, :] = (- E2d[p, :, -3, :] + 0.75 * E2d[p, :, -2, :] + 0.25 * E2d[p, :, -1, :]) / dxi / P_half_2[Nxi_half - 3]
    dE2d1[p, :, Nxi_half - 2, :] = (- 0.25 * E2d[p, :, -2, :] + 0.25 * E2d[p, :, -1, :]) / dxi / P_half_2[Nxi_half - 2]
    dE2d1[p, :, Nxi_half - 1, :] = (- 0.5 * E2d[p, :, -2, :] + 0.5 * E2d[p, :, -1, :]) / dxi / P_half_2[Nxi_half - 1]
    dE2d1[p, :, 3:(Nxi_half - 3), :] = (E2d[p, :, 3:(Nxi_half - 3), :] - N.roll(E2d, 1, axis = 2)[p, :, 3:(Nxi_half - 3), :]) / dxi

    dE1d2[p, :, :, 0] = (- 0.50 * E1d[p, :, :, 0] + 0.50 * E1d[p, :, :, 1]) / deta / P_half_2[0]
    dE1d2[p, :, :, 1] = (- 0.25 * E1d[p, :, :, 0] + 0.25 * E1d[p, :, :, 1]) / deta / P_half_2[1]
    dE1d2[p, :, :, 2] = (- 0.25 * E1d[p, :, :, 0] - 0.75 * E1d[p, :, :, 1] + E1d[p, :, :, 2]) / deta / P_half_2[2]
    dE1d2[p, :, :, Neta_half - 3] = (- E1d[p, :, :, -3] + 0.75 * E1d[p, :, :, -2] + 0.25 * E1d[p, :, :, -1]) / deta / P_half_2[Nxi_half - 3]
    dE1d2[p, :, :, Neta_half - 2] = (- 0.25 * E1d[p, :, :, -2] + 0.25 * E1d[p, :, :, -1]) / deta / P_half_2[Nxi_half - 2]
    dE1d2[p, :, :, Neta_half - 1] = (- 0.50 * E1d[p, :, :, -2] + 0.50 * E1d[p, :, :, -1]) / deta / P_half_2[Nxi_half - 1]
    dE1d2[p, :, :, 3:(Neta_half - 3)] = (E1d[p, :, :, 3:(Neta_half - 3)] - N.roll(E1d, 1, axis = 3)[p, :, :, 3:(Neta_half - 3)]) / deta

    dE1dr[p, 0, :, :] = (- 0.50 * E1d[p, 0, :, :] + 0.50 * E1d[p, 1, :, :]) / dl / P_half_2[0]
    dE1dr[p, 1, :, :] = (- 0.25 * E1d[p, 0, :, :] + 0.25 * E1d[p, 1, :, :]) / dl / P_half_2[1]
    dE1dr[p, 2, :, :] = (- 0.25 * E1d[p, 0, :, :] - 0.75 * E1d[p, 1, :, :] + E1d[p, 2, :, :]) / dl / P_half_2[2]
    dE1dr[p, Nl_half - 3, :, :] = (- E1d[p, -3, :, :] + 0.75 * E1d[p, -2, :, :] + 0.25 * E1d[p, -1, :, :]) / dl / P_half_2[Nxi_half - 3]
    dE1dr[p, Nl_half - 2, :, :] = (- 0.25 * E1d[p, -2, :, :] + 0.25 * E1d[p, -1, :, :]) / dl / P_half_2[Nxi_half - 2]
    dE1dr[p, Nl_half - 1, :, :] = (- 0.50 * E1d[p, -2, :, :] + 0.50 * E1d[p, -1, :, :]) / dl / P_half_2[Nxi_half - 1]
    dE1dr[p, 3:(Nl_half - 3), :, :] = (E1d[p, 3:(Nl_half - 3), :, :] - N.roll(E1d, 1, axis = 1)[p, 3:(Nl_half - 3), :, :]) / dl

    dE2dr[p, 0, :, :] = (- 0.50 * E2d[p, 0, :, :] + 0.50 * E2d[p, 1, :, :]) / dl / P_half_2[0]
    dE2dr[p, 1, :, :] = (- 0.25 * E2d[p, 0, :, :] + 0.25 * E2d[p, 1, :, :]) / dl / P_half_2[1]
    dE2dr[p, 2, :, :] = (- 0.25 * E2d[p, 0, :, :] - 0.75 * E2d[p, 1, :, :] + E2d[p, 2, :, :]) / dl / P_half_2[2]
    dE2dr[p, Nl_half - 3, :, :] = (- E2d[p, -3, :, :] + 0.75 * E2d[p, -2, :, :] + 0.25 * E2d[p, -1, :, :]) / dl / P_half_2[Nxi_half - 3]
    dE2dr[p, Nl_half - 2, :, :] = (- 0.25 * E2d[p, -2, :, :] + 0.25 * E2d[p, -1, :, :]) / dl / P_half_2[Nxi_half - 2]
    dE2dr[p, Nl_half - 1, :, :] = (- 0.50 * E2d[p, -2, :, :] + 0.50 * E2d[p, -1, :, :]) / dl / P_half_2[Nxi_half - 1]
    dE2dr[p, 3:(Nl_half - 3), :, :] = (E2d[p, 3:(Nl_half - 3), :, :] - N.roll(E2d, 1, axis = 1)[p, 3:(Nl_half - 3), :, :]) / dl

    dErd1[p, :, 0, :] = (- 0.50 * Erd[p, :, 0, :] + 0.50 * Erd[p, :, 1, :]) / dxi / P_half_2[0]
    dErd1[p, :, 1, :] = (- 0.25 * Erd[p, :, 0, :] + 0.25 * Erd[p, :, 1, :]) / dxi / P_half_2[1]
    dErd1[p, :, 2, :] = (- 0.25 * Erd[p, :, 0, :] - 0.75 * Erd[p, :, 1, :] + Erd[p, :, 2, :]) / dxi / P_half_2[2]
    dErd1[p, :, Nxi_half - 3, :] = (- Erd[p, :, -3, :] + 0.75 * Erd[p, :, -2, :] + 0.25 * Erd[p, :, -1, :]) / dxi / P_half_2[Nxi_half - 3]
    dErd1[p, :, Nxi_half - 2, :] = (- 0.25 * Erd[p, :, -2, :] + 0.25 * Erd[p, :, -1, :]) / dxi / P_half_2[Nxi_half - 2]
    dErd1[p, :, Nxi_half - 1, :] = (- 0.5 * Erd[p, :, -2, :] + 0.5 * Erd[p, :, -1, :]) / dxi / P_half_2[Nxi_half - 1]
    dErd1[p, :, 3:(Nxi_half - 3), :] = (Erd[p, :, 3:(Nxi_half - 3), :] - N.roll(Erd, 1, axis = 2)[p, :, 3:(Nxi_half - 3), :]) / dxi

    dErd2[p, :, :, 0] = (- 0.50 * Erd[p, :, :, 0] + 0.50 * Erd[p, :, :, 1]) / deta / P_half_2[0]
    dErd2[p, :, :, 1] = (- 0.25 * Erd[p, :, :, 0] + 0.25 * Erd[p, :, :, 1]) / deta / P_half_2[1]
    dErd2[p, :, :, 2] = (- 0.25 * Erd[p, :, :, 0] - 0.75 * Erd[p, :, :, 1] + Erd[p, :, :, 2]) / deta / P_half_2[2]
    dErd2[p, :, :, Neta_half - 3] = (- Erd[p, :, :, -3] + 0.75 * Erd[p, :, :, -2] + 0.25 * Erd[p, :, :, -1]) / deta / P_half_2[Nxi_half - 3]
    dErd2[p, :, :, Neta_half - 2] = (- 0.25 * Erd[p, :, :, -2] + 0.25 * Erd[p, :, :, -1]) / deta / P_half_2[Nxi_half - 2]
    dErd2[p, :, :, Neta_half - 1] = (- 0.50 * Erd[p, :, :, -2] + 0.50 * Erd[p, :, :, -1]) / deta / P_half_2[Nxi_half - 1]
    dErd2[p, :, :, 3:(Neta_half - 3)] = (Erd[p, :, :, 3:(Neta_half - 3)] - N.roll(Erd, 1, axis = 3)[p, :, :, 3:(Neta_half - 3)]) / deta

def average_field(p, fieldrin0, field1in0, field2in0, fieldrin1, field1in1, field2in1, fieldrout, field1out, field2out):
    fieldrout[p, :, :, :] = 0.5 * (fieldrin0[p, :, :, :] + fieldrin1[p, :, :, :])
    field1out[p, :, :, :] = 0.5 * (field1in0[p, :, :, :] + field1in1[p, :, :, :])
    field2out[p, :, :, :] = 0.5 * (field2in0[p, :, :, :] + field2in1[p, :, :, :])

########
# Single-patch push routines
########

def push_D(p, Drin, D1in, D2in, dtin):

    Drin[p, :, :, :] += dtin * (dH2d1[p, :, :, :] - dH1d2[p, :, :, :]) / sqrt_det_h_Dr[p, :, :, :] 
    D1in[p, :, :, :] += dtin * (dHrd2[p, :, :, :] - dH2dr[p, :, :, :]) / sqrt_det_h_D1[p, :, :, :] 
    D2in[p, :, :, :] += dtin * (dH1dr[p, :, :, :] - dHrd1[p, :, :, :]) / sqrt_det_h_D2[p, :, :, :]

def push_B(p, Brin, B1in, B2in, dtin):

    Brin[p, :, :, :] += dtin * (dE1d2[p, :, :, :] - dE2d1[p, :, :, :]) / sqrt_det_h_Br[p, :, :, :] 
    B1in[p, :, :, :] += dtin * (dE2dr[p, :, :, :] - dErd2[p, :, :, :]) / sqrt_det_h_B1[p, :, :, :]
    B2in[p, :, :, :] += dtin * (dErd1[p, :, :, :] - dE1dr[p, :, :, :]) / sqrt_det_h_B2[p, :, :, :] 

########
# Auxiliary field computation
########

def contra_to_cov_D(p, Drin, D1in, D2in):

    ########
    # Dr
    ########

    # Interior
    Drd[p, 1:-1, 1:-1, 1:-1] = hrrd_Dr[p, 1:-1, 1:-1, 1:-1] *  Drin[p, 1:-1, 1:-1, 1:-1] \
                      + 0.25 * hr1d_Dr[p, 1:-1, 1:-1, 1:-1] * (D1in[p, 1:, 1:-2, 1:-1] + N.roll(N.roll(D1in, 1, axis = 1), -1, axis = 2)[p, 1:, 1:-2, 1:-1]  \
                                                                                       + N.roll(D1in, 1, axis = 1)[p, 1:, 1:-2, 1:-1] + N.roll(D1in, -1, axis = 2)[p, 1:, 1:-2, 1:-1]) \
                      + 0.25 * hr2d_Dr[p, 1:-1, 1:-1, 1:-1] * (D2in[p, 1:, 1:-1, 1:-2] + N.roll(N.roll(D2in, 1, axis = 1), -1, axis = 3)[p, 1:, 1:-1, 1:-2]  \
                                                                                       + N.roll(D2in, 1, axis = 1)[p, 1:, 1:-1, 1:-2] + N.roll(D2in, -1, axis = 3)[p, 1:, 1:-1, 1:-2]) \

    # Left/right face
    for j in [0, -1]:
        Drd[p, 1:-1, j, 1:-1] = hrrd_Dr[p, 1:-1, j, 1:-1] *  Drin[p, 1:-1, j, 1:-1] \
                              + 0.5  * hr1d_Dr[p, 1:-1, j, 1:-1] * (D1in[p, 1:, j, 1:-1] + N.roll(D1in, 1, axis = 1)[p, 1:, j, 1:-1]) \
                              + 0.25 * hr2d_Dr[p, 1:-1, j, 1:-1] * (D2in[p, 1:, j, 1:-2] + N.roll(N.roll(D2in, 1, axis = 1), -1, axis = 3)[p, 1:, j, 1:-2]  \
                                                                                       + N.roll(D2in, 1, axis = 1)[p, 1:, j, 1:-2] + N.roll(D2in, -1, axis = 3)[p, 1:, j, 1:-2]) \
    # Top/bottom face
    for k in [0, -1]:
        Drd[p, 1:-1, 1:-1, k] = hrrd_Dr[p, 1:-1, 1:-1, k] * Drin[p, 1:-1, 1:-1, k] \
                                 + 0.25 * hr1d_Dr[p, 1:-1, 1:-1, k] * (D1in[p, 1:, 1:-2, k] + N.roll(N.roll(D1in, 1, axis = 1), -1, axis = 2)[p, 1:, 1:-2, k]  \
                                                                              +   N.roll(D1in, 1, axis = 1)[p, 1:, 1:-2, k] + N.roll(D1in, -1, axis = 2)[p, 1:, 1:-2, k]) \
                                 + 0.5  * hr2d_Dr[p, 1:-1, 1:-1, k] * (D2in[p, 1:, 1:-1, k] + N.roll(D2in, 1, axis = 1)[p, 1:, 1:-1, k])
                                 
    # Back/front face
    for i in [0, -1]:
        Drd[p, i, 1:-1, 1:-1] = hrrd_Dr[p, i, 1:-1, 1:-1] * Drin[p, i, 1:-1, 1:-1] \
                              + 0.5 * hr1d_Dr[p, i, 1:-1, 1:-1] * (D1in[p, i, 1:-2, 1:-1] + N.roll(D1in, -1, axis = 2)[p, i, 1:-2, 1:-1]) \
                              + 0.5 * hr2d_Dr[p, i, 1:-1, 1:-1] * (D2in[p, i, 1:-1, 1:-2] + N.roll(D2in, -1, axis = 3)[p, i, 1:-1, 1:-2])
    
    # Edges
    for i in [0, -1]:
        for j in [0, -1]:
            Drd[p, i, j, 1:-1] = hrrd_Dr[p, i, j, 1:-1] * Drin[p, i, j, 1:-1] \
                               + hr1d_Dr[p, i, j, 1:-1] * D1in[p, i, j, 1:-1] \
                               + 0.5 * hr2d_Dr[p, i, j, 1:-1] * (D2in[p, i, j, 1:-2] + N.roll(D2in, -1, axis = 3)[p, i, j, 1:-2]) 

    for i in [0, -1]:
        for k in [0, -1]:
            Drd[p, i, 1:-1, k] = hrrd_Dr[p, i, 1:-1, k] * Drin[p, i, 1:-1, k] \
                                 + 0.5 * hr1d_Dr[p, i, 1:-1, k] * (D1in[p, i, 1:-2, k] + N.roll(D1in, -1, axis = 2)[p, i, 1:-2, k]) \
                                 + hr2d_Dr[p, i, 1:-1, k] * D2in[p, i, 1:-1, k]

    for j in [0, -1]:
        for k in [0, -1]:
            Drd[p, 1:-1, j, k] = hrrd_Dr[p, 1:-1, j, k] *  Drin[p, 1:-1, j, k] \
                              + 0.5 * hr1d_Dr[p, 1:-1, j, k] * (D1in[p, 1:, j, k] + N.roll(D1in, 1, axis = 1)[p, 1:, j, k]) \
                              + 0.5 * hr2d_Dr[p, 1:-1, j, k] * (D2in[p, 1:, j, k] + N.roll(D2in, 1, axis = 1)[p, 1:, j, k])
                               
    # Corners
    for i in [0, -1]:
        for j in [0, -1]:
            for k in [0, -1]:
                Drd[p, i, j, k] = hrrd_Dr[p, i, j, k] * Drin[p, i, j, k] \
                                + hr1d_Dr[p, i, j, k] * D1in[p, i, j, k] \
                                + hr2d_Dr[p, i, j, k] * D2in[p, i, j, k] 
                                
    ########
    # Dxi
    ########

    # Interior
    D1d[p, 1:-1, 1:-1, 1:-1] = h11d_D1[p, 1:-1, 1:-1, 1:-1] * D1in[p, 1:-1, 1:-1, 1:-1] \
                             + 0.25 * h12d_D1[p, 1:-1, 1:-1, 1:-1] * (D2in[p, 1:-1, 1:, 1:-2] + N.roll(N.roll(D2in, 1, axis = 2), -1, axis = 3)[p, 1:-1, 1:, 1:-2] \
                                                                                    +  N.roll(D2in, 1, axis = 2)[p, 1:-1, 1:, 1:-2] + N.roll(D2in, -1, axis = 3)[p, 1:-1, 1:, 1:-2]) \
                             + 0.25 * hr1d_D1[p, 1:-1, 1:-1, 1:-1] * (Drin[p, 1:-2, 1:, 1:-1] + N.roll(N.roll(Drin, -1, axis = 1), 1, axis = 2)[p, 1:-2, 1:, 1:-1] \
                                                                                    +  N.roll(Drin, -1, axis = 1)[p, 1:-2, 1:, 1:-1] + N.roll(Drin, 1, axis = 2)[p, 1:-2, 1:, 1:-1])

    # Left/right face
    for j in [0, -1]:
        D1d[p, 1:-1, j, 1:-1] = h11d_D1[p, 1:-1, j, 1:-1] * D1in[p, 1:-1, j, 1:-1] \
                             + 0.5 * h12d_D1[p, 1:-1, j, 1:-1] * (D2in[p, 1:-1, j, 1:-2] + N.roll(D2in, -1, axis = 3)[p, 1:-1, j, 1:-2]) \
                             + 0.5 * hr1d_D1[p, 1:-1, j, 1:-1] * (Drin[p, 1:-2, j, 1:-1] + N.roll(Drin, -1, axis = 1)[p, 1:-2, j, 1:-1])

    # Top/bottom face
    for k in [0, -1]:
        D1d[p, 1:-1, 1:-1, k] = h11d_D1[p, 1:-1, 1:-1, k] * D1in[p, 1:-1, 1:-1, k] \
                             + 0.5  * h12d_D1[p, 1:-1, 1:-1, k] * (D2in[p, 1:-1, 1:, k] +  N.roll(D2in, 1, axis = 2)[p, 1:-1, 1:, k]) \
                             + 0.25 * hr1d_D1[p, 1:-1, 1:-1, k] * (Drin[p, 1:-2, 1:, k] + N.roll(N.roll(Drin, -1, axis = 1), 1, axis = 2)[p, 1:-2, 1:, k] \
                                                                                    +  N.roll(Drin, -1, axis = 1)[p, 1:-2, 1:, k] + N.roll(Drin, 1, axis = 2)[p, 1:-2, 1:, k])
                                 
    # Back/front face
    for i in [0, -1]:
        D1d[p, i, 1:-1, 1:-1] = h11d_D1[p, i, 1:-1, 1:-1] * D1in[p, i, 1:-1, 1:-1] \
                             + 0.25 * h12d_D1[p, i, 1:-1, 1:-1] * (D2in[p, i, 1:, 1:-2] + N.roll(N.roll(D2in, 1, axis = 2), -1, axis = 3)[p, i, 1:, 1:-2] \
                                                                                    +  N.roll(D2in, 1, axis = 2)[p, i, 1:, 1:-2] + N.roll(D2in, -1, axis = 3)[p, i, 1:, 1:-2]) \
                             + 0.5 * hr1d_D1[p, i, 1:-1, 1:-1] * (Drin[p, i, 1:, 1:-1] + N.roll(Drin, 1, axis = 2)[p, i, 1:, 1:-1])
    
    # Edges
    for i in [0, -1]:
        for j in [0, -1]:
            D1d[p, i, j, 1:-1] = h11d_D1[p, i, j, 1:-1] * D1in[p, i, j, 1:-1] \
                                + 0.5 * h12d_D1[p, i, j, 1:-1] * (D2in[p, i, j, 1:-2] + N.roll(D2in, -1, axis = 3)[p, i, j, 1:-2]) \
                                + hr1d_D1[p, i, j, 1:-1] * Drin[p, i, j, 1:-1]

    for i in [0, -1]:
        for k in [0, -1]:
            D1d[p, i, 1:-1, k] = h11d_D1[p, i, 1:-1, k] * D1in[p, i, 1:-1, k] \
                             + 0.5 * h12d_D1[p, i, 1:-1, k] * (D2in[p, i, 1:, k] + N.roll(D2in, 1, axis = 2)[p, i, 1:, k]) \
                             + 0.5 * hr1d_D1[p, i, 1:-1, k] * (Drin[p, i, 1:, k] + N.roll(Drin, 1, axis = 2)[p, i, 1:, k])

    for j in [0, -1]:
        for k in [0, -1]:
            D1d[p, 1:-1, j, k] = h11d_D1[p, 1:-1, j, k] * D1in[p, 1:-1, j, k] \
                             + h12d_D1[p, 1:-1, j, k] * D2in[p, 1:-1, j, k] \
                             + 0.5 * hr1d_D1[p, 1:-1, j, k] * (Drin[p, 1:-2, j, k] + N.roll(Drin, -1, axis = 1)[p, 1:-2, j, k])
                               
    # Corners
    for i in [0, -1]:
        for j in [0, -1]:
            for k in [0, -1]:
                D1d[p, i, j, k] = h11d_D1[p, i, j, k] * D1in[p, i, j, k] \
                                + h12d_D1[p, i, j, k] * D2in[p, i, j, k] \
                                + hr1d_D1[p, i, j, k] * Drin[p, i, j, k] 

    ########
    # Deta
    ########

    # Interior
    D2d[p, 1:-1, 1:-1, 1:-1] = h22d_D2[p, 1:-1, 1:-1, 1:-1] * D2in[p, 1:-1, 1:-1, 1:-1] \
                             + 0.25 * h12d_D2[p, 1:-1, 1:-1, 1:-1] * (D1in[p, 1:-1, 1:-2, 1:] + N.roll(N.roll(D1in, -1, axis = 2), 1, axis = 3)[p, 1:-1, 1:-2, 1:] \
                             + N.roll(D1in, -1, axis = 2)[p, 1:-1, 1:-2, 1:] + N.roll(D1in, 1, axis = 3)[p, 1:-1, 1:-2, 1:]) \
                             + 0.25 * hr2d_D2[p, 1:-1, 1:-1, 1:-1] * (Drin[p, 1:-2, 1:-1, 1:] + N.roll(N.roll(Drin, -1, axis = 1), 1, axis = 3)[p, 1:-2, 1:-1, 1:] \
                             + N.roll(Drin, -1, axis = 1)[p, 1:-2, 1:-1, 1:] + N.roll(Drin, 1, axis = 3)[p, 1:-2, 1:-1, 1:])

    # Left/right face
    for j in [0, -1]:
        D2d[p, 1:-1, j, 1:-1] = h22d_D2[p, 1:-1, j, 1:-1] * D2in[p, 1:-1, j, 1:-1] \
                             + 0.5  * h12d_D2[p, 1:-1, j, 1:-1] * (D1in[p, 1:-1, j, 1:] + N.roll(D1in, 1, axis = 3)[p, 1:-1, j, 1:]) \
                             + 0.25 * hr2d_D2[p, 1:-1, j, 1:-1] * (Drin[p, 1:-2, j, 1:] + N.roll(N.roll(Drin, -1, axis = 1), 1, axis = 3)[p, 1:-2, j, 1:] \
                             + N.roll(Drin, -1, axis = 1)[p, 1:-2, j, 1:] + N.roll(Drin, 1, axis = 3)[p, 1:-2, j, 1:])

    # Top/bottom face
    for k in [0, -1]:
        D2d[p, 1:-1, 1:-1, k] = h22d_D2[p, 1:-1, 1:-1, k] * D2in[p, 1:-1, 1:-1, k] \
                             + 0.5 * h12d_D2[p, 1:-1, 1:-1, k] * (D1in[p, 1:-1, 1:-2, k] + N.roll(D1in, -1, axis = 2)[p, 1:-1, 1:-2, k]) \
                             + 0.5 * hr2d_D2[p, 1:-1, 1:-1, k] * (Drin[p, 1:-2, 1:-1, k] + N.roll(Drin, -1, axis = 1)[p, 1:-2, 1:-1, k])
                                 
    # Back/front face
    for i in [0, -1]:
        D2d[p, i, 1:-1, 1:-1] = h22d_D2[p, i, 1:-1, 1:-1] * D2in[p, i, 1:-1, 1:-1] \
                             + 0.25 * h12d_D2[p, i, 1:-1, 1:-1] * (D1in[p, i, 1:-2, 1:] + N.roll(N.roll(D1in, -1, axis = 2), 1, axis = 3)[p, i, 1:-2, 1:] \
                             + N.roll(D1in, -1, axis = 2)[p, i, 1:-2, 1:] + N.roll(D1in, 1, axis = 3)[p, i, 1:-2, 1:]) \
                             + 0.5 * hr2d_D2[p, i, 1:-1, 1:-1] * (Drin[p, i, 1:-1, 1:] + N.roll(Drin, 1, axis = 3)[p, i, 1:-1, 1:])
    
    # Edges
    for i in [0, -1]:
        for j in [0, -1]:
            D2d[p, i, j, 1:-1] = h22d_D2[p, i, j, 1:-1] * D2in[p, i, j, 1:-1] \
                             + 0.5 * h12d_D2[p, i, j, 1:-1] * (D1in[p, i, j, 1:] + N.roll(D1in, 1, axis = 3)[p, i, j, 1:]) \
                             + 0.5 * hr2d_D2[p, i, j, 1:-1] * (Drin[p, i, j, 1:] + N.roll(Drin, 1, axis = 3)[p, i, j, 1:])

    # Top/bottom edges
    for i in [0, -1]:
        for k in [0, -1]:
            D2d[p, i, 1:-1, k] = h22d_D2[p, i, 1:-1, k] * D2in[p, i, 1:-1, k] \
                             + 0.5 * h12d_D2[p, i, 1:-1, k] * (D1in[p, i, 1:-2, k] + N.roll(D1in, -1, axis = 2)[p, i, 1:-2, k]) \
                             + hr2d_D2[p, i, 1:-1, k] * Drin[p, i, 1:-1, k]

    for j in [0, -1]:
        for k in [0, -1]:
            D2d[p, 1:-1, j, k] = h22d_D2[p, 1:-1, j, k] * D2in[p, 1:-1, j, k] \
                               + h12d_D2[p, 1:-1, j, k] * D1in[p, 1:-1, j, k] \
                               + 0.5 * hr2d_D2[p, 1:-1, j, k] * (Drin[p, 1:-2, j, k] + N.roll(Drin, -1, axis = 1)[p, 1:-2, j, k])
                               
    # Corners
    for i in [0, -1]:
        for j in [0, -1]:
            for k in [0, -1]:
                D2d[p, i, j, k] = h22d_D2[p, i, j, k] * D2in[p, i, j, k] \
                                + h12d_D2[p, i, j, k] * D1in[p, i, j, k] \
                                + hr2d_D2[p, i, j, k] * Drin[p, i, j, k] 
                                
                                
def compute_E_aux(p, Drin, D1in, D2in, Brin, B1in, B2in):

    ##### Er
    Erd[p, :, :, :] = alpha_Dr[p, :, :, :] * Drin[p, :, :, :]

    ##### Exi

    # Interior
    E1d[p, 1:-1, :, :] = alpha_D1[p, 1:-1, :, :] * D1in[p, 1:-1, :, :] \
                       - sqrt_det_h_D1[p, 1:-1, :, :] * beta_D1[p, 1:-1, :, :] \
                       * 0.5 * (B2in[p, 1:-2, :, :] + N.roll(B2in, -1, axis = 1)[p, 1:-2, :, :])

    # Back/front face
    for i in [0, -1]:
        E1d[p, i, :, :] = alpha_D1[p, i, :, :] * D1in[p, i, :, :] \
                        - sqrt_det_h_D1[p, i, :, :] * beta_D1[p, i, :, :] * B2in[p, i, :, :]

    ##### Eeta
    
    ##### Interior
    E2d[p, 1:-1, :, :] = alpha_D2[p, 1:-1, :, :] * D2in[p, 1:-1, :, :] \
                          + 0.5 * sqrt_det_h_D2[p, 1:-1, :, :] * beta_D2[p, 1:-1, :, :] \
                          * (B1in[p, 1:-2, :, :] + N.roll(B1in, -1, axis = 1)[p, 1:-2, :, :])

    # Back/front face
    for i in [0, -1]:
        E2d[p, i, :, :] = alpha_D2[p, i, :, :] * D2in[p, i, :, :] \
                           + sqrt_det_h_D2[p, i, :, :] * beta_D2[p, i, :, :] * B1in[p, i, :, :]


def contra_to_cov_B(p, Brin, B1in, B2in):

    ########
    # Br
    ########

    # Interior
    Brd[p, 1:-1, 1:-1, 1:-1] = hrrd_Br[p, 1:-1, 1:-1, 1:-1] * Brin[p, 1:-1, 1:-1, 1:-1] \
                                      + 0.25 * hr1d_Br[p, 1:-1, 1:-1, 1:-1] * (B1in[p, 1:-2, 1:, 1:-1] + N.roll(N.roll(B1in, -1, axis = 1), 1, axis = 2)[p, 1:-2, 1:, 1:-1]  \
                                                                                   +  N.roll(B1in, -1, axis = 1)[p, 1:-2, 1:, 1:-1] + N.roll(B1in, 1, axis = 2)[p, 1:-2, 1:, 1:-1]) \
                                      + 0.25 * hr2d_Br[p, 1:-1, 1:-1, 1:-1] * (B2in[p, 1:-2, 1:-1, 1:] + N.roll(N.roll(B2in, -1, axis = 1), 1, axis = 3)[p, 1:-2, 1:-1, 1:]  \
                                                                                   +  N.roll(B2in, -1, axis = 1)[p, 1:-2, 1:-1, 1:] + N.roll(B2in, 1, axis = 3)[p, 1:-2, 1:-1, 1:])

    # Left/right face
    for j in [0, -1]:
        Brd[p, 1:-1, j, 1:-1] = hrrd_Br[p, 1:-1, j, 1:-1] * Brin[p, 1:-1, j, 1:-1] \
                                      + 0.5  * hr1d_Br[p, 1:-1, j, 1:-1] * (B1in[p, 1:-2, j, 1:-1] +  N.roll(B1in, -1, axis = 1)[p, 1:-2, j, 1:-1]) \
                                      + 0.25 * hr2d_Br[p, 1:-1, j, 1:-1] * (B2in[p, 1:-2, j, 1:] + N.roll(N.roll(B2in, -1, axis = 1), 1, axis = 3)[p, 1:-2, j, 1:]  \
                                                                                   +  N.roll(B2in, -1, axis = 1)[p, 1:-2, j, 1:] + N.roll(B2in, 1, axis = 3)[p, 1:-2, j, 1:])
                                      
    # Top/bottom face
    for k in [0, -1]:
        Brd[p, 1:-1, 1:-1, k] = hrrd_Br[p, 1:-1, 1:-1, k] * Brin[p, 1:-1, 1:-1, k] \
                                      + 0.25 * hr1d_Br[p, 1:-1, 1:-1, k] * (B1in[p, 1:-2, 1:, k] + N.roll(N.roll(B1in, -1, axis = 1), 1, axis = 2)[p, 1:-2, 1:, k]  \
                                                                                   +  N.roll(B1in, -1, axis = 1)[p, 1:-2, 1:, k] + N.roll(B1in, 1, axis = 2)[p, 1:-2, 1:, k]) \
                                      + 0.5 * hr2d_Br[p, 1:-1, 1:-1, k] * (B2in[p, 1:-2, 1:-1, k] +  N.roll(B2in, -1, axis = 1)[p, 1:-2, 1:-1, k])
                                 
    # Back/front face
    for i in [0, -1]:
        Brd[p, i, 1:-1, 1:-1] = hrrd_Br[p, i, 1:-1, 1:-1] * Brin[p, i, 1:-1, 1:-1] \
                                      + 0.5 * hr1d_Br[p, i, 1:-1, 1:-1] * (B1in[p, i, 1:, 1:-1] + N.roll(B1in, 1, axis = 2)[p, i, 1:, 1:-1]) \
                                      + 0.5 * hr2d_Br[p, i, 1:-1, 1:-1] * (B2in[p, i, 1:-1, 1:] + N.roll(B2in, 1, axis = 3)[p, i, 1:-1, 1:])
    
    # Edges
    for i in [0, -1]:
        for j in [0, -1]:
            Brd[p, i, j, 1:-1] = hrrd_Br[p, i, j, 1:-1] * Brin[p, i, j, 1:-1] \
                                      + hr1d_Br[p, i, j, 1:-1] * B1in[p, i, j, 1:-1] \
                                      + 0.5 * hr2d_Br[p, i, j, 1:-1] * (B2in[p, i, j, 1:] + N.roll(B2in, 1, axis = 3)[p, i, j, 1:])

    for i in [0, -1]:
        for k in [0, -1]:
            Brd[p, i, 1:-1, k] = hrrd_Br[p, i, 1:-1, k] * Brin[p, i, 1:-1, k] \
                                      + 0.5 * hr1d_Br[p, i, 1:-1, k] * (B1in[p, i, 1:, k] + N.roll(B1in, 1, axis = 2)[p, i, 1:, k]) \
                                      + hr2d_Br[p, i, 1:-1, k] * B2in[p, i, 1:-1, k]

    for j in [0, -1]:
        for k in [0, -1]:
            Brd[p, 1:-1, j, k] = hrrd_Br[p, 1:-1, j, k] * Brin[p, 1:-1, j, k] \
                                      + 0.5 * hr1d_Br[p, 1:-1, j, k] * (B1in[p, 1:-2, j, k] + N.roll(B1in, -1, axis = 1)[p, 1:-2, j, k]) \
                                      + 0.5 * hr2d_Br[p, 1:-1, j, k] * (B2in[p, 1:-2, j, k] + N.roll(B2in, -1, axis = 1)[p, 1:-2, j, k])
                                      
                               
    # Corners
    for i in [0, -1]:
        for j in [0, -1]:
            for k in [0, -1]:
                Brd[p, i, j, k] = hrrd_Br[p, i, j, k] * Brin[p, i, j, k] \
                                + hr1d_Br[p, i, j, k] * B1in[p, i, j, k] \
                                + hr2d_Br[p, i, j, k] * B2in[p, i, j, k] 

    ########
    # Bxi
    ########

    # Interior
    B1d[p, 1:-1, 1:-1, 1:-1] = h11d_B1[p, 1:-1, 1:-1, 1:-1] * B1in[p, 1:-1, 1:-1, 1:-1] \
                             + 0.25 * h12d_B1[p, 1:-1, 1:-1, 1:-1] * (B2in[p, 1:-1, 1:-2, 1:] + N.roll(N.roll(B2in, -1, axis = 2), 1, axis = 3)[p, 1:-1, 1:-2, 1:] \
                                                                                    +  N.roll(B2in, -1, axis = 2)[p, 1:-1, 1:-2, 1:] + N.roll(B2in, 1, axis = 3)[p, 1:-1, 1:-2, 1:]) \
                             + 0.25 * hr1d_B1[p, 1:-1, 1:-1, 1:-1] * (Brin[p, 1:, 1:-2, 1:-1] + N.roll(N.roll(Brin, 1, axis = 1), -1, axis = 2)[p, 1:, 1:-2, 1:-1] \
                                                                                    +  N.roll(Brin, 1, axis = 1)[p, 1:, 1:-2, 1:-1] + N.roll(Brin, -1, axis = 2)[p, 1:, 1:-2, 1:-1])
    # Left/right face
    for j in [0, -1]:
        B1d[p, 1:-1, j, 1:-1] = h11d_B1[p, 1:-1, j, 1:-1] * B1in[p, 1:-1, j, 1:-1] \
                             + 0.5 * h12d_B1[p, 1:-1, j, 1:-1] * (B2in[p, 1:-1, j, 1:] + N.roll(B2in, 1, axis = 3)[p, 1:-1, j, 1:]) \
                             + 0.5 * hr1d_B1[p, 1:-1, j, 1:-1] * (Brin[p, 1:, j, 1:-1] + N.roll(Brin, 1, axis = 1)[p, 1:, j, 1:-1])
                                      
    # Top/bottom face
    for k in [0, -1]:
        B1d[p, 1:-1, 1:-1, k] = h11d_B1[p, 1:-1, 1:-1, k] * B1in[p, 1:-1, 1:-1, k] \
                             + 0.5 * h12d_B1[p, 1:-1, 1:-1, k] * (B2in[p, 1:-1, 1:-2, k] +  N.roll(B2in, -1, axis = 2)[p, 1:-1, 1:-2, k]) \
                             + 0.25 * hr1d_B1[p, 1:-1, 1:-1, k] * (Brin[p, 1:, 1:-2, k] + N.roll(N.roll(Brin, 1, axis = 1), -1, axis = 2)[p, 1:, 1:-2, k] \
                                                                                    +  N.roll(Brin, 1, axis = 1)[p, 1:, 1:-2, k] + N.roll(Brin, -1, axis = 2)[p, 1:, 1:-2, k])
                             
    # Back/front face
    for i in [0, -1]:
        B1d[p, i, 1:-1, 1:-1] = h11d_B1[p, i, 1:-1, 1:-1] * B1in[p, i, 1:-1, 1:-1] \
                             + 0.25 * h12d_B1[p, i, 1:-1, 1:-1] * (B2in[p, i, 1:-2, 1:] + N.roll(N.roll(B2in, -1, axis = 2), 1, axis = 3)[p, i, 1:-2, 1:] \
                                                                                    +  N.roll(B2in, -1, axis = 2)[p, i, 1:-2, 1:] + N.roll(B2in, 1, axis = 3)[p, i, 1:-2, 1:]) \
                             + 0.5 * hr1d_B1[p, i, 1:-1, 1:-1] * (Brin[p, i, 1:-2, 1:-1] + N.roll(Brin, -1, axis = 2)[p, i, 1:-2, 1:-1])
    
    # Edges
    for i in [0, -1]:
        for j in [0, -1]:
            B1d[p, i, j, 1:-1] = h11d_B1[p, i, j, 1:-1] * B1in[p, i, j, 1:-1] \
                             + 0.5 * h12d_B1[p, i, j, 1:-1] * (B2in[p, i, j, 1:] + N.roll(B2in, 1, axis = 3)[p, i, j, 1:]) \
                             + hr1d_B1[p, i, j, 1:-1] * Brin[p, i, j, 1:-1]

    for i in [0, -1]:
        for k in [0, -1]:
            B1d[p, i, 1:-1, k] = h11d_B1[p, i, 1:-1, k] * B1in[p, i, 1:-1, k] \
                             + 0.5 * h12d_B1[p, i, 1:-1, k] * (B2in[p, i, 1:-2, k] + N.roll(B2in, -1, axis = 2)[p, i, 1:-2, k]) \
                             + 0.5 * hr1d_B1[p, i, 1:-1, k] * (Brin[p, i, 1:-2, k] + N.roll(Brin, -1, axis = 2)[p, i, 1:-2, k])

    for j in [0, -1]:
        for k in [0, -1]:
            B1d[p, 1:-1, j, k] = h11d_B1[p, 1:-1, j, k] * B1in[p, 1:-1, j, k] \
                             + h12d_B1[p, 1:-1, j, k] * B2in[p, 1:-1, j, k] \
                             + 0.5 * hr1d_B1[p, 1:-1, j, k] * (Brin[p, 1:, j, k] + N.roll(Brin, 1, axis = 1)[p, 1:, j, k])
                               
    # Corners
    for i in [0, -1]:
        for j in [0, -1]:
            for k in [0, -1]:
                B1d[p, i, j, k] = h11d_B1[p, i, j, k] * B1in[p, i, j, k] \
                                + h12d_B1[p, i, j, k] * B2in[p, i, j, k] \
                                + hr1d_B1[p, i, j, k] * Brin[p, i, j, k] 

    ########
    # Beta
    ########

    # Interior
    B2d[p, 1:-1, 1:-1, 1:-1] = h22d_B2[p, 1:-1, 1:-1, 1:-1] * B2in[p, 1:-1, 1:-1, 1:-1] \
                             + 0.25 * h12d_B2[p, 1:-1, 1:-1, 1:-1] * (B1in[p, 1:-1, 1:, 1:-2] + N.roll(N.roll(B1in, 1, axis = 2), -1, axis = 3)[p, 1:-1, 1:, 1:-2] \
                                                                                    +  N.roll(B1in, 1, axis = 2)[p, 1:-1, 1:, 1:-2] + N.roll(B1in, -1, axis = 3)[p, 1:-1, 1:, 1:-2]) \
                             + 0.25 * hr2d_B2[p, 1:-1, 1:-1, 1:-1] * (Brin[p, 1:, 1:-1, 1:-2] + N.roll(N.roll(Brin, 1, axis = 1), -1, axis = 3)[p, 1:, 1:-1, 1:-2] \
                                                                                    +  N.roll(Brin, 1, axis = 1)[p, 1:, 1:-1, 1:-2] + N.roll(Brin, -1, axis = 3)[p, 1:, 1:-1, 1:-2])
                             
    # Left/right face
    for j in [0, -1]:
        B2d[p, 1:-1, j, 1:-1] = h22d_B2[p, 1:-1, j, 1:-1] * B2in[p, 1:-1, j, 1:-1] \
                              + 0.5  * h12d_B2[p, 1:-1, j, 1:-1] * (B1in[p, 1:-1, j, 1:-2] + N.roll(B1in, -1, axis = 3)[p, 1:-1, j, 1:-2]) \
                              + 0.25 * hr2d_B2[p, 1:-1, j, 1:-1] * (Brin[p, 1:, j, 1:-2] + N.roll(N.roll(Brin, 1, axis = 1), -1, axis = 3)[p, 1:, j, 1:-2] \
                                                                                    +  N.roll(Brin, 1, axis = 1)[p, 1:, j, 1:-2] + N.roll(Brin, -1, axis = 3)[p, 1:, j, 1:-2])

    # Top/bottom face
    for k in [0, -1]:
        B2d[p, 1:-1, 1:-1, k] = h22d_B2[p, 1:-1, 1:-1, k] * B2in[p, 1:-1, 1:-1, k] \
                              + 0.5 * h12d_B2[p, 1:-1, 1:-1, k] * (B1in[p, 1:-1, 1:, k] +  N.roll(B1in, 1, axis = 2)[p, 1:-1, 1:, k]) \
                              + 0.5 * hr2d_B2[p, 1:-1, 1:-1, k] * (Brin[p, 1:, 1:-1, k] +  N.roll(Brin, 1, axis = 1)[p, 1:, 1:-1, k])
                             
    # Back/front face
    for i in [0, -1]:
        B2d[p, i, 1:-1, 1:-1] = h22d_B2[p, i, 1:-1, 1:-1] * B2in[p, i, 1:-1, 1:-1] \
                              + 0.25 * h12d_B2[p, i, 1:-1, 1:-1] * (B1in[p, i, 1:, 1:-2] + N.roll(N.roll(B1in, 1, axis = 2), -1, axis = 3)[p, i, 1:, 1:-2] \
                                                                                    +  N.roll(B1in, 1, axis = 2)[p, i, 1:, 1:-2] + N.roll(B1in, -1, axis = 3)[p, i, 1:, 1:-2]) \
                              + 0.5 * hr2d_B2[p, i, 1:-1, 1:-1] * (Brin[p, i, 1:-1, 1:-2] + N.roll(Brin, -1, axis = 3)[p, i, 1:-1, 1:-2])
    
    # Edges
    for i in [0, -1]:
        for j in [0, -1]:
            B2d[p, i, j, 1:-1] = h22d_B2[p, i, j, 1:-1] * B2in[p, i, j, 1:-1] \
                              + 0.5 * h12d_B2[p, i, j, 1:-1] * (B1in[p, i, j, 1:-2] + N.roll(B1in, -1, axis = 3)[p, i, j, 1:-2]) \
                              + 0.5 * hr2d_B2[p, i, j, 1:-1] * (Brin[p, i, j, 1:-2] + N.roll(Brin, -1, axis = 3)[p, i, j, 1:-2])

    for i in [0, -1]:
        for k in [0, -1]:
            B2d[p, i, 1:-1, k] = h22d_B2[p, i, 1:-1, k] * B2in[p, i, 1:-1, k] \
                              + 0.5 * h12d_B2[p, i, 1:-1, k] * (B1in[p, i, 1:, k] +  N.roll(B1in, 1, axis = 2)[p, i, 1:, k]) \
                              + hr2d_B2[p, i, 1:-1, k] * Brin[p, i, 1:-1, k]

    for j in [0, -1]:
        for k in [0, -1]:
            B2d[p, 1:-1, j, k] = h22d_B2[p, 1:-1, j, k] * B2in[p, 1:-1, j, k] \
                               + h12d_B2[p, 1:-1, j, k] * B1in[p, 1:-1, j, k] \
                               + 0.5 * hr2d_B2[p, 1:-1, j, k] * (Brin[p, 1:, j, k] +  N.roll(Brin, 1, axis = 1)[p, 1:, j, k])
                        
    # Corners
    for i in [0, -1]:
        for j in [0, -1]:
            for k in [0, -1]:
                B2d[p, i, j, k] = h22d_B2[p, i, j, k] * B2in[p, i, j, k] \
                                + h12d_B2[p, i, j, k] * B1in[p, i, j, k] \
                                + hr2d_B2[p, i, j, k] * Brin[p, i, j, k] 


def compute_H_aux(p, Drin, D1in, D2in, Brin, B1in, B2in):

    ##### Hr
    Hrd[p, :, :, :] = alpha_Br[p, :, :, :] * Brin[p, :, :, :]

    ##### Hxi

    # Interior
    H1d[p, 1:-1, :, :] = alpha_B1[p, 1:-1, :, :] * B1in[p, 1:-1, :, :] \
                       + sqrt_det_h_B1[p, 1:-1, :, :] * beta_B1[p, 1:-1, :, :] \
                       * 0.5 * (D2in[p, 1:, :, :] + N.roll(D2in, 1, axis = 1)[p, 1:, :, :])

    # Back/front faces
    for i in [0, -1]:
        H1d[p, i, :, :] = alpha_B1[p, i, :, :] * B1in[p, i, :, :] \
                        + sqrt_det_h_B1[p, i, :, :] * beta_B1[p, i, :, :] * D2in[p, i, :, :]

    ##### Heta

    # Interior
    H2d[p, 1:-1, :, :] = alpha_B2[p, 1:-1, :, :] * B2in[p, 1:-1, :, :] \
                       - 0.5 * sqrt_det_h_B2[p, 1:-1, :, :] * beta_B2[p, 1:-1, :, :] \
                       * (D1in[p, 1:, :, :] + N.roll(D1in, 1, axis = 1)[p, 1:, :, :])

    # Back/front faces
    for i in [0, -1]:
        H2d[p, i, :, :] = alpha_B2[p, i, :, :] * B2in[p, i, :, :] \
                        - sqrt_det_h_B2[p, i, :, :] * beta_B2[p, i, :, :] * D1in[p, i, :, :]

def compute_H_up(p, Drin, D1in, D2in, Brin, B1in, B2in):

    ##### Hr
    Hru[p, :, :, 1:-1] = alpha_Br[p, :, :, 1:-1] * Brin[p, :, :, 1:-1] \
                       + beta2d_Br[p, :, :, 1:-1] * 0.5 * (D1in[p, :, :, 1:] + N.roll(D1in, 1, axis = 1)[p, :, :, 1:]) / sqrt_det_h_Br[p, :, :, 1:-1]

    Hru[p, :, :, 0] = alpha_Br[p, :, :, 0] * Brin[p, :, :, 0] \
                       + beta2d_Br[p, :, :, 0] * D1in[p, :, :, 0] / sqrt_det_h_Br[p, :, :, 0]
    Hru[p, :, :, -1] = alpha_Br[p, :, :, -1] * Brin[p, :, :, -1] \
                       + beta2d_Br[p, :, :, -1] * D1in[p, :, :, -1] / sqrt_det_h_Br[p, :, :, -1]


    ##### Hxi

    # Interior
    H1d[p, 1:-1, :, :] = alpha_B1[p, 1:-1, :, :] * B1in[p, 1:-1, :, :] \
                       + sqrt_det_h_B1[p, 1:-1, :, :] * beta_B1[p, 1:-1, :, :] \
                       * 0.5 * (D2in[p, 1:, :, :] + N.roll(D2in, 1, axis = 1)[p, 1:, :, :])

    # Back/front faces
    for i in [0, -1]:
        H1d[p, i, :, :] = alpha_B1[p, i, :, :] * B1in[p, i, :, :] \
                        + sqrt_det_h_B1[p, i, :, :] * beta_B1[p, i, :, :] * D2in[p, i, :, :]

    ##### Heta

    # Interior
    H2d[p, 1:-1, :, :] = alpha_B2[p, 1:-1, :, :] * B2in[p, 1:-1, :, :] \
                       - 0.5 * sqrt_det_h_B2[p, 1:-1, :, :] * beta_B2[p, 1:-1, :, :] \
                       * (D1in[p, 1:, :, :] + N.roll(D1in, 1, axis = 1)[p, 1:, :, :])

    # Back/front faces
    for i in [0, -1]:
        H2d[p, i, :, :] = alpha_B2[p, i, :, :] * B2in[p, i, :, :] \
                        - sqrt_det_h_B2[p, i, :, :] * beta_B2[p, i, :, :] * D1in[p, i, :, :]


########
# Radial boundary conditions
########

def BC_Du(patch, Drin, D1in, D2in):
    Drin[patch, 0, :, :] = Drin[patch, 1, :, :] ## ????
    D1in[patch, 0, :, :] = D1in[patch, 1, :, :]
    D2in[patch, 0, :, :] = D2in[patch, 1, :, :]           
    return

def BC_Bu(patch, Brin, B1in, B2in):
    Brin[patch, 0, :, :] = Brin[patch, 1, :, :] ## ????
    B1in[patch, 0, :, :] = B1in[patch, 1, :, :]
    B2in[patch, 0, :, :] = B2in[patch, 1, :, :]
    return

def BC_Dd(patch, Drin, D1in, D2in):
    Drin[patch, 0, :, :] = Drin[patch, 1, :, :] ## ????
    D1in[patch, 0, :, :] = D1in[patch, 1, :, :]
    D2in[patch, 0, :, :] = D2in[patch, 1, :, :]           
    return

def BC_Bd(patch, Brin, B1in, B2in):
    Brin[patch, 0, :, :] = Brin[patch, 1, :, :] ## ????
    B1in[patch, 0, :, :] = B1in[patch, 1, :, :]
    B2in[patch, 0, :, :] = B2in[patch, 1, :, :]
    return

def BC_Ed(patch, Drin, D1in, D2in):
    Drin[patch, 0, :, :] = Drin[patch, 1, :, :] ## ????
    D1in[patch, 0, :, :] = D1in[patch, 1, :, :]
    D2in[patch, 0, :, :] = D2in[patch, 1, :, :]
    return        

def BC_Hd(patch, Brin, B1in, B2in):
    Brin[patch, 0, :, :] = Brin[patch, 1, :, :] ## ????
    B1in[patch, 0, :, :] = B1in[patch, 1, :, :]
    B2in[patch, 0, :, :] = B2in[patch, 1, :, :]
    return


########
# Compute interface terms
########

sig_in  = 1.0

def radial_penalty_D(dtin, Drin, D1in, D2in, Brin, B1in, B2in):

    #######
    # Inner boundary
    #######    
    
    #######
    # D1
    #######
    
    lambda_0 = alpha_D1[:, 0, :, :] * N.sqrt(hrru_D1[:, 0, :, :]) + beta_D1[:, 0, :, :]
            
    Dr_0 = interp(Drin[:, 0, :, :], eta_int, eta_half, 1)
    D1_0 = D1in[:, 0, :, :]
    B2_0 = B2in[:, 0, :, :]

    carac_0 = (D1_0 - hr1u_D1[:, 0, :, :] / hrru_D1[:, 0, :, :] * Dr_0 + B2_0 / N.sqrt(sqrt_det_h_D1**2 * hrru_D1)[:, 0, :, :])
    # carac_0 = (B2_0 / N.sqrt(sqrt_det_h_D1**2 * hrru_D1)[:, 0, :, :])

    diff_D1u[:, 0, :, :]  += dtin * sig_in * 0.5 * carac_0 * lambda_0 / dl / P_int_2[0]
    
    #######
    # D2
    #######

    lambda_0 = alpha_D2[:, 0, :, :] * N.sqrt(hrru_D2[:, 0, :, :]) + beta_D2[:, 0, :, :]
            
    Dr_0 = interp(Drin[:, 0, :, :], xi_int, xi_half, 2)
    D2_0 = D2in[:, 0, :, :]
    B1_0 = B1in[:, 0, :, :]

    carac_0 = (D2_0 - hr2u_D2[:, 0, :, :] / hrru_D2[:, 0, :, :] * Dr_0 - B1_0 / N.sqrt(sqrt_det_h_D2**2 * hrru_D2)[:, 0, :, :])
    # carac_0 = (- B1_0 / N.sqrt(sqrt_det_h_D2**2 * hrru_D2)[:, 0, :, :])

    diff_D2u[:, 0, :, :]  += dtin * sig_in * 0.5 * carac_0 * lambda_0 / dl / P_int_2[0]

    #######
    # Outer boundary
    #######    
    
    #######
    # D1
    #######

    lambda_0 = alpha_D1[:, -1, :, :] * N.sqrt(hrru_D1[:, -1, :, :]) - beta_D1[:, -1, :, :]
        
    Dr_0 = interp(Drin[:, -1, :, :], eta_int, eta_half, 1)
    D1_0 = D1in[:, -1, :, :]
    B2_0 = B2in[:, -1, :, :]

    carac_0 = (D1_0 - hr1u_D1[:, -1, :, :] / hrru_D1[:, -1, :, :] * Dr_0 - B2_0 / N.sqrt(sqrt_det_h_D1**2 * hrru_D1)[:, -1, :, :])

    diff_D1u[:, -1, :, :]  += dtin * sig_in * 0.5 * carac_0 * lambda_0 / dl / P_int_2[0]
    
    #######
    # D2
    #######

    lambda_0 = alpha_D2[:, -1, :, :] * N.sqrt(hrru_D2[:, -1, :, :]) - beta_D2[:, -1, :, :]
            
    Dr_0 = interp(Drin[:, -1, :, :], xi_int, xi_half, 2)
    D2_0 = D2in[:, -1, :, :]
    B1_0 = B1in[:, -1, :, :]

    carac_0 = (D2_0 - hr2u_D2[:, -1, :, :] / hrru_D2[:, -1, :, :] * Dr_0 + B1_0 / N.sqrt(sqrt_det_h_D2**2 * hrru_D2)[:, -1, :, :])

    diff_D2u[:, -1, :, :]  += dtin * sig_in * 0.5 * carac_0 * lambda_0 / dl / P_int_2[0]


def radial_penalty_B(dtin, Drin, D1in, D2in, Brin, B1in, B2in):

    #######
    # Inner boundary
    #######    

    #######
    # B1
    #######

    lambda_0 = alpha_B1[:, 0, :, :] * N.sqrt(hrru_B1[:, 0, :, :]) + beta_B1[:, 0, :, :]
            
    Br_0 = interp(Brin[:, 0, :, :], xi_half, xi_int, 1)
    B1_0 = B1in[:, 0, :, :]
    D2_0 = D2in[:, 0, :, :]

    carac_0 = (B1_0 - hr1u_B1[:, 0, :, :] / hrru_B1[:, 0, :, :] * Br_0 - D2_0 / N.sqrt(sqrt_det_h_B1**2 * hrru_B1)[:, 0, :, :])
    # carac_0 = (D2_0 / N.sqrt(sqrt_det_h_B1**2 * hrru_B1)[:, 0, :, :])

    diff_B1u[:, 0, :, :]  += dtin * sig_in * 0.5 * carac_0 * lambda_0 / dl / P_int_2[0]

    #######
    # B2
    #######

    lambda_0 = alpha_B2[:, 0, :, :] * N.sqrt(hrru_B2[:, 0, :, :]) + beta_B2[:, 0, :, :]
            
    Br_0 = interp(Brin[:, 0, :, :], eta_half, eta_int, 2)
    B2_0 = B2in[:, 0, :, :]
    D1_0 = D1in[:, 0, :, :]

    carac_0 = (B2_0 - hr2u_B2[:, 0, :, :] / hrru_B2[:, 0, :, :] * Br_0 + D1_0 / N.sqrt(sqrt_det_h_B2**2 * hrru_B2)[:, 0, :, :])
    # carac_0 = (- D1_0 / N.sqrt(sqrt_det_h_B2**2 * hrru_B2)[:, 0, :, :])

    diff_B2u[:, 0, :, :]  += dtin * sig_in * 0.5 * carac_0 * lambda_0 / dl / P_int_2[0]

    #######
    # Outer boundary
    #######    

    #######
    # B1
    #######

    lambda_0 = alpha_B1[:, -1, :, :] * N.sqrt(hrru_B1[:, -1, :, :]) - beta_B1[:, -1, :, :]
            
    Br_0 = interp(Brin[:, -1, :, :], xi_half, xi_int, 1)
    B1_0 = B1in[:, -1, :, :]
    D2_0 = D2in[:, -1, :, :]

    carac_0 = (- B1_0 + hr1u_B1[:, -1, :, :] / hrru_B1[:, -1, :, :] * Br_0 - D2_0 / N.sqrt(sqrt_det_h_B1**2 * hrru_B1)[:, -1, :, :])

    diff_B1u[:, -1, :, :]  -= dtin * sig_in * 0.5 * carac_0 * lambda_0 / dl / P_int_2[0]

    #######
    # B2
    #######
    
    lambda_0 = alpha_B2[:, -1, :, :] * N.sqrt(hrru_B2[:, -1, :, :]) - beta_B2[:, -1, :, :]
            
    Br_0 = interp(Brin[:, -1, :, :], eta_half, eta_int, 2)
    B2_0 = B2in[:, -1, :, :]
    D1_0 = D1in[:, -1, :, :]

    carac_0 = (- B2_0 + hr2u_B2[:, -1, :, :] / hrru_B2[:, -1, :, :] * Br_0 + D1_0 / N.sqrt(sqrt_det_h_B2**2 * hrru_B2)[:, -1, :, :])

    diff_B2u[:, -1, :, :]  -= dtin * sig_in * 0.5 * carac_0 * lambda_0 / dl / P_int_2[0]


def compute_penalty_D(p0, p1, dtin, Drin, D1in, D2in, Brin, B1in, B2in):

    top = topology[p0, p1]
    
    if (top == 'xx'):

        #######
        # Dr
        #######

        # lambda_0 = alpha_Dr[p0, :, -1, :] * N.sqrt(h11u_Dr[p0, :, -1, :]) * sqrt_det_h_Dr[p0, :, -1, :]
        # lambda_1 = alpha_Dr[p1, :, 0, :] * N.sqrt(h11u_Dr[p1, :, -1, :]) * sqrt_det_h_Dr[p1, :, -1, :]

        lambda_0 = alpha_Dr[p0, :, -1, :] * N.sqrt(h11u_Dr[p0, :, -1, :])
        lambda_1 = alpha_Dr[p1, :, 0, :]  * N.sqrt(h11u_Dr[p1, :, 0, :]) 
        
        Dr_0 = Drin[p0, :, -1, :]
        D1_0 = interp_int_to_half(D1in[p0, :, -1, :])
        B2_0 = B2in[p0, :, -1, :]

        Dr_1 = Drin[p1, :, 0, :]
        D1_1, D2_1 = transform_vect(p1, p0, xi_int[None, 0], eta_int[None, :], interp_int_to_half(D1in[p1, :, 0, :]), interp(interp_int_to_half(D2in[p1, :, 0, :]), eta_half, eta_int, 1))
        B2_1 = B2in[p1, :, 0, :]

        carac_0 = (Dr_0 - hr1u_Dr[p0, :, -1, :] / h11u_Dr[p0, :, -1, :] * D1_0 + B2_0 / N.sqrt(sqrt_det_h_Dr**2 * h11u_Dr)[p0, :, -1, :])
        carac_1 = (Dr_1 - hr1u_Dr[p0, :, -1, :] / h11u_Dr[p0, :, -1, :] * D1_1 + B2_1 / N.sqrt(sqrt_det_h_Dr**2 * h11u_Dr)[p0, :, -1, :])

        diff_Dru[p0, :, -1, :] += dtin * sig_in * 0.5 * (carac_0 - carac_1) * lambda_0 / dxi / P_int_2[0]

        Dr_0 = Drin[p0, :, -1, :]
        D1_0, D2_0 = transform_vect(p0, p1, xi_int[None, -1], eta_int[None, :], interp_int_to_half(D1in[p0, :, -1, :]), interp(interp_int_to_half(D2in[p0, :, -1, :]), eta_half, eta_int, 1))
        B2_0 = B2in[p0, :, -1, :]

        Dr_1 = Drin[p1, :, 0, :]
        D1_1 = interp_int_to_half(D1in[p1, :, 0, :])
        B2_1 = B2in[p1, :, 0, :]
        
        carac_1 = (Dr_1 - hr1u_Dr[p1, :, 0, :] / h11u_Dr[p1, :, 0, :] * D1_1 - B2_1 / N.sqrt(sqrt_det_h_Dr**2 * h11u_Dr)[p1, :, 0, :])
        carac_0 = (Dr_0 - hr1u_Dr[p1, :, 0, :] / h11u_Dr[p1, :, 0, :] * D1_0 - B2_0 / N.sqrt(sqrt_det_h_Dr**2 * h11u_Dr)[p1, :, 0, :])
        
        diff_Dru[p1, :, 0, :]  += dtin * sig_in * 0.5 * (carac_1 - carac_0) * lambda_1 / dxi / P_int_2[0]

        #######
        # D2
        #######

        # lambda_0 = alpha_D2[p0, :, -1, :] * N.sqrt(h11u_D2[p0, :, -1, :]) * sqrt_det_h_D2[p0, :, -1, :]
        # lambda_1 = alpha_D2[p1, :, 0, :]  * N.sqrt(h11u_D2[p1, :, 0, :])  * sqrt_det_h_D2[p1, :, 0, :]

        lambda_0 = alpha_D2[p0, :, -1, :] * N.sqrt(h11u_D2[p0, :, -1, :])
        lambda_1 = alpha_D2[p1, :, 0, :]  * N.sqrt(h11u_D2[p1, :, 0, :]) 

        D1_0 = interp(D1in[p0, :, -1, :], eta_int, eta_half, 1)
        D2_0 = D2in[p0, :, -1, :]
        Br_0 = Brin[p0, :, -1, :]
        
        D1_1, D2_1 = transform_vect(p1, p0, xi_half[None, 0], eta_half[None, :], interp(D1in[p1, :, 0, :], eta_int, eta_half, 1), D2in[p1, :, 0, :])
        Br_1 = Brin[p1, :, 0, :]
        
        carac_0 = (D2_0 - h12u_D2[p0, :, -1, :] / h11u_D2[p0, :, -1, :] * D1_0 - Br_0 / N.sqrt(sqrt_det_h_D2**2 * h11u_D2)[p0, :, -1, :])
        carac_1 = (D2_1 - h12u_D2[p0, :, -1, :] / h11u_D2[p0, :, -1, :] * D1_1 - Br_1 / N.sqrt(sqrt_det_h_D2**2 * h11u_D2)[p0, :, -1, :])

        diff_D2u[p0, :, -1, :] += dtin * sig_in * 0.5 * (carac_0 - carac_1) * lambda_0 / dxi / P_half_2[0]

        D1_0, D2_0 = transform_vect(p0, p1, xi_half[None, -1], eta_half[None, :], interp(D1in[p0, :, -1, :], eta_int, eta_half, 1), D2in[p0, :, -1, :])
        Br_0 = Brin[p0, :, -1, :]

        D1_1 = interp(D1in[p1, :, 0, :], eta_int, eta_half, 1)
        D2_1 = D2in[p1, :, 0, :]
        Br_1 = Brin[p1, :, 0, :]        

        carac_1 = (D2_1 - h12u_D2[p1, :, 0, :] / h11u_D2[p1, :, 0, :] * D1_1 + Br_1 / N.sqrt(sqrt_det_h_D2**2 * h11u_D2)[p1, :, 0, :])
        carac_0 = (D2_0 - h12u_D2[p1, :, 0, :] / h11u_D2[p1, :, 0, :] * D1_0 + Br_0 / N.sqrt(sqrt_det_h_D2**2 * h11u_D2)[p1, :, 0, :])
        
        diff_D2u[p1, :, 0, :]  += dtin * sig_in * 0.5 * (carac_1 - carac_0) * lambda_1 / dxi / P_half_2[0]

    if (top == 'xy'):

        #######
        # Dr
        #######

        # lambda_0 = alpha_Dr[p0, :, -1, :] * N.sqrt(h11u_Dr[p0, :, -1, :]) * sqrt_det_h_Dr[p0, :, -1, :]
        # lambda_1 = alpha_Dr[p1, :, :, 0] * N.sqrt(h22u_Dr[p1, :, :, 0]) * sqrt_det_h_Dr[p1, :, :, 0]

        lambda_0 = alpha_Dr[p0, :, -1, :]  * N.sqrt(h11u_Dr[p0, :, -1, :]) 
        lambda_1 = alpha_Dr[p1, :, :, 0] * N.sqrt(h22u_Dr[p1, :, :, 0])

        Dr_0 = Drin[p0, :, -1, :]
        D1_0 = interp_int_to_half(D1in[p0, :, -1, :])
        B2_0 = B2in[p0, :, -1, :]

        Dr_1 = Drin[p1, :, :, 0]
        D1_1, D2_1 = transform_vect(p1, p0, xi_int[None, :], eta_int[None, 0], interp(interp_int_to_half(D1in[p1, :, :, 0]), xi_half, xi_int, 1), interp_int_to_half(D2in[p1, :, :, 0]))
        B1_1 = B1in[p1, :, :, 0]
        
        carac_0 = (Dr_0          - hr1u_Dr[p0, :, -1, :] / h11u_Dr[p0, :, -1, :] * D1_0          + B2_0          / N.sqrt(sqrt_det_h_Dr**2 * h11u_Dr)[p0, :, -1, :])
        carac_1 = (Dr_1[:, ::-1] - hr1u_Dr[p0, :, -1, :] / h11u_Dr[p0, :, -1, :] * D1_1[:, ::-1] - B1_1[:, ::-1] / N.sqrt(sqrt_det_h_Dr**2 * h11u_Dr)[p0, :, -1, :])
        
        diff_Dru[p0, :, -1, :] += dtin * sig_in * 0.5 * (carac_0 - carac_1) * lambda_0 / dxi / P_int_2[0]

        Dr_0 = Drin[p0, :, -1, :]
        D1_0, D2_0 = transform_vect(p0, p1, xi_int[None, -1], eta_int[None, :], interp_int_to_half(D1in[p0, :, -1, :]), interp(interp_int_to_half(D2in[p0, :, -1, :]), eta_half, eta_int, 1))
        B2_0 = B2in[p0, :, -1, :]

        Dr_1 = Drin[p1, :, :, 0]
        D2_1 = interp_int_to_half(D2in[p1, :, :, 0])
        B1_1 = B1in[p1, :, :, 0]
        
        carac_1 = (Dr_1          - hr2u_Dr[p1, :, :, 0] / h22u_Dr[p1, :, :, 0] * D2_1          + B1_1          / N.sqrt(sqrt_det_h_Dr**2 * h22u_Dr)[p1, :, :, 0])
        carac_0 = (Dr_0[:, ::-1] - hr2u_Dr[p1, :, :, 0] / h22u_Dr[p1, :, :, 0] * D2_0[:, ::-1] - B2_0[:, ::-1] / N.sqrt(sqrt_det_h_Dr**2 * h22u_Dr)[p1, :, :, 0])
        
        diff_Dru[p1, :, :, 0]  += dtin * sig_in * 0.5 * (carac_1 - carac_0) * lambda_1 / dxi / P_int_2[0]

        #######
        # D1, D2
        #######

        # lambda_0 = alpha_D2[p0, :, -1, :]  * N.sqrt(h11u_D2[p0, :, -1, :])  * sqrt_det_h_D2[p0, :, -1, :]
        # lambda_1 = alpha_D1[p1, :, :, 0] * N.sqrt(h22u_D1[p1, :, :, 0]) * sqrt_det_h_D1[p1, :, :, 0]

        lambda_0 = alpha_D2[p0, :, -1, :]  * N.sqrt(h11u_D2[p0, :, -1, :]) 
        lambda_1 = alpha_D1[p1, :, :, 0] * N.sqrt(h22u_D1[p1, :, :, 0])

        D1_0 = interp(D1in[p0, :, -1, :], eta_int, eta_half, 1)
        D2_0 = D2in[p0, :, -1, :]
        Br_0 = Brin[p0, :, -1, :]

        D1_1, D2_1 = transform_vect(p1, p0, xi_half[None, :], eta_half[None, 0], D1in[p1, :, :, 0], interp(D2in[p1, :, :, 0], xi_int, xi_half, 1))
        Br_1 = Brin[p1, :, :, 0]

        carac_0 = (D2_0          - h12u_D2[p0, :, -1, :] / h11u_D2[p0, :, -1, :] * D1_0          - Br_0          / N.sqrt(sqrt_det_h_D2**2 * h11u_D2)[p0, :, -1, :])
        carac_1 = (D2_1[:, ::-1] - h12u_D2[p0, :, -1, :] / h11u_D2[p0, :, -1, :] * D1_1[:, ::-1] - Br_1[:, ::-1] / N.sqrt(sqrt_det_h_D2**2 * h11u_D2)[p0, :, -1, :])
        
        diff_D2u[p0, :, -1, :] += dtin * sig_in * 0.5 * (carac_0 - carac_1) * lambda_0 / dxi / P_half_2[0]

        D1_0, D2_0 = transform_vect(p0, p1, xi_half[None, -1], eta_half[None, :], interp(D1in[p0, :, -1, :], eta_int, eta_half, 1), D2in[p0, :, -1, :])
        Br_0 = Brin[p0, :, -1, :]

        D1_1 = D1in[p1, :, :, 0]
        D2_1 = interp(D2in[p1, :, :, 0], xi_int, xi_half, 1)
        Br_1 = Brin[p1, :, :, 0]
        
        carac_1 = (D1_1          - h12u_D1[p1, :, :, 0] / h22u_D1[p1, :, :, 0] * D2_1          - Br_1          / N.sqrt(sqrt_det_h_D1**2 * h22u_D1)[p1, :, :, 0])
        carac_0 = (D1_0[:, ::-1] - h12u_D1[p1, :, :, 0] / h22u_D1[p1, :, :, 0] * D2_0[:, ::-1] - Br_0[:, ::-1] / N.sqrt(sqrt_det_h_D1**2 * h22u_D1)[p1, :, :, 0])
        
        diff_D1u[p1, :, :, 0]  += dtin * sig_in * 0.5 * (carac_1 - carac_0) * lambda_1 / dxi / P_half_2[0]

    if (top == 'yy'):

        #######
        # Dr
        #######

        # lambda_0 = alpha_Dr[p0, :, :, -1]    * N.sqrt(h22u_Dr[p0, :, :, -1])    * sqrt_det_h_Dr[p0, :, :, -1]
        # lambda_1 = alpha_Dr[p1, :, :, 0] * N.sqrt(h22u_Dr[p1, :, :, 0]) * sqrt_det_h_Dr[p1, :, :, 0]

        lambda_0 = alpha_Dr[p0, :, :, -1]    * N.sqrt(h22u_Dr[p0, :, :, -1])   
        lambda_1 = alpha_Dr[p1, :, :, 0] * N.sqrt(h22u_Dr[p1, :, :, 0])

        Dr_0 = Drin[p0, :, :, -1]
        D2_0 = interp_int_to_half(D2in[p0, :, :, -1])
        B1_0 = B1in[p0, :, :, -1]
        
        Dr_1 = Drin[p1, :, :, 0]
        D1_1, D2_1 = transform_vect(p1, p0, xi_int[None, :], eta_int[None, 0], interp(interp_int_to_half(D1in[p1, :, :, 0]), xi_half, xi_int, 1), interp_int_to_half(D2in[p1, :, :, 0]))
        B1_1 = B1in[p1, :, :, 0]

        carac_0 = (Dr_0 - hr2u_Dr[p0, :, :, -1] / h22u_Dr[p0, :, :, -1] * D2_0 - B1_0 / N.sqrt(sqrt_det_h_Dr**2 * h22u_Dr)[p0, :, :, -1])
        carac_1 = (Dr_1 - hr2u_Dr[p0, :, :, -1] / h22u_Dr[p0, :, :, -1] * D2_1 - B1_1 / N.sqrt(sqrt_det_h_Dr**2 * h22u_Dr)[p0, :, :, -1])
        
        diff_Dru[p0, :, :, -1] += dtin * sig_in * 0.5 * (carac_0 - carac_1) * lambda_0 / dxi / P_int_2[0]

        Dr_0 = Drin[p0, :, :, -1]
        D1_0, D2_0 = transform_vect(p0, p1, xi_int[None, :], eta_int[None, -1], interp(interp_int_to_half(D1in[p0, :, :, -1]), xi_half, xi_int, 1), interp_int_to_half(D2in[p0, :, :, -1]))
        B1_0 = B1in[p0, :, :, -1]
        
        Dr_1 = Drin[p1, :, :, 0]
        D2_1 = interp_int_to_half(D2in[p1, :, :, 0])
        B1_1 = B1in[p1, :, :, 0]   

        carac_1 = (Dr_1 - hr2u_Dr[p1, :, :, 0] / h22u_Dr[p1, :, :, 0] * D2_1 + B1_1 / N.sqrt(sqrt_det_h_Dr**2 * h22u_Dr)[p1, :, :, 0])
        carac_0 = (Dr_0 - hr2u_Dr[p1, :, :, 0] / h22u_Dr[p1, :, :, 0] * D2_0 + B1_0 / N.sqrt(sqrt_det_h_Dr**2 * h22u_Dr)[p1, :, :, 0])    
    
        diff_Dru[p1, :, :, 0]  += dtin * sig_in * 0.5 * (carac_1 - carac_0) * lambda_1 / dxi / P_int_2[0]

        #######
        # D1
        #######

        # lambda_0 = alpha_D1[p0, :, :, -1]    * N.sqrt(h22u_D1[p0, :, :, -1])    * sqrt_det_h_D1[p0, :, :, -1]
        # lambda_1 = alpha_D1[p1, :, :, 0] * N.sqrt(h22u_D1[p1, :, :, 0]) * sqrt_det_h_D1[p1, :, :, 0]

        lambda_0 = alpha_D1[p0, :, :, -1]    * N.sqrt(h22u_D1[p0, :, :, -1])   
        lambda_1 = alpha_D1[p1, :, :, 0] * N.sqrt(h22u_D1[p1, :, :, 0])

        D1_0 = D1in[p0, :, :, -1]
        D2_0 = interp(D2in[p0, :, :, -1], xi_int, xi_half, 1)
        Br_0 = Brin[p0, :, :, -1]

        D1_1, D2_1 = transform_vect(p1, p0, xi_half[None, :], eta_half[None, 0], D1in[p1, :, :, 0], interp(D2in[p1, :, :, 0], xi_int, xi_half, 1))
        Br_1 = Brin[p1, :, :, 0]
        
        carac_0 = (D1_0 - h12u_D1[p0, :, :, -1] / h22u_D1[p0, :, :, -1] * D2_0 + Br_0 / N.sqrt(sqrt_det_h_D1**2 * h22u_D1)[p0, :, :, -1])
        carac_1 = (D1_1 - h12u_D1[p0, :, :, -1] / h22u_D1[p0, :, :, -1] * D2_1 + Br_1 / N.sqrt(sqrt_det_h_D1**2 * h22u_D1)[p0, :, :, -1])
        
        diff_D1u[p0, :, :, -1] += dtin * sig_in * 0.5 * (carac_0 - carac_1) * lambda_0 / dxi / P_half_2[0]

        D1_0, D2_0 = transform_vect(p0, p1, xi_half[None, :], eta_half[None, -1], D1in[p0, :, :, -1], interp(D2in[p0, :, :, -1], xi_int, xi_half, 1))
        Br_0 = Brin[p0, :, :, -1]

        D1_1 = D1in[p1, :, :, 0]
        D2_1 = interp(D2in[p1, :, :, 0], xi_int, xi_half, 1)
        Br_1 = Brin[p1, :, :, 0]

        carac_1 = (D1_1 - h12u_D1[p1, :, :, 0] / h22u_D1[p1, :, :, 0] * D2_1 - Br_1 / N.sqrt(sqrt_det_h_D1**2 * h22u_D1)[p1, :, :, 0])
        carac_0 = (D1_0 - h12u_D1[p1, :, :, 0] / h22u_D1[p1, :, :, 0] * D2_0 - Br_0 / N.sqrt(sqrt_det_h_D1**2 * h22u_D1)[p1, :, :, 0]) 
    
        diff_D1u[p1, :, :, 0]  += dtin * sig_in * 0.5 * (carac_1 - carac_0) * lambda_1 / dxi / P_half_2[0]

    if (top == 'yx'):

        #######
        # Dr
        #######

        # lambda_0 = alpha_Dr[p0, :, :, -1]  * N.sqrt(h22u_Dr[p0, :, :, -1])  * sqrt_det_h_Dr[p0, :, :, -1]
        # lambda_1 = alpha_Dr[p1, :, 0, :] * N.sqrt(h11u_Dr[p1, :, 0, :]) * sqrt_det_h_Dr[p1, :, 0, :] 

        lambda_0 = alpha_Dr[p0, :, :, -1]  * N.sqrt(h22u_Dr[p0, :, :, -1]) 
        lambda_1 = alpha_Dr[p1, :, 0, :] * N.sqrt(h11u_Dr[p1, :, 0, :])

        Dr_0 = Drin[p0, :, :, -1]
        D2_0 = interp_int_to_half(D2in[p0, :, :, -1])
        B1_0 = B1in[p0, :, :, -1]

        Dr_1 = Drin[p1, :, 0, :]
        D1_1, D2_1 = transform_vect(p1, p0, xi_int[None, 0], eta_int[None, :], interp_int_to_half(D1in[p1, :, 0, :]), interp(interp_int_to_half(D2in[p1, :, 0, :]), eta_half, eta_int, 1))
        B2_1 = B2in[p1, :, 0, :]

        carac_0 = (Dr_0          - hr2u_Dr[p0, :, :, -1] / h22u_Dr[p0, :, :, -1] * D2_0          - B1_0          / N.sqrt(sqrt_det_h_Dr**2 * h22u_Dr)[p0, :, :, -1])
        carac_1 = (Dr_1[:, ::-1] - hr2u_Dr[p0, :, :, -1] / h22u_Dr[p0, :, :, -1] * D2_1[:, ::-1] + B2_1[:, ::-1] / N.sqrt(sqrt_det_h_Dr**2 * h22u_Dr)[p0, :, :, -1])

        diff_Dru[p0, :, :, -1] += dtin * sig_in * 0.5 * (carac_0 - carac_1) * lambda_0 / dxi / P_int_2[0]
        
        Dr_0 = Drin[p0, :, :, -1]
        D1_0, D2_0 = transform_vect(p0, p1, xi_int[None, :], eta_int[None, -1], interp(interp_int_to_half(D1in[p0, :, :, -1]), xi_half, xi_int, 1), interp_int_to_half(D2in[p0, :, :, -1]))
        B1_0 = B1in[p0, :, :, -1]

        Dr_1 = Drin[p1, :, 0, :]
        D1_1 = interp_int_to_half(D1in[p1, :, 0, :])
        B2_1 = B2in[p1, :, 0, :]        
    
        carac_1 = (Dr_1          - hr1u_Dr[p1, :, 0, :] / h11u_Dr[p1, :, 0, :] * D1_1          - B2_1          / N.sqrt(sqrt_det_h_Dr**2 * h11u_Dr)[p1, :, 0, :])
        carac_0 = (Dr_0[:, ::-1] - hr1u_Dr[p1, :, 0, :] / h11u_Dr[p1, :, 0, :] * D1_0[:, ::-1] + B1_0[:, ::-1] / N.sqrt(sqrt_det_h_Dr**2 * h11u_Dr)[p1, :, 0, :])
        
        diff_Dru[p1, :, 0, :]  += dtin * sig_in * 0.5 * (carac_1 - carac_0) * lambda_1 / dxi / P_int_2[0]

        #######
        # D1, D2
        #######
        
        # lambda_0 = alpha_D1[p0, :, :, -1]  * N.sqrt(h22u_D1[p0, :, :, -1])  * sqrt_det_h_D1[p0, :, :, -1]
        # lambda_1 = alpha_D2[p1, :, 0, :] * N.sqrt(h11u_D2[p1, :, 0, :]) * sqrt_det_h_D2[p1, :, 0, :]

        lambda_0 = alpha_D1[p0, :, :, -1]  * N.sqrt(h22u_D1[p0, :, :, -1]) 
        lambda_1 = alpha_D2[p1, :, 0, :] * N.sqrt(h11u_D2[p1, :, 0, :])

        D1_0 = D1in[p0, :, :, -1]
        D2_0 = interp(D2in[p0, :, :, -1], xi_int, xi_half, 1)
        Br_0 = Brin[p0, :, :, -1]

        D1_1, D2_1 = transform_vect(p1, p0, xi_half[None, 0], eta_half[None, :], interp(D1in[p1, :, 0, :], eta_int, eta_half, 1), D2in[p1, :, 0, :])
        Br_1 = Brin[p1, :, 0, :]

        carac_0 = (D1_0          - h12u_D1[p0, :, :, -1] / h22u_D1[p0, :, :, -1] * D2_0          + Br_0          / N.sqrt(sqrt_det_h_D1**2 * h22u_D1)[p0, :, :, -1])
        carac_1 = (D1_1[:, ::-1] - h12u_D1[p0, :, :, -1] / h22u_D1[p0, :, :, -1] * D2_1[:, ::-1] + Br_1[:, ::-1] / N.sqrt(sqrt_det_h_D1**2 * h22u_D1)[p0, :, :, -1])

        diff_D1u[p0, :, :, -1] += dtin * sig_in * 0.5 * (carac_0 - carac_1) * lambda_0 / dxi / P_half_2[0]

        D1_0, D2_0 = transform_vect(p0, p1, xi_half[None, :], eta_half[None, -1], D1in[p0, :, :, -1], interp(D2in[p0, :, :, -1], xi_int, xi_half, 1))
        Br_0 = Brin[p0, :, :, -1]

        D1_1 = interp(D1in[p1, :, 0, :], eta_int, eta_half, 1)
        D2_1 = D2in[p1, :, 0, :]
        Br_1 = Brin[p1, :, 0, :]

        carac_1 = (D2_1          - h12u_D2[p1, :, 0, :] / h11u_D2[p1, :, 0, :] * D1_1          + Br_1          / N.sqrt(sqrt_det_h_D2**2 * h11u_D2)[p1, :, 0, :])
        carac_0 = (D2_0[:, ::-1] - h12u_D2[p1, :, 0, :] / h11u_D2[p1, :, 0, :] * D1_0[:, ::-1] + Br_0[:, ::-1] / N.sqrt(sqrt_det_h_D2**2 * h11u_D2)[p1, :, 0, :])

        diff_D2u[p1, :, 0, :]  += dtin * sig_in * 0.5 * (carac_1 - carac_0) * lambda_1 / dxi / P_half_2[0]


def compute_penalty_B(p0, p1, dtin, Drin, D1in, D2in, Brin, B1in, B2in):

    top = topology[p0, p1]
    
    if (top == 'xx'):

        #######
        # Br
        #######

        # lambda_0 = alpha_Br[p0, :,  -1, :] * N.sqrt(h11u_Br[p0, :,  -1, :]) * sqrt_det_h_Br[p0, :, -1, :]
        # lambda_1 = alpha_Br[p1, :,  0, :]  * N.sqrt(h11u_Br[p1, :,  0, :])  * sqrt_det_h_Br[p1, :, 0, :]

        lambda_0 = alpha_Br[p0, :,  -1, :] * N.sqrt(h11u_Br[p0, :,  -1, :])
        lambda_1 = alpha_Br[p1, :,  0, :]  * N.sqrt(h11u_Br[p1, :,  0, :]) 

        D2_0 = D2in[p0, :,  -1, :]
        Br_0 = Brin[p0, :,  -1, :]
        B1_0 = interp_half_to_int(B1in[p0, :,  -1, :])

        D2_1 = D2in[p1, :,  0, :]
        Br_1 = Brin[p1, :,  0, :]
        B1_1, B2_1 = transform_vect(p1, p0, xi_half[None, 0], eta_half[None, :], interp_half_to_int(B1in[p1, :,  0, :]), interp(interp_half_to_int(B2in[p1, :,  0, :]), eta_int, eta_half, 1))

        carac_0 = (- D2_0 / N.sqrt(sqrt_det_h_Br**2 * h11u_Br)[p0, :,  -1, :] + Br_0 - hr1u_Br[p0, :,  -1, :] / h11u_Br[p0, :,  -1, :] * B1_0)
        carac_1 = (- D2_1 / N.sqrt(sqrt_det_h_Br**2 * h11u_Br)[p0, :,  -1, :] + Br_1 - hr1u_Br[p0, :,  -1, :] / h11u_Br[p0, :,  -1, :] * B1_1)

        diff_Bru[p0, :,  -1, :] += dtin * sig_in * 0.5 * (carac_0 - carac_1) * lambda_0 / dxi / P_half_2[0]

        D2_0 = D2in[p0, :,  -1, :]
        Br_0 = Brin[p0, :,  -1, :]
        B1_0, B2_0 = transform_vect(p0, p1, xi_half[None, -1], eta_half[None, :], interp_half_to_int(B1in[p0, :,  -1, :]), interp(interp_half_to_int(B2in[p0, :,  -1, :]), eta_int, eta_half, 1))

        D2_1 = D2in[p1, :,  0, :]
        Br_1 = Brin[p1, :,  0, :]
        B1_1 = interp_half_to_int(B1in[p1, :,  0, :])
 
        carac_1 = (D2_1 / N.sqrt(sqrt_det_h_Br**2 * h11u_Br)[p1, :,  0, :] + Br_1 - hr1u_Br[p1, :,  0, :] / h11u_Br[p1, :,  0, :] * B1_1)
        carac_0 = (D2_0 / N.sqrt(sqrt_det_h_Br**2 * h11u_Br)[p1, :,  0, :] + Br_0 - hr1u_Br[p1, :,  0, :] / h11u_Br[p1, :,  0, :] * B1_0)
        
        diff_Bru[p1, :,  0, :]  += dtin * sig_in * 0.5 * (carac_1 - carac_0) * lambda_1 / dxi / P_half_2[0]

        #######
        # B2
        #######

        # lambda_0 = alpha_B2[p0, :,  -1, :] * N.sqrt(h11u_B2[p0, :,  -1, :]) * sqrt_det_h_B2[p0, :,  -1, :]
        # lambda_1 = alpha_B2[p1, :,  0, :]  * N.sqrt(h11u_B2[p1, :,  0, :])  * sqrt_det_h_B2[p1, :,  0, :]

        lambda_0 = alpha_B2[p0, :,  -1, :] * N.sqrt(h11u_B2[p0, :,  -1, :])
        lambda_1 = alpha_B2[p1, :,  0, :]  * N.sqrt(h11u_B2[p1, :,  0, :]) 

        Dr_0 = Drin[p0, :,  -1, :]
        B1_0 = interp(B1in[p0, :,  -1, :], eta_half, eta_int, 1)
        B2_0 = B2in[p0, :,  -1, :]

        Dr_1 = Drin[p1, :,  0, :]
        B1_1, B2_1 = transform_vect(p1, p0, xi_int[None, 0], eta_int[None, :], interp(B1in[p1, :,  0, :], eta_half, eta_int, 1), B2in[p1, :,  0, :])

        carac_0 = (Dr_0 / N.sqrt(sqrt_det_h_B2**2 * h11u_B2)[p0, :,  -1, :] + B2_0 - h12u_B2[p0, :,  -1, :] / h11u_B2[p0, :,  -1, :] * B1_0)
        carac_1 = (Dr_1 / N.sqrt(sqrt_det_h_B2**2 * h11u_B2)[p0, :,  -1, :] + B2_1 - h12u_B2[p0, :,  -1, :] / h11u_B2[p0, :,  -1, :] * B1_1)

        diff_B2u[p0, :,  -1, :] += dtin * sig_in * 0.5 * (carac_0 - carac_1) * lambda_0 / dxi / P_int_2[0]

        Dr_0 = Drin[p0, :,  -1, :]
        B1_0, B2_0 = transform_vect(p0, p1, xi_int[None, -1], eta_int[None, :], interp(B1in[p0, :,  -1, :], eta_half, eta_int, 1), B2in[p0, :,  -1, :])

        Dr_1 = Drin[p1, :,  0, :]
        B1_1 = interp(B1in[p1, :,  0, :], eta_half, eta_int, 1)
        B2_1 = B2in[p1, :,  0, :]        

        carac_1 = (- Dr_1 / N.sqrt(sqrt_det_h_B2**2 * h11u_B2)[p1, :,  0, :] + B2_1 - h12u_B2[p1, :,  0, :] / h11u_B2[p1, :,  0, :] * B1_1)
        carac_0 = (- Dr_0 / N.sqrt(sqrt_det_h_B2**2 * h11u_B2)[p1, :,  0, :] + B2_0 - h12u_B2[p1, :,  0, :] / h11u_B2[p1, :,  0, :] * B1_0)
        
        diff_B2u[p1, :,  0, :]  += dtin * sig_in * 0.5 * (carac_1 - carac_0) * lambda_1 / dxi / P_int_2[0]

    if (top == 'xy'):

        #######
        # Br
        #######

        # lambda_0 = alpha_Br[p0, :,  -1, :]  * N.sqrt(h11u_Br[p0, :,  -1, :])  * sqrt_det_h_Br[p0, :,  -1, :]
        # lambda_1 = alpha_Br[p1, :,  :, 0] * N.sqrt(h22u_Br[p1, :,  :, 0]) * sqrt_det_h_Br[p1, :,  :, 0]

        lambda_0 = alpha_Br[p0, :,  -1, :]  * N.sqrt(h11u_Br[p0, :,  -1, :]) 
        lambda_1 = alpha_Br[p1, :,  :, 0] * N.sqrt(h22u_Br[p1, :,  :, 0])

        D2_0 = D2in[p0, :,  -1, :]
        Br_0 = Brin[p0, :,  -1, :]
        B1_0 = interp_half_to_int(B1in[p0, :,  -1, :])

        D1_1 = D1in[p1, :,  :, 0]
        Br_1 = Brin[p1, :,  :, 0]
        B1_1, B2_1 = transform_vect(p1, p0, xi_half[None, :], eta_half[None, 0], interp(interp_half_to_int(B1in[p1, :,  :, 0]), xi_int, xi_half, 1), interp_half_to_int(B2in[p1, :,  :, 0]))

        carac_0 = (- D2_0          / N.sqrt(sqrt_det_h_Br**2 * h11u_Br)[p0, :,  -1, :] + Br_0          - hr1u_Br[p0, :,  -1, :] / h11u_Br[p0, :,  -1, :] * B1_0)
        carac_1 = (  D1_1[:, ::-1] / N.sqrt(sqrt_det_h_Br**2 * h11u_Br)[p0, :,  -1, :] + Br_1[:, ::-1] - hr1u_Br[p0, :,  -1, :] / h11u_Br[p0, :,  -1, :] * B1_1[:, ::-1])
        
        diff_Bru[p0, :,  -1, :] += dtin * sig_in * 0.5 * (carac_0 - carac_1) * lambda_0 / dxi / P_half_2[0]

        D2_0 = D2in[p0, :,  -1, :]
        Br_0 = Brin[p0, :,  -1, :]
        B1_0, B2_0 = transform_vect(p0, p1, xi_half[None, -1], eta_half[None, :], interp_half_to_int(B1in[p0, :,  -1, :]), interp(interp_half_to_int(B2in[p0, :,  -1, :]), eta_int, eta_half, 1))

        D1_1 = D1in[p1, :,  :, 0]
        Br_1 = Brin[p1, :,  :, 0]
        B2_1 = interp_half_to_int(B2in[p1, :,  :, 0])
        
        carac_1 = (- D1_1          / N.sqrt(sqrt_det_h_Br**2 * h22u_Br)[p1, :,  :, 0] + Br_1          - hr2u_Br[p1, :,  :, 0] / h22u_Br[p1, :,  :, 0] * B2_1)
        carac_0 = (  D2_0[:, ::-1] / N.sqrt(sqrt_det_h_Br**2 * h22u_Br)[p1, :,  :, 0] + Br_0[:, ::-1] - hr2u_Br[p1, :,  :, 0] / h22u_Br[p1, :,  :, 0] * B2_0[:, ::-1])
        
        diff_Bru[p1, :,  :, 0]  += dtin * sig_in * 0.5 * (carac_1 - carac_0) * lambda_1 / dxi / P_half_2[0]

        #######
        # B1, B2
        #######

        # lambda_0 = alpha_B2[p0, :,  -1, :]  * N.sqrt(h11u_B2[p0, :,  -1, :])  * sqrt_det_h_B2[p0, :,  -1, :]
        # lambda_1 = alpha_B1[p1, :,  :, 0] * N.sqrt(h22u_B1[p1, :,  :, 0]) * sqrt_det_h_B1[p1, :,  :, 0]

        lambda_0 = alpha_B2[p0, :,  -1, :]  * N.sqrt(h11u_B2[p0, :,  -1, :]) 
        lambda_1 = alpha_B1[p1, :,  :, 0] * N.sqrt(h22u_B1[p1, :,  :, 0])

        Dr_0 = Drin[p0, :,  -1, :]
        B1_0 = interp(B1in[p0, :,  -1, :], eta_half, eta_int, 1)
        B2_0 = B2in[p0, :,  -1, :]

        Dr_1 = Drin[p1, :,  :, 0]
        B1_1, B2_1 = transform_vect(p1, p0, xi_int[None, :], eta_int[None, 0], B1in[p1, :,  :, 0], interp(B2in[p1, :,  :, 0], xi_half, xi_int, 1))

        carac_0 = (Dr_0          / N.sqrt(sqrt_det_h_B2**2 * h11u_B2)[p0, :,  -1, :] + B2_0          - h12u_B2[p0, :,  -1, :] / h11u_B2[p0, :,  -1, :] * B1_0)
        carac_1 = (Dr_1[:, ::-1] / N.sqrt(sqrt_det_h_B2**2 * h11u_B2)[p0, :,  -1, :] + B2_1[:, ::-1] - h12u_B2[p0, :,  -1, :] / h11u_B2[p0, :,  -1, :] * B1_1[:, ::-1])

        diff_B2u[p0, :,  -1, :] += dtin * sig_in * 0.5 * (carac_0 - carac_1) * lambda_0 / dxi / P_int_2[0]

        Dr_0 = Drin[p0, :,  -1, :]
        B1_0, B2_0 = transform_vect(p0, p1, xi_int[None, -1], eta_int[None, :], interp(B1in[p0, :,  -1, :], eta_half, eta_int, 1), B2in[p0, :,  -1, :])

        Dr_1 = Drin[p1, :,  :, 0]
        B1_1 = B1in[p1, :,  :, 0]
        B2_1 = interp(B2in[p1, :,  :, 0], xi_half, xi_int, 1)
        
        carac_1 = (Dr_1          / N.sqrt(sqrt_det_h_B1**2 * h22u_B1)[p1, :,  :, 0] + B1_1          - h12u_B1[p1, :,  :, 0] / h22u_B1[p1, :,  :, 0] * B2_1)
        carac_0 = (Dr_0[:, ::-1] / N.sqrt(sqrt_det_h_B1**2 * h22u_B1)[p1, :,  :, 0] + B1_0[:, ::-1] - h12u_B1[p1, :,  :, 0] / h22u_B1[p1, :,  :, 0] * B2_0[:, ::-1])
        
        diff_B1u[p1, :,  :, 0]  += dtin * sig_in * 0.5 * (carac_1 - carac_0) * lambda_1 / dxi / P_int_2[0]

    if (top == 'yy'):
        
        #######
        # Br
        #######

        # lambda_0 = alpha_Br[p0, :,  :, -1]    * N.sqrt(h22u_Br[p0, :,  :, -1])    * sqrt_det_h_Br[p0, :,  :, -1]
        # lambda_1 = alpha_Br[p1, :,  :, 0] * N.sqrt(h22u_Br[p1, :,  :, 0]) * sqrt_det_h_Br[p1, :,  :, 0]
        
        lambda_0 = alpha_Br[p0, :,  :, -1]    * N.sqrt(h22u_Br[p0, :,  :, -1])   
        lambda_1 = alpha_Br[p1, :,  :, 0] * N.sqrt(h22u_Br[p1, :,  :, 0])

        D1_0 = D1in[p0, :,  :, -1]
        Br_0 = Brin[p0, :,  :, -1]
        B2_0 = interp_half_to_int(B2in[p0, :,  :, -1])

        D1_1 = D1in[p1, :,  :, 0]
        Br_1 = Brin[p1, :,  :, 0]
        B1_1, B2_1 = transform_vect(p1, p0, xi_half[None, :], eta_half[None, 0], interp(interp_half_to_int(B1in[p1, :,  :, 0]), xi_int, xi_half, 1), interp_half_to_int(B2in[p1, :,  :, 0]))

        carac_0 = (  D1_0 / N.sqrt(sqrt_det_h_Br**2 * h22u_Br)[p0, :,  :, -1] + Br_0 - hr2u_Br[p0, :,  :, -1] / h22u_Br[p0, :,  :, -1] * B2_0)
        carac_1 = (  D1_1 / N.sqrt(sqrt_det_h_Br**2 * h22u_Br)[p0, :,  :, -1] + Br_1 - hr2u_Br[p0, :,  :, -1] / h22u_Br[p0, :,  :, -1] * B2_1)

        diff_Bru[p0, :,  :, -1] += dtin * sig_in * 0.5 * (carac_0 - carac_1) * lambda_0 / dxi / P_half_2[0]

        D1_0 = D1in[p0, :,  :, -1]
        Br_0 = Brin[p0, :,  :, -1]
        B1_0, B2_0 = transform_vect(p0, p1, xi_half[None, :], eta_half[None, -1], interp(interp_half_to_int(B1in[p0, :,  :, -1]), xi_int, xi_half, 1), interp_half_to_int(B2in[p0, :,  :, -1]))

        D1_1 = D1in[p1, :,  :, 0]
        Br_1 = Brin[p1, :,  :, 0]
        B2_1 = interp_half_to_int(B2in[p1, :,  :, 0])

        carac_1 = (- D1_1 / N.sqrt(sqrt_det_h_Br**2 * h11u_Br)[p1, :,  :, 0] + Br_1 - hr2u_Br[p1, :,  :, 0] / h22u_Br[p1, :,  :, 0] * B2_1)
        carac_0 = (- D1_0 / N.sqrt(sqrt_det_h_Br**2 * h11u_Br)[p1, :,  :, 0] + Br_0 - hr2u_Br[p1, :,  :, 0] / h22u_Br[p1, :,  :, 0] * B2_0)
        
        diff_Bru[p1, :,  :, 0]  += dtin * sig_in * 0.5 * (carac_1 - carac_0) * lambda_1 / dxi / P_half_2[0]

        #######
        # B1
        #######

        # lambda_0 = alpha_B1[p0, :,  :, -1]    * N.sqrt(h22u_B1[p0, :,  :, -1])    * sqrt_det_h_B1[p0, :,  :, -1]
        # lambda_1 = alpha_B1[p1, :,  :, 0] * N.sqrt(h22u_B1[p1, :,  :, 0]) * sqrt_det_h_B1[p1, :,  :, 0]

        lambda_0 = alpha_B1[p0, :,  :, -1]    * N.sqrt(h22u_B1[p0, :,  :, -1])   
        lambda_1 = alpha_B1[p1, :,  :, 0] * N.sqrt(h22u_B1[p1, :,  :, 0])

        Dr_0 = Drin[p0, :,  :, -1]
        B1_0 = B1in[p0, :,  :, -1]
        B2_0 = interp(B2in[p0, :,  :, -1], xi_half, xi_int, 1)

        Dr_1 = Drin[p1, :,  :, 0]
        B1_1, B2_1 = transform_vect(p1, p0, xi_int[None, :], eta_int[None, 0], B1in[p1, :,  :, 0], interp(B2in[p1, :,  :, 0], xi_half, xi_int, 1))
        
        carac_0 = (- Dr_0 / N.sqrt(sqrt_det_h_B1**2 * h22u_B1)[p0, :,  :, -1] + B1_0 - h12u_B1[p0, :,  :, -1] / h22u_B1[p0, :,  :, -1] * B2_0)
        carac_1 = (- Dr_1 / N.sqrt(sqrt_det_h_B1**2 * h22u_B1)[p0, :,  :, -1] + B1_1 - h12u_B1[p0, :,  :, -1] / h22u_B1[p0, :,  :, -1] * B2_1)
        
        diff_B1u[p0, :,  :, -1] += dtin * sig_in * 0.5 * (carac_0 - carac_1) * lambda_0 / dxi / P_int_2[0]

        Dr_0 = Drin[p0, :,  :, -1]
        B1_0, B2_0 = transform_vect(p0, p1, xi_int[None, :], eta_int[None, -1], B1in[p0, :,  :, -1], interp(B2in[p0, :,  :, -1], xi_half, xi_int, 1))

        Dr_1 = Drin[p1, :,  :, 0]
        B1_1 = B1in[p1, :,  :, 0]
        B2_1 = interp(B2in[p1, :,  :, 0], xi_half, xi_int, 1)

        carac_1 = (Dr_1 / N.sqrt(sqrt_det_h_B1**2 * h22u_B1)[p1, :,  :, 0] + B1_1 - h12u_B1[p1, :,  :, 0] / h22u_B1[p1, :,  :, 0] * B2_1)
        carac_0 = (Dr_0 / N.sqrt(sqrt_det_h_B1**2 * h22u_B1)[p1, :,  :, 0] + B1_0 - h12u_B1[p1, :,  :, 0] / h22u_B1[p1, :,  :, 0] * B2_0)

        diff_B1u[p1, :,  :, 0]  += dtin * sig_in * 0.5 * (carac_1 - carac_0) * lambda_0 / dxi / P_int_2[0]
        
    if (top == 'yx'):

        #######
        # Br
        #######

        # lambda_0 = alpha_Br[p0, :,  :, -1]  * N.sqrt(h22u_Br[p0, :,  :, -1])  * sqrt_det_h_Br[p0, :,  :, -1]
        # lambda_1 = alpha_Br[p1, :,  0, :] * N.sqrt(h11u_Br[p1, :,  0, :]) * sqrt_det_h_Br[p1, :,  0, :]

        lambda_0 = alpha_Br[p0, :,  :, -1]  * N.sqrt(h22u_Br[p0, :,  :, -1]) 
        lambda_1 = alpha_Br[p1, :,  0, :] * N.sqrt(h11u_Br[p1, :,  0, :])

        D1_0 = D1in[p0, :,  :, -1]
        Br_0 = Brin[p0, :,  :, -1]
        B2_0 = interp_half_to_int(B2in[p0, :,  :, -1])

        D2_1 = D2in[p1, :,  0, :]
        Br_1 = Brin[p1, :,  0, :]
        B1_1, B2_1 = transform_vect(p1, p0, xi_half[None, 0], eta_half[None, :], interp_half_to_int(B1in[p1, :,  0, :]), interp(interp_half_to_int(B2in[p1, :,  0, :]), eta_int, eta_half, 1))

        carac_0 = (  D1_0          / N.sqrt(sqrt_det_h_Br**2 * h22u_Br)[p0, :,  :, -1] + Br_0          - hr2u_Br[p0, :,  :, -1] / h22u_Br[p0, :,  :, -1] * B2_0)
        carac_1 = (- D2_1[:, ::-1] / N.sqrt(sqrt_det_h_Br**2 * h22u_Br)[p0, :,  :, -1] + Br_1[:, ::-1] - hr2u_Br[p0, :,  :, -1] / h22u_Br[p0, :,  :, -1] * B2_1[:, ::-1])

        diff_Bru[p0, :,  :, -1] += dtin * sig_in * 0.5 * (carac_0 - carac_1) * lambda_0 / dxi / P_half_2[0]

        D1_0 = D1in[p0, :,  :, -1]
        Br_0 = Brin[p0, :,  :, -1]
        B1_0, B2_0 = transform_vect(p0, p1, xi_half[None, :], eta_half[None, -1], interp(interp_half_to_int(B1in[p0, :,  :, -1]), xi_int, xi_half, 1), interp_half_to_int(B2in[p0, :,  :, -1]))

        D2_1 = D2in[p1, :,  0, :]
        Br_1 = Brin[p1, :,  0, :]
        B1_1 = interp_half_to_int(B1in[p1, :,  0, :])
        
        carac_1 = (  D2_1          / N.sqrt(sqrt_det_h_Br**2 * h11u_Br)[p1, :,  0, :] + Br_1          - hr1u_Br[p1, :,  0, :] / h11u_Br[p1, :,  0, :] * B1_1)
        carac_0 = (- D1_0[:, ::-1] / N.sqrt(sqrt_det_h_Br**2 * h11u_Br)[p1, :,  0, :] + Br_0[:, ::-1] - hr1u_Br[p1, :,  0, :] / h11u_Br[p1, :,  0, :] * B1_0[:, ::-1])
        
        diff_Bru[p1, :,  0, :]  += dtin * sig_in * 0.5 * (carac_1 - carac_0) * lambda_1 / dxi / P_half_2[0]

        #######
        # B1, B2
        #######

        # lambda_0 = alpha_B1[p0, :,  :, -1]  * N.sqrt(h22u_B1[p0, :,  :, -1])  * sqrt_det_h_B1[p0, :,  :, -1]
        # lambda_1 = alpha_B2[p1, :,  0, :] * N.sqrt(h11u_B2[p1, :,  0, :]) * sqrt_det_h_B2[p1, :,  0, :]

        lambda_0 = alpha_B1[p0, :,  :, -1]  * N.sqrt(h22u_B1[p0, :,  :, -1]) 
        lambda_1 = alpha_B2[p1, :,  0, :] * N.sqrt(h11u_B2[p1, :,  0, :])

        Dr_0 = Drin[p0, :,  :, -1]
        B1_0 = B1in[p0, :,  :, -1]
        B2_0 = interp(B2in[p0, :,  :, -1], xi_half, xi_int, 1)

        Dr_1 = Drin[p1, :,  0, :]
        B1_1, B2_1 = transform_vect(p1, p0, xi_int[None, 0], eta_int[None, :], interp(B1in[p1, :,  0, :], eta_half, eta_int, 1), B2in[p1, :,  0, :])

        carac_0 = (- Dr_0          / N.sqrt(sqrt_det_h_B1**2 * h22u_B1)[p0, :,  :, -1] + B1_0          - h12u_B1[p0, :,  :, -1] / h22u_B1[p0, :,  :, -1] * B2_0)
        carac_1 = (- Dr_1[:, ::-1] / N.sqrt(sqrt_det_h_B1**2 * h22u_B1)[p0, :,  :, -1] + B1_1[:, ::-1] - h12u_B1[p0, :,  :, -1] / h22u_B1[p0, :,  :, -1] * B2_1[:, ::-1])

        diff_B1u[p0, :,  :, -1] += dtin * sig_in * 0.5 * (carac_0 - carac_1) * lambda_0 / dxi / P_int_2[0]

        Dr_0 = Drin[p0, :,  :, -1]
        B1_0, B2_0 = transform_vect(p0, p1, xi_int[None, :], eta_int[None, -1], B1in[p0, :,  :, -1], interp(B2in[p0, :,  :, -1], xi_half, xi_int, 1))

        Dr_1 = Drin[p1, :,  0, :]
        B1_1 = interp(B1in[p1, :,  0, :], xi_half, xi_int, 1)
        B2_1 = B2in[p1, :,  0, :]

        carac_1 = (- Dr_1          / N.sqrt(sqrt_det_h_B2**2 * h11u_B2)[p1, :,  0, :] + B2_1          - h12u_B2[p1, :,  0, :] / h11u_B2[p1, :,  0, :] * B1_1)
        carac_0 = (- Dr_0[:, ::-1] / N.sqrt(sqrt_det_h_B2**2 * h11u_B2)[p1, :,  0, :] + B2_0[:, ::-1] - h12u_B2[p1, :,  0, :] / h11u_B2[p1, :,  0, :] * B1_0[:, ::-1])

        diff_B2u[p1, :,  0, :]  += dtin * sig_in * 0.5 * (carac_1 - carac_0) * lambda_1 / dxi / P_int_2[0]

########
# Apply penalty terms to E, D
########

def interface_D(p0, p1, Drin, D1in, D2in):

    i0 =  1
    i1_half = Nxi_half - 1
    i1_int  = Nxi_int - 1
    ir_half = Nl_half - 1
    ir_int = Nl_int - 1

    top = topology[p0, p1]
    
    if (top == 'xx'):
        Drin[p0, i0:ir_half, -1, i0:i1_int] -= diff_Dru[p0, i0:ir_half, -1, i0:i1_int] #/ sqrt_det_h_Dr[p0, i0:ir_half, -1, i0:i1_int]
        Drin[p1, i0:ir_half, 0, i0:i1_int]  -= diff_Dru[p1, i0:ir_half, 0, i0:i1_int]  #/ sqrt_det_h_Dr[p1, i0:ir_half, 0, i0:i1_int]
        
        D2in[p0, i0:ir_int, -1, i0:i1_half] -= diff_D2u[p0, i0:ir_int, -1, i0:i1_half] #/ sqrt_det_h_D2[p0, i0:ir_int, -1, i0:i1_half]
        D2in[p1, i0:ir_int, 0, i0:i1_half]  -= diff_D2u[p1, i0:ir_int, 0, i0:i1_half]  #/ sqrt_det_h_D2[p1, i0:ir_int, 0, i0:i1_half]

    if (top == 'xy'):
        Drin[p0, i0:ir_half, -1, i0:i1_int] -= diff_Dru[p0, i0:ir_half, -1, i0:i1_int] #/ sqrt_det_h_Dr[p0, i0:ir_half, -1, i0:i1_int]
        Drin[p1, i0:ir_half, i0:i1_int, 0]  -= diff_Dru[p1, i0:ir_half, i0:i1_int, 0]  #/ sqrt_det_h_Dr[p1, i0:ir_half, i0:i1_int, 0]

        D2in[p0, i0:ir_int, -1, i0:i1_half] -= diff_D2u[p0, i0:ir_int, -1, i0:i1_half] #/ sqrt_det_h_D2[p0, i0:ir_int, -1, i0:i1_half]
        D1in[p1, i0:ir_int, i0:i1_half, 0]  -= diff_D1u[p1, i0:ir_int, i0:i1_half, 0]  #/ sqrt_det_h_D2[p1, i0:ir_int, i0:i1_half, 0]

    if (top == 'yy'):
        Drin[p0, i0:ir_half, i0:i1_int, -1] -= diff_Dru[p0, i0:ir_half, i0:i1_int, -1] #/ sqrt_det_h_Dr[p0, i0:ir_half, i0:i1_int, -1]
        Drin[p1, i0:ir_half, i0:i1_int, 0]  -= diff_Dru[p1, i0:ir_half, i0:i1_int, 0]  #/ sqrt_det_h_Dr[p1, i0:ir_half, i0:i1_int, 0]

        D1in[p0, i0:ir_int, i0:i1_half, -1] -= diff_D1u[p0, i0:ir_int, i0:i1_half, -1] #/ sqrt_det_h_D1[p0, i0:ir_int, i0:i1_half, -1]
        D1in[p1, i0:ir_int, i0:i1_half, 0]  -= diff_D1u[p1, i0:ir_int, i0:i1_half, 0]  #/ sqrt_det_h_D1[p1, i0:ir_int, i0:i1_half, 0]

    if (top == 'yx'):
        Drin[p0, i0:ir_half, i0:i1_int, -1] -= diff_Dru[p0, i0:ir_half, i0:i1_int, -1] #/ sqrt_det_h_Dr[p0, i0:ir_half, i0:i1_int, -1]
        Drin[p1, i0:ir_half, 0, i0:i1_int]  -= diff_Dru[p1, i0:ir_half, 0, i0:i1_int]  #/ sqrt_det_h_Dr[p1, i0:ir_half, 0, i0:i1_int]

        D1in[p0, i0:ir_int, i0:i1_half, -1] -= diff_D1u[p0, i0:ir_int, i0:i1_half, -1] #/ sqrt_det_h_D1[p0, i0:ir_int, i0:i1_half, -1]
        D2in[p1, i0:ir_int, 0, i0:i1_half]  -= diff_D2u[p1, i0:ir_int, 0, i0:i1_half]  #/ sqrt_det_h_D2[p1, i0:ir_int, 0, i0:i1_half]

def corners_D(p0, Drin, D1in, D2in):

    i0 =  1
    ir_half = Nl_half - 1
    ir_int = Nl_int - 1

    Drin[p0, i0:ir_half, 0, 0]   -= diff_Dru[p0, i0:ir_half, 0, 0]   #/ sqrt_det_h_Dr[p0, i0:ir_half, 0, 0]
    Drin[p0, i0:ir_half, -1, 0]  -= diff_Dru[p0, i0:ir_half, -1, 0]  #/ sqrt_det_h_Dr[p0, i0:ir_half, -1, 0]
    Drin[p0, i0:ir_half, 0, -1]  -= diff_Dru[p0, i0:ir_half, 0, -1]  #/ sqrt_det_h_Dr[p0, i0:ir_half, 0, -1]
    Drin[p0, i0:ir_half, -1, -1] -= diff_Dru[p0, i0:ir_half, -1, -1] #/ sqrt_det_h_Dr[p0, i0:ir_half, -1, -1]

    D1in[p0, i0:ir_int, 0, 0]   -= diff_D1u[p0, i0:ir_int, 0, 0]   #/ sqrt_det_h_D1[p0, i0:ir_int, 0, 0]
    D1in[p0, i0:ir_int, -1, 0]  -= diff_D1u[p0, i0:ir_int, -1, 0]  #/ sqrt_det_h_D1[p0, i0:ir_int, -1, 0]
    D1in[p0, i0:ir_int, 0, -1]  -= diff_D1u[p0, i0:ir_int, 0, -1]  #/ sqrt_det_h_D1[p0, i0:ir_int, 0, -1] 
    D1in[p0, i0:ir_int, -1, -1] -= diff_D1u[p0, i0:ir_int, -1, -1] #/ sqrt_det_h_D1[p0, i0:ir_int, -1, -1]

    D2in[p0, i0:ir_int, 0, 0]   -= diff_D2u[p0, i0:ir_int, 0, 0]   #/ sqrt_det_h_D2[p0, i0:ir_int, 0, 0]
    D2in[p0, i0:ir_int, -1, 0]  -= diff_D2u[p0, i0:ir_int, -1, 0]  #/ sqrt_det_h_D2[p0, i0:ir_int, -1, 0]
    D2in[p0, i0:ir_int, 0, -1]  -= diff_D2u[p0, i0:ir_int, 0, -1]  #/ sqrt_det_h_D2[p0, i0:ir_int, 0, -1]
    D2in[p0, i0:ir_int, -1, -1] -= diff_D2u[p0, i0:ir_int, -1, -1] #/ sqrt_det_h_D2[p0, i0:ir_int, -1, -1]

def radial_interface_D(p0, Drin, D1in, D2in):

    Drin[p0, 0, :, :]   -= diff_Dru[p0, 0, :, :]  #/ sqrt_det_h_Dr[p0, 0, :, :]
    D1in[p0, 0, :, :]   -= diff_D1u[p0, 0, :, :]  #/ sqrt_det_h_D1[p0, 0, :, :]
    D2in[p0, 0, :, :]   -= diff_D2u[p0, 0, :, :]  #/ sqrt_det_h_D2[p0, 0, :, :]
    Drin[p0, -1, :, :]  -= diff_Dru[p0, -1, :, :] #/ sqrt_det_h_Dr[p0, -1, :, :]
    D1in[p0, -1, :, :]  -= diff_D1u[p0, -1, :, :] #/ sqrt_det_h_D1[p0, -1, :, :]
    D2in[p0, -1, :, :]  -= diff_D2u[p0, -1, :, :] #/ sqrt_det_h_D2[p0, -1, :, :]

def penalty_edges_D(dtin, Drin, D1in, D2in, Brin, B1in, B2in, Drout, D1out, D2out):

    diff_Dru[:, :, :, :] = 0.0
    diff_D1u[:, :, :, :] = 0.0
    diff_D2u[:, :, :, :] = 0.0
    
    for i in range(n_zeros):
        p0, p1 = index_row[i], index_col[i]
        compute_penalty_D(p0, p1, dtin, Drin, D1in, D2in, Brin, B1in, B2in)

    # radial_penalty_D(dtin, Drin, D1in, D2in, Brin, B1in, B2in)

    for i in range(n_zeros):
        p0, p1 = index_row[i], index_col[i]
        interface_D(p0, p1, Drout, D1out, D2out)
    
    radial_interface_D(patches, Drout, D1out, D2out)
    
    corners_D(patches, Drout, D1out, D2out)

########
# Apply penalty terms to B, H
########

def interface_B(p0, p1, Brin, B1in, B2in):

    i0 =  1
    i1_half = Nxi_half - 1
    i1_int  = Nxi_int - 1
    ir_half = Nl_half - 1
    ir_int = Nl_int - 1

    top = topology[p0, p1]
    
    if (top == 'xx'):
        Brin[p0, i0:ir_int, -1, i0:i1_half] -= diff_Bru[p0, i0:ir_int, -1, i0:i1_half] #/ sqrt_det_h_Br[p0, i0:ir_int, -1, i0:i1_half]
        Brin[p1, i0:ir_int, 0, i0:i1_half]  -= diff_Bru[p1, i0:ir_int, 0, i0:i1_half]  #/ sqrt_det_h_Br[p1, i0:ir_int, 0, i0:i1_half]

        B2in[p0, i0:ir_half, -1, i0:i1_int] -= diff_B2u[p0, i0:ir_half, -1, i0:i1_int] #/ sqrt_det_h_B2[p0, i0:ir_half, -1, i0:i1_int]
        B2in[p1, i0:ir_half, 0, i0:i1_int]  -= diff_B2u[p1, i0:ir_half, 0, i0:i1_int]  #/ sqrt_det_h_B2[p1, i0:ir_half, 0, i0:i1_int]

    if (top == 'xy'):
        Brin[p0, i0:ir_int, -1, i0:i1_half] -= diff_Bru[p0, i0:ir_int, -1, i0:i1_half] #/ sqrt_det_h_Br[p0, i0:ir_int, -1, i0:i1_half]
        Brin[p1, i0:ir_int, i0:i1_half, 0]  -= diff_Bru[p1, i0:ir_int, i0:i1_half, 0]  #/ sqrt_det_h_Br[p1, i0:ir_int, i0:i1_half, 0]

        B2in[p0, i0:ir_half, -1, i0:i1_int] -= diff_B2u[p0, i0:ir_half, -1, i0:i1_int] #/ sqrt_det_h_B2[p0, i0:ir_half, -1, i0:i1_int]
        B1in[p1, i0:ir_half, i0:i1_int, 0]  -= diff_B1u[p1, i0:ir_half, i0:i1_int, 0]  #/ sqrt_det_h_B1[p1, i0:ir_half, i0:i1_int, 0]

    if (top == 'yy'):
        Brin[p0, i0:ir_int, i0:i1_half, -1] -= diff_Bru[p0, i0:ir_int, i0:i1_half, -1] #/ sqrt_det_h_Br[p0, i0:ir_int, i0:i1_half, -1]
        Brin[p1, i0:ir_int, i0:i1_half, 0]  -= diff_Bru[p1, i0:ir_int, i0:i1_half, 0]  #/ sqrt_det_h_Br[p1, i0:ir_int, i0:i1_half, 0]

        B1in[p0, i0:ir_half, i0:i1_int, -1] -= diff_B1u[p0, i0:ir_half, i0:i1_int, -1] #/ sqrt_det_h_B1[p0, i0:ir_half, i0:i1_int, -1]
        B1in[p1, i0:ir_half, i0:i1_int, 0]  -= diff_B1u[p1, i0:ir_half, i0:i1_int, 0]  #/ sqrt_det_h_B1[p1, i0:ir_half, i0:i1_int, 0]

    if (top == 'yx'):
        Brin[p0, i0:ir_int, i0:i1_half, -1] -= diff_Bru[p0, i0:ir_int, i0:i1_half, -1] #/ sqrt_det_h_Br[p0, i0:ir_int, i0:i1_half, -1]
        Brin[p1, i0:ir_int, 0, i0:i1_half]  -= diff_Bru[p1, i0:ir_int, 0, i0:i1_half]  #/ sqrt_det_h_Br[p1, i0:ir_int, 0, i0:i1_half]

        B1in[p0, i0:ir_half, i0:i1_int, -1] -= diff_B1u[p0, i0:ir_half, i0:i1_int, -1] #/ sqrt_det_h_B1[p0, i0:ir_half, i0:i1_int, -1]
        B2in[p1, i0:ir_half, 0, i0:i1_int]  -= diff_B2u[p1, i0:ir_half, 0, i0:i1_int]  #/ sqrt_det_h_B2[p1, i0:ir_half, 0, i0:i1_int]

def corners_B(p0, Brin, B1in, B2in):

    i0 = 1
    ir_half = Nl_half - 1
    ir_int = Nl_int - 1

    Brin[p0, i0:ir_int, 0, 0]   -= diff_Bru[p0, i0:ir_int, 0, 0]   #/ sqrt_det_h_Br[p0, i0:ir_int, 0, 0]
    Brin[p0, i0:ir_int, -1, 0]  -= diff_Bru[p0, i0:ir_int, -1, 0]  #/ sqrt_det_h_Br[p0, i0:ir_int, -1, 0] 
    Brin[p0, i0:ir_int, 0, -1]  -= diff_Bru[p0, i0:ir_int, 0, -1]  #/ sqrt_det_h_Br[p0, i0:ir_int, 0, -1]
    Brin[p0, i0:ir_int, -1, -1] -= diff_Bru[p0, i0:ir_int, -1, -1] #/ sqrt_det_h_Br[p0, i0:ir_int, -1, -1] 

    B1in[p0, i0:ir_half, 0, 0]   -= diff_B1u[p0, i0:ir_half, 0, 0]   #/ sqrt_det_h_B1[p0, i0:ir_half, 0, 0]
    B1in[p0, i0:ir_half, -1, 0]  -= diff_B1u[p0, i0:ir_half, -1, 0]  #/ sqrt_det_h_B1[p0, i0:ir_half, -1, 0] 
    B1in[p0, i0:ir_half, 0, -1]  -= diff_B1u[p0, i0:ir_half, 0, -1]  #/ sqrt_det_h_B1[p0, i0:ir_half, 0, -1]
    B1in[p0, i0:ir_half, -1, -1] -= diff_B1u[p0, i0:ir_half, -1, -1] #/ sqrt_det_h_B1[p0, i0:ir_half, -1, -1] 

    B2in[p0, i0:ir_half, 0, 0]   -= diff_B2u[p0, i0:ir_half, 0, 0]   #/ sqrt_det_h_B2[p0, i0:ir_half, 0, 0]
    B2in[p0, i0:ir_half, -1, 0]  -= diff_B2u[p0, i0:ir_half, -1, 0]  #/ sqrt_det_h_B2[p0, i0:ir_half, -1, 0] 
    B2in[p0, i0:ir_half, 0, -1]  -= diff_B2u[p0, i0:ir_half, 0, -1]  #/ sqrt_det_h_B2[p0, i0:ir_half, 0, -1]
    B2in[p0, i0:ir_half, -1, -1] -= diff_B2u[p0, i0:ir_half, -1, -1] #/ sqrt_det_h_B2[p0, i0:ir_half, -1, -1] 

def radial_interface_B(p0, Brin, B1in, B2in):

    Brin[p0, 0, :, :]   -= diff_Bru[p0, 0, :, :]  #/ sqrt_det_h_Br[p0, 0, :, :]
    B1in[p0, 0, :, :]   -= diff_B1u[p0, 0, :, :]  #/ sqrt_det_h_B1[p0, 0, :, :]
    B2in[p0, 0, :, :]   -= diff_B2u[p0, 0, :, :]  #/ sqrt_det_h_B2[p0, 0, :, :]
    Brin[p0, -1, :, :]  -= diff_Bru[p0, -1, :, :] #/ sqrt_det_h_Br[p0, -1, :, :]
    B1in[p0, -1, :, :]  -= diff_B1u[p0, -1, :, :] #/ sqrt_det_h_B1[p0, -1, :, :]
    B2in[p0, -1, :, :]  -= diff_B2u[p0, -1, :, :] #/ sqrt_det_h_B2[p0, -1, :, :]

def penalty_edges_B(dtin, Drin, D1in, D2in, Brin, B1in, B2in, Brout, B1out, B2out):

    diff_Bru[:, :, :, :] = 0.0
    diff_B1u[:, :, :, :] = 0.0
    diff_B2u[:, :, :, :] = 0.0
    
    for i in range(n_zeros):
        p0, p1 = index_row[i], index_col[i]
        compute_penalty_B(p0, p1, dtin, Drin, D1in, D2in, Brin, B1in, B2in)

    # radial_penalty_B(dtin, Drin, D1in, D2in, Brin, B1in, B2in)

    for i in range(n_zeros):
        p0, p1 = index_row[i], index_col[i]
        interface_B(p0, p1, Brout, B1out, B2out)

    radial_interface_B(patches, Brout, B1out, B2out)
    
    corners_B(patches, Brout, B1out, B2out)

########
# Absorbing boundary conditions at r_max
########

i_abs = 3 # Thickness of absorbing layer in number of cells

r_abs_out = r_int[Nl_int - i_abs]
kappa_out = 10.0 

delta = ((r_int - r_abs_out) / (r_max - r_abs_out)) * N.heaviside(r_int - r_abs_out, 0.0)
sigma_int = N.exp(- kappa_out * delta**3)

delta = ((r_half - r_abs_out) / (r_max - r_abs_out)) * N.heaviside(r_half - r_abs_out, 0.0)
sigma_half = N.exp(- kappa_out * delta**3)

r_abs_in = r_int[i_abs]
kappa_in = 0.0 

delta = ((r_abs_in - r_int) / (r_abs_in - r_min)) * N.heaviside(r_abs_in - r_int, 0.0)
sigma_int2 = N.exp(- kappa_in * delta**3)

delta = ((r_abs_in - r_half) / (r_abs_in - r_min)) * N.heaviside(r_abs_in - r_half, 0.0)
sigma_half2 = N.exp(- kappa_in * delta**3)

def BC_D_absorb(patch, Drin, D1in, D2in):
    Drin[patch, :, :, :] *= sigma_half[:, None, None]
    D1in[patch, :, :, :] *= sigma_int[:, None, None]
    D2in[patch, :, :, :] *= sigma_int[:, None, None]

    Drin[patch, :, :, :] *= sigma_half2[:, None, None]
    D1in[patch, :, :, :] *= sigma_int2[:, None, None]
    D2in[patch, :, :, :] *= sigma_int2[:, None, None]

def BC_B_absorb(patch, Brin, B1in, B2in):
    Brin[patch, :, :, :] = INBr[patch, :, :, :] + (Brin[patch, :, :, :] - INBr[patch, :, :, :]) * sigma_int[:, None, None]
    B1in[patch, :, :, :] = INB1[patch, :, :, :] + (B1in[patch, :, :, :] - INB1[patch, :, :, :]) * sigma_half[:, None, None]
    B2in[patch, :, :, :] = INB2[patch, :, :, :] + (B2in[patch, :, :, :] - INB2[patch, :, :, :]) * sigma_half[:, None, None]

    Brin[patch, :, :, :] *= sigma_int2[:, None, None]
    B1in[patch, :, :, :] *= sigma_half2[:, None, None]
    B2in[patch, :, :, :] *= sigma_half2[:, None, None]

########
# Define initial data
########

def alpha_sph(r, theta, phi, spin):
    z = 2.0 * r / (r * r + spin * spin * N.cos(theta)**2)
    return 1.0 / N.sqrt(1.0 + z)

def sqrtdeth(r, theta, phi, spin):
    return (r * r + spin * spin * N.cos(theta)**2) * N.sin(theta) / alpha_sph(r, theta, phi, spin)

B0 = 1.0
tilt = 0.0 / 180.0 * N.pi

def func_Br(r0, th0, ph0):
    # return 2.0 * B0 * (N.cos(th0) * N.cos(tilt) + N.sin(th0) * N.sin(ph0) * N.sin(tilt)) / r0**3
    return B0 * r0 * r0 * N.cos(th0) * N.sin(th0) / sqrtdeth(r0, th0, ph0, a) / r0
    # return B0 * N.sin(th0) / sqrtdeth(r0, th0, ph0, a)

def func_Bth(r0, th0, ph0):
#    return B0 * (N.cos(tilt) * N.sin(th0) - N.cos(th0) * N.sin(ph0) * N.sin(tilt)) / r0**4
   return - B0 * r0 * N.sin(th0)**2 / sqrtdeth(r0, th0, ph0, a)
    # return 0.0

def func_Bph(r0, th0, ph0):
    # return - B0 * (N.cos(ph0) / N.sin(th0) * N.sin(tilt)) / r0**4
    return 0.0

def InitialData():

    for patch in range(n_patches):

        fvec = (globals()["vec_sph_to_" + sphere[patch]])
        fcoord = (globals()["coord_" + sphere[patch] + "_to_sph"])

        for i in range(Nxi_half):
            for j in range(Neta_half):

                r0 = r_int[:]
                th0, ph0 = fcoord(xi_half[i], eta_half[j])
                BrTMP = func_Br(r0, th0, ph0)

                Bru[patch, :, i, j] = BrTMP
                INBr[patch,:, i, j] = BrTMP
                    
        for i in range(Nxi_int):
            for j in range(Neta_half):

                    r0 = r_half[:]
                    th0, ph0 = fcoord(xi_int[i], eta_half[j])
                    BtTMP = func_Bth(r0, th0, ph0)
                    BpTMP = func_Bph(r0, th0, ph0)
                    BCStmp = fvec(th0, ph0, BtTMP, BpTMP)

                    B1u[patch, :, i, j]  = BCStmp[0]
                    INB1[patch, :, i, j] = BCStmp[0]
                    D2u[patch, :, i, j] = 0.0

        for i in range(Nxi_half):
            for j in range(Neta_int):

                    r0 = r_half[:]
                    th0, ph0 = fcoord(xi_half[i], eta_int[j])
                    BtTMP = func_Bth(r0, th0, ph0)
                    BpTMP = func_Bph(r0, th0, ph0)
                    BCStmp = fvec(th0, ph0, BtTMP, BpTMP)

                    B2u[patch, :, i, j]  = BCStmp[1]
                    INB2[patch, :, i, j] = BCStmp[1]
                    D1u[patch, :, i, j]  = 0.0

        Dru[patch, :, :, :] = 0.0

InitialData()

########
# Plotting fields on an unfolded sphere
########

# Figure parameters
scale, aspect = 2.0, 0.7
vm = 0.2
ratio = 2.0
fig_size=deffigsize(scale, aspect)

ratio = 0.5

def plot_fields_unfolded_Br(it, vm, ir):

    xi_grid_c, eta_grid_c = unflip_eq(xBr_grid, yBr_grid)
    xi_grid_d, eta_grid_d = unflip_eq(xBr_grid, yBr_grid)
    xi_grid_n, eta_grid_n = unflip_po(xBr_grid, yBr_grid)

    fig = P.figure(1, facecolor='w')
    ax = P.subplot(111)

    ax.pcolormesh(xBr_grid, yBr_grid, Bru[Sphere.A, ir, :, :], cmap = "RdBu_r", vmin = - vm, vmax = vm)
    ax.pcolormesh(xBr_grid + N.pi / 2.0, yBr_grid, Bru[Sphere.B, ir, :, :], cmap = "RdBu_r", vmin = - vm, vmax = vm)
    ax.pcolormesh(xBr_grid, yBr_grid - N.pi / 2.0, Bru[Sphere.S, ir, :, :], cmap = "RdBu_r", vmin = - vm, vmax = vm)

    ax.pcolormesh(xi_grid_c + N.pi, eta_grid_c, Bru[Sphere.C, ir, :, :], cmap = "RdBu_r", vmin = - vm, vmax = vm)
    ax.pcolormesh(xi_grid_d - N.pi / 2.0, eta_grid_d, Bru[Sphere.D, ir, :, :], cmap = "RdBu_r", vmin = - vm, vmax = vm)
    ax.pcolormesh(xi_grid_n, eta_grid_n + N.pi / 2.0, Bru[Sphere.N, ir, :, :], cmap = "RdBu_r", vmin = - vm, vmax = vm)
    
    P.title(r'$t={:.3f} R/c$'.format(FDUMP*it*dt))
    
    figsave_png(fig, "../snapshots_3d/Br_" + str(it))

    P.close('all')

def plot_fields_unfolded_B1(it, vm, ir):

    xi_grid_c, eta_grid_c = unflip_eq(xE2_grid, yE2_grid)
    xi_grid_d, eta_grid_d = unflip_eq(xE2_grid, yE2_grid)
    xi_grid_n, eta_grid_n = unflip_po(xE2_grid, yE2_grid)

    fig = P.figure(1, facecolor='w')
    ax = P.subplot(111)

    ax.pcolormesh(xE2_grid, yE2_grid, B1u[Sphere.A, ir, :, :], cmap = "RdBu_r", vmin = - vm, vmax = vm)
    ax.pcolormesh(xE2_grid + N.pi / 2.0 + 0.1, yE2_grid, B1u[Sphere.B, ir, :, :], cmap = "RdBu_r", vmin = - vm, vmax = vm)
    ax.pcolormesh(xE2_grid, yE2_grid - N.pi / 2.0 - 0.1, B1u[Sphere.S, ir, :, :], cmap = "RdBu_r", vmin = - vm, vmax = vm)

    ax.pcolormesh(xi_grid_c + N.pi + 0.2, eta_grid_c, B1u[Sphere.C, ir, :, :], cmap = "RdBu_r", vmin = - vm, vmax = vm)
    ax.pcolormesh(xi_grid_d - N.pi / 2.0 - 0.1, eta_grid_d, B1u[Sphere.D, ir, :, :], cmap = "RdBu_r", vmin = - vm, vmax = vm)
    ax.pcolormesh(xi_grid_n, eta_grid_n + N.pi / 2.0 + 0.1, B1u[Sphere.N, ir, :, :], cmap = "RdBu_r", vmin = - vm, vmax = vm)
    
    P.title(r'$t={:.3f} R/c$'.format(FDUMP*it*dt))
    
    figsave_png(fig, "../snapshots_3d/B1u_" + str(it))

    P.close('all')

def plot_fields_unfolded_B2(it, vm, ir):

    xi_grid_c, eta_grid_c = unflip_eq(xE1_grid, yE1_grid)
    xi_grid_d, eta_grid_d = unflip_eq(xE1_grid, yE1_grid)
    xi_grid_n, eta_grid_n = unflip_po(xE1_grid, yE1_grid)

    fig = P.figure(1, facecolor='w')
    ax = P.subplot(111)

    ax.pcolormesh(xE1_grid, yE1_grid, B2u[Sphere.A, ir, :, :], cmap = "RdBu_r", vmin = - vm, vmax = vm)
    ax.pcolormesh(xE1_grid + N.pi / 2.0 + 0.1, yE1_grid, B2u[Sphere.B, ir, :, :], cmap = "RdBu_r", vmin = - vm, vmax = vm)
    ax.pcolormesh(xE1_grid, yE1_grid - N.pi / 2.0 - 0.1, B2u[Sphere.S, ir, :, :], cmap = "RdBu_r", vmin = - vm, vmax = vm)

    ax.pcolormesh(xi_grid_c + N.pi + 0.2, eta_grid_c, B2u[Sphere.C, ir, :, :], cmap = "RdBu_r", vmin = - vm, vmax = vm)
    ax.pcolormesh(xi_grid_d - N.pi / 2.0 - 0.1, eta_grid_d, B2u[Sphere.D, ir, :, :], cmap = "RdBu_r", vmin = - vm, vmax = vm)
    ax.pcolormesh(xi_grid_n, eta_grid_n + N.pi / 2.0 + 0.1, B2u[Sphere.N, ir, :, :], cmap = "RdBu_r", vmin = - vm, vmax = vm)
    
    P.title(r'$t={:.3f} R/c$'.format(FDUMP*it*dt))
    
    figsave_png(fig, "../snapshots_3d/B2u_" + str(it))

    P.close('all')

def plot_fields_unfolded_D2(it, vm, ir):

    xi_grid_c, eta_grid_c = unflip_eq(xE2_grid, yE2_grid)
    xi_grid_d, eta_grid_d = unflip_eq(xE2_grid, yE2_grid)
    xi_grid_n, eta_grid_n = unflip_po(xE2_grid, yE2_grid)

    fig = P.figure(1, facecolor='w')
    ax = P.subplot(111)

    ax.pcolormesh(xE2_grid, yE2_grid, D2u[Sphere.A, ir, :, :], cmap = "RdBu_r", vmin = - vm, vmax = vm)
    ax.pcolormesh(xE2_grid + N.pi / 2.0 + 0.1, yE2_grid, D2u[Sphere.B, ir, :, :], cmap = "RdBu_r", vmin = - vm, vmax = vm)
    ax.pcolormesh(xE2_grid, yE2_grid - N.pi / 2.0 - 0.1, D2u[Sphere.S, ir, :, :], cmap = "RdBu_r", vmin = - vm, vmax = vm)

    ax.pcolormesh(xi_grid_c + N.pi + 0.2, eta_grid_c, D2u[Sphere.C, ir, :, :], cmap = "RdBu_r", vmin = - vm, vmax = vm)
    ax.pcolormesh(xi_grid_d - N.pi / 2.0 - 0.1, eta_grid_d, D2u[Sphere.D, ir, :, :], cmap = "RdBu_r", vmin = - vm, vmax = vm)
    ax.pcolormesh(xi_grid_n, eta_grid_n + N.pi / 2.0 + 0.1, D2u[Sphere.N, ir, :, :], cmap = "RdBu_r", vmin = - vm, vmax = vm)
    
    P.title(r'$t={:.3f} R/c$'.format(FDUMP*it*dt))
    
    figsave_png(fig, "../snapshots_3d/D2u_" + str(it))

    P.close('all')
    
########
# Initialization
########

idump = 0

Nt = 20000 # Number of iterations
FDUMP = 50 # Dump frequency
time = dt * N.arange(Nt)
energy = N.zeros((n_patches, Nt))

WriteCoordsHDF5()

Bru0[:, :, :, :] = Bru[:, :, :, :]
B1u0[:, :, :, :] = B1u[:, :, :, :]
B2u0[:, :, :, :] = B2u[:, :, :, :]
Dru0[:, :, :, :] = Dru[:, :, :, :]
D1u0[:, :, :, :] = D1u[:, :, :, :]
D2u0[:, :, :, :] = D2u[:, :, :, :]

########
# Main routine
########

for it in tqdm(range(Nt), "Progression"):
    if ((it % FDUMP) == 0):
        plot_fields_unfolded_Br(idump, 1.0, 4)
        plot_fields_unfolded_B1(idump, 1.0, 4)
        plot_fields_unfolded_B2(idump, 1.0, 4)
        plot_fields_unfolded_D2(idump, 1.0, 4)
        WriteAllFieldsHDF5(idump)
        idump += 1

    # average_field(patches, Bru, B1u, B2u, Bru0, B1u0, B2u0, Bru1, B1u1, B2u1)
    # average_field(patches, Dru, D1u, D2u, Dru0, D1u0, D2u0, Dru1, D1u1, D2u1)
    
    # contra_to_cov_D(patches, Dru1, D1u1, D2u1)
    # compute_E_aux(patches, Drd, D1d, D2d, Bru, B1u, B2u)

    # compute_diff_E(patches)
    # push_B(patches, Bru1, B1u1, B2u1, dt)

    # # Penalty terms ??
    # # penalty_edges_B(dt, Erd, E1d, E2d, Bru, B1u, B2u, Bru1, B1u1, B2u1)

    # BC_B_absorb(patches, Bru1, B1u1, B2u1)

    # contra_to_cov_D(patches, Dru, D1u, D2u)
    # compute_E_aux(patches, Drd, D1d, D2d, Bru1, B1u1, B2u1)

    # Bru0[:, :, :, :] = Bru[:, :, :, :]
    # B1u0[:, :, :, :] = B1u[:, :, :, :]
    # B2u0[:, :, :, :] = B2u[:, :, :, :]

    # compute_diff_E(patches)
    # push_B(patches, Bru, B1u, B2u, dt)

    # # Penalty terms
    # penalty_edges_B(dt, Erd, E1d, E2d, Bru1, B1u1, B2u1, Bru, B1u, B2u)

    # BC_B_absorb(patches, Bru, B1u, B2u)

    # average_field(patches, Bru, B1u, B2u, Bru0, B1u0, B2u0, Bru1, B1u1, B2u1)

    # contra_to_cov_B(patches, Bru1, B1u1, B2u1)
    # compute_H_aux(patches, Dru, D1u, D2u, Brd, B1d, B2d)

    # compute_diff_H(patches)
    # push_D(patches, Dru1, D1u1, D2u1, dt)

    # # Penalty terms ??
    # # penalty_edges_D(dt, Dru, D1u, D2u, Hrd, H1d, H2d, Dru1, D1u1, D2u1)

    # BC_D_absorb(patches, Dru1, D1u1, D2u1)

    # contra_to_cov_B(patches, Bru, B1u, B2u)
    # compute_H_aux(patches, Dru1, D1u1, D2u1, Brd, B1d, B2d)

    # Dru0[:, :, :, :] = Dru[:, :, :, :]
    # D1u0[:, :, :, :] = D1u[:, :, :, :]
    # D2u0[:, :, :, :] = D2u[:, :, :, :]

    # compute_diff_H(patches)
    # push_D(patches, Dru, D1u, D2u, dt)

    # # Penalty terms
    # penalty_edges_D(dt, Dru1, D1u1, D2u1, Hrd, H1d, H2d, Dru, D1u, D2u)

    # BC_D_absorb(patches, Dru, D1u, D2u)
    
    #### WITH BC

    average_field(patches, Bru, B1u, B2u, Bru0, B1u0, B2u0, Bru1, B1u1, B2u1)
    average_field(patches, Dru, D1u, D2u, Dru0, D1u0, D2u0, Dru1, D1u1, D2u1)
    
    contra_to_cov_D(patches, Dru1, D1u1, D2u1)
    BC_Dd(patches, Drd, D1d, D2d)
    compute_E_aux(patches, Drd, D1d, D2d, Bru, B1u, B2u)
    BC_Ed(patches, Erd, E1d, E2d)

    compute_diff_E(patches)
    push_B(patches, Bru1, B1u1, B2u1, dt)

    # # Penalty terms ??
    # penalty_edges_B(dt, Erd, E1d, E2d, Bru, B1u, B2u, Bru1, B1u1, B2u1)

    BC_Bu(patches, Bru1, B1u1, B2u1)
    BC_B_absorb(patches, Bru1, B1u1, B2u1)

    contra_to_cov_D(patches, Dru, D1u, D2u)
    BC_Dd(patches, Drd, D1d, D2d)
    compute_E_aux(patches, Drd, D1d, D2d, Bru1, B1u1, B2u1)
    BC_Ed(patches, Erd, E1d, E2d)

    Bru0[:, :, :, :] = Bru[:, :, :, :]
    B1u0[:, :, :, :] = B1u[:, :, :, :]
    B2u0[:, :, :, :] = B2u[:, :, :, :]

    compute_diff_E(patches)
    push_B(patches, Bru, B1u, B2u, dt)

    # Penalty terms
    # penalty_edges_B(dt, Erd, E1d, E2d, Hru, H1u, H2u, Bru, B1u, B2u)
    # penalty_edges_B(dt, Erd, E1d, E2d, Bru1, B1u1, B2u1, Bru, B1u, B2u)
    penalty_edges_B(dt, Drd, D1d, D2d, Bru1, B1u1, B2u1, Bru, B1u, B2u)

    BC_Bu(patches, Bru, B1u, B2u)
    BC_B_absorb(patches, Bru, B1u, B2u)

    average_field(patches, Bru, B1u, B2u, Bru0, B1u0, B2u0, Bru1, B1u1, B2u1)

    contra_to_cov_B(patches, Bru1, B1u1, B2u1)
    BC_Bd(patches, Brd, B1d, B2d)
    compute_H_aux(patches, Dru, D1u, D2u, Brd, B1d, B2d)
    BC_Hd(patches, Hrd, H1d, H2d)

    compute_diff_H(patches)
    push_D(patches, Dru1, D1u1, D2u1, dt)

    # # Penalty terms ??
    # penalty_edges_D(dt, Dru, D1u, D2u, Hrd, H1d, H2d, Dru1, D1u1, D2u1)

    BC_Du(patches, Dru1, D1u1, D2u1)
    BC_D_absorb(patches, Dru1, D1u1, D2u1)

    contra_to_cov_B(patches, Bru, B1u, B2u)
    BC_Bd(patches, Brd, B1d, B2d)
    compute_H_aux(patches, Dru1, D1u1, D2u1, Brd, B1d, B2d)
    BC_Hd(patches, Hrd, H1d, H2d)

    Dru0[:, :, :, :] = Dru[:, :, :, :]
    D1u0[:, :, :, :] = D1u[:, :, :, :]
    D2u0[:, :, :, :] = D2u[:, :, :, :]

    compute_diff_H(patches)
    push_D(patches, Dru, D1u, D2u, dt)

    # Penalty terms
    # penalty_edges_D(dt, Eru, E1u, E2u, Hrd, H1d, H2d, Dru, D1u, D2u)
    # penalty_edges_D(dt, Dru1, D1u1, D2u1, Hrd, H1d, H2d, Dru, D1u, D2u)
    penalty_edges_D(dt, Dru1, D1u1, D2u1, Brd, B1d, B2d, Dru, D1u, D2u)

    BC_Du(patches, Dru, D1u, D2u)
    BC_D_absorb(patches, Dru, D1u, D2u)
