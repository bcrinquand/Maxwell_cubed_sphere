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
cfl  = 0.4 #0.1 for a=0.99
Nl0  = 60
Nxi  = 24
Neta = 24

# Spin parameter
a = 0.5
rh = 1.0 + N.sqrt(1.0 - a * a)

Nxi_int   = Nxi + 1  # Number of integer points
Nxi_half  = Nxi + 2  # Number of half-step points
Neta_int  = Neta + 1 # Number of integer points
Neta_half = Neta + 2 # NUmber of half-step points

NG = 1            # Radial ghost cells
Nl = Nl0 + 2 * NG # Total number of radial points

r_min, r_max     = 0.85 * rh, 4.0 * rh
l_min, l_max     = N.log(r_min), N.log(r_max)
xi_min, xi_max   = - N.pi / 4.0, N.pi / 4.0
eta_min, eta_max = - N.pi / 4.0, N.pi / 4.0

dl   = (l_max - l_min) / Nl0
dxi  = (xi_max - xi_min) / Nxi
deta = (eta_max - eta_min) / Neta

# Define grids
l     = l_min + N.arange(- NG, NG + Nl0, 1) * dl
l_yee = l + 0.5 * dl
r     = N.exp(l)
r_yee = N.exp(l_yee)

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
Bru = N.zeros((n_patches, Nl, Nxi_half, Neta_half))
B1u = N.zeros((n_patches, Nl, Nxi_int, Neta_half))
B2u = N.zeros((n_patches, Nl, Nxi_half,  Neta_int))
Brd = N.zeros((n_patches, Nl, Nxi_half, Neta_half))
B1d = N.zeros((n_patches, Nl, Nxi_int, Neta_half))
B2d = N.zeros((n_patches, Nl, Nxi_half,  Neta_int))

Dru = N.zeros((n_patches, Nl, Nxi_int, Neta_int))
D1u = N.zeros((n_patches, Nl, Nxi_half, Neta_int))
D2u = N.zeros((n_patches, Nl, Nxi_int,  Neta_half))
Drd = N.zeros((n_patches, Nl, Nxi_int, Neta_int))
D1d = N.zeros((n_patches, Nl, Nxi_half, Neta_int))
D2d = N.zeros((n_patches, Nl, Nxi_int,  Neta_half))

# Shifted by one time step
Bru0 = N.zeros((n_patches, Nl, Nxi_half, Neta_half))
B1u0 = N.zeros((n_patches, Nl, Nxi_int, Neta_half))
B2u0 = N.zeros((n_patches, Nl, Nxi_half,  Neta_int))
Bru1 = N.zeros((n_patches, Nl, Nxi_half, Neta_half))
B1u1 = N.zeros((n_patches, Nl, Nxi_int, Neta_half))
B2u1 = N.zeros((n_patches, Nl, Nxi_half,  Neta_int))

Dru0 = N.zeros((n_patches, Nl, Nxi_int, Neta_int))
D1u0 = N.zeros((n_patches, Nl, Nxi_half, Neta_int))
D2u0 = N.zeros((n_patches, Nl, Nxi_int,  Neta_half))
Dru1 = N.zeros((n_patches, Nl, Nxi_int, Neta_int))
D1u1 = N.zeros((n_patches, Nl, Nxi_half, Neta_int))
D2u1 = N.zeros((n_patches, Nl, Nxi_int,  Neta_half))

# Auxiliary fields and gradients
Erd = N.zeros((n_patches, Nl, Nxi_int, Neta_int))
E1d = N.zeros((n_patches, Nl, Nxi_half, Neta_int))
E2d = N.zeros((n_patches, Nl, Nxi_int,  Neta_half))

Hrd = N.zeros((n_patches, Nl, Nxi_half, Neta_half))
H1d = N.zeros((n_patches, Nl, Nxi_int, Neta_half))
H2d = N.zeros((n_patches, Nl, Nxi_half,  Neta_int))

dE1d2 = N.zeros((n_patches, Nl, Nxi_half, Neta_half))
dE2d1 = N.zeros((n_patches, Nl, Nxi_half, Neta_half))
dErd1 = N.zeros((n_patches, Nl, Nxi_half,  Neta_int))
dErd2 = N.zeros((n_patches, Nl, Nxi_int, Neta_half))
dE1dl = N.zeros((n_patches, Nl, Nxi_half, Neta_int))
dE2dl = N.zeros((n_patches, Nl, Nxi_int, Neta_half))

dHrd1 = N.zeros((n_patches, Nl, Nxi_int,  Neta_half))
dHrd2 = N.zeros((n_patches, Nl, Nxi_half, Neta_int))
dH1d2 = N.zeros((n_patches, Nl, Nxi_int, Neta_int))
dH2d1 = N.zeros((n_patches, Nl, Nxi_int, Neta_int))
dH1dl = N.zeros((n_patches, Nl, Nxi_int, Neta_half))
dH2dl = N.zeros((n_patches, Nl, Nxi_half, Neta_int))

# Interface terms
diff_Bru = N.zeros((n_patches, Nl, Nxi_half, Neta_half))
diff_B1u = N.zeros((n_patches, Nl, Nxi_int, Neta_half))
diff_B2u = N.zeros((n_patches, Nl, Nxi_half, Neta_int))
diff_Dru = N.zeros((n_patches, Nl, Nxi_int, Neta_int))
diff_D1u = N.zeros((n_patches, Nl, Nxi_half, Neta_int))
diff_D2u = N.zeros((n_patches, Nl, Nxi_int, Neta_half))

# Initial magnetic field
INBr = N.zeros((n_patches, Nl, Nxi_half, Neta_half))
INB1 = N.zeros((n_patches, Nl, Nxi_int, Neta_half))
INB2 = N.zeros((n_patches, Nl, Nxi_half,  Neta_int))
INDr = N.zeros((n_patches, Nl, Nxi_int, Neta_int))
IND1 = N.zeros((n_patches, Nl, Nxi_half, Neta_int))
IND2 = N.zeros((n_patches, Nl, Nxi_int,  Neta_half))

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

    h5f.create_dataset('r', data = r)
    h5f.create_dataset('xi_int', data = xi_int)
    h5f.create_dataset('eta_int', data = eta_int)
    h5f.create_dataset('xi_half', data = xi_half)
    h5f.create_dataset('eta_half', data = eta_half)
    
    h5f.close()

########
# Define metric tensor
########

# 7 positions in the Yee elementary cubic cell
hlld = N.empty((n_patches, Nl, Nxi_int, Neta_int, 7))
hl1d = N.empty((n_patches, Nl, Nxi_int, Neta_int, 7))
hl2d = N.empty((n_patches, Nl, Nxi_int, Neta_int, 7))
h11d = N.empty((n_patches, Nl, Nxi_int, Neta_int, 7))
h12d = N.empty((n_patches, Nl, Nxi_int, Neta_int, 7))
h22d = N.empty((n_patches, Nl, Nxi_int, Neta_int, 7))
alpha= N.empty((n_patches, Nl, Nxi_int, Neta_int, 7))
beta = N.empty((n_patches, Nl, Nxi_int, Neta_int, 7))
sqrt_det_h = N.empty((n_patches, Nl, Nxi_int, Neta_int, 7))

hllu = N.empty((n_patches, Nl, Nxi_int, Neta_int))
hl1u = N.empty((n_patches, Nl, Nxi_int, Neta_int))
hl2u = N.empty((n_patches, Nl, Nxi_int, Neta_int))
h11u = N.empty((n_patches, Nl, Nxi_int, Neta_int))
h12u = N.empty((n_patches, Nl, Nxi_int, Neta_int))
h22u = N.empty((n_patches, Nl, Nxi_int, Neta_int))

# 4 sides of a patch
sqrt_det_h_half = N.empty((n_patches, Nl, Nxi_half, 4))
h12d_half = N.empty((n_patches, Nl, Nxi_half, 4))
h11d_half = N.empty((n_patches, Nl, Nxi_half, 4))
h22d_half = N.empty((n_patches, Nl, Nxi_half, 4))
hlld_half = N.empty((n_patches, Nl, Nxi_half, 4))
hl1d_half = N.empty((n_patches, Nl, Nxi_half, 4))
hl2d_half = N.empty((n_patches, Nl, Nxi_half, 4))
h12u_half = N.empty((n_patches, Nl, Nxi_half, 4))
h11u_half = N.empty((n_patches, Nl, Nxi_half, 4))
h22u_half = N.empty((n_patches, Nl, Nxi_half, 4))
hllu_half = N.empty((n_patches, Nl, Nxi_half, 4))
hl1u_half = N.empty((n_patches, Nl, Nxi_half, 4))
hl2u_half = N.empty((n_patches, Nl, Nxi_half, 4))
alpha_half = N.empty((n_patches, Nl, Nxi_half, 4))

sqrt_det_h_int = N.empty((n_patches, Nl, Nxi_int, 4))
h12d_int = N.empty((n_patches, Nl, Nxi_int, 4))
h11d_int = N.empty((n_patches, Nl, Nxi_int, 4))
h22d_int = N.empty((n_patches, Nl, Nxi_int, 4))
hlld_int = N.empty((n_patches, Nl, Nxi_int, 4))
hl1d_int = N.empty((n_patches, Nl, Nxi_int, 4))
hl2d_int = N.empty((n_patches, Nl, Nxi_int, 4))
h12u_int = N.empty((n_patches, Nl, Nxi_int, 4))
h11u_int = N.empty((n_patches, Nl, Nxi_int, 4))
h22u_int = N.empty((n_patches, Nl, Nxi_int, 4))
hllu_int = N.empty((n_patches, Nl, Nxi_int, 4))
hl1u_int = N.empty((n_patches, Nl, Nxi_int, 4))
hl2u_int = N.empty((n_patches, Nl, Nxi_int, 4))
alpha_int = N.empty((n_patches, Nl, Nxi_int, 4))

sqrt_det_h_Br = N.empty((n_patches, Nl, Nxi_half, Neta_half))
sqrt_det_h_B1 = N.empty((n_patches, Nl, Nxi_int, Neta_half))
sqrt_det_h_B2 = N.empty((n_patches, Nl, Nxi_half, Neta_int))

for p in range(n_patches):
    for i in range(Nxi_int):
        print(i, p)
        for j in range(Neta_int):

            # 0 at (k, i, j)
            l0 = l[:]
            xi0 = xi_int[i]
            eta0 = eta_int[j]
            h11d[p, :, i, j, 0] = g11d(p, l0, xi0, eta0, a)
            h22d[p, :, i, j, 0] = g22d(p, l0, xi0, eta0, a)
            h12d[p, :, i, j, 0] = g12d(p, l0, xi0, eta0, a)
            hlld[p, :, i, j, 0] = glld(p, l0, xi0, eta0, a)
            hl1d[p, :, i, j, 0] = gl1d(p, l0, xi0, eta0, a)
            hl2d[p, :, i, j, 0] = gl2d(p, l0, xi0, eta0, a)
            alpha[p, :, i, j, 0]=  alphas(p, l0, xi0, eta0, a)
            beta[p, :, i, j, 0] =  betalu(p, l0, xi0, eta0, a)
            # sqrt_det_h[p, :, i, j, 0] = sqrtdeth(p, l0, xi0, eta0, a)

            metric = N.array([[hlld[p, :, i, j, 0], hl1d[p, :, i, j, 0], hl2d[p, :, i, j, 0]], \
                              [hl1d[p, :, i, j, 0], h11d[p, :, i, j, 0], h12d[p, :, i, j, 0]], \
                              [hl2d[p, :, i, j, 0], h12d[p, :, i, j, 0], h22d[p, :, i, j, 0]]])
            sqrt_det_h[p, :, i, j, 0] = N.sqrt(N.linalg.det(N.moveaxis(metric, 2, 0)))
            
            inv_metric = N.linalg.inv(N.moveaxis(metric, 2, 0))
            hllu[p, :, i, j] = inv_metric[:, 0, 0]
            hl1u[p, :, i, j] = inv_metric[:, 0, 1]
            hl2u[p, :, i, j] = inv_metric[:, 0, 2]
            h11u[p, :, i, j] = inv_metric[:, 1, 1]
            h12u[p, :, i, j] = inv_metric[:, 1, 2]
            h22u[p, :, i, j] = inv_metric[:, 2, 2]

            # 1 at (k, i + 1/2, j)
            l0 = l[:]
            xi0  = xi_int[i] + 0.5 * dxi
            eta0 = eta_int[j]
            h11d[p, :, i, j, 1] = g11d(p, l0, xi0, eta0, a)
            h22d[p, :, i, j, 1] = g22d(p, l0, xi0, eta0, a)
            h12d[p, :, i, j, 1] = g12d(p, l0, xi0, eta0, a)
            hlld[p, :, i, j, 1] = glld(p, l0, xi0, eta0, a)
            hl1d[p, :, i, j, 1] = gl1d(p, l0, xi0, eta0, a)
            hl2d[p, :, i, j, 1] = gl2d(p, l0, xi0, eta0, a)
            alpha[p, :, i, j,1] =  alphas(p, l0, xi0, eta0, a)
            beta[p, :, i, j, 1] =  betalu(p, l0, xi0, eta0, a)

            metric = N.array([[hlld[p, :, i, j, 1], hl1d[p, :, i, j, 1], hl2d[p, :, i, j, 1]], \
                              [hl1d[p, :, i, j, 1], h11d[p, :, i, j, 1], h12d[p, :, i, j, 1]], \
                              [hl2d[p, :, i, j, 1], h12d[p, :, i, j, 1], h22d[p, :, i, j, 1]]])
            sqrt_det_h[p, :, i, j, 1] = N.sqrt(N.linalg.det(N.moveaxis(metric, 2, 0)))

            # 2 at (k, i, j + 1/2)
            l0 = l[:]
            xi0  = xi_int[i]
            eta0 = eta_int[j] + 0.5 * deta
            h11d[p, :, i, j, 2] = g11d(p, l0, xi0, eta0, a)
            h22d[p, :, i, j, 2] = g22d(p, l0, xi0, eta0, a)
            h12d[p, :, i, j, 2] = g12d(p, l0, xi0, eta0, a)
            hlld[p, :, i, j, 2] = glld(p, l0, xi0, eta0, a)
            hl1d[p, :, i, j, 2] = gl1d(p, l0, xi0, eta0, a)
            hl2d[p, :, i, j, 2] = gl2d(p, l0, xi0, eta0, a)
            alpha[p, :, i, j,2] =  alphas(p, l0, xi0, eta0, a)
            beta[p, :, i, j, 2] =  betalu(p, l0, xi0, eta0, a)

            metric = N.array([[hlld[p, :, i, j, 2], hl1d[p, :, i, j, 2], hl2d[p, :, i, j, 2]], \
                              [hl1d[p, :, i, j, 2], h11d[p, :, i, j, 2], h12d[p, :, i, j, 2]], \
                              [hl2d[p, :, i, j, 2], h12d[p, :, i, j, 2], h22d[p, :, i, j, 2]]])
            sqrt_det_h[p, :, i, j, 2] = N.sqrt(N.linalg.det(N.moveaxis(metric, 2, 0)))

            # 3 at (k, i + 1/2, j + 1/2)
            l0 = l[:]
            xi0  = xi_int[i] + 0.5 * dxi
            eta0 = eta_int[j] + 0.5 * deta
            h11d[p, :, i, j, 3] = g11d(p, l0, xi0, eta0, a)
            h22d[p, :, i, j, 3] = g22d(p, l0, xi0, eta0, a)
            h12d[p, :, i, j, 3] = g12d(p, l0, xi0, eta0, a)
            hlld[p, :, i, j, 3] = glld(p, l0, xi0, eta0, a)
            hl1d[p, :, i, j, 3] = gl1d(p, l0, xi0, eta0, a)
            hl2d[p, :, i, j, 3] = gl2d(p, l0, xi0, eta0, a)
            alpha[p, :, i, j,3] =  alphas(p, l0, xi0, eta0, a)
            beta[p, :, i, j, 3] =  betalu(p, l0, xi0, eta0, a)

            metric = N.array([[hlld[p, :, i, j, 3], hl1d[p, :, i, j, 3], hl2d[p, :, i, j, 3]], \
                              [hl1d[p, :, i, j, 3], h11d[p, :, i, j, 3], h12d[p, :, i, j, 3]], \
                              [hl2d[p, :, i, j, 3], h12d[p, :, i, j, 3], h22d[p, :, i, j, 3]]])
            sqrt_det_h[p, :, i, j, 3] = N.sqrt(N.linalg.det(N.moveaxis(metric, 2, 0)))

            # 4 at (k + 1/2, i, j)
            l0 = l_yee[:]
            xi0  = xi_int[i]
            eta0 = eta_int[j]
            h11d[p, :, i, j, 4] = g11d(p, l0, xi0, eta0, a)
            h22d[p, :, i, j, 4] = g22d(p, l0, xi0, eta0, a)
            h12d[p, :, i, j, 4] = g12d(p, l0, xi0, eta0, a)
            hlld[p, :, i, j, 4] = glld(p, l0, xi0, eta0, a)
            hl1d[p, :, i, j, 4] = gl1d(p, l0, xi0, eta0, a)
            hl2d[p, :, i, j, 4] = gl2d(p, l0, xi0, eta0, a)
            alpha[p, :, i, j,4] =  alphas(p, l0, xi0, eta0, a)
            beta[p, :, i, j, 4] =  betalu(p, l0, xi0, eta0, a)

            metric = N.array([[hlld[p, :, i, j, 4], hl1d[p, :, i, j, 4], hl2d[p, :, i, j, 4]], \
                              [hl1d[p, :, i, j, 4], h11d[p, :, i, j, 4], h12d[p, :, i, j, 4]], \
                              [hl2d[p, :, i, j, 4], h12d[p, :, i, j, 4], h22d[p, :, i, j, 4]]])
            sqrt_det_h[p, :, i, j, 4] = N.sqrt(N.linalg.det(N.moveaxis(metric, 2, 0)))

            # 5 at (k + 1/2, i + 1/2, j)
            l0 = l_yee[:]
            xi0  = xi_int[i] + 0.5 * dxi
            eta0 = eta_int[j]
            h11d[p, :, i, j, 5] = g11d(p, l0, xi0, eta0, a)
            h22d[p, :, i, j, 5] = g22d(p, l0, xi0, eta0, a)
            h12d[p, :, i, j, 5] = g12d(p, l0, xi0, eta0, a)
            hlld[p, :, i, j, 5] = glld(p, l0, xi0, eta0, a)
            hl1d[p, :, i, j, 5] = gl1d(p, l0, xi0, eta0, a)
            hl2d[p, :, i, j, 5] = gl2d(p, l0, xi0, eta0, a)
            alpha[p, :, i, j,5] =  alphas(p, l0, xi0, eta0, a)
            beta[p, :, i, j, 5] =  betalu(p, l0, xi0, eta0, a)

            metric = N.array([[hlld[p, :, i, j, 5], hl1d[p, :, i, j, 5], hl2d[p, :, i, j, 5]], \
                              [hl1d[p, :, i, j, 5], h11d[p, :, i, j, 5], h12d[p, :, i, j, 5]], \
                              [hl2d[p, :, i, j, 5], h12d[p, :, i, j, 5], h22d[p, :, i, j, 5]]])
            sqrt_det_h[p, :, i, j, 5] = N.sqrt(N.linalg.det(N.moveaxis(metric, 2, 0)))

            # 6 at (k + 1/2, i, j + 1/2)
            l0 = l_yee[:]
            xi0  = xi_int[i]
            eta0 = eta_int[j] + 0.5 * deta
            h11d[p, :, i, j, 6] = g11d(p, l0, xi0, eta0, a)
            h22d[p, :, i, j, 6] = g22d(p, l0, xi0, eta0, a)
            h12d[p, :, i, j, 6] = g12d(p, l0, xi0, eta0, a)
            hlld[p, :, i, j, 6] = glld(p, l0, xi0, eta0, a)
            hl1d[p, :, i, j, 6] = gl1d(p, l0, xi0, eta0, a)
            hl2d[p, :, i, j, 6] = gl2d(p, l0, xi0, eta0, a)
            alpha[p, :, i, j,6] =  alphas(p, l0, xi0, eta0, a)
            beta[p, :, i, j, 6] =  betalu(p, l0, xi0, eta0, a)

            metric = N.array([[hlld[p, :, i, j, 6], hl1d[p, :, i, j, 6], hl2d[p, :, i, j, 6]], \
                              [hl1d[p, :, i, j, 6], h11d[p, :, i, j, 6], h12d[p, :, i, j, 6]], \
                              [hl2d[p, :, i, j, 6], h12d[p, :, i, j, 6], h22d[p, :, i, j, 6]]])
            sqrt_det_h[p, :, i, j, 6] = N.sqrt(N.linalg.det(N.moveaxis(metric, 2, 0)))

    # for i in range(Nxi_half):
    #     for j in range(Neta_half):

    #         l0 = l[:]
    #         xi0 = xi_half[i]
    #         eta0 = eta_half[j]
    #         h11d0 = g11d(p, l0, xi0, eta0, a)
    #         h22d0 = g22d(p, l0, xi0, eta0, a)
    #         h12d0 = g12d(p, l0, xi0, eta0, a)
    #         hrrd0 = glld(p, l0, xi0, eta0, a)
    #         hr1d0 = gl1d(p, l0, xi0, eta0, a)
    #         hr2d0 = gl2d(p, l0, xi0, eta0, a)
    #         alpha0=  alphas(p, l0, xi0, eta0, a)
    #         beta0 =  betalu(p, l0, xi0, eta0, a)

    #         metric = N.array([[hrrd0, hr1d0, hr2d0], \
    #                           [hr1d0, h11d0, h12d0], \
    #                           [hr2d0, h12d0, h22d0]])
    #         sqrt_det_h_Br[p, :, i, j] = N.sqrt(N.linalg.det(N.moveaxis(metric, 2, 0)))

    # for i in range(Nxi_int):
    #     for j in range(Neta_half):

    #         l0 = l_yee[:]
    #         xi0 = xi_int[i]
    #         eta0 = eta_half[j]
    #         h11d0 = g11d(p, l0, xi0, eta0, a)
    #         h22d0 = g22d(p, l0, xi0, eta0, a)
    #         h12d0 = g12d(p, l0, xi0, eta0, a)
    #         hrrd0 = glld(p, l0, xi0, eta0, a)
    #         hr1d0 = gl1d(p, l0, xi0, eta0, a)
    #         hr2d0 = gl2d(p, l0, xi0, eta0, a)
    #         alpha0=  alphas(p, l0, xi0, eta0, a)
    #         beta0 =  betalu(p, l0, xi0, eta0, a)

    #         metric = N.array([[hrrd0, hr1d0, hr2d0], \
    #                           [hr1d0, h11d0, h12d0], \
    #                           [hr2d0, h12d0, h22d0]])
    #         sqrt_det_h_B1[p, :, i, j] = N.sqrt(N.linalg.det(N.moveaxis(metric, 2, 0)))

    # for i in range(Nxi_half):
    #     for j in range(Neta_int):

    #         l0 = l_yee[:]
    #         xi0 = xi_half[i]
    #         eta0 = eta_int[j]
    #         h11d0 = g11d(p, l0, xi0, eta0, a)
    #         h22d0 = g22d(p, l0, xi0, eta0, a)
    #         h12d0 = g12d(p, l0, xi0, eta0, a)
    #         hrrd0 = glld(p, l0, xi0, eta0, a)
    #         hr1d0 = gl1d(p, l0, xi0, eta0, a)
    #         hr2d0 = gl2d(p, l0, xi0, eta0, a)
    #         alpha0=  alphas(p, l0, xi0, eta0, a)
    #         beta0 =  betalu(p, l0, xi0, eta0, a)

    #         metric = N.array([[hrrd0, hr1d0, hr2d0], \
    #                           [hr1d0, h11d0, h12d0], \
    #                           [hr2d0, h12d0, h22d0]])
    #         sqrt_det_h_B2[p, :, i, j] = N.sqrt(N.linalg.det(N.moveaxis(metric, 2, 0)))

# Define sqrt(det(h)), h11d, h22d, h12d, h1rd, and h2rd, on the edge of a patch, for convenience
# 0 for left, 1 for bottom, 2 for right, 3 for top

class loc:
    left = 0
    bottom = 1
    right = 2
    top = 3

for p in range(n_patches):
    
    # On half grid
    for i in range(Nxi_half):
        
        l0 = l[:]
        
        # Left edge
        xi0 = xi_half[0]
        eta0 = eta_half[i]
        h11d_half[p, :, i, loc.left] = g11d(p, l0, xi0, eta0, a)
        h22d_half[p, :, i, loc.left] = g22d(p, l0, xi0, eta0, a)
        h12d_half[p, :, i, loc.left] = g12d(p, l0, xi0, eta0, a)
        hlld_half[p, :, i, loc.left] = glld(p, l0, xi0, eta0, a)
        hl1d_half[p, :, i, loc.left] = gl1d(p, l0, xi0, eta0, a)
        hl2d_half[p, :, i, loc.left] = gl2d(p, l0, xi0, eta0, a)
        alpha_half[p, :, i, loc.left] = alphas(p, l0, xi0, eta0, a)

        metric = N.array([[hlld_half[p, :, i, loc.left], hl1d_half[p, :, i, loc.left], hl2d_half[p, :, i, loc.left]], \
                          [hl1d_half[p, :, i, loc.left], h11d_half[p, :, i, loc.left], h12d_half[p, :, i, loc.left]], \
                          [hl2d_half[p, :, i, loc.left], h12d_half[p, :, i, loc.left], h22d_half[p, :, i, loc.left]]])

        inv_metric = N.linalg.inv(N.moveaxis(metric, 2, 0))
        hllu_half[p, :, i, loc.left] = inv_metric[:, 0, 0]
        hl1u_half[p, :, i, loc.left] = inv_metric[:, 0, 1]
        hl2u_half[p, :, i, loc.left] = inv_metric[:, 0, 2]
        h11u_half[p, :, i, loc.left] = inv_metric[:, 1, 1]
        h12u_half[p, :, i, loc.left] = inv_metric[:, 1, 2]
        h22u_half[p, :, i, loc.left] = inv_metric[:, 2, 2]
        sqrt_det_h_half[p, :, i, loc.left] = N.sqrt(N.linalg.det(N.moveaxis(metric, 2, 0)))

        # Bottom edge
        xi0 = xi_half[i]
        eta0 = eta_half[0]
        h11d_half[p, :, i, loc.bottom] = g11d(p, l0, xi0, eta0, a)
        h22d_half[p, :, i, loc.bottom] = g22d(p, l0, xi0, eta0, a)
        h12d_half[p, :, i, loc.bottom] = g12d(p, l0, xi0, eta0, a)
        hlld_half[p, :, i, loc.bottom] = glld(p, l0, xi0, eta0, a)
        hl1d_half[p, :, i, loc.bottom] = gl1d(p, l0, xi0, eta0, a)
        hl2d_half[p, :, i, loc.bottom] = gl2d(p, l0, xi0, eta0, a)
        alpha_half[p, :, i, loc.bottom] = alphas(p, l0, xi0, eta0, a)

        metric = N.array([[hlld_half[p, :, i, loc.bottom], hl1d_half[p, :, i, loc.bottom], hl2d_half[p, :, i, loc.bottom]], \
                          [hl1d_half[p, :, i, loc.bottom], h11d_half[p, :, i, loc.bottom], h12d_half[p, :, i, loc.bottom]], \
                          [hl2d_half[p, :, i, loc.bottom], h12d_half[p, :, i, loc.bottom], h22d_half[p, :, i, loc.bottom]]])

        inv_metric = N.linalg.inv(N.moveaxis(metric, 2, 0))
        hllu_half[p, :, i, loc.bottom] = inv_metric[:, 0, 0]
        hl1u_half[p, :, i, loc.bottom] = inv_metric[:, 0, 1]
        hl2u_half[p, :, i, loc.bottom] = inv_metric[:, 0, 2]
        h11u_half[p, :, i, loc.bottom] = inv_metric[:, 1, 1]
        h12u_half[p, :, i, loc.bottom] = inv_metric[:, 1, 2]
        h22u_half[p, :, i, loc.bottom] = inv_metric[:, 2, 2]
        sqrt_det_h_half[p, :, i, loc.bottom] = N.sqrt(N.linalg.det(N.moveaxis(metric, 2, 0)))

        # Right edge
        xi0 = xi_half[-1]
        eta0 = eta_half[i]
        h11d_half[p, :, i, loc.right] = g11d(p, l0, xi0, eta0, a)
        h22d_half[p, :, i, loc.right] = g22d(p, l0, xi0, eta0, a)
        h12d_half[p, :, i, loc.right] = g12d(p, l0, xi0, eta0, a)
        hlld_half[p, :, i, loc.right] = glld(p, l0, xi0, eta0, a)
        hl1d_half[p, :, i, loc.right] = gl1d(p, l0, xi0, eta0, a)
        hl2d_half[p, :, i, loc.right] = gl2d(p, l0, xi0, eta0, a)
        alpha_half[p, :, i, loc.right] = alphas(p, l0, xi0, eta0, a)

        metric = N.array([[hlld_half[p, :, i, loc.right], hl1d_half[p, :, i, loc.right], hl2d_half[p, :, i, loc.right]], \
                          [hl1d_half[p, :, i, loc.right], h11d_half[p, :, i, loc.right], h12d_half[p, :, i, loc.right]], \
                          [hl2d_half[p, :, i, loc.right], h12d_half[p, :, i, loc.right], h22d_half[p, :, i, loc.right]]])

        inv_metric = N.linalg.inv(N.moveaxis(metric, 2, 0))
        hllu_half[p, :, i, loc.right] = inv_metric[:, 0, 0]
        hl1u_half[p, :, i, loc.right] = inv_metric[:, 0, 1]
        hl2u_half[p, :, i, loc.right] = inv_metric[:, 0, 2]
        h11u_half[p, :, i, loc.right] = inv_metric[:, 1, 1]
        h12u_half[p, :, i, loc.right] = inv_metric[:, 1, 2]
        h22u_half[p, :, i, loc.right] = inv_metric[:, 2, 2]
        sqrt_det_h_half[p, :, i, loc.right] = N.sqrt(N.linalg.det(N.moveaxis(metric, 2, 0)))

        # Top edge
        xi0 = xi_half[i]
        eta0 = eta_half[-1]
        h11d_half[p, :, i, loc.top] = g11d(p, l0, xi0, eta0, a)
        h22d_half[p, :, i, loc.top] = g22d(p, l0, xi0, eta0, a)
        h12d_half[p, :, i, loc.top] = g12d(p, l0, xi0, eta0, a)
        hlld_half[p, :, i, loc.top] = glld(p, l0, xi0, eta0, a)
        hl1d_half[p, :, i, loc.top] = gl1d(p, l0, xi0, eta0, a)
        hl2d_half[p, :, i, loc.top] = gl2d(p, l0, xi0, eta0, a)
        alpha_half[p, :, i, loc.top] = alphas(p, l0, xi0, eta0, a)

        metric = N.array([[hlld_half[p, :, i, loc.top], hl1d_half[p, :, i, loc.top], hl2d_half[p, :, i, loc.top]], \
                          [hl1d_half[p, :, i, loc.top], h11d_half[p, :, i, loc.top], h12d_half[p, :, i, loc.top]], \
                          [hl2d_half[p, :, i, loc.top], h12d_half[p, :, i, loc.top], h22d_half[p, :, i, loc.top]]])

        inv_metric = N.linalg.inv(N.moveaxis(metric, 2, 0))
        hllu_half[p, :, i, loc.top] = inv_metric[:, 0, 0]
        hl1u_half[p, :, i, loc.top] = inv_metric[:, 0, 1]
        hl2u_half[p, :, i, loc.top] = inv_metric[:, 0, 2]
        h11u_half[p, :, i, loc.top] = inv_metric[:, 1, 1]
        h12u_half[p, :, i, loc.top] = inv_metric[:, 1, 2]
        h22u_half[p, :, i, loc.top] = inv_metric[:, 2, 2]
        sqrt_det_h_half[p, :, i, loc.top] = N.sqrt(N.linalg.det(N.moveaxis(metric, 2, 0)))

    # On int grid
    for i in range(Nxi_int):

        l0 = l_yee[:]
        
        # Left edge
        xi0 = xi_int[0]
        eta0 = eta_int[i]
        h11d_int[p, :, i, loc.left] = g11d(p, l0, xi0, eta0, a)
        h22d_int[p, :, i, loc.left] = g22d(p, l0, xi0, eta0, a)
        h12d_int[p, :, i, loc.left] = g12d(p, l0, xi0, eta0, a)
        hlld_int[p, :, i, loc.left] = glld(p, l0, xi0, eta0, a)
        hl1d_int[p, :, i, loc.left] = gl1d(p, l0, xi0, eta0, a)
        hl2d_int[p, :, i, loc.left] = gl2d(p, l0, xi0, eta0, a)
        alpha_int[p, :, i, loc.left] = alphas(p, l0, xi0, eta0, a)

        metric = N.array([[hlld_int[p, :, i, loc.left], hl1d_int[p, :, i, loc.left], hl2d_int[p, :, i, loc.left]], \
                          [hl1d_int[p, :, i, loc.left], h11d_int[p, :, i, loc.left], h12d_int[p, :, i, loc.left]], \
                          [hl2d_int[p, :, i, loc.left], h12d_int[p, :, i, loc.left], h22d_int[p, :, i, loc.left]]])
        
        inv_metric = N.linalg.inv(N.moveaxis(metric, 2, 0))
        hllu_int[p, :, i, loc.left] = inv_metric[:, 0, 0]
        hl1u_int[p, :, i, loc.left] = inv_metric[:, 0, 1]
        hl2u_int[p, :, i, loc.left] = inv_metric[:, 0, 2]
        h11u_int[p, :, i, loc.left] = inv_metric[:, 1, 1]
        h12u_int[p, :, i, loc.left] = inv_metric[:, 1, 2]
        h22u_int[p, :, i, loc.left] = inv_metric[:, 2, 2]
        sqrt_det_h_int[p, :, i, loc.left] = N.sqrt(N.linalg.det(N.moveaxis(metric, 2, 0)))

        # Bottom edge
        xi0 = xi_int[i]
        eta0 = eta_int[0]
        h11d_int[p, :, i, loc.bottom] = g11d(p, l0, xi0, eta0, a)
        h22d_int[p, :, i, loc.bottom] = g22d(p, l0, xi0, eta0, a)
        h12d_int[p, :, i, loc.bottom] = g12d(p, l0, xi0, eta0, a)
        hlld_int[p, :, i, loc.bottom] = glld(p, l0, xi0, eta0, a)
        hl1d_int[p, :, i, loc.bottom] = gl1d(p, l0, xi0, eta0, a)
        hl2d_int[p, :, i, loc.bottom] = gl2d(p, l0, xi0, eta0, a)
        alpha_int[p, :, i, loc.bottom] = alphas(p, l0, xi0, eta0, a)

        metric = N.array([[hlld_int[p, :, i, loc.bottom], hl1d_int[p, :, i, loc.bottom], hl2d_int[p, :, i, loc.bottom]], \
                          [hl1d_int[p, :, i, loc.bottom], h11d_int[p, :, i, loc.bottom], h12d_int[p, :, i, loc.bottom]], \
                          [hl2d_int[p, :, i, loc.bottom], h12d_int[p, :, i, loc.bottom], h22d_int[p, :, i, loc.bottom]]])
        
        inv_metric = N.linalg.inv(N.moveaxis(metric, 2, 0))
        hllu_int[p, :, i, loc.bottom] = inv_metric[:, 0, 0]
        hl1u_int[p, :, i, loc.bottom] = inv_metric[:, 0, 1]
        hl2u_int[p, :, i, loc.bottom] = inv_metric[:, 0, 2]
        h11u_int[p, :, i, loc.bottom] = inv_metric[:, 1, 1]
        h12u_int[p, :, i, loc.bottom] = inv_metric[:, 1, 2]
        h22u_int[p, :, i, loc.bottom] = inv_metric[:, 2, 2]
        sqrt_det_h_int[p, :, i, loc.bottom] = N.sqrt(N.linalg.det(N.moveaxis(metric, 2, 0)))

        # Right edge
        xi0 = xi_int[-1]
        eta0 = eta_int[i]
        h11d_int[p, :, i, loc.right] = g11d(p, l0, xi0, eta0, a)
        h22d_int[p, :, i, loc.right] = g22d(p, l0, xi0, eta0, a)
        h12d_int[p, :, i, loc.right] = g12d(p, l0, xi0, eta0, a)
        hlld_int[p, :, i, loc.right] = glld(p, l0, xi0, eta0, a)
        hl1d_int[p, :, i, loc.right] = gl1d(p, l0, xi0, eta0, a)
        hl2d_int[p, :, i, loc.right] = gl2d(p, l0, xi0, eta0, a)
        alpha_int[p, :, i, loc.right] = alphas(p, l0, xi0, eta0, a)

        metric = N.array([[hlld_int[p, :, i, loc.right], hl1d_int[p, :, i, loc.right], hl2d_int[p, :, i, loc.right]], \
                          [hl1d_int[p, :, i, loc.right], h11d_int[p, :, i, loc.right], h12d_int[p, :, i, loc.right]], \
                          [hl2d_int[p, :, i, loc.right], h12d_int[p, :, i, loc.right], h22d_int[p, :, i, loc.right]]])
        
        inv_metric = N.linalg.inv(N.moveaxis(metric, 2, 0))
        hllu_int[p, :, i, loc.right] = inv_metric[:, 0, 0]
        hl1u_int[p, :, i, loc.right] = inv_metric[:, 0, 1]
        hl2u_int[p, :, i, loc.right] = inv_metric[:, 0, 2]
        h11u_int[p, :, i, loc.right] = inv_metric[:, 1, 1]
        h12u_int[p, :, i, loc.right] = inv_metric[:, 1, 2]
        h22u_int[p, :, i, loc.right] = inv_metric[:, 2, 2]
        sqrt_det_h_int[p, :, i, loc.right] = N.sqrt(N.linalg.det(N.moveaxis(metric, 2, 0)))

        # Top edge
        xi0 = xi_int[i]
        eta0 = eta_int[-1]
        h11d_int[p, :, i, loc.top] = g11d(p, l0, xi0, eta0, a)
        h22d_int[p, :, i, loc.top] = g22d(p, l0, xi0, eta0, a)
        h12d_int[p, :, i, loc.top] = g12d(p, l0, xi0, eta0, a)
        hlld_int[p, :, i, loc.top] = glld(p, l0, xi0, eta0, a)
        hl1d_int[p, :, i, loc.top] = gl1d(p, l0, xi0, eta0, a)
        hl2d_int[p, :, i, loc.top] = gl2d(p, l0, xi0, eta0, a)
        alpha_int[p, :, i, loc.top] = alphas(p, l0, xi0, eta0, a)

        metric = N.array([[hlld_int[p, :, i, loc.top], hl1d_int[p, :, i, loc.top], hl2d_int[p, :, i, loc.top]], \
                          [hl1d_int[p, :, i, loc.top], h11d_int[p, :, i, loc.top], h12d_int[p, :, i, loc.top]], \
                          [hl2d_int[p, :, i, loc.top], h12d_int[p, :, i, loc.top], h22d_int[p, :, i, loc.top]]])
        
        inv_metric = N.linalg.inv(N.moveaxis(metric, 2, 0))
        hllu_int[p, :, i, loc.top] = inv_metric[:, 0, 0]
        hl1u_int[p, :, i, loc.top] = inv_metric[:, 0, 1]
        hl2u_int[p, :, i, loc.top] = inv_metric[:, 0, 2]
        h11u_int[p, :, i, loc.top] = inv_metric[:, 1, 1]
        h12u_int[p, :, i, loc.top] = inv_metric[:, 1, 2]
        h22u_int[p, :, i, loc.top] = inv_metric[:, 2, 2]
        sqrt_det_h_int[p, :, i, loc.top] = N.sqrt(N.linalg.det(N.moveaxis(metric, 2, 0)))

# Time step
# dt = cfl * N.min(1.0 / N.sqrt(1.0 / (dr * dr) + 1.0 / (r_max * r_max * dxi * dxi) + 1.0 / (r_max * r_max * deta * deta)))
dt = cfl * N.min(1.0 / N.sqrt(hllu / (dl * dl) + h11u / (dxi * dxi) + h22u / (deta * deta))) # + 2.0 * h12u / (dxi * deta) + 2.0 * hl1u / (dr * dxi) + 2.0 * hl2u / (dr * deta)))

# dt = cfl * N.min(1.0 / (beta[:, :, :, :, 0] / dl + alpha[:, :, :, :, 0] * N.sqrt(hllu / (dl * dl) + h11u / (dxi * dxi) + h22u / (deta * deta)))) # + 2.0 * h12u / (dxi * deta) + 2.0 * hl1u / (dr * dxi) + 2.0 * hl2u / (dr * deta)))

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

# def interp(arr_in, xA, xB):
#     f = interpolate.interp1d(xA, arr_in, axis = 1, kind='linear', fill_value=(0,0), bounds_error=False)
#     return f(xB)

# def interp(arr_in, xA, xB):
#     return N.interp(xB, xA, arr_in)

def interp_half_to_int(tab_in):
    tab_out = N.zeros((Nl - NG, Nxi_int))
    tab_out[:, 0]    = tab_in[:, 0]
    tab_out[:, -1]   = tab_in[:, -1]
    tab_out[:, 1:-1] = 0.5 * (tab_in[:, 1:-2] + N.roll(tab_in, -1, axis = 1)[:, 1:-2])
    return tab_out

def interp_int_to_half(tab_in):
    tab_out = N.zeros((Nl - NG, Nxi_half))
    tab_out[:, 0]    = tab_in[:, 0]
    tab_out[:, -1]   = tab_in[:, -1]
    tab_out[:, 1:-1] = 0.5 * (tab_in[:, 1:] + N.roll(tab_in, 1, axis = 1)[:, 1:])
    return tab_out

def interp_r_to_ryee(tab_in):
    ir1 = NG - 1
    ir2 = Nl0 + NG
    tab_out = 0.5 * (tab_in[:, ir1:ir2, :, :] + N.roll(tab_in, -1, axis = 1)[:, ir1:ir2, :, :])
    return tab_out

def interp_ryee_to_r(tab_in):
    ir1 = NG
    ir2 = Nl0 + NG + 1
    tab_out = 0.5 * (tab_in[:, ir1:ir2, :, :] + N.roll(tab_in, 1, axis = 1)[:, ir1:ir2, :, :])
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
    
    ir1 = NG
    ir2 = Nl0 + NG + 1
    
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

    dH1dl[p, ir1:ir2, :, :] = (H1d[p, ir1:ir2, :, :] - N.roll(H1d, 1, axis = 1)[p, ir1:ir2, :, :]) / dl
    dH2dl[p, ir1:ir2, :, :] = (H2d[p, ir1:ir2, :, :] - N.roll(H2d, 1, axis = 1)[p, ir1:ir2, :, :]) / dl

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

    ir1 = NG - 1
    ir2 = Nl0 + NG

    dE2d1[p, :, 0, :] = (- 0.50 * E2d[p, :, 0, :] + 0.50 * E2d[p, :, 1, :]) / dxi / P_half_2[0]
    dE2d1[p, :, 1, :] = (- 0.25 * E2d[p, :, 0, :] + 0.25 * E2d[p, :, 1, :]) / dxi / P_half_2[1]
    dE2d1[p, :, 2, :] = (- 0.25 * E2d[p, :, 0, :] - 0.75 * E2d[p, :, 1, :] + E2d[p, :, 2, :]) / dxi / P_half_2[2]
    dE2d1[p, :, Nxi_half - 3, :] = (- E2d[p, :, -3, :] + 0.75 * E2d[p, :, -2, :] + 0.25 * E2d[p, :, -1, :]) / dxi / P_half_2[Nxi_half - 3]
    dE2d1[p, :, Nxi_half - 2, :] = (- 0.25 * E2d[p, :, -2, :] + 0.25 * E2d[p, :, -1, :]) / dxi / P_half_2[Nxi_half - 2]
    dE2d1[p, :, Nxi_half - 1, :] = (- 0.5 * E2d[p, :, -2, :] + 0.5 * E2d[p, :, -1, :]) / dxi / P_half_2[Nxi_half - 1]
    dE2d1[p, :, 3:(Nxi_half - 3), :] = (E2d[p, :, 3:(Nxi_half - 3), :] - N.roll(E2d, 1, axis = 2)[p, :, 3:(Nxi_half - 3), :]) / dxi

    dE1d2[p, :, :, 0] = (- 0.50 * E1d[p, :, :, 0] + 0.50 * E1d[p, :, :, 1]) / dxi / P_half_2[0]
    dE1d2[p, :, :, 1] = (- 0.25 * E1d[p, :, :, 0] + 0.25 * E1d[p, :, :, 1]) / dxi / P_half_2[1]
    dE1d2[p, :, :, 2] = (- 0.25 * E1d[p, :, :, 0] - 0.75 * E1d[p, :, :, 1] + E1d[p, :, :, 2]) / dxi / P_half_2[2]
    dE1d2[p, :, :, Neta_half - 3] = (- E1d[p, :, :, -3] + 0.75 * E1d[p, :, :, -2] + 0.25 * E1d[p, :, :, -1]) / deta / P_half_2[Nxi_half - 3]
    dE1d2[p, :, :, Neta_half - 2] = (- 0.25 * E1d[p, :, :, -2] + 0.25 * E1d[p, :, :, -1]) / deta / P_half_2[Nxi_half - 2]
    dE1d2[p, :, :, Neta_half - 1] = (- 0.50 * E1d[p, :, :, -2] + 0.50 * E1d[p, :, :, -1]) / deta / P_half_2[Nxi_half - 1]
    dE1d2[p, :, :, 3:(Neta_half - 3)] = (E1d[p, :, :, 3:(Neta_half - 3)] - N.roll(E1d, 1, axis = 3)[p, :, :, 3:(Neta_half - 3)]) / deta

    dE1dl[p, ir1:ir2, :, :] = (N.roll(E1d, -1, axis = 1)[p, ir1:ir2, :, :] - E1d[p, ir1:ir2, :, :]) / dl
    dE2dl[p, ir1:ir2, :, :] = (N.roll(E2d, -1, axis = 1)[p, ir1:ir2, :, :] - E2d[p, ir1:ir2, :, :]) / dl

    dErd1[p, :, 0, :] = (- 0.50 * Erd[p, :, 0, :] + 0.50 * Erd[p, :, 1, :]) / dxi / P_half_2[0]
    dErd1[p, :, 1, :] = (- 0.25 * Erd[p, :, 0, :] + 0.25 * Erd[p, :, 1, :]) / dxi / P_half_2[1]
    dErd1[p, :, 2, :] = (- 0.25 * Erd[p, :, 0, :] - 0.75 * Erd[p, :, 1, :] + Erd[p, :, 2, :]) / dxi / P_half_2[2]
    dErd1[p, :, Nxi_half - 3, :] = (- Erd[p, :, -3, :] + 0.75 * Erd[p, :, -2, :] + 0.25 * Erd[p, :, -1, :]) / dxi / P_half_2[Nxi_half - 3]
    dErd1[p, :, Nxi_half - 2, :] = (- 0.25 * Erd[p, :, -2, :] + 0.25 * Erd[p, :, -1, :]) / dxi / P_half_2[Nxi_half - 2]
    dErd1[p, :, Nxi_half - 1, :] = (- 0.5 * Erd[p, :, -2, :] + 0.5 * Erd[p, :, -1, :]) / dxi / P_half_2[Nxi_half - 1]
    dErd1[p, :, 3:(Nxi_half - 3), :] = (Erd[p, :, 3:(Nxi_half - 3), :] - N.roll(Erd, 1, axis = 2)[p, :, 3:(Nxi_half - 3), :]) / dxi

    dErd2[p, :, :, 0] = (- 0.50 * Erd[p, :, :, 0] + 0.50 * Erd[p, :, :, 1]) / dxi / P_half_2[0]
    dErd2[p, :, :, 1] = (- 0.25 * Erd[p, :, :, 0] + 0.25 * Erd[p, :, :, 1]) / dxi / P_half_2[1]
    dErd2[p, :, :, 2] = (- 0.25 * Erd[p, :, :, 0] - 0.75 * Erd[p, :, :, 1] + Erd[p, :, :, 2]) / dxi / P_half_2[2]
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

    ir1 = NG - 1
    ir2 = Nl0 + NG + 1

    Drin[p, ir1:ir2, :, :] += dtin * (dH2d1[p, ir1:ir2, :, :] - dH1d2[p, ir1:ir2, :, :]) / sqrt_det_h[p, ir1:ir2, :, :, 4] 

    ir1 = NG
    ir2 = Nl0 + NG + 1
    
    # Interior
    D1in[p, ir1:ir2, 1:-1, :] += dtin * (dHrd2[p, ir1:ir2, 1:-1, :] - dH2dl[p, ir1:ir2, 1:-1, :]) / sqrt_det_h[p, ir1:ir2, :-1, :, 1] 
    # Left edge
    D1in[p, ir1:ir2, 0, :] += dtin * (dHrd2[p, ir1:ir2, 0, :] - dH2dl[p, ir1:ir2, 0, :]) / sqrt_det_h[p, ir1:ir2, 0, :, 0] 
    # Right edge
    D1in[p, ir1:ir2, -1, :] += dtin * (dHrd2[p, ir1:ir2, -1, :] - dH2dl[p, ir1:ir2, -1, :]) / sqrt_det_h[p, ir1:ir2, -1, :, 0]

    ir1 = NG
    ir2 = Nl0 + NG + 1    

    # Interior
    D2in[p, ir1:ir2, :, 1:-1] += dtin * (dH1dl[p, ir1:ir2, :, 1:-1] - dHrd1[p, ir1:ir2, :, 1:-1]) / sqrt_det_h[p, ir1:ir2, :, :-1, 2]
    # Bottom edge
    D2in[p, ir1:ir2, :, 0] += dtin * (dH1dl[p, ir1:ir2, :, 0] - dHrd1[p, ir1:ir2, :, 0]) / sqrt_det_h[p, ir1:ir2, :, 0, 0]
    # Top edge
    D2in[p, ir1:ir2, :, -1] += dtin * (dH1dl[p, ir1:ir2, :, -1] - dHrd1[p, ir1:ir2, :, -1]) / sqrt_det_h[p, ir1:ir2, :, -1, 0]

def push_B(p, Brin, B1in, B2in, dtin):

    ir1 = NG
    ir2 = Nl0 + NG + 1

    # Interior
    Brin[p, ir1:ir2, 1:-1, 1:-1] += dtin * (dE1d2[p, ir1:ir2, 1:-1, 1:-1] - dE2d1[p, ir1:ir2, 1:-1, 1:-1]) / sqrt_det_h[p, ir1:ir2, :-1, :-1, 3] 
    # Left edge
    Brin[p, ir1:ir2, 0, 1:-1] += dtin * (dE1d2[p, ir1:ir2, 0, 1:-1] - dE2d1[p, ir1:ir2, 0, 1:-1]) / sqrt_det_h[p, ir1:ir2, 0, :-1, 2] 
    # Right edge
    Brin[p, ir1:ir2, -1, 1:-1] += dtin * (dE1d2[p, ir1:ir2, -1, 1:-1] - dE2d1[p, ir1:ir2, -1, 1:-1]) / sqrt_det_h[p, ir1:ir2, -1, :-1, 2] 
    # Bottom edge
    Brin[p, ir1:ir2, 1:-1, 0] += dtin * (dE1d2[p, ir1:ir2, 1:-1, 0] - dE2d1[p, ir1:ir2, 1:-1, 0]) / sqrt_det_h[p, ir1:ir2, :-1, 0, 1] 
    # Top edge
    Brin[p, ir1:ir2, 1:-1, -1] += dtin * (dE1d2[p, ir1:ir2, 1:-1, -1] - dE2d1[p, ir1:ir2, 1:-1, -1]) / sqrt_det_h[p, ir1:ir2, :-1, -1, 1] 
    # Bottom left corner
    Brin[p, ir1:ir2, 0, 0] += dtin * (dE1d2[p, ir1:ir2, 0, 0] - dE2d1[p, ir1:ir2, 0, 0]) / sqrt_det_h[p, ir1:ir2, 0, 0, 0] 
    # Bottom right corner
    Brin[p, ir1:ir2, -1, 0] += dtin * (dE1d2[p, ir1:ir2, -1, 0] - dE2d1[p, ir1:ir2, -1, 0]) / sqrt_det_h[p, ir1:ir2, -1, 0, 0] 
    # Top left corner
    Brin[p, ir1:ir2, 0, -1] += dtin * (dE1d2[p, ir1:ir2, 0, -1] - dE2d1[p, ir1:ir2, 0, -1]) / sqrt_det_h[p, ir1:ir2, 0, -1, 0] 
    # Top right corner
    Brin[p, ir1:ir2, -1, -1] += dtin * (dE1d2[p, ir1:ir2, -1, -1] - dE2d1[p, ir1:ir2, -1, -1]) / sqrt_det_h[p, ir1:ir2, -1, -1, 0] 

    ir1 = NG - 1
    ir2 = Nl0 + NG

    # Interior
    B1in[p, ir1:ir2, :, 1:-1] += dtin * (dE2dl[p, ir1:ir2, :, 1:-1] - dErd2[p, ir1:ir2, :, 1:-1]) / sqrt_det_h[p, ir1:ir2, :, :-1, 6]
    # Bottom edge
    B1in[p, ir1:ir2, :, 0] += dtin * (dE2dl[p, ir1:ir2, :, 0] - dErd2[p, ir1:ir2, :, 0]) / sqrt_det_h[p, ir1:ir2, :, 0, 4]
    # Top edge
    B1in[p, ir1:ir2, :, -1] += dtin * (dE2dl[p, ir1:ir2, :, -1] - dErd2[p, ir1:ir2, :, -1]) / sqrt_det_h[p, ir1:ir2, :, -1, 4]

    ir1 = NG - 1
    ir2 = Nl0 + NG

    # Interior
    B2in[p, ir1:ir2, 1:-1, :] += dtin * (dErd1[p, ir1:ir2, 1:-1, :] - dE1dl[p, ir1:ir2, 1:-1, :]) / sqrt_det_h[p, ir1:ir2, :-1, :, 5] 
    # Left edge
    B2in[p, ir1:ir2, 0, :] += dtin * (dErd1[p, ir1:ir2, 0, :] - dE1dl[p, ir1:ir2, 0, :]) / sqrt_det_h[p, ir1:ir2, 0, :, 4] 
    # Right edge
    B2in[p, ir1:ir2, -1, :] += dtin * (dErd1[p, ir1:ir2, -1, :] - dE1dl[p, ir1:ir2, -1, :]) / sqrt_det_h[p, ir1:ir2, -1, :, 4]

########
# Auxiliary field computation
########

def contra_to_cov_D(p, Drin, D1in, D2in):

    ########
    # Dr
    ########
    
    ir1 = NG - 1
    ir2 = Nl0 + NG

    # Interior
    Drd[p, ir1:ir2, 1:-1, 1:-1] = hlld[p, ir1:ir2, 1:-1, 1:-1, 4] * Drin[p, ir1:ir2, 1:-1, 1:-1] \
                                      + 0.25 * hl1d[p, ir1:ir2, 1:-1, 1:-1, 4] * (D1in[p, ir1:ir2, 1:-2, 1:-1] + N.roll(N.roll(D1in, -1, axis = 1), -1, axis = 2)[p, ir1:ir2, 1:-2, 1:-1]  \
                                                                                     +  N.roll(D1in, -1, axis = 1)[p, ir1:ir2, 1:-2, 1:-1] + N.roll(D1in, -1, axis = 2)[p, ir1:ir2, 1:-2, 1:-1]) \
                                      + 0.25 * hl2d[p, ir1:ir2, 1:-1, 1:-1, 4] * (D2in[p, ir1:ir2, 1:-1, 1:-2] + N.roll(N.roll(D2in, -1, axis = 1), -1, axis = 3)[p, ir1:ir2, 1:-1, 1:-2]  \
                                                                                     +  N.roll(D2in, -1, axis = 1)[p, ir1:ir2, 1:-1, 1:-2] + N.roll(D2in, -1, axis = 3)[p, ir1:ir2, 1:-1, 1:-2]) \

    # Left edge
    Drd[p, ir1:ir2, 0, 1:-1] = hlld[p, ir1:ir2, 0, 1:-1, 4] * Drin[p, ir1:ir2, 0, 1:-1] \
                                   + 0.5  * hl1d[p, ir1:ir2, 0, 1:-1, 4] * (D1in[p, ir1:ir2, 0, 1:-1] + N.roll(D1in, -1, axis = 1)[p, ir1:ir2, 0, 1:-1]) \
                                   + 0.25 * hl2d[p, ir1:ir2, 0, 1:-1, 4] * (D2in[p, ir1:ir2, 0, 1:-2] + N.roll(N.roll(D2in, -1, axis = 1), -1, axis = 3)[p, ir1:ir2, 0, 1:-2]  \
                                                                               +  N.roll(D2in, -1, axis = 1)[p, ir1:ir2, 0, 1:-2] + N.roll(D2in, -1, axis = 3)[p, ir1:ir2, 0, 1:-2])
    # Right edge
    Drd[p, ir1:ir2, -1, 1:-1] = hlld[p, ir1:ir2, -1, 1:-1, 4] * Drin[p, ir1:ir2, -1, 1:-1] \
                                    + 0.5  * hl1d[p, ir1:ir2, -1, 1:-1, 4] * (D1in[p, ir1:ir2, -1, 1:-1] + N.roll(D1in, -1, axis = 1)[p, ir1:ir2, -1, 1:-1]) \
                                    + 0.25 * hl2d[p, ir1:ir2, -1, 1:-1, 4] * (D2in[p, ir1:ir2, -1, 1:-2] + N.roll(N.roll(D2in, -1, axis = 1), -1, axis = 3)[p, ir1:ir2, -1, 1:-2]  \
                                                                                 +  N.roll(D2in, -1, axis = 1)[p, ir1:ir2, -1, 1:-2] + N.roll(D2in, -1, axis = 3)[p, ir1:ir2, -1, 1:-2])
    # Bottom edge
    Drd[p, ir1:ir2, 1:-1, 0] = hlld[p, ir1:ir2, 1:-1, 0, 4] * Drin[p, ir1:ir2, 1:-1, 0] \
                                   + 0.25 * hl1d[p, ir1:ir2, 1:-1, 0, 4] * (D1in[p, ir1:ir2, 1:-2, 0] + N.roll(N.roll(D1in, -1, axis = 1), -1, axis = 2)[p, ir1:ir2, 1:-2, 0]  \
                                                                              +   N.roll(D1in, -1, axis = 1)[p, ir1:ir2, 1:-2, 0] + N.roll(D1in, -1, axis = 2)[p, ir1:ir2, 1:-2, 0]) \
                                   + 0.5  * hl2d[p, ir1:ir2, 1:-1, 0, 4] * (D2in[p, ir1:ir2, 1:-1, 0] + N.roll(D2in, -1, axis = 1)[p, ir1:ir2, 1:-1, 0])
    # Top edge
    Drd[p, ir1:ir2, 1:-1, -1] = hlld[p, ir1:ir2, 1:-1, -1, 4] * Drin[p, ir1:ir2, 1:-1, -1] \
                                    + 0.25 * hl1d[p, ir1:ir2, 1:-1, -1, 4] * (D1in[p, ir1:ir2, 1:-2, -1] + N.roll(N.roll(D1in, -1, axis = 1), -1, axis = 2)[p, ir1:ir2, 1:-2, -1]  \
                                                                                +   N.roll(D1in, -1, axis = 1)[p, ir1:ir2, 1:-2, -1] + N.roll(D1in, -1, axis = 2)[p, ir1:ir2, 1:-2, -1]) \
                                    + 0.5  * hl2d[p, ir1:ir2, 1:-1, -1, 4] * (D2in[p, ir1:ir2, 1:-1, -1] + N.roll(D2in, -1, axis = 1)[p, ir1:ir2, 1:-1, -1])
    # Bottom-left corner
    Drd[p, ir1:ir2, 0, 0] = hlld[p, ir1:ir2, 0, 0, 4] * Drin[p, ir1:ir2, 0, 0] \
                                + 0.5 * hl1d[p, ir1:ir2, 0, 0, 4] * (D1in[p, ir1:ir2, 0, 0] + N.roll(D1in, -1, axis = 1)[p, ir1:ir2, 0, 0]) \
                                + 0.5 * hl2d[p, ir1:ir2, 0, 0, 4] * (D2in[p, ir1:ir2, 0, 0] + N.roll(D2in, -1, axis = 1)[p, ir1:ir2, 0, 0])
    # Top-left corner
    Drd[p, ir1:ir2, 0, -1] = hlld[p, ir1:ir2, 0, -1, 4] * Drin[p, ir1:ir2, 0, -1] \
                                 + 0.5 * hl1d[p, ir1:ir2, 0, -1, 4] * (D1in[p, ir1:ir2, 0, -1] + N.roll(D1in, -1, axis = 1)[p, ir1:ir2, 0, -1]) \
                                 + 0.5 * hl2d[p, ir1:ir2, 0, -1, 4] * (D2in[p, ir1:ir2, 0, -1] + N.roll(D2in, -1, axis = 1)[p, ir1:ir2, 0, -1])
    # Bottom-right corner
    Drd[p, ir1:ir2, -1, 0] = hlld[p, ir1:ir2, -1, 0, 4] * Drin[p, ir1:ir2, -1, 0] \
                                 + 0.5 * hl1d[p, ir1:ir2, -1, 0, 4] * (D1in[p, ir1:ir2, -1, 0] + N.roll(D1in, -1, axis = 1)[p, ir1:ir2, -1, 0]) \
                                 + 0.5 * hl2d[p, ir1:ir2, -1, 0, 4] * (D2in[p, ir1:ir2, -1, 0] + N.roll(D2in, -1, axis = 1)[p, ir1:ir2, -1, 0])
    # Top-right corner
    Drd[p, ir1:ir2, -1, -1] = hlld[p, ir1:ir2, -1, -1, 4] * Drin[p, ir1:ir2, -1, -1] \
                                  + 0.5 * hl1d[p, ir1:ir2, -1, -1, 4] * (D1in[p, ir1:ir2, -1, -1] + N.roll(D1in, -1, axis = 1)[p, ir1:ir2, -1, -1]) \
                                  + 0.5 * hl2d[p, ir1:ir2, -1, -1, 4] * (D2in[p, ir1:ir2, -1, -1] + N.roll(D2in, -1, axis = 1)[p, ir1:ir2, -1, -1])

    ########
    # Dxi
    ########

    ir1 = NG
    ir2 = Nl0 + NG + 1

    # Interior
    D1d[p, ir1:ir2, 1:-1, 1:-1] = h11d[p, ir1:ir2, :-1, 1:-1, 1] * D1in[p, ir1:ir2, 1:-1, 1:-1] \
                                      + 0.25 * h12d[p, ir1:ir2, :-1, 1:-1, 1] * (D2in[p, ir1:ir2, 1:, 1:-2] + N.roll(N.roll(D2in, 1, axis = 2), -1, axis = 3)[p, ir1:ir2, 1:, 1:-2] \
                                                                                    +  N.roll(D2in, 1, axis = 2)[p, ir1:ir2, 1:, 1:-2] + N.roll(D2in, -1, axis = 3)[p, ir1:ir2, 1:, 1:-2]) \
                                      + 0.25 * hl1d[p, ir1:ir2, :-1, 1:-1, 1] * (Drin[p, ir1:ir2, 1:, 1:-1] + N.roll(N.roll(Drin, 1, axis = 1), 1, axis = 2)[p, ir1:ir2, 1:, 1:-1] \
                                                                                    +  N.roll(Drin, 1, axis = 1)[p, ir1:ir2, 1:, 1:-1] + N.roll(Drin, 1, axis = 2)[p, ir1:ir2, 1:, 1:-1])

    # Left edge
    D1d[p, ir1:ir2, 0, 1:-1] = h11d[p, ir1:ir2, 0, 1:-1, 0] * D1in[p, ir1:ir2, 0, 1:-1] \
                                   + 0.5 * h12d[p, ir1:ir2, 0, 1:-1, 0] * (D2in[p, ir1:ir2, 0, 1:-2] + N.roll(D2in, -1, axis = 3)[p, ir1:ir2, 0, 1:-2]) \
                                   + 0.5 * hl1d[p, ir1:ir2, 0, 1:-1, 0] * (Drin[p, ir1:ir2, 0, 1:-1] + N.roll(Drin, 1, axis = 1)[p, ir1:ir2, 0, 1:-1])
    # Right edge
    D1d[p, ir1:ir2, -1, 1:-1] = h11d[p, ir1:ir2, -1, 1:-1, 0] * D1in[p, ir1:ir2, -1, 1:-1] \
                                    + 0.5 * h12d[p, ir1:ir2, -1, 1:-1, 0] * (D2in[p, ir1:ir2, -1, 1:-2] + N.roll(D2in, -1, axis = 3)[p, ir1:ir2, -1, 1:-2]) \
                                    + 0.5 * hl1d[p, ir1:ir2, -1, 1:-1, 0] * (Drin[p, ir1:ir2, -1, 1:-1] + N.roll(Drin, 1, axis = 1)[p, ir1:ir2, -1, 1:-1])
    # Bottom edge
    D1d[p, ir1:ir2, 1:-1, 0] = h11d[p, ir1:ir2, :-1, 0, 1] * D1in[p, ir1:ir2, 1:-1, 0] \
                                   + 0.5  * h12d[p, ir1:ir2, :-1, 0, 1] * (D2in[p, ir1:ir2, 1:, 0] + N.roll(D2in, 1, axis = 2)[p, ir1:ir2, 1:, 0]) \
                                   + 0.25 * hl1d[p, ir1:ir2, :-1, 0, 1] * (Drin[p, ir1:ir2, 1:, 0] + N.roll(N.roll(Drin, 1, axis = 1), 1, axis = 2)[p, ir1:ir2, 1:, 0] \
                                                                              +  N.roll(Drin, 1, axis = 1)[p, ir1:ir2, 1:, 0] + N.roll(Drin, 1, axis = 2)[p, ir1:ir2, 1:, 0])
    # Top edge
    D1d[p, ir1:ir2, 1:-1, -1] = h11d[p, ir1:ir2, :-1, -1, 1] * D1in[p, ir1:ir2, 1:-1, -1] \
                                    + 0.5  * h12d[p, ir1:ir2, :-1, -1, 1] * (D2in[p, ir1:ir2, 1:, -1] + N.roll(D2in, 1, axis = 2)[p, ir1:ir2, 1:, -1]) \
                                    + 0.25 * hl1d[p, ir1:ir2, :-1, -1, 1] * (Drin[p, ir1:ir2, 1:, -1] + N.roll(N.roll(Drin, 1, axis = 1), 1, axis = 2)[p, ir1:ir2, 1:, -1] \
                                                                                +  N.roll(Drin, 1, axis = 1)[p, ir1:ir2, 1:, -1] + N.roll(Drin, 1, axis = 2)[p, ir1:ir2, 1:, -1])
    # Bottom-left corner
    D1d[p, ir1:ir2, 0, 0] = h11d[p, ir1:ir2, 0, 0, 0] * D1in[p, ir1:ir2, 0, 0] \
                                + h12d[p, ir1:ir2, 0, 0, 0] * D2in[p, ir1:ir2, 0, 0] \
                                + 0.5 * hl1d[p, ir1:ir2, 0, 0, 0] * (Drin[p, ir1:ir2, 0, 0] + N.roll(Drin, 1, axis = 1)[p, ir1:ir2, 0, 0])
    # Top-left corner
    D1d[p, ir1:ir2, 0, -1] = h11d[p, ir1:ir2, 0, -1, 0] * D1in[p, ir1:ir2, 0, -1] \
                                 + h12d[p, ir1:ir2, 0, -1, 0] * D2in[p, ir1:ir2, 0, -1] \
                                 + 0.5 * hl1d[p, ir1:ir2, 0, -1, 0] * (Drin[p, ir1:ir2, 0, -1] + N.roll(Drin, 1, axis = 1)[p, ir1:ir2, 0, -1])
    # Bottom-right corner
    D1d[p, ir1:ir2, -1, 0] = h11d[p, ir1:ir2, -1, 0, 0] * D1in[p, ir1:ir2, -1, 0] \
                                 + h12d[p, ir1:ir2, -1, 0, 0] * D2in[p, ir1:ir2, -1, 0] \
                                 + 0.5 * hl1d[p, ir1:ir2, -1, 0, 0] * (Drin[p, ir1:ir2, -1, 0] + N.roll(Drin, 1, axis = 1)[p, ir1:ir2, -1, 0])
    # Top-right corner
    D1d[p, ir1:ir2, -1, -1] = h11d[p, ir1:ir2, -1, -1, 0] * D1in[p, ir1:ir2, -1, -1] \
                                  + h12d[p, ir1:ir2, -1, -1, 0] * D2in[p, ir1:ir2, -1, -1] \
                                  + 0.5 * hl1d[p, ir1:ir2, -1, -1, 0] * (Drin[p, ir1:ir2, -1, -1] + N.roll(Drin, 1, axis = 1)[p, ir1:ir2, -1, -1])

    ########
    # Deta
    ########
    
    ir1 = NG
    ir2 = Nl0 + NG + 1

    # Interior
    D2d[p, ir1:ir2, 1:-1, 1:-1] = h22d[p, ir1:ir2, 1:-1, :-1, 2] * D2in[p, ir1:ir2, 1:-1, 1:-1] \
                          + 0.25 * h12d[p, ir1:ir2, 1:-1, :-1, 2] * (D1in[p, ir1:ir2, 1:-2, 1:] + N.roll(N.roll(D1in, -1, axis = 2), 1, axis = 3)[p, ir1:ir2, 1:-2, 1:] \
                          + N.roll(D1in, -1, axis = 2)[p, ir1:ir2, 1:-2, 1:] + N.roll(D1in, 1, axis = 3)[p, ir1:ir2, 1:-2, 1:]) \
                          + 0.25 * hl2d[p, ir1:ir2, 1:-1, :-1, 2] * (Drin[p, ir1:ir2, 1:-1, 1:] + N.roll(N.roll(Drin, 1, axis = 1), 1, axis = 3)[p, ir1:ir2, 1:-1, 1:] \
                          + N.roll(Drin, 1, axis = 1)[p, ir1:ir2, 1:-1, 1:] + N.roll(Drin, 1, axis = 3)[p, ir1:ir2, 1:-1, 1:])

    # Left edge
    D2d[p, ir1:ir2, 0, 1:-1] = h22d[p, ir1:ir2, 0, :-1, 2] * D2in[p, ir1:ir2, 0, 1:-1] \
                       + 0.5  * h12d[p, ir1:ir2, 0, :-1, 2] * (D1in[p, ir1:ir2, 0, 1:] + N.roll(D1in, 1, axis = 3)[p, ir1:ir2, 0, 1:]) \
                       + 0.25 * hl2d[p, ir1:ir2, 0, :-1, 2] * (Drin[p, ir1:ir2, 0, 1:] + N.roll(N.roll(Drin, 1, axis = 1), 1, axis = 3)[p, ir1:ir2, 0, 1:] \
                       + N.roll(Drin, 1, axis = 1)[p, ir1:ir2, 0, 1:] + N.roll(Drin, 1, axis = 3)[p, ir1:ir2, 0, 1:])
    # Right edge
    D2d[p, ir1:ir2, -1, 1:-1] = h22d[p, ir1:ir2, -1, :-1, 2] * D2in[p, ir1:ir2, -1, 1:-1] \
                        + 0.5  * h12d[p, ir1:ir2, -1, :-1, 2] * (D1in[p, ir1:ir2, -1, 1:] + N.roll(D1in, 1, axis = 3)[p, ir1:ir2, -1, 1:]) \
                        + 0.25 * hl2d[p, ir1:ir2, -1, :-1, 2] * (Drin[p, ir1:ir2, -1, 1:] + N.roll(N.roll(Drin, 1, axis = 1), 1, axis = 3)[p, ir1:ir2, -1, 1:] \
                        + N.roll(Drin, 1, axis = 1)[p, ir1:ir2, -1, 1:] + N.roll(Drin, 1, axis = 3)[p, ir1:ir2, -1, 1:])
    # Bottom edge
    D2d[p, ir1:ir2, 1:-1, 0] = h22d[p, ir1:ir2, 1:-1, 0, 0] * D2in[p, ir1:ir2, 1:-1, 0] \
                       + 0.5 * h12d[p, ir1:ir2, 1:-1, 0, 0] * (D1in[p, ir1:ir2, 1:-2, 0] + N.roll(D1in, -1, axis = 2)[p, ir1:ir2, 1:-2, 0]) \
                       + 0.5 * hl2d[p, ir1:ir2, 1:-1, 0, 0] * (Drin[p, ir1:ir2, 1:-1, 0] + N.roll(Drin, 1, axis = 1)[p, ir1:ir2, 1:-1, 0])
    # Top edge
    D2d[p, ir1:ir2, 1:-1, -1] = h22d[p, ir1:ir2, 1:-1, -1, 0] * D2in[p, ir1:ir2, 1:-1, -1] \
                        + 0.5 * h12d[p, ir1:ir2, 1:-1, -1, 0] * (D1in[p, ir1:ir2, 1:-2, -1] + N.roll(D1in, -1, axis = 2)[p, ir1:ir2, 1:-2, -1]) \
                        + 0.5 * hl2d[p, ir1:ir2, 1:-1, -1, 0] * (Drin[p, ir1:ir2, 1:-1, -1] + N.roll(Drin, 1, axis = 1)[p, ir1:ir2, 1:-1, -1])
    # Bottom-left corner
    D2d[p, ir1:ir2, 0, 0] = h22d[p, ir1:ir2, 0, 0, 0] * D2in[p, ir1:ir2, 0, 0] \
                                + h12d[p, ir1:ir2, 0, 0, 0] * D1in[p, ir1:ir2, 0, 0] \
                                + 0.5 * hl2d[p, ir1:ir2, 0, 0, 0] * (Drin[p, ir1:ir2, 0, 0] + N.roll(Drin, 1, axis = 1)[p, ir1:ir2, 0, 0])
    # Top-left corner
    D2d[p, ir1:ir2, 0, -1] = h22d[p, ir1:ir2, 0, -1, 0] * D2in[p, ir1:ir2, 0, -1] \
                                 + h12d[p, ir1:ir2, 0, -1, 0] * D1in[p, ir1:ir2, 0, -1] \
                                 + 0.5 * hl2d[p, ir1:ir2, 0, -1, 0] * (Drin[p, ir1:ir2, 0, -1] + N.roll(Drin, 1, axis = 1)[p, ir1:ir2, 0, -1])
    # Bottom-right corner
    D2d[p, ir1:ir2, -1, 0] = h22d[p, ir1:ir2, -1, 0, 0] * D2in[p, ir1:ir2, -1, 0] \
                                 + h12d[p, ir1:ir2, -1, 0, 0] * D1in[p, ir1:ir2, -1, 0] \
                                 + 0.5 * hl2d[p, ir1:ir2, -1, 0, 0] * (Drin[p, ir1:ir2, -1, 0] + N.roll(Drin, 1, axis = 1)[p, ir1:ir2, -1, 0])
    # Top-right corner
    D2d[p, ir1:ir2, -1, -1] = h22d[p, ir1:ir2, -1, -1, 0] * D2in[p, ir1:ir2, -1, -1] \
                                  + h12d[p, ir1:ir2, -1, -1, 0] * D1in[p, ir1:ir2, -1, -1] \
                                  + 0.5 * hl2d[p, ir1:ir2, -1, -1, 0] * (Drin[p, ir1:ir2, -1, -1] + N.roll(Drin, 1, axis = 1)[p, ir1:ir2, -1, -1])

def compute_E_aux(p, Drin, D1in, D2in, Brin, B1in, B2in):

    ir1 = NG - 1
    ir2 = Nl0 + NG + 1

    ##### Er
    Erd[p, ir1:ir2, :, :] = alpha[p, ir1:ir2, :, :, 4] * Drin[p, ir1:ir2, :, :]

    ir1 = NG
    ir2 = Nl0 + NG + 1

    ##### Exi
    # Interior
    E1d[p, ir1:ir2, 1:-1, :] = alpha[p, ir1:ir2, :-1, :, 1] * D1in[p, ir1:ir2, 1:-1, :] \
                                   - sqrt_det_h[p, ir1:ir2, :-1, :, 1] * beta[p, ir1:ir2, :-1, :, 1] \
                                   * 0.5 * (B2in[p, ir1:ir2, 1:-1, :] + N.roll(B2in, 1, axis = 1)[p, ir1:ir2, 1:-1, :])
    # Left edge
    E1d[p, ir1:ir2, 0, :] = alpha[p, ir1:ir2, 0, :, 0] * D1in[p, ir1:ir2, 0, :] \
                                - sqrt_det_h[p, ir1:ir2, 0, :, 0] * beta[p, ir1:ir2, 0, :, 0] \
                                * 0.5 * (B2in[p, ir1:ir2, 0, :] + N.roll(B2in, 1, axis = 1)[p, ir1:ir2, 0, :])
    # Right edge
    E1d[p, ir1:ir2, -1, :] = alpha[p, ir1:ir2, -1, :, 0] * D1in[p, ir1:ir2, -1, :] \
                                 - sqrt_det_h[p, ir1:ir2, -1, :, 0] * beta[p, ir1:ir2, -1, :, 0] \
                                 * 0.5 * (B2in[p, ir1:ir2, -1, :] + N.roll(B2in, 1, axis = 1)[p, ir1:ir2, -1, :])

    ir1 = NG
    ir2 = Nl0 + NG + 1

    ##### Eeta
    ##### Interior
    E2d[p, ir1:ir2, :, 1:-1] = alpha[p, ir1:ir2, :, :-1, 2] * D2in[p, ir1:ir2, :, 1:-1] \
                                + 0.5 * sqrt_det_h[p, ir1:ir2, :, :-1, 2] * beta[p, ir1:ir2, :, :-1, 2] \
                                * (B1in[p, ir1:ir2, :, 1:-1] + N.roll(B1in, 1, axis = 1)[p, ir1:ir2, :, 1:-1]) 
    ##### Bottom edge
    E2d[p, ir1:ir2, :, 0] = alpha[p, ir1:ir2, :, 0, 0] * D2in[p, ir1:ir2, :, 0] \
                                + 0.5 * sqrt_det_h[p, ir1:ir2, :, 0, 0] * beta[p, ir1:ir2, :, 0, 0] \
                                * (B1in[p, ir1:ir2, :, 0] + N.roll(B1in, 1, axis = 1)[p, ir1:ir2, :, 0]) 
    ##### Top edge
    E2d[p, ir1:ir2, :, -1] = alpha[p, ir1:ir2, :, -1, 0] * D2in[p, ir1:ir2, :, -1] \
                                + 0.5 * sqrt_det_h[p, ir1:ir2, :, -1, 0] * beta[p, ir1:ir2, :, -1, 0] \
                                * (B1in[p, ir1:ir2, :, -1] + N.roll(B1in, 1, axis = 1)[p, ir1:ir2, :, -1]) 


def contra_to_cov_B(p, Brin, B1in, B2in):

    ########
    # Br
    ########

    ir1 = NG
    ir2 = Nl0 + NG + 1

    # Interior
    Brd[p, ir1:ir2, 1:-1, 1:-1] = hlld[p, ir1:ir2, :-1, :-1, 3] * Brin[p, ir1:ir2, 1:-1, 1:-1] \
                                      + 0.25 * hl1d[p, ir1:ir2, :-1, :-1, 3] * (B1in[p, ir1:ir2, 1:, 1:-1] + N.roll(N.roll(B1in, 1, axis = 1), 1, axis = 2)[p, ir1:ir2, 1:, 1:-1]  \
                                                                                   +  N.roll(B1in, 1, axis = 1)[p, ir1:ir2, 1:, 1:-1] + N.roll(B1in, 1, axis = 2)[p, ir1:ir2, 1:, 1:-1]) \
                                      + 0.25 * hl2d[p, ir1:ir2, :-1, :-1, 3] * (B2in[p, ir1:ir2, 1:-1, 1:] + N.roll(N.roll(B2in, 1, axis = 1), 1, axis = 3)[p, ir1:ir2, 1:-1, 1:]  \
                                                                                   +  N.roll(B2in, 1, axis = 1)[p, ir1:ir2, 1:-1, 1:] + N.roll(B2in, 1, axis = 3)[p, ir1:ir2, 1:-1, 1:])

    # Left edge
    Brd[p, ir1:ir2, 0, 1:-1] = hlld[p, ir1:ir2, 0, :-1, 2] * Brin[p, ir1:ir2, 0, 1:-1] \
                                   + 0.5  * hl1d[p, ir1:ir2, 0, :-1, 2] * (B1in[p, ir1:ir2, 0, 1:-1] +  N.roll(B1in, 1, axis = 1)[p, ir1:ir2, 0, 1:-1]) \
                                   + 0.25 * hl2d[p, ir1:ir2, 0, :-1, 2] * (B2in[p, ir1:ir2, 0, 1:] + N.roll(N.roll(B2in, 1, axis = 1), 1, axis = 3)[p, ir1:ir2, 0, 1:]  \
                                                                              +  N.roll(B2in, 1, axis = 1)[p, ir1:ir2, 0, 1:] + N.roll(B2in, 1, axis = 3)[p, ir1:ir2, 0, 1:])

    # Right edge
    Brd[p, ir1:ir2, -1, 1:-1] = hlld[p, ir1:ir2, -1, :-1, 2] * Brin[p, ir1:ir2, -1, 1:-1] \
                                    + 0.5  * hl1d[p, ir1:ir2, -1, :-1, 2] * (B1in[p, ir1:ir2, -1, 1:-1] +  N.roll(B1in, 1, axis = 1)[p, ir1:ir2, -1, 1:-1]) \
                                    + 0.25 * hl2d[p, ir1:ir2, -1, :-1, 2] * (B2in[p, ir1:ir2, -1, 1:] + N.roll(N.roll(B2in, 1, axis = 1), 1, axis = 3)[p, ir1:ir2, -1, 1:]  \
                                                                                +  N.roll(B2in, 1, axis = 1)[p, ir1:ir2, -1, 1:] + N.roll(B2in, 1, axis = 3)[p, ir1:ir2, -1, 1:])
    
    # Bottom edge
    Brd[p, ir1:ir2, 1:-1, 0] = hlld[p, ir1:ir2, :-1, 0, 1] * Brin[p, ir1:ir2, 1:-1, 0] \
                                   + 0.25 * hl1d[p, ir1:ir2, :-1, 0, 1] * (B1in[p, ir1:ir2, 1:, 0] + N.roll(N.roll(B1in, 1, axis = 1), 1, axis = 2)[p, ir1:ir2, 1:, 0]  \
                                                                              +  N.roll(B1in, 1, axis = 1)[p, ir1:ir2, 1:, 0] + N.roll(B1in, 1, axis = 2)[p, ir1:ir2, 1:, 0]) \
                                   + 0.5  * hl2d[p, ir1:ir2, :-1, 0, 1] * (B2in[p, ir1:ir2, 1:-1, 0] +  N.roll(B2in, 1, axis = 1)[p, ir1:ir2, 1:-1, 0])
    # Top edge
    Brd[p, ir1:ir2, 1:-1, -1] = hlld[p, ir1:ir2, :-1, -1, 1] * Brin[p, ir1:ir2, 1:-1, -1] \
                                    + 0.25 * hl1d[p, ir1:ir2, :-1, -1, 1] * (B1in[p, ir1:ir2, 1:, -1] + N.roll(N.roll(B1in, 1, axis = 1), 1, axis = 2)[p, ir1:ir2, 1:, -1]  \
                                                                                +  N.roll(B1in, 1, axis = 1)[p, ir1:ir2, 1:, -1] + N.roll(B1in, 1, axis = 2)[p, ir1:ir2, 1:, -1]) \
                                    + 0.5  * hl2d[p, ir1:ir2, :-1, -1, 1] * (B2in[p, ir1:ir2, 1:-1, -1] +  N.roll(B2in, 1, axis = 1)[p, ir1:ir2, 1:-1, -1])
                                      
    # Bottom-left corner
    Brd[p, ir1:ir2, 0, 0] = hlld[p, ir1:ir2, 0, 0, 0] * Brin[p, ir1:ir2, 0, 0] \
                                + 0.5 * hl1d[p, ir1:ir2, 0, 0, 0] * (B1in[p, ir1:ir2, 0, 0] + N.roll(B1in, 1, axis = 1)[p, ir1:ir2, 0, 0]) \
                                + 0.5 * hl2d[p, ir1:ir2, 0, 0, 0] * (B2in[p, ir1:ir2, 0, 0] + N.roll(B2in, 1, axis = 1)[p, ir1:ir2, 0, 0])
    # Top-left corner
    Brd[p, ir1:ir2, 0, -1] = hlld[p, ir1:ir2, 0, -1, 0] * Brin[p, ir1:ir2, 0, -1] \
                                 + 0.5 * hl1d[p, ir1:ir2, 0, -1, 0] * (B1in[p, ir1:ir2, 0, -1] + N.roll(B1in, 1, axis = 1)[p, ir1:ir2, 0, -1]) \
                                 + 0.5 * hl2d[p, ir1:ir2, 0, -1, 0] * (B2in[p, ir1:ir2, 0, -1] + N.roll(B2in, 1, axis = 1)[p, ir1:ir2, 0, -1])
    # Bottom-right corner
    Brd[p, ir1:ir2, -1, 0] = hlld[p, ir1:ir2, -1, 0, 0] * Brin[p, ir1:ir2, -1, 0] \
                                 + 0.5 * hl1d[p, ir1:ir2, -1, 0, 0] * (B1in[p, ir1:ir2, -1, 0] + N.roll(B1in, 1, axis = 1)[p, ir1:ir2, -1, 0]) \
                                 + 0.5 * hl2d[p, ir1:ir2, -1, 0, 0] * (B2in[p, ir1:ir2, -1, 0] + N.roll(B2in, 1, axis = 1)[p, ir1:ir2, -1, 0])
    # Top-right corner
    Brd[p, ir1:ir2, -1, -1] = hlld[p, ir1:ir2, -1, -1, 0] * Brin[p, ir1:ir2, -1, -1] \
                                  + 0.5 * hl1d[p, ir1:ir2, -1, -1, 0] * (B1in[p, ir1:ir2, -1, -1] + N.roll(B1in, 1, axis = 1)[p, ir1:ir2, -1, -1]) \
                                  + 0.5 * hl2d[p, ir1:ir2, -1, -1, 0] * (B2in[p, ir1:ir2, -1, -1] + N.roll(B2in, 1, axis = 1)[p, ir1:ir2, -1, -1])

    ########
    # Bxi
    ########

    ir1 = NG - 1
    ir2 = Nl0 + NG

    # Interior
    B1d[p, ir1:ir2, 1:-1, 1:-1] = h11d[p, ir1:ir2, 1:-1, :-1, 6] * B1in[p, ir1:ir2, 1:-1, 1:-1] \
                                      + 0.25 * h12d[p, ir1:ir2, 1:-1, :-1, 6] * (B2in[p, ir1:ir2, 1:-2, 1:] + N.roll(N.roll(B2in, -1, axis = 2), 1, axis = 3)[p, ir1:ir2, 1:-2, 1:] \
                                                                                    +  N.roll(B2in, -1, axis = 2)[p, ir1:ir2, 1:-2, 1:] + N.roll(B2in, 1, axis = 3)[p, ir1:ir2, 1:-2, 1:]) \
                                      + 0.25 * hl1d[p, ir1:ir2, 1:-1, :-1, 6] * (Brin[p, ir1:ir2, 1:-2, 1:-1] + N.roll(N.roll(Brin, -1, axis = 1), -1, axis = 2)[p, ir1:ir2, 1:-2, 1:-1] \
                                                                                    +  N.roll(Brin, -1, axis = 1)[p, ir1:ir2, 1:-2, 1:-1] + N.roll(Brin, -1, axis = 2)[p, ir1:ir2, 1:-2, 1:-1])
    # Left edge
    B1d[p, ir1:ir2, 0, 1:-1] = h11d[p, ir1:ir2, 0, :-1, 6] * B1in[p, ir1:ir2, 0, 1:-1] \
                                   + 0.5 * h12d[p, ir1:ir2, 0, :-1, 6] * (B2in[p, ir1:ir2, 0, 1:] + N.roll(B2in, 1, axis = 3)[p, ir1:ir2, 0, 1:]) \
                                   + 0.5 * hl1d[p, ir1:ir2, 0, :-1, 6] * (Brin[p, ir1:ir2, 0, 1:-1] + N.roll(Brin, -1, axis = 1)[p, ir1:ir2, 0, 1:-1])
    # Right edge
    B1d[p, ir1:ir2, -1, 1:-1] = h11d[p, ir1:ir2, -1, :-1, 6] * B1in[p, ir1:ir2, -1, 1:-1] \
                                   + 0.5 * h12d[p, ir1:ir2, -1, :-1, 6] * (B2in[p, ir1:ir2, -1, 1:] + N.roll(B2in, 1, axis = 3)[p, ir1:ir2, -1, 1:]) \
                                   + 0.5 * hl1d[p, ir1:ir2, -1, :-1, 6] * (Brin[p, ir1:ir2, -1, 1:-1] + N.roll(Brin, -1, axis = 1)[p, ir1:ir2, -1, 1:-1])
    # Bottom edge
    B1d[p, ir1:ir2, 1:-1, 0] = h11d[p, ir1:ir2, 1:-1, 0, 4] * B1in[p, ir1:ir2, 1:-1, 0] \
                                   + 0.5  * h12d[p, ir1:ir2, 1:-1, 0, 4] * (B2in[p, ir1:ir2, 1:-2, 0] +  N.roll(B2in, -1, axis = 2)[p, ir1:ir2, 1:-2, 0]) \
                                   + 0.25 * hl1d[p, ir1:ir2, 1:-1, 0, 4] * (Brin[p, ir1:ir2, 1:-2, 0] + N.roll(N.roll(Brin, -1, axis = 1), -1, axis = 2)[p, ir1:ir2, 1:-2, 0] \
                                                                               +  N.roll(Brin, -1, axis = 1)[p, ir1:ir2, 1:-2, 0] + N.roll(Brin, -1, axis = 2)[p, ir1:ir2, 1:-2, 0])
    # Top edge
    B1d[p, ir1:ir2, 1:-1, -1] = h11d[p, ir1:ir2, 1:-1, -1, 4] * B1in[p, ir1:ir2, 1:-1, -1] \
                                    + 0.5  * h12d[p, ir1:ir2, 1:-1, -1, 4] * (B2in[p, ir1:ir2, 1:-2, -1] +  N.roll(B2in, -1, axis = 2)[p, ir1:ir2, 1:-2, -1]) \
                                    + 0.25 * hl1d[p, ir1:ir2, 1:-1, -1, 4] * (Brin[p, ir1:ir2, 1:-2, -1] + N.roll(N.roll(Brin, -1, axis = 1), -1, axis = 2)[p, ir1:ir2, 1:-2, -1] \
                                                                                +  N.roll(Brin, -1, axis = 1)[p, ir1:ir2, 1:-2, -1] + N.roll(Brin, -1, axis = 2)[p, ir1:ir2, 1:-2, -1])

    # Bottom-left corner
    B1d[p, ir1:ir2, 0, 0] = h11d[p, ir1:ir2, 0, 0, 4] * B1in[p, ir1:ir2, 0, 0] \
                                + h12d[p, ir1:ir2, 0, 0, 4] * B2in[p, ir1:ir2, 0, 0] \
                                + 0.5 * hl1d[p, ir1:ir2, 0, 0, 4] * (Brin[p, ir1:ir2, 0, 0] + N.roll(Brin, -1, axis = 1)[p, ir1:ir2, 0, 0])
    # Top-left corner
    B1d[p, ir1:ir2, 0, -1] = h11d[p, ir1:ir2, 0, -1, 4] * B1in[p, ir1:ir2, 0, -1] \
                                 + h12d[p, ir1:ir2, 0, -1, 4] * B2in[p, ir1:ir2, 0, -1] \
                                 + 0.5 * hl1d[p, ir1:ir2, 0, -1, 4] * (Brin[p, ir1:ir2, 0, -1] + N.roll(Brin, -1, axis = 1)[p, ir1:ir2, 0, -1])
    # Bottom-right corner
    B1d[p, ir1:ir2, -1, 0] = h11d[p, ir1:ir2, -1, 0, 4] * B1in[p, ir1:ir2, -1, 0] \
                                 + h12d[p, ir1:ir2, -1, 0, 4] * B2in[p, ir1:ir2, -1, 0] \
                                 + 0.5 * hl1d[p, ir1:ir2, -1, 0, 4] * (Brin[p, ir1:ir2, -1, 0] + N.roll(Brin, -1, axis = 1)[p, ir1:ir2, -1, 0])
    # Top-right corner
    B1d[p, ir1:ir2, -1, -1] = h11d[p, ir1:ir2, -1, -1, 4] * B1in[p, ir1:ir2, -1, -1] \
                                  + h12d[p, ir1:ir2, -1, -1, 4] * B2in[p, ir1:ir2, -1, -1] \
                                  + 0.5 * hl1d[p, ir1:ir2, -1, -1, 4] * (Brin[p, ir1:ir2, -1, -1] + N.roll(Brin, -1, axis = 1)[p, ir1:ir2, -1, -1])

    ########
    # Beta
    ########

    ir1 = NG - 1
    ir2 = Nl0 + NG

    # Interior
    B2d[p, ir1:ir2, 1:-1, 1:-1] = h22d[p, ir1:ir2, :-1, 1:-1, 5] * B2in[p, ir1:ir2, 1:-1, 1:-1] \
                                      + 0.25 * h12d[p, ir1:ir2, :-1, 1:-1, 5] * (B1in[p, ir1:ir2, 1:, 1:-2] + N.roll(N.roll(B1in, 1, axis = 2), -1, axis = 3)[p, ir1:ir2, 1:, 1:-2] \
                                                                                    +  N.roll(B1in, 1, axis = 2)[p, ir1:ir2, 1:, 1:-2] + N.roll(B1in, -1, axis = 3)[p, ir1:ir2, 1:, 1:-2]) \
                                      + 0.25 * hl2d[p, ir1:ir2, :-1, 1:-1, 5] * (Brin[p, ir1:ir2, 1:-1, 1:-2] + N.roll(N.roll(Brin, -1, axis = 1), -1, axis = 3)[p, ir1:ir2, 1:-1, 1:-2] \
                                                                                    +  N.roll(Brin, -1, axis = 1)[p, ir1:ir2, 1:-1, 1:-2] + N.roll(Brin, -1, axis = 3)[p, ir1:ir2, 1:-1, 1:-2])
    # Left edge
    B2d[p, ir1:ir2, 0, 1:-1] = h22d[p, ir1:ir2, 0, 1:-1, 4] * B2in[p, ir1:ir2, 0, 1:-1] \
                                   + 0.5  * h12d[p, ir1:ir2, 0, 1:-1, 4] * (B1in[p, ir1:ir2, 0, 1:-2] + N.roll(B1in, -1, axis = 3)[p, ir1:ir2, 0, 1:-2]) \
                                   + 0.25 * hl2d[p, ir1:ir2, 0, 1:-1, 4] * (Brin[p, ir1:ir2, 0, 1:-2] + N.roll(N.roll(Brin, -1, axis = 1), -1, axis = 3)[p, ir1:ir2, 0, 1:-2] \
                                                                               +  N.roll(Brin, -1, axis = 1)[p, ir1:ir2, 0, 1:-2] + N.roll(Brin, -1, axis = 3)[p, ir1:ir2, 0, 1:-2])
    # Right edge
    B2d[p, ir1:ir2, -1, 1:-1] = h22d[p, ir1:ir2, -1, 1:-1, 4] * B2in[p, ir1:ir2, -1, 1:-1] \
                                    + 0.5  * h12d[p, ir1:ir2, -1, 1:-1, 4] * (B1in[p, ir1:ir2, -1, 1:-2] + N.roll(B1in, -1, axis = 3)[p, ir1:ir2, -1, 1:-2]) \
                                    + 0.25 * hl2d[p, ir1:ir2, -1, 1:-1, 4] * (Brin[p, ir1:ir2, -1, 1:-2] + N.roll(N.roll(Brin, -1, axis = 1), -1, axis = 3)[p, ir1:ir2, -1, 1:-2] \
                                                                                 +  N.roll(Brin, -1, axis = 1)[p, ir1:ir2, -1, 1:-2] + N.roll(Brin, -1, axis = 3)[p, ir1:ir2, -1, 1:-2])
    # Bottom edge
    B2d[p, ir1:ir2, 1:-1, 0] = h22d[p, ir1:ir2, :-1, 0, 5] * B2in[p, ir1:ir2, 1:-1, 0] \
                                   + 0.5 * h12d[p, ir1:ir2, :-1, 0, 5] * (B1in[p, ir1:ir2, 1:, 0] + N.roll(B1in, 1, axis = 2)[p, ir1:ir2, 1:, 0]) \
                                   + 0.5 * hl2d[p, ir1:ir2, :-1, 0, 5] * (Brin[p, ir1:ir2, 1:-1, 0] + N.roll(Brin, -1, axis = 1)[p, ir1:ir2, 1:-1, 0])
    # Top edge
    B2d[p, ir1:ir2, 1:-1, -1] = h22d[p, ir1:ir2, :-1, -1, 5] * B2in[p, ir1:ir2, 1:-1, -1] \
                                   + 0.5 * h12d[p, ir1:ir2, :-1, -1, 5] * (B1in[p, ir1:ir2, 1:, -1] + N.roll(B1in, 1, axis = 2)[p, ir1:ir2, 1:, -1]) \
                                   + 0.5 * hl2d[p, ir1:ir2, :-1, -1, 5] * (Brin[p, ir1:ir2, 1:-1, -1] + N.roll(Brin, -1, axis = 1)[p, ir1:ir2, 1:-1, -1])
    # Bottom-left corner
    B2d[p, ir1:ir2, 0, 0] = h22d[p, ir1:ir2, 0, 0, 4] * B2in[p, ir1:ir2, 0, 0] \
                                + h12d[p, ir1:ir2, 0, 0, 4] * B1in[p, ir1:ir2, 0, 0] \
                                + 0.5 * hl2d[p, ir1:ir2, 0, 0, 4] * (Brin[p, ir1:ir2, 0, 0] + N.roll(Brin, -1, axis = 1)[p, ir1:ir2, 0, 0])
    # Top-left corner
    B2d[p, ir1:ir2, 0, -1] = h22d[p, ir1:ir2, 0, -1, 4] * B2in[p, ir1:ir2, 0, -1] \
                                 + h12d[p, ir1:ir2, 0, -1, 4] * B1in[p, ir1:ir2, 0, -1] \
                                 + 0.5 * hl2d[p, ir1:ir2, 0, -1, 4] * (Brin[p, ir1:ir2, 0, -1] + N.roll(Brin, -1, axis = 1)[p, ir1:ir2, 0, -1])
    # Bottom-right corner
    B2d[p, ir1:ir2, -1, 0] = h22d[p, ir1:ir2, -1, 0, 4] * B2in[p, ir1:ir2, -1, 0] \
                                 + h12d[p, ir1:ir2, -1, 0, 4] * B1in[p, ir1:ir2, -1, 0] \
                                 + 0.5 * hl2d[p, ir1:ir2, -1, 0, 4] * (Brin[p, ir1:ir2, -1, 0] + N.roll(Brin, -1, axis = 1)[p, ir1:ir2, -1, 0])
    # Top-right corner
    B2d[p, ir1:ir2, -1, -1] = h22d[p, ir1:ir2, -1, -1, 4] * B2in[p, ir1:ir2, -1, -1] \
                                  + h12d[p, ir1:ir2, -1, -1, 4] * B1in[p, ir1:ir2, -1, -1] \
                                  + 0.5 * hl2d[p, ir1:ir2, -1, -1, 4] * (Brin[p, ir1:ir2, -1, -1] + N.roll(Brin, -1, axis = 1)[p, ir1:ir2, -1, -1])

def compute_H_aux(p, Drin, D1in, D2in, Brin, B1in, B2in):

    ir1 = NG - 1
    ir2 = Nl0 + NG + 1

    ##### Hr
    # Interior
    Hrd[p, ir1:ir2, 1:-1, 1:-1] = alpha[p, ir1:ir2, :-1, :-1, 3] * Brin[p, ir1:ir2, 1:-1, 1:-1]
    # Left edge
    Hrd[p, ir1:ir2, 0, 1:-1] = alpha[p, ir1:ir2, 0, :-1, 2] * Brin[p, ir1:ir2, 0, 1:-1]
    # Right edge
    Hrd[p, ir1:ir2, -1, 1:-1] = alpha[p, ir1:ir2, -1, :-1, 2] * Brin[p, ir1:ir2, -1, 1:-1]
    # Bottom edge
    Hrd[p, ir1:ir2, 1:-1, 0] = alpha[p, ir1:ir2, :-1, 0, 1] * Brin[p, ir1:ir2, 1:-1, 0]
    # Top edge
    Hrd[p, ir1:ir2, 1:-1, -1] = alpha[p, ir1:ir2, :-1, -1, 1] * Brin[p, ir1:ir2, 1:-1, -1]
    # Corners
    Hrd[p, ir1:ir2, 0, 0]   = alpha[p, ir1:ir2, 0, 0, 0]  * Brin[p, ir1:ir2, 0, 0]
    Hrd[p, ir1:ir2, -1, 0]  = alpha[p, ir1:ir2, -1, 0, 0] * Brin[p, ir1:ir2, -1, 0]
    Hrd[p, ir1:ir2, 0, -1]  = alpha[p, ir1:ir2, 0, -1, 0] * Brin[p, ir1:ir2, 0, -1]
    Hrd[p, ir1:ir2, -1, -1] = alpha[p, ir1:ir2, -1, -1, 0]* Brin[p, ir1:ir2, -1, -1]

    ir1 = NG - 1
    ir2 = Nl0 + NG

    ##### Hxi
    # Interior
    H1d[p, ir1:ir2, :, 1:-1] = alpha[p, ir1:ir2, :, :-1, 6] * B1in[p, ir1:ir2, :, 1:-1] \
                                   + sqrt_det_h[p, ir1:ir2, :, :-1, 6] * beta[p, ir1:ir2, :, :-1, 6] \
                                   * 0.5 * (D2in[p, ir1:ir2, :, 1:-1] + N.roll(D2in, -1, axis = 1)[p, ir1:ir2, :, 1:-1])
    # Bottom edge
    H1d[p, ir1:ir2, :, 0] = alpha[p, ir1:ir2, :, 0, 4] * B1in[p, ir1:ir2, :, 0] \
                                + sqrt_det_h[p, ir1:ir2, :, 0, 4] * beta[p, ir1:ir2, :, 0, 4] \
                                * 0.5 * (D2in[p, ir1:ir2, :, 0] + N.roll(D2in, -1, axis = 1)[p, ir1:ir2, :, 0])
    # Top edge
    H1d[p, ir1:ir2, :, -1] = alpha[p, ir1:ir2, :, -1, 4] * B1in[p, ir1:ir2, :, -1] \
                                   + sqrt_det_h[p, ir1:ir2, :, -1, 4] * beta[p, ir1:ir2, :, -1, 4] \
                                   * 0.5 * (D2in[p, ir1:ir2, :, -1] + N.roll(D2in, -1, axis = 1)[p, ir1:ir2, :, -1])

    ir1 = NG - 1
    ir2 = Nl0 + NG

    ##### Heta
    ##### Interior
    H2d[p, ir1:ir2, 1:-1, :] = alpha[p, ir1:ir2, :-1, :, 5] * B2in[p, ir1:ir2, 1:-1, :] \
                                   - 0.5 * sqrt_det_h[p, ir1:ir2, :-1, :, 5] * beta[p, ir1:ir2, :-1, :, 5] \
                                   * (D1in[p, ir1:ir2, 1:-1, :] + N.roll(D1in, -1, axis = 1)[p, ir1:ir2, 1:-1, :])
    ##### Left edge
    H2d[p, ir1:ir2, 0, :] = alpha[p, ir1:ir2, 0, :, 4] * B2in[p, ir1:ir2, 0, :] \
                                   - 0.5 * sqrt_det_h[p, ir1:ir2, 0, :, 4] * beta[p, ir1:ir2, 0, :, 4] \
                                   * (D1in[p, ir1:ir2, 0, :] + N.roll(D1in, -1, axis = 1)[p, ir1:ir2, 0, :])
    ##### Right edge
    H2d[p, ir1:ir2, -1, :] = alpha[p, ir1:ir2, -1, :, 4] * B2in[p, ir1:ir2, -1, :] \
                                   - 0.5 * sqrt_det_h[p, ir1:ir2, -1, :, 4] * beta[p, ir1:ir2, -1, :, 4] \
                                   * (D1in[p, ir1:ir2, -1, :] + N.roll(D1in, -1, axis = 1)[p, ir1:ir2, -1, :])

########
# Zero-gradient boundary conditions at r_min
########

#TODO: perfectly matched BC at rmin with characteristics!

# def BC_D_rmin(patch, Drin, D1in, D2in):
#     Drin[patch, NG - 1, :, :] = Drin[patch, NG, :, :]
#     D1in[patch, NG - 1, :, :] = D1in[patch, NG, :, :]
#     D1in[patch, NG, :, :]     = D1in[patch, NG + 1, :, :]
#     D2in[patch, NG - 1, :, :] = D2in[patch, NG, :, :]
#     D2in[patch, NG, :, :]     = D2in[patch, NG + 1, :, :]

# def BC_B_rmin(patch, Brin, B1in, B2in):
#     Brin[patch, NG - 1, :, :] = Brin[patch, NG, :, :]
#     Brin[patch, NG, :, :]     = Brin[patch, NG + 1, :, :]
#     B1in[patch, NG - 1, :, :] = B1in[patch, NG, :, :]
#     B2in[patch, NG - 1, :, :] = B2in[patch, NG, :, :]

########
# Boundary conditions at r_max
########

# def BC_D_rmax(patch, Drin, D1in, D2in):
#     D1in[patch, (Nl0 + NG), :, :] = D1in[patch, (Nl0 + NG) - 1, :, :]
#     D2in[patch, (Nl0 + NG), :, :] = D2in[patch, (Nl0 + NG) - 1, :, :]

# def BC_B_rmax(patch, Brin, B1in, B2in):
#     Brin[patch, (Nl0 + NG), :, :] = Brin[patch, (Nl0 + NG) - 1, :, :]

########
# Radial boundary conditions
########

def BC_Hd(patch, Hrin, H1in, H2in):
    H1in[patch, (Nl0 + NG), :, :] = H1in[patch, (Nl0 + NG) - 1, :, :]
    H2in[patch, (Nl0 + NG), :, :] = H2in[patch, (Nl0 + NG) - 1, :, :]

def BC_Du(patch, Drin, D1in, D2in):
    Drin[patch, (Nl0 + NG), :, :] = Drin[patch, (Nl0 + NG) - 1, :, :] ## ????

    D1in[patch, NG - 1, :, :] = D1in[patch, NG, :, :]
    D2in[patch, NG - 1, :, :] = D2in[patch, NG, :, :]           

def BC_Bu(patch, Brin, B1in, B2in):
    Brin[patch, NG - 1, :, :] = Brin[patch, NG, :, :] ## ????

    B1in[patch, (Nl0 + NG), :, :] = B1in[patch, (Nl0 + NG) - 1, :, :]
    B2in[patch, (Nl0 + NG), :, :] = B2in[patch, (Nl0 + NG) - 1, :, :]

def BC_Ed(patch, Erin, E1in, E2in):
    E1in[patch, NG - 1, :, :] = E1in[patch, NG, :, :]
    E2in[patch, NG - 1, :, :] = E2in[patch, NG, :, :]

def BC_Dd(patch, Drin, D1in, D2in):
    Drin[patch, (Nl0 + NG), :, :] = Drin[patch, (Nl0 + NG) - 1, :, :]
    D1in[patch, NG - 1, :, :] = D1in[patch, NG, :, :]
    D2in[patch, NG - 1, :, :] = D2in[patch, NG, :, :]    

def BC_Bd(patch, Brin, B1in, B2in):
    Brin[patch, NG - 1, :, :] = Brin[patch, NG, :, :]
    B1in[patch, (Nl0 + NG), :, :] = B1in[patch, (Nl0 + NG) - 1, :, :]
    B2in[patch, (Nl0 + NG), :, :] = B2in[patch, (Nl0 + NG) - 1, :, :]

########
# Compute interface terms
########

sig_in  = 1.0

def compute_penalty_D(p0, p1, dtin, Drin, D1in, D2in, Brin, B1in, B2in):

    top = topology[p0, p1]

    if (top == 'xx'):

        #######
        # Dr
        #######

        ir1 = NG - 1
        ir2 = Nl0 + NG

        lambda_0 = alpha_int[p0, ir1:ir2, :, loc.right] * N.sqrt(h11u_int[p0, ir1:ir2, :, loc.right]) * sqrt_det_h_int[p0, ir1:ir2, :, loc.right]
        lambda_1 = alpha_int[p1, ir1:ir2, :, loc.left]  * N.sqrt(h11u_int[p1, ir1:ir2, :, loc.left])  * sqrt_det_h_int[p1, ir1:ir2, :, loc.left]

        # lambda_0 = alpha_int[p0, ir1:ir2, :, loc.right] / N.sqrt(h11u_int[p0, ir1:ir2, :, loc.right])
        # lambda_1 = alpha_int[p1, ir1:ir2, :, loc.left]  / N.sqrt(h11u_int[p1, ir1:ir2, :, loc.left])         
        
        Dr_0 = Drin[p0, ir1:ir2, -1, :]
        D1_0 = interp_r_to_ryee(D1in)[p0, :, -1, :]
        B2_0 = B2in[p0, ir1:ir2, -1, :]

        Dr_1 = Drin[p1, ir1:ir2, 0, :]
        D1_1, D2_1 = transform_vect(p1, p0, xi_int[None, 0], eta_int[None, :], interp_r_to_ryee(D1in)[p1, :, 0, :], interp_half_to_int(interp_r_to_ryee(D2in)[p1, :, 0, :]))
        B2_1 = B2in[p1, ir1:ir2, 0, :]

        carac_0 = (Dr_0 - hl1u_int[p0, ir1:ir2, :, loc.right] / h11u_int[p0, ir1:ir2, :, loc.right] * D1_0 + B2_0 / N.sqrt(sqrt_det_h_int**2 * h11u_int)[p0, ir1:ir2, :, loc.right])
        carac_1 = (Dr_1 - hl1u_int[p0, ir1:ir2, :, loc.right] / h11u_int[p0, ir1:ir2, :, loc.right] * D1_1 + B2_1 / N.sqrt(sqrt_det_h_int**2 * h11u_int)[p0, ir1:ir2, :, loc.right])

        diff_Dru[p0, ir1:ir2, -1, :] += dtin * sig_in * 0.5 * (carac_0 - carac_1) * lambda_0 / dxi / P_int_2[0]

        Dr_0 = Drin[p0, ir1:ir2, -1, :]
        D1_0, D2_0 = transform_vect(p0, p1, xi_int[None, -1], eta_int[None, :], interp_r_to_ryee(D1in)[p0, :, -1, :], interp_half_to_int(interp_r_to_ryee(D2in)[p0, :, -1, :]))
        B2_0 = B2in[p0, ir1:ir2, -1, :]

        Dr_1 = Drin[p1, ir1:ir2, 0, :]
        D1_1 = interp_r_to_ryee(D1in)[p1, :, 0, :]
        B2_1 = B2in[p1, ir1:ir2, 0, :]
        
        carac_1 = (Dr_1 - hl1u_int[p1, ir1:ir2, :, loc.left] / h11u_int[p1, ir1:ir2, :, loc.left] * D1_1 - B2_1 / N.sqrt(sqrt_det_h_int**2 * h11u_int)[p1, ir1:ir2, :, loc.left])
        carac_0 = (Dr_0 - hl1u_int[p1, ir1:ir2, :, loc.left] / h11u_int[p1, ir1:ir2, :, loc.left] * D1_0 - B2_0 / N.sqrt(sqrt_det_h_int**2 * h11u_int)[p1, ir1:ir2, :, loc.left])
        
        diff_Dru[p1, ir1:ir2, 0, :]  += dtin * sig_in * 0.5 * (carac_1 - carac_0) * lambda_1 / dxi / P_int_2[0]

        #######
        # D2
        #######

        ir1 = NG
        ir2 = Nl0 + NG + 1

        lambda_0 = alpha_half[p0, ir1:ir2, :, loc.right] * N.sqrt(h11u_half[p0, ir1:ir2, :, loc.right]) * sqrt_det_h_half[p0, ir1:ir2, :, loc.right]
        lambda_1 = alpha_half[p1, ir1:ir2, :, loc.left]  * N.sqrt(h11u_half[p1, ir1:ir2, :, loc.left])  * sqrt_det_h_half[p1, ir1:ir2, :, loc.left]

        # lambda_0 = alpha_half[p0, ir1:ir2, :, loc.right] / N.sqrt(h11u_half[p0, ir1:ir2, :, loc.right])
        # lambda_1 = alpha_half[p1, ir1:ir2, :, loc.left]  / N.sqrt(h11u_half[p1, ir1:ir2, :, loc.left]) 
        
        D1_0 = interp_int_to_half(D1in[p0, ir1:ir2, -1, :])
        D2_0 = D2in[p0, ir1:ir2, -1, :]
        Br_0 = Brin[p0, ir1:ir2, -1, :]
        
        D1_1, D2_1 = transform_vect(p1, p0, xi_half[None, 0], eta_half[None, :], interp_int_to_half(D1in[p1, ir1:ir2, 0, :]), D2in[p1, ir1:ir2, 0, :])
        Br_1 = Brin[p1, ir1:ir2, 0, :]
        
        carac_0 = (D2_0 - h12u_half[p0, ir1:ir2, :, loc.right] / h11u_half[p0, ir1:ir2, :, loc.right] * D1_0 - Br_0 / N.sqrt(sqrt_det_h_half**2 * h11u_half)[p0, ir1:ir2, :, loc.right])
        carac_1 = (D2_1 - h12u_half[p0, ir1:ir2, :, loc.right] / h11u_half[p0, ir1:ir2, :, loc.right] * D1_1 - Br_1 / N.sqrt(sqrt_det_h_half**2 * h11u_half)[p0, ir1:ir2, :, loc.right])

        diff_D2u[p0, ir1:ir2, -1, :] += dtin * sig_in * 0.5 * (carac_0 - carac_1) * lambda_0 / dxi / P_half_2[0]

        D1_0, D2_0 = transform_vect(p0, p1, xi_half[None, -1], eta_half[None, :], interp_int_to_half(D1in[p0, ir1:ir2, -1, :]), D2in[p0, ir1:ir2, -1, :])
        Br_0 = Brin[p0, ir1:ir2, -1, :]

        D1_1 = interp_int_to_half(D1in[p1, ir1:ir2, 0, :])
        D2_1 = D2in[p1, ir1:ir2, 0, :]
        Br_1 = Brin[p1, ir1:ir2, 0, :]        

        carac_1 = (D2_1 - h12u_half[p1, ir1:ir2, :, loc.left] / h11u_half[p1, ir1:ir2, :, loc.left] * D1_1 + Br_1 / N.sqrt(sqrt_det_h_half**2 * h11u_half)[p1, ir1:ir2, :, loc.left])
        carac_0 = (D2_0 - h12u_half[p1, ir1:ir2, :, loc.left] / h11u_half[p1, ir1:ir2, :, loc.left] * D1_0 + Br_0 / N.sqrt(sqrt_det_h_half**2 * h11u_half)[p1, ir1:ir2, :, loc.left])
        
        diff_D2u[p1, ir1:ir2, 0, :]  += dtin * sig_in * 0.5 * (carac_1 - carac_0) * lambda_1 / dxi / P_half_2[0]

    if (top == 'xy'):

        #######
        # Dr
        #######

        ir1 = NG - 1
        ir2 = Nl0 + NG
        
        lambda_0 = alpha_int[p0, ir1:ir2, :, loc.right]  * N.sqrt(h11u_int[p0, ir1:ir2, :, loc.right])  * sqrt_det_h_int[p0, ir1:ir2, :, loc.right]
        lambda_1 = alpha_int[p1, ir1:ir2, :, loc.bottom] * N.sqrt(h22u_int[p1, ir1:ir2, :, loc.bottom]) * sqrt_det_h_int[p1, ir1:ir2, :, loc.bottom]

        # lambda_0 = alpha_int[p0, ir1:ir2, :, loc.right]  / N.sqrt(h11u_int[p0, ir1:ir2, :, loc.right]) 
        # lambda_1 = alpha_int[p1, ir1:ir2, :, loc.bottom] / N.sqrt(h22u_int[p1, ir1:ir2, :, loc.bottom])

        Dr_0 = Drin[p0, ir1:ir2, -1, :]
        D1_0 = interp_r_to_ryee(D1in)[p0, :, -1, :]
        B2_0 = B2in[p0, ir1:ir2, -1, :]

        Dr_1 = Drin[p1, ir1:ir2, :, 0]
        D1_1, D2_1 = transform_vect(p1, p0, xi_int[None, :], eta_int[None, 0], interp_half_to_int(interp_r_to_ryee(D1in)[p1, :, :, 0]), interp_r_to_ryee(D2in)[p1, :, :, 0])
        B1_1 = B1in[p1, ir1:ir2, :, 0]
        
        carac_0 = (Dr_0          - hl1u_int[p0, ir1:ir2, :, loc.right] / h11u_int[p0, ir1:ir2, :, loc.right] * D1_0          + B2_0          / N.sqrt(sqrt_det_h_int**2 * h11u_int)[p0, ir1:ir2, :, loc.right])
        carac_1 = (Dr_1[:, ::-1] - hl1u_int[p0, ir1:ir2, :, loc.right] / h11u_int[p0, ir1:ir2, :, loc.right] * D1_1[:, ::-1] - B1_1[:, ::-1] / N.sqrt(sqrt_det_h_int**2 * h11u_int)[p0, ir1:ir2, :, loc.right])
        
        diff_Dru[p0, ir1:ir2, -1, :] += dtin * sig_in * 0.5 * (carac_0 - carac_1) * lambda_0 / dxi / P_int_2[0]

        Dr_0 = Drin[p0, ir1:ir2, -1, :]
        D1_0, D2_0 = transform_vect(p0, p1, xi_int[None, -1], eta_int[None, :], interp_r_to_ryee(D1in)[p0, :, -1, :], interp_half_to_int(interp_r_to_ryee(D2in)[p0, :, -1, :]))
        B2_0 = B2in[p0, ir1:ir2, -1, :]

        Dr_1 = Drin[p1, ir1:ir2, :, 0]
        D2_1 = interp_r_to_ryee(D2in)[p1, :, :, 0]
        B1_1 = B1in[p1, ir1:ir2, :, 0]
        
        carac_1 = (Dr_1          - hl2u_int[p1, ir1:ir2, :, loc.bottom] / h22u_int[p1, ir1:ir2, :, loc.bottom] * D2_1          + B1_1          / N.sqrt(sqrt_det_h_int**2 * h22u_int)[p1, ir1:ir2, :, loc.bottom])
        carac_0 = (Dr_0[:, ::-1] - hl2u_int[p1, ir1:ir2, :, loc.bottom] / h22u_int[p1, ir1:ir2, :, loc.bottom] * D2_0[:, ::-1] - B2_0[:, ::-1] / N.sqrt(sqrt_det_h_int**2 * h22u_int)[p1, ir1:ir2, :, loc.bottom])
        
        diff_Dru[p1, ir1:ir2, :, 0]  += dtin * sig_in * 0.5 * (carac_1 - carac_0) * lambda_1 / dxi / P_int_2[0]

        #######
        # D1, D2
        #######

        ir1 = NG
        ir2 = Nl0 + NG + 1

        lambda_0 = alpha_half[p0, ir1:ir2, :, loc.right]  * N.sqrt(h11u_half[p0, ir1:ir2, :, loc.right])  * sqrt_det_h_half[p0, ir1:ir2, :, loc.right]
        lambda_1 = alpha_half[p1, ir1:ir2, :, loc.bottom] * N.sqrt(h22u_half[p1, ir1:ir2, :, loc.bottom]) * sqrt_det_h_half[p1, ir1:ir2, :, loc.bottom]

        # lambda_0 = alpha_half[p0, ir1:ir2, :, loc.right]  / N.sqrt(h11u_half[p0, ir1:ir2, :, loc.right]) 
        # lambda_1 = alpha_half[p1, ir1:ir2, :, loc.bottom] / N.sqrt(h22u_half[p1, ir1:ir2, :, loc.bottom])

        D1_0 = interp_int_to_half(D1in[p0, ir1:ir2, -1, :])
        D2_0 = D2in[p0, ir1:ir2, -1, :]
        Br_0 = Brin[p0, ir1:ir2, -1, :]

        D1_1, D2_1 = transform_vect(p1, p0, xi_half[None, :], eta_half[None, 0], D1in[p1, ir1:ir2, :, 0], interp_int_to_half(D2in[p1, ir1:ir2, :, 0]))
        Br_1 = Brin[p1, ir1:ir2, :, 0]

        carac_0 = (D2_0          - h12u_half[p0, ir1:ir2, :, loc.right] / h11u_half[p0, ir1:ir2, :, loc.right] * D1_0          - Br_0          / N.sqrt(sqrt_det_h_half**2 * h11u_half)[p0, ir1:ir2, :, loc.right])
        carac_1 = (D2_1[:, ::-1] - h12u_half[p0, ir1:ir2, :, loc.right] / h11u_half[p0, ir1:ir2, :, loc.right] * D1_1[:, ::-1] - Br_1[:, ::-1] / N.sqrt(sqrt_det_h_half**2 * h11u_half)[p0, ir1:ir2, :, loc.right])
        
        diff_D2u[p0, ir1:ir2, -1, :] += dtin * sig_in * 0.5 * (carac_0 - carac_1) * lambda_0 / dxi / P_half_2[0]

        D1_0, D2_0 = transform_vect(p0, p1, xi_half[None, -1], eta_half[None, :], interp_int_to_half(D1in[p0, ir1:ir2, -1, :]), D2in[p0, ir1:ir2, -1, :])
        Br_0 = Brin[p0, ir1:ir2, -1, :]

        D1_1 = D1in[p1, ir1:ir2, :, 0]
        D2_1 = interp_int_to_half(D2in[p1, ir1:ir2, :, 0])
        Br_1 = Brin[p1, ir1:ir2, :, 0]
        
        carac_1 = (D1_1          - h12u_half[p1, ir1:ir2, :, loc.bottom] / h22u_half[p1, ir1:ir2, :, loc.bottom] * D2_1          - Br_1          / N.sqrt(sqrt_det_h_half**2 * h22u_half)[p1, ir1:ir2, :, loc.bottom])
        carac_0 = (D1_0[:, ::-1] - h12u_half[p1, ir1:ir2, :, loc.bottom] / h22u_half[p1, ir1:ir2, :, loc.bottom] * D2_0[:, ::-1] - Br_0[:, ::-1] / N.sqrt(sqrt_det_h_half**2 * h22u_half)[p1, ir1:ir2, :, loc.bottom])
        
        diff_D1u[p1, ir1:ir2, :, 0]  += dtin * sig_in * 0.5 * (carac_1 - carac_0) * lambda_1 / dxi / P_half_2[0]

    if (top == 'yy'):

        #######
        # Dr
        #######

        ir1 = NG - 1
        ir2 = Nl0 + NG

        lambda_0 = alpha_int[p0, ir1:ir2, :, loc.top]    * N.sqrt(h22u_int[p0, ir1:ir2, :, loc.top])    * sqrt_det_h_int[p0, ir1:ir2, :, loc.top]
        lambda_1 = alpha_int[p1, ir1:ir2, :, loc.bottom] * N.sqrt(h22u_int[p1, ir1:ir2, :, loc.bottom]) * sqrt_det_h_int[p1, ir1:ir2, :, loc.bottom]

        # lambda_0 = alpha_int[p0, ir1:ir2, :, loc.top]    / N.sqrt(h22u_int[p0, ir1:ir2, :, loc.top])   
        # lambda_1 = alpha_int[p1, ir1:ir2, :, loc.bottom] / N.sqrt(h22u_int[p1, ir1:ir2, :, loc.bottom])

        Dr_0 = Drin[p0, ir1:ir2, :, -1]
        D2_0 = interp_r_to_ryee(D2in)[p0, :, :, -1]
        B1_0 = B1in[p0, ir1:ir2, :, -1]
        
        Dr_1 = Drin[p1, ir1:ir2, :, 0]
        D1_1, D2_1 = transform_vect(p1, p0, xi_int[None, :], eta_int[None, 0], interp_half_to_int(interp_r_to_ryee(D1in)[p1, :, :, 0]), interp_r_to_ryee(D2in)[p1, :, :, 0])
        B1_1 = B1in[p1, ir1:ir2, :, 0]

        carac_0 = (Dr_0 - hl2u_int[p0, ir1:ir2, :, loc.top] / h22u_int[p0, ir1:ir2, :, loc.top] * D2_0 - B1_0 / N.sqrt(sqrt_det_h_int**2 * h22u_int)[p0, ir1:ir2, :, loc.top])
        carac_1 = (Dr_1 - hl2u_int[p0, ir1:ir2, :, loc.top] / h22u_int[p0, ir1:ir2, :, loc.top] * D2_1 - B1_1 / N.sqrt(sqrt_det_h_int**2 * h22u_int)[p0, ir1:ir2, :, loc.top])
        
        diff_Dru[p0, ir1:ir2, :, -1] += dtin * sig_in * 0.5 * (carac_0 - carac_1) * lambda_0 / dxi / P_int_2[0]

        Dr_0 = Drin[p0, ir1:ir2, :, -1]
        D1_0, D2_0 = transform_vect(p0, p1, xi_int[None, :], eta_int[None, -1], interp_half_to_int(interp_r_to_ryee(D1in)[p0, :, :, -1]), interp_r_to_ryee(D2in)[p0, :, :, -1])
        B1_0 = B1in[p0, ir1:ir2, :, -1]
        
        Dr_1 = Drin[p1, ir1:ir2, :, 0]
        D2_1 = interp_r_to_ryee(D2in)[p1, :, :, 0]
        B1_1 = B1in[p1, ir1:ir2, :, 0]   

        carac_1 = (Dr_1 - hl2u_int[p1, ir1:ir2, :, loc.bottom] / h22u_int[p1, ir1:ir2, :, loc.bottom] * D2_1 + B1_1 / N.sqrt(sqrt_det_h_int**2 * h22u_int)[p1, ir1:ir2, :, loc.bottom])
        carac_0 = (Dr_0 - hl2u_int[p1, ir1:ir2, :, loc.bottom] / h22u_int[p1, ir1:ir2, :, loc.bottom] * D2_0 + B1_0 / N.sqrt(sqrt_det_h_int**2 * h22u_int)[p1, ir1:ir2, :, loc.bottom])    
    
        diff_Dru[p1, ir1:ir2, :, 0]  += dtin * sig_in * 0.5 * (carac_1 - carac_0) * lambda_1 / dxi / P_int_2[0]

        #######
        # D1
        #######

        ir1 = NG
        ir2 = Nl0 + NG + 1

        lambda_0 = alpha_half[p0, ir1:ir2, :, loc.top]    * N.sqrt(h22u_half[p0, ir1:ir2, :, loc.top])    * sqrt_det_h_half[p0, ir1:ir2, :, loc.top]
        lambda_1 = alpha_half[p1, ir1:ir2, :, loc.bottom] * N.sqrt(h22u_half[p1, ir1:ir2, :, loc.bottom]) * sqrt_det_h_half[p1, ir1:ir2, :, loc.bottom]

        # lambda_0 = alpha_half[p0, ir1:ir2, :, loc.top]    / N.sqrt(h22u_half[p0, ir1:ir2, :, loc.top])   
        # lambda_1 = alpha_half[p1, ir1:ir2, :, loc.bottom] / N.sqrt(h22u_half[p1, ir1:ir2, :, loc.bottom])

        D1_0 = D1in[p0, ir1:ir2, :, -1]
        D2_0 = interp_int_to_half(D2in[p0, ir1:ir2, :, -1])
        Br_0 = Brin[p0, ir1:ir2, :, -1]

        D1_1, D2_1 = transform_vect(p1, p0, xi_half[None, :], eta_half[None, 0], D1in[p1, ir1:ir2, :, 0], interp_int_to_half(D2in[p1, ir1:ir2, :, 0]))
        Br_1 = Brin[p1, ir1:ir2, :, 0]
        
        carac_0 = (D1_0 - h12u_half[p0, ir1:ir2, :, loc.top] / h22u_half[p0, ir1:ir2, :, loc.top] * D2_0 + Br_0 / N.sqrt(sqrt_det_h_half**2 * h22u_half)[p0, ir1:ir2, :, loc.top])
        carac_1 = (D1_1 - h12u_half[p0, ir1:ir2, :, loc.top] / h22u_half[p0, ir1:ir2, :, loc.top] * D2_1 + Br_1 / N.sqrt(sqrt_det_h_half**2 * h22u_half)[p0, ir1:ir2, :, loc.top])
        
        diff_D1u[p0, ir1:ir2, :, -1] += dtin * sig_in * 0.5 * (carac_0 - carac_1) * lambda_0 / dxi / P_half_2[0]

        D1_0, D2_0 = transform_vect(p0, p1, xi_half[None, :], eta_half[None, -1], D1in[p0, ir1:ir2, :, -1], interp_int_to_half(D2in[p0, ir1:ir2, :, -1]))
        Br_0 = Brin[p0, ir1:ir2, :, -1]

        D1_1 = D1in[p1, ir1:ir2, :, 0]
        D2_1 = interp_int_to_half(D2in[p1, ir1:ir2, :, 0])
        Br_1 = Brin[p1, ir1:ir2, :, 0]

        carac_1 = (D1_1 - h12u_half[p1, ir1:ir2, :, loc.bottom] / h22u_half[p1, ir1:ir2, :, loc.bottom] * D2_1 - Br_1 / N.sqrt(sqrt_det_h_half**2 * h22u_half)[p1, ir1:ir2, :, loc.bottom])
        carac_0 = (D1_0 - h12u_half[p1, ir1:ir2, :, loc.bottom] / h22u_half[p1, ir1:ir2, :, loc.bottom] * D2_0 - Br_0 / N.sqrt(sqrt_det_h_half**2 * h22u_half)[p1, ir1:ir2, :, loc.bottom]) 
    
        diff_D1u[p1, ir1:ir2, :, 0]  += dtin * sig_in * 0.5 * (carac_1 - carac_0) * lambda_1 / dxi / P_half_2[0]

    if (top == 'yx'):

        #######
        # Dr
        #######

        ir1 = NG - 1
        ir2 = Nl0 + NG

        lambda_0 = alpha_int[p0, ir1:ir2, :, loc.top]  * N.sqrt(h22u_int[p0, ir1:ir2, :, loc.top])  * sqrt_det_h_int[p0, ir1:ir2, :, loc.top]
        lambda_1 = alpha_int[p1, ir1:ir2, :, loc.left] * N.sqrt(h11u_int[p1, ir1:ir2, :, loc.left]) * sqrt_det_h_int[p1, ir1:ir2, :, loc.left] 

        # lambda_0 = alpha_int[p0, ir1:ir2, :, loc.top]  / N.sqrt(h22u_int[p0, ir1:ir2, :, loc.top]) 
        # lambda_1 = alpha_int[p1, ir1:ir2, :, loc.left] / N.sqrt(h11u_int[p1, ir1:ir2, :, loc.left]) 

        Dr_0 = Drin[p0, ir1:ir2, :, -1]
        D2_0 = interp_r_to_ryee(D2in)[p0, :, :, -1]
        B1_0 = B1in[p0, ir1:ir2, :, -1]

        Dr_1 = Drin[p1, ir1:ir2, 0, :]
        D1_1, D2_1 = transform_vect(p1, p0, xi_int[None, 0], eta_int[None, :], interp_r_to_ryee(D1in)[p1, :, 0, :], interp_half_to_int(interp_r_to_ryee(D2in)[p1, :, 0, :]))
        B2_1 = B2in[p1, ir1:ir2, 0, :]

        carac_0 = (Dr_0          - hl2u_int[p0, ir1:ir2, :, loc.top] / h22u_int[p0, ir1:ir2, :, loc.top] * D2_0          - B1_0          / N.sqrt(sqrt_det_h_int**2 * h22u_int)[p0, ir1:ir2, :, loc.top])
        carac_1 = (Dr_1[:, ::-1] - hl2u_int[p0, ir1:ir2, :, loc.top] / h22u_int[p0, ir1:ir2, :, loc.top] * D2_1[:, ::-1] + B2_1[:, ::-1] / N.sqrt(sqrt_det_h_int**2 * h22u_int)[p0, ir1:ir2, :, loc.top])

        diff_Dru[p0, ir1:ir2, :, -1] += dtin * sig_in * 0.5 * (carac_0 - carac_1) * lambda_0 / dxi / P_int_2[0]
        
        Dr_0 = Drin[p0, ir1:ir2, :, -1]
        D1_0, D2_0 = transform_vect(p0, p1, xi_int[None, :], eta_int[None, -1], interp_half_to_int(interp_r_to_ryee(D1in)[p0, :, :, -1]), interp_r_to_ryee(D2in)[p0, :, :, -1])
        B1_0 = B1in[p0, ir1:ir2, :, -1]

        Dr_1 = Drin[p1, ir1:ir2, 0, :]
        D1_1 = interp_r_to_ryee(D1in)[p1, :, 0, :]
        B2_1 = B2in[p1, ir1:ir2, 0, :]        
    
        carac_1 = (Dr_1          - hl1u_int[p1, ir1:ir2, :, loc.left] / h11u_int[p1, ir1:ir2, :, loc.left] * D1_1          - B2_1          / N.sqrt(sqrt_det_h_int**2 * h11u_int)[p1, ir1:ir2, :, loc.left])
        carac_0 = (Dr_0[:, ::-1] - hl1u_int[p1, ir1:ir2, :, loc.left] / h11u_int[p1, ir1:ir2, :, loc.left] * D1_0[:, ::-1] + B1_0[:, ::-1] / N.sqrt(sqrt_det_h_int**2 * h11u_int)[p1, ir1:ir2, :, loc.left])
        
        diff_Dru[p1, ir1:ir2, 0, :]  += dtin * sig_in * 0.5 * (carac_1 - carac_0) * lambda_1 / dxi / P_int_2[0]

        #######
        # D1, D2
        #######

        ir1 = NG
        ir2 = Nl0 + NG + 1
        
        lambda_0 = alpha_half[p0, ir1:ir2, :, loc.top]  * N.sqrt(h22u_half[p0, ir1:ir2, :, loc.top])  * sqrt_det_h_half[p0, ir1:ir2, :, loc.top]
        lambda_1 = alpha_half[p1, ir1:ir2, :, loc.left] * N.sqrt(h11u_half[p1, ir1:ir2, :, loc.left]) * sqrt_det_h_half[p1, ir1:ir2, :, loc.left]

        # lambda_0 = alpha_half[p0, ir1:ir2, :, loc.top]  / N.sqrt(h22u_half[p0, ir1:ir2, :, loc.top]) 
        # lambda_1 = alpha_half[p1, ir1:ir2, :, loc.left] / N.sqrt(h11u_half[p1, ir1:ir2, :, loc.left])

        D1_0 = D1in[p0, ir1:ir2, :, -1]
        D2_0 = interp_int_to_half(D2in[p0, ir1:ir2, :, -1])
        Br_0 = Brin[p0, ir1:ir2, :, -1]

        D1_1, D2_1 = transform_vect(p1, p0, xi_half[None, 0], eta_half[None, :], interp_int_to_half(D1in[p1, ir1:ir2, 0, :]), D2in[p1, ir1:ir2, 0, :])
        Br_1 = Brin[p1, ir1:ir2, 0, :]

        carac_0 = (D1_0          - h12u_half[p0, ir1:ir2, :, loc.top] / h22u_half[p0, ir1:ir2, :, loc.top] * D2_0          + Br_0          / N.sqrt(sqrt_det_h_half**2 * h22u_half)[p0, ir1:ir2, :, loc.top])
        carac_1 = (D1_1[:, ::-1] - h12u_half[p0, ir1:ir2, :, loc.top] / h22u_half[p0, ir1:ir2, :, loc.top] * D2_1[:, ::-1] + Br_1[:, ::-1] / N.sqrt(sqrt_det_h_half**2 * h22u_half)[p0, ir1:ir2, :, loc.top])

        diff_D1u[p0, ir1:ir2, :, -1] += dtin * sig_in * 0.5 * (carac_0 - carac_1) * lambda_0 / dxi / P_half_2[0]

        D1_0, D2_0 = transform_vect(p0, p1, xi_half[None, :], eta_half[None, -1], D1in[p0, ir1:ir2, :, -1], interp_int_to_half(D2in[p0, ir1:ir2, :, -1]))
        Br_0 = Brin[p0, ir1:ir2, :, -1]

        D1_1 = interp_int_to_half(D1in[p1, ir1:ir2, 0, :])
        D2_1 = D2in[p1, ir1:ir2, 0, :]
        Br_1 = Brin[p1, ir1:ir2, 0, :]

        carac_1 = (D2_1          - h12u_half[p1, ir1:ir2, :, loc.left] / h11u_half[p1, ir1:ir2, :, loc.left] * D1_1          + Br_1          / N.sqrt(sqrt_det_h_half**2 * h11u_half)[p1, ir1:ir2, :, loc.left])
        carac_0 = (D2_0[:, ::-1] - h12u_half[p1, ir1:ir2, :, loc.left] / h11u_half[p1, ir1:ir2, :, loc.left] * D1_0[:, ::-1] + Br_0[:, ::-1] / N.sqrt(sqrt_det_h_half**2 * h11u_half)[p1, ir1:ir2, :, loc.left])

        diff_D2u[p1, ir1:ir2, 0, :]  += dtin * sig_in * 0.5 * (carac_1 - carac_0) * lambda_1 / dxi / P_half_2[0]


def compute_penalty_B(p0, p1, dtin, Drin, D1in, D2in, Brin, B1in, B2in):

    top = topology[p0, p1]
    
    if (top == 'xx'):

        #######
        # Br
        #######

        ir1 = NG
        ir2 = Nl0 + NG + 1

        lambda_0 = alpha_half[p0, ir1:ir2,  :, loc.right] * N.sqrt(h11u_half[p0, ir1:ir2,  :, loc.right]) * sqrt_det_h_half[p0, ir1:ir2,  :, loc.right]
        lambda_1 = alpha_half[p1, ir1:ir2,  :, loc.left]  * N.sqrt(h11u_half[p1, ir1:ir2,  :, loc.left])  * sqrt_det_h_half[p1, ir1:ir2,  :, loc.left]

        # lambda_0 = alpha_half[p0, ir1:ir2,  :, loc.right] / N.sqrt(h11u_half[p0, ir1:ir2,  :, loc.right])
        # lambda_1 = alpha_half[p1, ir1:ir2,  :, loc.left]  / N.sqrt(h11u_half[p1, ir1:ir2,  :, loc.left]) 

        D2_0 = D2in[p0, ir1:ir2,  -1, :]
        Br_0 = Brin[p0, ir1:ir2,  -1, :]
        B1_0 = interp_ryee_to_r(B1in)[p0, :, -1, :]

        D2_1 = D2in[p1, ir1:ir2,  0, :]
        Br_1 = Brin[p1, ir1:ir2,  0, :]
        B1_1, B2_1 = transform_vect(p1, p0, xi_half[None, 0], eta_half[None, :], interp_ryee_to_r(B1in)[p1, :,  0, :], interp_int_to_half(interp_ryee_to_r(B2in)[p1, :,  0, :]))

        carac_0 = (- D2_0 / N.sqrt(sqrt_det_h_half**2 * h11u_half)[p0, ir1:ir2,  :, loc.right] + Br_0 - hl1u_half[p0, ir1:ir2,  :, loc.right] / h11u_half[p0, ir1:ir2,  :, loc.right] * B1_0)
        carac_1 = (- D2_1 / N.sqrt(sqrt_det_h_half**2 * h11u_half)[p0, ir1:ir2,  :, loc.right] + Br_1 - hl1u_half[p0, ir1:ir2,  :, loc.right] / h11u_half[p0, ir1:ir2,  :, loc.right] * B1_1)

        diff_Bru[p0, ir1:ir2,  -1, :] += dtin * sig_in * 0.5 * (carac_0 - carac_1) * lambda_0 / dxi / P_half_2[0]

        D2_0 = D2in[p0, ir1:ir2,  -1, :]
        Br_0 = Brin[p0, ir1:ir2,  -1, :]
        B1_0, B2_0 = transform_vect(p0, p1, xi_half[None, -1], eta_half[None, :], interp_ryee_to_r(B1in)[p0, :,  -1, :], interp_int_to_half(interp_ryee_to_r(B2in)[p0, :,  -1, :]))

        D2_1 = D2in[p1, ir1:ir2,  0, :]
        Br_1 = Brin[p1, ir1:ir2,  0, :]
        B1_1 = interp_ryee_to_r(B1in)[p1, :,  0, :]
 
        carac_1 = (D2_1 / N.sqrt(sqrt_det_h_half**2 * h11u_half)[p1, ir1:ir2,  :, loc.left] + Br_1 - hl1u_half[p1, ir1:ir2,  :, loc.left] / h11u_half[p1, ir1:ir2,  :, loc.left] * B1_1)
        carac_0 = (D2_0 / N.sqrt(sqrt_det_h_half**2 * h11u_half)[p1, ir1:ir2,  :, loc.left] + Br_0 - hl1u_half[p1, ir1:ir2,  :, loc.left] / h11u_half[p1, ir1:ir2,  :, loc.left] * B1_0)
        
        diff_Bru[p1, ir1:ir2,  0, :]  += dtin * sig_in * 0.5 * (carac_1 - carac_0) * lambda_1 / dxi / P_half_2[0]

        #######
        # B2
        #######

        ir1 = NG - 1
        ir2 = Nl0 + NG

        lambda_0 = alpha_int[p0, ir1:ir2,  :, loc.right] * N.sqrt(h11u_int[p0, ir1:ir2,  :, loc.right]) * sqrt_det_h_int[p0, ir1:ir2,  :, loc.right]
        lambda_1 = alpha_int[p1, ir1:ir2,  :, loc.left]  * N.sqrt(h11u_int[p1, ir1:ir2,  :, loc.left])  * sqrt_det_h_int[p1, ir1:ir2,  :, loc.left]

        # lambda_0 = alpha_int[p0, ir1:ir2,  :, loc.right] / N.sqrt(h11u_int[p0, ir1:ir2,  :, loc.right])
        # lambda_1 = alpha_int[p1, ir1:ir2,  :, loc.left]  / N.sqrt(h11u_int[p1, ir1:ir2,  :, loc.left]) 

        Dr_0 = Drin[p0, ir1:ir2,  -1, :]
        B1_0 = interp_half_to_int(B1in[p0, ir1:ir2,  -1, :])
        B2_0 = B2in[p0, ir1:ir2,  -1, :]

        Dr_1 = Drin[p1, ir1:ir2,  0, :]
        B1_1, B2_1 = transform_vect(p1, p0, xi_int[None, 0], eta_int[None, :], interp_half_to_int(B1in[p1, ir1:ir2,  0, :]), B2in[p1, ir1:ir2,  0, :])

        carac_0 = (Dr_0 / N.sqrt(sqrt_det_h_int**2 * h11u_int)[p0, ir1:ir2,  :, loc.right] + B2_0 - h12u_int[p0, ir1:ir2,  :, loc.right] / h11u_int[p0, ir1:ir2,  :, loc.right] * B1_0)
        carac_1 = (Dr_1 / N.sqrt(sqrt_det_h_int**2 * h11u_int)[p0, ir1:ir2,  :, loc.right] + B2_1 - h12u_int[p0, ir1:ir2,  :, loc.right] / h11u_int[p0, ir1:ir2,  :, loc.right] * B1_1)

        diff_B2u[p0, ir1:ir2,  -1, :] += dtin * sig_in * 0.5 * (carac_0 - carac_1) * lambda_0 / dxi / P_int_2[0]

        Dr_0 = Drin[p0, ir1:ir2,  -1, :]
        B1_0, B2_0 = transform_vect(p0, p1, xi_int[None, -1], eta_int[None, :], interp_half_to_int(B1in[p0, ir1:ir2,  -1, :]), B2in[p0, ir1:ir2,  -1, :])

        Dr_1 = Drin[p1, ir1:ir2,  0, :]
        B1_1 = interp_half_to_int(B1in[p1, ir1:ir2,  0, :])
        B2_1 = B2in[p1, ir1:ir2,  0, :]        

        carac_1 = (- Dr_1 / N.sqrt(sqrt_det_h_int**2 * h11u_int)[p1, ir1:ir2,  :, loc.left] + B2_1 - h12u_int[p1, ir1:ir2,  :, loc.left] / h11u_int[p1, ir1:ir2,  :, loc.left] * B1_1)
        carac_0 = (- Dr_0 / N.sqrt(sqrt_det_h_int**2 * h11u_int)[p1, ir1:ir2,  :, loc.left] + B2_0 - h12u_int[p1, ir1:ir2,  :, loc.left] / h11u_int[p1, ir1:ir2,  :, loc.left] * B1_0)
        
        diff_B2u[p1, ir1:ir2,  0, :]  += dtin * sig_in * 0.5 * (carac_1 - carac_0) * lambda_1 / dxi / P_int_2[0]

    if (top == 'xy'):

        #######
        # Br
        #######

        ir1 = NG
        ir2 = Nl0 + NG + 1

        lambda_0 = alpha_half[p0, ir1:ir2,  :, loc.right]  * N.sqrt(h11u_half[p0, ir1:ir2,  :, loc.right])  * sqrt_det_h_half[p0, ir1:ir2,  :, loc.right]
        lambda_1 = alpha_half[p1, ir1:ir2,  :, loc.bottom] * N.sqrt(h22u_half[p1, ir1:ir2,  :, loc.bottom]) * sqrt_det_h_half[p1, ir1:ir2,  :, loc.bottom]

        # lambda_0 = alpha_half[p0, ir1:ir2,  :, loc.right]  / N.sqrt(h11u_half[p0, ir1:ir2,  :, loc.right]) 
        # lambda_1 = alpha_half[p1, ir1:ir2,  :, loc.bottom] / N.sqrt(h22u_half[p1, ir1:ir2,  :, loc.bottom])

        D2_0 = D2in[p0, ir1:ir2,  -1, :]
        Br_0 = Brin[p0, ir1:ir2,  -1, :]
        B1_0 = interp_ryee_to_r(B1in)[p0, :,  -1, :]

        D1_1 = D1in[p1, ir1:ir2,  :, 0]
        Br_1 = Brin[p1, ir1:ir2,  :, 0]
        B1_1, B2_1 = transform_vect(p1, p0, xi_half[None, :], eta_half[None, 0], interp_int_to_half(interp_ryee_to_r(B1in)[p1, :,  :, 0]), interp_ryee_to_r(B2in)[p1, :,  :, 0])

        carac_0 = (- D2_0          / N.sqrt(sqrt_det_h_half**2 * h11u_half)[p0, ir1:ir2,  :, loc.right] + Br_0          - hl1u_half[p0, ir1:ir2,  :, loc.right] / h11u_half[p0, ir1:ir2,  :, loc.right] * B1_0)
        carac_1 = (  D1_1[:, ::-1] / N.sqrt(sqrt_det_h_half**2 * h11u_half)[p0, ir1:ir2,  :, loc.right] + Br_1[:, ::-1] - hl1u_half[p0, ir1:ir2,  :, loc.right] / h11u_half[p0, ir1:ir2,  :, loc.right] * B1_1[:, ::-1])
        
        diff_Bru[p0, ir1:ir2,  -1, :] += dtin * sig_in * 0.5 * (carac_0 - carac_1) * lambda_0 / dxi / P_half_2[0]

        D2_0 = D2in[p0, ir1:ir2,  -1, :]
        Br_0 = Brin[p0, ir1:ir2,  -1, :]
        B1_0, B2_0 = transform_vect(p0, p1, xi_half[None, -1], eta_half[None, :], interp_ryee_to_r(B1in)[p0, :,  -1, :], interp_int_to_half(interp_ryee_to_r(B2in)[p0, :,  -1, :]))

        D1_1 = D1in[p1, ir1:ir2,  :, 0]
        Br_1 = Brin[p1, ir1:ir2,  :, 0]
        B2_1 = interp_ryee_to_r(B2in)[p1, :,  :, 0]
        
        carac_1 = (- D1_1          / N.sqrt(sqrt_det_h_half**2 * h22u_half)[p1, ir1:ir2,  :, loc.bottom] + Br_1          - hl2u_half[p1, ir1:ir2,  :, loc.bottom] / h22u_half[p1, ir1:ir2,  :, loc.bottom] * B2_1)
        carac_0 = (  D2_0[:, ::-1] / N.sqrt(sqrt_det_h_half**2 * h22u_half)[p1, ir1:ir2,  :, loc.bottom] + Br_0[:, ::-1] - hl2u_half[p1, ir1:ir2,  :, loc.bottom] / h22u_half[p1, ir1:ir2,  :, loc.bottom] * B2_0[:, ::-1])
        
        diff_Bru[p1, ir1:ir2,  :, 0]  += dtin * sig_in * 0.5 * (carac_1 - carac_0) * lambda_1 / dxi / P_half_2[0]

        #######
        # B1, B2
        #######

        ir1 = NG - 1
        ir2 = Nl0 + NG

        lambda_0 = alpha_int[p0, ir1:ir2,  :, loc.right]  * N.sqrt(h11u_int[p0, ir1:ir2,  :, loc.right])  * sqrt_det_h_int[p0, ir1:ir2,  :, loc.right]
        lambda_1 = alpha_int[p1, ir1:ir2,  :, loc.bottom] * N.sqrt(h22u_int[p1, ir1:ir2,  :, loc.bottom]) * sqrt_det_h_int[p1, ir1:ir2,  :, loc.bottom]

        # lambda_0 = alpha_int[p0, ir1:ir2,  :, loc.right]  / N.sqrt(h11u_int[p0, ir1:ir2,  :, loc.right]) 
        # lambda_1 = alpha_int[p1, ir1:ir2,  :, loc.bottom] / N.sqrt(h22u_int[p1, ir1:ir2,  :, loc.bottom])

        Dr_0 = Drin[p0, ir1:ir2,  -1, :]
        B1_0 = interp_half_to_int(B1in[p0, ir1:ir2,  -1, :])
        B2_0 = B2in[p0, ir1:ir2,  -1, :]

        Dr_1 = Drin[p1, ir1:ir2,  :, 0]
        B1_1, B2_1 = transform_vect(p1, p0, xi_int[None, :], eta_int[None, 0], B1in[p1, ir1:ir2,  :, 0], interp_half_to_int(B2in[p1, ir1:ir2,  :, 0]))

        carac_0 = (Dr_0          / N.sqrt(sqrt_det_h_int**2 * h11u_int)[p0, ir1:ir2,  :, loc.right] + B2_0          - h12u_int[p0, ir1:ir2,  :, loc.right] / h11u_int[p0, ir1:ir2,  :, loc.right] * B1_0)
        carac_1 = (Dr_1[:, ::-1] / N.sqrt(sqrt_det_h_int**2 * h11u_int)[p0, ir1:ir2,  :, loc.right] + B2_1[:, ::-1] - h12u_int[p0, ir1:ir2,  :, loc.right] / h11u_int[p0, ir1:ir2,  :, loc.right] * B1_1[:, ::-1])

        diff_B2u[p0, ir1:ir2,  -1, :] += dtin * sig_in * 0.5 * (carac_0 - carac_1) * lambda_0 / dxi / P_int_2[0]

        Dr_0 = Drin[p0, ir1:ir2,  -1, :]
        B1_0, B2_0 = transform_vect(p0, p1, xi_int[None, -1], eta_int[None, :], interp_half_to_int(B1in[p0, ir1:ir2,  -1, :]), B2in[p0, ir1:ir2,  -1, :])

        Dr_1 = Drin[p1, ir1:ir2,  :, 0]
        B1_1 = B1in[p1, ir1:ir2,  :, 0]
        B2_1 = interp_half_to_int(B2in[p1, ir1:ir2,  :, 0])
        
        carac_1 = (Dr_1          / N.sqrt(sqrt_det_h_int**2 * h22u_int)[p1, ir1:ir2,  :, loc.bottom] + B1_1          - h12u_int[p1, ir1:ir2,  :, loc.bottom] / h22u_int[p1, ir1:ir2,  :, loc.bottom] * B2_1)
        carac_0 = (Dr_0[:, ::-1] / N.sqrt(sqrt_det_h_int**2 * h22u_int)[p1, ir1:ir2,  :, loc.bottom] + B1_0[:, ::-1] - h12u_int[p1, ir1:ir2,  :, loc.bottom] / h22u_int[p1, ir1:ir2,  :, loc.bottom] * B2_0[:, ::-1])
        
        diff_B1u[p1, ir1:ir2,  :, 0]  += dtin * sig_in * 0.5 * (carac_1 - carac_0) * lambda_1 / dxi / P_int_2[0]

    if (top == 'yy'):
        
        #######
        # Br
        #######

        ir1 = NG
        ir2 = Nl0 + NG + 1

        lambda_0 = alpha_half[p0, ir1:ir2,  :, loc.top]    * N.sqrt(h22u_half[p0, ir1:ir2,  :, loc.top])    * sqrt_det_h_half[p0, ir1:ir2,  :, loc.top]
        lambda_1 = alpha_half[p1, ir1:ir2,  :, loc.bottom] * N.sqrt(h22u_half[p1, ir1:ir2,  :, loc.bottom]) * sqrt_det_h_half[p1, ir1:ir2,  :, loc.bottom]

        # lambda_0 = alpha_half[p0, ir1:ir2,  :, loc.top]    / N.sqrt(h22u_half[p0, ir1:ir2,  :, loc.top])   
        # lambda_1 = alpha_half[p1, ir1:ir2,  :, loc.bottom] / N.sqrt(h22u_half[p1, ir1:ir2,  :, loc.bottom])

        D1_0 = D1in[p0, ir1:ir2,  :, -1]
        Br_0 = Brin[p0, ir1:ir2,  :, -1]
        B2_0 = interp_ryee_to_r(B2in)[p0, :,  :, -1]

        D1_1 = D1in[p1, ir1:ir2,  :, 0]
        Br_1 = Brin[p1, ir1:ir2,  :, 0]
        B1_1, B2_1 = transform_vect(p1, p0, xi_half[None, :], eta_half[None, 0], interp_int_to_half(interp_ryee_to_r(B1in)[p1, :,  :, 0]), interp_ryee_to_r(B2in)[p1, :,  :, 0])

        carac_0 = (  D1_0 / N.sqrt(sqrt_det_h_half**2 * h22u_half)[p0, ir1:ir2,  :, loc.top] + Br_0 - hl2u_half[p0, ir1:ir2,  :, loc.top] / h22u_half[p0, ir1:ir2,  :, loc.top] * B2_0)
        carac_1 = (  D1_1 / N.sqrt(sqrt_det_h_half**2 * h22u_half)[p0, ir1:ir2,  :, loc.top] + Br_1 - hl2u_half[p0, ir1:ir2,  :, loc.top] / h22u_half[p0, ir1:ir2,  :, loc.top] * B2_1)

        diff_Bru[p0, ir1:ir2,  :, -1] += dtin * sig_in * 0.5 * (carac_0 - carac_1) * lambda_0 / dxi / P_half_2[0]

        D1_0 = D1in[p0, ir1:ir2,  :, -1]
        Br_0 = Brin[p0, ir1:ir2,  :, -1]
        B1_0, B2_0 = transform_vect(p0, p1, xi_half[None, :], eta_half[None, -1], interp_int_to_half(interp_ryee_to_r(B1in)[p0, :,  :, -1]), interp_ryee_to_r(B2in)[p0, :,  :, -1])

        D1_1 = D1in[p1, ir1:ir2,  :, 0]
        Br_1 = Brin[p1, ir1:ir2,  :, 0]
        B2_1 = interp_ryee_to_r(B2in)[p1, :,  :, 0]

        carac_1 = (- D1_1 / N.sqrt(sqrt_det_h_half**2 * h22u_half)[p1, ir1:ir2,  :, loc.bottom] + Br_1 - hl2u_half[p1, ir1:ir2,  :, loc.bottom] / h22u_half[p1, ir1:ir2,  :, loc.bottom] * B2_1)
        carac_0 = (- D1_0 / N.sqrt(sqrt_det_h_half**2 * h22u_half)[p1, ir1:ir2,  :, loc.bottom] + Br_0 - hl2u_half[p1, ir1:ir2,  :, loc.bottom] / h22u_half[p1, ir1:ir2,  :, loc.bottom] * B2_0)
        
        diff_Bru[p1, ir1:ir2,  :, 0]  += dtin * sig_in * 0.5 * (carac_1 - carac_0) * lambda_1 / dxi / P_half_2[0]

        #######
        # B1
        #######

        ir1 = NG - 1
        ir2 = Nl0 + NG

        lambda_0 = alpha_int[p0, ir1:ir2,  :, loc.top]    * N.sqrt(h22u_int[p0, ir1:ir2,  :, loc.top])    * sqrt_det_h_int[p0, ir1:ir2,  :, loc.top]
        lambda_1 = alpha_int[p1, ir1:ir2,  :, loc.bottom] * N.sqrt(h22u_int[p1, ir1:ir2,  :, loc.bottom]) * sqrt_det_h_int[p1, ir1:ir2,  :, loc.bottom]

        # lambda_0 = alpha_int[p0, ir1:ir2,  :, loc.top]    / N.sqrt(h22u_int[p0, ir1:ir2,  :, loc.top])   
        # lambda_1 = alpha_int[p1, ir1:ir2,  :, loc.bottom] / N.sqrt(h22u_int[p1, ir1:ir2,  :, loc.bottom])

        Dr_0 = Drin[p0, ir1:ir2,  :, -1]
        B1_0 = B1in[p0, ir1:ir2,  :, -1]
        B2_0 = interp_half_to_int(B2in[p0, ir1:ir2,  :, -1])

        Dr_1 = Drin[p1, ir1:ir2,  :, 0]
        B1_1, B2_1 = transform_vect(p1, p0, xi_int[None, :], eta_int[None, 0], B1in[p1, ir1:ir2,  :, 0], interp_half_to_int(B2in[p1, ir1:ir2,  :, 0]))
        
        carac_0 = (- Dr_0 / N.sqrt(sqrt_det_h_int**2 * h22u_int)[p0, ir1:ir2,  :, loc.top] + B1_0 - h12u_int[p0, ir1:ir2,  :, loc.top] / h22u_int[p0, ir1:ir2,  :, loc.top] * B2_0)
        carac_1 = (- Dr_1 / N.sqrt(sqrt_det_h_int**2 * h22u_int)[p0, ir1:ir2,  :, loc.top] + B1_1 - h12u_int[p0, ir1:ir2,  :, loc.top] / h22u_int[p0, ir1:ir2,  :, loc.top] * B2_1)
        
        diff_B1u[p0, ir1:ir2,  :, -1] += dtin * sig_in * 0.5 * (carac_0 - carac_1) * lambda_0 / dxi / P_int_2[0]

        Dr_0 = Drin[p0, ir1:ir2,  :, -1]
        B1_0, B2_0 = transform_vect(p0, p1, xi_int[None, :], eta_int[None, -1], B1in[p0, ir1:ir2,  :, -1], interp_half_to_int(B2in[p0, ir1:ir2,  :, -1]))

        Dr_1 = Drin[p1, ir1:ir2,  :, 0]
        B1_1 = B1in[p1, ir1:ir2,  :, 0]
        B2_1 = interp_half_to_int(B2in[p1, ir1:ir2,  :, 0])

        carac_1 = (Dr_1 / N.sqrt(sqrt_det_h_int**2 * h22u_int)[p1, ir1:ir2,  :, loc.bottom] + B1_1 - h12u_int[p1, ir1:ir2,  :, loc.bottom] / h22u_int[p1, ir1:ir2,  :, loc.bottom] * B2_1)
        carac_0 = (Dr_0 / N.sqrt(sqrt_det_h_int**2 * h22u_int)[p1, ir1:ir2,  :, loc.bottom] + B1_0 - h12u_int[p1, ir1:ir2,  :, loc.bottom] / h22u_int[p1, ir1:ir2,  :, loc.bottom] * B2_0)

        diff_B1u[p1, ir1:ir2,  :, 0]  += dtin * sig_in * 0.5 * (carac_1 - carac_0) * lambda_1 / dxi / P_int_2[0]
        
    if (top == 'yx'):

        #######
        # Br
        #######

        ir1 = NG
        ir2 = Nl0 + NG + 1

        lambda_0 = alpha_half[p0, ir1:ir2,  :, loc.top]  * N.sqrt(h22u_half[p0, ir1:ir2,  :, loc.top])  * sqrt_det_h_half[p0, ir1:ir2,  :, loc.top]
        lambda_1 = alpha_half[p1, ir1:ir2,  :, loc.left] * N.sqrt(h11u_half[p1, ir1:ir2,  :, loc.left]) * sqrt_det_h_half[p1, ir1:ir2,  :, loc.left]

        # lambda_0 = alpha_half[p0, ir1:ir2,  :, loc.top]  / N.sqrt(h22u_half[p0, ir1:ir2,  :, loc.top]) 
        # lambda_1 = alpha_half[p1, ir1:ir2,  :, loc.left] / N.sqrt(h11u_half[p1, ir1:ir2,  :, loc.left])

        D1_0 = D1in[p0, ir1:ir2,  :, -1]
        Br_0 = Brin[p0, ir1:ir2,  :, -1]
        B2_0 = interp_ryee_to_r(B2in)[p0, :,  :, -1]

        D2_1 = D2in[p1, ir1:ir2,  0, :]
        Br_1 = Brin[p1, ir1:ir2,  0, :]
        B1_1, B2_1 = transform_vect(p1, p0, xi_half[None, 0], eta_half[None, :], interp_ryee_to_r(B1in)[p1, :,  0, :], interp_int_to_half(interp_ryee_to_r(B2in)[p1, :,  0, :]))

        carac_0 = (  D1_0          / N.sqrt(sqrt_det_h_half**2 * h22u_half)[p0, ir1:ir2,  :, loc.top] + Br_0          - hl2u_half[p0, ir1:ir2,  :, loc.top] / h22u_half[p0, ir1:ir2,  :, loc.top] * B2_0)
        carac_1 = (- D2_1[:, ::-1] / N.sqrt(sqrt_det_h_half**2 * h22u_half)[p0, ir1:ir2,  :, loc.top] + Br_1[:, ::-1] - hl2u_half[p0, ir1:ir2,  :, loc.top] / h22u_half[p0, ir1:ir2,  :, loc.top] * B2_1[:, ::-1])

        diff_Bru[p0, ir1:ir2,  :, -1] += dtin * sig_in * 0.5 * (carac_0 - carac_1) * lambda_0 / dxi / P_half_2[0]

        D1_0 = D1in[p0, ir1:ir2,  :, -1]
        Br_0 = Brin[p0, ir1:ir2,  :, -1]
        B1_0, B2_0 = transform_vect(p0, p1, xi_half[None, :], eta_half[None, -1], interp_int_to_half(interp_ryee_to_r(B1in)[p0, :,  :, -1]), interp_ryee_to_r(B2in)[p0, :,  :, -1])

        D2_1 = D2in[p1, ir1:ir2,  0, :]
        Br_1 = Brin[p1, ir1:ir2,  0, :]
        B1_1 = interp_ryee_to_r(B1in)[p1, :,  0, :]
        
        carac_1 = (  D2_1          / N.sqrt(sqrt_det_h_half**2 * h11u_half)[p1, ir1:ir2,  :, loc.left] + Br_1          - hl1u_half[p1, ir1:ir2,  :, loc.left] / h11u_half[p1, ir1:ir2,  :, loc.left] * B1_1)
        carac_0 = (- D1_0[:, ::-1] / N.sqrt(sqrt_det_h_half**2 * h11u_half)[p1, ir1:ir2,  :, loc.left] + Br_0[:, ::-1] - hl1u_half[p1, ir1:ir2,  :, loc.left] / h11u_half[p1, ir1:ir2,  :, loc.left] * B1_0[:, ::-1])
        
        diff_Bru[p1, ir1:ir2,  0, :]  += dtin * sig_in * 0.5 * (carac_1 - carac_0) * lambda_1 / dxi / P_half_2[0]

        #######
        # B1, B2
        #######

        ir1 = NG - 1
        ir2 = Nl0 + NG

        lambda_0 = alpha_int[p0, ir1:ir2,  :, loc.top]  * N.sqrt(h22u_int[p0, ir1:ir2,  :, loc.top])  * sqrt_det_h_int[p0, ir1:ir2,  :, loc.top]
        lambda_1 = alpha_int[p1, ir1:ir2,  :, loc.left] * N.sqrt(h11u_int[p1, ir1:ir2,  :, loc.left]) * sqrt_det_h_int[p1, ir1:ir2,  :, loc.left]

        # lambda_0 = alpha_int[p0, ir1:ir2,  :, loc.top]  / N.sqrt(h22u_int[p0, ir1:ir2,  :, loc.top]) 
        # lambda_1 = alpha_int[p1, ir1:ir2,  :, loc.left] / N.sqrt(h11u_int[p1, ir1:ir2,  :, loc.left])

        Dr_0 = Drin[p0, ir1:ir2,  :, -1]
        B1_0 = B1in[p0, ir1:ir2,  :, -1]
        B2_0 = interp_half_to_int(B2in[p0, ir1:ir2,  :, -1])

        Dr_1 = Drin[p1, ir1:ir2,  0, :]
        B1_1, B2_1 = transform_vect(p1, p0, xi_int[None, 0], eta_int[None, :], interp_half_to_int(B1in[p1, ir1:ir2,  0, :]), B2in[p1, ir1:ir2,  0, :])

        carac_0 = (- Dr_0          / N.sqrt(sqrt_det_h_int**2 * h22u_int)[p0, ir1:ir2,  :, loc.top] + B1_0          - h12u_int[p0, ir1:ir2,  :, loc.top] / h22u_int[p0, ir1:ir2,  :, loc.top] * B2_0)
        carac_1 = (- Dr_1[:, ::-1] / N.sqrt(sqrt_det_h_int**2 * h22u_int)[p0, ir1:ir2,  :, loc.top] + B1_1[:, ::-1] - h12u_int[p0, ir1:ir2,  :, loc.top] / h22u_int[p0, ir1:ir2,  :, loc.top] * B2_1[:, ::-1])

        diff_B1u[p0, ir1:ir2,  :, -1] += dtin * sig_in * 0.5 * (carac_0 - carac_1) * lambda_0 / dxi / P_int_2[0]

        Dr_0 = Drin[p0, ir1:ir2,  :, -1]
        B1_0, B2_0 = transform_vect(p0, p1, xi_int[None, :], eta_int[None, -1], B1in[p0, ir1:ir2,  :, -1], interp_half_to_int(B2in[p0, ir1:ir2,  :, -1]))

        Dr_1 = Drin[p1, ir1:ir2,  0, :]
        B1_1 = interp_half_to_int(B1in[p1, ir1:ir2,  0, :])
        B2_1 = B2in[p1, ir1:ir2,  0, :]

        carac_1 = (- Dr_1          / N.sqrt(sqrt_det_h_int**2 * h11u_int)[p1, ir1:ir2,  :, loc.left] + B2_1          - h12u_int[p1, ir1:ir2,  :, loc.left] / h11u_int[p1, ir1:ir2,  :, loc.left] * B1_1)
        carac_0 = (- Dr_0[:, ::-1] / N.sqrt(sqrt_det_h_int**2 * h11u_int)[p1, ir1:ir2,  :, loc.left] + B2_0[:, ::-1] - h12u_int[p1, ir1:ir2,  :, loc.left] / h11u_int[p1, ir1:ir2,  :, loc.left] * B1_0[:, ::-1])

        diff_B2u[p1, ir1:ir2,  0, :]  += dtin * sig_in * 0.5 * (carac_1 - carac_0) * lambda_1 / dxi / P_int_2[0]

########
# Apply penalty terms to E, D
########

def interface_D(p0, p1, Drin, D1in, D2in):

    ir1 = NG -1
    ir2 = Nl0 + NG + 1

    irr1 = NG

    i0 =  1
    i1_half = Nxi_half - 1
    i1_int  = Nxi_int - 1

    top = topology[p0, p1]
    
    if (top == 'xx'):
        Drin[p0, ir1:ir2, -1, i0:i1_int] -= diff_Dru[p0, ir1:ir2, -1, i0:i1_int] / sqrt_det_h_int[p0, ir1:ir2, i0:i1_int, loc.right]
        Drin[p1, ir1:ir2, 0, i0:i1_int]  -= diff_Dru[p1, ir1:ir2, 0, i0:i1_int]  / sqrt_det_h_int[p1, ir1:ir2, i0:i1_int, loc.left]
        
        D2in[p0, irr1:ir2, -1, i0:i1_half] -= diff_D2u[p0, irr1:ir2, -1, i0:i1_half] / sqrt_det_h_half[p0, irr1:ir2, i0:i1_half, loc.right]
        D2in[p1, irr1:ir2, 0, i0:i1_half]  -= diff_D2u[p1, irr1:ir2, 0, i0:i1_half]  / sqrt_det_h_half[p1, irr1:ir2, i0:i1_half, loc.left]

    if (top == 'xy'):
        Drin[p0, ir1:ir2, -1, i0:i1_int] -= diff_Dru[p0, ir1:ir2, -1, i0:i1_int] / sqrt_det_h_int[p0, ir1:ir2, i0:i1_int, loc.right]
        Drin[p1, ir1:ir2, i0:i1_int, 0]  -= diff_Dru[p1, ir1:ir2, i0:i1_int, 0]  / sqrt_det_h_int[p1, ir1:ir2, i0:i1_int, loc.bottom]

        D2in[p0, irr1:ir2, -1, i0:i1_half] -= diff_D2u[p0, irr1:ir2, -1, i0:i1_half] / sqrt_det_h_half[p0, irr1:ir2, i0:i1_half, loc.right]
        D1in[p1, irr1:ir2, i0:i1_half, 0]  -= diff_D1u[p1, irr1:ir2, i0:i1_half, 0]  / sqrt_det_h_half[p1, irr1:ir2, i0:i1_half, loc.bottom]

    if (top == 'yy'):
        Drin[p0, ir1:ir2, i0:i1_int, -1] -= diff_Dru[p0, ir1:ir2, i0:i1_int, -1] / sqrt_det_h_int[p0, ir1:ir2, i0:i1_int, loc.top]
        Drin[p1, ir1:ir2, i0:i1_int, 0]  -= diff_Dru[p1, ir1:ir2, i0:i1_int, 0]  / sqrt_det_h_int[p1, ir1:ir2, i0:i1_int, loc.bottom]

        D1in[p0, irr1:ir2, i0:i1_half, -1] -= diff_D1u[p0, irr1:ir2, i0:i1_half, -1] / sqrt_det_h_half[p0, irr1:ir2, i0:i1_half, loc.top]
        D1in[p1, irr1:ir2, i0:i1_half, 0]  -= diff_D1u[p1, irr1:ir2, i0:i1_half, 0]  / sqrt_det_h_half[p1, irr1:ir2, i0:i1_half, loc.bottom]

    if (top == 'yx'):
        Drin[p0, ir1:ir2, i0:i1_int, -1] -= diff_Dru[p0, ir1:ir2, i0:i1_int, -1] / sqrt_det_h_int[p0, ir1:ir2, i0:i1_int, loc.top]
        Drin[p1, ir1:ir2, 0, i0:i1_int]  -= diff_Dru[p1, ir1:ir2, 0, i0:i1_int]  / sqrt_det_h_int[p1, ir1:ir2, i0:i1_int, loc.left]

        D1in[p0, irr1:ir2, i0:i1_half, -1] -= diff_D1u[p0, irr1:ir2, i0:i1_half, -1] / sqrt_det_h_half[p0, irr1:ir2, i0:i1_half, loc.top]
        D2in[p1, irr1:ir2, 0, i0:i1_half]  -= diff_D2u[p1, irr1:ir2, 0, i0:i1_half]  / sqrt_det_h_half[p1, irr1:ir2, i0:i1_half, loc.left]

def corners_D(p0, Drin, D1in, D2in):

    ir1 = NG - 1
    ir2 = Nl0 + NG + 1
    
    irr1 = NG

    Drin[p0, ir1:ir2, 0, 0]   -= diff_Dru[p0, ir1:ir2, 0, 0]   / sqrt_det_h_int[p0, ir1:ir2, 0, loc.bottom]
    Drin[p0, ir1:ir2, -1, 0]  -= diff_Dru[p0, ir1:ir2, -1, 0]  / sqrt_det_h_int[p0, ir1:ir2, -1, loc.bottom]
    Drin[p0, ir1:ir2, 0, -1]  -= diff_Dru[p0, ir1:ir2, 0, -1]  / sqrt_det_h_int[p0, ir1:ir2, 0, loc.top]
    Drin[p0, ir1:ir2, -1, -1] -= diff_Dru[p0, ir1:ir2, -1, -1] / sqrt_det_h_int[p0, ir1:ir2, -1, loc.top]

    D1in[p0, irr1:ir2, 0, 0]   -= diff_D1u[p0, irr1:ir2, 0, 0]   / sqrt_det_h_half[p0, irr1:ir2, 0, loc.bottom]
    D1in[p0, irr1:ir2, -1, 0]  -= diff_D1u[p0, irr1:ir2, -1, 0]  / sqrt_det_h_half[p0, irr1:ir2, -1, loc.bottom]
    D1in[p0, irr1:ir2, 0, -1]  -= diff_D1u[p0, irr1:ir2, 0, -1]  / sqrt_det_h_half[p0, irr1:ir2, 0, loc.top] 
    D1in[p0, irr1:ir2, -1, -1] -= diff_D1u[p0, irr1:ir2, -1, -1] / sqrt_det_h_half[p0, irr1:ir2, -1, loc.top]

    D2in[p0, irr1:ir2, 0, 0]   -= diff_D2u[p0, irr1:ir2, 0, 0]   / sqrt_det_h_half[p0, irr1:ir2, 0, loc.bottom]
    D2in[p0, irr1:ir2, -1, 0]  -= diff_D2u[p0, irr1:ir2, -1, 0]  / sqrt_det_h_half[p0, irr1:ir2, -1, loc.bottom]
    D2in[p0, irr1:ir2, 0, -1]  -= diff_D2u[p0, irr1:ir2, 0, -1]  / sqrt_det_h_half[p0, irr1:ir2, 0, loc.top]
    D2in[p0, irr1:ir2, -1, -1] -= diff_D2u[p0, irr1:ir2, -1, -1] / sqrt_det_h_half[p0, irr1:ir2, -1, loc.top]


def penalty_edges_D(dtin, Drin, D1in, D2in, Brin, B1in, B2in, Drout, D1out, D2out):

    diff_Dru[:, :, :, :] = 0.0
    diff_D1u[:, :, :, :] = 0.0
    diff_D2u[:, :, :, :] = 0.0
    
    for i in range(n_zeros):
        p0, p1 = index_row[i], index_col[i]
        compute_penalty_D(p0, p1, dtin, Drin, D1in, D2in, Brin, B1in, B2in)

    for i in range(n_zeros):
        p0, p1 = index_row[i], index_col[i]
        interface_D(p0, p1, Drout, D1out, D2out)

    corners_D(patches, Drout, D1out, D2out)

########
# Apply penalty terms to B, H
########

def interface_B(p0, p1, Brin, B1in, B2in):

    ir1 = NG - 1
    ir2 = Nl0 + NG + 1

    irr1 = NG
    
    i0 =  1
    i1_half = Nxi_half - 1
    i1_int  = Nxi_int - 1

    top = topology[p0, p1]
    
    if (top == 'xx'):
        Brin[p0, irr1:ir2, -1, i0:i1_half] -= diff_Bru[p0, irr1:ir2, -1, i0:i1_half] / sqrt_det_h_half[p0, irr1:ir2, i0:i1_half, loc.right]
        Brin[p1, irr1:ir2, 0, i0:i1_half]  -= diff_Bru[p1, irr1:ir2, 0, i0:i1_half]  / sqrt_det_h_half[p1, irr1:ir2, i0:i1_half, loc.left]

        B2in[p0, ir1:ir2, -1, i0:i1_int] -= diff_B2u[p0, ir1:ir2, -1, i0:i1_int] / sqrt_det_h_int[p0, ir1:ir2, i0:i1_int, loc.right]
        B2in[p1, ir1:ir2, 0, i0:i1_int]  -= diff_B2u[p1, ir1:ir2, 0, i0:i1_int]  / sqrt_det_h_int[p1, ir1:ir2, i0:i1_int, loc.left]

    if (top == 'xy'):
        Brin[p0, irr1:ir2, -1, i0:i1_half] -= diff_Bru[p0, irr1:ir2, -1, i0:i1_half] / sqrt_det_h_half[p0, irr1:ir2, i0:i1_half, loc.right]
        Brin[p1, irr1:ir2, i0:i1_half, 0]  -= diff_Bru[p1, irr1:ir2, i0:i1_half, 0]  / sqrt_det_h_half[p1, irr1:ir2, i0:i1_half, loc.bottom]

        B2in[p0, ir1:ir2, -1, i0:i1_int] -= diff_B2u[p0, ir1:ir2, -1, i0:i1_int] / sqrt_det_h_int[p0, ir1:ir2, i0:i1_int, loc.right]
        B1in[p1, ir1:ir2, i0:i1_int, 0]  -= diff_B1u[p1, ir1:ir2, i0:i1_int, 0]  / sqrt_det_h_int[p1, ir1:ir2, i0:i1_int, loc.bottom]

    if (top == 'yy'):
        Brin[p0, irr1:ir2, i0:i1_half, -1] -= diff_Bru[p0, irr1:ir2, i0:i1_half, -1] / sqrt_det_h_half[p0, irr1:ir2, i0:i1_half, loc.top]
        Brin[p1, irr1:ir2, i0:i1_half, 0]  -= diff_Bru[p1, irr1:ir2, i0:i1_half, 0]  / sqrt_det_h_half[p1, irr1:ir2, i0:i1_half, loc.bottom]

        B1in[p0, ir1:ir2, i0:i1_int, -1] -= diff_B1u[p0, ir1:ir2, i0:i1_int, -1] / sqrt_det_h_int[p0, ir1:ir2, i0:i1_int, loc.top]
        B1in[p1, ir1:ir2, i0:i1_int, 0]  -= diff_B1u[p1, ir1:ir2, i0:i1_int, 0]  / sqrt_det_h_int[p1, ir1:ir2, i0:i1_int, loc.bottom]

    if (top == 'yx'):
        Brin[p0, irr1:ir2, i0:i1_half, -1] -= diff_Bru[p0, irr1:ir2, i0:i1_half, -1] / sqrt_det_h_half[p0, irr1:ir2, i0:i1_half, loc.top]
        Brin[p1, irr1:ir2, 0, i0:i1_half]  -= diff_Bru[p1, irr1:ir2, 0, i0:i1_half]  / sqrt_det_h_half[p1, irr1:ir2, i0:i1_half, loc.left]

        B1in[p0, ir1:ir2, i0:i1_int, -1] -= diff_B1u[p0, ir1:ir2, i0:i1_int, -1] / sqrt_det_h_int[p0, ir1:ir2, i0:i1_int, loc.top]
        B2in[p1, ir1:ir2, 0, i0:i1_int]  -= diff_B2u[p1, ir1:ir2, 0, i0:i1_int]  / sqrt_det_h_int[p1, ir1:ir2, i0:i1_int, loc.left]

def corners_B(p0, Brin, B1in, B2in):

    ir1 = NG - 1
    ir2 = Nl0 + NG + 1
    
    irr1 = NG

    Brin[p0, irr1:ir2, 0, 0]   -= diff_Bru[p0, irr1:ir2, 0, 0]    / sqrt_det_h_half[p0, irr1:ir2, 0, loc.bottom]
    Brin[p0, irr1:ir2, -1, 0]  -= diff_Bru[p0, irr1:ir2, -1, 0]   / sqrt_det_h_half[p0, irr1:ir2, -1, loc.bottom] 
    Brin[p0, irr1:ir2, 0, -1]  -= diff_Bru[p0, irr1:ir2, 0, -1]   / sqrt_det_h_half[p0, irr1:ir2, 0, loc.top]
    Brin[p0, irr1:ir2, -1, -1] -= diff_Bru[p0, irr1:ir2, -1, -1]  / sqrt_det_h_half[p0, irr1:ir2, -1, loc.top] 

    B1in[p0, ir1:ir2, 0, 0]   -= diff_B1u[p0, ir1:ir2, 0, 0]    / sqrt_det_h_int[p0, ir1:ir2, 0, loc.bottom]
    B1in[p0, ir1:ir2, -1, 0]  -= diff_B1u[p0, ir1:ir2, -1, 0]   / sqrt_det_h_int[p0, ir1:ir2, -1, loc.bottom] 
    B1in[p0, ir1:ir2, 0, -1]  -= diff_B1u[p0, ir1:ir2, 0, -1]   / sqrt_det_h_int[p0, ir1:ir2, 0, loc.top]
    B1in[p0, ir1:ir2, -1, -1] -= diff_B1u[p0, ir1:ir2, -1, -1]  / sqrt_det_h_int[p0, ir1:ir2, -1, loc.top] 

    B2in[p0, ir1:ir2, 0, 0]   -= diff_B2u[p0, ir1:ir2, 0, 0]    / sqrt_det_h_int[p0, ir1:ir2, 0, loc.bottom]
    B2in[p0, ir1:ir2, -1, 0]  -= diff_B2u[p0, ir1:ir2, -1, 0]   / sqrt_det_h_int[p0, ir1:ir2, -1, loc.bottom] 
    B2in[p0, ir1:ir2, 0, -1]  -= diff_B2u[p0, ir1:ir2, 0, -1]   / sqrt_det_h_int[p0, ir1:ir2, 0, loc.top]
    B2in[p0, ir1:ir2, -1, -1] -= diff_B2u[p0, ir1:ir2, -1, -1]  / sqrt_det_h_int[p0, ir1:ir2, -1, loc.top] 

def penalty_edges_B(dtin, Drin, D1in, D2in, Brin, B1in, B2in, Brout, B1out, B2out):

    diff_Bru[:, :, :, :] = 0.0
    diff_B1u[:, :, :, :] = 0.0
    diff_B2u[:, :, :, :] = 0.0
    
    for i in range(n_zeros):
        p0, p1 = index_row[i], index_col[i]
        compute_penalty_B(p0, p1, dtin, Drin, D1in, D2in, Brin, B1in, B2in)

    for i in range(n_zeros):
        p0, p1 = index_row[i], index_col[i]
        interface_B(p0, p1, Brout, B1out, B2out)

    corners_B(patches, Brout, B1out, B2out)

########
# Absorbing boundary conditions at r_max
########

i_abs = 6 # Thickness of absorbing layer in number of cells

r_abs_out = r[Nl - i_abs]
kappa_out = 5.0 

delta = ((r - r_abs_out) / (r_max - r_abs_out)) * N.heaviside(r - r_abs_out, 0.0)
sigma_out = N.exp(- kappa_out * delta**3)

delta = ((r_yee - r_abs_out) / (r_max - r_abs_out)) * N.heaviside(r_yee - r_abs_out, 0.0)
sigma_yee_out = N.exp(- kappa_out * delta**3)

r_abs_in = 0.92 * rh
# r_abs_in = r[i_abs]
kappa_in = 5.0 

delta = ((r_abs_in - r) / (r_abs_in - r_min)) * N.heaviside(r_abs_in - r, 0.0)
sigma_in = N.exp(- kappa_in * delta**3)

delta = ((r_abs_in - r_yee) / (r_abs_in - r_min)) * N.heaviside(r_abs_in - r_yee, 0.0)
sigma_yee_in = N.exp(- kappa_in * delta**3)

def BC_D_absorb(patch, Drin, D1in, D2in):
    # Drin[patch, :, :, :] *= sigma_yee_out[:, None, None]
    # D1in[patch, :, :, :] *= sigma_out[:, None, None]
    # D2in[patch, :, :, :] *= sigma_out[:, None, None]
    Drin[patch, :, :, :] = INDr[patch, :, :, :] + (Drin[patch, :, :, :] - INDr[patch, :, :, :]) * sigma_yee_out[:, None, None]
    D1in[patch, :, :, :] = IND1[patch, :, :, :] + (D1in[patch, :, :, :] - IND1[patch, :, :, :]) * sigma_out[:, None, None]
    D2in[patch, :, :, :] = IND2[patch, :, :, :] + (D2in[patch, :, :, :] - IND2[patch, :, :, :]) * sigma_out[:, None, None]
    return

def BC_D_absorb_in(patch, Drin, D1in, D2in):
    Drin[patch, :, :, :] *= sigma_yee_in[:, None, None]
    D1in[patch, :, :, :] *= sigma_in[:, None, None]
    D2in[patch, :, :, :] *= sigma_in[:, None, None]
    return

def BC_B_absorb(patch, Brin, B1in, B2in):
    Brin[patch, :, :, :] = INBr[patch, :, :, :] + (Brin[patch, :, :, :] - INBr[patch, :, :, :]) * sigma_out[:, None, None]
    B1in[patch, :, :, :] = INB1[patch, :, :, :] + (B1in[patch, :, :, :] - INB1[patch, :, :, :]) * sigma_yee_out[:, None, None]
    B2in[patch, :, :, :] = INB2[patch, :, :, :] + (B2in[patch, :, :, :] - INB2[patch, :, :, :]) * sigma_yee_out[:, None, None]
    return

def BC_B_absorb_in(patch, Brin, B1in, B2in):
    # Brin[patch, :, :, :] *= sigma_in[:, None, None]
    # B1in[patch, :, :, :] *= sigma_yee_in[:, None, None]
    # B2in[patch, :, :, :] *= sigma_yee_in[:, None, None]
    return

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

# Divide by r0 if strecthed metric!
def func_Br(r0, th0, ph0):
    # return 2.0 * B0 * (N.cos(th0) * N.cos(tilt) + N.sin(th0) * N.sin(ph0) * N.sin(tilt)) / r0**3
    if (N.sin(th0)<1e-10):
        return B0 * r0 * r0 * N.cos(th0) / (r0 * r0 + a * a) / r0
    else:
        return B0 * r0 * r0 * N.cos(th0) * N.sin(th0) / sqrtdeth(r0, th0, ph0, a) / r0
    # return B0 * N.sin(th0) / sqrtdeth(r0, th0, ph0, a)

def func_Bth(r0, th0, ph0):
#    return B0 * (N.cos(tilt) * N.sin(th0) - N.cos(th0) * N.sin(ph0) * N.sin(tilt)) / r0**4
    if (N.sin(th0)<1e-10):
        return 0.0
    else:
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

                r0 = r[:]
                th0, ph0 = fcoord(xi_half[i], eta_half[j])
                BrTMP = func_Br(r0, th0, ph0)

                Bru[patch, :, i, j] = BrTMP
                INBr[patch,:, i, j] = BrTMP
                    
        for i in range(Nxi_int):
            for j in range(Neta_half):

                    r0 = r_yee[:]
                    th0, ph0 = fcoord(xi_int[i], eta_half[j])
                    BtTMP = func_Bth(r0, th0, ph0)
                    BpTMP = func_Bph(r0, th0, ph0)
                    BCStmp = fvec(th0, ph0, BtTMP, BpTMP)

                    B1u[patch, :, i, j]  = BCStmp[0]
                    INB1[patch, :, i, j] = BCStmp[0]
                    D2u[patch, :, i, j] = 0.0
                    IND2[patch, :, i, j] = 0.0

        for i in range(Nxi_half):
            for j in range(Neta_int):

                    r0 = r_yee[:]
                    th0, ph0 = fcoord(xi_half[i], eta_int[j])
                    BtTMP = func_Bth(r0, th0, ph0)
                    BpTMP = func_Bph(r0, th0, ph0)
                    BCStmp = fvec(th0, ph0, BtTMP, BpTMP)

                    B2u[patch, :, i, j]  = BCStmp[1]
                    INB2[patch, :, i, j] = BCStmp[1]
                    D1u[patch, :, i, j]  = 0.0
                    IND1[patch, :, i, j]  = 0.0

        for i in range(Nxi_int):
            for j in range(Neta_int):

                r0 = r_yee[:]
                th0, ph0 = fcoord(xi_int[i], eta_int[j])
                BtTMP = func_Bth(r0, th0, ph0)
                BpTMP = func_Bph(r0, th0, ph0)
                BCStmp = fvec(th0, ph0, BtTMP, BpTMP)

                Dru[patch, :, i, j]  = 0.0
                INDr[patch, :, i, j]  = 0.0

# def InitialData():

#     for patch in range(n_patches):

#         fvec = (globals()["vec_sph_to_" + sphere[patch]])
#         fcoord = (globals()["coord_" + sphere[patch] + "_to_sph"])

#         for i in range(Nxi_half):
#             for j in range(Neta_half):

#                 r0 = r[:]
#                 th0, ph0 = fcoord(xi_half[i], eta_half[j])

#                 Bru[patch, :, i, j] = 0.0
#                 INBr[patch,:, i, j] = 0.0
                    
#         for i in range(Nxi_int):
#             for j in range(Neta_half):

#                     r0 = r_yee[:]
#                     th0, ph0 = fcoord(xi_int[i], eta_half[j])
#                     BtTMP = func_Bth(r0, th0, ph0)
#                     BpTMP = func_Bph(r0, th0, ph0)
#                     BCStmp = fvec(th0, ph0, BtTMP, BpTMP)

#                     B1u[patch, :, i, j]  = 0.0
#                     INB1[patch, :, i, j] = 0.0
#                     D2u[patch, :, i, j]  = BCStmp[1]
#                     IND2[patch, :, i, j] = BCStmp[1]

#         for i in range(Nxi_half):
#             for j in range(Neta_int):

#                     r0 = r_yee[:]
#                     th0, ph0 = fcoord(xi_half[i], eta_int[j])
#                     BtTMP = func_Bth(r0, th0, ph0)
#                     BpTMP = func_Bph(r0, th0, ph0)
#                     BCStmp = fvec(th0, ph0, BtTMP, BpTMP)

#                     B2u[patch, :, i, j]  = 0.0 
#                     INB2[patch, :, i, j] = 0.0
#                     D1u[patch, :, i, j]  = BCStmp[0]
#                     IND1[patch, :, i, j] = BCStmp[0]

#         for i in range(Nxi_int):
#             for j in range(Neta_int):

#                 r0 = r_yee[:]
#                 th0, ph0 = fcoord(xi_int[i], eta_int[j])
#                 DrTMP = func_Br(r0, th0, ph0)

#                 Dru[patch, :, i, j]  = DrTMP
#                 INDr[patch, :, i, j] = DrTMP

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

idump = 0 # Number of iterations

Nt = 15000 # Number of iterations
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
        plot_fields_unfolded_Br(idump, 2.0, 10)
        plot_fields_unfolded_B1(idump, 2.0, 10)
        plot_fields_unfolded_B2(idump, 2.0, 10)
        plot_fields_unfolded_D2(idump, 2.0, 10)
        WriteAllFieldsHDF5(idump)
        idump += 1

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
    BC_B_absorb_in(patches, Bru1, B1u1, B2u1)

    contra_to_cov_D(patches, Dru, D1u, D2u)
    BC_Dd(patches, Drd, D1d, D2d)
    compute_E_aux(patches, Drd, D1d, D2d, Bru1, B1u1, B2u1)
    BC_Ed(patches, Erd, E1d, E2d)

    # ### USeful??
    # BC_Du(patches, Erd, E1d, E2d)
    # BC_D_absorb(patches, Erd, E1d, E2d)
    # BC_D_absorb_in(patches, Erd, E1d, E2d)

    Bru0[:, :, :, :] = Bru[:, :, :, :]
    B1u0[:, :, :, :] = B1u[:, :, :, :]
    B2u0[:, :, :, :] = B2u[:, :, :, :]

    compute_diff_E(patches)
    push_B(patches, Bru, B1u, B2u, dt)

    # # Penalty terms
    penalty_edges_B(dt, Erd, E1d, E2d, Bru1, B1u1, B2u1, Bru, B1u, B2u)
    # # penalty_edges_B(dt, Erd, E1d, E2d, Bru0, B1u0, B2u0, Bru, B1u, B2u)
    # # penalty_edges_B(dt, Drd, D1d, D2d, Bru, B1u, B2u, Bru, B1u, B2u)

    BC_Bu(patches, Bru, B1u, B2u)
    BC_B_absorb(patches, Bru, B1u, B2u)
    BC_B_absorb_in(patches, Bru, B1u, B2u)

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
    BC_D_absorb_in(patches, Dru1, D1u1, D2u1)

    contra_to_cov_B(patches, Bru, B1u, B2u)
    BC_Bd(patches, Brd, B1d, B2d)
    compute_H_aux(patches, Dru1, D1u1, D2u1, Brd, B1d, B2d)
    BC_Hd(patches, Hrd, H1d, H2d)
    
    # ### USeful??
    # BC_Bu(patches, Hrd, H1d, H2d)
    # BC_B_absorb(patches, Hrd, H1d, H2d)
    # BC_B_absorb_in(patches, Hrd, H1d, H2d)

    Dru0[:, :, :, :] = Dru[:, :, :, :]
    D1u0[:, :, :, :] = D1u[:, :, :, :]
    D2u0[:, :, :, :] = D2u[:, :, :, :]

    compute_diff_H(patches)
    push_D(patches, Dru, D1u, D2u, dt)

    # Penalty terms
    penalty_edges_D(dt, Dru1, D1u1, D2u1, Hrd, H1d, H2d, Dru, D1u, D2u)
    # penalty_edges_D(dt, Dru, D1u, D2u, Hrd, H1d, H2d, Dru, D1u, D2u)
    # penalty_edges_D(dt, Dru, D1u, D2u, Brd, B1d, B2d, Dru, D1u, D2u)

    BC_Du(patches, Dru, D1u, D2u)
    BC_D_absorb(patches, Dru, D1u, D2u)
    BC_D_absorb_in(patches, Dru, D1u, D2u)
