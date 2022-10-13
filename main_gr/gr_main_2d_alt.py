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
from gr_metric import *

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
cfl = 0.5
Nxi  = 32
Neta = 32

# Spin parameter
a = 0.95
rh = 1.0 + N.sqrt(1.0 - a * a)
r0 = 1.0 * rh

Nxi_int = Nxi + 1 # Number of integer points
Nxi_half = Nxi + 2 # Number of half-step points
Neta_int = Neta + 1 # Number of integer points
Neta_half = Neta + 2 # NUmber of half-step points

xi_min, xi_max = - N.pi / 4.0, N.pi / 4.0
eta_min, eta_max = - N.pi / 4.0, N.pi / 4.0

dxi = (xi_max - xi_min) / Nxi
deta = (eta_max - eta_min) / Neta

# Define grids
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

yEr_grid, xEr_grid = N.meshgrid(eta_int, xi_int)
yBr_grid, xBr_grid = N.meshgrid(eta_half, xi_half)
yE1_grid, xE1_grid = N.meshgrid(eta_int, xi_half)
yE2_grid, xE2_grid = N.meshgrid(eta_half, xi_int)

# Physical fields
Bru = N.zeros((n_patches, Nxi_half, Neta_half))
B1u = N.zeros((n_patches, Nxi_int, Neta_half))
B2u = N.zeros((n_patches, Nxi_half,  Neta_int))
Brd = N.zeros((n_patches, Nxi_half, Neta_half))
B1d = N.zeros((n_patches, Nxi_int, Neta_half))
B2d = N.zeros((n_patches, Nxi_half,  Neta_int))

Dru = N.zeros((n_patches, Nxi_int, Neta_int))
D1u = N.zeros((n_patches, Nxi_half, Neta_int))
D2u = N.zeros((n_patches, Nxi_int,  Neta_half))
Drd = N.zeros((n_patches, Nxi_int, Neta_int))
D1d = N.zeros((n_patches, Nxi_half, Neta_int))
D2d = N.zeros((n_patches, Nxi_int,  Neta_half))

# Shifted by one time step
Bru0 = N.zeros((n_patches, Nxi_half, Neta_half))
B1u0 = N.zeros((n_patches, Nxi_int, Neta_half))
B2u0 = N.zeros((n_patches, Nxi_half,  Neta_int))
Bru1 = N.zeros((n_patches, Nxi_half, Neta_half))
B1u1 = N.zeros((n_patches, Nxi_int, Neta_half))
B2u1 = N.zeros((n_patches, Nxi_half,  Neta_int))

Dru0 = N.zeros((n_patches, Nxi_int, Neta_int))
D1u0 = N.zeros((n_patches, Nxi_half, Neta_int))
D2u0 = N.zeros((n_patches, Nxi_int,  Neta_half))
Dru1 = N.zeros((n_patches, Nxi_int, Neta_int))
D1u1 = N.zeros((n_patches, Nxi_half, Neta_int))
D2u1 = N.zeros((n_patches, Nxi_int,  Neta_half))

# Auxiliary fields and gradients
Erd = N.zeros((n_patches, Nxi_int, Neta_int))
E1d = N.zeros((n_patches, Nxi_half, Neta_int))
E2d = N.zeros((n_patches, Nxi_int,  Neta_half))
Eru = N.zeros((n_patches, Nxi_int, Neta_int))
E1u = N.zeros((n_patches, Nxi_half, Neta_int))
E2u = N.zeros((n_patches, Nxi_int,  Neta_half))

Hrd = N.zeros((n_patches, Nxi_half, Neta_half))
H1d = N.zeros((n_patches, Nxi_int, Neta_half))
H2d = N.zeros((n_patches, Nxi_half,  Neta_int))
Hru = N.zeros((n_patches, Nxi_half, Neta_half))
H1u = N.zeros((n_patches, Nxi_int, Neta_half))
H2u = N.zeros((n_patches, Nxi_half,  Neta_int))

dE1d2 = N.zeros((n_patches, Nxi_half, Neta_half))
dE2d1 = N.zeros((n_patches, Nxi_half, Neta_half))
dErd1 = N.zeros((n_patches, Nxi_half,  Neta_int))
dErd2 = N.zeros((n_patches, Nxi_int, Neta_half))

dHrd1 = N.zeros((n_patches, Nxi_int,  Neta_half))
dHrd2 = N.zeros((n_patches, Nxi_half, Neta_int))
dH1d2 = N.zeros((n_patches, Nxi_int, Neta_int))
dH2d1 = N.zeros((n_patches, Nxi_int, Neta_int))

# Interface terms
diff_Bru = N.zeros((n_patches, Nxi_half, Neta_half))
diff_B1u = N.zeros((n_patches, Nxi_int, Neta_half))
diff_B2u = N.zeros((n_patches, Nxi_half, Neta_int))
diff_Dru = N.zeros((n_patches, Nxi_int, Neta_int))
diff_D1u = N.zeros((n_patches, Nxi_half, Neta_int))
diff_D2u = N.zeros((n_patches, Nxi_int, Neta_half))

# Initial magnetic field
INBr = N.zeros((n_patches, Nxi_half, Neta_half))
INB1 = N.zeros((n_patches, Nxi_int, Neta_half))
INB2 = N.zeros((n_patches, Nxi_half,  Neta_int))

Jz = N.zeros_like(Dru)
Jz[Sphere.D, :, :] = 0.0 * N.exp(- (xEr_grid**2 + yEr_grid**2) / 0.1**2)

########
# Dump HDF5 output
########

def WriteFieldHDF5(it, field):

    outvec = (globals()[field])
    h5f = h5py.File(outdir + field + '_' + str(it).rjust(5, '0') + '.h5', 'w')

    for patch in range(n_patches):
        h5f.create_dataset(field + str(patch), data=outvec[patch, :, :])

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

    h5f.create_dataset('xi_int', data = xi_int)
    h5f.create_dataset('eta_int', data = eta_int)
    h5f.create_dataset('xi_half', data = xi_half)
    h5f.create_dataset('eta_half', data = eta_half)
    
    h5f.close()

########
# Define metric tensor
########

# 4 positions in the 2D Yee elementary square cell
hrrd = N.empty((n_patches, Nxi_int, Neta_int, 4))
hr1d = N.empty((n_patches, Nxi_int, Neta_int, 4))
hr2d = N.empty((n_patches, Nxi_int, Neta_int, 4))
h11d = N.empty((n_patches, Nxi_int, Neta_int, 4))
h12d = N.empty((n_patches, Nxi_int, Neta_int, 4))
h22d = N.empty((n_patches, Nxi_int, Neta_int, 4))
hrru = N.empty((n_patches, Nxi_int, Neta_int, 4))
hr1u = N.empty((n_patches, Nxi_int, Neta_int, 4))
hr2u = N.empty((n_patches, Nxi_int, Neta_int, 4))
h11u = N.empty((n_patches, Nxi_int, Neta_int, 4))
h12u = N.empty((n_patches, Nxi_int, Neta_int, 4))
h22u = N.empty((n_patches, Nxi_int, Neta_int, 4))
alpha= N.empty((n_patches, Nxi_int, Neta_int, 4))
beta = N.empty((n_patches, Nxi_int, Neta_int, 4))
sqrt_det_h = N.empty((n_patches, Nxi_int, Neta_int, 4))

# 4 sides of a patch
sqrt_det_h_half = N.empty((n_patches, Nxi_half, 4))
h12d_half = N.empty((n_patches, Nxi_half, 4))
h11d_half = N.empty((n_patches, Nxi_half, 4))
h22d_half = N.empty((n_patches, Nxi_half, 4))
hrrd_half = N.empty((n_patches, Nxi_half, 4))
hr1d_half = N.empty((n_patches, Nxi_half, 4))
hr2d_half = N.empty((n_patches, Nxi_half, 4))
h12u_half = N.empty((n_patches, Nxi_half, 4))
h11u_half = N.empty((n_patches, Nxi_half, 4))
h22u_half = N.empty((n_patches, Nxi_half, 4))
hrru_half = N.empty((n_patches, Nxi_half, 4))
hr1u_half = N.empty((n_patches, Nxi_half, 4))
hr2u_half = N.empty((n_patches, Nxi_half, 4))
alpha_half = N.empty((n_patches, Nxi_half, 4))

sqrt_det_h_int = N.empty((n_patches, Nxi_int, 4))
h12d_int = N.empty((n_patches, Nxi_int, 4))
h11d_int = N.empty((n_patches, Nxi_int, 4))
h22d_int = N.empty((n_patches, Nxi_int, 4))
hrrd_int = N.empty((n_patches, Nxi_int, 4))
hr1d_int = N.empty((n_patches, Nxi_int, 4))
hr2d_int = N.empty((n_patches, Nxi_int, 4))
h12u_int = N.empty((n_patches, Nxi_int, 4))
h11u_int = N.empty((n_patches, Nxi_int, 4))
h22u_int = N.empty((n_patches, Nxi_int, 4))
hrru_int = N.empty((n_patches, Nxi_int, 4))
hr1u_int = N.empty((n_patches, Nxi_int, 4))
hr2u_int = N.empty((n_patches, Nxi_int, 4))
alpha_int = N.empty((n_patches, Nxi_int, 4))

for p in range(n_patches):
    for i in range(Nxi_int):
        print(i, p)
        for j in range(Neta_int):

            # 0 at (i, j)
            xi0 = xi_int[i]
            eta0 = eta_int[j]
            h11d[p, i, j, 0] = g11d(p, r0, xi0, eta0, a)
            h22d[p, i, j, 0] = g22d(p, r0, xi0, eta0, a)
            h12d[p, i, j, 0] = g12d(p, r0, xi0, eta0, a)
            hrrd[p, i, j, 0] = grrd(p, r0, xi0, eta0, a)
            hr1d[p, i, j, 0] = gr1d(p, r0, xi0, eta0, a)
            hr2d[p, i, j, 0] = gr2d(p, r0, xi0, eta0, a)
            alpha[p, i, j,0] = alphas(p, r0, xi0, eta0, a)
            beta[p, i, j, 0] = betaru(p, r0, xi0, eta0, a)

            metric = N.array([[hrrd[p, i, j, 0], hr1d[p, i, j, 0], hr2d[p, i, j, 0]], \
                              [hr1d[p, i, j, 0], h11d[p, i, j, 0], h12d[p, i, j, 0]], \
                              [hr2d[p, i, j, 0], h12d[p, i, j, 0], h22d[p, i, j, 0]]])

            sqrt_det_h[p, i, j, 0] = N.sqrt(N.linalg.det(metric))
            inv_metric = N.linalg.inv(metric)
            hrru[p, i, j, 0] = inv_metric[0, 0]
            hr1u[p, i, j, 0] = inv_metric[0, 1]
            hr2u[p, i, j, 0] = inv_metric[0, 2]
            h11u[p, i, j, 0] = inv_metric[1, 1]
            h12u[p, i, j, 0] = inv_metric[1, 2]
            h22u[p, i, j, 0] = inv_metric[2, 2]
                
            # 1 at (i + 1/2, j)
            xi0  = xi_int[i] + 0.5 * dxi
            eta0 = eta_int[j]
            h11d[p, i, j, 1] = g11d(p, r0, xi0, eta0, a)
            h22d[p, i, j, 1] = g22d(p, r0, xi0, eta0, a)
            h12d[p, i, j, 1] = g12d(p, r0, xi0, eta0, a)
            hrrd[p, i, j, 1] = grrd(p, r0, xi0, eta0, a)
            hr1d[p, i, j, 1] = gr1d(p, r0, xi0, eta0, a)
            hr2d[p, i, j, 1] = gr2d(p, r0, xi0, eta0, a)
            alpha[p, i, j,1] =  alphas(p, r0, xi0, eta0, a)
            beta[p, i, j, 1] =  betaru(p, r0, xi0, eta0, a)

            metric = N.array([[hrrd[p, i, j, 1], hr1d[p, i, j, 1], hr2d[p, i, j, 1]], \
                              [hr1d[p, i, j, 1], h11d[p, i, j, 1], h12d[p, i, j, 1]], \
                              [hr2d[p, i, j, 1], h12d[p, i, j, 1], h22d[p, i, j, 1]]])

            sqrt_det_h[p, i, j, 1] = N.sqrt(N.linalg.det(metric))
            inv_metric = N.linalg.inv(metric)
            hrru[p, i, j, 1] = inv_metric[0, 0]
            hr1u[p, i, j, 1] = inv_metric[0, 1]
            hr2u[p, i, j, 1] = inv_metric[0, 2]
            h11u[p, i, j, 1] = inv_metric[1, 1]
            h12u[p, i, j, 1] = inv_metric[1, 2]
            h22u[p, i, j, 1] = inv_metric[2, 2]

            # 2 at (i, j + 1/2)
            xi0  = xi_int[i]
            eta0 = eta_int[j] + 0.5 * deta
            h11d[p, i, j, 2] = g11d(p, r0, xi0, eta0, a)
            h22d[p, i, j, 2] = g22d(p, r0, xi0, eta0, a)
            h12d[p, i, j, 2] = g12d(p, r0, xi0, eta0, a)
            hrrd[p, i, j, 2] = grrd(p, r0, xi0, eta0, a)
            hr1d[p, i, j, 2] = gr1d(p, r0, xi0, eta0, a)
            hr2d[p, i, j, 2] = gr2d(p, r0, xi0, eta0, a)
            alpha[p, i, j,2] =  alphas(p, r0, xi0, eta0, a)
            beta[p, i, j, 2] =  betaru(p, r0, xi0, eta0, a)

            metric = N.array([[hrrd[p, i, j, 2], hr1d[p, i, j, 2], hr2d[p, i, j, 2]], \
                              [hr1d[p, i, j, 2], h11d[p, i, j, 2], h12d[p, i, j, 2]], \
                              [hr2d[p, i, j, 2], h12d[p, i, j, 2], h22d[p, i, j, 2]]])

            sqrt_det_h[p, i, j, 2] = N.sqrt(N.linalg.det(metric))
            inv_metric = N.linalg.inv(metric)
            hrru[p, i, j, 2] = inv_metric[0, 0]
            hr1u[p, i, j, 2] = inv_metric[0, 1]
            hr2u[p, i, j, 2] = inv_metric[0, 2]
            h11u[p, i, j, 2] = inv_metric[1, 1]
            h12u[p, i, j, 2] = inv_metric[1, 2]
            h22u[p, i, j, 2] = inv_metric[2, 2]

            # 3 at (i + 1/2, j + 1/2)
            xi0  = xi_int[i] + 0.5 * dxi
            eta0 = eta_int[j] + 0.5 * deta
            h11d[p, i, j, 3] = g11d(p, r0, xi0, eta0, a)
            h22d[p, i, j, 3] = g22d(p, r0, xi0, eta0, a)
            h12d[p, i, j, 3] = g12d(p, r0, xi0, eta0, a)
            hrrd[p, i, j, 3] = grrd(p, r0, xi0, eta0, a)
            hr1d[p, i, j, 3] = gr1d(p, r0, xi0, eta0, a)
            hr2d[p, i, j, 3] = gr2d(p, r0, xi0, eta0, a)
            alpha[p, i, j,3] =  alphas(p, r0, xi0, eta0, a)
            beta[p, i, j, 3] =  betaru(p, r0, xi0, eta0, a)

            metric = N.array([[hrrd[p, i, j, 3], hr1d[p, i, j, 3], hr2d[p, i, j, 3]], \
                              [hr1d[p, i, j, 3], h11d[p, i, j, 3], h12d[p, i, j, 3]], \
                              [hr2d[p, i, j, 3], h12d[p, i, j, 3], h22d[p, i, j, 3]]])

            sqrt_det_h[p, i, j, 3] = N.sqrt(N.linalg.det(metric))
            inv_metric = N.linalg.inv(metric)
            hrru[p, i, j, 3] = inv_metric[0, 0]
            hr1u[p, i, j, 3] = inv_metric[0, 1]
            hr2u[p, i, j, 3] = inv_metric[0, 2]
            h11u[p, i, j, 3] = inv_metric[1, 1]
            h12u[p, i, j, 3] = inv_metric[1, 2]
            h22u[p, i, j, 3] = inv_metric[2, 2]

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
        
        # Left edge
        xi0 = xi_half[0]
        eta0 = eta_half[i]
        h11d_half[p, i, loc.left] = g11d(p, r0, xi0, eta0, a)
        h22d_half[p, i, loc.left] = g22d(p, r0, xi0, eta0, a)
        h12d_half[p, i, loc.left] = g12d(p, r0, xi0, eta0, a)
        hrrd_half[p, i, loc.left] = grrd(p, r0, xi0, eta0, a)
        hr1d_half[p, i, loc.left] = gr1d(p, r0, xi0, eta0, a)
        hr2d_half[p, i, loc.left] = gr2d(p, r0, xi0, eta0, a)
        alpha_half[p, i, loc.left] = alphas(p, r0, xi0, eta0, a)

        metric = N.array([[hrrd_half[p, i, loc.left], hr1d_half[p, i, loc.left], hr2d_half[p, i, loc.left]], \
                          [hr1d_half[p, i, loc.left], h11d_half[p, i, loc.left], h12d_half[p, i, loc.left]], \
                          [hr2d_half[p, i, loc.left], h12d_half[p, i, loc.left], h22d_half[p, i, loc.left]]])

        inv_metric = N.linalg.inv(metric)
        hrru_half[p, i, loc.left] = inv_metric[0, 0]
        hr1u_half[p, i, loc.left] = inv_metric[0, 1]
        hr2u_half[p, i, loc.left] = inv_metric[0, 2]
        h11u_half[p, i, loc.left] = inv_metric[1, 1]
        h12u_half[p, i, loc.left] = inv_metric[1, 2]
        h22u_half[p, i, loc.left] = inv_metric[2, 2]
        sqrt_det_h_half[p, i, loc.left] = N.sqrt(N.linalg.det(metric))

        # Bottom edge
        xi0 = xi_half[i]
        eta0 = eta_half[0]
        h11d_half[p, i, loc.bottom] = g11d(p, r0, xi0, eta0, a)
        h22d_half[p, i, loc.bottom] = g22d(p, r0, xi0, eta0, a)
        h12d_half[p, i, loc.bottom] = g12d(p, r0, xi0, eta0, a)
        hrrd_half[p, i, loc.bottom] = grrd(p, r0, xi0, eta0, a)
        hr1d_half[p, i, loc.bottom] = gr1d(p, r0, xi0, eta0, a)
        hr2d_half[p, i, loc.bottom] = gr2d(p, r0, xi0, eta0, a)
        alpha_half[p, i, loc.bottom] = alphas(p, r0, xi0, eta0, a)

        metric = N.array([[hrrd_half[p, i, loc.bottom], hr1d_half[p, i, loc.bottom], hr2d_half[p, i, loc.bottom]], \
                          [hr1d_half[p, i, loc.bottom], h11d_half[p, i, loc.bottom], h12d_half[p, i, loc.bottom]], \
                          [hr2d_half[p, i, loc.bottom], h12d_half[p, i, loc.bottom], h22d_half[p, i, loc.bottom]]])

        inv_metric = N.linalg.inv(metric)
        hrru_half[p, i, loc.bottom] = inv_metric[0, 0]
        hr1u_half[p, i, loc.bottom] = inv_metric[0, 1]
        hr2u_half[p, i, loc.bottom] = inv_metric[0, 2]
        h11u_half[p, i, loc.bottom] = inv_metric[1, 1]
        h12u_half[p, i, loc.bottom] = inv_metric[1, 2]
        h22u_half[p, i, loc.bottom] = inv_metric[2, 2]
        sqrt_det_h_half[p, i, loc.bottom] = N.sqrt(N.linalg.det(metric))

        # Right edge
        xi0 = xi_half[-1]
        eta0 = eta_half[i]
        h11d_half[p, i, loc.right] = g11d(p, r0, xi0, eta0, a)
        h22d_half[p, i, loc.right] = g22d(p, r0, xi0, eta0, a)
        h12d_half[p, i, loc.right] = g12d(p, r0, xi0, eta0, a)
        hrrd_half[p, i, loc.right] = grrd(p, r0, xi0, eta0, a)
        hr1d_half[p, i, loc.right] = gr1d(p, r0, xi0, eta0, a)
        hr2d_half[p, i, loc.right] = gr2d(p, r0, xi0, eta0, a)
        alpha_half[p, i, loc.right] = alphas(p, r0, xi0, eta0, a)

        metric = N.array([[hrrd_half[p, i, loc.right], hr1d_half[p, i, loc.right], hr2d_half[p, i, loc.right]], \
                          [hr1d_half[p, i, loc.right], h11d_half[p, i, loc.right], h12d_half[p, i, loc.right]], \
                          [hr2d_half[p, i, loc.right], h12d_half[p, i, loc.right], h22d_half[p, i, loc.right]]])

        inv_metric = N.linalg.inv(metric)
        hrru_half[p, i, loc.right] = inv_metric[0, 0]
        hr1u_half[p, i, loc.right] = inv_metric[0, 1]
        hr2u_half[p, i, loc.right] = inv_metric[0, 2]
        h11u_half[p, i, loc.right] = inv_metric[1, 1]
        h12u_half[p, i, loc.right] = inv_metric[1, 2]
        h22u_half[p, i, loc.right] = inv_metric[2, 2]
        sqrt_det_h_half[p, i, loc.right] = N.sqrt(N.linalg.det(metric))

        # Top edge
        xi0 = xi_half[i]
        eta0 = eta_half[-1]
        h11d_half[p, i, loc.top] = g11d(p, r0, xi0, eta0, a)
        h22d_half[p, i, loc.top] = g22d(p, r0, xi0, eta0, a)
        h12d_half[p, i, loc.top] = g12d(p, r0, xi0, eta0, a)
        hrrd_half[p, i, loc.top] = grrd(p, r0, xi0, eta0, a)
        hr1d_half[p, i, loc.top] = gr1d(p, r0, xi0, eta0, a)
        hr2d_half[p, i, loc.top] = gr2d(p, r0, xi0, eta0, a)
        alpha_half[p, i, loc.top] = alphas(p, r0, xi0, eta0, a)

        metric = N.array([[hrrd_half[p, i, loc.top], hr1d_half[p, i, loc.top], hr2d_half[p, i, loc.top]], \
                          [hr1d_half[p, i, loc.top], h11d_half[p, i, loc.top], h12d_half[p, i, loc.top]], \
                          [hr2d_half[p, i, loc.top], h12d_half[p, i, loc.top], h22d_half[p, i, loc.top]]])

        inv_metric = N.linalg.inv(metric)
        hrru_half[p, i, loc.top] = inv_metric[0, 0]
        hr1u_half[p, i, loc.top] = inv_metric[0, 1]
        hr2u_half[p, i, loc.top] = inv_metric[0, 2]
        h11u_half[p, i, loc.top] = inv_metric[1, 1]
        h12u_half[p, i, loc.top] = inv_metric[1, 2]
        h22u_half[p, i, loc.top] = inv_metric[2, 2]
        sqrt_det_h_half[p, i, loc.top] = N.sqrt(N.linalg.det(metric))

    # On int grid
    for i in range(Nxi_int):
        
        # Left edge
        xi0 = xi_int[0]
        eta0 = eta_int[i]
        h11d_int[p, i, loc.left] = g11d(p, r0, xi0, eta0, a)
        h22d_int[p, i, loc.left] = g22d(p, r0, xi0, eta0, a)
        h12d_int[p, i, loc.left] = g12d(p, r0, xi0, eta0, a)
        hrrd_int[p, i, loc.left] = grrd(p, r0, xi0, eta0, a)
        hr1d_int[p, i, loc.left] = gr1d(p, r0, xi0, eta0, a)
        hr2d_int[p, i, loc.left] = gr2d(p, r0, xi0, eta0, a)
        alpha_int[p, i, loc.left] = alphas(p, r0, xi0, eta0, a)

        metric = N.array([[hrrd_int[p, i, loc.left], hr1d_int[p, i, loc.left], hr2d_int[p, i, loc.left]], \
                          [hr1d_int[p, i, loc.left], h11d_int[p, i, loc.left], h12d_int[p, i, loc.left]], \
                          [hr2d_int[p, i, loc.left], h12d_int[p, i, loc.left], h22d_int[p, i, loc.left]]])
        
        inv_metric = N.linalg.inv(metric)
        hrru_int[p, i, loc.left] = inv_metric[0, 0]
        hr1u_int[p, i, loc.left] = inv_metric[0, 1]
        hr2u_int[p, i, loc.left] = inv_metric[0, 2]
        h11u_int[p, i, loc.left] = inv_metric[1, 1]
        h12u_int[p, i, loc.left] = inv_metric[1, 2]
        h22u_int[p, i, loc.left] = inv_metric[2, 2]
        sqrt_det_h_int[p, i, loc.left] = N.sqrt(N.linalg.det(metric))

        # Bottom edge
        xi0 = xi_int[i]
        eta0 = eta_int[0]
        h11d_int[p, i, loc.bottom] = g11d(p, r0, xi0, eta0, a)
        h22d_int[p, i, loc.bottom] = g22d(p, r0, xi0, eta0, a)
        h12d_int[p, i, loc.bottom] = g12d(p, r0, xi0, eta0, a)
        hrrd_int[p, i, loc.bottom] = grrd(p, r0, xi0, eta0, a)
        hr1d_int[p, i, loc.bottom] = gr1d(p, r0, xi0, eta0, a)
        hr2d_int[p, i, loc.bottom] = gr2d(p, r0, xi0, eta0, a)
        alpha_int[p, i, loc.bottom] = alphas(p, r0, xi0, eta0, a)

        metric = N.array([[hrrd_int[p, i, loc.bottom], hr1d_int[p, i, loc.bottom], hr2d_int[p, i, loc.bottom]], \
                          [hr1d_int[p, i, loc.bottom], h11d_int[p, i, loc.bottom], h12d_int[p, i, loc.bottom]], \
                          [hr2d_int[p, i, loc.bottom], h12d_int[p, i, loc.bottom], h22d_int[p, i, loc.bottom]]])
        
        inv_metric = N.linalg.inv(metric)
        hrru_int[p, i, loc.bottom] = inv_metric[0, 0]
        hr1u_int[p, i, loc.bottom] = inv_metric[0, 1]
        hr2u_int[p, i, loc.bottom] = inv_metric[0, 2]
        h11u_int[p, i, loc.bottom] = inv_metric[1, 1]
        h12u_int[p, i, loc.bottom] = inv_metric[1, 2]
        h22u_int[p, i, loc.bottom] = inv_metric[2, 2]
        sqrt_det_h_int[p, i, loc.bottom] = N.sqrt(N.linalg.det(metric))

        # Right edge
        xi0 = xi_int[-1]
        eta0 = eta_int[i]
        h11d_int[p, i, loc.right] = g11d(p, r0, xi0, eta0, a)
        h22d_int[p, i, loc.right] = g22d(p, r0, xi0, eta0, a)
        h12d_int[p, i, loc.right] = g12d(p, r0, xi0, eta0, a)
        hrrd_int[p, i, loc.right] = grrd(p, r0, xi0, eta0, a)
        hr1d_int[p, i, loc.right] = gr1d(p, r0, xi0, eta0, a)
        hr2d_int[p, i, loc.right] = gr2d(p, r0, xi0, eta0, a)
        alpha_int[p, i, loc.right] = alphas(p, r0, xi0, eta0, a)

        metric = N.array([[hrrd_int[p, i, loc.right], hr1d_int[p, i, loc.right], hr2d_int[p, i, loc.right]], \
                          [hr1d_int[p, i, loc.right], h11d_int[p, i, loc.right], h12d_int[p, i, loc.right]], \
                          [hr2d_int[p, i, loc.right], h12d_int[p, i, loc.right], h22d_int[p, i, loc.right]]])
        
        inv_metric = N.linalg.inv(metric)
        hrru_int[p, i, loc.right] = inv_metric[0, 0]
        hr1u_int[p, i, loc.right] = inv_metric[0, 1]
        hr2u_int[p, i, loc.right] = inv_metric[0, 2]
        h11u_int[p, i, loc.right] = inv_metric[1, 1]
        h12u_int[p, i, loc.right] = inv_metric[1, 2]
        h22u_int[p, i, loc.right] = inv_metric[2, 2]
        sqrt_det_h_int[p, i, loc.right] = N.sqrt(N.linalg.det(metric))

        # Top edge
        xi0 = xi_int[i]
        eta0 = eta_int[-1]
        h11d_int[p, i, loc.top] = g11d(p, r0, xi0, eta0, a)
        h22d_int[p, i, loc.top] = g22d(p, r0, xi0, eta0, a)
        h12d_int[p, i, loc.top] = g12d(p, r0, xi0, eta0, a)
        hrrd_int[p, i, loc.top] = grrd(p, r0, xi0, eta0, a)
        hr1d_int[p, i, loc.top] = gr1d(p, r0, xi0, eta0, a)
        hr2d_int[p, i, loc.top] = gr2d(p, r0, xi0, eta0, a)
        alpha_int[p, i, loc.top] = alphas(p, r0, xi0, eta0, a)

        metric = N.array([[hrrd_int[p, i, loc.top], hr1d_int[p, i, loc.top], hr2d_int[p, i, loc.top]], \
                          [hr1d_int[p, i, loc.top], h11d_int[p, i, loc.top], h12d_int[p, i, loc.top]], \
                          [hr2d_int[p, i, loc.top], h12d_int[p, i, loc.top], h22d_int[p, i, loc.top]]])
        
        inv_metric = N.linalg.inv(metric)
        hrru_int[p, i, loc.top] = inv_metric[0, 0]
        hr1u_int[p, i, loc.top] = inv_metric[0, 1]
        hr2u_int[p, i, loc.top] = inv_metric[0, 2]
        h11u_int[p, i, loc.top] = inv_metric[1, 1]
        h12u_int[p, i, loc.top] = inv_metric[1, 2]
        h22u_int[p, i, loc.top] = inv_metric[2, 2]
        sqrt_det_h_int[p, i, loc.top] = N.sqrt(N.linalg.det(metric))


# Time step
dt = cfl * N.min(1.0 / N.sqrt(1.0 / (r0 * r0 * dxi * dxi) + 1.0 / (r0 * r0 * deta * deta)))
print("delta t = {}".format(dt))

########
# Interpolation routine
########

# def interp(arr_in, xA, xB):
#     f = interpolate.interp1d(xA, arr_in, axis = 1, kind='linear', fill_value=(0,0), bounds_error=False)
#     return f(xB)

def interp(arr_in, xA, xB):
    return N.interp(xB, xA, arr_in)

########
# Generic vector transformation 
########

def transform_vect(patch0, patch1, xi0, eta0, vxi0, veta0):
    fcoord0 = (globals()["coord_" + sphere[patch0] + "_to_sph"])
    theta0, phi0 = fcoord0(xi0, eta0)
    fvec0 = (globals()["vec_" + sphere[patch0] + "_to_sph"])
    fvec1 = (globals()["vec_sph_to_" + sphere[patch1]])
    return fvec1(theta0, phi0, *fvec0(xi0, eta0, vxi0, veta0))

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

P_half_4 = N.ones(Nxi_half)
P_half_4[0] = 5/63
P_half_4[1] = 121/128
P_half_4[2] = 1085/1152
P_half_4[3] = 401/384
P_half_4[4] = 2659/2688
P_half_4[-1] = 5/63
P_half_4[-2] = 121/128
P_half_4[-3] = 1085/1152
P_half_4[-4] = 401/384
P_half_4[-5] = 2659/2688

def compute_diff_H(p):
    
    dHrd1[p, 0, :] = (- 0.5 * Hrd[p, 0, :] + 0.25 * Hrd[p, 1, :] + 0.25 * Hrd[p, 2, :]) / dxi / P_int_2[0]
    dHrd1[p, 1, :] = (- 0.5 * Hrd[p, 0, :] - 0.25 * Hrd[p, 1, :] + 0.75 * Hrd[p, 2, :]) / dxi / P_int_2[1]
    dHrd1[p, Nxi_int - 2, :] = (- 0.75 * Hrd[p, -3, :] + 0.25 * Hrd[p, -2, :] + 0.5 * Hrd[p, -1, :]) / dxi / P_int_2[Nxi_int - 2]
    dHrd1[p, Nxi_int - 1, :] = (- 0.25 * Hrd[p, -3, :] - 0.25 * Hrd[p, -2, :] + 0.5 * Hrd[p, -1, :]) / dxi / P_int_2[Nxi_int - 1]
    dHrd1[p, 2:(Nxi_int - 2), :] = (N.roll(Hrd, -1, axis = 1)[p, 2:(Nxi_int - 2), :] - Hrd[p, 2:(Nxi_int - 2), :]) / dxi

    dHrd2[p, :, 0] = (- 0.5 * Hrd[p, :, 0] + 0.25 * Hrd[p, :, 1] + 0.25 * Hrd[p, :, 2]) / deta / P_int_2[0]
    dHrd2[p, :, 1] = (- 0.5 * Hrd[p, :, 0] - 0.25 * Hrd[p, :, 1] + 0.75 * Hrd[p, :, 2]) / deta / P_int_2[1]
    dHrd2[p, :, Nxi_int - 2] = (- 0.75 * Hrd[p, :, -3] + 0.25 * Hrd[p, :, -2] + 0.5 * Hrd[p, :, -1]) / deta / P_int_2[Nxi_int - 2]
    dHrd2[p, :, Nxi_int - 1] = (- 0.25 * Hrd[p, :, -3] - 0.25 * Hrd[p, :, -2] + 0.5 * Hrd[p, :, -1]) / deta / P_int_2[Nxi_int - 1]
    dHrd2[p, :, 2:(Neta_int - 2)] = (N.roll(Hrd, -1, axis = 2)[p, :, 2:(Neta_int - 2)] - Hrd[p, :, 2:(Neta_int - 2)]) / deta

    dH1d2[p, :, 0] = (- 0.5 * H1d[p, :, 0] + 0.25 * H1d[p, :, 1] + 0.25 * H1d[p, :, 2]) / deta / P_int_2[0]
    dH1d2[p, :, 1] = (- 0.5 * H1d[p, :, 0] - 0.25 * H1d[p, :, 1] + 0.75 * H1d[p, :, 2]) / deta / P_int_2[1]
    dH1d2[p, :, Nxi_int - 2] = (- 0.75 * H1d[p, :, -3] + 0.25 * H1d[p, :, -2] + 0.5 * H1d[p, :, -1]) / deta / P_int_2[Nxi_int - 2]
    dH1d2[p, :, Nxi_int - 1] = (- 0.25 * H1d[p, :, -3] - 0.25 * H1d[p, :, -2] + 0.5 * H1d[p, :, -1]) / deta / P_int_2[Nxi_int - 1]
    dH1d2[p, :, 2:(Neta_int - 2)] = (N.roll(H1d, -1, axis = 2)[p, :, 2:(Neta_int - 2)] - H1d[p, :, 2:(Neta_int - 2)]) / deta

    dH2d1[p, 0, :] = (- 0.5 * H2d[p, 0, :] + 0.25 * H2d[p, 1, :] + 0.25 * H2d[p, 2, :]) / dxi / P_int_2[0]
    dH2d1[p, 1, :] = (- 0.5 * H2d[p, 0, :] - 0.25 * H2d[p, 1, :] + 0.75 * H2d[p, 2, :]) / dxi / P_int_2[1]
    dH2d1[p, Nxi_int - 2, :] = (- 0.75 * H2d[p, -3, :] + 0.25 * H2d[p, -2, :] + 0.5 * H2d[p, -1, :]) / dxi / P_int_2[Nxi_int - 2]
    dH2d1[p, Nxi_int - 1, :] = (- 0.25 * H2d[p, -3, :] - 0.25 * H2d[p, -2, :] + 0.5 * H2d[p, -1, :]) / dxi / P_int_2[Nxi_int - 1]
    dH2d1[p, 2:(Nxi_int - 2), :] = (N.roll(H2d, -1, axis = 1)[p, 2:(Nxi_int - 2), :] - H2d[p, 2:(Nxi_int - 2), :]) / dxi

def compute_diff_E(p):

    # dE2d1[p, 0, :] = (-55/378 * E2d[p, 0, :] + 5/21 * E2d[p, 1, :] -5/42 * E2d[p, 2, :] + 5/189 * E2d[p, 3, :]) / dxi / P_half_4[0]
    # dE2d1[p, 1, :] = (-2783/3072 * E2d[p, 0, :] + 847/1024 * E2d[p, 1, :] + 121/1024 * E2d[p, 2, :] -121/3072 * E2d[p, 3, :]) / dxi / P_half_4[1]

    dE2d1[p, 0, :] = (- 0.50 * E2d[p, 0, :] + 0.50 * E2d[p, 1, :]) / dxi / P_half_2[0]
    dE2d1[p, 1, :] = (- 0.25 * E2d[p, 0, :] + 0.25 * E2d[p, 1, :]) / dxi / P_half_2[1]
    dE2d1[p, 2, :] = (- 0.25 * E2d[p, 0, :] - 0.75 * E2d[p, 1, :] + E2d[p, 2, :]) / dxi / P_half_2[2]
    dE2d1[p, Nxi_half - 3, :] = (- E2d[p, -3, :] + 0.75 * E2d[p, -2, :] + 0.25 * E2d[p, -1, :]) / dxi / P_half_2[Nxi_half - 3]
    
    dE2d1[p, Nxi_half - 2, :] = (- 0.25 * E2d[p, -2, :] + 0.25 * E2d[p, -1, :]) / dxi / P_half_2[Nxi_half - 2]
    dE2d1[p, Nxi_half - 1, :] = (- 0.5 * E2d[p, -2, :] + 0.5 * E2d[p, -1, :]) / dxi / P_half_2[Nxi_half - 1]
    
    # dE2d1[p, Nxi_half -1, :] = (55/378 * E2d[p, -1, :] - 5/21 * E2d[p, -2, :] + 5/42 * E2d[p, -3, :] - 5/189 * E2d[p, -4, :]) / dxi / P_half_4[-1]
    # dE2d1[p, Nxi_half -2, :] = (2783/3072 * E2d[p, -1, :] - 847/1024 * E2d[p, -2, :] - 121/1024 * E2d[p, -3, :] + 121/3072 * E2d[p, -4, :]) / dxi / P_half_4[-2]

    dE2d1[p, 3:(Nxi_half - 3), :] = (E2d[p, 3:(Nxi_half - 3), :] - N.roll(E2d, 1, axis = 1)[p, 3:(Nxi_half - 3), :]) / dxi


    # dE1d2[p, :, 0] = (-55/378 * E1d[p, :, 0] + 5/21 * E1d[p, :, 1] -5/42 * E1d[p, :, 2] + 5/189 * E1d[p, :, 3]) / deta / P_half_4[0]
    # dE1d2[p, :, 1] = (-2783/3072 * E1d[p, :, 0] + 847/1024 * E1d[p, :, 1] + 121/1024 * E1d[p, :, 2] -121/3072 * E1d[p, :, 3]) / deta / P_half_4[1]

    dE1d2[p, :, 0] = (- 0.50 * E1d[p, :, 0] + 0.50 * E1d[p, :, 1]) / dxi / P_half_2[0]
    dE1d2[p, :, 1] = (- 0.25 * E1d[p, :, 0] + 0.25 * E1d[p, :, 1]) / dxi / P_half_2[1]

    dE1d2[p, :, 2] = (- 0.25 * E1d[p, :, 0] - 0.75 * E1d[p, :, 1] + E1d[p, :, 2]) / dxi / P_half_2[2]
    dE1d2[p, :, Neta_half - 3] = (- E1d[p, :, -3] + 0.75 * E1d[p, :, -2] + 0.25 * E1d[p, :, -1]) / deta / P_half_2[Nxi_half - 3]

    dE1d2[p, :, Neta_half - 2] = (- 0.25 * E1d[p, :, -2] + 0.25 * E1d[p, :, -1]) / deta / P_half_2[Nxi_half - 2]
    dE1d2[p, :, Neta_half - 1] = (- 0.50 * E1d[p, :, -2] + 0.50 * E1d[p, :, -1]) / deta / P_half_2[Nxi_half - 1]

    # dE1d2[p, :, Neta_half -1] = (55/378 * E1d[p, :, -1] - 5/21 * E1d[p, :, -2] + 5/42 * E1d[p, :, -3] - 5/189 * E1d[p, :, -4]) / deta / P_half_4[-1]
    # dE1d2[p, :, Neta_half -2] = (2783/3072 * E1d[p, :, -1] - 847/1024 * E1d[p, :, -2] - 121/1024 * E1d[p, :, -3] + 121/3072 * E1d[p, :, -4]) / deta / P_half_4[-2]

    dE1d2[p, :, 3:(Neta_half - 3)] = (E1d[p, :, 3:(Neta_half - 3)] - N.roll(E1d, 1, axis = 2)[p, :, 3:(Neta_half - 3)]) / deta

    
    # dErd1[p, 0, :] = (-55/378 * Erd[p, 0, :] + 5/21 * Erd[p, 1, :] -5/42 * Erd[p, 2, :] + 5/189 * Erd[p, 3, :]) / dxi / P_half_4[0]
    # dErd1[p, 1, :] = (-2783/3072 * Erd[p, 0, :] + 847/1024 * Erd[p, 1, :] + 121/1024 * Erd[p, 2, :] -121/3072 * Erd[p, 3, :]) / dxi / P_half_4[1]

    dErd1[p, 0, :] = (- 0.50 * Erd[p, 0, :] + 0.50 * Erd[p, 1, :]) / dxi / P_half_2[0]
    dErd1[p, 1, :] = (- 0.25 * Erd[p, 0, :] + 0.25 * Erd[p, 1, :]) / dxi / P_half_2[1]

    dErd1[p, 2, :] = (- 0.25 * Erd[p, 0, :] - 0.75 * Erd[p, 1, :] + Erd[p, 2, :]) / dxi / P_half_2[2]
    dErd1[p, Nxi_half - 3, :] = (- Erd[p, -3, :] + 0.75 * Erd[p, -2, :] + 0.25 * Erd[p, -1, :]) / dxi / P_half_2[Nxi_half - 3]
    
    dErd1[p, Nxi_half - 2, :] = (- 0.25 * Erd[p, -2, :] + 0.25 * Erd[p, -1, :]) / dxi / P_half_2[Nxi_half - 2]
    dErd1[p, Nxi_half - 1, :] = (- 0.5 * Erd[p, -2, :] + 0.5 * Erd[p, -1, :]) / dxi / P_half_2[Nxi_half - 1]

    # dErd1[p, Nxi_half -1, :] = (55/378 * Erd[p, -1, :] - 5/21 * Erd[p, -2, :] + 5/42 * Erd[p, -3, :] - 5/189 * Erd[p, -4, :]) / dxi / P_half_4[-1]
    # dErd1[p, Nxi_half -2, :] = (2783/3072 * Erd[p, -1, :] - 847/1024 * Erd[p, -2, :] - 121/1024 * Erd[p, -3, :] + 121/3072 * Erd[p, -4, :]) / dxi / P_half_4[-2]
    
    dErd1[p, 3:(Nxi_half - 3), :] = (Erd[p, 3:(Nxi_half - 3), :] - N.roll(Erd, 1, axis = 1)[p, 3:(Nxi_half - 3), :]) / dxi


    # dErd2[p, :, 0] = (-55/378 * Erd[p, :, 0] + 5/21 * Erd[p, :, 1] -5/42 * Erd[p, :, 2] + 5/189 * Erd[p, :, 3]) / deta / P_half_4[0]
    # dErd2[p, :, 1] = (-2783/3072 * Erd[p, :, 0] + 847/1024 * Erd[p, :, 1] + 121/1024 * Erd[p, :, 2] -121/3072 * Erd[p, :, 3]) / deta / P_half_4[1]

    # dErd2[p, :, 0] = (- 3.0 * Erd[p, :, 0] + 4.0 * Erd[p, :, 1] - 1.0 * E2d[p, :, 2]) / dxi / 2.0
    # dErd2[p, :, 1] = (- 3.0 * Erd[p, :, 0] + 4.0 * Erd[p, :, 1] - 1.0 * E2d[p, :, 2]) / dxi / 2.0

    dErd2[p, :, 0] = (- 0.50 * Erd[p, :, 0] + 0.50 * Erd[p, :, 1]) / dxi / P_half_2[0]
    dErd2[p, :, 1] = (- 0.25 * Erd[p, :, 0] + 0.25 * Erd[p, :, 1]) / dxi / P_half_2[1]
    
    dErd2[p, :, 2] = (- 0.25 * Erd[p, :, 0] - 0.75 * Erd[p, :, 1] + Erd[p, :, 2]) / dxi / P_half_2[2]
    dErd2[p, :, Neta_half - 3] = (- Erd[p, :, -3] + 0.75 * Erd[p, :, -2] + 0.25 * Erd[p, :, -1]) / deta / P_half_2[Nxi_half - 3]
    
    dErd2[p, :, Neta_half - 2] = (- 0.25 * Erd[p, :, -2] + 0.25 * Erd[p, :, -1]) / deta / P_half_2[Nxi_half - 2]
    dErd2[p, :, Neta_half - 1] = (- 0.50 * Erd[p, :, -2] + 0.50 * Erd[p, :, -1]) / deta / P_half_2[Nxi_half - 1]
    
    # dErd2[p, :, Neta_half - 2] = (1.0 * Erd[p, :, -3] - 4.0 * Erd[p, :, -2] + 3.0 * Erd[p, :, -1]) / deta / 2.0
    # dErd2[p, :, Neta_half - 1] = (1.0 * Erd[p, :, -3] - 4.0 * Erd[p, :, -2] + 3.0 * Erd[p, :, -1]) / deta / 2.0

    # dErd2[p, :, Neta_half -1] = (55/378    * Erd[p, :, -1] - 5/21     * Erd[p, :, -2] + 5/42     * Erd[p, :, -3] - 5/189    * Erd[p, :, -4]) / deta / P_half_4[-1]
    # dErd2[p, :, Neta_half -2] = (2783/3072 * Erd[p, :, -1] - 847/1024 * Erd[p, :, -2] - 121/1024 * Erd[p, :, -3] + 121/3072 * Erd[p, :, -4]) / deta / P_half_4[-2]

    dErd2[p, :, 3:(Neta_half - 3)] = (Erd[p, :, 3:(Neta_half - 3)] - N.roll(Erd, 1, axis = 2)[p, :, 3:(Neta_half - 3)]) / deta

def average_field(p, fieldrin0, field1in0, field2in0, fieldrin1, field1in1, field2in1, fieldrout, field1out, field2out):
    fieldrout[p, :, :] = 0.5 * (fieldrin0[p, :, :] + fieldrin1[p, :, :])
    field1out[p, :, :] = 0.5 * (field1in0[p, :, :] + field1in1[p, :, :])
    field2out[p, :, :] = 0.5 * (field2in0[p, :, :] + field2in1[p, :, :])

########
# Single-patch push routines
########

def push_D(p, Drin, D1in, D2in, dtin, itime):

        Drin[p, :, :] += dtin * (dH2d1[p, :, :] - dH1d2[p, :, :]) / sqrt_det_h[p, :, :, 0] 

        # Interior
        D1in[p, 1:-1, :] += dtin * (dHrd2[p, 1:-1, :]) / sqrt_det_h[p, :-1, :, 1] 
        # Left edge
        D1in[p, 0, :] += dtin * (dHrd2[p, 0, :]) / sqrt_det_h[p, 0, :, 0] 
        # Right edge
        D1in[p, -1, :] += dtin * (dHrd2[p, -1, :]) / sqrt_det_h[p, -1, :, 0]

        # Interior
        D2in[p, :, 1:-1] += dtin * (- dHrd1[p, :, 1:-1]) / sqrt_det_h[p, :, :-1, 2]
        # Bottom edge
        D2in[p, :, 0] += dtin * (- dHrd1[p, :, 0]) / sqrt_det_h[p, :, 0, 0]
        # Top edge
        D2in[p, :, -1] += dtin * (- dHrd1[p, :, -1]) / sqrt_det_h[p, :, -1, 0]

        # Current
        Drin[p, :, :] += dtin * Jz[p, :, :] * N.sin(20.0 * itime * dtin) * (1 + N.tanh(20 - itime/5.))/2.

def push_B(p, Brin, B1in, B2in, dtin, itime):
        
        # Interior
        Brin[p, 1:-1, 1:-1] += dtin * (dE1d2[p, 1:-1, 1:-1] - dE2d1[p, 1:-1, 1:-1]) / sqrt_det_h[p, :-1, :-1, 3] 
        # Left edge
        Brin[p, 0, 1:-1] += dtin * (dE1d2[p, 0, 1:-1] - dE2d1[p, 0, 1:-1]) / sqrt_det_h[p, 0, :-1, 2] 
        # Right edge
        Brin[p, -1, 1:-1] += dtin * (dE1d2[p, -1, 1:-1] - dE2d1[p, -1, 1:-1]) / sqrt_det_h[p, -1, :-1, 2] 
        # Bottom edge
        Brin[p, 1:-1, 0] += dtin * (dE1d2[p, 1:-1, 0] - dE2d1[p, 1:-1, 0]) / sqrt_det_h[p, :-1, 0, 1] 
        # Top edge
        Brin[p, 1:-1, -1] += dtin * (dE1d2[p, 1:-1, -1] - dE2d1[p, 1:-1, -1]) / sqrt_det_h[p, :-1, -1, 1] 
        # Bottom left corner
        Brin[p, 0, 0] += dtin * (dE1d2[p, 0, 0] - dE2d1[p, 0, 0]) / sqrt_det_h[p, 0, 0, 0] 
        # Bottom right corner
        Brin[p, -1, 0] += dtin * (dE1d2[p, -1, 0] - dE2d1[p, -1, 0]) / sqrt_det_h[p, -1, 0, 0] 
        # Top left corner
        Brin[p, 0, -1] += dtin * (dE1d2[p, 0, -1] - dE2d1[p, 0, -1]) / sqrt_det_h[p, 0, -1, 0] 
        # Top right corner
        Brin[p, -1, -1] += dtin * (dE1d2[p, -1, -1] - dE2d1[p, -1, -1]) / sqrt_det_h[p, -1, -1, 0] 

        # Interior
        B1in[p, :, 1:-1] += dtin * (- dErd2[p, :, 1:-1]) / sqrt_det_h[p, :, :-1, 2]
        # Bottom edge
        B1in[p, :, 0] += dtin * (- dErd2[p, :, 0]) / sqrt_det_h[p, :, 0, 0]
        # Top edge
        B1in[p, :, -1] += dtin * (- dErd2[p, :, -1]) / sqrt_det_h[p, :, -1, 0]

        # Interior
        B2in[p, 1:-1, :] += dtin * (dErd1[p, 1:-1, :]) / sqrt_det_h[p, :-1, :, 1] 
        # Left edge
        B2in[p, 0, :] += dtin * (dErd1[p, 0, :]) / sqrt_det_h[p, 0, :, 0] 
        # Right edge
        B2in[p, -1, :] += dtin * (dErd1[p, -1, :]) / sqrt_det_h[p, -1, :, 0]

########
# Auxiliary field computation
########

def contra_to_cov_D(p, Drin, D1in, D2in):

    ########
    # Dr
    ########

    # Interior
    Drd[p, 1:-1, 1:-1] = hrrd[p, 1:-1, 1:-1, 0] * Drin[p, 1:-1, 1:-1] \
                                      + 0.5 * hr1d[p, 1:-1, 1:-1, 0] * (D1in[p, 1:-2, 1:-1] + N.roll(D1in, -1, axis = 1)[p, 1:-2, 1:-1]) \
                                      + 0.5 * hr2d[p, 1:-1, 1:-1, 0] * (D2in[p, 1:-1, 1:-2] + N.roll(D2in, -1, axis = 2)[p, 1:-1, 1:-2]) \

    # Left edge
    Drd[p, 0, 1:-1] = hrrd[p, 0, 1:-1, 0] * Drin[p, 0, 1:-1] \
                                   + hr1d[p, 0, 1:-1, 0] * D1in[p, 0, 1:-1] \
                                   + 0.5 * hr2d[p, 0, 1:-1, 0] * (D2in[p, 0, 1:-2] + N.roll(D2in, -1, axis = 2)[p, 0, 1:-2])
    # Right edge
    Drd[p, -1, 1:-1] = hrrd[p, -1, 1:-1, 0] * Drin[p, -1, 1:-1] \
                                    + hr1d[p, -1, 1:-1, 0] * D1in[p, -1, 1:-1] \
                                    + 0.5 * hr2d[p, -1, 1:-1, 0] * (D2in[p, -1, 1:-2] + N.roll(D2in, -1, axis = 2)[p, -1, 1:-2])

    # Bottom edge
    Drd[p, 1:-1, 0] = hrrd[p, 1:-1, 0, 0] * Drin[p, 1:-1, 0] \
                                   + 0.5 * hr1d[p, 1:-1, 0, 0] * (D1in[p, 1:-2, 0] + N.roll(D1in, -1, axis = 1)[p, 1:-2, 0]) \
                                   + hr2d[p, 1:-1, 0, 0] * D2in[p, 1:-1, 0]
    # Top edge
    Drd[p, 1:-1, -1] = hrrd[p, 1:-1, -1, 0] * Drin[p, 1:-1, -1] \
                                    + 0.5 * hr1d[p, 1:-1, -1, 0] * (D1in[p, 1:-2, -1] + N.roll(D1in, -1, axis = 1)[p, 1:-2, -1]) \
                                    + hr2d[p, 1:-1, -1, 0] * D2in[p, 1:-1, -1]
    # Bottom-left corner
    Drd[p, 0, 0] = hrrd[p, 0, 0, 0] * Drin[p, 0, 0] \
                                + hr1d[p, 0, 0, 0] * D1in[p, 0, 0] \
                                + hr2d[p, 0, 0, 0] * D2in[p, 0, 0]
    # Top-left corner
    Drd[p, 0, -1] = hrrd[p, 0, -1, 0] * Drin[p, 0, -1] \
                                 + hr1d[p, 0, -1, 0] * D1in[p, 0, -1] \
                                 + hr2d[p, 0, -1, 0] * D2in[p, 0, -1]
    # Bottom-right corner
    Drd[p, -1, 0] = hrrd[p, -1, 0, 0] * Drin[p, -1, 0] \
                                 + hr1d[p, -1, 0, 0] * D1in[p, -1, 0] \
                                 + hr2d[p, -1, 0, 0] * D2in[p, -1, 0]
    # Top-right corner
    Drd[p, -1, -1] = hrrd[p, -1, -1, 0] * Drin[p, -1, -1] \
                                  + hr1d[p, -1, -1, 0] * D1in[p, -1, -1] \
                                  + hr2d[p, -1, -1, 0] * D2in[p, -1, -1]

    ########
    # Dxi
    ########

    # Interior
    D1d[p, 1:-1, 1:-1] = h11d[p, :-1, 1:-1, 1] * D1in[p, 1:-1, 1:-1] \
                       + 0.25 * h12d[p, :-1, 1:-1, 1] * (D2in[p, 1:, 1:-2] + N.roll(N.roll(D2in, 1, axis = 1), -1, axis = 2)[p, 1:, 1:-2] \
                                                      +  N.roll(D2in, 1, axis = 1)[p, 1:, 1:-2] + N.roll(D2in, -1, axis = 2)[p, 1:, 1:-2]) \
                       + 0.5  * hr1d[p, :-1, 1:-1, 1] * (Drin[p, 1:, 1:-1] + N.roll(Drin, 1, axis = 1)[p, 1:, 1:-1])

    # Left edge
    D1d[p, 0, 1:-1] = h11d[p, 0, 1:-1, 0] * D1in[p, 0, 1:-1] \
                                   + 0.5 * h12d[p, 0, 1:-1, 0] * (D2in[p, 0, 1:-2] + N.roll(D2in, -1, axis = 2)[p, 0, 1:-2]) \
                                   + hr1d[p, 0, 1:-1, 0] * Drin[p, 0, 1:-1]
    # Right edge
    D1d[p, -1, 1:-1] = h11d[p, -1, 1:-1, 0] * D1in[p, -1, 1:-1] \
                                    + 0.5 * h12d[p, -1, 1:-1, 0] * (D2in[p, -1, 1:-2] + N.roll(D2in, -1, axis = 2)[p, -1, 1:-2]) \
                                    + hr1d[p, -1, 1:-1, 0] * Drin[p, -1, 1:-1]
    # Bottom edge
    D1d[p, 1:-1, 0] = h11d[p, :-1, 0, 1] * D1in[p, 1:-1, 0] \
                                   + 0.5 * h12d[p, :-1, 0, 1] * (D2in[p, 1:, 0] + N.roll(D2in, 1, axis = 1)[p, 1:, 0]) \
                                   + 0.5 * hr1d[p, :-1, 0, 1] * (Drin[p, 1:, 0] + N.roll(Drin, 1, axis = 1)[p, 1:, 0])
    # Top edge
    D1d[p, 1:-1, -1] = h11d[p, :-1, -1, 1] * D1in[p, 1:-1, -1] \
                                    + 0.5 * h12d[p, :-1, -1, 1] * (D2in[p, 1:, -1] + N.roll(D2in, 1, axis = 1)[p, 1:, -1]) \
                                    + 0.5 * hr1d[p, :-1, -1, 1] * (Drin[p, 1:, -1] + N.roll(Drin, 1, axis = 1)[p, 1:, -1])
    # Bottom-left corner
    D1d[p, 0, 0] = h11d[p, 0, 0, 0] * D1in[p, 0, 0] \
                                + h12d[p, 0, 0, 0] * D2in[p, 0, 0] \
                                + hr1d[p, 0, 0, 0] * Drin[p, 0, 0]
    # Top-left corner
    D1d[p, 0, -1] = h11d[p, 0, -1, 0] * D1in[p, 0, -1] \
                                 + h12d[p, 0, -1, 0] * D2in[p, 0, -1] \
                                 + hr1d[p, 0, -1, 0] * Drin[p, 0, -1]
    # Bottom-right corner
    D1d[p, -1, 0] = h11d[p, -1, 0, 0] * D1in[p, -1, 0] \
                                 + h12d[p, -1, 0, 0] * D2in[p, -1, 0] \
                                 + hr1d[p, -1, 0, 0] * Drin[p, -1, 0]
    # Top-right corner
    D1d[p, -1, -1] = h11d[p, -1, -1, 0] * D1in[p, -1, -1] \
                                  + h12d[p, -1, -1, 0] * D2in[p, -1, -1] \
                                  + hr1d[p, -1, -1, 0] * Drin[p, -1, -1]

    ########
    # Deta
    ########

    # Interior
    D2d[p, 1:-1, 1:-1] = h22d[p, 1:-1, :-1, 2] * D2in[p, 1:-1, 1:-1] \
                       + 0.25 * h12d[p, 1:-1, :-1, 2] * (D1in[p, 1:-2, 1:] + N.roll(N.roll(D1in, -1, axis = 1), 1, axis = 2)[p, 1:-2, 1:] \
                                                      +  N.roll(D1in, -1, axis = 1)[p, 1:-2, 1:] + N.roll(D1in, 1, axis = 2)[p, 1:-2, 1:]) \
                       + 0.5  * hr2d[p, 1:-1, :-1, 2] * (Drin[p, 1:-1, 1:] + N.roll(Drin, 1, axis = 2)[p, 1:-1, 1:])

    # Left edge
    D2d[p, 0, 1:-1] = h22d[p, 0, :-1, 2] * D2in[p, 0, 1:-1] \
                       + 0.5 * h12d[p, 0, :-1, 2] * (D1in[p, 0, 1:] + N.roll(D1in, 1, axis = 2)[p, 0, 1:]) \
                       + 0.5 * hr2d[p, 0, :-1, 2] * (Drin[p, 0, 1:] + N.roll(Drin, 1, axis = 2)[p, 0, 1:])
    # Right edge
    D2d[p, -1, 1:-1] = h22d[p, -1, :-1, 2] * D2in[p, -1, 1:-1] \
                        + 0.5 * h12d[p, -1, :-1, 2] * (D1in[p, -1, 1:] + N.roll(D1in, 1, axis = 2)[p, -1, 1:]) \
                        + 0.5 * hr2d[p, -1, :-1, 2] * (Drin[p, -1, 1:] + N.roll(Drin, 1, axis = 2)[p, -1, 1:])
    # Bottom edge
    D2d[p, 1:-1, 0] = h22d[p, 1:-1, 0, 0] * D2in[p, 1:-1, 0] \
                       + 0.5 * h12d[p, 1:-1, 0, 0] * (D1in[p, 1:-2, 0] + N.roll(D1in, -1, axis = 1)[p, 1:-2, 0]) \
                       + hr2d[p, 1:-1, 0, 0] * Drin[p, 1:-1, 0]
    # Top edge
    D2d[p, 1:-1, -1] = h22d[p, 1:-1, -1, 0] * D2in[p, 1:-1, -1] \
                        + 0.5 * h12d[p, 1:-1, -1, 0] * (D1in[p, 1:-2, -1] + N.roll(D1in, -1, axis = 1)[p, 1:-2, -1]) \
                        + hr2d[p, 1:-1, -1, 0] * Drin[p, 1:-1, -1]
    # Bottom-left corner
    D2d[p, 0, 0] = h22d[p, 0, 0, 0] * D2in[p, 0, 0] \
                                + h12d[p, 0, 0, 0] * D1in[p, 0, 0] \
                                + hr2d[p, 0, 0, 0] * Drin[p, 0, 0]
    # Top-left corner
    D2d[p, 0, -1] = h22d[p, 0, -1, 0] * D2in[p, 0, -1] \
                                 + h12d[p, 0, -1, 0] * D1in[p, 0, -1] \
                                 + hr2d[p, 0, -1, 0] * Drin[p, 0, -1]
    # Bottom-right corner
    D2d[p, -1, 0] = h22d[p, -1, 0, 0] * D2in[p, -1, 0] \
                                 + h12d[p, -1, 0, 0] * D1in[p, -1, 0] \
                                 + hr2d[p, -1, 0, 0] * Drin[p, -1, 0]
    # Top-right corner
    D2d[p, -1, -1] = h22d[p, -1, -1, 0] * D2in[p, -1, -1] \
                                  + h12d[p, -1, -1, 0] * D1in[p, -1, -1] \
                                  + hr2d[p, -1, -1, 0] * Drin[p, -1, -1]

def compute_E_aux(p, Drin, D1in, D2in, Brin, B1in, B2in):

    ##### Er
    Erd[p, :, :] = alpha[p, :, :, 0] * Drin[p, :, :]

    ##### Exi
    # Interior
    E1d[p, 1:-1, :] = alpha[p, :-1, :, 1] * D1in[p, 1:-1, :] \
                    - sqrt_det_h[p, :-1, :, 1] * beta[p, :-1, :, 1] * B2in[p, 1:-1, :]
    # Left edge
    E1d[p, 0, :] = alpha[p, 0, :, 0] * D1in[p, 0, :] \
                - sqrt_det_h[p, 0, :, 0] * beta[p, 0, :, 0] * B2in[p, 0, :]
    # Right edge
    E1d[p, -1, :] = alpha[p, -1, :, 0] * D1in[p, -1, :] \
                  - sqrt_det_h[p, -1, :, 0] * beta[p, -1, :, 0] * B2in[p, -1, :]
                  
    ##### Eeta
    ##### Interior
    E2d[p, :, 1:-1] = alpha[p, :, :-1, 2] * D2in[p, :, 1:-1] \
                    + sqrt_det_h[p, :, :-1, 2] * beta[p, :, :-1, 2] * B1in[p, :, 1:-1] 
    ##### Bottom edge
    E2d[p, :, 0] = alpha[p, :, 0, 0] * D2in[p, :, 0] \
                 + sqrt_det_h[p, :, 0, 0] * beta[p, :, 0, 0] * B1in[p, :, 0] 
    ##### Top edge
    E2d[p, :, -1] = alpha[p, :, -1, 0] * D2in[p, :, -1] \
                  + sqrt_det_h[p, :, -1, 0] * beta[p, :, -1, 0] * B1in[p, :, -1] 

def cov_to_contra_E(p, Drin, D1in, D2in):

    ########
    # Dr
    ########

    # Interior
    Eru[p, 1:-1, 1:-1] = hrru[p, 1:-1, 1:-1, 0] * Drin[p, 1:-1, 1:-1] \
                                      + 0.5 * hr1u[p, 1:-1, 1:-1, 0] * (D1in[p, 1:-2, 1:-1] + N.roll(D1in, -1, axis = 1)[p, 1:-2, 1:-1]) \
                                      + 0.5 * hr2u[p, 1:-1, 1:-1, 0] * (D2in[p, 1:-1, 1:-2] + N.roll(D2in, -1, axis = 2)[p, 1:-1, 1:-2]) \

    # Left edge
    Eru[p, 0, 1:-1] = hrru[p, 0, 1:-1, 0] * Drin[p, 0, 1:-1] \
                                   + hr1u[p, 0, 1:-1, 0] * D1in[p, 0, 1:-1] \
                                   + 0.5 * hr2u[p, 0, 1:-1, 0] * (D2in[p, 0, 1:-2] + N.roll(D2in, -1, axis = 2)[p, 0, 1:-2])
    # Right edge
    Eru[p, -1, 1:-1] = hrru[p, -1, 1:-1, 0] * Drin[p, -1, 1:-1] \
                                    + hr1u[p, -1, 1:-1, 0] * D1in[p, -1, 1:-1] \
                                    + 0.5 * hr2u[p, -1, 1:-1, 0] * (D2in[p, -1, 1:-2] + N.roll(D2in, -1, axis = 2)[p, -1, 1:-2])

    # Bottom edge
    Eru[p, 1:-1, 0] = hrru[p, 1:-1, 0, 0] * Drin[p, 1:-1, 0] \
                                   + 0.5 * hr1u[p, 1:-1, 0, 0] * (D1in[p, 1:-2, 0] + N.roll(D1in, -1, axis = 1)[p, 1:-2, 0]) \
                                   + hr2u[p, 1:-1, 0, 0] * D2in[p, 1:-1, 0]
    # Top edge
    Eru[p, 1:-1, -1] = hrru[p, 1:-1, -1, 0] * Drin[p, 1:-1, -1] \
                                    + 0.5 * hr1u[p, 1:-1, -1, 0] * (D1in[p, 1:-2, -1] + N.roll(D1in, -1, axis = 1)[p, 1:-2, -1]) \
                                    + hr2u[p, 1:-1, -1, 0] * D2in[p, 1:-1, -1]
    # Bottom-left corner
    Eru[p, 0, 0] = hrru[p, 0, 0, 0] * Drin[p, 0, 0] \
                                + hr1u[p, 0, 0, 0] * D1in[p, 0, 0] \
                                + hr2u[p, 0, 0, 0] * D2in[p, 0, 0]
    # Top-left corner
    Eru[p, 0, -1] = hrru[p, 0, -1, 0] * Drin[p, 0, -1] \
                                 + hr1u[p, 0, -1, 0] * D1in[p, 0, -1] \
                                 + hr2u[p, 0, -1, 0] * D2in[p, 0, -1]
    # Bottom-right corner
    Eru[p, -1, 0] = hrru[p, -1, 0, 0] * Drin[p, -1, 0] \
                                 + hr1u[p, -1, 0, 0] * D1in[p, -1, 0] \
                                 + hr2u[p, -1, 0, 0] * D2in[p, -1, 0]
    # Top-right corner
    Eru[p, -1, -1] = hrru[p, -1, -1, 0] * Drin[p, -1, -1] \
                                  + hr1u[p, -1, -1, 0] * D1in[p, -1, -1] \
                                  + hr2u[p, -1, -1, 0] * D2in[p, -1, -1]

    ########
    # Dxi
    ########

    # Interior
    E1u[p, 1:-1, 1:-1] = h11u[p, :-1, 1:-1, 1] * D1in[p, 1:-1, 1:-1] \
                       + 0.25 * h12u[p, :-1, 1:-1, 1] * (D2in[p, 1:, 1:-2] + N.roll(N.roll(D2in, 1, axis = 1), -1, axis = 2)[p, 1:, 1:-2] \
                                                      +  N.roll(D2in, 1, axis = 1)[p, 1:, 1:-2] + N.roll(D2in, -1, axis = 2)[p, 1:, 1:-2]) \
                       + 0.5  * hr1u[p, :-1, 1:-1, 1] * (Drin[p, 1:, 1:-1] + N.roll(Drin, 1, axis = 1)[p, 1:, 1:-1])

    # Left edge
    E1u[p, 0, 1:-1] = h11u[p, 0, 1:-1, 0] * D1in[p, 0, 1:-1] \
                                   + 0.5 * h12u[p, 0, 1:-1, 0] * (D2in[p, 0, 1:-2] + N.roll(D2in, -1, axis = 2)[p, 0, 1:-2]) \
                                   + hr1u[p, 0, 1:-1, 0] * Drin[p, 0, 1:-1]
    # Right edge
    E1u[p, -1, 1:-1] = h11u[p, -1, 1:-1, 0] * D1in[p, -1, 1:-1] \
                                    + 0.5 * h12u[p, -1, 1:-1, 0] * (D2in[p, -1, 1:-2] + N.roll(D2in, -1, axis = 2)[p, -1, 1:-2]) \
                                    + hr1u[p, -1, 1:-1, 0] * Drin[p, -1, 1:-1]
    # Bottom edge
    E1u[p, 1:-1, 0] = h11u[p, :-1, 0, 1] * D1in[p, 1:-1, 0] \
                                   + 0.5 * h12u[p, :-1, 0, 1] * (D2in[p, 1:, 0] + N.roll(D2in, 1, axis = 1)[p, 1:, 0]) \
                                   + 0.5 * hr1u[p, :-1, 0, 1] * (Drin[p, 1:, 0] + N.roll(Drin, 1, axis = 1)[p, 1:, 0])
    # Top edge
    E1u[p, 1:-1, -1] = h11u[p, :-1, -1, 1] * D1in[p, 1:-1, -1] \
                                    + 0.5 * h12u[p, :-1, -1, 1] * (D2in[p, 1:, -1] + N.roll(D2in, 1, axis = 1)[p, 1:, -1]) \
                                    + 0.5 * hr1u[p, :-1, -1, 1] * (Drin[p, 1:, -1] + N.roll(Drin, 1, axis = 1)[p, 1:, -1])
    # Bottom-left corner
    E1u[p, 0, 0] = h11u[p, 0, 0, 0] * D1in[p, 0, 0] \
                                + h12u[p, 0, 0, 0] * D2in[p, 0, 0] \
                                + hr1u[p, 0, 0, 0] * Drin[p, 0, 0]
    # Top-left corner
    E1u[p, 0, -1] = h11u[p, 0, -1, 0] * D1in[p, 0, -1] \
                                 + h12u[p, 0, -1, 0] * D2in[p, 0, -1] \
                                 + hr1u[p, 0, -1, 0] * Drin[p, 0, -1]
    # Bottom-right corner
    E1u[p, -1, 0] = h11u[p, -1, 0, 0] * D1in[p, -1, 0] \
                                 + h12u[p, -1, 0, 0] * D2in[p, -1, 0] \
                                 + hr1u[p, -1, 0, 0] * Drin[p, -1, 0]
    # Top-right corner
    E1u[p, -1, -1] = h11u[p, -1, -1, 0] * D1in[p, -1, -1] \
                                  + h12u[p, -1, -1, 0] * D2in[p, -1, -1] \
                                  + hr1u[p, -1, -1, 0] * Drin[p, -1, -1]

    ########
    # Deta
    ########

    # Interior
    E2u[p, 1:-1, 1:-1] = h22u[p, 1:-1, :-1, 2] * D2in[p, 1:-1, 1:-1] \
                       + 0.25 * h12u[p, 1:-1, :-1, 2] * (D1in[p, 1:-2, 1:] + N.roll(N.roll(D1in, -1, axis = 1), 1, axis = 2)[p, 1:-2, 1:] \
                                                      +  N.roll(D1in, -1, axis = 1)[p, 1:-2, 1:] + N.roll(D1in, 1, axis = 2)[p, 1:-2, 1:]) \
                       + 0.5  * hr2u[p, 1:-1, :-1, 2] * (Drin[p, 1:-1, 1:] + N.roll(Drin, 1, axis = 2)[p, 1:-1, 1:])

    # Left edge
    E2u[p, 0, 1:-1] = h22u[p, 0, :-1, 2] * D2in[p, 0, 1:-1] \
                       + 0.5 * h12u[p, 0, :-1, 2] * (D1in[p, 0, 1:] + N.roll(D1in, 1, axis = 2)[p, 0, 1:]) \
                       + 0.5 * hr2u[p, 0, :-1, 2] * (Drin[p, 0, 1:] + N.roll(Drin, 1, axis = 2)[p, 0, 1:])
    # Right edge
    E2u[p, -1, 1:-1] = h22u[p, -1, :-1, 2] * D2in[p, -1, 1:-1] \
                        + 0.5 * h12u[p, -1, :-1, 2] * (D1in[p, -1, 1:] + N.roll(D1in, 1, axis = 2)[p, -1, 1:]) \
                        + 0.5 * hr2u[p, -1, :-1, 2] * (Drin[p, -1, 1:] + N.roll(Drin, 1, axis = 2)[p, -1, 1:])
    # Bottom edge
    E2u[p, 1:-1, 0] = h22u[p, 1:-1, 0, 0] * D2in[p, 1:-1, 0] \
                       + 0.5 * h12u[p, 1:-1, 0, 0] * (D1in[p, 1:-2, 0] + N.roll(D1in, -1, axis = 1)[p, 1:-2, 0]) \
                       + hr2u[p, 1:-1, 0, 0] * Drin[p, 1:-1, 0]
    # Top edge
    E2u[p, 1:-1, -1] = h22u[p, 1:-1, -1, 0] * D2in[p, 1:-1, -1] \
                        + 0.5 * h12u[p, 1:-1, -1, 0] * (D1in[p, 1:-2, -1] + N.roll(D1in, -1, axis = 1)[p, 1:-2, -1]) \
                        + hr2u[p, 1:-1, -1, 0] * Drin[p, 1:-1, -1]
    # Bottom-left corner
    E2u[p, 0, 0] = h22u[p, 0, 0, 0] * D2in[p, 0, 0] \
                                + h12u[p, 0, 0, 0] * D1in[p, 0, 0] \
                                + hr2u[p, 0, 0, 0] * Drin[p, 0, 0]
    # Top-left corner
    E2u[p, 0, -1] = h22u[p, 0, -1, 0] * D2in[p, 0, -1] \
                                 + h12u[p, 0, -1, 0] * D1in[p, 0, -1] \
                                 + hr2u[p, 0, -1, 0] * Drin[p, 0, -1]
    # Bottom-right corner
    E2u[p, -1, 0] = h22u[p, -1, 0, 0] * D2in[p, -1, 0] \
                                 + h12u[p, -1, 0, 0] * D1in[p, -1, 0] \
                                 + hr2u[p, -1, 0, 0] * Drin[p, -1, 0]
    # Top-right corner
    E2u[p, -1, -1] = h22u[p, -1, -1, 0] * D2in[p, -1, -1] \
                                  + h12u[p, -1, -1, 0] * D1in[p, -1, -1] \
                                  + hr2u[p, -1, -1, 0] * Drin[p, -1, -1]


def contra_to_cov_B(p, Brin, B1in, B2in):

    ########
    # Br
    ########

    # Interior
    Brd[p, 1:-1, 1:-1] = hrrd[p, :-1, :-1, 3] * Brin[p, 1:-1, 1:-1] \
                                      + 0.5 * hr1d[p, :-1, :-1, 3] * (B1in[p, 1:, 1:-1] + N.roll(B1in, 1, axis = 1)[p, 1:, 1:-1]) \
                                      + 0.5 * hr2d[p, :-1, :-1, 3] * (B2in[p, 1:-1, 1:] + N.roll(B2in, 1, axis = 2)[p, 1:-1, 1:])

    # Left edge
    Brd[p, 0, 1:-1] = hrrd[p, 0, :-1, 2] * Brin[p, 0, 1:-1] \
                                      + hr1d[p, 0, :-1, 2] * B1in[p, 0, 1:-1] \
                                      + 0.5 * hr2d[p, 0, :-1, 2] * (B2in[p, 0, 1:] + N.roll(B2in, 1, axis = 2)[p, 0, 1:])

    # Right edge
    Brd[p, -1, 1:-1] = hrrd[p, -1, :-1, 2] * Brin[p, -1, 1:-1] \
                                      + hr1d[p, -1, :-1, 2] * B1in[p, -1, 1:-1] \
                                      + 0.5 * hr2d[p, -1, :-1, 2] * (B2in[p, -1, 1:] + N.roll(B2in, 1, axis = 2)[p, -1, 1:])
    # Bottom edge
    Brd[p, 1:-1, 0] = hrrd[p, :-1, 0, 1] * Brin[p, 1:-1, 0] \
                                      + 0.5 * hr1d[p, :-1, 0, 1] * (B1in[p, 1:, 0] + N.roll(B1in, 1, axis = 1)[p, 1:, 0]) \
                                      + hr2d[p, :-1, 0, 1] * B2in[p, 1:-1, 0]
    # Top edge
    Brd[p, 1:-1, -1] = hrrd[p, :-1, -1, 1] * Brin[p, 1:-1, -1] \
                                      + 0.5 * hr1d[p, :-1, -1, 1] * (B1in[p, 1:, -1] + N.roll(B1in, 1, axis = 1)[p, 1:, -1]) \
                                      + hr2d[p, :-1, -1, 1] * B2in[p, 1:-1, -1]
                                      
    # Bottom-left corner
    Brd[p, 0, 0] = hrrd[p, 0, 0, 0] * Brin[p, 0, 0] \
                                + hr1d[p, 0, 0, 0] * B1in[p, 0, 0] \
                                + hr2d[p, 0, 0, 0] * B2in[p, 0, 0]
    # Top-left corner
    Brd[p, 0, -1] = hrrd[p, 0, -1, 0] * Brin[p, 0, -1] \
                                 + hr1d[p, 0, -1, 0] * B1in[p, 0, -1] \
                                 + hr2d[p, 0, -1, 0] * B2in[p, 0, -1]
    # Bottom-right corner
    Brd[p, -1, 0] = hrrd[p, -1, 0, 0] * Brin[p, -1, 0] \
                                 + hr1d[p, -1, 0, 0] * B1in[p, -1, 0] \
                                 + hr2d[p, -1, 0, 0] * B2in[p, -1, 0]
    # Top-right corner
    Brd[p, -1, -1] = hrrd[p, -1, -1, 0] * Brin[p, -1, -1] \
                                  + hr1d[p, -1, -1, 0] * B1in[p, -1, -1] \
                                  + hr2d[p, -1, -1, 0] * B2in[p, -1, -1]

    ########
    # Bxi
    ########

    # Interior
    B1d[p, 1:-1, 1:-1] = h11d[p, 1:-1, :-1, 2] * B1in[p, 1:-1, 1:-1] \
                                      + 0.25 * h12d[p, 1:-1, :-1, 2] * (B2in[p, 1:-2, 1:] + N.roll(N.roll(B2in, -1, axis = 1), 1, axis = 2)[p, 1:-2, 1:] \
                                                                                    +  N.roll(B2in, -1, axis = 1)[p, 1:-2, 1:] + N.roll(B2in, 1, axis = 2)[p, 1:-2, 1:]) \
                                      + 0.5 * hr1d[p, 1:-1, :-1, 2] * (Brin[p, 1:-2, 1:-1] + N.roll(Brin, -1, axis = 1)[p, 1:-2, 1:-1])
    # Left edge
    B1d[p, 0, 1:-1] = h11d[p, 0, :-1, 2] * B1in[p, 0, 1:-1] \
                                   + 0.5 * h12d[p, 0, :-1, 2] * (B2in[p, 0, 1:] + N.roll(B2in, 1, axis = 2)[p, 0, 1:]) \
                                   + hr1d[p, 0, :-1, 2] * Brin[p, 0, 1:-1]
    # Right edge
    B1d[p, -1, 1:-1] = h11d[p, -1, :-1, 2] * B1in[p, -1, 1:-1] \
                                   + 0.5 * h12d[p, -1, :-1, 2] * (B2in[p, -1, 1:] + N.roll(B2in, 1, axis = 2)[p, -1, 1:]) \
                                   + hr1d[p, -1, :-1, 2] * Brin[p, -1, 1:-1]
    # Bottom edge
    B1d[p, 1:-1, 0] = h11d[p, 1:-1, 0, 0] * B1in[p, 1:-1, 0] \
                                   + 0.5 * h12d[p, 1:-1, 0, 0] * (B2in[p, 1:-2, 0] + N.roll(B2in, -1, axis = 1)[p, 1:-2, 0]) \
                                   + 0.5 * hr1d[p, 1:-1, 0, 0] * (Brin[p, 1:-2, 0] + N.roll(Brin, -1, axis = 1)[p, 1:-2, 0])
    # Top edge
    B1d[p, 1:-1, -1] = h11d[p, 1:-1, -1, 0] * B1in[p, 1:-1, -1] \
                                    + 0.5 * h12d[p, 1:-1, -1, 0] * (B2in[p, 1:-2, -1] + N.roll(B2in, -1, axis = 1)[p, 1:-2, -1]) \
                                    + 0.5 * hr1d[p, 1:-1, -1, 0] * (Brin[p, 1:-2, -1] + N.roll(Brin, -1, axis = 1)[p, 1:-2, -1])

    # Bottom-left corner
    B1d[p, 0, 0] = h11d[p, 0, 0, 0] * B1in[p, 0, 0] \
                                + h12d[p, 0, 0, 0] * B2in[p, 0, 0] \
                                + hr1d[p, 0, 0, 0] * Brin[p, 0, 0]
    # Top-left corner
    B1d[p, 0, -1] = h11d[p, 0, -1, 0] * B1in[p, 0, -1] \
                                 + h12d[p, 0, -1, 0] * B2in[p, 0, -1] \
                                 + hr1d[p, 0, -1, 0] * Brin[p, 0, -1]
    # Bottom-right corner
    B1d[p, -1, 0] = h11d[p, -1, 0, 0] * B1in[p, -1, 0] \
                                 + h12d[p, -1, 0, 0] * B2in[p, -1, 0] \
                                 + hr1d[p, -1, 0, 0] * Brin[p, -1, 0]
    # Top-right corner
    B1d[p, -1, -1] = h11d[p, -1, -1, 0] * B1in[p, -1, -1] \
                                  + h12d[p, -1, -1, 0] * B2in[p, -1, -1] \
                                  + hr1d[p, -1, -1, 0] * Brin[p, -1, -1]

    ########
    # Beta
    ########

    # Interior
    B2d[p, 1:-1, 1:-1] = h22d[p, :-1, 1:-1, 1] * B2in[p, 1:-1, 1:-1] \
                                      + 0.25 * h12d[p, :-1, 1:-1, 1] * (B1in[p, 1:, 1:-2] + N.roll(N.roll(B1in, 1, axis = 1), -1, axis = 2)[p, 1:, 1:-2] \
                                                                                     + N.roll(B1in, 1, axis = 1)[p, 1:, 1:-2] + N.roll(B1in, -1, axis = 2)[p, 1:, 1:-2]) \
                                      + 0.5 * hr2d[p, :-1, 1:-1, 1] * (Brin[p, 1:-1, 1:-2] + N.roll(Brin, -1, axis = 2)[p, 1:-1, 1:-2])
    # Left edge
    B2d[p, 0, 1:-1] = h22d[p, 0, 1:-1, 0] * B2in[p, 0, 1:-1] \
                                   + 0.5 * h12d[p, 0, 1:-1, 0] * (B1in[p, 0, 1:-2] + N.roll(B1in, -1, axis = 2)[p, 0, 1:-2]) \
                                   + 0.5 * hr2d[p, 0, 1:-1, 0] * (Brin[p, 0, 1:-2] + N.roll(Brin, -1, axis = 2)[p, 0, 1:-2])
    # Right edge
    B2d[p, -1, 1:-1] = h22d[p, -1, 1:-1, 0] * B2in[p, -1, 1:-1] \
                                    + 0.5 * h12d[p, -1, 1:-1, 0] * (B1in[p, -1, 1:-2] + N.roll(B1in, -1, axis = 2)[p, -1, 1:-2]) \
                                    + 0.5 * hr2d[p, -1, 1:-1, 0] * (Brin[p, -1, 1:-2] + N.roll(Brin, -1, axis = 2)[p, -1, 1:-2])
    # Bottom edge
    B2d[p, 1:-1, 0] = h22d[p, :-1, 0, 1] * B2in[p, 1:-1, 0] \
                                   + 0.5 * h12d[p, :-1, 0, 1] * (B1in[p, 1:, 0] + N.roll(B1in, 1, axis = 1)[p, 1:, 0]) \
                                   + hr2d[p, :-1, 0, 1] * Brin[p, 1:-1, 0]
    # Top edge
    B2d[p, 1:-1, -1] = h22d[p, :-1, -1, 1] * B2in[p, 1:-1, -1] \
                                   + 0.5 * h12d[p, :-1, -1, 1] * (B1in[p, 1:, -1] + N.roll(B1in, 1, axis = 1)[p, 1:, -1]) \
                                   + hr2d[p, :-1, -1, 1] * Brin[p, 1:-1, -1]
    # Bottom-left corner
    B2d[p, 0, 0] = h22d[p, 0, 0, 0] * B2in[p, 0, 0] \
                                + h12d[p, 0, 0, 0] * B1in[p, 0, 0] \
                                + hr2d[p, 0, 0, 0] * Brin[p, 0, 0]
    # Top-left corner
    B2d[p, 0, -1] = h22d[p, 0, -1, 0] * B2in[p, 0, -1] \
                                 + h12d[p, 0, -1, 0] * B1in[p, 0, -1] \
                                 + hr2d[p, 0, -1, 0] * Brin[p, 0, -1]
    # Bottom-right corner
    B2d[p, -1, 0] = h22d[p, -1, 0, 0] * B2in[p, -1, 0] \
                                 + h12d[p, -1, 0, 0] * B1in[p, -1, 0] \
                                 + hr2d[p, -1, 0, 0] * Brin[p, -1, 0]
    # Top-right corner
    B2d[p, -1, -1] = h22d[p, -1, -1, 0] * B2in[p, -1, -1] \
                                  + h12d[p, -1, -1, 0] * B1in[p, -1, -1] \
                                  + hr2d[p, -1, -1, 0] * Brin[p, -1, -1]

def compute_H_aux(p, Drin, D1in, D2in, Brin, B1in, B2in):

    ##### Hr
    # Interior
    Hrd[p, 1:-1, 1:-1] = alpha[p, :-1, :-1, 3] * Brin[p, 1:-1, 1:-1]
    # Left edge
    Hrd[p, 0, 1:-1] = alpha[p, 0, :-1, 2] * Brin[p, 0, 1:-1]
    # Right edge
    Hrd[p, -1, 1:-1] = alpha[p, -1, :-1, 2] * Brin[p, -1, 1:-1]
    # Bottom edge
    Hrd[p, 1:-1, 0] = alpha[p, :-1, 0, 1] * Brin[p, 1:-1, 0]
    # Top edge
    Hrd[p, 1:-1, -1] = alpha[p, :-1, -1, 1] * Brin[p, 1:-1, -1]
    # Corners
    Hrd[p, 0, 0]   = alpha[p, 0, 0, 0]  * Brin[p, 0, 0]
    Hrd[p, -1, 0]  = alpha[p, -1, 0, 0] * Brin[p, -1, 0]
    Hrd[p, 0, -1]  = alpha[p, 0, -1, 0] * Brin[p, 0, -1]
    Hrd[p, -1, -1] = alpha[p, -1, -1, 0]* Brin[p, -1, -1]
    
    ##### Hxi
    # Interior
    H1d[p, :, 1:-1] = alpha[p, :, :-1, 2] * B1in[p, :, 1:-1] \
                    + sqrt_det_h[p, :, :-1, 2] * beta[p, :, :-1, 2] * D2in[p, :, 1:-1]
    # Bottom edge
    H1d[p, :, 0] = alpha[p, :, 0, 0] * B1in[p, :, 0] \
                 + sqrt_det_h[p, :, 0, 0] * beta[p, :, 0, 0] * D2in[p, :, 0]
    # Top edge
    H1d[p, :, -1] = alpha[p, :, -1, 0] * B1in[p, :, -1] \
                  + sqrt_det_h[p, :, -1, 0] * beta[p, :, -1, 0] * D2in[p, :, -1]

    ##### Heta
    ##### Interior
    H2d[p, 1:-1, :] = alpha[p, :-1, :, 1] * B2in[p, 1:-1, :] \
                    - sqrt_det_h[p, :-1, :, 1] * beta[p, :-1, :, 1] * D1in[p, 1:-1, :]
    ##### Left edge
    H2d[p, 0, :] = alpha[p, 0, :, 0] * B2in[p, 0, :] \
                 - sqrt_det_h[p, 0, :, 0] * beta[p, 0, :, 0] * D1in[p, 0, :]
    ##### Right edge
    H2d[p, -1, :] = alpha[p, -1, :, 0] * B2in[p, -1, :] \
                  - sqrt_det_h[p, -1, :, 0] * beta[p, -1, :, 0] * D1in[p, -1, :]

def cov_to_contra_H(p, Brin, B1in, B2in):

    ########
    # Br
    ########

    # Interior
    Hru[p, 1:-1, 1:-1] = hrru[p, :-1, :-1, 3] * Brin[p, 1:-1, 1:-1] \
                                      + 0.5 * hr1u[p, :-1, :-1, 3] * (B1in[p, 1:, 1:-1] + N.roll(B1in, 1, axis = 1)[p, 1:, 1:-1]) \
                                      + 0.5 * hr2u[p, :-1, :-1, 3] * (B2in[p, 1:-1, 1:] + N.roll(B2in, 1, axis = 2)[p, 1:-1, 1:])

    # Left edge
    Hru[p, 0, 1:-1] = hrru[p, 0, :-1, 2] * Brin[p, 0, 1:-1] \
                                      + hr1u[p, 0, :-1, 2] * B1in[p, 0, 1:-1] \
                                      + 0.5 * hr2u[p, 0, :-1, 2] * (B2in[p, 0, 1:] + N.roll(B2in, 1, axis = 2)[p, 0, 1:])

    # Right edge
    Hru[p, -1, 1:-1] = hrru[p, -1, :-1, 2] * Brin[p, -1, 1:-1] \
                                      + hr1u[p, -1, :-1, 2] * B1in[p, -1, 1:-1] \
                                      + 0.5 * hr2u[p, -1, :-1, 2] * (B2in[p, -1, 1:] + N.roll(B2in, 1, axis = 2)[p, -1, 1:])
    # Bottom edge
    Hru[p, 1:-1, 0] = hrru[p, :-1, 0, 1] * Brin[p, 1:-1, 0] \
                                      + 0.5 * hr1u[p, :-1, 0, 1] * (B1in[p, 1:, 0] + N.roll(B1in, 1, axis = 1)[p, 1:, 0]) \
                                      + hr2u[p, :-1, 0, 1] * B2in[p, 1:-1, 0]
    # Top edge
    Hru[p, 1:-1, -1] = hrru[p, :-1, -1, 1] * Brin[p, 1:-1, -1] \
                                      + 0.5 * hr1u[p, :-1, -1, 1] * (B1in[p, 1:, -1] + N.roll(B1in, 1, axis = 1)[p, 1:, -1]) \
                                      + hr2u[p, :-1, -1, 1] * B2in[p, 1:-1, -1]
                                      
    # Bottom-left corner
    Hru[p, 0, 0] = hrru[p, 0, 0, 0] * Brin[p, 0, 0] \
                                + hr1u[p, 0, 0, 0] * B1in[p, 0, 0] \
                                + hr2u[p, 0, 0, 0] * B2in[p, 0, 0]
                                
    # Top-left corner
    Hru[p, 0, -1] = hrru[p, 0, -1, 0] * Brin[p, 0, -1] \
                                 + hr1u[p, 0, -1, 0] * B1in[p, 0, -1] \
                                 + hr2u[p, 0, -1, 0] * B2in[p, 0, -1]
    # Bottom-right corner
    Hru[p, -1, 0] = hrru[p, -1, 0, 0] * Brin[p, -1, 0] \
                                 + hr1u[p, -1, 0, 0] * B1in[p, -1, 0] \
                                 + hr2u[p, -1, 0, 0] * B2in[p, -1, 0]
    # Top-right corner
    Hru[p, -1, -1] = hrru[p, -1, -1, 0] * Brin[p, -1, -1] \
                                  + hr1u[p, -1, -1, 0] * B1in[p, -1, -1] \
                                  + hr2u[p, -1, -1, 0] * B2in[p, -1, -1]

    ########
    # Bxi
    ########

    # Interior
    H1u[p, 1:-1, 1:-1] = h11u[p, 1:-1, :-1, 2] * B1in[p, 1:-1, 1:-1] \
                                      + 0.25 * h12u[p, 1:-1, :-1, 2] * (B2in[p, 1:-2, 1:] + N.roll(N.roll(B2in, -1, axis = 1), 1, axis = 2)[p, 1:-2, 1:] \
                                                                                    +  N.roll(B2in, -1, axis = 1)[p, 1:-2, 1:] + N.roll(B2in, 1, axis = 2)[p, 1:-2, 1:]) \
                                      + 0.5 * hr1u[p, 1:-1, :-1, 2] * (Brin[p, 1:-2, 1:-1] + N.roll(Brin, -1, axis = 1)[p, 1:-2, 1:-1])
    # Left edge
    H1u[p, 0, 1:-1] = h11u[p, 0, :-1, 2] * B1in[p, 0, 1:-1] \
                                   + 0.5 * h12u[p, 0, :-1, 2] * (B2in[p, 0, 1:] + N.roll(B2in, 1, axis = 2)[p, 0, 1:]) \
                                   + hr1u[p, 0, :-1, 2] * Brin[p, 0, 1:-1]
    # Right edge
    H1u[p, -1, 1:-1] = h11u[p, -1, :-1, 2] * B1in[p, -1, 1:-1] \
                                   + 0.5 * h12u[p, -1, :-1, 2] * (B2in[p, -1, 1:] + N.roll(B2in, 1, axis = 2)[p, -1, 1:]) \
                                   + hr1u[p, -1, :-1, 2] * Brin[p, -1, 1:-1]
    # Bottom edge
    H1u[p, 1:-1, 0] = h11u[p, 1:-1, 0, 0] * B1in[p, 1:-1, 0] \
                                   + 0.5 * h12u[p, 1:-1, 0, 0] * (B2in[p, 1:-2, 0] + N.roll(B2in, -1, axis = 1)[p, 1:-2, 0]) \
                                   + 0.5 * hr1u[p, 1:-1, 0, 0] * (Brin[p, 1:-2, 0] + N.roll(Brin, -1, axis = 1)[p, 1:-2, 0])
    # Top edge
    H1u[p, 1:-1, -1] = h11u[p, 1:-1, -1, 0] * B1in[p, 1:-1, -1] \
                                    + 0.5 * h12u[p, 1:-1, -1, 0] * (B2in[p, 1:-2, -1] + N.roll(B2in, -1, axis = 1)[p, 1:-2, -1]) \
                                    + 0.5 * hr1u[p, 1:-1, -1, 0] * (Brin[p, 1:-2, -1] + N.roll(Brin, -1, axis = 1)[p, 1:-2, -1])

    # Bottom-left corner
    H1u[p, 0, 0] = h11u[p, 0, 0, 0] * B1in[p, 0, 0] \
                                + h12u[p, 0, 0, 0] * B2in[p, 0, 0] \
                                + hr1u[p, 0, 0, 0] * Brin[p, 0, 0]
    # Top-left corner
    H1u[p, 0, -1] = h11u[p, 0, -1, 0] * B1in[p, 0, -1] \
                                 + h12u[p, 0, -1, 0] * B2in[p, 0, -1] \
                                 + hr1u[p, 0, -1, 0] * Brin[p, 0, -1]
    # Bottom-right corner
    H1u[p, -1, 0] = h11u[p, -1, 0, 0] * B1in[p, -1, 0] \
                                 + h12u[p, -1, 0, 0] * B2in[p, -1, 0] \
                                 + hr1u[p, -1, 0, 0] * Brin[p, -1, 0]
    # Top-right corner
    H1u[p, -1, -1] = h11u[p, -1, -1, 0] * B1in[p, -1, -1] \
                                  + h12u[p, -1, -1, 0] * B2in[p, -1, -1] \
                                  + hr1u[p, -1, -1, 0] * Brin[p, -1, -1]

    ########
    # Beta
    ########

    # Interior
    H2u[p, 1:-1, 1:-1] = h22u[p, :-1, 1:-1, 1] * B2in[p, 1:-1, 1:-1] \
                                      + 0.25 * h12u[p, :-1, 1:-1, 1] * (B1in[p, 1:, 1:-2] + N.roll(N.roll(B1in, 1, axis = 1), -1, axis = 2)[p, 1:, 1:-2] \
                                                                                     + N.roll(B1in, 1, axis = 1)[p, 1:, 1:-2] + N.roll(B1in, -1, axis = 2)[p, 1:, 1:-2]) \
                                      + 0.5 * hr2u[p, :-1, 1:-1, 1] * (Brin[p, 1:-1, 1:-2] + N.roll(Brin, -1, axis = 2)[p, 1:-1, 1:-2])
    # Left edge
    H2u[p, 0, 1:-1] = h22u[p, 0, 1:-1, 0] * B2in[p, 0, 1:-1] \
                                   + 0.5 * h12u[p, 0, 1:-1, 0] * (B1in[p, 0, 1:-2] + N.roll(B1in, -1, axis = 2)[p, 0, 1:-2]) \
                                   + 0.5 * hr2u[p, 0, 1:-1, 0] * (Brin[p, 0, 1:-2] + N.roll(Brin, -1, axis = 2)[p, 0, 1:-2])
    # Right edge
    H2u[p, -1, 1:-1] = h22u[p, -1, 1:-1, 0] * B2in[p, -1, 1:-1] \
                                    + 0.5 * h12u[p, -1, 1:-1, 0] * (B1in[p, -1, 1:-2] + N.roll(B1in, -1, axis = 2)[p, -1, 1:-2]) \
                                    + 0.5 * hr2u[p, -1, 1:-1, 0] * (Brin[p, -1, 1:-2] + N.roll(Brin, -1, axis = 2)[p, -1, 1:-2])
    # Bottom edge
    H2u[p, 1:-1, 0] = h22u[p, :-1, 0, 1] * B2in[p, 1:-1, 0] \
                                   + 0.5 * h12u[p, :-1, 0, 1] * (B1in[p, 1:, 0] + N.roll(B1in, 1, axis = 1)[p, 1:, 0]) \
                                   + hr2u[p, :-1, 0, 1] * Brin[p, 1:-1, 0]
    # Top edge
    H2u[p, 1:-1, -1] = h22u[p, :-1, -1, 1] * B2in[p, 1:-1, -1] \
                                   + 0.5 * h12u[p, :-1, -1, 1] * (B1in[p, 1:, -1] + N.roll(B1in, 1, axis = 1)[p, 1:, -1]) \
                                   + hr2u[p, :-1, -1, 1] * Brin[p, 1:-1, -1]
    # Bottom-left corner
    H2u[p, 0, 0] = h22u[p, 0, 0, 0] * B2in[p, 0, 0] \
                                + h12u[p, 0, 0, 0] * B1in[p, 0, 0] \
                                + hr2u[p, 0, 0, 0] * Brin[p, 0, 0]
    # Top-left corner
    H2u[p, 0, -1] = h22u[p, 0, -1, 0] * B2in[p, 0, -1] \
                                 + h12u[p, 0, -1, 0] * B1in[p, 0, -1] \
                                 + hr2u[p, 0, -1, 0] * Brin[p, 0, -1]
    # Bottom-right corner
    H2u[p, -1, 0] = h22u[p, -1, 0, 0] * B2in[p, -1, 0] \
                                 + h12u[p, -1, 0, 0] * B1in[p, -1, 0] \
                                 + hr2u[p, -1, 0, 0] * Brin[p, -1, 0]
    # Top-right corner
    H2u[p, -1, -1] = h22u[p, -1, -1, 0] * B2in[p, -1, -1] \
                                  + h12u[p, -1, -1, 0] * B1in[p, -1, -1] \
                                  + hr2u[p, -1, -1, 0] * Brin[p, -1, -1]


########
# Compute interface terms
########

sig_in  = 1.0
sig_cor = 1.0

def compute_penalty_D(p0, p1, dtin, Drin, D1in, D2in, Brin, B1in, B2in):

    top = topology[p0, p1]
    
    if (top == 'xx'):

        #######
        # Dr
        #######

        # lambda_0 = alpha_int[p0, :, loc.right] * N.sqrt(h11u_int[p0, :, loc.right]) * sqrt_det_h_int[p0, :, loc.right]
        # lambda_1 = alpha_int[p1, :, loc.left]  * N.sqrt(h11u_int[p1, :, loc.left])  * sqrt_det_h_int[p1, :, loc.left]
        
        lambda_0 = alpha_int[p0, :, loc.right] * N.sqrt(h11u_int[p0, :, loc.right])
        lambda_1 = alpha_int[p1, :, loc.left]  * N.sqrt(h11u_int[p1, :, loc.left]) 
        
        Dr_0 = Drin[p0, -1, :]
        D1_0 = D1in[p0, -1, :]
        B2_0 = B2in[p0, -1, :]

        Dr_1 = Drin[p1, 0, :]
        D1_1, D2_1 = transform_vect(p1, p0, xi_int[0], eta_int[:], D1in[p1, 0, :], interp(D2in[p1, 0, :], eta_half, eta_int))
        B2_1 = B2in[p1, 0, :]

        carac_0 = (Dr_0 - hr1u_int[p0, :, loc.right] / h11u_int[p0, :, loc.right] * D1_0 + B2_0 / N.sqrt(sqrt_det_h_int**2 * h11u_int)[p0, :, loc.right])
        carac_1 = (Dr_1 - hr1u_int[p0, :, loc.right] / h11u_int[p0, :, loc.right] * D1_1 + B2_1 / N.sqrt(sqrt_det_h_int**2 * h11u_int)[p0, :, loc.right])

        diff_Dru[p0, -1, :] += dtin * sig_in * 0.5 * (carac_0 - carac_1) * lambda_0 / dxi / P_int_2[0]

        Dr_0 = Drin[p0, -1, :]
        D1_0, D2_0 = transform_vect(p0, p1, xi_int[-1], eta_int[:], D1in[p0, -1, :], interp(D2in[p0, -1, :], eta_half, eta_int))
        B2_0 = B2in[p0, -1, :]

        Dr_1 = Drin[p1, 0, :]
        D1_1 = D1in[p1, 0, :]
        B2_1 = B2in[p1, 0, :]
        
        carac_1 = (Dr_1 - hr1u_int[p1, :, loc.left] / h11u_int[p1, :, loc.left] * D1_1 - B2_1 / N.sqrt(sqrt_det_h_int**2 * h11u_int)[p1, :, loc.left])
        carac_0 = (Dr_0 - hr1u_int[p1, :, loc.left] / h11u_int[p1, :, loc.left] * D1_0 - B2_0 / N.sqrt(sqrt_det_h_int**2 * h11u_int)[p1, :, loc.left])
        
        diff_Dru[p1, 0, :]  += dtin * sig_in * 0.5 * (carac_1 - carac_0) * lambda_1 / dxi / P_int_2[0]

        #######
        # D2
        #######

        # lambda_0 = alpha_half[p0, :, loc.right] * N.sqrt(h11u_half[p0, :, loc.right]) * sqrt_det_h_half[p0, :, loc.right]
        # lambda_1 = alpha_half[p1, :, loc.left]  * N.sqrt(h11u_half[p1, :, loc.left])  * sqrt_det_h_half[p1, :, loc.left] 

        lambda_0 = alpha_half[p0, :, loc.right] * N.sqrt(h11u_half[p0, :, loc.right])
        lambda_1 = alpha_half[p1, :, loc.left]  * N.sqrt(h11u_half[p1, :, loc.left]) 
        
        D1_0 = interp(D1in[p0, -1, :], eta_int, eta_half)
        D2_0 = D2in[p0, -1, :]
        Br_0 = Brin[p0, -1, :]
        
        D1_1, D2_1 = transform_vect(p1, p0, xi_half[0], eta_half[:], interp(D1in[p1, 0, :], eta_int, eta_half), D2in[p1, 0, :])
        Br_1 = Brin[p1, 0, :]
        
        carac_0 = (D2_0 - h12u_half[p0, :, loc.right] / h11u_half[p0, :, loc.right] * D1_0 - Br_0 / N.sqrt(sqrt_det_h_half**2 * h11u_half)[p0, :, loc.right])
        carac_1 = (D2_1 - h12u_half[p0, :, loc.right] / h11u_half[p0, :, loc.right] * D1_1 - Br_1 / N.sqrt(sqrt_det_h_half**2 * h11u_half)[p0, :, loc.right])

        diff_D2u[p0, -1, :] += dtin * sig_in * 0.5 * (carac_0 - carac_1) * lambda_0 / dxi / P_half_2[0]

        D1_0, D2_0 = transform_vect(p0, p1, xi_half[-1], eta_half[:], interp(D1in[p0, -1, :], eta_int, eta_half), D2in[p0, -1, :])
        Br_0 = Brin[p0, -1, :]

        D1_1 = interp(D1in[p1, 0, :], eta_int, eta_half)
        D2_1 = D2in[p1, 0, :]
        Br_1 = Brin[p1, 0, :]        

        carac_1 = (D2_1 - h12u_half[p1, :, loc.left] / h11u_half[p1, :, loc.left] * D1_1 + Br_1 / N.sqrt(sqrt_det_h_half**2 * h11u_half)[p1, :, loc.left])
        carac_0 = (D2_0 - h12u_half[p1, :, loc.left] / h11u_half[p1, :, loc.left] * D1_0 + Br_0 / N.sqrt(sqrt_det_h_half**2 * h11u_half)[p1, :, loc.left])
        
        diff_D2u[p1, 0, :]  += dtin * sig_in * 0.5 * (carac_1 - carac_0) * lambda_1 / dxi / P_half_2[0]

    if (top == 'xy'):

        #######
        # Dr
        #######

        # lambda_0 = alpha_int[p0, :, loc.right]  * N.sqrt(h11u_int[p0, :, loc.right])  * sqrt_det_h_int[p0, :, loc.right] 
        # lambda_1 = alpha_int[p1, :, loc.bottom] * N.sqrt(h22u_int[p1, :, loc.bottom]) * sqrt_det_h_int[p1, :, loc.bottom]

        lambda_0 = alpha_int[p0, :, loc.right]  * N.sqrt(h11u_int[p0, :, loc.right]) 
        lambda_1 = alpha_int[p1, :, loc.bottom] * N.sqrt(h22u_int[p1, :, loc.bottom])

        Dr_0 = Drin[p0, -1, :]
        D1_0 = D1in[p0, -1, :]
        B2_0 = B2in[p0, -1, :]

        Dr_1 = Drin[p1, :, 0]
        D1_1, D2_1 = transform_vect(p1, p0, xi_int[:], eta_int[0], interp(D1in[p1, :, 0], xi_half, xi_int), D2in[p1, :, 0])
        B1_1 = B1in[p1, :, 0]
        
        carac_0 = (Dr_0       - hr1u_int[p0, :, loc.right] / h11u_int[p0, :, loc.right] * D1_0       + B2_0       / N.sqrt(sqrt_det_h_int**2 * h11u_int)[p0, :, loc.right])
        carac_1 = (Dr_1[::-1] - hr1u_int[p0, :, loc.right] / h11u_int[p0, :, loc.right] * D1_1[::-1] - B1_1[::-1] / N.sqrt(sqrt_det_h_int**2 * h11u_int)[p0, :, loc.right])
        
        diff_Dru[p0, -1, :] += dtin * sig_in * 0.5 * (carac_0 - carac_1) * lambda_0 / dxi / P_int_2[0]

        Dr_0 = Drin[p0, -1, :]
        D1_0, D2_0 = transform_vect(p0, p1, xi_int[-1], eta_int[:], D1in[p0, -1, :], interp(D2in[p0, -1, :], eta_half, eta_int))
        B2_0 = B2in[p0, -1, :]

        Dr_1 = Drin[p1, :, 0]
        D2_1 = D2in[p1, :, 0]
        B1_1 = B1in[p1, :, 0]
        
        carac_1 = (Dr_1       - hr2u_int[p1, :, loc.bottom] / h22u_int[p1, :, loc.bottom] * D2_1       + B1_1       / N.sqrt(sqrt_det_h_int**2 * h22u_int)[p1, :, loc.bottom])
        carac_0 = (Dr_0[::-1] - hr2u_int[p1, :, loc.bottom] / h22u_int[p1, :, loc.bottom] * D2_0[::-1] - B2_0[::-1] / N.sqrt(sqrt_det_h_int**2 * h22u_int)[p1, :, loc.bottom])
        
        diff_Dru[p1, :, 0]  += dtin * sig_in * 0.5 * (carac_1 - carac_0) * lambda_1 / dxi / P_int_2[0]

        #######
        # D1, D2
        #######

        # lambda_0 = alpha_half[p0, :, loc.right]  * N.sqrt(h11u_half[p0, :, loc.right])  * sqrt_det_h_half[p0, :, loc.right] 
        # lambda_1 = alpha_half[p1, :, loc.bottom] * N.sqrt(h22u_half[p1, :, loc.bottom]) * sqrt_det_h_half[p1, :, loc.bottom]

        lambda_0 = alpha_half[p0, :, loc.right]  * N.sqrt(h11u_half[p0, :, loc.right])  
        lambda_1 = alpha_half[p1, :, loc.bottom] * N.sqrt(h22u_half[p1, :, loc.bottom]) 

        D1_0 = interp(D1in[p0, -1, :], eta_int, eta_half)
        D2_0 = D2in[p0, -1, :]
        Br_0 = Brin[p0, -1, :]

        D1_1, D2_1 = transform_vect(p1, p0, xi_half[:], eta_half[0], D1in[p1, :, 0], interp(D2in[p1, :, 0], xi_int, xi_half))
        Br_1 = Brin[p1, :, 0]

        carac_0 = (D2_0       - h12u_half[p0, :, loc.right] / h11u_half[p0, :, loc.right] * D1_0       - Br_0       / N.sqrt(sqrt_det_h_half**2 * h11u_half)[p0, :, loc.right])
        carac_1 = (D2_1[::-1] - h12u_half[p0, :, loc.right] / h11u_half[p0, :, loc.right] * D1_1[::-1] - Br_1[::-1] / N.sqrt(sqrt_det_h_half**2 * h11u_half)[p0, :, loc.right])
        
        diff_D2u[p0, -1, :] += dtin * sig_in * 0.5 * (carac_0 - carac_1) * lambda_0 / dxi / P_half_2[0]

        D1_0, D2_0 = transform_vect(p0, p1, xi_half[-1], eta_half[:], interp(D1in[p0, -1, :], eta_int, eta_half), D2in[p0, -1, :])
        Br_0 = Brin[p0, -1, :]

        D1_1 = D1in[p1, :, 0]
        D2_1 = interp(D2in[p1, :, 0], xi_int, xi_half)
        Br_1 = Brin[p1, :, 0]
        
        carac_1 = (D1_1       - h12u_half[p1, :, loc.bottom] / h22u_half[p1, :, loc.bottom] * D2_1       - Br_1       / N.sqrt(sqrt_det_h_half**2 * h22u_half)[p1, :, loc.bottom])
        carac_0 = (D1_0[::-1] - h12u_half[p1, :, loc.bottom] / h22u_half[p1, :, loc.bottom] * D2_0[::-1] - Br_0[::-1] / N.sqrt(sqrt_det_h_half**2 * h22u_half)[p1, :, loc.bottom])
        
        diff_D1u[p1, :, 0]  += dtin * sig_in * 0.5 * (carac_1 - carac_0) * lambda_1 / dxi / P_half_2[0]

    if (top == 'yy'):

        #######
        # Dr
        #######

        # lambda_0 = alpha_int[p0, :, loc.top]    * N.sqrt(h22u_int[p0, :, loc.top])    * sqrt_det_h_int[p0, :, loc.top]
        # lambda_1 = alpha_int[p1, :, loc.bottom] * N.sqrt(h22u_int[p1, :, loc.bottom]) * sqrt_det_h_int[p1, :, loc.bottom]

        lambda_0 = alpha_int[p0, :, loc.top]    * N.sqrt(h22u_int[p0, :, loc.top])   
        lambda_1 = alpha_int[p1, :, loc.bottom] * N.sqrt(h22u_int[p1, :, loc.bottom])

        Dr_0 = Drin[p0, :, -1]
        D2_0 = D2in[p0, :, -1]
        B1_0 = B1in[p0, :, -1]
        
        Dr_1 = Drin[p1, :, 0]
        D1_1, D2_1 = transform_vect(p1, p0, xi_int, eta_int[0], interp(D1in[p1, :, 0], xi_half, xi_int), D2in[p1, :, 0])
        B1_1 = B1in[p1, :, 0]

        carac_0 = (Dr_0 - hr2u_int[p0, :, loc.top] / h22u_int[p0, :, loc.top] * D2_0 - B1_0 / N.sqrt(sqrt_det_h_int**2 * h22u_int)[p0, :, loc.top])
        carac_1 = (Dr_1 - hr2u_int[p0, :, loc.top] / h22u_int[p0, :, loc.top] * D2_1 - B1_1 / N.sqrt(sqrt_det_h_int**2 * h22u_int)[p0, :, loc.top])
        
        diff_Dru[p0, :, -1] += dtin * sig_in * 0.5 * (carac_0 - carac_1) * lambda_0 / dxi / P_int_2[0]

        Dr_0 = Drin[p0, :, -1]
        D1_0, D2_0 = transform_vect(p0, p1, xi_int, eta_int[-1], interp(D1in[p0, :, -1], xi_half, xi_int), D2in[p0, :, -1])
        B1_0 = B1in[p0, :, -1]
        
        Dr_1 = Drin[p1, :, 0]
        D2_1 = D2in[p1, :, 0]
        B1_1 = B1in[p1, :, 0]   

        carac_1 = (Dr_1 - hr2u_int[p1, :, loc.bottom] / h22u_int[p1, :, loc.bottom] * D2_1 + B1_1 / N.sqrt(sqrt_det_h_int**2 * h22u_int)[p1, :, loc.bottom])
        carac_0 = (Dr_0 - hr2u_int[p1, :, loc.bottom] / h22u_int[p1, :, loc.bottom] * D2_0 + B1_0 / N.sqrt(sqrt_det_h_int**2 * h22u_int)[p1, :, loc.bottom])    
    
        diff_Dru[p1, :, 0]  += dtin * sig_in * 0.5 * (carac_1 - carac_0) * lambda_1 / dxi / P_int_2[0]

        #######
        # D1
        #######

        # lambda_0 = alpha_half[p0, :, loc.top]    * N.sqrt(h22u_half[p0, :, loc.top])    * sqrt_det_h_half[p0, :, loc.top]
        # lambda_1 = alpha_half[p1, :, loc.bottom] * N.sqrt(h22u_half[p1, :, loc.bottom]) * sqrt_det_h_half[p1, :, loc.bottom]

        lambda_0 = alpha_half[p0, :, loc.top]    * N.sqrt(h22u_half[p0, :, loc.top])   
        lambda_1 = alpha_half[p1, :, loc.bottom] * N.sqrt(h22u_half[p1, :, loc.bottom])
        
        D1_0 = D1in[p0, :, -1]
        D2_0 = interp(D2in[p0, :, -1], xi_int, xi_half)
        Br_0 = Brin[p0, :, -1]

        D1_1, D2_1 = transform_vect(p1, p0, xi_half[:], eta_half[0], D1in[p1, :, 0], interp(D2in[p1, :, 0], xi_int, xi_half))
        Br_1 = Brin[p1, :, 0]
        
        carac_0 = (D1_0 - h12u_half[p0, :, loc.top] / h22u_half[p0, :, loc.top] * D2_0 + Br_0 / N.sqrt(sqrt_det_h_half**2 * h22u_half)[p0, :, loc.top])
        carac_1 = (D1_1 - h12u_half[p0, :, loc.top] / h22u_half[p0, :, loc.top] * D2_1 + Br_1 / N.sqrt(sqrt_det_h_half**2 * h22u_half)[p0, :, loc.top])
        
        diff_D1u[p0, :, -1] += dtin * sig_in * 0.5 * (carac_0 - carac_1) * lambda_0 / dxi / P_half_2[0]

        D1_0, D2_0 = transform_vect(p0, p1, xi_half[:], eta_half[-1], D1in[p0, :, -1], interp(D2in[p0, :, -1], xi_int, xi_half))
        Br_0 = Brin[p0, :, -1]

        D1_1 = D1in[p1, :, 0]
        D2_1 = interp(D2in[p1, :, 0], xi_int, xi_half)
        Br_1 = Brin[p1, :, 0]

        carac_1 = (D1_1 - h12u_half[p1, :, loc.bottom] / h22u_half[p1, :, loc.bottom] * D2_1 - Br_1 / N.sqrt(sqrt_det_h_half**2 * h22u_half)[p1, :, loc.bottom])
        carac_0 = (D1_0 - h12u_half[p1, :, loc.bottom] / h22u_half[p1, :, loc.bottom] * D2_0 - Br_0 / N.sqrt(sqrt_det_h_half**2 * h22u_half)[p1, :, loc.bottom]) 
    
        diff_D1u[p1, :, 0]  += dtin * sig_in * 0.5 * (carac_1 - carac_0) * lambda_1 / dxi / P_half_2[0]

    if (top == 'yx'):

        #######
        # Dr
        #######

        # lambda_0 = alpha_int[p0, :, loc.top]  * N.sqrt(h22u_int[p0, :, loc.top])  * sqrt_det_h_int[p0, :, loc.top]
        # lambda_1 = alpha_int[p1, :, loc.left] * N.sqrt(h11u_int[p1, :, loc.left]) * sqrt_det_h_int[p1, :, loc.left] 

        lambda_0 = alpha_int[p0, :, loc.top]  * N.sqrt(h22u_int[p0, :, loc.top]) 
        lambda_1 = alpha_int[p1, :, loc.left] * N.sqrt(h11u_int[p1, :, loc.left]) 

        Dr_0 = Drin[p0, :, -1]
        D2_0 = D2in[p0, :, -1]
        B1_0 = B1in[p0, :, -1]

        Dr_1 = Drin[p1, 0, :]
        D1_1, D2_1 = transform_vect(p1, p0, xi_int[0], eta_int, D1in[p1, 0, :], interp(D2in[p1, 0, :], eta_half, eta_int))
        B2_1 = B2in[p1, 0, :]

        carac_0 = (Dr_0       - hr2u_int[p0, :, loc.top] / h22u_int[p0, :, loc.top] * D2_0       - B1_0       / N.sqrt(sqrt_det_h_int**2 * h22u_int)[p0, :, loc.top])
        carac_1 = (Dr_1[::-1] - hr2u_int[p0, :, loc.top] / h22u_int[p0, :, loc.top] * D2_1[::-1] + B2_1[::-1] / N.sqrt(sqrt_det_h_int**2 * h22u_int)[p0, :, loc.top])

        diff_Dru[p0, :, -1] += dtin * sig_in * 0.5 * (carac_0 - carac_1) * lambda_0 / dxi / P_int_2[0]
        
        Dr_0 = Drin[p0, :, -1]
        D1_0, D2_0 = transform_vect(p0, p1, xi_int, eta_int[-1], interp(D1in[p0, :, -1], xi_half, xi_int), D2in[p0, :, -1])
        B1_0 = B1in[p0, :, -1]

        Dr_1 = Drin[p1, 0, :]
        D1_1 = D1in[p1, 0, :]
        B2_1 = B2in[p1, 0, :]        
    
        carac_1 = (Dr_1       - hr1u_int[p1, :, loc.left] / h11u_int[p1, :, loc.left] * D1_1       - B2_1       / N.sqrt(sqrt_det_h_int**2 * h11u_int)[p1, :, loc.left])
        carac_0 = (Dr_0[::-1] - hr1u_int[p1, :, loc.left] / h11u_int[p1, :, loc.left] * D1_0[::-1] + B1_0[::-1] / N.sqrt(sqrt_det_h_int**2 * h11u_int)[p1, :, loc.left])
        
        diff_Dru[p1, 0, :]  += dtin * sig_in * 0.5 * (carac_1 - carac_0) * lambda_1 / dxi / P_int_2[0]

        #######
        # D1, D2
        #######
        
        # lambda_0 = alpha_half[p0, :, loc.top]  * N.sqrt(h22u_half[p0, :, loc.top])  * sqrt_det_h_half[p0, :, loc.top]
        # lambda_1 = alpha_half[p1, :, loc.left] * N.sqrt(h11u_half[p1, :, loc.left]) * sqrt_det_h_half[p1, :, loc.left]

        lambda_0 = alpha_half[p0, :, loc.top]  * N.sqrt(h22u_half[p0, :, loc.top]) 
        lambda_1 = alpha_half[p1, :, loc.left] * N.sqrt(h11u_half[p1, :, loc.left])

        D1_0 = D1in[p0, :, -1]
        D2_0 = interp(D2in[p0, :, -1], xi_int, xi_half)
        Br_0 = Brin[p0, :, -1]

        D1_1, D2_1 = transform_vect(p1, p0, xi_half[0], eta_half, interp(D1in[p1, 0, :], eta_int, eta_half), D2in[p1, 0, :])
        Br_1 = Brin[p1, 0, :]

        carac_0 = (D1_0       - h12u_half[p0, :, loc.top] / h22u_half[p0, :, loc.top] * D2_0       + Br_0       / N.sqrt(sqrt_det_h_half**2 * h22u_half)[p0, :, loc.top])
        carac_1 = (D1_1[::-1] - h12u_half[p0, :, loc.top] / h22u_half[p0, :, loc.top] * D2_1[::-1] + Br_1[::-1] / N.sqrt(sqrt_det_h_half**2 * h22u_half)[p0, :, loc.top])

        diff_D1u[p0, :, -1] += dtin * sig_in * 0.5 * (carac_0 - carac_1) * lambda_0 / dxi / P_half_2[0]

        D1_0, D2_0 = transform_vect(p0, p1, xi_half, eta_half[-1], D1in[p0, :, -1], interp(D2in[p0, :, -1], xi_int, xi_half))
        Br_0 = Brin[p0, :, -1]

        D1_1 = interp(D1in[p1, 0, :], eta_int, eta_half)
        D2_1 = D2in[p1, 0, :]
        Br_1 = Brin[p1, 0, :]

        carac_1 = (D2_1       - h12u_half[p1, :, loc.left] / h11u_half[p1, :, loc.left] * D1_1       + Br_1       / N.sqrt(sqrt_det_h_half**2 * h11u_half)[p1, :, loc.left])
        carac_0 = (D2_0[::-1] - h12u_half[p1, :, loc.left] / h11u_half[p1, :, loc.left] * D1_0[::-1] + Br_0[::-1] / N.sqrt(sqrt_det_h_half**2 * h11u_half)[p1, :, loc.left])

        diff_D2u[p1, 0, :]  += dtin * sig_in * 0.5 * (carac_1 - carac_0) * lambda_1 / dxi / P_half_2[0]

#######
# PREVIOUS ATTEMPT
#######

# def compute_penalty_D(p0, p1, dtin, Drin, D1in, D2in, Brin, B1in, B2in):

#     top = topology[p0, p1]
    
#     if (top == 'xx'):

#         # Dr
#         lambda_0 = alpha_int[p0, :, loc.right] * N.sqrt(h11u_int[p0, :, loc.right]) * sqrt_det_h_int[p0, :, loc.right]
#         lambda_1 = alpha_int[p1, :, loc.left]  * N.sqrt(h11u_int[p1, :, loc.left])  * sqrt_det_h_int[p1, :, loc.left]

#         Dr_0 = Drin[p0, -1, :]
#         D1_0 = D1in[p0, -1, :]
#         B2_0 = B2in[p0, -1, :]

#         Dr_1 = Drin[p1, 0, :]
#         D1_1 = D1in[p1, 0, :]
#         B2_1 = B2in[p1, 0, :]

#         carac_0 = (Dr_0 - hr1u_int[p0, :, loc.right] / h11u_int[p0, :, loc.right] * D1_0 + B2_0 / N.sqrt(sqrt_det_h_int**2 * h11u_int)[p0, :, loc.right])
#         carac_1 = (Dr_1 - hr1u_int[p0, :, loc.right] / h11u_int[p0, :, loc.right] * D1_1 + B2_1 / N.sqrt(sqrt_det_h_int**2 * h11u_int)[p0, :, loc.right])

#         diff_Dru[p0, -1, :] += dtin * sig_in * 0.5 * (carac_0  * lambda_0 - carac_1 * lambda_1) / dxi / P_int_2[0]
        
#         carac_1 = (Dr_1 - hr1u_int[p1, :, loc.left]  / h11u_int[p1, :, loc.left]  * D1_1 - B2_1 / N.sqrt(sqrt_det_h_int**2 * h11u_int)[p1, :, loc.left])
#         carac_0 = (Dr_0 - hr1u_int[p0, :, loc.right] / h11u_int[p0, :, loc.right] * D1_0 - B2_0 / N.sqrt(sqrt_det_h_int**2 * h11u_int)[p0, :, loc.right])
        
#         diff_Dru[p1, 0, :]  += dtin * sig_in * 0.5 * (carac_1  * lambda_1 - carac_0 * lambda_0) / dxi / P_int_2[0]

#         # D2
#         lambda_0 = alpha_half[p0, :, loc.right] * N.sqrt(h11u_half[p0, :, loc.right]) * sqrt_det_h_half[p0, :, loc.right]
#         lambda_1 = alpha_half[p1, :, loc.left]  * N.sqrt(h11u_half[p1, :, loc.left])  * sqrt_det_h_half[p1, :, loc.left]

#         D1_0 = interp(D1in[p0, -1, :], eta_int, eta_half)
#         D2_0 = D2in[p0, -1, :]
#         Br_0 = Brin[p0, -1, :]

#         D1_1 = interp(D1in[p1, 0, :], eta_int, eta_half)
#         D2_1 = D2in[p1, 0, :]
#         Br_1 = Brin[p1, 0, :]
        
#         carac_0 = (D2_0 - h12u_half[p0, :, loc.right] / h11u_half[p0, :, loc.right] * D1_0 - Br_0 / N.sqrt(sqrt_det_h_half**2 * h11u_half)[p0, :, loc.right])
#         carac_1 = (D2_1 - h12u_half[p1, :, loc.left]  / h11u_half[p1, :, loc.left]  * D1_1 - Br_1 / N.sqrt(sqrt_det_h_half**2 * h11u_half)[p1, :, loc.left])

#         diff_D2u[p0, -1, :] += dtin * sig_in * 0.5 * (carac_0 * lambda_0 - carac_1 * lambda_1) / dxi / P_half_2[0]

#         carac_1 = (D2_1 - h12u_half[p1, :, loc.left]  / h11u_half[p1, :, loc.left]  * D1_1 + Br_1 / N.sqrt(sqrt_det_h_half**2 * h11u_half)[p1, :, loc.left])
#         carac_0 = (D2_0 - h12u_half[p0, :, loc.right] / h11u_half[p0, :, loc.right] * D1_0 + Br_0 / N.sqrt(sqrt_det_h_half**2 * h11u_half)[p0, :, loc.right])
        
#         diff_D2u[p1, 0, :]  += dtin * sig_in * 0.5 * (carac_1 * lambda_1 - carac_0 * lambda_0) / dxi / P_half_2[0]

#     if (top == 'xy'):

#         # Dr
#         lambda_0 = alpha_int[p0, :, loc.right]  * N.sqrt(h11u_int[p0, :, loc.right])  * sqrt_det_h_int[p0, :, loc.right]
#         lambda_1 = alpha_int[p1, :, loc.bottom] * N.sqrt(h22u_int[p1, :, loc.bottom]) * sqrt_det_h_int[p1, :, loc.bottom]

#         Dr_0 = Drin[p0, -1, :]
#         D1_0 = D1in[p0, -1, :]
#         B2_0 = B2in[p0, -1, :]

#         Dr_1 = Drin[p1, :, 0]
#         D2_1 = D2in[p1, :, 0]
#         B1_1 = B1in[p1, :, 0]
        
#         carac_0 = (Dr_0 - hr1u_int[p0, :, loc.right]  / h11u_int[p0, :, loc.right]  * D1_0 + B2_0 / N.sqrt(sqrt_det_h_int**2 * h11u_int)[p0, :, loc.right])
#         carac_1 = (Dr_1 - hr2u_int[p1, :, loc.bottom] / h22u_int[p1, :, loc.bottom] * D2_1 - B1_1 / N.sqrt(sqrt_det_h_int**2 * h22u_int)[p1, :, loc.bottom])
        
#         diff_Dru[p0, -1, :] += dtin * sig_in * 0.5 * (carac_0 * lambda_0 - carac_1[::-1] * lambda_1[::-1]) / dxi / P_int_2[0]
        
#         carac_1 = (Dr_1 - hr2u_int[p1, :, loc.bottom] / h22u_int[p1, :, loc.bottom] * D2_1 + B1_1 / N.sqrt(sqrt_det_h_int**2 * h22u_int)[p1, :, loc.bottom])
#         carac_0 = (Dr_0 - hr1u_int[p0, :, loc.right]  / h11u_int[p0, :, loc.right]  * D1_0 - B2_0 / N.sqrt(sqrt_det_h_int**2 * h11u_int)[p0, :, loc.right])
        
#         diff_Dru[p1, :, 0]  += dtin * sig_in * 0.5 * (carac_1 * lambda_1 - carac_0[::-1] * lambda_0[::-1]) / dxi / P_int_2[0]

#         # D1, D2
#         lambda_0 = alpha_half[p0, :, loc.right]  * N.sqrt(h11u_half[p0, :, loc.right])  * sqrt_det_h_half[p0, :, loc.right]
#         lambda_1 = alpha_half[p1, :, loc.bottom] * N.sqrt(h22u_half[p1, :, loc.bottom]) * sqrt_det_h_half[p1, :, loc.bottom]

#         D1_0 = interp(D1in[p0, -1, :], eta_int, eta_half)
#         D2_0 = D2in[p0, -1, :]
#         Br_0 = Brin[p0, -1, :]

#         D1_1 = D1in[p1, :, 0]
#         D2_1 = interp(D2in[p1, :, 0], xi_int, xi_half)
#         Br_1 = Brin[p1, :, 0]

#         carac_0 = (D2_0 - h12u_half[p0, :, loc.right]  / h11u_half[p0, :, loc.right]  * D1_0 - Br_0 / N.sqrt(sqrt_det_h_half**2 * h11u_half)[p0, :, loc.right])
#         carac_1 = (D1_1 - h12u_half[p1, :, loc.bottom] / h22u_half[p1, :, loc.bottom] * D2_1 + Br_1 / N.sqrt(sqrt_det_h_half**2 * h22u_half)[p1, :, loc.bottom])
        
#         diff_D2u[p0, -1, :] += dtin * sig_in * 0.5 * (carac_0 * lambda_0 + carac_1[::-1] * lambda_1[::-1]) / dxi / P_half_2[0]
        
#         carac_1 = (D1_1 - h12u_half[p1, :, loc.bottom] / h22u_half[p1, :, loc.bottom] * D2_1 - Br_1 / N.sqrt(sqrt_det_h_half**2 * h22u_half)[p1, :, loc.bottom])
#         carac_0 = (D2_0 - h12u_half[p0, :, loc.right]  / h11u_half[p0, :, loc.right]  * D1_0 + Br_0 / N.sqrt(sqrt_det_h_half**2 * h11u_half)[p0, :, loc.right])
        
#         diff_D1u[p1, :, 0]  += dtin * sig_in * 0.5 * (carac_1 * lambda_1 + carac_0[::-1] * lambda_0[::-1]) / dxi / P_half_2[0]
        
#     if (top == 'yy'):

#         # Dr
#         lambda_0 = alpha_int[p0, :, loc.top]    * N.sqrt(h22u_int[p0, :, loc.top])    * sqrt_det_h_int[p0, :, loc.top]
#         lambda_1 = alpha_int[p1, :, loc.bottom] * N.sqrt(h22u_int[p1, :, loc.bottom]) * sqrt_det_h_int[p1, :, loc.bottom]

#         Dr_0 = Drin[p0, :, -1]
#         D2_0 = D2in[p0, :, -1]
#         B1_0 = B1in[p0, :, -1]
        
#         Dr_1 = Drin[p1, :, 0]
#         D2_1 = D2in[p1, :, 0]
#         B1_1 = B1in[p1, :, 0]

#         carac_0 = (Dr_0 - hr2u_int[p0, :, loc.top]    / h22u_int[p0, :, loc.top]    * D2_0 - B1_0 / N.sqrt(sqrt_det_h_int**2 * h22u_int)[p0, :, loc.top])
#         carac_1 = (Dr_1 - hr2u_int[p1, :, loc.bottom] / h22u_int[p1, :, loc.bottom] * D2_1 - B1_1 / N.sqrt(sqrt_det_h_int**2 * h22u_int)[p1, :, loc.bottom])
        
#         diff_Dru[p0, :, -1] += dtin * sig_in * 0.5 * (carac_0 * lambda_0 - carac_1 * lambda_1) / dxi / P_int_2[0]
   
#         carac_1 = (Dr_1 - hr2u_int[p1, :, loc.bottom] / h22u_int[p1, :, loc.bottom] * D2_1 + B1_1 / N.sqrt(sqrt_det_h_int**2 * h22u_int)[p1, :, loc.bottom])
#         carac_0 = (Dr_0 - hr2u_int[p0, :, loc.top]    / h22u_int[p0, :, loc.top]    * D2_0 + B1_0 / N.sqrt(sqrt_det_h_int**2 * h22u_int)[p0, :, loc.top])    
    
#         diff_Dru[p1, :, 0]  += dtin * sig_in * 0.5 * (carac_1 * lambda_1 - carac_0 * lambda_0) / dxi / P_int_2[0]

#         # D1
#         lambda_0 = alpha_half[p0, :, loc.top]    * N.sqrt(h22u_half[p0, :, loc.top])    * sqrt_det_h_half[p0, :, loc.top]
#         lambda_1 = alpha_half[p1, :, loc.bottom] * N.sqrt(h22u_half[p1, :, loc.bottom]) * sqrt_det_h_half[p1, :, loc.bottom]

#         D1_0 = D1in[p0, :, -1]
#         D2_0 = interp(D2in[p0, :, -1], xi_int, xi_half)
#         Br_0 = Brin[p0, :, -1]

#         D1_1 = D1in[p1, :, 0]
#         D2_1 = interp(D2in[p1, :, 0], xi_int, xi_half)
#         Br_1 = Brin[p1, :, 0]
        
#         carac_0 = (D1_0 - h12u_half[p0, :, loc.top]    / h22u_half[p0, :, loc.top]    * D2_0 + Br_0 / N.sqrt(sqrt_det_h_half**2 * h22u_half)[p0, :, loc.top])
#         carac_1 = (D1_1 - h12u_half[p1, :, loc.bottom] / h22u_half[p1, :, loc.bottom] * D2_1 + Br_1 / N.sqrt(sqrt_det_h_half**2 * h22u_half)[p1, :, loc.bottom])
        
#         diff_D1u[p0, :, -1] += dtin * sig_in * 0.5 * (carac_0 * lambda_0 - carac_1 * lambda_1) / dxi / P_half_2[0]

#         carac_1 = (D1_1 - h12u_half[p1, :, loc.bottom] / h22u_half[p1, :, loc.bottom] * D2_1 - Br_1 / N.sqrt(sqrt_det_h_half**2 * h22u_half)[p1, :, loc.bottom])
#         carac_0 = (D1_0 - h12u_half[p0, :, loc.top]    / h22u_half[p0, :, loc.top]    * D2_0 - Br_0 / N.sqrt(sqrt_det_h_half**2 * h22u_half)[p0, :, loc.top]) 
    
#         diff_D1u[p1, :, 0]  += dtin * sig_in * 0.5 * (carac_1 * lambda_1 - carac_0 * lambda_0) / dxi / P_half_2[0]

#     if (top == 'yx'):

#         # Dr
#         lambda_0 = alpha_int[p0, :, loc.top]  * N.sqrt(h22u_int[p0, :, loc.top])  * sqrt_det_h_int[p0, :, loc.top]
#         lambda_1 = alpha_int[p1, :, loc.left] * N.sqrt(h11u_int[p1, :, loc.left]) * sqrt_det_h_int[p1, :, loc.left]
        
#         Dr_0 = Drin[p0, :, -1]
#         D2_0 = D2in[p0, :, -1]
#         B1_0 = B1in[p0, :, -1]

#         Dr_1 = Drin[p1, 0, :]
#         D1_1 = D1in[p1, 0, :]
#         B2_1 = B2in[p1, 0, :]        

#         carac_0 = (Dr_0 - hr2u_int[p0, :, loc.top]  / h22u_int[p0, :, loc.top]  * D2_0 - B1_0 / N.sqrt(sqrt_det_h_int**2 * h22u_int)[p0, :, loc.top])
#         carac_1 = (Dr_1 - hr1u_int[p1, :, loc.left] / h11u_int[p1, :, loc.left] * D1_1 + B2_1 / N.sqrt(sqrt_det_h_int**2 * h11u_int)[p1, :, loc.left])

#         diff_Dru[p0, :, -1] += dtin * sig_in * 0.5 * (carac_0 * lambda_0 - carac_1[::-1] * lambda_1[::-1]) / dxi / P_int_2[0]
                
#         carac_1 = (Dr_1 - hr1u_int[p1, :, loc.left] / h11u_int[p1, :, loc.left] * D1_1 - B2_1 / N.sqrt(sqrt_det_h_int**2 * h11u_int)[p1, :, loc.left])
#         carac_0 = (Dr_0 - hr2u_int[p0, :, loc.top]  / h22u_int[p0, :, loc.top]  * D2_0 + B1_0 / N.sqrt(sqrt_det_h_int**2 * h22u_int)[p0, :, loc.top])
        
#         diff_Dru[p1, 0, :]  += dtin * sig_in * 0.5 * (carac_1 * lambda_1 - carac_0[::-1] * lambda_0[::-1]) / dxi / P_int_2[0]

#         # D1, D2
#         lambda_0 = alpha_half[p0, :, loc.top]  * N.sqrt(h22u_half[p0, :, loc.top])  * sqrt_det_h_half[p0, :, loc.top]
#         lambda_1 = alpha_half[p1, :, loc.left] * N.sqrt(h11u_half[p1, :, loc.left]) * sqrt_det_h_half[p1, :, loc.left]

#         D1_0 = D1in[p0, :, -1]
#         D2_0 = interp(D2in[p0, :, -1], xi_int, xi_half)
#         Br_0 = Brin[p0, :, -1]

#         D1_1 = interp(D1in[p1, 0, :], eta_int, eta_half)
#         D2_1 = D2in[p1, 0, :]
#         Br_1 = Brin[p1, 0, :]

#         carac_0 = (D1_0 - h12u_half[p0, :, loc.top]  / h22u_half[p0, :, loc.top]  * D2_0 + Br_0 / N.sqrt(sqrt_det_h_half**2 * h22u_half)[p0, :, loc.top])
#         carac_1 = (D2_1 - h12u_half[p1, :, loc.left] / h11u_half[p1, :, loc.left] * D1_1 - Br_1 / N.sqrt(sqrt_det_h_half**2 * h11u_half)[p1, :, loc.left])

#         diff_D1u[p0, :, -1] += dtin * sig_in * 0.5 * (carac_0 * lambda_0 + carac_1[::-1] * lambda_1[::-1]) / dxi / P_half_2[0]

#         carac_1 = (D2_1 - h12u_half[p1, :, loc.left] / h11u_half[p1, :, loc.left] * D1_1 + Br_1 / N.sqrt(sqrt_det_h_half**2 * h11u_half)[p1, :, loc.left])
#         carac_0 = (D1_0 - h12u_half[p0, :, loc.top]  / h22u_half[p0, :, loc.top]  * D2_0 - Br_0 / N.sqrt(sqrt_det_h_half**2 * h22u_half)[p0, :, loc.top])

#         diff_D2u[p1, 0, :]  += dtin * sig_in * 0.5 * (carac_1 * lambda_1 + carac_0[::-1] * lambda_0[::-1]) / dxi / P_half_2[0]


def interface_D(p0, p1, Drin, D1in, D2in):

    i0 =  1
    i1_half = Nxi_half - 1
    i1_int  = Nxi_int - 1

    top = topology[p0, p1]
    
    if (top == 'xx'):
        Drin[p0, -1, i0:i1_int] -= diff_Dru[p0, -1, i0:i1_int] #/ sqrt_det_h_int[p0, i0:i1_int, loc.right]
        Drin[p1, 0, i0:i1_int]  -= diff_Dru[p1, 0, i0:i1_int]  #/ sqrt_det_h_int[p1, i0:i1_int, loc.left]
        
        D2in[p0, -1, i0:i1_half] -= diff_D2u[p0, -1, i0:i1_half] #/ sqrt_det_h_half[p0, i0:i1_half, loc.right]
        D2in[p1, 0, i0:i1_half]  -= diff_D2u[p1, 0, i0:i1_half]  #/ sqrt_det_h_half[p1, i0:i1_half, loc.left]

    if (top == 'xy'):
        Drin[p0, -1, i0:i1_int] -= diff_Dru[p0, -1, i0:i1_int] #/ sqrt_det_h_int[p0, i0:i1_int, loc.right]
        Drin[p1, i0:i1_int, 0]  -= diff_Dru[p1, i0:i1_int, 0]  #/ sqrt_det_h_int[p1, i0:i1_int, loc.bottom]

        D2in[p0, -1, i0:i1_half] -= diff_D2u[p0, -1, i0:i1_half] #/ sqrt_det_h_half[p0, i0:i1_half, loc.right]
        D1in[p1, i0:i1_half, 0]  -= diff_D1u[p1, i0:i1_half, 0]  #/ sqrt_det_h_half[p1, i0:i1_half, loc.bottom]

    if (top == 'yy'):
        Drin[p0, i0:i1_int, -1] -= diff_Dru[p0, i0:i1_int, -1] #/ sqrt_det_h_int[p0, i0:i1_int, loc.top]
        Drin[p1, i0:i1_int, 0]  -= diff_Dru[p1, i0:i1_int, 0]  #/ sqrt_det_h_int[p1, i0:i1_int, loc.bottom]

        D1in[p0, i0:i1_half, -1] -= diff_D1u[p0, i0:i1_half, -1] #/ sqrt_det_h_half[p0, i0:i1_half, loc.top]
        D1in[p1, i0:i1_half, 0]  -= diff_D1u[p1, i0:i1_half, 0]  #/ sqrt_det_h_half[p1, i0:i1_half, loc.bottom]

    if (top == 'yx'):
        Drin[p0, i0:i1_int, -1] -= diff_Dru[p0, i0:i1_int, -1] #/ sqrt_det_h_int[p0, i0:i1_int, loc.top]
        Drin[p1, 0, i0:i1_int]  -= diff_Dru[p1, 0, i0:i1_int]  #/ sqrt_det_h_int[p1, i0:i1_int, loc.left]

        D1in[p0, i0:i1_half, -1] -= diff_D1u[p0, i0:i1_half, -1] #/ sqrt_det_h_half[p0, i0:i1_half, loc.top]
        D2in[p1, 0, i0:i1_half]  -= diff_D2u[p1, 0, i0:i1_half]  #/ sqrt_det_h_half[p1, i0:i1_half, loc.left]

def corners_D(p0, Drin, D1in, D2in):
    
    Drin[p0, 0, 0]   -= diff_Dru[p0, 0, 0]   * sig_cor / sig_in #/ sqrt_det_h_int[p0, 0, loc.bottom]
    Drin[p0, -1, 0]  -= diff_Dru[p0, -1, 0]  * sig_cor / sig_in #/ sqrt_det_h_int[p0, -1, loc.bottom]
    Drin[p0, 0, -1]  -= diff_Dru[p0, 0, -1]  * sig_cor / sig_in #/ sqrt_det_h_int[p0, 0, loc.top]
    Drin[p0, -1, -1] -= diff_Dru[p0, -1, -1] * sig_cor / sig_in #/ sqrt_det_h_int[p0, -1, loc.top]

    D1in[p0, 0, 0]   -= diff_D1u[p0, 0, 0]   * sig_cor / sig_in #/ sqrt_det_h_half[p0, 0, loc.bottom]
    D1in[p0, -1, 0]  -= diff_D1u[p0, -1, 0]  * sig_cor / sig_in #/ sqrt_det_h_half[p0, -1, loc.bottom]
    D1in[p0, 0, -1]  -= diff_D1u[p0, 0, -1]  * sig_cor / sig_in #/ sqrt_det_h_half[p0, 0, loc.top] 
    D1in[p0, -1, -1] -= diff_D1u[p0, -1, -1] * sig_cor / sig_in #/ sqrt_det_h_half[p0, -1, loc.top]

    D2in[p0, 0, 0]   -= diff_D2u[p0, 0, 0]   * sig_cor / sig_in #/ sqrt_det_h_half[p0, 0, loc.bottom]
    D2in[p0, -1, 0]  -= diff_D2u[p0, -1, 0]  * sig_cor / sig_in #/ sqrt_det_h_half[p0, -1, loc.bottom]
    D2in[p0, 0, -1]  -= diff_D2u[p0, 0, -1]  * sig_cor / sig_in #/ sqrt_det_h_half[p0, 0, loc.top]
    D2in[p0, -1, -1] -= diff_D2u[p0, -1, -1] * sig_cor / sig_in #/ sqrt_det_h_half[p0, -1, loc.top]


def compute_penalty_B(p0, p1, dtin, Drin, D1in, D2in, Brin, B1in, B2in):

    top = topology[p0, p1]
    
    if (top == 'xx'):

        #######
        # Br
        #######

        # lambda_0 = alpha_half[p0, :, loc.right] * N.sqrt(h11u_half[p0, :, loc.right]) * sqrt_det_h_half[p0, :, loc.right]
        # lambda_1 = alpha_half[p1, :, loc.left]  * N.sqrt(h11u_half[p1, :, loc.left])  * sqrt_det_h_half[p1, :, loc.left]

        lambda_0 = alpha_half[p0, :, loc.right] * N.sqrt(h11u_half[p0, :, loc.right])
        lambda_1 = alpha_half[p1, :, loc.left]  * N.sqrt(h11u_half[p1, :, loc.left]) 
        
        D2_0 = D2in[p0, -1, :]
        Br_0 = Brin[p0, -1, :]
        B1_0 = B1in[p0, -1, :]

        D2_1 = D2in[p1, 0, :]
        Br_1 = Brin[p1, 0, :]
        B1_1, B2_1 = transform_vect(p1, p0, xi_half[0], eta_half[:], B1in[p1, 0, :], interp(B2in[p1, 0, :], eta_int, eta_half))

        carac_0 = (- D2_0 / N.sqrt(sqrt_det_h_half**2 * h11u_half)[p0, :, loc.right] + Br_0 - hr1u_half[p0, :, loc.right] / h11u_half[p0, :, loc.right] * B1_0)
        carac_1 = (- D2_1 / N.sqrt(sqrt_det_h_half**2 * h11u_half)[p0, :, loc.right] + Br_1 - hr1u_half[p0, :, loc.right] / h11u_half[p0, :, loc.right] * B1_1)

        diff_Bru[p0, -1, :] += dtin * sig_in * 0.5 * (carac_0 - carac_1) * lambda_0 / dxi / P_half_2[0]

        D2_0 = D2in[p0, -1, :]
        Br_0 = Brin[p0, -1, :]
        B1_0, B2_0 = transform_vect(p0, p1, xi_half[-1], eta_half[:], B1in[p0, -1, :], interp(B2in[p0, -1, :], eta_int, eta_half))

        D2_1 = D2in[p1, 0, :]
        Br_1 = Brin[p1, 0, :]
        B1_1 = B1in[p1, 0, :]
 
        carac_1 = (D2_1 / N.sqrt(sqrt_det_h_half**2 * h11u_half)[p1, :, loc.left] + Br_1 - hr1u_half[p1, :, loc.left] / h11u_half[p1, :, loc.left] * B1_1)
        carac_0 = (D2_0 / N.sqrt(sqrt_det_h_half**2 * h11u_half)[p1, :, loc.left] + Br_0 - hr1u_half[p1, :, loc.left] / h11u_half[p1, :, loc.left] * B1_0)
        
        diff_Bru[p1, 0, :]  += dtin * sig_in * 0.5 * (carac_1 - carac_0) * lambda_1 / dxi / P_half_2[0]

        #######
        # B2
        #######

        # lambda_0 = alpha_int[p0, :, loc.right] * N.sqrt(h11u_int[p0, :, loc.right]) * sqrt_det_h_int[p0, :, loc.right]
        # lambda_1 = alpha_int[p1, :, loc.left]  * N.sqrt(h11u_int[p1, :, loc.left])  * sqrt_det_h_int[p1, :, loc.left]
        
        lambda_0 = alpha_int[p0, :, loc.right] * N.sqrt(h11u_int[p0, :, loc.right])
        lambda_1 = alpha_int[p1, :, loc.left]  * N.sqrt(h11u_int[p1, :, loc.left]) 
        
        Dr_0 = Drin[p0, -1, :]
        B1_0 = interp(B1in[p0, -1, :], eta_half, eta_int)
        B2_0 = B2in[p0, -1, :]

        Dr_1 = Drin[p1, 0, :]
        B1_1, B2_1 = transform_vect(p1, p0, xi_int[0], eta_int[:], interp(B1in[p1, 0, :], eta_half, eta_int), B2in[p1, 0, :])

        carac_0 = (Dr_0 / N.sqrt(sqrt_det_h_int**2 * h11u_int)[p0, :, loc.right] + B2_0 - h12u_int[p0, :, loc.right] / h11u_int[p0, :, loc.right] * B1_0)
        carac_1 = (Dr_1 / N.sqrt(sqrt_det_h_int**2 * h11u_int)[p0, :, loc.right] + B2_1 - h12u_int[p0, :, loc.right] / h11u_int[p0, :, loc.right] * B1_1)

        diff_B2u[p0, -1, :] += dtin * sig_in * 0.5 * (carac_0 - carac_1) * lambda_0 / dxi / P_int_2[0]

        Dr_0 = Drin[p0, -1, :]
        B1_0, B2_0 = transform_vect(p0, p1, xi_int[-1], eta_int[:], interp(B1in[p0, -1, :], eta_half, eta_int), B2in[p0, -1, :])

        Dr_1 = Drin[p1, 0, :]
        B1_1 = interp(B1in[p1, 0, :], eta_half, eta_int)
        B2_1 = B2in[p1, 0, :]        

        carac_1 = (- Dr_1 / N.sqrt(sqrt_det_h_int**2 * h11u_int)[p1, :, loc.left] + B2_1 - h12u_int[p1, :, loc.left] / h11u_int[p1, :, loc.left] * B1_1)
        carac_0 = (- Dr_0 / N.sqrt(sqrt_det_h_int**2 * h11u_int)[p1, :, loc.left] + B2_0 - h12u_int[p1, :, loc.left] / h11u_int[p1, :, loc.left] * B1_0)
        
        diff_B2u[p1, 0, :]  += dtin * sig_in * 0.5 * (carac_1 - carac_0) * lambda_1 / dxi / P_int_2[0]

    if (top == 'xy'):

        #######
        # Br
        #######

        # lambda_0 = alpha_half[p0, :, loc.right]  * N.sqrt(h11u_half[p0, :, loc.right])  * sqrt_det_h_half[p0, :, loc.right]
        # lambda_1 = alpha_half[p1, :, loc.bottom] * N.sqrt(h22u_half[p1, :, loc.bottom]) * sqrt_det_h_half[p1, :, loc.bottom]

        lambda_0 = alpha_half[p0, :, loc.right]  * N.sqrt(h11u_half[p0, :, loc.right]) 
        lambda_1 = alpha_half[p1, :, loc.bottom] * N.sqrt(h22u_half[p1, :, loc.bottom])
        
        D2_0 = D2in[p0, -1, :]
        Br_0 = Brin[p0, -1, :]
        B1_0 = B1in[p0, -1, :]

        D1_1 = D1in[p1, :, 0]
        Br_1 = Brin[p1, :, 0]
        B1_1, B2_1 = transform_vect(p1, p0, xi_half[:], eta_half[0], interp(B1in[p1, :, 0], xi_int, xi_half), B2in[p1, :, 0])

        carac_0 = (- D2_0       / N.sqrt(sqrt_det_h_half**2 * h11u_half)[p0, :, loc.right] + Br_0       - hr1u_half[p0, :, loc.right] / h11u_half[p0, :, loc.right] * B1_0)
        carac_1 = (  D1_1[::-1] / N.sqrt(sqrt_det_h_half**2 * h11u_half)[p0, :, loc.right] + Br_1[::-1] - hr1u_half[p0, :, loc.right] / h11u_half[p0, :, loc.right] * B1_1[::-1])
        
        diff_Bru[p0, -1, :] += dtin * sig_in * 0.5 * (carac_0 - carac_1) * lambda_0 / dxi / P_half_2[0]

        D2_0 = D2in[p0, -1, :]
        Br_0 = Brin[p0, -1, :]
        B1_0, B2_0 = transform_vect(p0, p1, xi_half[-1], eta_half[:], B1in[p0, -1, :], interp(B2in[p0, -1, :], eta_int, eta_half))

        D1_1 = D1in[p1, :, 0]
        Br_1 = Brin[p1, :, 0]
        B2_1 = B2in[p1, :, 0]
        
        carac_1 = (- D1_1       / N.sqrt(sqrt_det_h_half**2 * h22u_half)[p1, :, loc.bottom] + Br_1       - hr2u_half[p1, :, loc.bottom] / h22u_half[p1, :, loc.bottom] * B2_1)
        carac_0 = (  D2_0[::-1] / N.sqrt(sqrt_det_h_half**2 * h22u_half)[p1, :, loc.bottom] + Br_0[::-1] - hr2u_half[p1, :, loc.bottom] / h22u_half[p1, :, loc.bottom] * B2_0[::-1])
        
        diff_Bru[p1, :, 0]  += dtin * sig_in * 0.5 * (carac_1 - carac_0) * lambda_1 / dxi / P_half_2[0]

        #######
        # B1, B2
        #######

        # lambda_0 = alpha_int[p0, :, loc.right]  * N.sqrt(h11u_int[p0, :, loc.right])  * sqrt_det_h_int[p0, :, loc.right]
        # lambda_1 = alpha_int[p1, :, loc.bottom] * N.sqrt(h22u_int[p1, :, loc.bottom]) * sqrt_det_h_int[p1, :, loc.bottom]

        lambda_0 = alpha_int[p0, :, loc.right]  * N.sqrt(h11u_int[p0, :, loc.right]) 
        lambda_1 = alpha_int[p1, :, loc.bottom] * N.sqrt(h22u_int[p1, :, loc.bottom])
        
        Dr_0 = Drin[p0, -1, :]
        B1_0 = interp(B1in[p0, -1, :], eta_half, eta_int)
        B2_0 = B2in[p0, -1, :]

        Dr_1 = Drin[p1, :, 0]
        B1_1, B2_1 = transform_vect(p1, p0, xi_int[:], eta_int[0], B1in[p1, :, 0], interp(B2in[p1, :, 0], xi_half, xi_int))

        carac_0 = (Dr_0       / N.sqrt(sqrt_det_h_int**2 * h11u_int)[p0, :, loc.right] + B2_0       - h12u_int[p0, :, loc.right] / h11u_int[p0, :, loc.right] * B1_0)
        carac_1 = (Dr_1[::-1] / N.sqrt(sqrt_det_h_int**2 * h11u_int)[p0, :, loc.right] + B2_1[::-1] - h12u_int[p0, :, loc.right] / h11u_int[p0, :, loc.right] * B1_1[::-1])

        diff_B2u[p0, -1, :] += dtin * sig_in * 0.5 * (carac_0 - carac_1) * lambda_0 / dxi / P_int_2[0]

        Dr_0 = Drin[p0, -1, :]
        B1_0, B2_0 = transform_vect(p0, p1, xi_int[-1], eta_int[:], interp(B1in[p0, -1, :], eta_half, eta_int), B2in[p0, -1, :])

        Dr_1 = Drin[p1, :, 0]
        B1_1 = B1in[p1, :, 0]
        B2_1 = interp(B2in[p1, :, 0], xi_half, xi_int)
        
        carac_1 = (Dr_1       / N.sqrt(sqrt_det_h_int**2 * h22u_int)[p1, :, loc.bottom] + B1_1       - h12u_int[p1, :, loc.bottom] / h22u_int[p1, :, loc.bottom] * B2_1)
        carac_0 = (Dr_0[::-1] / N.sqrt(sqrt_det_h_int**2 * h22u_int)[p1, :, loc.bottom] + B1_0[::-1] - h12u_int[p1, :, loc.bottom] / h22u_int[p1, :, loc.bottom] * B2_0[::-1])
        
        diff_B1u[p1, :, 0]  += dtin * sig_in * 0.5 * (carac_1 - carac_0) * lambda_1 / dxi / P_int_2[0]

    if (top == 'yy'):
        
        #######
        # Br
        #######

        # lambda_0 = alpha_half[p0, :, loc.top]    * N.sqrt(h22u_half[p0, :, loc.top])    * sqrt_det_h_half[p0, :, loc.top]
        # lambda_1 = alpha_half[p1, :, loc.bottom] * N.sqrt(h22u_half[p1, :, loc.bottom]) * sqrt_det_h_half[p1, :, loc.bottom]
        
        lambda_0 = alpha_half[p0, :, loc.top]    * N.sqrt(h22u_half[p0, :, loc.top])   
        lambda_1 = alpha_half[p1, :, loc.bottom] * N.sqrt(h22u_half[p1, :, loc.bottom])
        
        D1_0 = D1in[p0, :, -1]
        Br_0 = Brin[p0, :, -1]
        B2_0 = B2in[p0, :, -1]

        D1_1 = D1in[p1, :, 0]
        Br_1 = Brin[p1, :, 0]
        B1_1, B2_1 = transform_vect(p1, p0, xi_half[:], eta_half[0], interp(B1in[p1, :, 0], xi_int, xi_half), B2in[p1, :, 0])

        carac_0 = (  D1_0 / N.sqrt(sqrt_det_h_half**2 * h22u_half)[p0, :, loc.top] + Br_0 - hr2u_half[p0, :, loc.top] / h22u_half[p0, :, loc.top] * B2_0)
        carac_1 = (  D1_1 / N.sqrt(sqrt_det_h_half**2 * h22u_half)[p0, :, loc.top] + Br_1 - hr2u_half[p0, :, loc.top] / h22u_half[p0, :, loc.top] * B2_1)

        diff_Bru[p0, :, -1] += dtin * sig_in * 0.5 * (carac_0 - carac_1) * lambda_0 / dxi / P_half_2[0]

        D1_0 = D1in[p0, :, -1]
        Br_0 = Brin[p0, :, -1]
        B1_0, B2_0 = transform_vect(p0, p1, xi_half[:], eta_half[-1], interp(B1in[p0, :, -1], xi_int, xi_half), B2in[p0, :, -1])

        D1_1 = D1in[p1, :, 0]
        Br_1 = Brin[p1, :, 0]
        B2_1 = B2in[p1, :, 0]

        carac_1 = (- D1_1 / N.sqrt(sqrt_det_h_half**2 * h11u_half)[p1, :, loc.bottom] + Br_1 - hr2u_half[p1, :, loc.bottom] / h22u_half[p1, :, loc.bottom] * B2_1)
        carac_0 = (- D1_0 / N.sqrt(sqrt_det_h_half**2 * h11u_half)[p1, :, loc.bottom] + Br_0 - hr2u_half[p1, :, loc.bottom] / h22u_half[p1, :, loc.bottom] * B2_0)
        
        diff_Bru[p1, :, 0]  += dtin * sig_in * 0.5 * (carac_1 - carac_0) * lambda_1 / dxi / P_half_2[0]

        #######
        # B1
        #######

        # lambda_0 = alpha_int[p0, :, loc.top]    * N.sqrt(h22u_int[p0, :, loc.top])    * sqrt_det_h_int[p0, :, loc.top]
        # lambda_1 = alpha_int[p1, :, loc.bottom] * N.sqrt(h22u_int[p1, :, loc.bottom]) * sqrt_det_h_int[p1, :, loc.bottom]

        lambda_0 = alpha_int[p0, :, loc.top]    * N.sqrt(h22u_int[p0, :, loc.top])   
        lambda_1 = alpha_int[p1, :, loc.bottom] * N.sqrt(h22u_int[p1, :, loc.bottom])

        Dr_0 = Drin[p0, :, -1]
        B1_0 = B1in[p0, :, -1]
        B2_0 = interp(B2in[p0, :, -1], xi_half, xi_int)

        Dr_1 = Drin[p1, :, 0]
        B1_1, B2_1 = transform_vect(p1, p0, xi_int[:], eta_int[0], B1in[p1, :, 0], interp(B2in[p1, :, 0], xi_half, xi_int))
        
        carac_0 = (- Dr_0 / N.sqrt(sqrt_det_h_int**2 * h22u_int)[p0, :, loc.top] + B1_0 - h12u_int[p0, :, loc.top] / h22u_int[p0, :, loc.top] * B2_0)
        carac_1 = (- Dr_1 / N.sqrt(sqrt_det_h_int**2 * h22u_int)[p0, :, loc.top] + B1_1 - h12u_int[p0, :, loc.top] / h22u_int[p0, :, loc.top] * B2_1)
        
        diff_B1u[p0, :, -1] += dtin * sig_in * 0.5 * (carac_0 - carac_1) * lambda_0 / dxi / P_int_2[0]

        Dr_0 = Drin[p0, :, -1]
        B1_0, B2_0 = transform_vect(p0, p1, xi_int[:], eta_int[-1], B1in[p0, :, -1], interp(B2in[p0, :, -1], xi_half, xi_int))

        Dr_1 = Drin[p1, :, 0]
        B1_1 = B1in[p1, :, 0]
        B2_1 = interp(B2in[p1, :, 0], xi_half, xi_int)

        carac_1 = (Dr_1 / N.sqrt(sqrt_det_h_int**2 * h22u_int)[p1, :, loc.bottom] + B1_1 - h12u_int[p1, :, loc.bottom] / h22u_int[p1, :, loc.bottom] * B2_1)
        carac_0 = (Dr_0 / N.sqrt(sqrt_det_h_int**2 * h22u_int)[p1, :, loc.bottom] + B1_0 - h12u_int[p1, :, loc.bottom] / h22u_int[p1, :, loc.bottom] * B2_0)

        diff_B1u[p1, :, 0]  += dtin * sig_in * 0.5 * (carac_1 - carac_0) * lambda_0 / dxi / P_int_2[0]
        
    if (top == 'yx'):

        #######
        # Br
        #######

        # lambda_0 = alpha_half[p0, :, loc.top]  * N.sqrt(h22u_half[p0, :, loc.top])  * sqrt_det_h_half[p0, :, loc.top]
        # lambda_1 = alpha_half[p1, :, loc.left] * N.sqrt(h11u_half[p1, :, loc.left]) * sqrt_det_h_half[p1, :, loc.left]

        lambda_0 = alpha_half[p0, :, loc.top]  * N.sqrt(h22u_half[p0, :, loc.top]) 
        lambda_1 = alpha_half[p1, :, loc.left] * N.sqrt(h11u_half[p1, :, loc.left])
        
        D1_0 = D1in[p0, :, -1]
        Br_0 = Brin[p0, :, -1]
        B2_0 = B2in[p0, :, -1]

        D2_1 = D2in[p1, 0, :]
        Br_1 = Brin[p1, 0, :]
        B1_1, B2_1 = transform_vect(p1, p0, xi_half[0], eta_half, B1in[p1, 0, :], interp(B2in[p1, 0, :], eta_int, eta_half))

        carac_0 = (  D1_0       / N.sqrt(sqrt_det_h_half**2 * h22u_half)[p0, :, loc.top] + Br_0       - hr2u_half[p0, :, loc.top] / h22u_half[p0, :, loc.top] * B2_0)
        carac_1 = (- D2_1[::-1] / N.sqrt(sqrt_det_h_half**2 * h22u_half)[p0, :, loc.top] + Br_1[::-1] - hr2u_half[p0, :, loc.top] / h22u_half[p0, :, loc.top] * B2_1[::-1])

        diff_Bru[p0, :, -1] += dtin * sig_in * 0.5 * (carac_0 - carac_1) * lambda_0 / dxi / P_half_2[0]

        D1_0 = D1in[p0, :, -1]
        Br_0 = Brin[p0, :, -1]
        B1_0, B2_0 = transform_vect(p0, p1, xi_half, eta_half[-1], interp(B1in[p0, :, -1], xi_int, xi_half), B2in[p0, :, -1])

        D2_1 = D2in[p1, 0, :]
        Br_1 = Brin[p1, 0, :]
        B1_1 = B1in[p1, 0, :]
        
        carac_1 = (  D2_1       / N.sqrt(sqrt_det_h_half**2 * h11u_half)[p1, :, loc.left] + Br_1       - hr1u_half[p1, :, loc.left] / h11u_half[p1, :, loc.left] * B1_1)
        carac_0 = (- D1_0[::-1] / N.sqrt(sqrt_det_h_half**2 * h11u_half)[p1, :, loc.left] + Br_0[::-1] - hr1u_half[p1, :, loc.left] / h11u_half[p1, :, loc.left] * B1_0[::-1])
        
        diff_Bru[p1, 0, :]  += dtin * sig_in * 0.5 * (carac_1 - carac_0) * lambda_1 / dxi / P_half_2[0]

        #######
        # B1, B2
        #######

        # lambda_0 = alpha_int[p0, :, loc.top]  * N.sqrt(h22u_int[p0, :, loc.top])  * sqrt_det_h_int[p0, :, loc.top]
        # lambda_1 = alpha_int[p1, :, loc.left] * N.sqrt(h11u_int[p1, :, loc.left]) * sqrt_det_h_int[p1, :, loc.left]

        lambda_0 = alpha_int[p0, :, loc.top]  * N.sqrt(h22u_int[p0, :, loc.top]) 
        lambda_1 = alpha_int[p1, :, loc.left] * N.sqrt(h11u_int[p1, :, loc.left])

        Dr_0 = Drin[p0, :, -1]
        B1_0 = B1in[p0, :, -1]
        B2_0 = interp(B2in[p0, :, -1], xi_half, xi_int)

        Dr_1 = Drin[p1, 0, :]
        B1_1, B2_1 = transform_vect(p1, p0, xi_int[0], eta_int[:], interp(B1in[p1, 0, :], xi_half, xi_int), B2in[p1, 0, :])

        carac_0 = (- Dr_0       / N.sqrt(sqrt_det_h_int**2 * h22u_int)[p0, :, loc.top] + B1_0       - h12u_int[p0, :, loc.top] / h22u_int[p0, :, loc.top] * B2_0)
        carac_1 = (- Dr_1[::-1] / N.sqrt(sqrt_det_h_int**2 * h22u_int)[p0, :, loc.top] + B1_1[::-1] - h12u_int[p0, :, loc.top] / h22u_int[p0, :, loc.top] * B2_1[::-1])

        diff_B1u[p0, :, -1] += dtin * sig_in * 0.5 * (carac_0 - carac_1) * lambda_0 / dxi / P_int_2[0]

        Dr_0 = Drin[p0, :, -1]
        B1_0, B2_0 = transform_vect(p0, p1, xi_int[:], eta_int[-1], B1in[p0, :, -1], interp(B2in[p0, :, -1], xi_half, xi_int))

        Dr_1 = Drin[p1, 0, :]
        B1_1 = interp(B1in[p1, 0, :], xi_half, xi_int)
        B2_1 = B2in[p1, 0, :]

        carac_1 = (- Dr_1       / N.sqrt(sqrt_det_h_int**2 * h11u_int)[p1, :, loc.left] + B2_1       - h12u_int[p1, :, loc.left] / h11u_int[p1, :, loc.left] * B1_1)
        carac_0 = (- Dr_0[::-1] / N.sqrt(sqrt_det_h_int**2 * h11u_int)[p1, :, loc.left] + B2_0[::-1] - h12u_int[p1, :, loc.left] / h11u_int[p1, :, loc.left] * B1_0[::-1])

        diff_B2u[p1, 0, :]  += dtin * sig_in * 0.5 * (carac_1 - carac_0) * lambda_1 / dxi / P_int_2[0]

    # if (top == 'yx'):

    #     # Br
    #     lambda_0 = alpha_half[p0, :, loc.top]  * N.sqrt(h22u_half[p0, :, loc.top])  * sqrt_det_h_half[p0, :, loc.top]
    #     lambda_1 = alpha_half[p1, :, loc.left] * N.sqrt(h11u_half[p1, :, loc.left]) * sqrt_det_h_half[p1, :, loc.left]
        
    #     D1_0 = D1in[p0, :, -1]
    #     Br_0 = Brin[p0, :, -1]
    #     B2_0 = B2in[p0, :, -1]

    #     D2_1 = D2in[p1, 0, :]
    #     Br_1 = Brin[p1, 0, :]
    #     B1_1 = B1in[p1, 0, :]

    #     carac_0 = (  D1_0 / N.sqrt(sqrt_det_h_half**2 * h22u_half)[p0, :, loc.top]  + Br_0 - hr2u_half[p0, :, loc.top]  / h22u_half[p0, :, loc.top]  * B2_0)
    #     carac_1 = (- D2_1 / N.sqrt(sqrt_det_h_half**2 * h11u_half)[p1, :, loc.left] + Br_1 - hr1u_half[p1, :, loc.left] / h11u_half[p1, :, loc.left] * B1_1)

    #     diff_Bru[p0, :, -1] += dtin * sig_in * 0.5 * (carac_0 * lambda_0 - carac_1[::-1] * lambda_1[::-1]) / dxi / P_half_2[0]
                
    #     carac_1 = (  D2_1 / N.sqrt(sqrt_det_h_half**2 * h11u_half)[p1, :, loc.left] + Br_1 - hr1u_half[p1, :, loc.left] / h11u_half[p1, :, loc.left] * B1_1)
    #     carac_0 = (- D1_0 / N.sqrt(sqrt_det_h_half**2 * h22u_half)[p0, :, loc.top]  + Br_0 - hr2u_half[p0, :, loc.top]  / h22u_half[p0, :, loc.top]  * B2_0)
        
    #     diff_Bru[p1, 0, :]  += dtin * sig_in * 0.5 * (carac_1 * lambda_1 - carac_0[::-1] * lambda_0[::-1]) / dxi / P_half_2[0]

    #     # B1, B2
    #     lambda_0 = alpha_int[p0, :, loc.top]  * N.sqrt(h22u_int[p0, :, loc.top])  * sqrt_det_h_int[p0, :, loc.top]
    #     lambda_1 = alpha_int[p1, :, loc.left] * N.sqrt(h11u_int[p1, :, loc.left]) * sqrt_det_h_int[p1, :, loc.left]
        
    #     Dr_0 = Drin[p0, :, -1]
    #     B1_0 = B1in[p0, :, -1]
    #     B2_0 = interp(B2in[p0, :, -1], xi_half, xi_int)

    #     Dr_1 = Drin[p1, 0, :]
    #     B1_1 = interp(B1in[p1, 0, :], xi_half, xi_int)
    #     B2_1 = B2in[p1, 0, :]

    #     carac_0 = (- Dr_0 / N.sqrt(sqrt_det_h_int**2 * h22u_int)[p0, :, loc.top]  + B1_0 - h12u_int[p0, :, loc.top]  / h22u_int[p0, :, loc.top]  * B2_0)
    #     carac_1 = (  Dr_1 / N.sqrt(sqrt_det_h_int**2 * h11u_int)[p1, :, loc.left] + B2_1 - h12u_int[p1, :, loc.left] / h11u_int[p1, :, loc.left] * B1_1)

    #     diff_B1u[p0, :, -1] += dtin * sig_in * 0.5 * (carac_0 * lambda_0 + carac_1[::-1] * lambda_1[::-1]) / dxi / P_int_2[0]

    #     carac_1 = (- Dr_1 / N.sqrt(sqrt_det_h_int**2 * h11u_int)[p1, :, loc.left] + B2_1 - h12u_int[p1, :, loc.left] / h11u_int[p1, :, loc.left] * B1_1)
    #     carac_0 = (  Dr_0 / N.sqrt(sqrt_det_h_int**2 * h22u_int)[p0, :, loc.top]  + B1_0 - h12u_int[p0, :, loc.top]  / h22u_int[p0, :, loc.top]  * B2_0)

    #     diff_B2u[p1, 0, :]  += dtin * sig_in * 0.5 * (carac_1 * lambda_1 + carac_0[::-1] * lambda_0[::-1]) / dxi / P_int_2[0]


#######
# PREVIOUS ATTEMPT
#######

# def compute_penalty_B(p0, p1, dtin, Drin, D1in, D2in, Brin, B1in, B2in):

#     top = topology[p0, p1]
    
#     if (top == 'xx'):

#         # Br
#         lambda_0 = alpha_half[p0, :, loc.right] * N.sqrt(h11u_half[p0, :, loc.right]) * sqrt_det_h_half[p0, :, loc.right]
#         lambda_1 = alpha_half[p1, :, loc.left]  * N.sqrt(h11u_half[p1, :, loc.left])  * sqrt_det_h_half[p1, :, loc.left]

#         D2_0 = D2in[p0, -1, :]
#         Br_0 = Brin[p0, -1, :]
#         B1_0 = B1in[p0, -1, :]

#         D2_1 = D2in[p1, 0, :]
#         Br_1 = Brin[p1, 0, :]
#         B1_1 = B1in[p1, 0, :]

#         carac_0 = (- D2_0 / N.sqrt(sqrt_det_h_half**2 * h11u_half)[p0, :, loc.right] + Br_0 - hr1u_half[p0, :, loc.right] / h11u_half[p0, :, loc.right] * B1_0)
#         carac_1 = (- D2_1 / N.sqrt(sqrt_det_h_half**2 * h11u_half)[p1, :, loc.left]  + Br_1 - hr1u_half[p1, :, loc.left]  / h11u_half[p1, :, loc.left]  * B1_1)

#         diff_Bru[p0, -1, :] += dtin * sig_in * 0.5 * (carac_0 * lambda_0 - carac_1 * lambda_1) / dxi / P_half_2[0]
 
#         carac_1 = (D2_1 / N.sqrt(sqrt_det_h_half**2 * h11u_half)[p1, :, loc.left]  + Br_1 - hr1u_half[p1, :, loc.left]  / h11u_half[p1, :, loc.left]  * B1_1)
#         carac_0 = (D2_0 / N.sqrt(sqrt_det_h_half**2 * h11u_half)[p0, :, loc.right] + Br_0 - hr1u_half[p0, :, loc.right] / h11u_half[p0, :, loc.right] * B1_0)
        
#         diff_Bru[p1, 0, :]  += dtin * sig_in * 0.5 * (carac_1 * lambda_1 - carac_0 * lambda_0) / dxi / P_half_2[0]

#         # B2
#         lambda_0 = alpha_int[p0, :, loc.right] * N.sqrt(h11u_int[p0, :, loc.right]) * sqrt_det_h_int[p0, :, loc.right]
#         lambda_1 = alpha_int[p1, :, loc.left]  * N.sqrt(h11u_int[p1, :, loc.left])  * sqrt_det_h_int[p1, :, loc.left]

#         Dr_0 = Drin[p0, -1, :]
#         B1_0 = interp(B1in[p0, -1, :], eta_half, eta_int)
#         B2_0 = B2in[p0, -1, :]

#         Dr_1 = Drin[p1, 0, :]
#         B1_1 = interp(B1in[p1, 0, :], eta_half, eta_int)
#         B2_1 = B2in[p1, 0, :]

#         carac_0 = (Dr_0 / N.sqrt(sqrt_det_h_int**2 * h11u_int)[p0, :, loc.right] + B2_0 - h12u_int[p0, :, loc.right] / h11u_int[p0, :, loc.right] * B1_0)
#         carac_1 = (Dr_1 / N.sqrt(sqrt_det_h_int**2 * h11u_int)[p1, :, loc.left]  + B2_1 - h12u_int[p1, :, loc.left]  / h11u_int[p1, :, loc.left]  * B1_1)

#         diff_B2u[p0, -1, :] += dtin * sig_in * 0.5 * (carac_0 * lambda_0 - carac_1 * lambda_1) / dxi / P_int_2[0]

#         carac_1  = (- Dr_1 / N.sqrt(sqrt_det_h_int**2 * h11u_int)[p1, :, loc.left]  + B2_1 - h12u_int[p1, :, loc.left]  / h11u_int[p1, :, loc.left]  * B1_1)
#         carac_0 = (- Dr_0 / N.sqrt(sqrt_det_h_int**2 * h11u_int)[p0, :, loc.right] + B2_0 - h12u_int[p0, :, loc.right] / h11u_int[p0, :, loc.right] * B1_0)
        
#         diff_B2u[p1, 0, :]  += dtin * sig_in * 0.5 * (carac_1 * lambda_1 - carac_0 * lambda_0) / dxi / P_int_2[0]

#     if (top == 'xy'):

#         # Br
#         lambda_0 = alpha_half[p0, :, loc.right]  * N.sqrt(h11u_half[p0, :, loc.right])  * sqrt_det_h_half[p0, :, loc.right]
#         lambda_1 = alpha_half[p1, :, loc.bottom] * N.sqrt(h22u_half[p1, :, loc.bottom]) * sqrt_det_h_half[p1, :, loc.bottom]

#         D2_0 = D2in[p0, -1, :]
#         Br_0 = Brin[p0, -1, :]
#         B1_0 = B1in[p0, -1, :]

#         D1_1 = D1in[p1, :, 0]
#         Br_1 = Brin[p1, :, 0]
#         B2_1 = B2in[p1, :, 0]

#         carac_0 = (- D2_0 / N.sqrt(sqrt_det_h_half**2 * h11u_half)[p0, :, loc.right]  + Br_0 - hr1u_half[p0, :, loc.right]  / h11u_half[p0, :, loc.right]  * B1_0)
#         carac_1 = (  D1_1 / N.sqrt(sqrt_det_h_half**2 * h22u_half)[p1, :, loc.bottom] + Br_1 - hr2u_half[p1, :, loc.bottom] / h22u_half[p1, :, loc.bottom] * B2_1)
        
#         diff_Bru[p0, -1, :] += dtin * sig_in * 0.5 * (carac_0 * lambda_0 - carac_1[::-1] * lambda_1[::-1]) / dxi / P_half_2[0]
        
#         carac_1 = (- D1_1 / N.sqrt(sqrt_det_h_half**2 * h22u_half)[p1, :, loc.bottom] + Br_1 - hr2u_half[p1, :, loc.bottom] / h22u_half[p1, :, loc.bottom] * B2_1)
#         carac_0 = (  D2_0 / N.sqrt(sqrt_det_h_half**2 * h11u_half)[p0, :, loc.right]  + Br_0 - hr1u_half[p0, :, loc.right]  / h11u_half[p0, :, loc.right]  * B1_0)
        
#         diff_Bru[p1, :, 0]  += dtin * sig_in * 0.5 * (carac_1 * lambda_1 - carac_0[::-1] * lambda_0[::-1]) / dxi / P_half_2[0]

#         # B1, B2
#         lambda_0 = alpha_int[p0, :, loc.right]  * N.sqrt(h11u_int[p0, :, loc.right])  * sqrt_det_h_int[p0, :, loc.right]
#         lambda_1 = alpha_int[p1, :, loc.bottom] * N.sqrt(h22u_int[p1, :, loc.bottom]) * sqrt_det_h_int[p1, :, loc.bottom]

#         Dr_0 = Drin[p0, -1, :]
#         B1_0 = interp(B1in[p0, -1, :], eta_half, eta_int)
#         B2_0 = B2in[p0, -1, :]

#         Dr_1 = Drin[p1, :, 0]
#         B1_1 = B1in[p1, :, 0]
#         B2_1 = interp(B2in[p1, :, 0], xi_half, xi_int)

#         carac_0 = (  Dr_0 / N.sqrt(sqrt_det_h_int**2 * h11u_int)[p0, :, loc.right]  + B2_0 - h12u_int[p0, :, loc.right]  / h11u_int[p0, :, loc.right]  * B1_0)
#         carac_1 = (- Dr_1 / N.sqrt(sqrt_det_h_int**2 * h22u_int)[p1, :, loc.bottom] + B1_1 - h12u_int[p1, :, loc.bottom] / h22u_int[p1, :, loc.bottom] * B2_1)

#         diff_B2u[p0, -1, :] += dtin * sig_in * 0.5 * (carac_0 * lambda_0 + carac_1[::-1] * lambda_1[::-1]) / dxi / P_int_2[0]
        
#         carac_1 = (  Dr_1 / N.sqrt(sqrt_det_h_int**2 * h22u_int)[p1, :, loc.bottom] + B1_1 - h12u_int[p1, :, loc.bottom] / h22u_int[p1, :, loc.bottom] * B2_1)
#         carac_0 = (- Dr_0 / N.sqrt(sqrt_det_h_int**2 * h11u_int)[p0, :, loc.right]  + B2_0 - h12u_int[p0, :, loc.right]  / h11u_int[p0, :, loc.right]  * B1_0)
        
#         diff_B1u[p1, :, 0]  += dtin * sig_in * 0.5 * (carac_1 * lambda_1 + carac_0[::-1] * lambda_0[::-1]) / dxi / P_int_2[0]

#     if (top == 'yy'):
        
#         # Br
#         lambda_0 = alpha_half[p0, :, loc.top]    * N.sqrt(h22u_half[p0, :, loc.top])    * sqrt_det_h_half[p0, :, loc.top]
#         lambda_1 = alpha_half[p1, :, loc.bottom] * N.sqrt(h22u_half[p1, :, loc.bottom]) * sqrt_det_h_half[p1, :, loc.bottom]
        
#         D1_0 = D1in[p0, :, -1]
#         Br_0 = Brin[p0, :, -1]
#         B2_0 = B2in[p0, :, -1]

#         D1_1 = D1in[p1, :, 0]
#         Br_1 = Brin[p1, :, 0]
#         B2_1 = B2in[p1, :, 0]

#         carac_0 = (  D1_0 / N.sqrt(sqrt_det_h_half**2 * h22u_half)[p0, :, loc.top]    + Br_0 - hr2u_half[p0, :, loc.top]    / h22u_half[p0, :, loc.top]    * B2_0)
#         carac_1 = (  D1_1 / N.sqrt(sqrt_det_h_half**2 * h22u_half)[p1, :, loc.bottom] + Br_1 - hr2u_half[p1, :, loc.bottom] / h22u_half[p1, :, loc.bottom] * B2_1)

#         diff_Bru[p0, :, -1] += dtin * sig_in * 0.5 * (carac_0  * lambda_0 - carac_1 * lambda_1) / dxi / P_half_2[0]
        
#         carac_1 = (- D1_1 / N.sqrt(sqrt_det_h_half**2 * h11u_half)[p1, :, loc.bottom] + Br_1 - hr2u_half[p1, :, loc.bottom] / h22u_half[p1, :, loc.bottom] * B2_1)
#         carac_0 = (- D1_0 / N.sqrt(sqrt_det_h_half**2 * h11u_half)[p0, :, loc.top]    + Br_0 - hr2u_half[p0, :, loc.top]    / h22u_half[p0, :, loc.top]    * B2_0)
        
#         diff_Bru[p1, :, 0]  += dtin * sig_in * 0.5 * (carac_1 * lambda_1 - carac_0 * lambda_0) / dxi / P_half_2[0]

#         # B1
#         lambda_0 = alpha_int[p0, :, loc.top]    * N.sqrt(h22u_int[p0, :, loc.top])    * sqrt_det_h_int[p0, :, loc.top]
#         lambda_1 = alpha_int[p1, :, loc.bottom] * N.sqrt(h22u_int[p1, :, loc.bottom]) * sqrt_det_h_int[p1, :, loc.bottom]

#         Dr_0 = Drin[p0, :, -1]
#         B1_0 = B1in[p0, :, -1]
#         B2_0 = interp(B2in[p0, :, -1], xi_half, xi_int)

#         Dr_1 = Drin[p1, :, 0]
#         B1_1 = B1in[p1, :, 0]
#         B2_1 = interp(B2in[p1, :, 0], xi_half, xi_int)
        
#         carac_0 = (- Dr_0 / N.sqrt(sqrt_det_h_int**2 * h22u_int)[p0, :, loc.top]    + B1_0 - h12u_int[p0, :, loc.top]    / h22u_int[p0, :, loc.top]    * B2_0)
#         carac_1 = (- Dr_1 / N.sqrt(sqrt_det_h_int**2 * h22u_int)[p1, :, loc.bottom] + B1_1 - h12u_int[p1, :, loc.bottom] / h22u_int[p1, :, loc.bottom] * B2_1)
        
#         diff_B1u[p0, :, -1] += dtin * sig_in * 0.5 * (carac_0 * lambda_0 - carac_1 * lambda_1) / dxi / P_int_2[0]

#         carac_1  = (  Dr_1 / N.sqrt(sqrt_det_h_int**2 * h22u_int)[p1, :, loc.bottom] + B1_1 - h12u_int[p1, :, loc.bottom] / h22u_int[p1, :, loc.bottom] * B2_1)
#         carac_0 = (  Dr_0 / N.sqrt(sqrt_det_h_int**2 * h22u_int)[p0, :, loc.top]    + B1_0 - h12u_int[p0, :, loc.top]    / h22u_int[p0, :, loc.top]    * B2_0)

#         diff_B1u[p1, :, 0]  += dtin * sig_in * 0.5 * (carac_1 * lambda_0 - carac_0 * lambda_0) / dxi / P_int_2[0]
        
#     if (top == 'yx'):

#         # Br
#         lambda_0 = alpha_half[p0, :, loc.top]  * N.sqrt(h22u_half[p0, :, loc.top])  * sqrt_det_h_half[p0, :, loc.top]
#         lambda_1 = alpha_half[p1, :, loc.left] * N.sqrt(h11u_half[p1, :, loc.left]) * sqrt_det_h_half[p1, :, loc.left]
        
#         D1_0 = D1in[p0, :, -1]
#         Br_0 = Brin[p0, :, -1]
#         B2_0 = B2in[p0, :, -1]

#         D2_1 = D2in[p1, 0, :]
#         Br_1 = Brin[p1, 0, :]
#         B1_1 = B1in[p1, 0, :]

#         carac_0 = (  D1_0 / N.sqrt(sqrt_det_h_half**2 * h22u_half)[p0, :, loc.top]  + Br_0 - hr2u_half[p0, :, loc.top]  / h22u_half[p0, :, loc.top]  * B2_0)
#         carac_1 = (- D2_1 / N.sqrt(sqrt_det_h_half**2 * h11u_half)[p1, :, loc.left] + Br_1 - hr1u_half[p1, :, loc.left] / h11u_half[p1, :, loc.left] * B1_1)

#         diff_Bru[p0, :, -1] += dtin * sig_in * 0.5 * (carac_0 * lambda_0 - carac_1[::-1] * lambda_1[::-1]) / dxi / P_half_2[0]
                
#         carac_1 = (  D2_1 / N.sqrt(sqrt_det_h_half**2 * h11u_half)[p1, :, loc.left] + Br_1 - hr1u_half[p1, :, loc.left] / h11u_half[p1, :, loc.left] * B1_1)
#         carac_0 = (- D1_0 / N.sqrt(sqrt_det_h_half**2 * h22u_half)[p0, :, loc.top]  + Br_0 - hr2u_half[p0, :, loc.top]  / h22u_half[p0, :, loc.top]  * B2_0)
        
#         diff_Bru[p1, 0, :]  += dtin * sig_in * 0.5 * (carac_1 * lambda_1 - carac_0[::-1] * lambda_0[::-1]) / dxi / P_half_2[0]

#         # B1, B2
#         lambda_0 = alpha_int[p0, :, loc.top]  * N.sqrt(h22u_int[p0, :, loc.top])  * sqrt_det_h_int[p0, :, loc.top]
#         lambda_1 = alpha_int[p1, :, loc.left] * N.sqrt(h11u_int[p1, :, loc.left]) * sqrt_det_h_int[p1, :, loc.left]
        
#         Dr_0 = Drin[p0, :, -1]
#         B1_0 = B1in[p0, :, -1]
#         B2_0 = interp(B2in[p0, :, -1], xi_half, xi_int)

#         Dr_1 = Drin[p1, 0, :]
#         B1_1 = interp(B1in[p1, 0, :], xi_half, xi_int)
#         B2_1 = B2in[p1, 0, :]

#         carac_0 = (- Dr_0 / N.sqrt(sqrt_det_h_int**2 * h22u_int)[p0, :, loc.top]  + B1_0 - h12u_int[p0, :, loc.top]  / h22u_int[p0, :, loc.top]  * B2_0)
#         carac_1 = (  Dr_1 / N.sqrt(sqrt_det_h_int**2 * h11u_int)[p1, :, loc.left] + B2_1 - h12u_int[p1, :, loc.left] / h11u_int[p1, :, loc.left] * B1_1)

#         diff_B1u[p0, :, -1] += dtin * sig_in * 0.5 * (carac_0 * lambda_0 + carac_1[::-1] * lambda_1[::-1]) / dxi / P_int_2[0]

#         carac_1 = (- Dr_1 / N.sqrt(sqrt_det_h_int**2 * h11u_int)[p1, :, loc.left] + B2_1 - h12u_int[p1, :, loc.left] / h11u_int[p1, :, loc.left] * B1_1)
#         carac_0 = (  Dr_0 / N.sqrt(sqrt_det_h_int**2 * h22u_int)[p0, :, loc.top]  + B1_0 - h12u_int[p0, :, loc.top]  / h22u_int[p0, :, loc.top]  * B2_0)

#         diff_B2u[p1, 0, :]  += dtin * sig_in * 0.5 * (carac_1 * lambda_1 + carac_0[::-1] * lambda_0[::-1]) / dxi / P_int_2[0]


def interface_B(p0, p1, Brin, B1in, B2in):

    i0 =  1
    i1_half = Nxi_half - 1
    i1_int  = Nxi_int - 1

    top = topology[p0, p1]
    
    if (top == 'xx'):
        Brin[p0, -1, i0:i1_half] -= diff_Bru[p0, -1, i0:i1_half] #/ sqrt_det_h_half[p0, i0:i1_half, loc.right]
        Brin[p1, 0, i0:i1_half]  -= diff_Bru[p1, 0, i0:i1_half]  #/ sqrt_det_h_half[p1, i0:i1_half, loc.left]

        B2in[p0, -1, i0:i1_int] -= diff_B2u[p0, -1, i0:i1_int] #/ sqrt_det_h_int[p0, i0:i1_int, loc.right]
        B2in[p1, 0, i0:i1_int]  -= diff_B2u[p1, 0, i0:i1_int]  #/ sqrt_det_h_int[p1, i0:i1_int, loc.left]

    if (top == 'xy'):
        Brin[p0, -1, i0:i1_half] -= diff_Bru[p0, -1, i0:i1_half] #/ sqrt_det_h_half[p0, i0:i1_half, loc.right]
        Brin[p1, i0:i1_half, 0]  -= diff_Bru[p1, i0:i1_half, 0]  #/ sqrt_det_h_half[p1, i0:i1_half, loc.bottom]

        B2in[p0, -1, i0:i1_int] -= diff_B2u[p0, -1, i0:i1_int] #/ sqrt_det_h_int[p0, i0:i1_int, loc.right]
        B1in[p1, i0:i1_int, 0]  -= diff_B1u[p1, i0:i1_int, 0]  #/ sqrt_det_h_int[p1, i0:i1_int, loc.bottom]

    if (top == 'yy'):
        Brin[p0, i0:i1_half, -1] -= diff_Bru[p0, i0:i1_half, -1] #/ sqrt_det_h_half[p0, i0:i1_half, loc.top]
        Brin[p1, i0:i1_half, 0]  -= diff_Bru[p1, i0:i1_half, 0]  #/ sqrt_det_h_half[p1, i0:i1_half, loc.bottom]

        B1in[p0, i0:i1_int, -1] -= diff_B1u[p0, i0:i1_int, -1] #/ sqrt_det_h_int[p0, i0:i1_int, loc.top]
        B1in[p1, i0:i1_int, 0]  -= diff_B1u[p1, i0:i1_int, 0]  #/ sqrt_det_h_int[p1, i0:i1_int, loc.bottom]

    if (top == 'yx'):
        Brin[p0, i0:i1_half, -1] -= diff_Bru[p0, i0:i1_half, -1] #/ sqrt_det_h_half[p0, i0:i1_half, loc.top]
        Brin[p1, 0, i0:i1_half]  -= diff_Bru[p1, 0, i0:i1_half]  #/ sqrt_det_h_half[p1, i0:i1_half, loc.left]

        B1in[p0, i0:i1_int, -1] -= diff_B1u[p0, i0:i1_int, -1] #/ sqrt_det_h_int[p0, i0:i1_int, loc.top]
        B2in[p1, 0, i0:i1_int]  -= diff_B2u[p1, 0, i0:i1_int]  #/ sqrt_det_h_int[p1, i0:i1_int, loc.left]

def corners_B(p0, Brin, B1in, B2in):

    Brin[p0, 0, 0]   -= diff_Bru[p0, 0, 0]   * sig_cor / sig_in #/ sqrt_det_h_half[p0, 0, loc.bottom]
    Brin[p0, -1, 0]  -= diff_Bru[p0, -1, 0]  * sig_cor / sig_in #/ sqrt_det_h_half[p0, -1, loc.bottom] 
    Brin[p0, 0, -1]  -= diff_Bru[p0, 0, -1]  * sig_cor / sig_in #/ sqrt_det_h_half[p0, 0, loc.top]
    Brin[p0, -1, -1] -= diff_Bru[p0, -1, -1] * sig_cor / sig_in #/ sqrt_det_h_half[p0, -1, loc.top] 

    B1in[p0, 0, 0]   -= diff_B1u[p0, 0, 0] * sig_cor / sig_in  # / sqrt_det_h_int[p0, 0, loc.bottom]
    B1in[p0, -1, 0]  -= diff_B1u[p0, -1, 0] * sig_cor / sig_in # / sqrt_det_h_int[p0, -1, loc.bottom] 
    B1in[p0, 0, -1]  -= diff_B1u[p0, 0, -1] * sig_cor / sig_in # / sqrt_det_h_int[p0, 0, loc.top]
    B1in[p0, -1, -1] -= diff_B1u[p0, -1, -1] * sig_cor / sig_in# / sqrt_det_h_int[p0, -1, loc.top] 

    B2in[p0, 0, 0]   -= diff_B2u[p0, 0, 0] * sig_cor / sig_in   #/ sqrt_det_h_int[p0, 0, loc.bottom]
    B2in[p0, -1, 0]  -= diff_B2u[p0, -1, 0] * sig_cor / sig_in  #/ sqrt_det_h_int[p0, -1, loc.bottom] 
    B2in[p0, 0, -1]  -= diff_B2u[p0, 0, -1] * sig_cor / sig_in  #/ sqrt_det_h_int[p0, 0, loc.top]
    B2in[p0, -1, -1] -= diff_B2u[p0, -1, -1] * sig_cor / sig_in #/ sqrt_det_h_int[p0, -1, loc.top] 


def penalty_edges_D(dtin, Drin, D1in, D2in, Brin, B1in, B2in, Drout, D1out, D2out):

    diff_Dru[:, :, :] = 0.0
    diff_D1u[:, :, :] = 0.0
    diff_D2u[:, :, :] = 0.0
    
    for i in range(n_zeros):
        p0, p1 = index_row[i], index_col[i]
        compute_penalty_D(p0, p1, dtin, Drin, D1in, D2in, Brin, B1in, B2in)

    for i in range(n_zeros):
        p0, p1 = index_row[i], index_col[i]
        interface_D(p0, p1, Drout, D1out, D2out)

    corners_D(patches, Drout, D1out, D2out)

def penalty_edges_B(dtin, Drin, D1in, D2in, Brin, B1in, B2in, Brout, B1out, B2out):

    diff_Bru[:, :, :] = 0.0
    diff_B1u[:, :, :] = 0.0
    diff_B2u[:, :, :] = 0.0
    
    for i in range(n_zeros):
        p0, p1 = index_row[i], index_col[i]
        compute_penalty_B(p0, p1, dtin, Drin, D1in, D2in, Brin, B1in, B2in)

    for i in range(n_zeros):
        p0, p1 = index_row[i], index_col[i]
        interface_B(p0, p1, Brout, B1out, B2out)

    corners_B(patches, Brout, B1out, B2out)

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
    # return B0 * N.sin(th0)**3 * N.cos(3.0 * ph0)
    return B0 * r0 * r0 * N.cos(th0) * N.sin(th0) / sqrtdeth(r0, th0, ph0, a)

def func_Bth(r0, th0, ph0):
    # return - B0 * N.sin(th0)
    # return 0.0
   return - B0 * r0 * N.sin(th0)**2 / sqrtdeth(r0, th0, ph0, a)

def func_Bph(r0, th0, ph0):
    # return - B0 * (N.cos(ph0) / N.sin(th0) * N.sin(tilt)) / r0**4
    return 0.0

def InitialData():

    for patch in range(n_patches):

        fvec = (globals()["vec_sph_to_" + sphere[patch]])
        fcoord = (globals()["coord_" + sphere[patch] + "_to_sph"])

        for i in range(Nxi_int):
            for j in range(Neta_int):

                th0, ph0 = fcoord(xi_int[i], eta_int[j])
                # DrTMP = func_Br(r0, th0, ph0)
                DrTMP = 0.0 

                Dru[patch,  i, j] = DrTMP

        for i in range(Nxi_half):
            for j in range(Neta_half):

                th0, ph0 = fcoord(xi_half[i], eta_half[j])
                BrTMP = func_Br(r0, th0, ph0)
                # BrTMP = 0.0

                Bru[patch,  i, j] = BrTMP
                INBr[patch, i, j] = BrTMP
                    
        for i in range(Nxi_int):
            for j in range(Neta_half):

                    th0, ph0 = fcoord(xi_int[i], eta_half[j])
                    BtTMP = func_Bth(r0, th0, ph0)
                    BpTMP = func_Bph(r0, th0, ph0)
                    BCStmp = fvec(th0, ph0, BtTMP, BpTMP)

                    B1u[patch, i, j]  = BCStmp[0]
                    INB1[patch, i, j] = BCStmp[0]
                    D2u[patch, i, j] = 0.0

        for i in range(Nxi_half):
            for j in range(Neta_int):

                    th0, ph0 = fcoord(xi_half[i], eta_int[j])
                    BtTMP = func_Bth(r0, th0, ph0)
                    BpTMP = func_Bph(r0, th0, ph0)
                    BCStmp = fvec(th0, ph0, BtTMP, BpTMP)

                    B2u[patch, i, j]  = BCStmp[1]
                    INB2[patch, i, j] = BCStmp[1]
                    D1u[patch, i, j]  = 0.0

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

def plot_fields_unfolded_Br(it, vm):

    xi_grid_c, eta_grid_c = unflip_eq(xBr_grid, yBr_grid)
    xi_grid_d, eta_grid_d = unflip_eq(xBr_grid, yBr_grid)
    xi_grid_n, eta_grid_n = unflip_po(xBr_grid, yBr_grid)

    fig = P.figure(1, facecolor='w')
    ax = P.subplot(111)

    ax.pcolormesh(xBr_grid, yBr_grid, Bru[Sphere.A, :, :], cmap = "RdBu_r", vmin = - vm, vmax = vm)
    ax.pcolormesh(xBr_grid + N.pi / 2.0, yBr_grid, Bru[Sphere.B, :, :], cmap = "RdBu_r", vmin = - vm, vmax = vm)
    ax.pcolormesh(xBr_grid, yBr_grid - N.pi / 2.0, Bru[Sphere.S, :, :], cmap = "RdBu_r", vmin = - vm, vmax = vm)

    ax.pcolormesh(xi_grid_c + N.pi, eta_grid_c, Bru[Sphere.C, :, :], cmap = "RdBu_r", vmin = - vm, vmax = vm)
    ax.pcolormesh(xi_grid_d - N.pi / 2.0, eta_grid_d, Bru[Sphere.D, :, :], cmap = "RdBu_r", vmin = - vm, vmax = vm)
    ax.pcolormesh(xi_grid_n, eta_grid_n + N.pi / 2.0, Bru[Sphere.N, :, :], cmap = "RdBu_r", vmin = - vm, vmax = vm)
    
    P.title(r'$t={:.3f} R/c$'.format(FDUMP*it*dt/r0))
    
    figsave_png(fig, "../snapshots_2d/Br_" + str(it))

    P.close('all')

def plot_fields_unfolded_Dr(it, vm):

    xi_grid_c, eta_grid_c = unflip_eq(xEr_grid, yEr_grid)
    xi_grid_d, eta_grid_d = unflip_eq(xEr_grid, yEr_grid)
    xi_grid_n, eta_grid_n = unflip_po(xEr_grid, yEr_grid)

    fig = P.figure(1, facecolor='w')
    ax = P.subplot(111)

    ax.pcolormesh(xEr_grid, yEr_grid, Dru[Sphere.A, :, :], cmap = "RdBu_r", vmin = - vm, vmax = vm)
    ax.pcolormesh(xEr_grid + N.pi / 2.0, yEr_grid, Dru[Sphere.B, :, :], cmap = "RdBu_r", vmin = - vm, vmax = vm)
    ax.pcolormesh(xEr_grid, yEr_grid - N.pi / 2.0, Dru[Sphere.S, :, :], cmap = "RdBu_r", vmin = - vm, vmax = vm)

    ax.pcolormesh(xi_grid_c + N.pi, eta_grid_c, Dru[Sphere.C, :, :], cmap = "RdBu_r", vmin = - vm, vmax = vm)
    ax.pcolormesh(xi_grid_d - N.pi / 2.0, eta_grid_d, Dru[Sphere.D, :, :], cmap = "RdBu_r", vmin = - vm, vmax = vm)
    ax.pcolormesh(xi_grid_n, eta_grid_n + N.pi / 2.0, Dru[Sphere.N, :, :], cmap = "RdBu_r", vmin = - vm, vmax = vm)
    
    P.title(r'$t={:.3f} R/c$'.format(FDUMP*it*dt/r0))
    
    figsave_png(fig, "../snapshots_2d/Dr_" + str(it))

    P.close('all')

def plot_fields_unfolded_D1(it, vm):

    xi_grid_c, eta_grid_c = unflip_eq(xE1_grid, yE1_grid)
    xi_grid_d, eta_grid_d = unflip_eq(xE1_grid, yE1_grid)
    xi_grid_n, eta_grid_n = unflip_po(xE1_grid, yE1_grid)

    fig = P.figure(1, facecolor='w')
    ax = P.subplot(111)

    ax.pcolormesh(xE1_grid, yE1_grid, D1u[Sphere.A, :, :], cmap = "RdBu_r", vmin = - vm, vmax = vm)
    ax.pcolormesh(xE1_grid + N.pi / 2.0 + 0.1, yE1_grid, D1u[Sphere.B, :, :], cmap = "RdBu_r", vmin = - vm, vmax = vm)
    ax.pcolormesh(xE1_grid, yE1_grid - N.pi / 2.0 - 0.1, D1u[Sphere.S, :, :], cmap = "RdBu_r", vmin = - vm, vmax = vm)

    ax.pcolormesh(xi_grid_c + N.pi + 0.2, eta_grid_c, D1u[Sphere.C, :, :], cmap = "RdBu_r", vmin = - vm, vmax = vm)
    ax.pcolormesh(xi_grid_d - N.pi / 2.0 - 0.1, eta_grid_d, D1u[Sphere.D, :, :], cmap = "RdBu_r", vmin = - vm, vmax = vm)
    ax.pcolormesh(xi_grid_n, eta_grid_n + N.pi / 2.0 + 0.1, D1u[Sphere.N, :, :], cmap = "RdBu_r", vmin = - vm, vmax = vm)
    
    P.title(r'$t={:.3f} R/c$'.format(FDUMP*it*dt/r0))
    
    figsave_png(fig, "../snapshots_2d/D1u_" + str(it))

    P.close('all')

def plot_fields_unfolded_B1(it, vm):

    xi_grid_c, eta_grid_c = unflip_eq(xE2_grid, yE2_grid)
    xi_grid_d, eta_grid_d = unflip_eq(xE2_grid, yE2_grid)
    xi_grid_n, eta_grid_n = unflip_po(xE2_grid, yE2_grid)

    fig = P.figure(1, facecolor='w')
    ax = P.subplot(111)

    ax.pcolormesh(xE2_grid, yE2_grid, B1u[Sphere.A, :, :], cmap = "RdBu_r", vmin = - vm, vmax = vm)
    ax.pcolormesh(xE2_grid + N.pi / 2.0 + 0.1, yE2_grid, B1u[Sphere.B, :, :], cmap = "RdBu_r", vmin = - vm, vmax = vm)
    ax.pcolormesh(xE2_grid, yE2_grid - N.pi / 2.0 - 0.1, B1u[Sphere.S, :, :], cmap = "RdBu_r", vmin = - vm, vmax = vm)

    ax.pcolormesh(xi_grid_c + N.pi + 0.2, eta_grid_c, B1u[Sphere.C, :, :], cmap = "RdBu_r", vmin = - vm, vmax = vm)
    ax.pcolormesh(xi_grid_d - N.pi / 2.0 - 0.1, eta_grid_d, B1u[Sphere.D, :, :], cmap = "RdBu_r", vmin = - vm, vmax = vm)
    ax.pcolormesh(xi_grid_n, eta_grid_n + N.pi / 2.0 + 0.1, B1u[Sphere.N, :, :], cmap = "RdBu_r", vmin = - vm, vmax = vm)
    
    P.title(r'$t={:.3f} R/c$'.format(FDUMP*it*dt/r0))
    
    figsave_png(fig, "../snapshots_2d/B1u_" + str(it))

    P.close('all')

def plot_fields_unfolded_B2(it, vm):

    xi_grid_c, eta_grid_c = unflip_eq(xE1_grid, yE1_grid)
    xi_grid_d, eta_grid_d = unflip_eq(xE1_grid, yE1_grid)
    xi_grid_n, eta_grid_n = unflip_po(xE1_grid, yE1_grid)

    fig = P.figure(1, facecolor='w')
    ax = P.subplot(111)

    ax.pcolormesh(xE1_grid, yE1_grid, B2u[Sphere.A, :, :], cmap = "RdBu_r", vmin = - vm, vmax = vm)
    ax.pcolormesh(xE1_grid + N.pi / 2.0 + 0.1, yE1_grid, B2u[Sphere.B, :, :], cmap = "RdBu_r", vmin = - vm, vmax = vm)
    ax.pcolormesh(xE1_grid, yE1_grid - N.pi / 2.0 - 0.1, B2u[Sphere.S, :, :], cmap = "RdBu_r", vmin = - vm, vmax = vm)

    ax.pcolormesh(xi_grid_c + N.pi + 0.2, eta_grid_c, B2u[Sphere.C, :, :], cmap = "RdBu_r", vmin = - vm, vmax = vm)
    ax.pcolormesh(xi_grid_d - N.pi / 2.0 - 0.1, eta_grid_d, B2u[Sphere.D, :, :], cmap = "RdBu_r", vmin = - vm, vmax = vm)
    ax.pcolormesh(xi_grid_n, eta_grid_n + N.pi / 2.0 + 0.1, B2u[Sphere.N, :, :], cmap = "RdBu_r", vmin = - vm, vmax = vm)
    
    P.title(r'$t={:.3f} R/c$'.format(FDUMP*it*dt/r0))
    
    figsave_png(fig, "../snapshots_2d/B2u_" + str(it))

    P.close('all')
    
def plot_unfolded_metric(field):
    
    xi_grid_c, eta_grid_c = unflip_eq(xEr_grid, yEr_grid)
    xi_grid_d, eta_grid_d = unflip_eq(xEr_grid, yEr_grid)
    xi_grid_n, eta_grid_n = unflip_po(xEr_grid, yEr_grid)
    
    tab = (globals()[field])[:, :, :, 0]

    vm = N.max(N.abs(tab))

    fig = P.figure(1, facecolor='w')
    ax = P.subplot(111)

    ax.pcolormesh(xEr_grid, yEr_grid, tab[Sphere.A, :, :], cmap = "RdBu_r", vmin = - vm, vmax = vm)
    ax.pcolormesh(xEr_grid + N.pi / 2.0, yEr_grid, tab[Sphere.B, :, :], cmap = "RdBu_r", vmin = - vm, vmax = vm)
    ax.pcolormesh(xEr_grid, yEr_grid - N.pi / 2.0, tab[Sphere.S, :, :], cmap = "RdBu_r", vmin = - vm, vmax = vm)

    ax.pcolormesh(xi_grid_c + N.pi, eta_grid_c, tab[Sphere.C, :, :], cmap = "RdBu_r", vmin = - vm, vmax = vm)
    ax.pcolormesh(xi_grid_d - N.pi / 2.0, eta_grid_d, tab[Sphere.D, :, :], cmap = "RdBu_r", vmin = - vm, vmax = vm)
    ax.pcolormesh(xi_grid_n, eta_grid_n + N.pi / 2.0, tab[Sphere.N, :, :], cmap = "RdBu_r", vmin = - vm, vmax = vm)

########
# Initialization
########

idump = 0

Nt = 20000 # Number of iterations
FDUMP = 100 # Dump frequency
time = dt * N.arange(Nt)
energy = N.zeros((n_patches, Nt))

Bru0[:, :, :] = Bru[:, :, :]
B1u0[:, :, :] = B1u[:, :, :]
B2u0[:, :, :] = B2u[:, :, :]
Dru0[:, :, :] = Dru[:, :, :]
D1u0[:, :, :] = D1u[:, :, :]
D2u0[:, :, :] = D2u[:, :, :]

WriteCoordsHDF5()

########
# Main routine
########

for it in tqdm(range(Nt), "Progression"):
    
    average_field(patches, Bru, B1u, B2u, Bru0, B1u0, B2u0, Bru1, B1u1, B2u1)
    average_field(patches, Dru, D1u, D2u, Dru0, D1u0, D2u0, Dru1, D1u1, D2u1)
    
    contra_to_cov_D(patches, Dru1, D1u1, D2u1)
    compute_E_aux(patches, Drd, D1d, D2d, Bru, B1u, B2u)

    compute_diff_E(patches)
    push_B(patches, Bru1, B1u1, B2u1, dt, it)

    # # Penalty terms
    # penalty_edges_B(dt, Erd, E1d, E2d, Bru, B1u, B2u, Bru1, B1u1, B2u1)
    # penalty_edges_B(dt, Drd, D1d, D2d, Bru, B1u, B2u, Bru1, B1u1, B2u1)
    
    contra_to_cov_D(patches, Dru, D1u, D2u)
    compute_E_aux(patches, Drd, D1d, D2d, Bru1, B1u1, B2u1)

    Bru0[:, :, :] = Bru[:, :, :]
    B1u0[:, :, :] = B1u[:, :, :]
    B2u0[:, :, :] = B2u[:, :, :]

    compute_diff_E(patches)
    push_B(patches, Bru, B1u, B2u, dt, it)

    ##### TEST
    contra_to_cov_B(patches, Bru1, B1u1, B2u1)
    compute_H_aux(patches, Dru, D1u, D2u, Brd, B1d, B2d)
    cov_to_contra_H(patches, Hrd, H1d, H2d)

    # # Penalty terms
    # penalty_edges_B(dt, Erd, E1d, E2d, Hru, H1u, H2u, Bru, B1u, B2u)
    # penalty_edges_B(dt, Erd, E1d, E2d, Bru1, B1u1, B2u1, Bru, B1u, B2u)
    penalty_edges_B(dt, Drd, D1d, D2d, Bru1, B1u1, B2u1, Bru, B1u, B2u)

    average_field(patches, Bru, B1u, B2u, Bru0, B1u0, B2u0, Bru1, B1u1, B2u1)

    contra_to_cov_B(patches, Bru1, B1u1, B2u1)
    compute_H_aux(patches, Dru, D1u, D2u, Brd, B1d, B2d)

    compute_diff_H(patches)
    push_D(patches, Dru1, D1u1, D2u1, dt, it)

    # # Penalty terms
    # penalty_edges_D(dt, Dru, D1u, D2u, Hrd, H1d, H2d, Dru1, D1u1, D2u1)
    # penalty_edges_D(dt, Dru, D1u, D2u, Brd, B1d, B2d, Dru1, D1u1, D2u1)

    contra_to_cov_B(patches, Bru, B1u, B2u)
    compute_H_aux(patches, Dru1, D1u1, D2u1, Brd, B1d, B2d)

    Dru0[:, :, :] = Dru[:, :, :]
    D1u0[:, :, :] = D1u[:, :, :]
    D2u0[:, :, :] = D2u[:, :, :]

    compute_diff_H(patches)
    push_D(patches, Dru, D1u, D2u, dt, it)

    ##### TEST
    contra_to_cov_D(patches, Dru1, D1u1, D2u1)
    compute_E_aux(patches, Drd, D1d, D2d, Bru, B1u, B2u)
    cov_to_contra_E(patches, Erd, E1d, E2d)

    # # Penalty terms
    # penalty_edges_D(dt, Eru, E1u, E2u, Hrd, H1d, H2d, Dru, D1u, D2u)
    # penalty_edges_D(dt, Dru1, D1u1, D2u1, Hrd, H1d, H2d, Dru, D1u, D2u)
    penalty_edges_D(dt, Dru1, D1u1, D2u1, Brd, B1d, B2d, Dru, D1u, D2u)

    if ((it % FDUMP) == 0):
        plot_fields_unfolded_Br(idump, 0.5)
        plot_fields_unfolded_D1(idump, 2.0)
        plot_fields_unfolded_Dr(idump, 0.5)
        plot_fields_unfolded_B1(idump, 0.5)
        plot_fields_unfolded_B2(idump, 0.5)
        # WriteAllFieldsHDF5(idump)
        idump += 1
    