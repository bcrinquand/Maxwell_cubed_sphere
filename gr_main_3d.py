## BIG CHANGES
## - Change r-sweeps of pusher to reproduce curls
## - Change power of r in the diagonal metric components to r**2 instead of r**4
## - Output and initial data wrappers
## - Fixing electric field on the surface to reproduce aligned rotator profiles
## - Outer boundary damp to dipole

## TODO:
## - Outer and inner boundary to inclined dipole

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
from figure_module import *

# Import metric functions
from gr_metric import *

outdir = '/home/bcrinqua/GitHub/Maxwell_cubed_sphere/data/'

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
cfl = 0.6
Nr0 = 60
Nxi = 64
Neta = 64

# Spin parameter
a = 0.99

Nxi_int = Nxi + 1 # Number of integer points
Nxi_half = Nxi + 2 # Number of half-step points
Neta_int = Neta + 1 # Number of integer points
Neta_half = Neta + 2 # NUmber of half-step points

NG = 1 # Radial ghost cells
Nr = Nr0 + 2 * NG # Total number of radial points

r_min, r_max = 1.0, 12.0
xi_min, xi_max = - N.pi / 4.0, N.pi / 4.0
eta_min, eta_max = - N.pi / 4.0, N.pi / 4.0

dr = (r_max - r_min) / Nr0
dxi = (xi_max - xi_min) / Nxi
deta = (eta_max - eta_min) / Neta

# Define grids
r = r_min + N.arange(- NG, NG + Nr0, 1) * dr
r_yee = r + 0.5 * dr

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

# Define fields
Erd = N.zeros((n_patches, Nr, Nxi_int, Neta_int))
E1d = N.zeros((n_patches, Nr, Nxi_half, Neta_int))
E2d = N.zeros((n_patches, Nr, Nxi_int,  Neta_half))

Hrd = N.zeros((n_patches, Nr, Nxi_half, Neta_half))
H1d = N.zeros((n_patches, Nr, Nxi_int, Neta_half))
H2d = N.zeros((n_patches, Nr, Nxi_half,  Neta_int))

Bru = N.zeros((n_patches, Nr, Nxi_half, Neta_half))
B1u = N.zeros((n_patches, Nr, Nxi_int, Neta_half))
B2u = N.zeros((n_patches, Nr, Nxi_half,  Neta_int))
Brd = N.zeros((n_patches, Nr, Nxi_half, Neta_half))
B1d = N.zeros((n_patches, Nr, Nxi_int, Neta_half))
B2d = N.zeros((n_patches, Nr, Nxi_half,  Neta_int))

Dru = N.zeros((n_patches, Nr, Nxi_int, Neta_int))
D1u = N.zeros((n_patches, Nr, Nxi_half, Neta_int))
D2u = N.zeros((n_patches, Nr, Nxi_int,  Neta_half))
Drd = N.zeros((n_patches, Nr, Nxi_int, Neta_int))
D1d = N.zeros((n_patches, Nr, Nxi_half, Neta_int))
D2d = N.zeros((n_patches, Nr, Nxi_int,  Neta_half))

dE1d2 = N.zeros((n_patches, Nr, Nxi_half, Neta_half))
dE2d1 = N.zeros((n_patches, Nr, Nxi_half, Neta_half))
dErd1 = N.zeros((n_patches, Nr, Nxi_half,  Neta_int))
dErd2 = N.zeros((n_patches, Nr, Nxi_int, Neta_half))
dE1dr = N.zeros((n_patches, Nr, Nxi_half, Neta_int))
dE2dr = N.zeros((n_patches, Nr, Nxi_int, Neta_half))

dHrd1 = N.zeros((n_patches, Nr, Nxi_int,  Neta_half))
dHrd2 = N.zeros((n_patches, Nr, Nxi_half, Neta_int))
dH1d2 = N.zeros((n_patches, Nr, Nxi_int, Neta_int))
dH2d1 = N.zeros((n_patches, Nr, Nxi_int, Neta_int))
dH1dr = N.zeros((n_patches, Nr, Nxi_int, Neta_half))
dH2dr = N.zeros((n_patches, Nr, Nxi_half, Neta_int))

# Interface terms
diff_Br = N.zeros((n_patches, Nr, Nxi_half, Neta_half))
diff_D1 = N.zeros((n_patches, Nr, Nxi_half, Neta_int))
diff_D2 = N.zeros((n_patches, Nr, Nxi_int, Neta_half))
diff_Dr = N.zeros((n_patches, Nr, Nxi_int, Neta_int))
diff_B1 = N.zeros((n_patches, Nr, Nxi_int, Neta_half))
diff_B2 = N.zeros((n_patches, Nr, Nxi_half, Neta_int))

# Initial fields
Br0  = N.zeros((n_patches, Nr, Nxi_half, Neta_half))
B1u0 = N.zeros((n_patches, Nr, Nxi_int, Neta_half))
B2u0 = N.zeros((n_patches, Nr, Nxi_half,  Neta_int))

########
# Dump HDF5 output
########

def WriteFieldHDF5(it, field):

    outvec = (globals()[field])
    h5f = h5py.File(outdir + field + '_' + str(it).rjust(5, '0') + '.h5', 'w')

    for patch in range(6):
        h5f.create_dataset(field + str(patch), data=outvec[patch, :, :, :])

    h5f.close()

def WriteAllFieldsHDF5(idump):

    WriteFieldHDF5(idump, "Br")
    WriteFieldHDF5(idump, "B1u")
    WriteFieldHDF5(idump, "B2u")
    WriteFieldHDF5(idump, "Er")
    WriteFieldHDF5(idump, "E1u")
    WriteFieldHDF5(idump, "E2u")

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

hrrd = N.empty((Nr, Nxi_int, Neta_int, 7))
hr1d = N.empty((Nr, Nxi_int, Neta_int, 7))
hr2d = N.empty((Nr, Nxi_int, Neta_int, 7))
h11d = N.empty((Nr, Nxi_int, Neta_int, 7))
h12d = N.empty((Nr, Nxi_int, Neta_int, 7))
h22d = N.empty((Nr, Nxi_int, Neta_int, 7))
alpha= N.empty((Nr, Nxi_int, Neta_int, 7))
beta = N.empty((Nr, Nxi_int, Neta_int, 7))

sqrt_det_h_half = N.empty((Nr, Nxi_half))
h12d_half = N.empty((Nr, Nxi_half))
h11d_half = N.empty((Nr, Nxi_half))
h22d_half = N.empty((Nr, Nxi_half))

sqrt_det_h_int = N.empty((Nr, Nxi_int))
h12d_int = N.empty((Nr, Nxi_int))
h11d_int = N.empty((Nr, Nxi_int))
h22d_int = N.empty((Nr, Nxi_int))

for p in range(6):
    for i in range(Nxi_int):
        for j in range(Neta_int):
            for k in range(Nr):

                # 0 at (k, i, j)
                r0 = r[k]
                xi0 = xi_int[i]
                eta0 = eta_int[j]
                h11d[p, k, i, j, 0] = g11d(p, r0, xi0, eta0, a)
                h22d[p, k, i, j, 0] = g22d(p, r0, xi0, eta0, a)
                h12d[p, k, i, j, 0] = g12d(p, r0, xi0, eta0, a)
                hrrd[p, k, i, j, 0] = grrd(p, r0, xi0, eta0, a)
                hr1d[p, k, i, j, 0] = gr1d(p, r0, xi0, eta0, a)
                hr2d[p, k, i, j, 0] = gr2d(p, r0, xi0, eta0, a)
                alpha[p, k, i, j, 0]=  alphas(p, r0, xi0, eta0, a)
                beta[p, k, i, j, 0] =  betaru(p, r0, xi0, eta0, a)
                
                # 1 at (k, i + 1/2, j)
                r0 = r[k]
                xi0  = xi_int[i] + 0.5 * dxi
                eta0 = eta_int[j]
                h11d[p, k, i, j, 0] = g11d(p, r0, xi0, eta0, a)
                h22d[p, k, i, j, 0] = g22d(p, r0, xi0, eta0, a)
                h12d[p, k, i, j, 0] = g12d(p, r0, xi0, eta0, a)
                hrrd[p, k, i, j, 0] = grrd(p, r0, xi0, eta0, a)
                hr1d[p, k, i, j, 0] = gr1d(p, r0, xi0, eta0, a)
                hr2d[p, k, i, j, 0] = gr2d(p, r0, xi0, eta0, a)
                alpha[p, k, i, j, 0]=  alphas(p, r0, xi0, eta0, a)
                beta[p, k, i, j, 0] =  betaru(p, r0, xi0, eta0, a)

                # 2 at (k, i, j + 1/2)
                r0 = r[k]
                xi0  = xi_int[i]
                eta0 = eta_int[j] + 0.5 * deta
                h11d[p, k, i, j, 0] = g11d(p, r0, xi0, eta0, a)
                h22d[p, k, i, j, 0] = g22d(p, r0, xi0, eta0, a)
                h12d[p, k, i, j, 0] = g12d(p, r0, xi0, eta0, a)
                hrrd[p, k, i, j, 0] = grrd(p, r0, xi0, eta0, a)
                hr1d[p, k, i, j, 0] = gr1d(p, r0, xi0, eta0, a)
                hr2d[p, k, i, j, 0] = gr2d(p, r0, xi0, eta0, a)
                alpha[p, k, i, j, 0]=  alphas(p, r0, xi0, eta0, a)
                beta[p, k, i, j, 0] =  betaru(p, r0, xi0, eta0, a)

                # 3 at (k, i + 1/2, j + 1/2)
                r0 = r[k]
                xi0  = xi_int[i] + 0.5 * dxi
                eta0 = eta_int[j] + 0.5 * deta
                h11d[p, k, i, j, 0] = g11d(p, r0, xi0, eta0, a)
                h22d[p, k, i, j, 0] = g22d(p, r0, xi0, eta0, a)
                h12d[p, k, i, j, 0] = g12d(p, r0, xi0, eta0, a)
                hrrd[p, k, i, j, 0] = grrd(p, r0, xi0, eta0, a)
                hr1d[p, k, i, j, 0] = gr1d(p, r0, xi0, eta0, a)
                hr2d[p, k, i, j, 0] = gr2d(p, r0, xi0, eta0, a)
                alpha[p, k, i, j, 0]=  alphas(p, r0, xi0, eta0, a)
                beta[p, k, i, j, 0] =  betaru(p, r0, xi0, eta0, a)

                # 4 at (k + 1/2, i, j)
                r0 = r_yee[k]
                xi0  = xi_int[i]
                eta0 = eta_int[j]
                h11d[p, k, i, j, 0] = g11d(p, r0, xi0, eta0, a)
                h22d[p, k, i, j, 0] = g22d(p, r0, xi0, eta0, a)
                h12d[p, k, i, j, 0] = g12d(p, r0, xi0, eta0, a)
                hrrd[p, k, i, j, 0] = grrd(p, r0, xi0, eta0, a)
                hr1d[p, k, i, j, 0] = gr1d(p, r0, xi0, eta0, a)
                hr2d[p, k, i, j, 0] = gr2d(p, r0, xi0, eta0, a)
                alpha[p, k, i, j, 0]=  alphas(p, r0, xi0, eta0, a)
                beta[p, k, i, j, 0] =  betaru(p, r0, xi0, eta0, a)

                # 5 at (k + 1/2, i + 1/2, j)
                r0 = r_yee[k]
                xi0  = xi_int[i] + 0.5 * dxi
                eta0 = eta_int[j]
                h11d[p, k, i, j, 0] = g11d(p, r0, xi0, eta0, a)
                h22d[p, k, i, j, 0] = g22d(p, r0, xi0, eta0, a)
                h12d[p, k, i, j, 0] = g12d(p, r0, xi0, eta0, a)
                hrrd[p, k, i, j, 0] = grrd(p, r0, xi0, eta0, a)
                hr1d[p, k, i, j, 0] = gr1d(p, r0, xi0, eta0, a)
                hr2d[p, k, i, j, 0] = gr2d(p, r0, xi0, eta0, a)
                alpha[p, k, i, j, 0]=  alphas(p, r0, xi0, eta0, a)
                beta[p, k, i, j, 0] =  betaru(p, r0, xi0, eta0, a)

                # 6 at (k + 1/2, i, j + 1/2)
                r0 = r_yee[k]
                xi0  = xi_int[i]
                eta0 = eta_int[j] + 0.5 * deta
                h11d[p, k, i, j, 0] = g11d(p, r0, xi0, eta0, a)
                h22d[p, k, i, j, 0] = g22d(p, r0, xi0, eta0, a)
                h12d[p, k, i, j, 0] = g12d(p, r0, xi0, eta0, a)
                hrrd[p, k, i, j, 0] = grrd(p, r0, xi0, eta0, a)
                hr1d[p, k, i, j, 0] = gr1d(p, r0, xi0, eta0, a)
                hr2d[p, k, i, j, 0] = gr2d(p, r0, xi0, eta0, a)
                alpha[p, k, i, j, 0]=  alphas(p, r0, xi0, eta0, a)
                beta[p, k, i, j, 0] =  betaru(p, r0, xi0, eta0, a)

# Define sqrt of determinant of metric
sqrt_det_g = N.sqrt(hrrd*h11d*h22d + 2.0 * hr1d*hr2d*h12d - h22d*hr2d*hr2d - hrrd*h12d*h12d - h22d*hr1d*hr1d)

# Define sqrt(det(g)), g11d, g22d, g12d on the edge of a patch, for convenience
for i in range(Nxi_half): 
    for k in range(Nr): 
        r0 = r[k]
        X = N.tan(xi_half[i])
        Y = N.tan(eta_int[0])
        C = N.sqrt(1.0 + X * X)
        D = N.sqrt(1.0 + Y * Y)
        delta = N.sqrt(1.0 + X * X + Y * Y)
        g11d_half[k, i] = (r0 * C * C * D / (delta * delta))**2
        g22d_half[k, i] = (r0 * C * D * D / (delta * delta))**2
        g12d_half[k, i] = - r0 * r0 * X * Y * C * C * D * D / (delta)**4
        sqrt_det_g_half[k, i] = N.sqrt(g11d_half[k, i] * g22d_half[k, i] - g12d_half[k, i] * g12d_half[k, i])

for i in range(Nxi_int): 
    for k in range(Nr): 
        r0 = r_yee[k]
        X = N.tan(xi_int[i])
        Y = N.tan(eta_int[0])
        C = N.sqrt(1.0 + X * X)
        D = N.sqrt(1.0 + Y * Y)
        delta = N.sqrt(1.0 + X * X + Y * Y)
        g11d_int[k, i] = (r0 * C * C * D / (delta * delta))**2
        g22d_int[k, i] = (r0 * C * D * D / (delta * delta))**2
        g12d_int[k, i] = - r0 * r0 * X * Y * C * C * D * D / (delta)**4
        sqrt_det_g_int[k, i] = N.sqrt(g11d_half[k, i] * g22d_half[k, i] - g12d_half[k, i] * g12d_half[k, i])

sqrt_gd_half = N.sqrt(g22d_half)
sqrt_gd_int = N.sqrt(g22d_int)

# Time step
dt = cfl * N.min(1.0 / N.sqrt(1.0 / (dr * dr) + g11d / (sqrt_det_g * sqrt_det_g) / (dxi * dxi) + g22d / (sqrt_det_g * sqrt_det_g) / (deta * deta) - 2.0 * g12d / (sqrt_det_g * sqrt_det_g) / (dxi * deta)))
print("delta t = {}".format(dt))

########
# Interpolation routine
########

def interp(arr_in, xA, xB):
    f = interpolate.interp1d(xA, arr_in, axis = 1, kind='linear', fill_value=(0,0), bounds_error=False)
    return f(xB)

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

    dH1dr[p, NG:(Nr0 + NG), :, :] = (H1d[p, NG:(Nr0 + NG), :, :] - N.roll(H1d, 1, axis = 1)[p, NG:(Nr0 + NG), :, :]) / dr
    dH2dr[p, NG:(Nr0 + NG), :, :] = (H2d[p, NG:(Nr0 + NG), :, :] - N.roll(H2d, 1, axis = 1)[p, NG:(Nr0 + NG), :, :]) / dr

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

    dE1d2[p, :, :, 0] = (- 0.50 * E1d[p, :, :, 0] + 0.50 * E1d[p, :, :, 1]) / dxi / P_half_2[0]
    dE1d2[p, :, :, 1] = (- 0.25 * E1d[p, :, :, 0] + 0.25 * E1d[p, :, :, 1]) / dxi / P_half_2[1]
    dE1d2[p, :, :, 2] = (- 0.25 * E1d[p, :, :, 0] - 0.75 * E1d[p, :, :, 1] + E1d[p, :, :, 2]) / dxi / P_half_2[2]
    dE1d2[p, :, :, Neta_half - 3] = (- E1d[p, :, :, -3] + 0.75 * E1d[p, :, :, -2] + 0.25 * E1d[p, :, :, -1]) / deta / P_half_2[Nxi_half - 3]
    dE1d2[p, :, :, Neta_half - 2] = (- 0.25 * E1d[p, :, :, -2] + 0.25 * E1d[p, :, :, -1]) / deta / P_half_2[Nxi_half - 2]
    dE1d2[p, :, :, Neta_half - 1] = (- 0.50 * E1d[p, :, :, -2] + 0.50 * E1d[p, :, :, -1]) / deta / P_half_2[Nxi_half - 1]
    dE1d2[p, :, :, 3:(Neta_half - 3)] = (E1d[p, :, :, 3:(Neta_half - 3)] - N.roll(E1d, 1, axis = 3)[p, :, :, 3:(Neta_half - 3)]) / deta

    dE1dr[p, NG:(Nr0 + NG), :, :] = (N.roll(E1d, -1, axis = 1)[p, NG:(Nr0 + NG), :, :] - E1d[p, NG:(Nr0 + NG), :, :]) / dr
    dE2dr[p, NG:(Nr0 + NG), :, :] = (N.roll(E2d, -1, axis = 1)[p, NG:(Nr0 + NG), :, :] - E2d[p, NG:(Nr0 + NG), :, :]) / dr

    dErd1[p, :, 0, :] = (- 0.50 * Erd[p, :, 0, :] + 0.50 * Erd[p, :, 1, :]) / dxi / P_half_2[0]
    dErd1[p, :, 1, :] = (- 0.25 * Erd[p, :, 0, :] + 0.25 * Erd[p, :, 1, :]) / dxi / P_half_2[1]
    dErd1[p, :, 2, :] = (- 0.25 * Erd[p, :, 0, :] - 0.75 * Erd[p, :, 1, :] + Erd[p, :, 2, :]) / dxi / P_half_2[2]
    dErd1[p, :, Nxi_half - 3, :] = (- Erd[p, :, -3, :] + 0.75 * Erd[p, :, -2, :] + 0.25 * Erd[p, :, -1, :]) / dxi / P_half_2[Nxi_half - 3]
    dErd1[p, :, Nxi_half - 2, :] = (- 0.25 * Erd[p, :, -2, :] + 0.25 * Erd[p, :, -1, :]) / dxi / P_half_2[Nxi_half - 2]
    dErd1[p, :, Nxi_half - 1, :] = (- 0.5 * Erd[p, :, -2, :] + 0.5 * Erd[p, :, -1, :]) / dxi / P_half_2[Nxi_half - 1]
    dErd1[p, :, 3:(Nxi_half - 3), :] = (Erd[p, :, 3:(Nxi_half - 3), :] - N.roll(Er, 1, axis = 2)[p, :, 3:(Nxi_half - 3), :]) / dxi

    dErd2[p, :, :, 0] = (- 0.50 * Erd[p, :, :, 0] + 0.50 * Erd[p, :, :, 1]) / dxi / P_half_2[0]
    dErd2[p, :, :, 1] = (- 0.25 * Erd[p, :, :, 0] + 0.25 * Erd[p, :, :, 1]) / dxi / P_half_2[1]
    dErd2[p, :, :, 2] = (- 0.25 * Erd[p, :, :, 0] - 0.75 * Erd[p, :, :, 1] + Erd[p, :, :, 2]) / dxi / P_half_2[2]
    dErd2[p, :, :, Neta_half - 3] = (- Erd[p, :, :, -3] + 0.75 * Erd[p, :, :, -2] + 0.25 * Erd[p, :, :, -1]) / deta / P_half_2[Nxi_half - 3]
    dErd2[p, :, :, Neta_half - 2] = (- 0.25 * Erd[p, :, :, -2] + 0.25 * Erd[p, :, :, -1]) / deta / P_half_2[Nxi_half - 2]
    dErd2[p, :, :, Neta_half - 1] = (- 0.50 * Erd[p, :, :, -2] + 0.50 * Erd[p, :, :, -1]) / deta / P_half_2[Nxi_half - 1]
    dErd2[p, :, :, 3:(Neta_half - 3)] = (Erd[p, :, :, 3:(Neta_half - 3)] - N.roll(Er, 1, axis = 3)[p, :, :, 3:(Neta_half - 3)]) / deta

########
# Single-patch push routines
########

def push_D(p, dtin):

        Dru[p, :, :, :] += dtin * (dH2d1[p, :, :, :] - dH1d2[p, :, :, :]) / sqrt_det_g[p, :, :, :, 4] 

        # Interior
        D1u[p, :, 1:-1, :] += dtin * (dHrd2[p, :, 1:-1, :] - dH2dr[p, :, 1:-1, :]) / sqrt_det_g[p, :, 0:-1, :, 1] 
        # Left edge
        D1u[p, :, 0, :] += dtin * (dHrd2[p, :, 0, :] - dH2dr[p, :, 0, :]) / sqrt_det_g[p, :, 0, :, 0] 
        # Right edge
        D1u[p, :, -1, :] += dtin * (dHrd2[p, :, -1, :] - dH2dr[p, :, -1, :]) / sqrt_det_g[p, :, -1, :, 0]

        # Interior
        D2u[p, :, :, 1:-1] += dtin * (dH1dr[p, :, :, 1:-1] - dHrd1[p, :, :, 1:-1]) / sqrt_det_g[p, :, :, 0:-1, 2]
        # Bottom edge
        D2u[p, :, :, 0] += dtin * (dH1dr[p, :, :, 0] - dHrd1[p, :, :, 0]) / sqrt_det_g[p, :, :, 0, 0]
        # Top edge
        D2u[p, :, :, -1] += dtin * (dH1dr[p, :, :, -1] - dHrd1[p, :, :, -1]) / sqrt_det_g[p, :, :, -1, 0]

def push_B(p, dtin):
        
        # Interior
        Bru[p, :, 1:-1, 1:-1] += dtin * (dE1d2[p, :, 1:-1, 1:-1] - dE2d1[p, :, 1:-1, 1:-1]) / sqrt_det_g[:, 0:-1, 0:-1, 3] 
        # Left edge
        Bru[p, :, 0, 1:-1] += dtin * (dE1d2[p, :, 0, 1:-1] - dE2d1[p, :, 0, 1:-1]) / sqrt_det_g[:, 0, 0:-1, 2] 
        # Right edge
        Bru[p, :, -1, 1:-1] += dtin * (dE1d2[p, :, -1, 1:-1] - dE2d1[p, :, -1, 1:-1]) / sqrt_det_g[:, -1, 0:-1, 2] 
        # Bottom edge
        Bru[p, :, 1:-1, 0] += dtin * (dE1d2[p, :, 1:-1, 0] - dE2d1[p, :, 1:-1, 0]) / sqrt_det_g[:, 0:-1, 0, 1] 
        # Top edge
        Bru[p, :, 1:-1, -1] += dtin * (dE1d2[p, :, 1:-1, -1] - dE2d1[p, :, 1:-1, -1]) / sqrt_det_g[:, 0:-1, -1, 1] 
        # Bottom left corner
        Bru[p, :, 0, 0] += dtin * (dE1d2[p, :, 0, 0] - dE2d1[p, :, 0, 0]) / sqrt_det_g[:, 0, 0, 0] 
        # Bottom right corner
        Bru[p, :, -1, 0] += dtin * (dE1d2[p, :, -1, 0] - dE2d1[p, :, -1, 0]) / sqrt_det_g[:, -1, 0, 0] 
        # Top left corner
        Bru[p, :, 0, -1] += dtin * (dE1d2[p, :, 0, -1] - dE2d1[p, :, 0, -1]) / sqrt_det_g[:, 0, -1, 0] 
        # Top right corner
        Bru[p, :, -1, -1] += dtin * (dE1d2[p, :, -1, -1] - dE2d1[p, :, -1, -1]) / sqrt_det_g[:, -1, -1, 0] 

        # Interior
        B1u[p, :, :, 1:-1] += dtin * (dE2dr[p, :, :, 1:-1] - dErd2[p, :, :, 1:-1]) / sqrt_det_g[:, :, 0:-1, 6]
        # Bottom edge
        B1u[p, :, :, 0] += dtin * (dE2dr[p, :, :, 0] - dErd2[p, :, :, 0]) / sqrt_det_g[:, :, 0, 4]
        # Top edge
        B1u[p, :, :, -1] += dtin * (dE2dr[p, :, :, -1] - dErd2[p, :, :, -1]) / sqrt_det_g[:, :, -1, 4]

        # Interior
        B2u[p, :, 1:-1, :] += dtin * (dErd1[p, :, 1:-1, :] - dE1dr[p, :, 1:-1, :]) / sqrt_det_g[:, 0:-1, :, 5] 
        # Left edge
        B2u[p, :, 0, :] += dtin * (dErd1[p, :, 0, :] - dE1dr[p, :, 0, :]) / sqrt_det_g[:, 0, :, 4] 
        # Right edge
        B2u[p, :, -1, :] += dtin * (dErd1[p, :, -1, :] - dE1dr[p, :, -1, :]) / sqrt_det_g[:, -1, :, 4]

########
# Auxiliary field computation
########

def contra_to_cov_D(p, Drin, D1in, D2in):

    ########
    # Dr
    ########

    # Interior
    Drd[p, :, 1:-1, 1:-1] = hrrd[p, :, 1:-1, 1:-1, 4] * Drin[p, :, 1:-1, 1:-1] \
                          + 0.25 * hr1d[p, :, 1:-1, 1:-1, 4] * (D1in[p, :, 1:-2, 1:-1] + N.roll(N.roll(D1in, -1, axis = 1), 1, axis = 2)[p, :, 1:-2, 1:-1]  \
                                                                + N.roll(D1in, -1, axis = 1)[p, :, 1:-2, 1:-1] + N.roll(D1in, 1, axis = 2)[p, :, 1:-2, 1:-1]) \
                          + 0.25 * hr2d[p, :, 1:-1, 1:-1, 4] * (D2in[p, :, 1:-1, 1:-2] + N.roll(N.roll(D2in, -1, axis = 1), 1, axis = 3)[p, :, 1:-1, 1:-2]  \
                                                                + N.roll(D2in, -1, axis = 1)[p, :, 1:-1, 1:-2] + N.roll(D2in, 1, axis = 3)[p, :, 1:-1, 1:-2]) \

    # Left edge
    Drd[p, :, 0, 1:-1] = hrrd[p, :, 0, 1:-1, 4] * Drin[p, :, 0, 1:-1] \
                       + 0.5 * hr1d[p, :, 0, 1:-1, 4]  * (D1in[p, :, 0, 1:-1] + N.roll(D1in, -1, axis = 1)[p, :, 0, 1:-1]) \
                       + 0.25 * hr2d[p, :, 0, 1:-1, 4] * (D2in[p, :, 0, 1:-2] + N.roll(N.roll(D2in, -1, axis = 1), 1, axis = 3)[p, :, 0, 1:-2]  \
                                                       + N.roll(D2in, -1, axis = 1)[p, :, 0, 1:-2] + N.roll(D2in, 1, axis = 3)[p, :, 0, 1:-2]) \
    # Right edge
    Drd[p, :, -1, 1:-1] = hrrd[p, :, -1, 1:-1, 4] * Drin[p, :, -1, 1:-1] \
                       + 0.5 * hr1d[p, :, -1, 1:-1, 4]  * (D1in[p, :, -1, 1:-1] + N.roll(D1in, -1, axis = 1)[p, :, -1, 1:-1]) \
                       + 0.25 * hr2d[p, :, -1, 1:-1, 4] * (D2in[p, :, -1, 1:-2] + N.roll(N.roll(D2in, -1, axis = 1), 1, axis = 3)[p, :, -1, 1:-2]  \
                                                       + N.roll(D2in, -1, axis = 1)[p, :, -1, 1:-2] + N.roll(D2in, 1, axis = 3)[p, :, -1, 1:-2]) \
    # Bottom edge
    Drd[p, :, 1:-1, 0] = hrrd[p, :, 1:-1, 0, 4] * Drin[p, :, 1:-1, 0] \
                       + 0.25 * hr1d[p, :, 1:-1, 0, 4] * (D1in[p, :, 1:-2, 0] + N.roll(N.roll(D1in, -1, axis = 1), 1, axis = 2)[p, :, 1:-2, 0]  \
                                                           + N.roll(D1in, -1, axis = 1)[p, :, 1:-2, 0] + N.roll(D1in, 1, axis = 2)[p, :, 1:-2, 0]) \
                       + 0.5 * hr2d[p, :, 1:-1, 0, 4] * (D2in[p, :, 1:-1, 0] + N.roll(D2in, -1, axis = 1)[p, :, 1:-1, 0])
    # Top edge
    Drd[p, :, 1:-1, -1] = hrrd[p, :, 1:-1, -1, 4] * Drin[p, :, 1:-1, -1] \
                       + 0.25 * hr1d[p, :, 1:-1, -1, 4] * (D1in[p, :, 1:-2, -1] + N.roll(N.roll(D1in, -1, axis = 1), 1, axis = 2)[p, :, 1:-2, -1]  \
                                                           + N.roll(D1in, -1, axis = 1)[p, :, 1:-2, -1] + N.roll(D1in, 1, axis = 2)[p, :, 1:-2, -1]) \
                       + 0.5 * hr2d[p, :, 1:-1, -1, 4] * (D2in[p, :, 1:-1, -1] + N.roll(D2in, -1, axis = 1)[p, :, 1:-1, -1])
    # Bottom-left corner
    Drd[p, :, 0, 0] = hrrd[p, :, 0, 0, 4] * Drin[p, :, 0, 0] \
                    + 0.5 * hr1d[p, :, 0, 0, 4] * (D1in[p, :, 0, 0] + N.roll(D1in, -1, axis = 1)[p, :, 0, 0]) \
                    + 0.5 * hr2d[p, :, 0, 0, 4] * (D2in[p, :, 0, 0] + N.roll(D2in, -1, axis = 1)[p, :, 0, 0])
    # Top-left corner
    Drd[p, :, 0, -1] = hrrd[p, :, 0, -1, 4] * Drin[p, :, 0, -1] \
                    + 0.5 * hr1d[p, :, 0, -1, 4] * (D1in[p, :, 0, -1] + N.roll(D1in, -1, axis = 1)[p, :, 0, -1]) \
                    + 0.5 * hr2d[p, :, 0, -1, 4] * (D2in[p, :, 0, -1] + N.roll(D2in, -1, axis = 1)[p, :, 0, -1])
    # Bottom-right corner
    Drd[p, :, -1, 0] = hrrd[p, :, -1, 0, 4] * Drin[p, :, -1, 0] \
                    + 0.5 * hr1d[p, :, -1, 0, 4] * (D1in[p, :, -1, 0] + N.roll(D1in, -1, axis = 1)[p, :, -1, 0]) \
                    + 0.5 * hr2d[p, :, -1, 0, 4] * (D2in[p, :, -1, 0] + N.roll(D2in, -1, axis = 1)[p, :, -1, 0])
    # Top-right corner
    Drd[p, :, -1, -1] = hrrd[p, :, -1, -1, 4] * Drin[p, :, -1, -1] \
                    + 0.5 * hr1d[p, :, -1, -1, 4] * (D1in[p, :, -1, -1] + N.roll(D1in, -1, axis = 1)[p, :, -1, -1]) \
                    + 0.5 * hr2d[p, :, -1, -1, 4] * (D2in[p, :, -1, -1] + N.roll(D2in, -1, axis = 1)[p, :, -1, -1])

    ########
    # Dxi
    ########

    # Interior
    D1d[p, :, 1:-1, 1:-1] = h11d[p, :, 0:-1, 1:-1, 1] * D1in[p, :, 1:-1, 1:-1] \
                          + 0.25 * h12d[p, :, 0:-1, 1:-1, 1] * (D2in[p, :, 1:, 1:-2] + N.roll(N.roll(D2in, 1, axis = 2), -1, axis = 3)[p, :, 1:, 1:-2] \
                          + N.roll(D2in, 1, axis = 2)[p, :, 1:, 1:-2] + N.roll(D2in, -1, axis = 3)[p, :, 1:, 1:-2]) \
                          + 0.25 * hr1d[p, :, 0:-1, 1:-1, 1] * (Drin[p, :, :-1, 1:-1] + N.roll(N.roll(Drin, 1, axis = 1), -1, axis = 2)[p, :, :-1, 1:-1] \
                          + N.roll(Drin, 1, axis = 1)[p, :, :-1, 1:-1] + N.roll(Drin, -1, axis = 2)[p, :, :-1, 1:-1])

    # Left edge
    D1d[p, :, 0, 1:-1] = h11d[p, :, 0, 1:-1, 0] * D1in[p, :, 0, 1:-1] \
                       + 0.5 * h12d[p, :, 0, 1:-1, 0] * (D2in[p, :, 0, 1:-2] + N.roll(D2in, -1, axis = 3)[p, :, 0, 1:-2]) \
                       + 0.5 * hr1d[p, :, 0, 1:-1, 0] * (Drin[p, :, 0, 1:-1] + N.roll(Drin, 1, axis = 1)[p, :, 0, 1:-1])
    # Right edge
    D1d[p, :, -1, 1:-1] = h11d[p, :, -1, 1:-1, 0] * D1in[p, :, -1, 1:-1] \
                        + 0.5 * h12d[p, :, -1, 1:-1, 0] * (D2in[p, :, -1, 1:-2] + N.roll(D2in, -1, axis = 3)[p, :, -1, 1:-2]) \
                        + 0.5 * hr1d[p, :, -1, 1:-1, 0] * (Drin[p, :, -1, 1:-1] + N.roll(Drin, 1, axis = 1)[p, :, -1, 1:-1])
    # Bottom edge
    D1d[p, :, 1:-1, 0] = h11d[p, :, 0:-1, 0, 1] * D1in[p, :, 1:-1, 0] \
                       + 0.5 * h12d[p, :, 0:-1, 0, 1] * (D2in[p, :, 1:, 0] + N.roll(D2in, 1, axis = 2)[p, :, 1:, 0]) \
                       + 0.25 * hr1d[p, :, 0:-1, 0, 1] * (Drin[p, :, :-1, 0] + N.roll(N.roll(Drin, 1, axis = 1), -1, axis = 2)[p, :, :-1, 0] \
                       + N.roll(Drin, 1, axis = 1)[p, :, :-1, 0] + N.roll(Drin, -1, axis = 2)[p, :, :-1, 0])
    # Top edge
    D1d[p, :, 1:-1, -1] = h11d[p, :, -1:-1, -1, 1] * D1in[p, :, 1:-1, -1] \
                        + 0.5 * h12d[p, :, -1:-1, -1, 1] * (D2in[p, :, 1:, -1] + N.roll(D2in, 1, axis = 2)[p, :, 1:, -1]) \
                        + 0.25 * hr1d[p, :, -1:-1, -1, 1] * (Drin[p, :, :-1, -1] + N.roll(N.roll(Drin, 1, axis = 1), -1, axis = 2)[p, :, :-1, -1] \
                        + N.roll(Drin, 1, axis = 1)[p, :, :-1, -1] + N.roll(Drin, -1, axis = 2)[p, :, :-1, -1])
    # Bottom-left corner
    D1d[p, :, 0, 0] = h11d[p, :, 0, 0, 0] * D1in[p, :, 0, 0] \
                    + h12d[p, :, 0, 0, 0] * D2in[p, :, 0, 0] \
                    + 0.5 * hr1d[p, :, 0, 0, 0] * (Drin[p, :, 0, 0] + N.roll(Drin, 1, axis = 1)[p, :, 0, 0])
    # Top-left corner
    D1d[p, :, 0, -1] = h11d[p, :, 0, -1, 0] * D1in[p, :, 0, -1] \
                    + h12d[p, :, 0, -1, 0] * D2in[p, :, 0, -1] \
                    + 0.5 * hr1d[p, :, 0, -1, 0] * (Drin[p, :, 0, -1] + N.roll(Drin, 1, axis = 1)[p, :, 0, -1])
    # Bottom-right corner
    D1d[p, :, -1, 0] = h11d[p, :, -1, 0, 0] * D1in[p, :, -1, 0] \
                     + h12d[p, :, -1, 0, 0] * D2in[p, :, -1, 0] \
                     + 0.5 * hr1d[p, :, -1, 0, 0] * (Drin[p, :, -1, 0] + N.roll(Drin, 1, axis = 1)[p, :, -1, 0])
    # Top-right corner
    D1d[p, :, -1, -1] = h11d[p, :, -1, -1, 0] * D1in[p, :, -1, -1] \
                     + h12d[p, :, -1, -1, 0] * D2in[p, :, -1, -1] \
                     + 0.5 * hr1d[p, :, -1, -1, 0] * (Drin[p, :, -1, -1] + N.roll(Drin, 1, axis = 1)[p, :, -1, -1])

    ########
    # Deta
    ########

    # Interior
    D2d[p, :, 1:-1, 1:-1] = h22d[p, :, 1:-1, 0:-1, 2] * D2in[p, :, 1:-1, 1:-1] \
                          + 0.25 * h12d[p, :, 1:-1, 0:-1, 2] * (D1in[p, :, 1:-2, 1:] + N.roll(N.roll(D1in, -1, axis = 2), 1, axis = 3)[p, :, 1:-2, 1:] \
                          + N.roll(D1in, -1, axis = 2)[p, :, 1:-2, 1:] + N.roll(D1in, 1, axis = 3)[p, :, 1:-2, 1:]) \
                          + 0.25 * hr2d[p, :, 1:-1, :-1, 2] * (Drin[p, :, 1:-1, :-1] + N.roll(N.roll(Drin, -1, axis = 2), 1, axis = 3)[p, :, 1:-1, :-1] \
                          + N.roll(Drin, 1, axis = 1)[p, :, 1:-1, :-1] + N.roll(Drin, 1, axis = 3)[p, :, 1:-1, :-1])

    # Left edge
    D2d[p, :, 0, 1:-1] = h22d[p, :, 0, 0:-1, 2] * D2in[p, :, 0, 1:-1] \
                          + 0.5 * h12d[p, :, 0, 0:-1, 2] * (D1in[p, :, 0, 1:] + N.roll(D1in, 1, axis = 3)[p, :, 0, 1:]) \
                          + 0.25 * hr2d[p, :, 0, :-1, 2] * (Drin[p, :, 0, :-1] + N.roll(N.roll(Drin, -1, axis = 2), 1, axis = 3)[p, :, 0, :-1] \
                          + N.roll(Drin, 1, axis = 1)[p, :, 0, :-1] + N.roll(Drin, 1, axis = 3)[p, :, 0, :-1])

    # Right edge
    D2d[p, :, -1, 1:-1] = h22d[p, :, -1, 0:-1, 2] * D2in[p, :, -1, 1:-1] \
                          + 0.5 * h12d[p, :, -1, 0:-1, 2] * (D1in[p, :, -1, 1:] + N.roll(D1in, 1, axis = 3)[p, :, -1, 1:]) \
                          + 0.25 * hr2d[p, :, -1, :-1, 2] * (Drin[p, :, -1, :-1] + N.roll(N.roll(Drin, -1, axis = 2), 1, axis = 3)[p, :, -1, :-1] \
                          + N.roll(Drin, 1, axis = 1)[p, :, -1, :-1] + N.roll(Drin, 1, axis = 3)[p, :, -1, :-1])

    # Interior
    D2d[p, :, 1:-1, 1:-1] = h22d[p, :, 1:-1, 0:-1, 2] * D2in[p, :, 1:-1, 1:-1] \
                          + 0.25 * h12d[p, :, 1:-1, 0:-1, 2] * (D1in[p, :, 1:-2, 1:] + N.roll(N.roll(D1in, -1, axis = 2), 1, axis = 3)[p, :, 1:-2, 1:] \
                          + N.roll(D1in, -1, axis = 2)[p, :, 1:-2, 1:] + N.roll(D1in, 1, axis = 3)[p, :, 1:-2, 1:]) \
                          + 0.25 * hr2d[p, :, 1:-1, :-1, 2] * (Drin[p, :, 1:-1, :-1] + N.roll(N.roll(Drin, -1, axis = 2), 1, axis = 3)[p, :, 1:-1, :-1] \
                          + N.roll(Drin, 1, axis = 1)[p, :, 1:-1, :-1] + N.roll(Drin, 1, axis = 3)[p, :, 1:-1, :-1])


def compute_E(p, Din, Bin):

    ##### Exi

    # Interior
    E1d[p, :, 1:-1, 1:-1] = g11d[:, 0:-1, 1:-1, 1] * E1u[p, :, 1:-1, 1:-1] \
                    + 0.25 * g12d[:, 0:-1, 1:-1, 1] * (E2u[p, :, 1:, 1:-2] + N.roll(N.roll(E2u, 1, axis = 2), -1, axis = 3)[p, :, 1:, 1:-2] \
                    + N.roll(E2u, 1, axis = 2)[p, :, 1:, 1:-2] + N.roll(E2u, -1, axis = 3)[p, :, 1:, 1:-2])
    # Left edge
    E1d[p, :, 0, 1:-1] = g11d[:, 0, 1:-1, 0] * E1u[p, :, 0, 1:-1] + 0.5 * g12d[:, 0, 1:-1, 0] * (E2u[p, :, 0, 1:-2] + N.roll(E2u, -1, axis = 3)[p, :, 0, 1:-2])
    # Right edge
    E1d[p, :, -1, 1:-1] = g11d[:, -1, 1:-1, 0] * E1u[p, :, -1, 1:-1] + 0.5 * g12d[:, -1, 1:-1, 0] * (E2u[p, :, -1, 1:-2] + N.roll(E2u, -1, axis = 3)[p, :, -1, 1:-2])
    # Bottom edge
    E1d[p, :, 1:-1, 0] = g11d[:, 0:-1, 0, 1] * E1u[p, :, 1:-1, 0] + 0.5 * g12d[:, 0:-1, 0, 1] * (E2u[p, :, 1:, 0] + N.roll(E2u, 1, axis = 2)[p, :, 1:, 0])
    # Top edge
    E1d[p, :, 1:-1, -1] = g11d[:, 0:-1, -1, 1] * E1u[p, :, 1:-1, -1] + 0.5 * g12d[:, 0:-1, -1, 1] * (E2u[p, :, 1:, -1] + N.roll(E2u, 1, axis = 2)[p, :, 1:, -1])
    # Bottom left corner
    E1d[p, :, 0, 0] = g11d[:, 0, 0, 0] * E1u[p, :, 0, 0] + g12d[:, 0, 0, 0] * E2u[p, :, 0, 0]
    # Bottom right corner
    E1d[p, :, -1, 0] = g11d[:, -1, 0, 0] * E1u[p, :, -1, 0] + g12d[:, -1, 0, 0] * E2u[p, :, -1, 0]
    # Top left corner
    E1d[p, :, 0, -1] = g11d[:, 0, -1, 0] * E1u[p, :, 0, -1] + g12d[:, 0, -1, 0] * E2u[p, :, 0, -1]
    # Top right corner
    E1d[p, :, -1, -1] = g11d[:, -1, -1, 0] * E1u[p, :, -1, -1] + g12d[:, -1, -1, 0] * E2u[p, :, -1, -1]

    ##### Eeta

    # Interior
    E2d[p, :, 1:-1, 1:-1] = g22d[:, 1:-1, 0:-1, 2] * E2u[p, :, 1:-1, 1:-1] \
                    + 0.25 * g12d[:, 1:-1, 0:-1, 2] * (E1u[p, :, 1:-2, 1:] + N.roll(N.roll(E1u, -1, axis = 2), 1, axis = 3)[p, :, 1:-2, 1:] \
                    + N.roll(E1u, -1, axis = 2)[p, :, 1:-2, 1:] + N.roll(E1u, 1, axis = 3)[p, :, 1:-2, 1:])
    # Left edge
    E2d[p, :, 0, 1:-1] = g22d[:, 0, 0:-1, 2] * E2u[p, :, 0, 1:-1] + 0.5 * g12d[:, 0, 0:-1, 2] * (E1u[p, :, 0, 1:] + N.roll(E1u, 1, axis = 3)[p, :, 0, 1:])
    # Right edge
    E2d[p, :, -1, 1:-1] = g22d[:, -1, 0:-1, 2] * E2u[p, :, -1, 1:-1] + 0.5 * g12d[:, -1, 0:-1, 2] * (E1u[p, :, -1, 1:] + N.roll(E1u, 1, axis = 3)[p, :, -1, 1:])
    # Bottom edge
    E2d[p, :, 1:-1, 0] = g22d[:, 1:-1, 0, 0] * E2u[p, :, 1:-1, 0] + 0.5 * g12d[:, 1:-1, 0, 0] * (E1u[p, :, 1:-2, 0] + N.roll(E1u, -1, axis = 2)[p, :, 1:-2, 0])
    # Top edge
    E2d[p, :, 1:-1, -1] = g22d[:, 1:-1, -1, 0] * E2u[p, :, 1:-1, -1] + 0.5 * g12d[:, 1:-1, -1, 0] * (E1u[p, :, 1:-2, -1] + N.roll(E1u, -1, axis = 2)[p, :, 1:-2, -1])
    # Bottom left corner
    E2d[p, :, 0, 0] = g22d[:, 0, 0, 0] * E2u[p, :, 0, 0] + g12d[:, 0, 0, 0] * E1u[p, :, 0, 0]
    # Bottom right corner
    E2d[p, :, -1, 0] = g22d[:, -1, 0, 0] * E2u[p, :, -1, 0] + g12d[:, -1, 0, 0] * E1u[p, :, -1, 0]
    # Top left corner
    E2d[p, :, 0, -1] = g22d[:, 0, -1, 0] * E2u[p, :, 0, -1] + g12d[:, 0, -1, 0] * E1u[p, :, 0, -1]
    # Top right corner
    E2d[p, :, -1, -1] = g22d[:, -1, -1, 0] * E2u[p, :, -1, -1] + g12d[:, -1, -1, 0] * E1u[p, :, -1, -1]

def contra_to_cov_B(p):

    ##### Bxi

    # Interior
    B1d[p, :, 1:-1, 1:-1] = g11d[:, 1:-1, 0:-1, 6] * B1u[p, :, 1:-1, 1:-1] \
                    + 0.25 * g12d[:, 1:-1, 0:-1, 6] * (B2u[p, :, 1:-2, 1:] + N.roll(N.roll(B2u, -1, axis = 2), 1, axis = 3)[p, :, 1:-2, 1:] \
                    + N.roll(B2u, -1, axis = 2)[p, :, 1:-2, 1:] + N.roll(B2u, 1, axis = 3)[p, :, 1:-2, 1:])
    # Left edge
    B1d[p, :, 0, 1:-1] = g11d[:, 0, 0:-1, 6] * B1u[p, :, 0, 1:-1] + 0.5 * g12d[:, 0, 0:-1, 6] * (B2u[p, :, 0, 1:] + N.roll(B2u, 1, axis = 3)[p, :, 0, 1:])
    # Right edge
    B1d[p, :, -1, 1:-1] = g11d[:, -1, 0:-1, 6] * B1u[p, :, -1, 1:-1] + 0.5 * g12d[:, -1, 0:-1, 6] * (B2u[p, :, -1, 1:] + N.roll(B2u, 1, axis = 3)[p, :, -1, 1:])
    # Bottom edge
    B1d[p, :, 1:-1, 0] = g11d[:, 1:-1, 0, 4] * B1u[p, :, 1:-1, 0] + 0.5 * g12d[:, 1:-1, 0, 4] * (B2u[p, :, 1:-2, 0] + N.roll(B2u, -1, axis = 2)[p, :, 1:-2, 0])
    # Top edge
    B1d[p, :, 1:-1, -1] = g11d[:, 1:-1, -1, 4] * B1u[p, :, 1:-1, -1] + 0.5 * g12d[:, 1:-1, -1, 4] * (B2u[p, :, 1:-2, -1] + N.roll(B2u, -1, axis = 2)[p, :, 1:-2, -1])
    # Bottom left corner
    B1d[p, :, 0, 0] = g11d[:, 0, 0, 4] * B1u[p, :, 0, 0] + g12d[:, 0, 0, 4] * B2u[p, :, 0, 0]
    # Bottom right corner
    B1d[p, :, -1, 0] = g11d[:, -1, 0, 4] * B1u[p, :, -1, 0] + g12d[:, -1, 0, 4] * B2u[p, :, -1, 0]
    # Top left corner
    B1d[p, :, 0, -1] = g11d[:, 0, -1, 4] * B1u[p, :, 0, -1] + g12d[:, 0, -1, 4] * B2u[p, :, 0, -1]
    # Top right corner
    B1d[p, :, -1, -1] = g11d[:, -1, -1, 4] * B1u[p, :, -1, -1] + g12d[:, -1, -1, 4] * B2u[p, :, -1, -1]

    ##### Beta

    # Interior
    B2d[p, :, 1:-1, 1:-1] = g22d[:, 0:-1, 1:-1, 5] * B2u[p, :, 1:-1, 1:-1] \
                    + 0.25 * g12d[:, 0:-1, 1:-1, 5] * (B1u[p, :, 1:, 1:-2] + N.roll(N.roll(B1u, 1, axis = 2), -1, axis = 3)[p, :, 1:, 1:-2] \
                    + N.roll(B1u, 1, axis = 2)[p, :, 1:, 1:-2] + N.roll(B1u, -1, axis = 3)[p, :, 1:, 1:-2])
    # Left edge
    B2d[p, :, 0, 1:-1] = g22d[:, 0, 1:-1, 4] * B2u[p, :, 0, 1:-1] + 0.5 * g12d[:, 0, 1:-1, 4] * (B1u[p, :, 0, 1:-2] + N.roll(B1u, -1, axis = 3)[p, :, 0, 1:-2])
    # Right edge
    B2d[p, :, -1, 1:-1] = g22d[:, -1, 1:-1, 4] * B2u[p, :, -1, 1:-1] + 0.5 * g12d[:, -1, 1:-1, 4] * (B1u[p, :, -1, 1:-2] + N.roll(B1u, -1, axis = 3)[p, :, -1, 1:-2])
    # Bottom edge
    B2d[p, :, 1:-1, 0] = g22d[:, 0:-1, 0, 5] * B2u[p, :, 1:-1, 0] + 0.5 * g12d[:, 0:-1, 0, 5] * (B1u[p, :, 1:, 0] + N.roll(B1u, 1, axis = 2)[p, :, 1:, 0])
    # Top edge
    B2d[p, :, 1:-1, -1] = g22d[:, 0:-1, -1, 5] * B2u[p, :, 1:-1, -1] + 0.5 * g12d[:, 0:-1, -1, 5] * (B1u[p, :, 1:, -1] + N.roll(B1u, 1, axis = 2)[p, :, 1:, -1])
    # Bottom left corner
    B2d[p, :, 0, 0] = g22d[:, 0, 0, 4] * B2u[p, :, 0, 0] + g12d[:, 0, 0, 4] * B1u[p, :, 0, 0]
    # Bottom right corner
    B2d[p, :, -1, 0] = g22d[:, -1, 0, 4] * B2u[p, :, -1, 0] + g12d[:, -1, 0, 4] * B1u[p, :, -1, 0]
    # Top left corner
    B2d[p, :, 0, -1] = g22d[:, 0, -1, 4] * B2u[p, :, 0, -1] + g12d[:, 0, -1, 4] * B1u[p, :, 0, -1]
    # Top right corner
    B2d[p, :, -1, -1] = g22d[:, -1, -1, 4] * B2u[p, :, -1, -1] + g12d[:, -1, -1, 4] * B1u[p, :, -1, -1]

########
# Compute interface terms
########

sig_in  = 1.0

def compute_penalty_E(p0, p1, dtin, Erin, E1in, E2in, Brin, B1in, B2in):

    top = topology[p0, p1]
    
    if (top == 'xx'):

        Er_0 = Erin[p0, :, -1, :]
        Er_1 = Erin[p1, :, 0, :]
        Eeta_0 = E2in[p0, :, -1, :]
        Eeta_1 = E2in[p1, :, 0, :]
        Br_0 = Brin[p0, :, -1, :]
        Br_1 = Brin[p1, :, 0, :]
        Beta_0 = B2in[p0, :, -1, :]
        Beta_1 = B2in[p1, :, 0, :]

        diff_Er[p0, :, -1, :] += dtin * sig_in * 0.5 * ((  Beta_0 + sqrt_gd_int * Er_0) - \
                                                        (  Beta_1 + sqrt_gd_int * Er_1)) / dxi / P_int_2[0]
        diff_Er[p1, :, 0, :]  += dtin * sig_in * 0.5 * ((- Beta_1 + sqrt_gd_int * Er_1) - \
                                                        (- Beta_0 + sqrt_gd_int * Er_0)) / dxi / P_int_2[0]

        diff_E2[p0, :, -1, :] += dtin * sig_in * 0.5 * ((Eeta_0 / sqrt_gd_half - Br_0) - \
                                                        (Eeta_1 / sqrt_gd_half - Br_1)) / dxi / P_int_2[0]
        diff_E2[p1, :, 0, :]  += dtin * sig_in * 0.5 * ((Eeta_1 / sqrt_gd_half + Br_1) - \
                                                        (Eeta_0 / sqrt_gd_half + Br_0)) / dxi / P_int_2[0]

    if (top == 'xy'):

        Er_0 = Erin[p0, :, -1, :]
        Er_1 = Erin[p1, :, :, 0]
        Eeta_0 = E2in[p0, :, -1, :]
        Exi_1  = E1in[p1, :, :, 0]
        Br_0 = Br[p0, :, -1, :]
        Br_1 = Br[p1, :, :, 0]
        Beta_0 = B2in[p0, :, -1, :]
        Bxi_1 = B1in[p1, :, :, 0]
        
        diff_Er[p0, :, -1, :] += dtin * sig_in * 0.5 * ((  Beta_0 + sqrt_gd_int * Er_0) - \
                                                        (- Bxi_1[:, ::-1] + sqrt_gd_int[:, ::-1] * Er_1[:, ::-1])) / dxi / P_int_2[0]
        diff_Er[p1, :, :, 0]  += dtin * sig_in * 0.5 * ((  Bxi_1 + sqrt_gd_int * Er_1) - \
                                                        (- Beta_0[:, ::-1] + sqrt_gd_int[:, ::-1] * Er_0[:, ::-1])) / dxi / P_int_2[0]

        diff_E2[p0, :, -1, :] += dtin * sig_in * 0.5 * (( Eeta_0 / sqrt_gd_half - Br_0) - \
                                                       (- Exi_1[:, ::-1] / sqrt_gd_half[:, ::-1] - Br_1[:, ::-1])) / dxi / P_int_2[0]
        diff_E1[p1, :, :, 0]  += dtin * sig_in * 0.5 * (( Exi_1 / sqrt_gd_half - Br_1) - \
                                                       (- Eeta_0[:, ::-1] / sqrt_gd_half[:, ::-1] - Br_0[:, ::-1])) / dxi / P_int_2[0]
        
    if (top == 'yy'):

        Er_0 = Erin[p0, :, :, -1]
        Er_1 = Erin[p1, :, :, 0]
        Exi_0 = E1in[p0, :, :, -1]
        Exi_1 = E1in[p1, :, :, 0]
        Br_0 = Brin[p0, :, :, -1]
        Br_1 = Brin[p1, :, :, 0]
        Bxi_0 = B1in[p0, :, :, -1]
        Bxi_1 = B1in[p1, :, :, 0]

        diff_Er[p0, :, :, -1] += dtin * sig_in * 0.5 * ((- Bxi_0 + sqrt_gd_int * Er_0) - \
                                                        (- Bxi_1 + sqrt_gd_int * Er_1)) / deta / P_int_2[0]
        diff_Er[p1, :, :, 0]  += dtin * sig_in * 0.5 * ((  Bxi_1 + sqrt_gd_int * Er_1) - \
                                                        (  Bxi_0 + sqrt_gd_int * Er_0)) / deta / P_int_2[0]

        diff_E1[p0, :, :, -1] += dtin * sig_in * 0.5 * ((Exi_0 / sqrt_gd_half + Br_0) - \
                                                        (Exi_1 / sqrt_gd_half + Br_1)) / deta / P_int_2[0]
        diff_E1[p1, :, :, 0]  += dtin * sig_in * 0.5 * ((Exi_1 / sqrt_gd_half - Br_1) - \
                                                        (Exi_0 / sqrt_gd_half - Br_0)) / deta / P_int_2[0]

    if (top == 'yx'):

        Er_0 = Erin[p0, :, :, -1]
        Er_1 = Erin[p1, :, 0, :]
        Exi_0 = E1in[p0, :, :, -1]
        Eeta_1 = E2in[p1, :, 0, :]
        Br_0 = Brin[p0, :, :, -1]
        Br_1 = Brin[p1, :, 0, :]
        Bxi_0 = B1in[p0, :, :, -1]
        Beta_1 = B2in[p1, :, 0, :]

        diff_Er[p0, :, :, -1] += dtin * sig_in * 0.5 * ((- Bxi_0 + sqrt_gd_int * Er_0) - \
                                                        (  Beta_1[:, ::-1] + sqrt_gd_int[:, ::-1] * Er_1[:, ::-1])) / deta / P_int_2[0]
        diff_Er[p1, :, 0, :]  += dtin * sig_in * 0.5 * ((- Beta_1 + sqrt_gd_int * Er_1) - \
                                                        (  Bxi_0[:, ::-1] + sqrt_gd_int[:, ::-1] * Er_0[:, ::-1])) / deta / P_int_2[0]        

        diff_E1[p0, :, :, -1] += dtin * sig_in * 0.5 * ((Exi_0 / sqrt_gd_half + Br_0) - \
                                                        (- Eeta_1[:, ::-1] / sqrt_gd_half[:, ::-1] + Br_1[:, ::-1])) / deta / P_int_2[0]
        diff_E2[p1, :, 0, :]  += dtin * sig_in * 0.5 * ((Eeta_1 / sqrt_gd_half + Br_1) - \
                                                        (- Exi_0[:, ::-1] / sqrt_gd_half[:, ::-1] + Br_0[:, ::-1])) / deta / P_int_2[0]

def interface_E(p0, p1):

    i0 =  1
    i1_half = Nxi_half - 1
    i1_int  = Nxi_int - 1

    top = topology[p0, p1]
    
    if (top == 'xx'):
        Er[p0, :, -1, i0:i1_int] -= diff_Er[p0, :, -1, i0:i1_int] / sqrt_det_g_int[:, i0:i1_int]
        Er[p1, :, 0, i0:i1_int]  -= diff_Er[p1, :, 0, i0:i1_int]  / sqrt_det_g_int[:, i0:i1_int]
        
        E2u[p0, :, -1, i0:i1_half] -= diff_E2[p0, :, -1, i0:i1_half] / sqrt_det_g_half[:, i0:i1_half]
        E2u[p1, :, 0, i0:i1_half]  -= diff_E2[p1, :, 0, i0:i1_half]  / sqrt_det_g_half[:, i0:i1_half]

    if (top == 'xy'):
        Er[p0, :, -1, i0:i1_int] -= diff_Er[p0, :, -1, i0:i1_int] / sqrt_det_g_int[:, i0:i1_int]
        Er[p1, :, i0:i1_int, 0]  -= diff_Er[p1, :, i0:i1_int, 0]  / sqrt_det_g_int[:, i0:i1_int]

        E2u[p0, :, -1, i0:i1_half] -= diff_E2[p0, :, -1, i0:i1_half] / sqrt_det_g_half[:, i0:i1_half]
        E1u[p1, :, i0:i1_half, 0]  -= diff_E1[p1, :, i0:i1_half, 0]  / sqrt_det_g_half[:, i0:i1_half] 

    if (top == 'yy'):
        Er[p0, :, i0:i1_int, -1] -= diff_Er[p0, :, i0:i1_int, -1] / sqrt_det_g_int[:, i0:i1_int]
        Er[p1, :, i0:i1_int, 0]  -= diff_Er[p1, :, i0:i1_int, 0]  / sqrt_det_g_int[:, i0:i1_int]

        E1u[p0, :, i0:i1_half, -1] -= diff_E1[p0, :, i0:i1_half, -1] / sqrt_det_g_half[:, i0:i1_half]
        E1u[p1, :, i0:i1_half, 0]  -= diff_E1[p1, :, i0:i1_half, 0]  / sqrt_det_g_half[:, i0:i1_half] 

    if (top == 'yx'):
        Er[p0, :, i0:i1_int, -1] -= diff_Er[p0, :, i0:i1_int, -1] / sqrt_det_g_int[:, i0:i1_int]
        Er[p1, :, 0, i0:i1_int]  -= diff_Er[p1, :, 0, i0:i1_int]  / sqrt_det_g_int[:, i0:i1_int]

        E1u[p0, :, i0:i1_half, -1] -= diff_E1[p0, :, i0:i1_half, -1] / sqrt_det_g_half[:, i0:i1_half]
        E2u[p1, :, 0, i0:i1_half]  -= diff_E2[p1, :, 0, i0:i1_half]  / sqrt_det_g_half[:, i0:i1_half]

def corners_E(p0):
    
    Er[p0, :, 0, 0]   -= diff_Er[p0, :, 0, 0]   * sig_in / sqrt_det_g_int[:, 0]
    Er[p0, :, -1, 0]  -= diff_Er[p0, :, -1, 0]  * sig_in / sqrt_det_g_int[:, 0]
    Er[p0, :, 0, -1]  -= diff_Er[p0, :, 0, -1]  * sig_in / sqrt_det_g_int[:, 0]
    Er[p0, :, -1, -1] -= diff_Er[p0, :, -1, -1] * sig_in / sqrt_det_g_int[:, 0]

    E1u[p0, :, 0, 0]   -= diff_E1[p0, :, 0, 0]   * sig_in / sqrt_det_g_half[:, 0]
    E1u[p0, :, -1, 0]  -= diff_E1[p0, :, -1, 0]  * sig_in / sqrt_det_g_half[:, 0]
    E1u[p0, :, 0, -1]  -= diff_E1[p0, :, 0, -1]  * sig_in / sqrt_det_g_half[:, 0]
    E1u[p0, :, -1, -1] -= diff_E1[p0, :, -1, -1] * sig_in / sqrt_det_g_half[:, 0]

    E2u[p0, :, 0, 0]   -= diff_E2[p0, :, 0, 0]   * sig_in / sqrt_det_g_half[:, 0]
    E2u[p0, :, -1, 0]  -= diff_E2[p0, :, -1, 0]  * sig_in / sqrt_det_g_half[:, 0]
    E2u[p0, :, 0, -1]  -= diff_E2[p0, :, 0, -1]  * sig_in / sqrt_det_g_half[:, 0]
    E2u[p0, :, -1, -1] -= diff_E2[p0, :, -1, -1] * sig_in / sqrt_det_g_half[:, 0]


def compute_penalty_B(p0, p1, dtin, Erin, E1in, E2in, Brin, B1in, B2in):

    top = topology[p0, p1]
    
    if (top == 'xx'):

        Eeta_0 = E2in[p0, :, -1, :]
        Eeta_1 = E2in[p1, :, 0, :]
        Br_0 = Brin[p0, :, -1, :]
        Br_1 = Brin[p1, :, 0, :]
        Beta_0 = B2in[p0, :, -1, :]
        Beta_1 = B2in[p1, :, 0, :]
        Er_0 = Erin[p0, :, -1, :]
        Er_1 = Erin[p1, :, 0, :]

        diff_Br[p0, :, -1, :] += dtin * sig_in * 0.5 * ((- Eeta_0 + sqrt_gd_half * Br_0) - \
                                                        (- Eeta_1 + sqrt_gd_half * Br_1)) / dxi / P_int_2[0]
        diff_Br[p1, :, 0, :]  += dtin * sig_in * 0.5 * ((  Eeta_1 + sqrt_gd_half * Br_1) - \
                                                         ( Eeta_0 + sqrt_gd_half * Br_0)) / dxi / P_int_2[0]

        diff_B2[p0, :, -1, :] += dtin * sig_in * 0.5 * ((Beta_0 / sqrt_gd_int + Er_0) - \
                                                        (Beta_1 / sqrt_gd_int + Er_1)) / dxi / P_int_2[0]
        diff_B2[p1, :, 0, :]  += dtin * sig_in * 0.5 * ((Beta_1 / sqrt_gd_int - Er_1) - \
                                                        (Beta_0 / sqrt_gd_int - Er_0)) / dxi / P_int_2[0]

    if (top == 'xy'):

        Eeta_0 = E2in[p0, :, -1, :]
        Exi_1  = E1in[p1, :, :, 0]
        Br_0 = Br[p0, :, -1, :]
        Br_1 = Br[p1, :, :, 0]
        Beta_0 = B2in[p0, :, -1, :]
        Bxi_1  = B1in[p1, :, :, 0]
        Er_0 = Er[p0, :, -1, :]
        Er_1 = Er[p1, :, :, 0]
        
        diff_Br[p0, :, -1, :] += dtin * sig_in * 0.5 * ((- Eeta_0 + sqrt_gd_half * Br_0) - \
                                                        (  Exi_1[:, ::-1] + sqrt_gd_half[:, ::-1] * Br_1[:, ::-1])) / dxi / P_int_2[0]
        diff_Br[p1, :, :, 0]  += dtin * sig_in * 0.5 * ((- Exi_1 + sqrt_gd_half * Br_1) - \
                                                        (  Eeta_0[:, ::-1] + sqrt_gd_half[:, ::-1] * Br_0[:, ::-1])) / dxi / P_int_2[0]

        diff_B2[p0, :, -1, :] += dtin * sig_in * 0.5 * ((  Beta_0 / sqrt_gd_int + Er_0) - \
                                                        (- Bxi_1[:, ::-1] / sqrt_gd_int[:, ::-1] + Er_1[:, ::-1])) / dxi / P_int_2[0]
        diff_B1[p1, :, :, 0]  += dtin * sig_in * 0.5 * ((  Bxi_1 / sqrt_gd_int + Er_1) - \
                                                        (- Beta_0[:, ::-1] / sqrt_gd_int[:, ::-1] + Er_0[:, ::-1])) / dxi / P_int_2[0]

    if (top == 'yy'):

        Exi_0 = E1in[p0, :, :, -1]
        Exi_1 = E1in[p1, :, :, 0]
        Br_0 = Brin[p0, :, :, -1]
        Br_1 = Brin[p1, :, :, 0]
        Bxi_0 = B1in[p0, :, :, -1]
        Bxi_1 = B1in[p1, :, :, 0]
        Er_0 = Erin[p0, :, :, -1]
        Er_1 = Erin[p1, :, :, 0]
        
        diff_Br[p0, :, :, -1] += dtin * sig_in * 0.5 * ((  Exi_0 + sqrt_gd_half * Br_0) - \
                                                        (  Exi_1 + sqrt_gd_half * Br_1)) / deta / P_int_2[0]
        diff_Br[p1, :, :, 0]  += dtin * sig_in * 0.5 * ((- Exi_1 + sqrt_gd_half * Br_1) - \
                                                        (- Exi_0 + sqrt_gd_half * Br_0)) / deta / P_int_2[0]

        diff_B1[p0, :, :, -1] += dtin * sig_in * 0.5 * ((Bxi_0 / sqrt_gd_int - Er_0) - \
                                                        (Bxi_1 / sqrt_gd_int - Er_1)) / deta / P_int_2[0]
        diff_B1[p1, :, :, 0]  += dtin * sig_in * 0.5 * ((Bxi_1 / sqrt_gd_int + Er_1) - \
                                                        (Bxi_0 / sqrt_gd_int + Er_0)) / deta / P_int_2[0]

    if (top == 'yx'):

        Exi_0 = E1in[p0, :, :, -1]
        Eeta_1 = E2in[p1, :, 0, :]
        Br_0 = Brin[p0, :, :, -1]
        Br_1 = Brin[p1, :, 0, :]
        Bxi_0 = B1in[p0, :, :, -1]
        Beta_1 = B2in[p1, :, 0, :]
        Er_0 = Erin[p0, :, :, -1]
        Er_1 = Erin[p1, :, 0, :]

        diff_Br[p0, :, :, -1] += dtin * sig_in * 0.5 * ((  Exi_0 + sqrt_gd_half * Br_0) - \
                                                        (- Eeta_1[:, ::-1] + sqrt_gd_half[:, ::-1] * Br_1[:, ::-1])) / deta / P_int_2[0]
        diff_Br[p1, :, 0, :]  += dtin * sig_in * 0.5 * ((  Eeta_1 + sqrt_gd_half * Br_1) - \
                                                        (- Exi_0[:, ::-1] + sqrt_gd_half[:, ::-1] * Br_0[:, ::-1])) / deta / P_int_2[0]

        diff_B1[p0, :, :, -1] += dtin * sig_in * 0.5 * ((  Bxi_0 / sqrt_gd_int - Er_0) - \
                                                        (- Beta_1[:, ::-1] / sqrt_gd_int[:, ::-1] - Er_1[:, ::-1])) / deta / P_int_2[0]
        diff_B2[p1, :, 0, :]  += dtin * sig_in * 0.5 * ((  Beta_1 / sqrt_gd_int - Er_1) - \
                                                        (- Bxi_0[:, ::-1] / sqrt_gd_int[:, ::-1] - Er_0[:, ::-1])) / deta / P_int_2[0]


def interface_B(p0, p1):

    i0 =  1
    i1_half = Nxi_half - 1
    i1_int  = Nxi_int - 1

    top = topology[p0, p1]
    
    if (top == 'xx'):
        Br[p0, :, -1, i0:i1_half] -= diff_Br[p0, :, -1, i0:i1_half] / sqrt_det_g_half[:, i0:i1_half]
        Br[p1, :, 0, i0:i1_half]  -= diff_Br[p1, :, 0, i0:i1_half]  / sqrt_det_g_half[:, i0:i1_half]

        B2u[p0, :, -1, i0:i1_int] -= diff_B2[p0, :, -1, i0:i1_int] / sqrt_det_g_int[:, i0:i1_int]
        B2u[p1, :, 0, i0:i1_int]  -= diff_B2[p1, :, 0, i0:i1_int] / sqrt_det_g_int[:, i0:i1_int]

    if (top == 'xy'):
        Br[p0, :, -1, i0:i1_half] -= diff_Br[p0, :, -1, i0:i1_half] / sqrt_det_g_half[:, i0:i1_half]
        Br[p1, :, i0:i1_half, 0]  -= diff_Br[p1, :, i0:i1_half, 0]  / sqrt_det_g_half[:, i0:i1_half]

        B2u[p0, :, -1, i0:i1_int] -= diff_B2[p0, :, -1, i0:i1_int] / sqrt_det_g_int[:, i0:i1_int]
        B1u[p1, :, i0:i1_int, 0]  -= diff_B1[p1, :, i0:i1_int, 0] / sqrt_det_g_int[:, i0:i1_int] 

    if (top == 'yy'):
        Br[p0, :, i0:i1_half, -1] -= diff_Br[p0, :, i0:i1_half, -1] / sqrt_det_g_half[:, i0:i1_half]
        Br[p1, :, i0:i1_half, 0]  -= diff_Br[p1, :, i0:i1_half, 0]  / sqrt_det_g_half[:, i0:i1_half]

        B1u[p0, :, i0:i1_int, -1] -= diff_B1[p0, :, i0:i1_int, -1] / sqrt_det_g_int[:, i0:i1_int]
        B1u[p1, :, i0:i1_int, 0]  -= diff_B1[p1, :, i0:i1_int, 0] / sqrt_det_g_int[:, i0:i1_int] 

    if (top == 'yx'):
        Br[p0, :, i0:i1_half, -1] -= diff_Br[p0, :, i0:i1_half, -1] / sqrt_det_g_half[:, i0:i1_half]
        Br[p1, :, 0, i0:i1_half]  -= diff_Br[p1, :, 0, i0:i1_half]  / sqrt_det_g_half[:, i0:i1_half]

        B1u[p0, :, i0:i1_int, -1] -= diff_B1[p0, :,i0:i1_int, -1] / sqrt_det_g_int[:, i0:i1_int]
        B2u[p1, :, 0, i0:i1_int]  -= diff_B2[p1, :, 0, i0:i1_int] / sqrt_det_g_int[:, i0:i1_int]

def corners_B(p0):

    Br[p0, :, 0, 0]   -= diff_Br[p0, :, 0, 0]   * sig_in / sqrt_det_g_half[:, 0]
    Br[p0, :, -1, 0]  -= diff_Br[p0, :, -1, 0]  * sig_in / sqrt_det_g_half[:, 0]
    Br[p0, :, 0, -1]  -= diff_Br[p0, :, 0, -1]  * sig_in / sqrt_det_g_half[:, 0]
    Br[p0, :, -1, -1] -= diff_Br[p0, :, -1, -1] * sig_in / sqrt_det_g_half[:, 0]

    B1u[p0, :, 0, 0]   -= diff_B1[p0, :, 0, 0] * sig_in   / sqrt_det_g_int[:, 0]
    B1u[p0, :, -1, 0]  -= diff_B1[p0, :, -1, 0] * sig_in  / sqrt_det_g_int[:, 0]
    B1u[p0, :, 0, -1]  -= diff_B1[p0, :, 0, -1] * sig_in  / sqrt_det_g_int[:, 0]
    B1u[p0, :, -1, -1] -= diff_B1[p0, :, -1, -1] * sig_in / sqrt_det_g_int[:, 0]

    B2u[p0, :, 0, 0]   -= diff_B2[p0, :, 0, 0] * sig_in   / sqrt_det_g_int[:, 0]
    B2u[p0, :, -1, 0]  -= diff_B2[p0, :, -1, 0] * sig_in  / sqrt_det_g_int[:, 0]
    B2u[p0, :, 0, -1]  -= diff_B2[p0, :, 0, -1] * sig_in  / sqrt_det_g_int[:, 0]
    B2u[p0, :, -1, -1] -= diff_B2[p0, :, -1, -1] * sig_in / sqrt_det_g_int[:, 0]

########
# Perfectly conducting boundary conditions at r_min
########

# NS rotation proile to avoid high frequency waves in FD solver
def BC_E_metal_rmin(it, patch):
    Er[patch,  NG, :, :] = 0.0
    E1u[patch, NG, :, :] = E1_surf[patch, :, :] * 0.5 * (1.0 - N.tanh(5.0 - it / 0.5))
    E2u[patch, NG, :, :] = E2_surf[patch, :, :] * 0.5 * (1.0 - N.tanh(5.0 - it / 0.5))
    
def BC_B_metal_rmin(patch):
    Br[patch, NG, :, :]   = Br_surf[patch, :, :]
    B1u[patch, :NG, :, :] = 0.0
    B2u[patch, :NG, :, :] = 0.0

########
# Absorbing boundary conditions at r_max
########

i_abs = 5 # Thickness of absorbing layer in number of cells
r_abs = r[Nr - i_abs]
delta = ((r - r_abs) / (r_max - r_abs)) * N.heaviside(r - r_abs, 0.0)
sigma = N.exp(- 10.0 * delta**3)

delta = ((r_yee - r_abs) / (r_max - r_abs)) * N.heaviside(r_yee - r_abs, 0.0)
sigma_yee = N.exp(- 10.0 * delta**3)

# def BC_B_absorb(patch):
#     for k in range(Nr - i_abs, Nr):
#         Br[patch, k, :, :]  = Br0[patch, k, :, :] + (Br[patch, k, :, :] - Br0[patch, k, :, :]) * sigma[k]
#         B1u[patch, k, :, :] = B1u0[patch, k, :, :] + (B1u[patch, k, :, :] - B1u0[patch, k, :, :]) * sigma_yee[k]
#         B2u[patch, k, :, :] = B2u0[patch, k, :, :] + (B2u[patch, k, :, :] - B2u0[patch, k, :, :]) * sigma_yee[k]

# def BC_E_absorb(patch):
#     for k in range(Nr - i_abs, Nr):
#         Er[patch, k, :, :]  *= sigma_yee[k]
#         E1u[patch, k, :, :] *= sigma[k]
#         E2u[patch, k, :, :] *= sigma[k]

def BC_B_absorb(patch):
    Br[patch, :, :, :]  = Br0[patch, :, :, :] + (Br[patch, :, :, :] - Br0[patch, :, :, :]) * sigma[:, None, None]
    B1u[patch, :, :, :] = B1u0[patch, :, :, :] + (B1u[patch, :, :, :] - B1u0[patch, :, :, :]) * sigma_yee[:, None, None]
    B2u[patch, :, :, :] = B2u0[patch, :, :, :] + (B2u[patch, :, :, :] - B2u0[patch, :, :, :]) * sigma_yee[:, None, None]

def BC_E_absorb(patch):
    Er[patch, :, :, :]  *= sigma_yee[:, None, None]
    E1u[patch, :, :, :] *= sigma[:, None, None]
    E2u[patch, :, :, :] *= sigma[:, None, None]

########
# Boundary conditions at r_max
########

def BC_E_metal_rmax(patch):
    E1u[patch, (Nr0 + NG), :, :] = 0.0
    E2u[patch, (Nr0 + NG), :, :] = 0.0

def BC_B_metal_rmax(patch):
    Br[patch, (Nr0 + NG), :, :] = Br0[patch, (Nr0 + NG), :, :]

########
# Define initial data
########

B0 = 1.0
tilt = 30.0 / 180.0 * N.pi

def func_Br(r0, th0, ph0):
    return 2.0 * B0 * (N.cos(th0) * N.cos(tilt) + N.sin(th0) * N.sin(ph0) * N.sin(tilt)) / r0**3
    # return B0 * N.sin(th0)**3 * N.cos(3.0 * ph0) 

def func_Bth(r0, th0, ph0):
    return B0 * (N.cos(tilt) * N.sin(th0) - N.cos(th0) * N.sin(ph0) * N.sin(tilt)) / r0**4
#    return 0.0

def func_Bph(r0, th0, ph0):
    return - B0 * (N.cos(ph0) / N.sin(th0) * N.sin(tilt)) / r0**4
    # return 0.0

def InitialData():

    for patch in range(n_patches):

        fvec = (globals()["vec_sph_to_" + sphere[patch]])
        fcoord = (globals()["coord_" + sphere[patch] + "_to_sph"])

        for i in range(Nxi_half):
            for j in range(Neta_half):

                r0 = r[:]
                th0, ph0 = fcoord(xi_half[i], eta_half[j])
                BrTMP = func_Br(r0, th0, ph0)

                Br[patch, :, i, j] = BrTMP
                Br0[patch,:, i, j] = BrTMP
                    
        for i in range(Nxi_int):
            for j in range(Neta_half):

                    r0 = r_yee[:]
                    th0, ph0 = fcoord(xi_int[i], eta_half[j])
                    BtTMP = func_Bth(r0, th0, ph0)
                    BpTMP = func_Bph(r0, th0, ph0)
                    BCStmp = fvec(th0, ph0, BtTMP, BpTMP)

                    B1u[patch, :, i, j]  = BCStmp[0]
                    B1u0[patch, :, i, j] = BCStmp[0]
                    E2u[patch, :, i, j] = 0.0

        for i in range(Nxi_half):
            for j in range(Neta_int):

                    r0 = r_yee[:]
                    th0, ph0 = fcoord(xi_half[i], eta_int[j])
                    BtTMP = func_Bth(r0, th0, ph0)
                    BpTMP = func_Bph(r0, th0, ph0)
                    BCStmp = fvec(th0, ph0, BtTMP, BpTMP)

                    B2u[patch, :, i, j]  = BCStmp[1]
                    B2u0[patch, :, i, j] = BCStmp[1]
                    E1u[patch, :, i, j]  = 0.0

        Er[patch,  :, :, :] = 0.0

########
# Boundary conditions at r_min
########

omega = 0.0 # Angular velocity of the conductor at r_min
InitialData()

def func_Eth(th0, ph0, omega0):
    return - omega0 * func_Br(1.0, th0, ph0) * N.sin(th0)

# Fields at r_min
E1_surf = N.zeros((n_patches, Nxi_half, Neta_int))
E2_surf = N.zeros((n_patches, Nxi_int, Neta_half))
Br_surf = N.zeros((n_patches, Nxi_half, Neta_half))

Br_surf[:, :, :] = Br0[:, 0, :, :]

for patch in range(6):

    fvec = (globals()["vec_sph_to_" + sphere[patch]])
    fcoord = (globals()["coord_" + sphere[patch] + "_to_sph"])

    for i in range(Nxi_half):
        for j in range(Neta_int):

            r0 = r[0]
            th0, ph0 = fcoord(xi_half[i], eta_int[j])
            EtTMP = func_Eth(th0, ph0, omega)
            EpTMP = 0.0
            omega_patch = fvec(th0, ph0, EtTMP, EpTMP)

            E1_surf[patch, i, j] = omega_patch[0]

    for i in range(Nxi_int):
        for j in range(Neta_half):

            r0 = r[0]
            th0, ph0 = fcoord(xi_int[i], eta_half[j])
            EtTMP = func_Eth(th0, ph0, omega)
            EpTMP = 0.0
            omega_patch = fvec(th0, ph0, EtTMP, EpTMP)

            E2_surf[patch, i, j] = omega_patch[1]

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

    ax.pcolormesh(xBr_grid, yBr_grid, Br[Sphere.A, ir, :, :], cmap = "RdBu_r", vmin = - vm, vmax = vm)
    ax.pcolormesh(xBr_grid + N.pi / 2.0, yBr_grid, Br[Sphere.B, ir, :, :], cmap = "RdBu_r", vmin = - vm, vmax = vm)
    ax.pcolormesh(xBr_grid, yBr_grid - N.pi / 2.0, Br[Sphere.S, ir, :, :], cmap = "RdBu_r", vmin = - vm, vmax = vm)

    ax.pcolormesh(xi_grid_c + N.pi, eta_grid_c, Br[Sphere.C, ir, :, :], cmap = "RdBu_r", vmin = - vm, vmax = vm)
    ax.pcolormesh(xi_grid_d - N.pi / 2.0, eta_grid_d, Br[Sphere.D, ir, :, :], cmap = "RdBu_r", vmin = - vm, vmax = vm)
    ax.pcolormesh(xi_grid_n, eta_grid_n + N.pi / 2.0, Br[Sphere.N, ir, :, :], cmap = "RdBu_r", vmin = - vm, vmax = vm)
    
    P.title(r'$t={:.3f} R/c$'.format(FDUMP*it*dt))
    
    figsave_png(fig, "snapshots_3d/Br_" + str(it))

    P.close('all')

def plot_fields_unfolded_E1(it, vm, ir):

    xi_grid_c, eta_grid_c = unflip_eq(xE1_grid, yE1_grid)
    xi_grid_d, eta_grid_d = unflip_eq(xE1_grid, yE1_grid)
    xi_grid_n, eta_grid_n = unflip_po(xE1_grid, yE1_grid)

    fig = P.figure(1, facecolor='w')
    ax = P.subplot(111)

    ax.pcolormesh(xE1_grid, yE1_grid, E1u[Sphere.A, ir, :, :], cmap = "RdBu_r", vmin = - vm, vmax = vm)
    ax.pcolormesh(xE1_grid + N.pi / 2.0 + 0.1, yE1_grid, E1u[Sphere.B, ir, :, :], cmap = "RdBu_r", vmin = - vm, vmax = vm)
    ax.pcolormesh(xE1_grid, yE1_grid - N.pi / 2.0 - 0.1, E1u[Sphere.S, ir, :, :], cmap = "RdBu_r", vmin = - vm, vmax = vm)

    ax.pcolormesh(xi_grid_c + N.pi + 0.2, eta_grid_c, E1u[Sphere.C, ir, :, :], cmap = "RdBu_r", vmin = - vm, vmax = vm)
    ax.pcolormesh(xi_grid_d - N.pi / 2.0 - 0.1, eta_grid_d, E1u[Sphere.D, ir, :, :], cmap = "RdBu_r", vmin = - vm, vmax = vm)
    ax.pcolormesh(xi_grid_n, eta_grid_n + N.pi / 2.0 + 0.1, E1u[Sphere.N, ir, :, :], cmap = "RdBu_r", vmin = - vm, vmax = vm)
    
    P.title(r'$t={:.3f} R/c$'.format(FDUMP*it*dt))
    
    figsave_png(fig, "snapshots_3d/E1u_" + str(it))

    P.close('all')

def plot_fields_unfolded_E2(it, vm, ir):

    xi_grid_c, eta_grid_c = unflip_eq(xE2_grid, yE2_grid)
    xi_grid_d, eta_grid_d = unflip_eq(xE2_grid, yE2_grid)
    xi_grid_n, eta_grid_n = unflip_po(xE2_grid, yE2_grid)

    fig = P.figure(1, facecolor='w')
    ax = P.subplot(111)

    ax.pcolormesh(xE2_grid, yE2_grid, E2u[Sphere.A, ir, :, :], cmap = "RdBu_r", vmin = - vm, vmax = vm)
    ax.pcolormesh(xE2_grid + N.pi / 2.0 + 0.1, yE2_grid, E2u[Sphere.B, ir, :, :], cmap = "RdBu_r", vmin = - vm, vmax = vm)
    ax.pcolormesh(xE2_grid, yE2_grid - N.pi / 2.0 - 0.1, E2u[Sphere.S, ir, :, :], cmap = "RdBu_r", vmin = - vm, vmax = vm)

    ax.pcolormesh(xi_grid_c + N.pi + 0.2, eta_grid_c, E2u[Sphere.C, ir, :, :], cmap = "RdBu_r", vmin = - vm, vmax = vm)
    ax.pcolormesh(xi_grid_d - N.pi / 2.0 - 0.1, eta_grid_d, E2u[Sphere.D, ir, :, :], cmap = "RdBu_r", vmin = - vm, vmax = vm)
    ax.pcolormesh(xi_grid_n, eta_grid_n + N.pi / 2.0 + 0.1, E2u[Sphere.N, ir, :, :], cmap = "RdBu_r", vmin = - vm, vmax = vm)
    
    P.title(r'$t={:.3f} R/c$'.format(FDUMP*it*dt))
    
    figsave_png(fig, "snapshots_3d/E2u_" + str(it))

    P.close('all')

########
# Initialization
########

idump = 0

Nt = 10001 # Number of iterations
FDUMP = 20 # Dump frequency
time = dt * N.arange(Nt)
energy = N.zeros((n_patches, Nt))

WriteCoordsHDF5()

# # Initialize half time step
# compute_diff_E(patches)
# push_B(patches, 0, 0.5 * dt)
# for i in range(n_zeros):
#     p0, p1 = index_row[i], index_col[i]
#     compute_penalty_B(p0, p1, dt, E1d, E2d, Br)
# for i in range(n_zeros):
#     p0, p1 = index_row[i], index_col[i]
#     interface_B(p0, p1)
# corners_B(patches)

########
# Main routine
########

for it in tqdm(range(Nt), "Progression"):
    if ((it % FDUMP) == 0):
        # plot_fields_unfolded_Br(idump, 0.5, 10)
        plot_fields_unfolded_E1(idump, 0.1, 2)
        # plot_fields_unfolded_E2(idump, 0.1, 2)
        WriteAllFieldsHDF5(idump)
        idump += 1

    diff_Br[:, :, :, :] = 0.0
    diff_E1[:, :, :, :] = 0.0
    diff_E2[:, :, :, :] = 0.0
    diff_Er[:, :, :, :] = 0.0
    diff_B1[:, :, :, :] = 0.0
    diff_B2[:, :, :, :] = 0.0
    
    compute_diff_B(patches)

    push_E(patches, dt)

    for i in range(n_zeros):
        p0, p1 = index_row[i], index_col[i]
        compute_penalty_E(p0, p1, dt, Er, E1d, E2d, Br, B1d, B2d)
        interface_E(p0, p1)
        # compute_delta_E(p0, p1, dt, E1d0, E2d0, Br) # Different time-stepping?

    corners_E(patches)

    BC_E_metal_rmin(it, patches)
    BC_E_absorb(patches)
    BC_E_metal_rmax(patches)

    contra_to_cov_E(patches)

    compute_diff_E(patches)
    
    push_B(patches, dt)
    
    for i in range(n_zeros):
        p0, p1 = index_row[i], index_col[i]
        compute_penalty_B(p0, p1, dt, Er, E1d, E2d, Br, B1d, B2d)
        interface_B(p0, p1)
        # compute_delta_B(p0, p1, dt, E1d, E2d, Br) # Different time-stepping?

    corners_B(patches)

    BC_B_metal_rmin(patches)
    BC_B_absorb(patches)
    BC_B_metal_rmax(patches)

    contra_to_cov_B(patches)
