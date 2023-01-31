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

# Import figure routines
import sys
sys.path.append('../')
sys.path.append('../transformations/')

from figure_module import *

outdir = '/home/bcrinqua/GitHub/Maxwell_cubed_sphere/data_3d/'

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
cfl = 0.2
Nr0 = 64
Nxi = 32 #32
Neta = 32 #32

Nxi_int = Nxi + 1 # Number of integer points
Nxi_half = Nxi + 2 # Number of half-step points
Neta_int = Neta + 1 # Number of integer points
Neta_half = Neta + 2 # NUmber of half-step points

NG = 1 # Radial ghost cells
Nr = Nr0 + 2 * NG # Total number of radial points

r_min, r_max = 1.0, 8.0
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
Er  = N.zeros((n_patches, Nr, Nxi_int, Neta_int))
E1u = N.zeros((n_patches, Nr, Nxi_half, Neta_int))
E2u = N.zeros((n_patches, Nr, Nxi_int,  Neta_half))
E1d = N.zeros((n_patches, Nr, Nxi_half, Neta_int))
E2d = N.zeros((n_patches, Nr, Nxi_int,  Neta_half))

Br  = N.zeros((n_patches, Nr, Nxi_half, Neta_half))
B1u = N.zeros((n_patches, Nr, Nxi_int, Neta_half))
B2u = N.zeros((n_patches, Nr, Nxi_half,  Neta_int))
B1d = N.zeros((n_patches, Nr, Nxi_int, Neta_half))
B2d = N.zeros((n_patches, Nr, Nxi_half,  Neta_int))

# Gradients
dE1d2 = N.zeros((n_patches, Nr, Nxi_half, Neta_half))
dE2d1 = N.zeros((n_patches, Nr, Nxi_half, Neta_half))
dErd1 = N.zeros((n_patches, Nr, Nxi_half,  Neta_int))
dErd2 = N.zeros((n_patches, Nr, Nxi_int, Neta_half))
dE1dr = N.zeros((n_patches, Nr, Nxi_half, Neta_int))
dE2dr = N.zeros((n_patches, Nr, Nxi_int, Neta_half))

dBrd1 = N.zeros((n_patches, Nr, Nxi_int,  Neta_half))
dBrd2 = N.zeros((n_patches, Nr, Nxi_half, Neta_int))
dB1d2 = N.zeros((n_patches, Nr, Nxi_int, Neta_int))
dB2d1 = N.zeros((n_patches, Nr, Nxi_int, Neta_int))
dB1dr = N.zeros((n_patches, Nr, Nxi_int, Neta_half))
dB2dr = N.zeros((n_patches, Nr, Nxi_half, Neta_int))

# Interface terms
diff_Br = N.zeros((n_patches, Nr, Nxi_half, Neta_half))
diff_E1 = N.zeros((n_patches, Nr, Nxi_half, Neta_int))
diff_E2 = N.zeros((n_patches, Nr, Nxi_int, Neta_half))
diff_Er = N.zeros((n_patches, Nr, Nxi_int, Neta_int))
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

g11d = N.empty((Nr, Nxi_int, Neta_int, 7))
g12d = N.empty((Nr, Nxi_int, Neta_int, 7))
g22d = N.empty((Nr, Nxi_int, Neta_int, 7))

sqrt_det_g_half = N.empty((Nr, Nxi_half))
g12d_half = N.empty((Nr, Nxi_half))
g11d_half = N.empty((Nr, Nxi_half))
g22d_half = N.empty((Nr, Nxi_half))

sqrt_det_g_int = N.empty((Nr, Nxi_int))
g12d_int = N.empty((Nr, Nxi_int))
g11d_int = N.empty((Nr, Nxi_int))
g22d_int = N.empty((Nr, Nxi_int))

for i in range(Nxi_int):
    for j in range(Neta_int):
        for k in range(Nr):

            # 0 at (k, i, j)
            r0 = r[k]
            X = N.tan(xi_int[i])
            Y = N.tan(eta_int[j])
            C = N.sqrt(1.0 + X * X)
            D = N.sqrt(1.0 + Y * Y)
            delta = N.sqrt(1.0 + X * X + Y * Y)

            g11d[k, i, j, 0] = (r0 * C * C * D / (delta * delta))**2
            g22d[k, i, j, 0] = (r0 * C * D * D / (delta * delta))**2
            g12d[k, i, j, 0] = - r0 * r0 * X * Y * C * C * D * D / (delta)**4

            # 1 at (k, i + 1/2, j)
            r0 = r[k]
            X = N.tan(xi_int[i] + 0.5 * dxi)
            Y = N.tan(eta_int[j])
            C = N.sqrt(1.0 + X * X)
            D = N.sqrt(1.0 + Y * Y)
            delta = N.sqrt(1.0 + X * X + Y * Y)

            g11d[k, i, j, 1] = (r0 * C * C * D / (delta * delta))**2
            g22d[k, i, j, 1] = (r0 * C * D * D / (delta * delta))**2
            g12d[k, i, j, 1] = - r0 * r0 * X * Y * C * C * D * D / (delta)**4

            # 2 at (k, i, j + 1/2)
            r0 = r[k]
            X = N.tan(xi_int[i])
            Y = N.tan(eta_int[j] + 0.5 * deta)
            C = N.sqrt(1.0 + X * X)
            D = N.sqrt(1.0 + Y * Y)
            delta = N.sqrt(1.0 + X * X + Y * Y)

            g11d[k, i, j, 2] = (r0 * C * C * D / (delta * delta))**2
            g22d[k, i, j, 2] = (r0 * C * D * D / (delta * delta))**2
            g12d[k, i, j, 2] = - r0 * r0 * X * Y * C * C * D * D / (delta)**4

            # 3 at (k, i + 1/2, j + 1/2)
            r0 = r[k]
            X = N.tan(xi_int[i] + 0.5 * dxi)
            Y = N.tan(eta_int[j] + 0.5 * deta)
            C = N.sqrt(1.0 + X * X)
            D = N.sqrt(1.0 + Y * Y)
            delta = N.sqrt(1.0 + X * X + Y * Y)

            g11d[k, i, j, 3] = (r0 * C * C * D / (delta * delta))**2
            g22d[k, i, j, 3] = (r0 * C * D * D / (delta * delta))**2
            g12d[k, i, j, 3] = - r0 * r0 * X * Y * C * C * D * D / (delta)**4

            # 4 at (k + 1/2, i, j)
            r0 = r_yee[k]
            X = N.tan(xi_int[i])
            Y = N.tan(eta_int[j])
            C = N.sqrt(1.0 + X * X)
            D = N.sqrt(1.0 + Y * Y)
            delta = N.sqrt(1.0 + X * X + Y * Y)

            g11d[k, i, j, 4] = (r0 * C * C * D / (delta * delta))**2
            g22d[k, i, j, 4] = (r0 * C * D * D / (delta * delta))**2
            g12d[k, i, j, 4] = - r0 * r0 * X * Y * C * C * D * D / (delta)**4

            # 5 at (k + 1/2, i + 1/2, j)
            r0 = r_yee[k]
            X = N.tan(xi_int[i] + 0.5 * dxi)
            Y = N.tan(eta_int[j])
            C = N.sqrt(1.0 + X * X)
            D = N.sqrt(1.0 + Y * Y)
            delta = N.sqrt(1.0 + X * X + Y * Y)

            g11d[k, i, j, 5] = (r0 * C * C * D / (delta * delta))**2
            g22d[k, i, j, 5] = (r0 * C * D * D / (delta * delta))**2
            g12d[k, i, j, 5] = - r0 * r0 * X * Y * C * C * D * D / (delta)**4

            # 6 at (k + 1/2, i, j + 1/2)
            r0 = r_yee[k]
            X = N.tan(xi_int[i])
            Y = N.tan(eta_int[j] + 0.5 * deta)
            C = N.sqrt(1.0 + X * X)
            D = N.sqrt(1.0 + Y * Y)
            delta = N.sqrt(1.0 + X * X + Y * Y)

            g11d[k, i, j, 6] = (r0 * C * C * D / (delta * delta))**2
            g22d[k, i, j, 6] = (r0 * C * D * D / (delta * delta))**2
            g12d[k, i, j, 6] = - r0 * r0 * X * Y * C * C * D * D / (delta)**4

# Define sqrt of determinant of metric
sqrt_det_g = N.sqrt(g11d * g22d - g12d * g12d)

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

def interp(arr_in, xA, xB):
    return N.interp(xB, xA, arr_in)

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

def compute_diff_B(p):
    
    dBrd1[p, :, 0, :] = (- 0.5 * Br[p, :, 0, :] + 0.25 * Br[p, :, 1, :] + 0.25 * Br[p, :, 2, :]) / dxi / P_int_2[0]
    dBrd1[p, :, 1, :] = (- 0.5 * Br[p, :, 0, :] - 0.25 * Br[p, :, 1, :] + 0.75 * Br[p, :, 2, :]) / dxi / P_int_2[1]
    dBrd1[p, :, Nxi_int - 2, :] = (- 0.75 * Br[p, :, -3, :] + 0.25 * Br[p, :, -2, :] + 0.5 * Br[p, :, -1, :]) / dxi / P_int_2[Nxi_int - 2]
    dBrd1[p, :, Nxi_int - 1, :] = (- 0.25 * Br[p, :, -3, :] - 0.25 * Br[p, :, -2, :] + 0.5 * Br[p, :, -1, :]) / dxi / P_int_2[Nxi_int - 1]
    dBrd1[p, :, 2:(Nxi_int - 2), :] = (N.roll(Br, -1, axis = 2)[p, :, 2:(Nxi_int - 2), :] - Br[p, :, 2:(Nxi_int - 2), :]) / dxi

    dBrd2[p, :, :, 0] = (- 0.5 * Br[p, :, :, 0] + 0.25 * Br[p, :, :, 1] + 0.25 * Br[p, :, :, 2]) / deta / P_int_2[0]
    dBrd2[p, :, :, 1] = (- 0.5 * Br[p, :, :, 0] - 0.25 * Br[p, :, :, 1] + 0.75 * Br[p, :, :, 2]) / deta / P_int_2[1]
    dBrd2[p, :, :, Nxi_int - 2] = (- 0.75 * Br[p, :, :, -3] + 0.25 * Br[p, :, :, -2] + 0.5 * Br[p, :, :, -1]) / deta / P_int_2[Nxi_int - 2]
    dBrd2[p, :, :, Nxi_int - 1] = (- 0.25 * Br[p, :, :, -3] - 0.25 * Br[p, :, :, -2] + 0.5 * Br[p, :, :, -1]) / deta / P_int_2[Nxi_int - 1]
    dBrd2[p, :, :, 2:(Neta_int - 2)] = (N.roll(Br, -1, axis = 3)[p, :, :, 2:(Neta_int - 2)] - Br[p, :, :, 2:(Neta_int - 2)]) / deta

    dB1dr[p, NG:(Nr0 + NG), :, :] = (B1d[p, NG:(Nr0 + NG), :, :] - N.roll(B1d, 1, axis = 1)[p, NG:(Nr0 + NG), :, :]) / dr
    dB2dr[p, NG:(Nr0 + NG), :, :] = (B2d[p, NG:(Nr0 + NG), :, :] - N.roll(B2d, 1, axis = 1)[p, NG:(Nr0 + NG), :, :]) / dr

    dB1d2[p, :, :, 0] = (- 0.5 * B1d[p, :, :, 0] + 0.25 * B1d[p, :, :, 1] + 0.25 * B1d[p, :, :, 2]) / deta / P_int_2[0]
    dB1d2[p, :, :, 1] = (- 0.5 * B1d[p, :, :, 0] - 0.25 * B1d[p, :, :, 1] + 0.75 * B1d[p, :, :, 2]) / deta / P_int_2[1]
    dB1d2[p, :, :, Nxi_int - 2] = (- 0.75 * B1d[p, :, :, -3] + 0.25 * B1d[p, :, :, -2] + 0.5 * B1d[p, :, :, -1]) / deta / P_int_2[Nxi_int - 2]
    dB1d2[p, :, :, Nxi_int - 1] = (- 0.25 * B1d[p, :, :, -3] - 0.25 * B1d[p, :, :, -2] + 0.5 * B1d[p, :, :, -1]) / deta / P_int_2[Nxi_int - 1]
    dB1d2[p, :, :, 2:(Neta_int - 2)] = (N.roll(B1d, -1, axis = 3)[p, :, :, 2:(Neta_int - 2)] - B1d[p, :, :, 2:(Neta_int - 2)]) / deta

    dB2d1[p, :, 0, :] = (- 0.5 * B2d[p, :, 0, :] + 0.25 * B2d[p, :, 1, :] + 0.25 * B2d[p, :, 2, :]) / dxi / P_int_2[0]
    dB2d1[p, :, 1, :] = (- 0.5 * B2d[p, :, 0, :] - 0.25 * B2d[p, :, 1, :] + 0.75 * B2d[p, :, 2, :]) / dxi / P_int_2[1]
    dB2d1[p, :, Nxi_int - 2, :] = (- 0.75 * B2d[p, :, -3, :] + 0.25 * B2d[p, :, -2, :] + 0.5 * B2d[p, :, -1, :]) / dxi / P_int_2[Nxi_int - 2]
    dB2d1[p, :, Nxi_int - 1, :] = (- 0.25 * B2d[p, :, -3, :] - 0.25 * B2d[p, :, -2, :] + 0.5 * B2d[p, :, -1, :]) / dxi / P_int_2[Nxi_int - 1]
    dB2d1[p, :, 2:(Nxi_int - 2), :] = (N.roll(B2d, -1, axis = 2)[p, :, 2:(Nxi_int - 2), :] - B2d[p, :, 2:(Nxi_int - 2), :]) / dxi

def compute_diff_B_low(p):
    
    dBrd1[p, :, 0, :] = (- Br[p, :, 0, :] + Br[p, :, 1, :]) / dxi / 0.5
    dBrd1[p, :, Nxi_int - 1, :] = (-  Br[p, :, -2, :] + Br[p, :, -1, :]) / dxi / 0.5
    dBrd1[p, :, 1:(Nxi_int - 1), :] = (N.roll(Br, -1, axis = 2)[p, :, 1:(Nxi_int - 1), :] - Br[p, :, 1:(Nxi_int - 1), :]) / dxi

    dBrd2[p, :, :, 0] = (- Br[p, :, :, 0] + Br[p, :, :, 1]) / deta / 0.5
    dBrd2[p, :, :, Nxi_int - 1] = (- Br[p, :, :, -2] + Br[p, :, :, -1]) / deta / 0.5
    dBrd2[p, :, :, 1:(Neta_int - 1)] = (N.roll(Br, -1, axis = 3)[p, :, :, 1:(Neta_int - 1)] - Br[p, :, :, 1:(Neta_int - 1)]) / deta

    dB1dr[p, NG:(Nr0 + NG), :, :] = (B1d[p, NG:(Nr0 + NG), :, :] - N.roll(B1d, 1, axis = 1)[p, NG:(Nr0 + NG), :, :]) / dr
    dB2dr[p, NG:(Nr0 + NG), :, :] = (B2d[p, NG:(Nr0 + NG), :, :] - N.roll(B2d, 1, axis = 1)[p, NG:(Nr0 + NG), :, :]) / dr

    dB1d2[p, :, :, 0] = (- B1d[p, :, :, 0] + B1d[p, :, :, 1]) / deta / 0.5
    dB1d2[p, :, :, Nxi_int - 1] = (- B1d[p, :, :, -2] + B1d[p, :, :, -1]) / deta / 0.5
    dB1d2[p, :, :, 1:(Neta_int - 1)] = (N.roll(B1d, -1, axis = 3)[p, :, :, 1:(Neta_int - 1)] - B1d[p, :, :, 1:(Neta_int - 1)]) / deta

    dB2d1[p, :, 0, :] = (- B2d[p, :, 0, :] + B2d[p, :, 1, :]) / dxi / 0.5
    dB2d1[p, :, Nxi_int - 1, :] = (- B2d[p, :, -2, :] + B2d[p, :, -1, :]) / dxi / 0.5
    dB2d1[p, :, 1:(Nxi_int - 1), :] = (N.roll(B2d, -1, axis = 2)[p, :, 1:(Nxi_int - 1), :] - B2d[p, :, 1:(Nxi_int - 1), :]) / dxi

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

    dErd1[p, :, 0, :] = (- 0.50 * Er[p, :, 0, :] + 0.50 * Er[p, :, 1, :]) / dxi / P_half_2[0]
    dErd1[p, :, 1, :] = (- 0.25 * Er[p, :, 0, :] + 0.25 * Er[p, :, 1, :]) / dxi / P_half_2[1]
    dErd1[p, :, 2, :] = (- 0.25 * Er[p, :, 0, :] - 0.75 * Er[p, :, 1, :] + Er[p, :, 2, :]) / dxi / P_half_2[2]
    dErd1[p, :, Nxi_half - 3, :] = (- Er[p, :, -3, :] + 0.75 * Er[p, :, -2, :] + 0.25 * Er[p, :, -1, :]) / dxi / P_half_2[Nxi_half - 3]
    dErd1[p, :, Nxi_half - 2, :] = (- 0.25 * Er[p, :, -2, :] + 0.25 * Er[p, :, -1, :]) / dxi / P_half_2[Nxi_half - 2]
    dErd1[p, :, Nxi_half - 1, :] = (- 0.5 * Er[p, :, -2, :] + 0.5 * Er[p, :, -1, :]) / dxi / P_half_2[Nxi_half - 1]
    dErd1[p, :, 3:(Nxi_half - 3), :] = (Er[p, :, 3:(Nxi_half - 3), :] - N.roll(Er, 1, axis = 2)[p, :, 3:(Nxi_half - 3), :]) / dxi

    dErd2[p, :, :, 0] = (- 0.50 * Er[p, :, :, 0] + 0.50 * Er[p, :, :, 1]) / dxi / P_half_2[0]
    dErd2[p, :, :, 1] = (- 0.25 * Er[p, :, :, 0] + 0.25 * Er[p, :, :, 1]) / dxi / P_half_2[1]
    dErd2[p, :, :, 2] = (- 0.25 * Er[p, :, :, 0] - 0.75 * Er[p, :, :, 1] + Er[p, :, :, 2]) / dxi / P_half_2[2]
    dErd2[p, :, :, Neta_half - 3] = (- Er[p, :, :, -3] + 0.75 * Er[p, :, :, -2] + 0.25 * Er[p, :, :, -1]) / deta / P_half_2[Nxi_half - 3]
    dErd2[p, :, :, Neta_half - 2] = (- 0.25 * Er[p, :, :, -2] + 0.25 * Er[p, :, :, -1]) / deta / P_half_2[Nxi_half - 2]
    dErd2[p, :, :, Neta_half - 1] = (- 0.50 * Er[p, :, :, -2] + 0.50 * Er[p, :, :, -1]) / deta / P_half_2[Nxi_half - 1]
    dErd2[p, :, :, 3:(Neta_half - 3)] = (Er[p, :, :, 3:(Neta_half - 3)] - N.roll(Er, 1, axis = 3)[p, :, :, 3:(Neta_half - 3)]) / deta

def compute_diff_E_low(p):

    dE2d1[p, :, 0, :] = (- E2d[p, :, 0, :] + E2d[p, :, 1, :]) / dxi
    dE2d1[p, :, 1, :] = (- E2d[p, :, 0, :] + E2d[p, :, 1, :]) / dxi
    dE2d1[p, :, Nxi_half - 2, :] = (- E2d[p, :, -2, :] + E2d[p, :, -1, :]) / dxi
    dE2d1[p, :, Nxi_half - 1, :] = (- E2d[p, :, -2, :] + E2d[p, :, -1, :]) / dxi
    dE2d1[p, :, 2:(Nxi_half - 2), :] = (E2d[p, :, 2:(Nxi_half - 2), :] - N.roll(E2d, 1, axis = 2)[p, :, 2:(Nxi_half - 2), :]) / dxi

    dE1d2[p, :, :, 0] = (- E1d[p, :, :, 0] + E1d[p, :, :, 1]) / dxi
    dE1d2[p, :, :, 1] = (- E1d[p, :, :, 0] + E1d[p, :, :, 1]) / dxi
    dE1d2[p, :, :, Neta_half - 2] = (- E1d[p, :, :, -2] + E1d[p, :, :, -1]) / deta
    dE1d2[p, :, :, Neta_half - 1] = (- E1d[p, :, :, -2] + E1d[p, :, :, -1]) / deta
    dE1d2[p, :, :, 2:(Neta_half - 2)] = (E1d[p, :, :, 2:(Neta_half - 2)] - N.roll(E1d, 1, axis = 3)[p, :, :, 2:(Neta_half - 2)]) / deta

    dE1dr[p, NG:(Nr0 + NG), :, :] = (N.roll(E1d, -1, axis = 1)[p, NG:(Nr0 + NG), :, :] - E1d[p, NG:(Nr0 + NG), :, :]) / dr
    dE2dr[p, NG:(Nr0 + NG), :, :] = (N.roll(E2d, -1, axis = 1)[p, NG:(Nr0 + NG), :, :] - E2d[p, NG:(Nr0 + NG), :, :]) / dr

    dErd1[p, :, 0, :] = (- Er[p, :, 0, :] + Er[p, :, 1, :]) / dxi
    dErd1[p, :, 1, :] = (- Er[p, :, 0, :] + Er[p, :, 1, :]) / dxi
    dErd1[p, :, Nxi_half - 2, :] = (- Er[p, :, -2, :] + Er[p, :, -1, :]) / dxi
    dErd1[p, :, Nxi_half - 1, :] = (- Er[p, :, -2, :] + Er[p, :, -1, :]) / dxi
    dErd1[p, :, 3:(Nxi_half - 3), :] = (Er[p, :, 3:(Nxi_half - 3), :] - N.roll(Er, 1, axis = 2)[p, :, 3:(Nxi_half - 3), :]) / dxi

    dErd2[p, :, :, 0] = (- Er[p, :, :, 0] + Er[p, :, :, 1]) / dxi
    dErd2[p, :, :, 1] = (- Er[p, :, :, 0] + Er[p, :, :, 1]) / dxi
    dErd2[p, :, :, Neta_half - 2] = (- Er[p, :, :, -2] + Er[p, :, :, -1]) / deta
    dErd2[p, :, :, Neta_half - 1] = (- Er[p, :, :, -2] + Er[p, :, :, -1]) / deta
    dErd2[p, :, :, 2:(Neta_half - 2)] = (Er[p, :, :, 2:(Neta_half - 2)] - N.roll(Er, 1, axis = 3)[p, :, :, 2:(Neta_half - 2)]) / deta

########
# Single-patch push routines
########

def contra_to_cov_E(p):

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

def push_E(p, dtin):

        Er[p, :, :, :] += dtin * (dB2d1[p, :, :, :] - dB1d2[p, :, :, :]) / sqrt_det_g[:, :, :, 4] 

        # Interior
        E1u[p, :, 1:-1, :] += dtin * (dBrd2[p, :, 1:-1, :] - dB2dr[p, :, 1:-1, :]) / sqrt_det_g[:, 0:-1, :, 1] 
        # Left edge
        E1u[p, :, 0, :] += dtin * (dBrd2[p, :, 0, :] - dB2dr[p, :, 0, :]) / sqrt_det_g[:, 0, :, 0] 
        # Right edge
        E1u[p, :, -1, :] += dtin * (dBrd2[p, :, -1, :] - dB2dr[p, :, -1, :]) / sqrt_det_g[:, -1, :, 0]

        # Interior
        E2u[p, :, :, 1:-1] += dtin * (dB1dr[p, :, :, 1:-1] - dBrd1[p, :, :, 1:-1]) / sqrt_det_g[:, :, 0:-1, 2]
        # Bottom edge
        E2u[p, :, :, 0] += dtin * (dB1dr[p, :, :, 0] - dBrd1[p, :, :, 0]) / sqrt_det_g[:, :, 0, 0]
        # Top edge
        E2u[p, :, :, -1] += dtin * (dB1dr[p, :, :, -1] - dBrd1[p, :, :, -1]) / sqrt_det_g[:, :, -1, 0]

def push_B(p, dtin):
        
        # Interior
        Br[p, :, 1:-1, 1:-1] += dtin * (dE1d2[p, :, 1:-1, 1:-1] - dE2d1[p, :, 1:-1, 1:-1]) / sqrt_det_g[:, 0:-1, 0:-1, 3] 
        # Left edge
        Br[p, :, 0, 1:-1] += dtin * (dE1d2[p, :, 0, 1:-1] - dE2d1[p, :, 0, 1:-1]) / sqrt_det_g[:, 0, 0:-1, 2] 
        # Right edge
        Br[p, :, -1, 1:-1] += dtin * (dE1d2[p, :, -1, 1:-1] - dE2d1[p, :, -1, 1:-1]) / sqrt_det_g[:, -1, 0:-1, 2] 
        # Bottom edge
        Br[p, :, 1:-1, 0] += dtin * (dE1d2[p, :, 1:-1, 0] - dE2d1[p, :, 1:-1, 0]) / sqrt_det_g[:, 0:-1, 0, 1] 
        # Top edge
        Br[p, :, 1:-1, -1] += dtin * (dE1d2[p, :, 1:-1, -1] - dE2d1[p, :, 1:-1, -1]) / sqrt_det_g[:, 0:-1, -1, 1] 
        # Bottom left corner
        Br[p, :, 0, 0] += dtin * (dE1d2[p, :, 0, 0] - dE2d1[p, :, 0, 0]) / sqrt_det_g[:, 0, 0, 0] 
        # Bottom right corner
        Br[p, :, -1, 0] += dtin * (dE1d2[p, :, -1, 0] - dE2d1[p, :, -1, 0]) / sqrt_det_g[:, -1, 0, 0] 
        # Top left corner
        Br[p, :, 0, -1] += dtin * (dE1d2[p, :, 0, -1] - dE2d1[p, :, 0, -1]) / sqrt_det_g[:, 0, -1, 0] 
        # Top right corner
        Br[p, :, -1, -1] += dtin * (dE1d2[p, :, -1, -1] - dE2d1[p, :, -1, -1]) / sqrt_det_g[:, -1, -1, 0] 

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
        Br_0 = Brin[p0, :, -1, :]
        Br_1 = Brin[p1, :, :, 0]
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
        Br_0 = Brin[p0, :, -1, :]
        Br_1 = Brin[p1, :, :, 0]
        Beta_0 = B2in[p0, :, -1, :]
        Bxi_1  = B1in[p1, :, :, 0]
        Er_0 = Erin[p0, :, -1, :]
        Er_1 = Erin[p1, :, :, 0]
        
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
# Absorbing boundary conditions at r_max
########

i_abs = 5 # Thickness of absorbing layer in number of cells
r_abs = r[Nr - i_abs]
delta = ((r - r_abs) / (r_max - r_abs)) * N.heaviside(r - r_abs, 0.0)
sigma = N.exp(- 1.0 * delta**3)

delta = ((r_yee - r_abs) / (r_max - r_abs)) * N.heaviside(r_yee - r_abs, 0.0)
sigma_yee = N.exp(- 1.0 * delta**3)

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
    # Br[patch, :, :, :] *= sigma[:, None, None]
    # B1u[patch, :, :, :] *= sigma_yee[:, None, None]
    # B2u[patch, :, :, :] *= sigma_yee[:, None, None]
    return

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
    return 2.0 * B0 * (N.sin(tilt) * N.sin(th0) * N.cos(ph0) + N.cos(th0) * N.cos(tilt)) / r0**3
    # return B0 * N.sin(th0)**3 * N.cos(3.0 * ph0) 
    # return B0 * N.cos(th0) 


def func_Bth(r0, th0, ph0):
    # return - B0 * N.sin(th0) / r0
    return B0 * (N.cos(tilt) * N.sin(th0) - N.sin(tilt) * N.cos(th0) * N.cos(ph0)) / r0**4
#    return 0.0

def func_Bph(r0, th0, ph0):
    return B0 * (N.sin(tilt) * N.sin(ph0) / N.sin(th0)) / r0**4
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

InitialData()

########
# Boundary conditions at r_min
########

rlc = 2.0 # Light cylinder radius
omega = 1.0 / rlc # Spin velocity of the conductor at r_min
# period = 2.0 * N.pi / omega
pem = (2.0/3.0)*(omega**4)*N.sin(tilt)**2

def func_Br_a(r0, th0, ph0):
    return 2.0 * B0 * N.sin(tilt) * N.sin(th0) * N.cos(ph0) / r0**3

def func_Br_b(r0, th0, ph0):
    return 2.0 * B0 * N.sin(tilt) * N.sin(th0) * N.sin(ph0) / r0**3

def func_Br_c(r0, th0, ph0):
    return 2.0 * B0 * N.cos(tilt) * N.cos(th0) / r0**3

def func_Eth_a(r0, th0, ph0, omega0):
    return - r0 * r0 * omega0 * func_Br_a(r_min, th0, ph0) * N.sin(th0)

def func_Eth_b(r0, th0, ph0, omega0):
    return - r0 * r0 * omega0 * func_Br_b(r_min, th0, ph0) * N.sin(th0)

def func_Eth_c(r0, th0, ph0, omega0):
    return - r0 * r0 * omega0 * func_Br_c(r_min, th0, ph0) * N.sin(th0)

# Fields at r_min
E1_surf_a = N.zeros((n_patches, Nxi_half, Neta_int))
E2_surf_a = N.zeros((n_patches, Nxi_int, Neta_half))
E1_surf_b = N.zeros((n_patches, Nxi_half, Neta_int))
E2_surf_b = N.zeros((n_patches, Nxi_int, Neta_half))
E1_surf_c = N.zeros((n_patches, Nxi_half, Neta_int))
E2_surf_c = N.zeros((n_patches, Nxi_int, Neta_half))
Br_surf_a = N.zeros((n_patches, Nxi_half, Neta_half))
Br_surf_b = N.zeros((n_patches, Nxi_half, Neta_half))
Br_surf_c = N.zeros((n_patches, Nxi_half, Neta_half))

for patch in range(6):

    fvec = (globals()["vec_sph_to_" + sphere[patch]])
    fcoord = (globals()["coord_" + sphere[patch] + "_to_sph"])

    for i in range(Nxi_half):
        for j in range(Neta_half):

            r0 = r_min
            th0, ph0 = fcoord(xi_half[i], eta_half[j])
            Br_surf_a[patch, i, j] = func_Br_a(r0, th0, ph0)
            Br_surf_b[patch, i, j] = func_Br_b(r0, th0, ph0)
            Br_surf_c[patch, i, j] = func_Br_c(r0, th0, ph0)

    for i in range(Nxi_half):
        for j in range(Neta_int):

            r0 = r_min
            th0, ph0 = fcoord(xi_half[i], eta_int[j])
            
            EtTMP = func_Eth_a(r0, th0, ph0, omega)
            EpTMP = 0.0
            omega_patch = fvec(th0, ph0, EtTMP, EpTMP)
            E1_surf_a[patch, i, j] = omega_patch[0]

            EtTMP = func_Eth_b(r0, th0, ph0, omega)
            EpTMP = 0.0
            omega_patch = fvec(th0, ph0, EtTMP, EpTMP)
            E1_surf_b[patch, i, j] = omega_patch[0]

            EtTMP = func_Eth_c(r0, th0, ph0, omega)
            EpTMP = 0.0
            omega_patch = fvec(th0, ph0, EtTMP, EpTMP)
            E1_surf_c[patch, i, j] = omega_patch[0]

    for i in range(Nxi_int):
        for j in range(Neta_half):

            r0 = r_min
            th0, ph0 = fcoord(xi_int[i], eta_half[j])

            EtTMP = func_Eth_a(r0, th0, ph0, omega)
            EpTMP = 0.0
            omega_patch = fvec(th0, ph0, EtTMP, EpTMP)
            E2_surf_a[patch, i, j] = omega_patch[1]

            EtTMP = func_Eth_b(r0, th0, ph0, omega)
            EpTMP = 0.0
            omega_patch = fvec(th0, ph0, EtTMP, EpTMP)
            E2_surf_b[patch, i, j] = omega_patch[1]

            EtTMP = func_Eth_c(r0, th0, ph0, omega)
            EpTMP = 0.0
            omega_patch = fvec(th0, ph0, EtTMP, EpTMP)
            E2_surf_c[patch, i, j] = omega_patch[1]

########
# Perfectly conducting boundary conditions at r_min
########

def BC_B_metal_rmin(it, patch):
    t0 = omega * it * dt
    Br[patch, NG, :, :]   = Br_surf_a[patch, :, :] * N.cos(t0) + Br_surf_b[patch, :, :] * N.sin(t0) + Br_surf_c[patch, :, :]
    # B1u[patch, :NG, :, :] = 0.0
    # B2u[patch, :NG, :, :] = 0.0

def BC_E_metal_rmin(it, patch):
    t0 = omega * it * dt
    # Er[patch, :NG, :, :] = 0.0
    E1u[patch, NG, :, :] = E1_surf_a[patch, :, :] * N.cos(t0) + E1_surf_b[patch, :, :] * N.sin(t0) + E1_surf_c[patch, :, :]
    E2u[patch, NG, :, :] = E2_surf_a[patch, :, :] * N.cos(t0) + E2_surf_b[patch, :, :] * N.sin(t0) + E2_surf_c[patch, :, :]

    # NS rotation profile to avoid high frequency waves in FD solver    
    E1u[patch, NG, :, :] *= 0.5 * (1.0 - N.tanh(2.0 - it*dt / 0.5))
    E2u[patch, NG, :, :] *= 0.5 * (1.0 - N.tanh(2.0 - it*dt / 0.5))

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
    
    figsave_png(fig, "../snapshots_3d/E1u_" + str(it))

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
    
    figsave_png(fig, "../snapshots_3d/E2u_" + str(it))

    P.close('all')

########
# Initialization
########

idump = 0

Nt = 10001 # Number of iterations
FDUMP = 100 # Dump frequency
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
        plot_fields_unfolded_Br(idump, 0.5, 5)
        plot_fields_unfolded_B1(idump, 0.1, 5)
        plot_fields_unfolded_E1(idump, 0.1, 5)
        plot_fields_unfolded_E2(idump, 0.1, 5)
        WriteAllFieldsHDF5(idump)
        idump += 1

    diff_Br[:, :, :, :] = 0.0
    diff_E1[:, :, :, :] = 0.0
    diff_E2[:, :, :, :] = 0.0
    diff_Er[:, :, :, :] = 0.0
    diff_B1[:, :, :, :] = 0.0
    diff_B2[:, :, :, :] = 0.0
    
    # compute_diff_B(patches)
    compute_diff_B_low(patches)

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
    # compute_diff_E_low(patches)
    
    push_B(patches, dt)
    
    for i in range(n_zeros):
        p0, p1 = index_row[i], index_col[i]
        compute_penalty_B(p0, p1, dt, Er, E1d, E2d, Br, B1d, B2d)
        interface_B(p0, p1)
        # compute_delta_B(p0, p1, dt, E1d, E2d, Br) # Different time-stepping?

    corners_B(patches)

    BC_B_metal_rmin(it, patches)
    BC_B_absorb(patches)
    BC_B_metal_rmax(patches)

    contra_to_cov_B(patches)

