# Import modules
import numpy as N
import matplotlib.pyplot as P
import matplotlib
from skimage.measure import find_contours
from math import *
from tqdm import tqdm
import matplotlib.pyplot as plt
from scipy import interpolate, ndimage
import scipy.integrate as spi

# Import my figure routines
import sys
sys.path.append('../')
sys.path.append('../transformations/')

from figure_module import *

import warnings
import matplotlib.cbook

import warnings
import matplotlib.cbook
warnings.filterwarnings("ignore",category=matplotlib.cbook.mplDeprecation)

# I/O
import h5py

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

########    
# Topology of the patches
########

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

# Corners triplets
n_corners = 8
triplet = N.zeros((n_corners,3), dtype = int)
triplet[0, :] = Sphere.A, Sphere.B, Sphere.N
triplet[1, :] = Sphere.A, Sphere.B, Sphere.S
triplet[2, :] = Sphere.B, Sphere.C, Sphere.N
triplet[3, :] = Sphere.B, Sphere.C, Sphere.S
triplet[4, :] = Sphere.C, Sphere.D, Sphere.N
triplet[5, :] = Sphere.C, Sphere.D, Sphere.S
triplet[6, :] = Sphere.D, Sphere.A, Sphere.N
triplet[7, :] = Sphere.D, Sphere.A, Sphere.S

# Indices of corners for patches p0, p1, p2
index_corners = N.zeros((n_corners, 3), dtype = object)
index_corners[0,0] = (-1, -1)
index_corners[0,1] = (0 , -1)
index_corners[0,2] = (0 ,  0)
index_corners[1,0] = (-1,  0)
index_corners[1,1] = (0 ,  0)
index_corners[1,2] = (-1, -1)
index_corners[2,0] = (-1, -1)
index_corners[2,1] = (0 ,  0)
index_corners[2,2] = (-1,  0)
index_corners[3,0] = (-1,  0)
index_corners[3,1] = (-1,  0)
index_corners[3,2] = (-1,  0)
index_corners[4,0] = (0 , -1)
index_corners[4,1] = (0 ,  0)
index_corners[4,2] = (-1, -1)
index_corners[5,0] = (-1, -1)
index_corners[5,1] = (-1,  0)
index_corners[5,2] = (0 ,  0)
index_corners[6,0] = (0 , -1)
index_corners[6,1] = (0 , -1)
index_corners[6,2] = (0 , -1)
index_corners[7,0] = (-1, -1)
index_corners[7,1] = (0 ,  0)
index_corners[7,2] = (0 , -1)

# Parameters
cfl = 0.5
Nxi  = 64 # Number of cells in xi
Neta = 64 # Number of cells in eta

Nxi_int = Nxi + 1 # Number of integer points
Nxi_half = Nxi + 2 # Number of hlaf-step points
Neta_int = Neta + 1 # Number of integer points
Neta_half = Neta + 2 # NUmber of hlaf-step points

xi_min, xi_max = - N.pi/4.0, N.pi/4.0
dxi = (xi_max - xi_min) / Nxi
eta_min, eta_max = - N.pi/4.0, N.pi/4.0
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
yB1_grid, xB1_grid = N.meshgrid(eta_half, xi_int)
yB2_grid, xB2_grid = N.meshgrid(eta_int, xi_half)

########
# Define metric tensor
########

g11d = N.empty((Nxi_int, Neta_int, 4))
g12d = N.empty((Nxi_int, Neta_int, 4))
g22d = N.empty((Nxi_int, Neta_int, 4))
g11u = N.empty((Nxi_int, Neta_int, 4))
g12u = N.empty((Nxi_int, Neta_int, 4))
g22u = N.empty((Nxi_int, Neta_int, 4))
sqrt_det_g_half = N.empty(Nxi_int)
g12d_half = N.empty(Nxi_int)
g11d_half = N.empty(Nxi_int)
g22d_half = N.empty(Nxi_int)

sqrt_det_g_B1 = N.empty((Nxi_int, Neta_half))
sqrt_det_g_B2 = N.empty((Nxi_half, Neta_int))

for i in range(Nxi_int):
    for j in range(Neta_int):
        
        # 0 at (i, j)
        X = N.tan(xi_int[i])
        Y = N.tan(eta_int[j])
        C = N.sqrt(1.0 + X * X)
        D = N.sqrt(1.0 + Y * Y)
        delta = N.sqrt(1.0 + X * X + Y * Y)
        
        g11d[i, j, 0] = (C * C * D / (delta * delta))**2

        g22d[i, j, 0] = (C * D * D / (delta * delta))**2
        g12d[i, j, 0] = - X * Y * C * C * D * D / (delta)**4
        
        # 1 at (i + 1/2, j)
        X = N.tan(xi_int[i] + 0.5 * dxi)
        Y = N.tan(eta_int[j])
        C = N.sqrt(1.0 + X * X)
        D = N.sqrt(1.0 + Y * Y)
        delta = N.sqrt(1.0 + X * X + Y * Y)
        
        g11d[i, j, 1] = (C * C * D / (delta * delta))**2
        g22d[i, j, 1] = (C * D * D / (delta * delta))**2
        g12d[i, j, 1] = - X * Y * C * C * D * D / (delta)**4
        
        # 2 at (i, j + 1/2)
        X = N.tan(xi_int[i])
        Y = N.tan(eta_int[j] + 0.5 * deta)
        C = N.sqrt(1.0 + X * X)
        D = N.sqrt(1.0 + Y * Y)
        delta = N.sqrt(1.0 + X * X + Y * Y)
        
        g11d[i, j, 2] = (C * C * D / (delta * delta))**2
        g22d[i, j, 2] = (C * D * D / (delta * delta))**2
        g12d[i, j, 2] = - X * Y * C * C * D * D / (delta)**4

        # 3 at (i + 1/2, j + 1/2)
        X = N.tan(xi_int[i] + 0.5 * dxi)
        Y = N.tan(eta_int[j] + 0.5 * deta)
        C = N.sqrt(1.0 + X * X)
        D = N.sqrt(1.0 + Y * Y)
        delta = N.sqrt(1.0 + X * X + Y * Y)
        
        g11d[i, j, 3] = (C * C * D / (delta * delta))**2
        g22d[i, j, 3] = (C * D * D / (delta * delta))**2
        g12d[i, j, 3] = - X * Y * C * C * D * D / (delta)**4

for i in range(Nxi_int):
    for j in range(Neta_half):
        
        # at B1 locations
        X = N.tan(xi_int[i])
        Y = N.tan(eta_half[j])
        C = N.sqrt(1.0 + X * X)
        D = N.sqrt(1.0 + Y * Y)
        delta = N.sqrt(1.0 + X * X + Y * Y)
        
        g11d0 = (C * C * D / (delta * delta))**2
        g22d0 = (C * D * D / (delta * delta))**2
        g12d0 = - X * Y * C * C * D * D / (delta)**4
        sqrt_det_g_B1[i, j] = N.sqrt(g11d0 * g22d0 - g12d0 * g12d0)

for i in range(Nxi_half):
    for j in range(Neta_int):

        # at B2 locations
        X = N.tan(xi_half[i])
        Y = N.tan(eta_int[j])
        C = N.sqrt(1.0 + X * X)
        D = N.sqrt(1.0 + Y * Y)
        delta = N.sqrt(1.0 + X * X + Y * Y)
        
        g11d0 = (C * C * D / (delta * delta))**2
        g22d0 = (C * D * D / (delta * delta))**2
        g12d0 = - X * Y * C * C * D * D / (delta)**4
        sqrt_det_g_B2[i, j] = N.sqrt(g11d0 * g22d0 - g12d0 * g12d0)


for i in range(Nxi_int):
    for j in range(Neta_int):
        for i0 in range(4):
            
            metric = N.array([[g11d[i, j, i0], g12d[i, j, i0]], [g12d[i, j, i0], g22d[i, j, i0]]])
            inv_metric = N.linalg.inv(metric)
            g11u[i, j, i0] = inv_metric[0, 0]
            g12u[i, j, i0] = inv_metric[0, 1]
            g22u[i, j, i0] = inv_metric[1, 1]

for i in range(Nxi_int):        
        X = N.tan(xi_int[i])
        Y = N.tan(eta_int[0])
        C = N.sqrt(1.0 + X * X)
        D = N.sqrt(1.0 + Y * Y)
        delta = N.sqrt(1.0 + X * X + Y * Y)
        g11d_half[i] = (C * C * D / (delta * delta))**2
        g22d_half[i] = (C * D * D / (delta * delta))**2
        g12d_half[i] = - X * Y * C * C * D * D / (delta)**4
        sqrt_det_g_half[i] = N.sqrt(g11d_half[i] * g22d_half[i] - g12d_half[i] * g12d_half[i])

sqrt_det_g = N.sqrt(g11d * g22d - g12d * g12d)

dt = cfl * N.min(1.0 / N.sqrt(g11u / (dxi * dxi) + g22u / (deta * deta) + 2.0 * g12d/(dxi*deta)))
print("delta t = {}".format(dt))

# Define fields
Er = N.zeros((n_patches, Nxi_int, Neta_int))
B1u = N.zeros((n_patches, Nxi_int, Neta_half))
B2u = N.zeros((n_patches, Nxi_half,  Neta_int))
B1d = N.zeros((n_patches, Nxi_int, Neta_half))
B2d = N.zeros((n_patches, Nxi_half,  Neta_int))
B1d0 = N.zeros((n_patches, Nxi_int, Neta_half))
B2d0 = N.zeros((n_patches, Nxi_half,  Neta_int))
Er0 = N.zeros((n_patches, Nxi_int, Neta_int))
dErd1 = N.zeros((n_patches, Nxi_half,  Neta_int))
dErd2 = N.zeros((n_patches, Nxi_int, Neta_half))
dB1d2 = N.zeros((n_patches, Nxi_int, Neta_int))
dB2d1 = N.zeros((n_patches, Nxi_int, Neta_int))

diff_Er = N.zeros((n_patches, Nxi_int, Neta_int))
diff_B1 = N.zeros((n_patches, Nxi_int, Neta_half))
diff_B2 = N.zeros((n_patches, Nxi_half, Neta_int))

########
# Generic coordinate transformation
########

from coord_transformations_flip import *

def transform_coords(patch0, patch1, xi0, eta0):
    fcoord0 = (globals()["coord_" + sphere[patch0] + "_to_sph"])
    fcoord1 = (globals()["coord_sph_to_" + sphere[patch1]])
    return fcoord1(*fcoord0(xi0, eta0))

# Generic vector transformation 
########

from vec_transformations_flip import *

def transform_vect(patch0, patch1, xi0, eta0, vxi0, veta0):
    fcoord0 = (globals()["coord_" + sphere[patch0] + "_to_sph"])
    theta0, phi0 = fcoord0(xi0, eta0)
    fvec0 = (globals()["vec_" + sphere[patch0] + "_to_sph"])
    fvec1 = (globals()["vec_sph_to_" + sphere[patch1]])
    return fvec1(theta0, phi0, *fvec0(xi0, eta0, vxi0, veta0))

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

P_int_4 = N.ones(Nxi_int)
P_int_4[0] = 407/1152
P_int_4[1] = 473/384
P_int_4[2] = 343/384
P_int_4[3] = 1177/1152
P_int_4[-1] = 407/1152
P_int_4[-2] = 473/384
P_int_4[-3] = 343/384
P_int_4[-4] = 1177/1152

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

def compute_diff_B(p):
    
    dB2d1[p, 0, :] = (- 0.5 * B2d[p, 0, :] + 0.25 * B2d[p, 1, :] + 0.25 * B2d[p, 2, :]) / dxi / P_int_2[0]
    dB2d1[p, 1, :] = (- 0.5 * B2d[p, 0, :] - 0.25 * B2d[p, 1, :] + 0.75 * B2d[p, 2, :]) / dxi / P_int_2[1]
    
    dB2d1[p, Nxi_int - 2, :] = (- 0.75 * B2d[p, -3, :] + 0.25 * B2d[p, -2, :] + 0.5 * B2d[p, -1, :]) / dxi / P_int_2[Nxi_int - 2]
    dB2d1[p, Nxi_int - 1, :] = (- 0.25 * B2d[p, -3, :] - 0.25 * B2d[p, -2, :] + 0.5 * B2d[p, -1, :]) / dxi / P_int_2[Nxi_int - 1]

    dB2d1[p, 2:(Nxi_int - 2), :] = (N.roll(B2d, -1, axis = 1)[p, 2:(Nxi_int - 2), :] - B2d[p, 2:(Nxi_int - 2), :]) / dxi

    dB1d2[p, :, 0] = (- 0.5 * B1d[p, :, 0] + 0.25 * B1d[p, :, 1] + 0.25 * B1d[p, :, 2]) / deta / P_int_2[0]
    dB1d2[p, :, 1] = (- 0.5 * B1d[p, :, 0] - 0.25 * B1d[p, :, 1] + 0.75 * B1d[p, :, 2]) / deta / P_int_2[1]

    dB1d2[p, :, Nxi_int - 2] = (- 0.75 * B1d[p, :, -3] + 0.25 * B1d[p, :, -2] + 0.5 * B1d[p, :, -1]) / deta / P_int_2[Nxi_int - 2]
    dB1d2[p, :, Nxi_int - 1] = (- 0.25 * B1d[p, :, -3] - 0.25 * B1d[p, :, -2] + 0.5 * B1d[p, :, -1]) / deta / P_int_2[Nxi_int - 1]
        
    dB1d2[p, :, 2:(Neta_int - 2)] = (N.roll(B1d, -1, axis = 2)[p, :, 2:(Neta_int - 2)] - B1d[p, :, 2:(Neta_int - 2)]) / deta

def compute_diff_E(p):

    dErd1[p, 0, :] = (- 0.5 * Er[p, 0, :] + 0.5 * Er[p, 1, :]) / dxi / P_half_2[0]
    dErd1[p, 1, :] = (- 0.25 * Er[p, 0, :] + 0.25 * Er[p, 1, :]) / dxi / P_half_2[1]
    dErd1[p, 2, :] = (- 0.25 * Er[p, 0, :] - 0.75 * Er[p, 1, :] + Er[p, 2, :]) / dxi / P_half_2[2]

    dErd1[p, Nxi_half - 3, :] = (- Er[p, -3, :] + 0.75 * Er[p, -2, :] + 0.25 * Er[p, -1, :]) / dxi / P_half_2[Nxi_half - 3]
    dErd1[p, Nxi_half - 2, :] = (- 0.25 * Er[p, -2, :] + 0.25 * Er[p, -1, :]) / dxi / P_half_2[Nxi_half - 2]
    dErd1[p, Nxi_half - 1, :] = (- 0.5 * Er[p, -2, :] + 0.5 * Er[p, -1, :]) / dxi / P_half_2[Nxi_half - 1]

    dErd1[p, 3:(Nxi_half - 3), :] = (Er[p, 3:(Nxi_half - 3), :] - N.roll(Er, 1, axis = 1)[p, 3:(Nxi_half - 3), :]) / dxi

    dErd2[p, :, 0] = (- 0.5 * Er[p, :, 0] + 0.5 * Er[p, :, 1]) / dxi / P_half_2[0]
    dErd2[p, :, 1] = (- 0.25 * Er[p, :, 0] + 0.25 * Er[p, :, 1]) / dxi / P_half_2[1]
    dErd2[p, :, 2] = (- 0.25 * Er[p, :, 0] - 0.75 * Er[p, :, 1] + Er[p, :, 2]) / dxi / P_half_2[2]

    dErd2[p, :, Neta_half - 3] = (- Er[p, :, -3] + 0.75 * Er[p, :, -2] + 0.25 * Er[p, :, -1]) / deta / P_half_2[Nxi_half - 3]
    dErd2[p, :, Neta_half - 2] = (- 0.25 * Er[p, :, -2] + 0.25 * Er[p, :, -1]) / deta / P_half_2[Nxi_half - 2]
    dErd2[p, :, Neta_half - 1] = (- 0.5 * Er[p, :, -2] + 0.5 * Er[p, :, -1]) / deta / P_half_2[Nxi_half - 1]

    dErd2[p, :, 3:(Neta_half - 3)] = (Er[p, :, 3:(Neta_half - 3)] - N.roll(Er, 1, axis = 2)[p, :, 3:(Neta_half - 3)]) / deta

Jz = N.zeros_like(Er)
Jz[Sphere.B, :, :] = 0.0 * N.exp(- (xEr_grid**2 + yEr_grid**2) / 0.1**2)

def contra_to_cov_B(p):

    ##### Bxi

    # Interior
    B1d[p, 1:-1, 1:-1] = g11d[1:-1, 0:-1, 2] * B1u[p, 1:-1, 1:-1] \
                    + 0.25 * g12d[1:-1, 0:-1, 2] * (B2u[p, 1:-2, 1:] + N.roll(N.roll(B2u, -1, axis = 1), 1, axis = 2)[p, 1:-2, 1:] \
                    + N.roll(B2u, -1, axis = 1)[p, 1:-2, 1:] + N.roll(B2u, 1, axis = 2)[p, 1:-2, 1:])
    # Left edge
    B1d[p, 0, 1:-1] = g11d[0, 0:-1, 2] * B1u[p, 0, 1:-1] + 0.5 * g12d[0, 0:-1, 2] * (B2u[p, 0, 1:] + N.roll(B2u, 1, axis = 2)[p, 0, 1:])
    # Right edge
    B1d[p, -1, 1:-1] = g11d[-1, 0:-1, 2] * B1u[p, -1, 1:-1] + 0.5 * g12d[-1, 0:-1, 2] * (B2u[p, -1, 1:] + N.roll(B2u, 1, axis = 2)[p, -1, 1:])
    # Bottom edge
    B1d[p, 1:-1, 0] = g11d[1:-1, 0, 0] * B1u[p, 1:-1, 0] + 0.5 * g12d[1:-1, 0, 0] * (B2u[p, 1:-2, 0] + N.roll(B2u, -1, axis = 1)[p, 1:-2, 0])
    # Top edge
    B1d[p, 1:-1, -1] = g11d[1:-1, -1, 0] * B1u[p, 1:-1, -1] + 0.5 * g12d[1:-1, -1, 0] * (B2u[p, 1:-2, -1] + N.roll(B2u, -1, axis = 1)[p, 1:-2, -1])
    # Bottom left corner
    B1d[p, 0, 0] = g11d[0, 0, 0] * B1u[p, 0, 0] + g12d[0, 0, 0] * B2u[p, 0, 0]
    # Bottom right corner
    B1d[p, -1, 0] = g11d[-1, 0, 0] * B1u[p, -1, 0] + g12d[-1, 0, 0] * B2u[p, -1, 0]
    # Top left corner
    B1d[p, 0, -1] = g11d[0, -1, 0] * B1u[p, 0, -1] + g12d[0, -1, 0] * B2u[p, 0, -1]
    # Top right corner
    B1d[p, -1, -1] = g11d[-1, -1, 0] * B1u[p, -1, -1] + g12d[-1, -1, 0] * B2u[p, -1, -1]

    ##### Beta

    # Interior
    B2d[p, 1:-1, 1:-1] = g22d[0:-1, 1:-1, 1] * B2u[p, 1:-1, 1:-1] \
                    + 0.25 * g12d[0:-1, 1:-1, 1] * (B1u[p, 1:, 1:-2] + N.roll(N.roll(B1u, 1, axis = 1), -1, axis = 2)[p, 1:, 1:-2] \
                    + N.roll(B1u, 1, axis = 1)[p, 1:, 1:-2] + N.roll(B1u, -1, axis = 2)[p, 1:, 1:-2])
    # Left edge
    B2d[p, 0, 1:-1] = g22d[0, 1:-1, 0] * B2u[p, 0, 1:-1] + 0.5 * g12d[0, 1:-1, 0] * (B1u[p, 0, 1:-2] + N.roll(B1u, -1, axis = 2)[p, 0, 1:-2])
    # Right edge
    B2d[p, -1, 1:-1] = g22d[-1, 1:-1, 0] * B2u[p, -1, 1:-1] + 0.5 * g12d[-1, 1:-1, 0] * (B1u[p, -1, 1:-2] + N.roll(B1u, -1, axis = 2)[p, -1, 1:-2])
    # Bottom edge
    B2d[p, 1:-1, 0] = g22d[0:-1, 0, 1] * B2u[p, 1:-1, 0] + 0.5 * g12d[0:-1, 0, 1] * (B1u[p, 1:, 0] + N.roll(B1u, 1, axis = 1)[p, 1:, 0])
    # Top edge
    B2d[p, 1:-1, -1] = g22d[0:-1, -1, 1] * B2u[p, 1:-1, -1] + 0.5 * g12d[0:-1, -1, 1] * (B1u[p, 1:, -1] + N.roll(B1u, 1, axis = 1)[p, 1:, -1])
    # Bottom left corner
    B2d[p, 0, 0] = g22d[0, 0, 0] * B2u[p, 0, 0] + g12d[0, 0, 0] * B1u[p, 0, 0]
    # Bottom right corner
    B2d[p, -1, 0] = g22d[-1, 0, 0] * B2u[p, -1, 0] + g12d[-1, 0, 0] * B1u[p, -1, 0]
    # Top left corner
    B2d[p, 0, -1] = g22d[0, -1, 0] * B2u[p, 0, -1] + g12d[0, -1, 0] * B1u[p, 0, -1]
    # Top right corner
    B2d[p, -1, -1] = g22d[-1, -1, 0] * B2u[p, -1, -1] + g12d[-1, -1, 0] * B1u[p, -1, -1]

def push_B(p, itime, dtin):

        # Interior
        B1u[p, :, 1:-1] -= dtin * dErd2[p, :, 1:-1] / sqrt_det_g[:, 0:-1, 2]
        # Bottom edge
        B1u[p, :, 0] -= dtin * dErd2[p, :, 0] / sqrt_det_g[:, 0, 0]
        # Top edge
        B1u[p, :, -1] -= dtin * dErd2[p, :, -1] / sqrt_det_g[:, -1, 0]
        
        # Interior
        B2u[p, 1:-1, :] += dtin * dErd1[p, 1:-1, :] / sqrt_det_g[0:-1, :, 1] 
        # Left edge
        B2u[p, 0, :] += dtin * dErd1[p, 0, :] / sqrt_det_g[0, :, 0] 
        # Right edge
        B2u[p, -1, :] += dtin * dErd1[p, -1, :] / sqrt_det_g[-1, :, 0] 

def push_E(p, itime, dtin):

        # Interior
        Er[p, :, :] += dtin * (dB2d1[p, :, :] - dB1d2[p, :, :]) / sqrt_det_g[:, :, 0] 
        
        # Current
        Er[p, :, :] += dtin * Jz[p, :, :] * N.sin(20.0 * itime * dtin) * (1 + N.tanh(20 - itime/5.))/2.

# divB = N.zeros((n_patches, Nxi, Neta))

# def compute_divB(p):

#     divB[p, :, :] = ((N.roll(B1u[p, :-1, 1:-1] * sqrt_det_g[:-1, :-1, 2], -1, axis = 0) - B1u[p, :-1, 1:-1] * sqrt_det_g[:-1, :-1, 2]) / dxi   \
#                    + (N.roll(B2u[p, 1:-1, :-1] * sqrt_det_g[:-1, :-1, 1], -1, axis = 1) - B2u[p, 1:-1, :-1] * sqrt_det_g[:-1, :-1, 1]) / deta) / sqrt_det_g[:-1, :-1, 3]

divB = N.zeros((n_patches, Nxi_half, Neta_half))
dB1d1 = N.zeros((n_patches, Nxi_half, Neta_half))
dB2d2 = N.zeros((n_patches, Nxi_half, Neta_half))

def compute_divB(p):
    
    dB1d1[p, 0, :] = (- 0.5 * B1u[p, 0, :] * sqrt_det_g_B1[0, :] + 0.5 * B1u[p, 1, :] * sqrt_det_g_B1[1, :]) / dxi / P_half_2[0]
    dB1d1[p, 1, :] = (- 0.25 * B1u[p, 0, :] * sqrt_det_g_B1[0, :] + 0.25 * B1u[p, 1, :] * sqrt_det_g_B1[1, :]) / dxi / P_half_2[1]
    dB1d1[p, 2, :] = (- 0.25 * B1u[p, 0, :] * sqrt_det_g_B1[0, :] - 0.75 * B1u[p, 1, :] * sqrt_det_g_B1[1, :] + B1u[p, 2, :] * sqrt_det_g_B1[2, :]) / dxi / P_half_2[2]

    dB1d1[p, Nxi_half - 3, :] = (- B1u[p, -3, :] * sqrt_det_g_B1[-3, :] + 0.75 * B1u[p, -2, :] * sqrt_det_g_B1[-2, :] + 0.25 * B1u[p, -1, :] * sqrt_det_g_B1[-1, :]) / dxi / P_half_2[Nxi_half - 3]
    dB1d1[p, Nxi_half - 2, :] = (- 0.25 * B1u[p, -2, :] * sqrt_det_g_B1[-2, :] + 0.25 * B1u[p, -1, :] * sqrt_det_g_B1[-1, :]) / dxi / P_half_2[Nxi_half - 2]
    dB1d1[p, Nxi_half - 1, :] = (- 0.5 * B1u[p, -2, :] * sqrt_det_g_B1[-2, :] + 0.5 * B1u[p, -1, :] * sqrt_det_g_B1[-1, :]) / dxi / P_half_2[Nxi_half - 1]

    dB1d1[p, 3:(Nxi_half - 3), :] = (B1u[p, 3:(Nxi_half - 3), :] * sqrt_det_g_B1[3:(Nxi_half - 3), :] - N.roll(B1u[p, :, :] * sqrt_det_g_B1[:, :], 1, axis = 0)[3:(Nxi_half - 3), :]) / dxi

    dB2d2[p, :, 0] = (- 0.5 * B2u[p, :, 0] * sqrt_det_g_B2[:, 0] + 0.5 * B2u[p, :, 1] * sqrt_det_g_B2[:, 1]) / dxi / P_half_2[0]
    dB2d2[p, :, 1] = (- 0.25 * B2u[p, :, 0] * sqrt_det_g_B2[:, 0] + 0.25 * B2u[p, :, 1] * sqrt_det_g_B2[:, 1]) / dxi / P_half_2[1]
    dB2d2[p, :, 2] = (- 0.25 * B2u[p, :, 0] * sqrt_det_g_B2[:, 0] - 0.75 * B2u[p, :, 1] * sqrt_det_g_B2[:, 1] + B2u[p, :, 2] * sqrt_det_g_B2[:, 2]) / dxi / P_half_2[2]

    dB2d2[p, :, Neta_half - 3] = (- B2u[p, :, -3] * sqrt_det_g_B2[:, -3] + 0.75 * B2u[p, :, -2] * sqrt_det_g_B2[:, -2] + 0.25 * B2u[p, :, -1] * sqrt_det_g_B2[:, -1]) / deta / P_half_2[Nxi_half - 3]
    dB2d2[p, :, Neta_half - 2] = (- 0.25 * B2u[p, :, -2] * sqrt_det_g_B2[:, -2] + 0.25 * B2u[p, :, -1] * sqrt_det_g_B2[:, -1]) / deta / P_half_2[Nxi_half - 2]
    dB2d2[p, :, Neta_half - 1] = (- 0.5 * B2u[p, :, -2] * sqrt_det_g_B2[:, -2] + 0.5 * B2u[p, :, -1] * sqrt_det_g_B2[:, -1]) / deta / P_half_2[Nxi_half - 1]

    dB2d2[p, :, 3:(Neta_half - 3)] = (B2u[p, :, 3:(Neta_half - 3)] * sqrt_det_g_B2[:, 3:(Neta_half - 3)] - N.roll(B2u[p, :, :] * sqrt_det_g_B2[:, :], 1, axis = 1)[:, 3:(Neta_half - 3)]) / deta

    # print(dB2d2[p, :5, Neta_half - 1], dB1d1[p, :5, Neta_half - 1])
    divB[p, :, :] = (dB1d1[p, :, :] + dB2d2[p, :, :])

def correct_B(p):
    
    B1u[p, -1, :] = B1u[p, -2, :] * sqrt_det_g_B1[-2, :] / sqrt_det_g_B1[-1, :] - dxi * dB2d2[p, -1, :] / sqrt_det_g_B1[-1, :]
    B2u[p, :, -1] = B2u[p, :, -2] * sqrt_det_g_B2[:, -2] / sqrt_det_g_B2[:, -1] - deta* dB1d1[p, :, -1] / sqrt_det_g_B2[:, -1]


########
# Boundary conditions
########

def interp(arr_in, xA, xB):
    return N.interp(xB, xA, arr_in)

sqrt_gd_half = N.sqrt(g22d_half)

# Interface inner boundary 
sig_in  = 1.0
sig_cor = sig_in

def compute_delta_E(p0, p1, dtin, B1in, B2in, Erin):

    i0 = 0
    i1 = Nxi_int

    top = topology[p0, p1]
    
    if (top == 'xx'):

        Beta_0 = B2in[p0, -1, :]
        Beta_1 = B2in[p1, 0, :]
        Er_0 = Erin[p0, -1, :]
        Er_1 = Erin[p1, 0, :]

        diff_Er[p0, -1, i0:i1] += dtin * sig_in * 0.5 * ((  Beta_0[i0:i1] + sqrt_gd_half[i0:i1] * Er_0[i0:i1]) - \
                                                         (  Beta_1[i0:i1] + sqrt_gd_half[i0:i1] * Er_1[i0:i1])) / dxi / P_int_2[0]
        diff_Er[p1, 0, i0:i1]  += dtin * sig_in * 0.5 * ((- Beta_1[i0:i1] + sqrt_gd_half[i0:i1] * Er_1[i0:i1]) - \
                                                         (- Beta_0[i0:i1] + sqrt_gd_half[i0:i1] * Er_0[i0:i1])) / dxi / P_int_2[0]

    if (top == 'xy'):

        Beta_0 = B2in[p0, -1, :]
        Bxi_1  = B1in[p1, :, 0]
        Er_0 = Er[p0, -1, :]
        Er_1 = Er[p1, :, 0]
        
        diff_Er[p0, -1, i0:i1] += dtin * sig_in * 0.5 * ((Beta_0[i0:i1] + sqrt_gd_half[i0:i1] * Er_0[i0:i1]) - \
                                                         (- (Bxi_1[::-1])[i0:i1] + sqrt_gd_half[i0:i1] * (Er_1[::-1])[i0:i1])) / dxi / P_int_2[0]
        diff_Er[p1, i0:i1, 0]  += dtin * sig_in * 0.5 * ((Bxi_1[i0:i1] + sqrt_gd_half[i0:i1] * Er_1[i0:i1]) - \
                                                         (- (Beta_0[::-1])[i0:i1] + sqrt_gd_half[i0:i1] * (Er_0[::-1])[i0:i1])) / dxi / P_int_2[0]

    if (top == 'yy'):

        Bxi_0 = B1in[p0, :, -1]
        Bxi_1 = B1in[p1, :, 0]
        Er_0 = Erin[p0, :, -1]
        Er_1 = Erin[p1, :, 0]
        
        diff_Er[p0, i0:i1, -1] += dtin * sig_in * 0.5 * ((- Bxi_0[i0:i1] + sqrt_gd_half[i0:i1] * Er_0[i0:i1]) - \
                                                         (- Bxi_1[i0:i1] + sqrt_gd_half[i0:i1] * Er_1[i0:i1])) / deta / P_int_2[0]
        diff_Er[p1, i0:i1, 0]  += dtin * sig_in * 0.5 * ((Bxi_1[i0:i1] + sqrt_gd_half[i0:i1] * Er_1[i0:i1]) - \
                                                         (Bxi_0[i0:i1] + sqrt_gd_half[i0:i1] * Er_0[i0:i1])) / deta / P_int_2[0]

    if (top == 'yx'):

        Bxi_0 = B1in[p0, :, -1]
        Beta_1 = B2in[p1, 0, :]
        Er_0 = Erin[p0, :, -1]
        Er_1 = Erin[p1, 0, :]

        diff_Er[p0, i0:i1, -1] += dtin * sig_in * 0.5 * ((- Bxi_0[i0:i1] + sqrt_gd_half[i0:i1] * Er_0[i0:i1]) - \
                                                         ((Beta_1[::-1])[i0:i1] + sqrt_gd_half[i0:i1] * (Er_1[::-1])[i0:i1])) / deta / P_int_2[0]
        diff_Er[p1, 0, i0:i1]  += dtin * sig_in * 0.5 * ((- Beta_1[i0:i1] + sqrt_gd_half[i0:i1] * Er_1[i0:i1]) - \
                                                         ((Bxi_0[::-1])[i0:i1] + sqrt_gd_half[i0:i1] * (Er_0[::-1])[i0:i1])) / deta / P_int_2[0]

def interface_E(p0, p1):

    i0 = 1
    i1 = Nxi_int - 1

    top = topology[p0, p1]
    
    if (top == 'xx'):
        Er[p0, -1, i0:i1] -= diff_Er[p0, -1, i0:i1] / sqrt_det_g_half[i0:i1]
        Er[p1, 0, i0:i1]  -= diff_Er[p1, 0, i0:i1]  / sqrt_det_g_half[i0:i1]

    if (top == 'xy'):
        Er[p0, -1, i0:i1] -= diff_Er[p0, -1, i0:i1] / sqrt_det_g_half[i0:i1]
        Er[p1, i0:i1, 0]  -= diff_Er[p1, i0:i1, 0]  / sqrt_det_g_half[i0:i1]

    if (top == 'yy'):
        Er[p0, i0:i1, -1] -= diff_Er[p0, i0:i1, -1] / sqrt_det_g_half[i0:i1]
        Er[p1, i0:i1, 0]  -= diff_Er[p1, i0:i1, 0]  / sqrt_det_g_half[i0:i1]

    if (top == 'yx'):
        Er[p0, i0:i1, -1] -= diff_Er[p0, i0:i1, -1] / sqrt_det_g_half[i0:i1]
        Er[p1, 0, i0:i1]  -= diff_Er[p1, 0, i0:i1]  / sqrt_det_g_half[i0:i1]

def corners_E(p0):

    Er[p0, 0, 0]   -= diff_Er[p0, 0, 0] * sig_cor / sig_in / sqrt_det_g_half[0]
    Er[p0, -1, 0]  -= diff_Er[p0, -1, 0] * sig_cor / sig_in  / sqrt_det_g_half[0]
    Er[p0, 0, -1]  -= diff_Er[p0, 0, -1] * sig_cor / sig_in  / sqrt_det_g_half[0]
    Er[p0, -1, -1] -= diff_Er[p0, -1, -1] * sig_cor / sig_in  / sqrt_det_g_half[0]

def compute_delta_B(p0, p1, dtin, B1in, B2in, Erin):

    i0 = 0
    i1 = Nxi_int

    top = topology[p0, p1]
    
    if (top == 'xx'):

        Beta_0 = B2in[p0, -1, :]
        Beta_1 = B2in[p1, 0, :]
        Er_0 = Erin[p0, -1, :]
        Er_1 = Erin[p1, 0, :]
        
        diff_B2[p0, -1, i0:i1] += dtin * sig_in * 0.5 * ((Beta_0[i0:i1] / sqrt_gd_half[i0:i1] + Er_0[i0:i1]) - \
                                                         (Beta_1[i0:i1] / sqrt_gd_half[i0:i1] + Er_1[i0:i1])) / dxi / P_int_2[0]
        diff_B2[p1, 0, i0:i1]  += dtin * sig_in * 0.5 * ((Beta_1[i0:i1] / sqrt_gd_half[i0:i1] - Er_1[i0:i1]) - \
                                                         (Beta_0[i0:i1] / sqrt_gd_half[i0:i1] - Er_0[i0:i1])) / dxi / P_int_2[0]

    if (top == 'xy'):

        Beta_0 = B2in[p0, -1, :]
        Bxi_1  = B1in[p1, :, 0]
        Er_0 = Er[p0, -1, :]
        Er_1 = Er[p1, :, 0]

        diff_B2[p0, -1, i0:i1] += dtin * sig_in * 0.5 * ((Beta_0[i0:i1] / sqrt_gd_half[i0:i1] + Er_0[i0:i1]) - \
                                                         (-(Bxi_1[::-1])[i0:i1] / sqrt_gd_half[i0:i1] + (Er_1[::-1])[i0:i1])) / dxi / P_int_2[0]
        diff_B1[p1, i0:i1, 0]  += dtin * sig_in * 0.5 * ((Bxi_1[i0:i1] / sqrt_gd_half[i0:i1] + Er_1[i0:i1]) - \
                                                         (-(Beta_0[::-1])[i0:i1] / sqrt_gd_half[i0:i1] + (Er_0[::-1])[i0:i1])) / dxi / P_int_2[0]

    if (top == 'yy'):

        Bxi_0 = B1in[p0, :, -1]
        Bxi_1 = B1in[p1, :, 0]
        Er_0 = Erin[p0, :, -1]
        Er_1 = Erin[p1, :, 0]

        diff_B1[p0, i0:i1, -1] += dtin * sig_in * 0.5 * ((Bxi_0[i0:i1] / sqrt_gd_half[i0:i1] - Er_0[i0:i1]) - \
                                                         (Bxi_1[i0:i1] / sqrt_gd_half[i0:i1] - Er_1[i0:i1])) / deta / P_int_2[0]
        diff_B1[p1, i0:i1, 0]  += dtin * sig_in * 0.5 * ((Bxi_1[i0:i1] / sqrt_gd_half[i0:i1] + Er_1[i0:i1]) - \
                                                         (Bxi_0[i0:i1] / sqrt_gd_half[i0:i1] + Er_0[i0:i1])) / deta / P_int_2[0]

    if (top == 'yx'):

        Bxi_0 = B1in[p0, :, -1]
        Beta_1 = B2in[p1, 0, :]
        Er_0 = Erin[p0, :, -1]
        Er_1 = Erin[p1, 0, :]

        diff_B1[p0, i0:i1, -1] += dtin * sig_in * 0.5 * ((Bxi_0[i0:i1] / sqrt_gd_half[i0:i1] - Er_0[i0:i1]) - \
                                                         (- (Beta_1[::-1])[i0:i1] / sqrt_gd_half[i0:i1] - (Er_1[::-1])[i0:i1])) / deta / P_int_2[0]
        diff_B2[p1, 0, i0:i1]  += dtin * sig_in * 0.5 * ((Beta_1[i0:i1] / sqrt_gd_half[i0:i1] - Er_1[i0:i1]) - \
                                                         (- (Bxi_0[::-1])[i0:i1] / sqrt_gd_half[i0:i1] - (Er_0[::-1])[i0:i1])) / deta / P_int_2[0]

def interface_B(p0, p1):

    i0 =  1
    i1 = Nxi_int - 1

    top = topology[p0, p1]
    
    if (top == 'xx'):
        B2u[p0, -1, i0:i1] -= diff_B2[p0, -1, i0:i1] / sqrt_det_g_half[i0:i1]
        B2u[p1, 0, i0:i1]  -= diff_B2[p1, 0, i0:i1] / sqrt_det_g_half[i0:i1]

    if (top == 'xy'):
        B2u[p0, -1, i0:i1] -= diff_B2[p0, -1, i0:i1] / sqrt_det_g_half[i0:i1]
        B1u[p1, i0:i1, 0]  -= diff_B1[p1, i0:i1, 0] / sqrt_det_g_half[i0:i1] 

    if (top == 'yy'):
        B1u[p0, i0:i1, -1] -= diff_B1[p0, i0:i1, -1] / sqrt_det_g_half[i0:i1]
        B1u[p1, i0:i1, 0]  -= diff_B1[p1, i0:i1, 0] / sqrt_det_g_half[i0:i1] 

    if (top == 'yx'):
        B1u[p0, i0:i1, -1] -= diff_B1[p0, i0:i1, -1] / sqrt_det_g_half[i0:i1]
        B2u[p1, 0, i0:i1]  -= diff_B2[p1, 0, i0:i1] / sqrt_det_g_half[i0:i1]

def corners_B(p0):

    B1u[p0, 0, 0]   -= diff_B1[p0, 0, 0] * sig_cor / sig_in / sqrt_det_g_half[0]
    B1u[p0, -1, 0]  -= diff_B1[p0, -1, 0] * sig_cor / sig_in / sqrt_det_g_half[0]
    B1u[p0, 0, -1]  -= diff_B1[p0, 0, -1] * sig_cor / sig_in / sqrt_det_g_half[0]
    B1u[p0, -1, -1] -= diff_B1[p0, -1, -1] * sig_cor / sig_in / sqrt_det_g_half[0]

    B2u[p0, 0, 0]   -= diff_B2[p0, 0, 0] * sig_cor / sig_in / sqrt_det_g_half[0]
    B2u[p0, -1, 0]  -= diff_B2[p0, -1, 0] * sig_cor / sig_in / sqrt_det_g_half[0]
    B2u[p0, 0, -1]  -= diff_B2[p0, 0, -1] * sig_cor / sig_in / sqrt_det_g_half[0]
    B2u[p0, -1, -1] -= diff_B2[p0, -1, -1] * sig_cor / sig_in / sqrt_det_g_half[0]


########
# Triple points
########

def plus(index):
    if (index == 0):
        return index  + 1
    elif (index == -1):
        return index - 1
    else:
        return

sig_fil = 1.0 / dt # 0.25 / dt # 0.1 / dt

def filter_B_corner(n_corner, dtin):

    p0, p1, p2 = triplet[n_corner, 0], triplet[n_corner, 1], triplet[n_corner, 2]
    i0, j0 = index_corners[n_corner, 0][0], index_corners[n_corner, 0][1]
    i1, j1 = index_corners[n_corner, 1][0], index_corners[n_corner, 1][1]
    i2, j2 = index_corners[n_corner, 2][0], index_corners[n_corner, 2][1]

    i0p, j0p = plus(i0), plus(j0)
    i1p, j1p = plus(i1), plus(j1)
    i2p, j2p = plus(i2), plus(j2)

    Er0 = Er[p0, i0, j0]
    Er1 = Er[p1, i1, j1]
    Er2 = Er[p2, i2, j2]
    mean0 = (Er0 + Er1 + Er2) / 3.0

    Era = Er[p0, i0p, j0]
    Erb = Er[p0, i0, j0p]
    Erc = Er[p0, i0p, j0p]
    Erd = Er[p1, i1p, j1]
    Ere = Er[p1, i1, j1p]
    Erf = Er[p1, i1p, j1p]
    Erg = Er[p2, i2p, j2]
    Erh = Er[p2, i2, j2p]
    Eri = Er[p2, i2p, j2p]
    # mean = (0.5 * Era + 0.5 * Erb + Erc + 0.5 * Erd + 0.5 * Ere + Erf + 0.5 * Erg + 0.5 * Erh + Eri) / 6.0
    mean = (Erc + Erf + Eri) / 3.0
    # mean = (Era + Erb + Erd + Ere + Erg + Erh) / 6.0
    # mean = (Era + Erb + Erd + Ere + Erg + Erh + Er0 + Er1 + Er2) / 9.0

    # Er[p0, i0, j0] -= dtin * sig_fil * (Er[p0, i0, j0] - mean0)
    # Er[p1, i1, j1] -= dtin * sig_fil * (Er[p1, i1, j1] - mean0)
    # Er[p2, i2, j2] -= dtin * sig_fil * (Er[p2, i2, j2] - mean0)
    
    Er[p0, i0, j0] -= dtin * sig_fil * (Er[p0, i0, j0] - mean)
    Er[p1, i1, j1] -= dtin * sig_fil * (Er[p1, i1, j1] - mean)
    Er[p2, i2, j2] -= dtin * sig_fil * (Er[p2, i2, j2] - mean)

def filter_E_corner(n_corner, dtin):

    p0, p1, p2 = triplet[n_corner, 0], triplet[n_corner, 1], triplet[n_corner, 2]
    i0, j0 = index_corners[n_corner, 0][0], index_corners[n_corner, 0][1]
    i1, j1 = index_corners[n_corner, 1][0], index_corners[n_corner, 1][1]
    i2, j2 = index_corners[n_corner, 2][0], index_corners[n_corner, 2][1]

    fcoord0 = (globals()["coord_" + sphere[p0] + "_to_sph"])
    th0, ph0 = fcoord0(xi_int[i0], eta_int[j0])

    i0p, j0p = plus(i0), plus(j0)
    i1p, j1p = plus(i1), plus(j1)
    i2p, j2p = plus(i2), plus(j2)

    fvec0 = (globals()["vec_" + sphere[p0] + "_to_sph"])
    fvec1 = (globals()["vec_" + sphere[p1] + "_to_sph"])
    fvec2 = (globals()["vec_" + sphere[p2] + "_to_sph"])
    fvec0i = (globals()["vec_sph_to_" + sphere[p0]])
    fvec1i = (globals()["vec_sph_to_" + sphere[p1]])
    fvec2i = (globals()["vec_sph_to_" + sphere[p2]])

    B1m = 0.5 * (B1u[p0, i0, j0] + B1u[p0, i0, j0p])    
    B2m = 0.5 * (B2u[p0, i0, j0] + B2u[p0, i0p, j0])
    Etha, Epha = fvec0(xi_half[i0p], eta_int[j0], B1u[p0, i0p, j0], B2m)
    Ethb, Ephb = fvec0(xi_int[i0], eta_half[j0p], B1m, B2u[p0, i0, j0p])
    B1m = 0.5 * (B1u[p1, i1, j1] + B1u[p1, i1, j1p])    
    B2m = 0.5 * (B2u[p1, i1, j1] + B2u[p1, i1p, j1])
    Ethc, Ephc = fvec1(xi_half[i1p], eta_int[j1], B1u[p1, i1p, j1], B2m)
    Ethd, Ephd = fvec1(xi_int[i1], eta_half[j1p], B1m, B2u[p1, i1, j1p])
    B1m = 0.5 * (B1u[p2, i2, j2] + B1u[p2, i2, j2p])    
    B2m = 0.5 * (B2u[p2, i2, j2] + B2u[p2, i2p, j2])
    Ethe, Ephe = fvec2(xi_half[i2p], eta_int[j2], B1u[p2, i2p, j2], B2m)
    Ethf, Ephf = fvec2(xi_int[i2], eta_half[j2p], B1m, B2u[p2, i2, j2p])

    Ethm = (Etha + Ethb + Ethc + Ethd + Ethe + Ethf) / 6.0
    Ephm = (Epha + Ephb + Ephc + Ephd + Ephe + Ephf) / 6.0

    Eth0, Eph0 = fvec0(xi_int[i0], eta_int[j0], B1u[p0, i0, j0], B2u[p0, i0, j0])
    Eth1, Eph1 = fvec1(xi_int[i1], eta_int[j1], B1u[p1, i1, j1], B2u[p1, i1, j1])
    Eth2, Eph2 = fvec2(xi_int[i2], eta_int[j2], B1u[p2, i2, j2], B2u[p2, i2, j2])
    Ethmm = (Eth0 + Eth1 + Eth2) / 3.0
    Ephmm = (Eph0 + Eph1 + Eph2) / 3.0
    B10, B20 = fvec0i(th0, ph0, Ethmm, Ephmm)
    B11, B21 = fvec1i(th0, ph0, Ethmm, Ephmm)
    B12, B22 = fvec2i(th0, ph0, Ethmm, Ephmm)

    # Ethm = (Etha + Ethb + Ethc + Ethd + Ethe + Ethf + Eth0 + Eth1 + Eth2) / 9.0
    # Ephm = (Epha + Ephb + Ephc + Ephd + Ephe + Ephf + Eph0 + Eph1 + Eph2) / 9.0
    
    Exi0, Eeta0 = fvec0i(th0, ph0, Ethm, Ephm)
    Exi1, Eeta1 = fvec1i(th0, ph0, Ethm, Ephm)
    Exi2, Eeta2 = fvec2i(th0, ph0, Ethm, Ephm)
    
    # Exi0, Eeta0 = fvec0i(th0, ph0, Ethmm, Ephmm)
    # Exi1, Eeta1 = fvec1i(th0, ph0, Ethmm, Ephmm)
    # Exi2, Eeta2 = fvec2i(th0, ph0, Ethmm, Ephmm)

    B1u[p0, i0, j0] -= sig_fil * dtin * (B1u[p0, i0, j0] - Exi0)
    B2u[p0, i0, j0] -= sig_fil * dtin * (B2u[p0, i0, j0] - Eeta0)
    B1u[p1, i1, j1] -= sig_fil * dtin * (B1u[p1, i1, j1] - Exi1)
    B2u[p1, i1, j1] -= sig_fil * dtin * (B2u[p1, i1, j1] - Eeta1)
    B1u[p2, i2, j2] -= sig_fil * dtin * (B1u[p2, i2, j2] - Exi2)
    B2u[p2, i2, j2] -= sig_fil * dtin * (B2u[p2, i2, j2] - Eeta2)


########
# Initialization
########

amp = 1.0
n_mode = 2
wave = 2.0 * (xi_max - xi_min) / n_mode
B1ui = N.zeros_like(B1u)
B2ui = N.zeros_like(B2u)

for patch in range(6):

    fvec = (globals()["vec_sph_to_" + sphere[patch]])
    fveci = (globals()["vec_"+sphere[patch]+"_to_sph"])
    fcoord = (globals()["coord_" + sphere[patch] + "_to_sph"])

    r0 = 1.0

    for i in range(Nxi_int):
        for j in range(Neta_int):

            th0, ph0 = fcoord(xEr_grid[i, j], yEr_grid[i, j])
            Er[patch, i, j] = amp * N.sin(th0)**3 * N.cos(3.0 * ph0)

    for i in range(Nxi_int):
        for j in range(Neta_half):

            th0, ph0 = fcoord(xB1_grid[i, j], yB1_grid[i, j])

            EtTMP = 0.0 # amp * N.sin(th0)**3 * N.cos(3.0 * ph0)
            EpTMP = 0.0

            ECStmp = fvec(th0, ph0, EtTMP, EpTMP)

            B1u[patch, i, j] = ECStmp[0]
            B1ui[patch, i, j] = ECStmp[0]

    for i in range(Nxi_half):
        for j in range(Neta_int):

            th0, ph0 = fcoord(xB2_grid[i, j], yB2_grid[i, j])

            EtTMP = 0.0 # amp * N.sin(th0)**3 * N.cos(3.0 * ph0)
            EpTMP = 0.0

            ECStmp = fvec(th0, ph0, EtTMP, EpTMP)

            B2u[patch, i, j] = ECStmp[1]
            B2ui[patch, i, j] = ECStmp[1]

########
# Visualization
########

ratio = 0.5

xi_grid_c, eta_grid_c = unflip_eq(xEr_grid, yEr_grid)
xi_grid_d, eta_grid_d = unflip_eq(xEr_grid, yEr_grid)
xi_grid_n, eta_grid_n = unflip_po(xEr_grid, yEr_grid)

xi_grid_c2, eta_grid_c2 = unflip_eq(xB1_grid, yB1_grid)
xi_grid_d2, eta_grid_d2 = unflip_eq(xB1_grid, yB1_grid)
xi_grid_n2, eta_grid_n2 = unflip_po(xB1_grid, yB1_grid)

def plot_fields_unfolded_Er(it, vm):

    fig = P.figure(1, facecolor='w')
    ax = P.subplot(111)

    ax.pcolormesh(xEr_grid, yEr_grid, Er[Sphere.A, :, :], cmap = "RdBu_r", vmin = - vm, vmax = vm)
    ax.pcolormesh(xEr_grid + N.pi / 2.0, yEr_grid, Er[Sphere.B, :, :], cmap = "RdBu_r", vmin = - vm, vmax = vm)
    ax.pcolormesh(xEr_grid, yEr_grid - N.pi / 2.0, Er[Sphere.S, :, :], cmap = "RdBu_r", vmin = - vm, vmax = vm)

    ax.pcolormesh(xi_grid_c + N.pi, eta_grid_c, Er[Sphere.C, :, :], cmap = "RdBu_r", vmin = - vm, vmax = vm)
    ax.pcolormesh(xi_grid_d - N.pi / 2.0, eta_grid_d, Er[Sphere.D, :, :], cmap = "RdBu_r", vmin = - vm, vmax = vm)
    ax.pcolormesh(xi_grid_n, eta_grid_n + N.pi / 2.0, Er[Sphere.N, :, :], cmap = "RdBu_r", vmin = - vm, vmax = vm)
    
    P.title(r'$t={:.3f} R/c$'.format(FDUMP*it*dt))
    
    figsave_png(fig, "../snapshots_penalty/Er_" + str(it))

    P.close('all')

def plot_fields_unfolded_B1(it, vm):

    fig = P.figure(1, facecolor='w')
    ax = P.subplot(111)

    ax.pcolormesh(xB1_grid, yB1_grid, B1u[Sphere.A, :, :], cmap = "RdBu_r", vmin = - vm, vmax = vm)
    ax.pcolormesh(xB1_grid + N.pi / 2.0, yB1_grid, B1u[Sphere.B, :, :], cmap = "RdBu_r", vmin = - vm, vmax = vm)
    ax.pcolormesh(xB1_grid, yB1_grid - N.pi / 2.0, B1u[Sphere.S, :, :], cmap = "RdBu_r", vmin = - vm, vmax = vm)

    ax.pcolormesh(xi_grid_c2 + N.pi, eta_grid_c2, B1u[Sphere.C, :, :], cmap = "RdBu_r", vmin = - vm, vmax = vm)
    ax.pcolormesh(xi_grid_d2 - N.pi / 2.0, eta_grid_d2, B1u[Sphere.D, :, :], cmap = "RdBu_r", vmin = - vm, vmax = vm)
    ax.pcolormesh(xi_grid_n2, eta_grid_n2 + N.pi / 2.0, B1u[Sphere.N, :, :], cmap = "RdBu_r", vmin = - vm, vmax = vm)
    
    figsave_png(fig, "../snapshots_penalty/B1u_" + str(it))

    P.close('all')

def plot_fields_unfolded_divB(it, vm):

    fig = P.figure(1, facecolor='w')
    ax = P.subplot(111)

    # ax.pcolormesh(xi_half[1:-1], xi_half[1:-1], divB[Sphere.A, :, :], cmap = "RdBu_r", vmin = - vm, vmax = vm)
    # ax.pcolormesh(xi_half[1:-1] + N.pi / 2.0 + 0.1, xi_half[1:-1], divB[Sphere.B, :, :], cmap = "RdBu_r", vmin = - vm, vmax = vm)
    # ax.pcolormesh(xi_half[1:-1], xi_half[1:-1] - N.pi / 2.0 - 0.1, divB[Sphere.S, :, :], cmap = "RdBu_r", vmin = - vm, vmax = vm)

    # ax.pcolormesh(xi_half[1:-1] + N.pi + 0.2, xi_half[1:-1], divB[Sphere.C, :, :], cmap = "RdBu_r", vmin = - vm, vmax = vm)
    # ax.pcolormesh(xi_half[1:-1] - N.pi / 2.0 - 0.1, xi_half[1:-1], divB[Sphere.D, :, :], cmap = "RdBu_r", vmin = - vm, vmax = vm)
    # ax.pcolormesh(xi_half[1:-1], xi_half[1:-1] + N.pi / 2.0 + 0.1, divB[Sphere.N, :, :], cmap = "RdBu_r", vmin = - vm, vmax = vm)

    ax.pcolormesh(xi_half, xi_half, divB[Sphere.A, :, :], cmap = "RdBu_r", vmin = - vm, vmax = vm)
    ax.pcolormesh(xi_half + N.pi / 2.0 + 0.1, xi_half, divB[Sphere.B, :, :], cmap = "RdBu_r", vmin = - vm, vmax = vm)
    ax.pcolormesh(xi_half, xi_half - N.pi / 2.0 - 0.1, divB[Sphere.S, :, :], cmap = "RdBu_r", vmin = - vm, vmax = vm)

    ax.pcolormesh(xi_half + N.pi + 0.2, xi_half, divB[Sphere.C, :, :], cmap = "RdBu_r", vmin = - vm, vmax = vm)
    ax.pcolormesh(xi_half - N.pi / 2.0 - 0.1, xi_half, divB[Sphere.D, :, :], cmap = "RdBu_r", vmin = - vm, vmax = vm)
    ax.pcolormesh(xi_half, xi_half + N.pi / 2.0 + 0.1, divB[Sphere.N, :, :], cmap = "RdBu_r", vmin = - vm, vmax = vm)
    
    figsave_png(fig, "../snapshots_penalty/divB_" + str(it))

    P.close('all')

########
# Main routine
########

idump = 0

patch = range(n_patches)

Nt = 500000 # Number of iterations
FDUMP = 2 # Dump frequency
time = dt * N.arange(Nt)
energy = N.zeros((n_patches, Nt))

# Initialize half time step

compute_diff_E(patch)
push_B(patch, 0, 0.5 * dt)
for i in range(n_zeros):
    p0, p1 = index_row[i], index_col[i]
    compute_delta_B(p0, p1, dt, B1d, B2d, Er)
for i in range(n_zeros):
    p0, p1 = index_row[i], index_col[i]
    interface_B(p0, p1)
corners_B(patch)

for it in tqdm(range(Nt), "Progression"):
    if ((it % FDUMP) == 0):

        plot_fields_unfolded_Er(idump, 1.0)
        plot_fields_unfolded_divB(idump, 0.01)
        idump += 1

    diff_Er[:, :, :] = 0.0
    diff_B1[:, :, :] = 0.0
    diff_B2[:, :, :] = 0.0

    Er0[:, :, :] = Er[:, :, :]   

    compute_diff_B(patch)
    # compute_diff_B_alt(patch)
    # compute_diff_B_order(patch)

    push_E(patch, it, dt)
    
    Er0[:, :, :] = 0.5 * (Er[:, :, :] + Er0[:, :, :])

    for i in range(n_zeros):
        p0, p1 = index_row[i], index_col[i]
        compute_delta_E(p0, p1, dt, B1d, B2d, Er0)
        # compute_delta_E(p0, p1, dt, B1d, B2d, Er)

        interface_E(p0, p1)

    corners_E(patch)

    compute_diff_E(patch)
    # compute_diff_E_alt(patch)
    # compute_diff_E_order(patch)

    B1d0[:, :, :] = B1d[:, :, :]
    B2d0[:, :, :] = B2d[:, :, :]
    
    push_B(patch, it, dt)

    B1d0[:, :, :] = 0.5 * (B1d[:, :, :] + B1d0[:, :, :])
    B2d0[:, :, :] = 0.5 * (B2d[:, :, :] + B2d0[:, :, :])
    
    for i in range(n_zeros):
        p0, p1 = index_row[i], index_col[i]
        compute_delta_B(p0, p1, dt, B1d0, B2d0, Er)
        # compute_delta_B(p0, p1, dt, B1d, B2d, Er)

        interface_B(p0, p1)

    corners_B(patch)
    
    for p0 in patch:
        compute_divB(p0)
        
    contra_to_cov_B(patch)
    # contra_to_cov_E_weights(patch)

    # for p in range(n_patches):
    #     # energy[p, it] = dxi * deta * (N.sum(Er[p, :, :]**2) \
    #     # + N.sum(B1u[p, :, :]**2) + N.sum(B2u[p, :, :]**2))
    #     energy[:, it] = compute_energy()
        
# for p in range(n_patches):
#     P.plot(time, energy[p, :])


