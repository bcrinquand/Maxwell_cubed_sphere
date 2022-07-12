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
from figure_module import *

import warnings
import matplotlib.cbook

import warnings
import matplotlib.cbook
warnings.filterwarnings("ignore",category=matplotlib.cbook.mplDeprecation)

n_patches = 3
# Connect patch indices and names
sphere = {0: "A", 1: "B", 2: "C", 3: "D", 4: "N", 5: "S"}

class Sphere:
    A = 0
    B = 1
    C = 2

########    
# Topology of the patches
########

# topology = N.array([
#     [   0, 'xx',    0,    0, 'yx',    0],
#     [   0,    0, 'xy',    0, 'yy',    0],
#     [   0,    0,    0, 'yy',    0, 'xy'],
#     ['yx',    0,    0,    0,    0, 'xx'],
#     [   0,    0, 'xx', 'yx',    0,    0],
#     ['yy', 'xy',    0,    0,    0,    0]
#                     ])

topology = N.zeros((6, 6), dtype = object)
topology[Sphere.A, Sphere.B] = 'xx'
topology[Sphere.A, Sphere.N] = 'yx'
topology[Sphere.B, Sphere.N] = 'yy'

# Gets indices where topology is nonzero
index_row, index_col = N.nonzero(topology)[0], N.nonzero(topology)[1]
n_zeros = N.size(index_row) # Total number of interactions (12)

# Corners triplets
n_corners = 1
triplet = N.zeros((n_corners,3), dtype='int')
triplet[0, :] = Sphere.A, Sphere.B, Sphere.N

# Indices of corners
index_corners = N.zeros((n_corners, 3), dtype='object')
index_corners[0,0] = (-1, -1)
index_corners[0,1] = (0, -1)
index_corners[0,2] = (0, 0)

# Parameters
cfl = 0.7
Nxi  = 128 # Number of cells in xi
Neta = 128 # Number of cells in eta

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

yBr_grid, xBr_grid = N.meshgrid(eta_half, xi_half)
yE1_grid, xE1_grid = N.meshgrid(eta_int, xi_half)
yE2_grid, xE2_grid = N.meshgrid(eta_half, xi_int)

########
# Define metric tensor
########

g11d = N.empty((Nxi_int, Neta_int, 4))
g12d = N.empty((Nxi_int, Neta_int, 4))
g22d = N.empty((Nxi_int, Neta_int, 4))
g11u = N.empty((Nxi_int, Neta_int, 4))
g12u = N.empty((Nxi_int, Neta_int, 4))
g22u = N.empty((Nxi_int, Neta_int, 4))

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
    for j in range(Neta_int):
        for i0 in range(4):
            
            metric = N.array([[g11d[i, j, i0], g12d[i, j, i0]], [g12d[i, j, i0], g22d[i, j, i0]]])
            inv_metric = N.linalg.inv(metric)
            g11u[i, j, i0] = inv_metric[0, 0]
            g12u[i, j, i0] = inv_metric[0, 1]
            g22u[i, j, i0] = inv_metric[1, 1]

sqrt_det_g = N.sqrt(g11d * g22d - g12d * g12d)

dt = cfl * N.min(1.0 / N.sqrt(g11d / (sqrt_det_g * sqrt_det_g) / (dxi * dxi) + g22d / (sqrt_det_g * sqrt_det_g) / (deta * deta) ))
print("delta t = {}".format(dt))

# Define fields
Br = N.zeros((n_patches, Nxi_half, Neta_half))
E1u = N.zeros((n_patches, Nxi_half, Neta_int))
E2u = N.zeros((n_patches, Nxi_int,  Neta_half))
E1d = N.zeros((n_patches, Nxi_half, Neta_int))
E2d = N.zeros((n_patches, Nxi_int,  Neta_half))
dBrd1 = N.zeros((n_patches, Nxi_int,  Neta_half))
dBrd2 = N.zeros((n_patches, Nxi_half, Neta_int))
dE1d2 = N.zeros((n_patches, Nxi_half, Neta_half))
dE2d1 = N.zeros((n_patches, Nxi_half, Neta_half))

diff_Br = N.zeros((n_patches, Nxi_half, Neta_half))
diff_E1 = N.zeros((n_patches, Nxi_half, Neta_int))
diff_E2 = N.zeros((n_patches, Nxi_int, Neta_half))

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

def compute_diff_B(p):
    
    dBrd1[p, 0, :] = (- 0.5 * Br[p, 0, :] + 0.25 * Br[p, 1, :] + 0.25 * Br[p, 2, :]) / dxi / P_int_2[0]
    dBrd1[p, 1, :] = (- 0.5 * Br[p, 0, :] - 0.25 * Br[p, 1, :] + 0.75 * Br[p, 2, :]) / dxi / P_int_2[1]
    
    dBrd1[p, Nxi_int - 2, :] = (- 0.75 * Br[p, -3, :] + 0.25 * Br[p, -2, :] + 0.5 * Br[p, -1, :]) / dxi / P_int_2[Nxi_int - 2]
    dBrd1[p, Nxi_int - 1, :] = (- 0.25 * Br[p, -3, :] - 0.25 * Br[p, -2, :] + 0.5 * Br[p, -1, :]) / dxi / P_int_2[Nxi_int - 1]

    dBrd1[p, 2:(Nxi_int - 2), :] = (N.roll(Br, -1, axis = 1)[p, 2:(Nxi_int - 2), :] - Br[p, 2:(Nxi_int - 2), :]) / dxi

    dBrd2[p, :, 0] = (- 0.5 * Br[p, :, 0] + 0.25 * Br[p, :, 1] + 0.25 * Br[p, :, 2]) / deta / P_int_2[0]
    dBrd2[p, :, 1] = (- 0.5 * Br[p, :, 0] - 0.25 * Br[p, :, 1] + 0.75 * Br[p, :, 2]) / deta / P_int_2[1]

    dBrd2[p, :, Nxi_int - 2] = (- 0.75 * Br[p, :, -3] + 0.25 * Br[p, :, -2] + 0.5 * Br[p, :, -1]) / deta / P_int_2[Nxi_int - 2]
    dBrd2[p, :, Nxi_int - 1] = (- 0.25 * Br[p, :, -3] - 0.25 * Br[p, :, -2] + 0.5 * Br[p, :, -1]) / deta / P_int_2[Nxi_int - 1]
        
    dBrd2[p, :, 2:(Neta_int - 2)] = (N.roll(Br, -1, axis = 2)[p, :, 2:(Neta_int - 2)] - Br[p, :, 2:(Neta_int - 2)]) / deta

def compute_diff_B_alt(p):
    
    dBrd1[p, 0, :] = (- 8.0 * Br[p, 0, :] + 9.0 * Br[p, 1, :] - 1.0 * Br[p, 2, :]) / dxi / 3.0
    
    dBrd1[p, Nxi_int - 1, :] = (1.0 * Br[p, -3, :] - 9.0 * Br[p, -2, :] + 8.0 * Br[p, -1, :]) / dxi / 3.0

    dBrd1[p, 1:(Nxi_int - 1), :] = (N.roll(Br, -1, axis = 1)[p, 1:(Nxi_int - 1), :] - Br[p, 1:(Nxi_int - 1), :]) / dxi

    dBrd2[p, :, 0] = (- 8.0 * Br[p, :, 0] + 9.0 * Br[p, :, 1] -1.0 * Br[p, :, 2]) / deta / 3.0

    dBrd2[p, :, Nxi_int - 1] = (1.0 * Br[p, :, -3] - 9.0 * Br[p, :, -2] + 8.0 * Br[p, :, -1]) / deta / 3.0
        
    dBrd2[p, :, 1:(Neta_int - 1)] = (N.roll(Br, -1, axis = 2)[p, :, 1:(Neta_int - 1)] - Br[p, :, 1:(Neta_int - 1)]) / deta

def compute_diff_E(p):

    dE2d1[p, 0, :] = (- 0.5 * E2d[p, 0, :] + 0.5 * E2d[p, 1, :]) / dxi / P_half_2[0]
    dE2d1[p, 1, :] = (- 0.25 * E2d[p, 0, :] + 0.25 * E2d[p, 1, :]) / dxi / P_half_2[1]
    dE2d1[p, 2, :] = (- 0.25 * E2d[p, 0, :] - 0.75 * E2d[p, 1, :] + E2d[p, 2, :]) / dxi / P_half_2[2]

    dE2d1[p, Nxi_half - 3, :] = (- E2d[p, -3, :] + 0.75 * E2d[p, -2, :] + 0.25 * E2d[p, -1, :]) / dxi / P_half_2[Nxi_half - 3]
    dE2d1[p, Nxi_half - 2, :] = (- 0.25 * E2d[p, -2, :] + 0.25 * E2d[p, -1, :]) / dxi / P_half_2[Nxi_half - 2]
    dE2d1[p, Nxi_half - 1, :] = (- 0.5 * E2d[p, -2, :] + 0.5 * E2d[p, -1, :]) / dxi / P_half_2[Nxi_half - 1]

    dE2d1[p, 3:(Nxi_half - 3), :] = (E2d[p, 3:(Nxi_half - 3), :] - N.roll(E2d, 1, axis = 1)[p, 3:(Nxi_half - 3), :]) / dxi

    dE1d2[p, :, 0] = (- 0.5 * E1d[p, :, 0] + 0.5 * E1d[p, :, 1]) / dxi / P_half_2[0]
    dE1d2[p, :, 1] = (- 0.25 * E1d[p, :, 0] + 0.25 * E1d[p, :, 1]) / dxi / P_half_2[1]
    dE1d2[p, :, 2] = (- 0.25 * E1d[p, :, 0] - 0.75 * E1d[p, :, 1] + E1d[p, :, 2]) / dxi / P_half_2[2]

    dE1d2[p, :, Neta_half - 3] = (- E1d[p, :, -3] + 0.75 * E1d[p, :, -2] + 0.25 * E1d[p, :, -1]) / deta / P_half_2[Nxi_half - 3]
    dE1d2[p, :, Neta_half - 2] = (- 0.25 * E1d[p, :, -2] + 0.25 * E1d[p, :, -1]) / deta / P_half_2[Nxi_half - 2]
    dE1d2[p, :, Neta_half - 1] = (- 0.5 * E1d[p, :, -2] + 0.5 * E1d[p, :, -1]) / deta / P_half_2[Nxi_half - 1]

    dE1d2[p, :, 3:(Neta_half - 3)] = (E1d[p, :, 3:(Neta_half - 3)] - N.roll(E1d, 1, axis = 2)[p, :, 3:(Neta_half - 3)]) / deta

def compute_diff_E_alt(p):

    # dE2d1[p, 0, :] = (- 3.0 * E2d[p, 0, :] + 4.0 * E2d[p, 1, :] - 1.0 * E2d[p, 2, :]) / dxi / 2.0
    dE2d1[p, 0, :] = (- 0.5 * E2d[p, 0, :] + 0.5 * E2d[p, 1, :]) / dxi / P_half_2[0]
    dE2d1[p, 1, :] = (- 0.25 * E2d[p, 0, :] + 0.25 * E2d[p, 1, :]) / dxi / P_half_2[1]

    dE2d1[p, Nxi_half - 2, :] = (- 0.25 * E2d[p, -2, :] + 0.25 * E2d[p, -1, :]) / dxi / P_half_2[Nxi_half - 2]
    dE2d1[p, Nxi_half - 1, :] = (- 0.5 * E2d[p, -2, :] + 0.5 * E2d[p, -1, :]) / dxi / P_half_2[Nxi_half - 1]
    # dE2d1[p, Nxi_half - 1, :] = (1.0 * E2d[p, -3, :] - 4.0 * E2d[p, -2, :] + 3.0 * E2d[p, -1, :]) / dxi / 2.0

    dE2d1[p, 2:(Nxi_half - 2), :] = (E2d[p, 2:(Nxi_half - 2), :] - N.roll(E2d, 1, axis = 1)[p, 2:(Nxi_half - 2), :]) / dxi

    # dE1d2[p, :, 0] = (- 3.0 * E1d[p, :, 0] + 4.0 * E1d[p, :, 1] - 1.0 * E1d[p, :, 2]) / dxi / 2.0
    dE1d2[p, :, 0] = (- 0.5 * E1d[p, :, 0] + 0.5 * E1d[p, :, 1]) / dxi / P_half_2[0]
    dE1d2[p, :, 1] = (- 0.25 * E1d[p, :, 0] + 0.25 * E1d[p, :, 1]) / dxi / P_half_2[1]

    dE1d2[p, :, Neta_half - 2] = (- 0.25 * E1d[p, :, -2] + 0.25 * E1d[p, :, -1]) / deta / P_half_2[Nxi_half - 2]
    dE1d2[p, :, Neta_half - 1] = (- 0.5 * E1d[p, :, -2] + 0.5 * E1d[p, :, -1]) / deta / P_half_2[Nxi_half - 1]
    # dE1d2[p, :, Nxi_half - 1] = (1.0 * E1d[p, :, -3] - 4.0 * E1d[p, :, -2] + 3.0 * E1d[p, :, -1]) / dxi / 2.0

    dE1d2[p, :, 2:(Neta_half - 2)] = (E1d[p, :, 2:(Neta_half - 2)] - N.roll(E1d, 1, axis = 2)[p, :, 2:(Neta_half - 2)]) / deta


Jz = N.zeros_like(Br)
Jz[Sphere.B, :, :] = 0.0 * N.exp(- (xBr_grid**2 + yBr_grid**2) / 0.1**2)

def contra_to_cov_E(p):

    ##### Exi

    # Interior
    E1d[p, 1:-1, 1:-1] = g11d[0:-1, 1:-1, 1] * E1u[p, 1:-1, 1:-1] \
                    + 0.25 * g12d[0:-1, 1:-1, 1] * (E2u[p, 1:, 1:-2] + N.roll(N.roll(E2u, 1, axis = 1), -1, axis = 2)[p, 1:, 1:-2] \
                    + N.roll(E2u, 1, axis = 1)[p, 1:, 1:-2] + N.roll(E2u, -1, axis = 2)[p, 1:, 1:-2])
    # Left edge
    E1d[p, 0, 1:-1] = g11d[0, 1:-1, 0] * E1u[p, 0, 1:-1] + 0.5 * g12d[0, 1:-1, 0] * (E2u[p, 0, 1:-2] + N.roll(E2u, -1, axis = 2)[p, 0, 1:-2])
    # Right edge
    E1d[p, -1, 1:-1] = g11d[-1, 1:-1, 0] * E1u[p, -1, 1:-1] + 0.5 * g12d[-1, 1:-1, 0] * (E2u[p, -1, 1:-2] + N.roll(E2u, -1, axis = 2)[p, -1, 1:-2])
    # Bottom edge
    E1d[p, 1:-1, 0] = g11d[0:-1, 0, 1] * E1u[p, 1:-1, 0] + 0.5 * g12d[0:-1, 0, 1] * (E2u[p, 1:, 0] + N.roll(E2u, 1, axis = 1)[p, 1:, 0])
    # Top edge
    E1d[p, 1:-1, -1] = g11d[0:-1, -1, 1] * E1u[p, 1:-1, -1] + 0.5 * g12d[0:-1, -1, 1] * (E2u[p, 1:, -1] + N.roll(E2u, 1, axis = 1)[p, 1:, -1])
    # Bottom left corner
    E1d[p, 0, 0] = g11d[0, 0, 0] * E1u[p, 0, 0] + g12d[0, 0, 0] * E2u[p, 0, 0]
    # Bottom right corner
    E1d[p, -1, 0] = g11d[-1, 0, 0] * E1u[p, -1, 0] + g12d[-1, 0, 0] * E2u[p, -1, 0]
    # Top left corner
    E1d[p, 0, -1] = g11d[0, -1, 0] * E1u[p, 0, -1] + g12d[0, -1, 0] * E2u[p, 0, -1]
    # Top right corner
    E1d[p, -1, -1] = g11d[-1, -1, 0] * E1u[p, -1, -1] + g12d[-1, -1, 0] * E2u[p, -1, -1]

    ##### Eeta

    # Interior
    E2d[p, 1:-1, 1:-1] = g22d[1:-1, 0:-1, 2] * E2u[p, 1:-1, 1:-1] \
                    + 0.25 * g12d[1:-1, 0:-1, 2] * (E1u[p, 1:-2, 1:] + N.roll(N.roll(E1u, -1, axis = 1), 1, axis = 2)[p, 1:-2, 1:] \
                    + N.roll(E1u, -1, axis = 1)[p, 1:-2, 1:] + N.roll(E1u, 1, axis = 2)[p, 1:-2, 1:])
    # Left edge
    E2d[p, 0, 1:-1] = g22d[0, 0:-1, 2] * E2u[p, 0, 1:-1] + 0.5 * g12d[0, 0:-1, 2] * (E1u[p, 0, 1:] + N.roll(E1u, 1, axis = 2)[p, 0, 1:])
    # Right edge
    E2d[p, -1, 1:-1] = g22d[-1, 0:-1, 2] * E2u[p, -1, 1:-1] + 0.5 * g12d[-1, 0:-1, 2] * (E1u[p, -1, 1:] + N.roll(E1u, 1, axis = 2)[p, -1, 1:])
    # Bottom edge
    E2d[p, 1:-1, 0] = g22d[1:-1, 0, 0] * E2u[p, 1:-1, 0] + 0.5 * g12d[1:-1, 0, 0] * (E1u[p, 1:-2, 0] + N.roll(E1u, -1, axis = 1)[p, 1:-2, 0])
    # Top edge
    E2d[p, 1:-1, -1] = g22d[1:-1, -1, 0] * E2u[p, 1:-1, -1] + 0.5 * g12d[1:-1, -1, 0] * (E1u[p, 1:-2, -1] + N.roll(E1u, -1, axis = 1)[p, 1:-2, -1])
    # Bottom left corner
    E2d[p, 0, 0] = g22d[0, 0, 0] * E2u[p, 0, 0] + g12d[0, 0, 0] * E1u[p, 0, 0]
    # Bottom right corner
    E2d[p, -1, 0] = g22d[-1, 0, 0] * E2u[p, -1, 0] + g12d[-1, 0, 0] * E1u[p, -1, 0]
    # Top left corner
    E2d[p, 0, -1] = g22d[0, -1, 0] * E2u[p, 0, -1] + g12d[0, -1, 0] * E1u[p, 0, -1]
    # Top right corner
    E2d[p, -1, -1] = g22d[-1, -1, 0] * E2u[p, -1, -1] + g12d[-1, -1, 0] * E1u[p, -1, -1]

def push_B(p, dtin):
        
        # Interior
        Br[p, 1:-1, 1:-1] += dtin * (dE1d2[p, 1:-1, 1:-1] - dE2d1[p, 1:-1, 1:-1]) / sqrt_det_g[0:-1, 0:-1, 3] 
        # Left edge
        Br[p, 0, 1:-1] += dtin * (dE1d2[p, 0, 1:-1] - dE2d1[p, 0, 1:-1]) / sqrt_det_g[0, 0:-1, 2] 
        # Right edge
        Br[p, -1, 1:-1] += dtin * (dE1d2[p, -1, 1:-1] - dE2d1[p, -1, 1:-1]) / sqrt_det_g[-1, 0:-1, 2] 
        # Bottom edge
        Br[p, 1:-1, 0] += dtin * (dE1d2[p, 1:-1, 0] - dE2d1[p, 1:-1, 0]) / sqrt_det_g[0:-1, 0, 1] 
        # Top edge
        Br[p, 1:-1, -1] += dtin * (dE1d2[p, 1:-1, -1] - dE2d1[p, 1:-1, -1]) / sqrt_det_g[0:-1, -1, 1] 
        # Bottom left corner
        Br[p, 0, 0] += dtin * (dE1d2[p, 0, 0] - dE2d1[p, 0, 0]) / sqrt_det_g[0, 0, 0] 
        # Bottom right corner
        Br[p, -1, 0] += dtin * (dE1d2[p, -1, 0] - dE2d1[p, -1, 0]) / sqrt_det_g[-1, 0, 0] 
        # Top left corner
        Br[p, 0, -1] += dtin * (dE1d2[p, 0, -1] - dE2d1[p, 0, -1]) / sqrt_det_g[0, -1, 0] 
        # Top right corner
        Br[p, -1, -1] += dtin * (dE1d2[p, -1, -1] - dE2d1[p, -1, -1]) / sqrt_det_g[-1, -1, 0] 
        
        # Current
        Br[p, :, :] += dtin * Jz[p, :, :] * N.sin(20.0 * it * dtin) * (1 + N.tanh(20 - it/5.))/2.

def push_E(p, it, dtin):

        # Interior
        E1u[p, 1:-1, :] += dtin * dBrd2[p, 1:-1, :] / sqrt_det_g[0:-1, :, 1] 
        # Left edge
        E1u[p, 0, :] += dtin * dBrd2[p, 0, :] / sqrt_det_g[0, :, 0] 
        # Right edge
        E1u[p, -1, :] += dtin * dBrd2[p, -1, :] / sqrt_det_g[-1, :, 0] 

        # Interior
        E2u[p, :, 1:-1] -= dtin * dBrd1[p, :, 1:-1] / sqrt_det_g[:, 0:-1, 2]
        # Bottom edge
        E2u[p, :, 0] -= dtin * dBrd1[p, :, 0] / sqrt_det_g[:, 0, 0]
        # Top edge
        E2u[p, :, -1] -= dtin * dBrd1[p, :, -1] / sqrt_det_g[:, -1, 0]

########
# Boundary conditions
########

def interp(arr_in, xA, xB):
    return N.interp(xB, xA, arr_in)

# Interface inner boundary 
sig_in  = 0.45 / dt #0.45 / dt # 75.0 # 0.5 / dt # 200.0

def compute_delta_B(p0, p1, dtin):

    i0 = 1
    i1 = Nxi_half - 1

    top = topology[p0, p1]
    
    if (top == 'xx'):

        # p0 -> p1
        xi1 = xi_int[0]
        eta1 = eta_half #[i0:i1]
        xi0, eta0 = transform_coords(p1, p0, xi1, eta1)
        E1 = interp(E1u[p0, -1, :], eta_int, eta0)
        E2 = interp(E2u[p0, -1, :], eta_half, eta0)
        E_target = transform_vect(p0, p1, xi0, eta0, E1, E2)[1]
        delta1 = dtin * sig_in * (E2u[p1, 0, :] - E_target)

        # p1 -> p0
        xi0 = xi_int[-1]
        eta0 = eta_half #[i0:i1]
        xi1, eta1 = transform_coords(p0, p1, xi0, eta0)
        E1 = interp(E1u[p1, 0, :], eta_int, eta1)
        E2 = interp(E2u[p1, 0, :], eta_half, eta1)
        E_target = transform_vect(p1, p0, xi1, eta1, E1, E2)[1]
        delta2 = dtin * sig_in * (E_target - E2u[p0, -1, :])

        delta = 0.5 * (delta1 + delta2)
        diff_Br[p0, -1, i0:i1] += delta[i0:i1]
        diff_Br[p1, 0, i0:i1]  += delta[i0:i1]

    if (top == 'xy'):

        # p0 -> p1
        xi1 = xi_half #[i0:i1]
        eta1 = eta_int[0]
        xi0, eta0 = transform_coords(p1, p0, xi1, eta1)
        E1 = interp(E1u[p0, -1, :], eta_int, eta0)
        E2 = interp(E2u[p0, -1, :], eta_half, eta0)
        E_target = transform_vect(p0, p1, xi0, eta0, E1, E2)[0]
        delta1 = dtin * sig_in * (E_target - E1u[p1, :, 0])

        # p1 -> p0
        xi0 = xi_int[-1]
        eta0 = eta_half #[i0:i1]
        xi1, eta1 = transform_coords(p0, p1, xi0, eta0)
        E1 = interp(E1u[p1, :, 0], xi_half, xi1)
        E2 = interp(E2u[p1, :, 0], xi_int, xi1)
        E_target = transform_vect(p1, p0, xi1, eta1, E1, E2)[1]
        delta2 = dtin * sig_in * (E_target - E2u[p0, -1, :])

        diff_Br[p0, -1, i0:i1] += 0.5 * (delta1[::-1] + delta2)[i0:i1]
        diff_Br[p1, i0:i1, 0]  += 0.5 * (delta1 + delta2[::-1])[i0:i1]

    if (top == 'yy'):

        # p0 -> p1
        xi1 = xi_half #[i0:i1]
        eta1 = eta_int[0]
        xi0, eta0 = transform_coords(p1, p0, xi1, eta1)
        E1 = interp(E1u[p0, :, -1], xi_half, xi0)
        E2 = interp(E2u[p0, :, -1], xi_int, xi0)
        E_target = transform_vect(p0, p1, xi0, eta0, E1, E2)[0]
        delta1 = dtin * sig_in * (E_target - E1u[p1, :, 0])

        # p1 -> p0
        xi0 = xi_half #[i0:i1]
        eta0 = eta_int[-1]
        xi1, eta1 = transform_coords(p0, p1, xi0, eta0)
        E1 = interp(E1u[p1, :, 0], xi_half, xi1)
        E2 = interp(E2u[p1, :, 0], xi_int, xi1)
        E_target = transform_vect(p1, p0, xi1, eta1, E1, E2)[0]
        delta2 = dtin * sig_in * (E1u[p0, :, -1] - E_target)

        delta = 0.5 * (delta1 + delta2)
        diff_Br[p0, i0:i1, -1] += delta[i0:i1]
        diff_Br[p1, i0:i1, 0]  += delta[i0:i1]

    if (top == 'yx'):

        # p0 -> p1
        xi1 = xi_int[0]
        eta1 = eta_half #[i0:i1]
        xi0, eta0 = transform_coords(p1, p0, xi1, eta1)
        E1 = interp(E1u[p0, :, -1], xi_half, xi0)
        E2 = interp(E2u[p0, :, -1], xi_int, xi0)
        E_target = transform_vect(p0, p1, xi0, eta0, E1, E2)[1]
        delta1 = dtin * sig_in * (E2u[p1, 0, :] - E_target)
        
        # p1 -> p0
        xi0 = xi_half #[i0:i1]
        eta0 = eta_int[-1]
        xi1, eta1 = transform_coords(p0, p1, xi0, eta0)
        E1 = interp(E1u[p1, 0, :], eta_int, eta1)
        E2 = interp(E2u[p1, 0, :], eta_half, eta1)
        E_target = transform_vect(p1, p0, xi1, eta1, E1, E2)[0]
        delta2 = dtin * sig_in * (E1u[p0, :, -1] - E_target)

        diff_Br[p0, i0:i1, -1] += 0.5 * (delta1[::-1] + delta2)[i0:i1]
        diff_Br[p1, 0, i0:i1]  += 0.5 * (delta1 + delta2[::-1])[i0:i1]


def interface_B(p0, p1):

    i0 = 0
    i1 = Nxi_half

    top = topology[p0, p1]
    
    if (top == 'xx'):
        Br[p0, -1, i0:i1] -= diff_Br[p0, -1, i0:i1]
        Br[p1, 0, i0:i1]  -= diff_Br[p1, 0, i0:i1] 

    if (top == 'xy'):
        Br[p0, -1, i0:i1] -= diff_Br[p0, -1, i0:i1]
        Br[p1, i0:i1, 0]  -= diff_Br[p1, i0:i1, 0] 

    if (top == 'yy'):
        Br[p0, i0:i1, -1] -= diff_Br[p0, i0:i1, -1]
        Br[p1, i0:i1, 0]  -= diff_Br[p1, i0:i1, 0] 

    if (top == 'yx'):
        Br[p0, i0:i1, -1] -= diff_Br[p0, i0:i1, -1]
        Br[p1, 0, i0:i1]  -= diff_Br[p1, 0, i0:i1] 

def compute_delta_E(p0, p1, dtin):

    i0 = 1
    i1 = Nxi_half - 1

    top = topology[p0, p1]
    
    if (top == 'xx'):

        # p0 -> p1
        xi1 = xi_int[0]
        eta1 = eta_half #[i0:i1]
        xi0, eta0 = transform_coords(p1, p0, xi1, eta1)
        B_target = interp(Br[p0, -1, :], eta_half, eta0)
        delta1 = dtin * sig_in * (Br[p1, 0, :] - B_target)

        # p1 -> p0
        xi0 = xi_int[-1]
        eta0 = eta_half #[i0:i1]
        xi1, eta1 = transform_coords(p0, p1, xi0, eta0)
        B_target = interp(Br[p1, 0, :], eta_half, eta1)
        delta2 = dtin * sig_in * (B_target - Br[p0, -1, :])

        delta = 0.5 * (delta1 + delta2)
        diff_E2[p0, -1, i0:i1] += delta[i0:i1]
        diff_E2[p1, 0, i0:i1]  += delta[i0:i1]

    if (top == 'xy'):

        # p0 -> p1
        xi1 = xi_half #[i0:i1]
        eta1 = eta_int[0]
        xi0, eta0 = transform_coords(p1, p0, xi1, eta1)
        B_target = interp(Br[p0, -1, :], eta_half, eta0)
        delta1 = dtin * sig_in * (Br[p1, :, 0] - B_target)

        # p1 -> p0
        xi0 = xi_int[-1]
        eta0 = eta_half #[i0:i1]
        xi1, eta1 = transform_coords(p0, p1, xi0, eta0)
        B_target = interp(Br[p1, :, 0], xi_half, xi1)
        delta2 = dtin * sig_in * (B_target - Br[p0, -1, :])

        diff_E2[p0, -1, i0:i1] += 0.5 * (delta1[::-1] + delta2)[i0:i1]
        diff_E1[p1, i0:i1, 0]  += 0.5 * (delta1 + delta2[::-1])[i0:i1]

    if (top == 'yy'):

        # p0 -> p1
        xi1 = xi_half #[i0:i1]
        eta1 = eta_int[0]
        xi0, eta0 = transform_coords(p1, p0, xi1, eta1)
        B_target = interp(Br[p0, :, -1], xi_half, xi0)
        delta1 = dtin * sig_in * (B_target - Br[p1, :, 0])

        # p1 -> p0
        xi0 = xi_half #[i0:i1]
        eta0 = eta_int[-1]
        xi1, eta1 = transform_coords(p0, p1, xi0, eta0)
        B_target = interp(Br[p1, :, 0], xi_half, xi1)
        delta2 = dtin * sig_in * (Br[p0, :, -1] - B_target)

        delta = 0.5 * (delta1 + delta2)
        diff_E1[p0, i0:i1, -1] += delta[i0:i1]
        diff_E1[p1, i0:i1, 0]  += delta[i0:i1]

    if (top == 'yx'):

        # p0 -> p1
        xi1 = xi_int[0]
        eta1 = eta_half #[i0:i1]
        xi0, eta0 = transform_coords(p1, p0, xi1, eta1)
        B_target = interp(Br[p0, :, -1], xi_half, xi0)
        delta1 = dtin * sig_in * (B_target - Br[p1, 0, :])

        # p1 -> p0
        xi0 = xi_half #[i0:i1]
        eta0 = eta_int[-1]
        xi1, eta1 = transform_coords(p0, p1, xi0, eta0)
        B_target = interp(Br[p1, 0, :], eta_half, eta1)
        delta2 = dtin * sig_in * (Br[p0, :, -1] - B_target)

        diff_E1[p0, i0:i1, -1] += 0.5 * (delta1[::-1] + delta2)[i0:i1]
        diff_E2[p1, 0, i0:i1]  += 0.5 * (delta1 + delta2[::-1])[i0:i1]

def interface_E(p0, p1):

    i0 =  1 # 1
    i1 = Nxi_half - 1

    top = topology[p0, p1]
    
    if (top == 'xx'):
        E2u[p0, -1, i0:i1] -= diff_E2[p0, -1, i0:i1]
        E2u[p1, 0, i0:i1]  -= diff_E2[p1, 0, i0:i1] 

    if (top == 'xy'):
        E2u[p0, -1, i0:i1] -= diff_E2[p0, -1, i0:i1]
        E1u[p1, i0:i1, 0]  += diff_E1[p1, i0:i1, 0] 

    if (top == 'yy'):
        E1u[p0, i0:i1, -1] -= diff_E1[p0, i0:i1, -1]
        E1u[p1, i0:i1, 0]  -= diff_E1[p1, i0:i1, 0] 

    if (top == 'yx'):
        E1u[p0, i0:i1, -1] -= diff_E1[p0, i0:i1, -1]
        E2u[p1, 0, i0:i1]  += diff_E2[p1, 0, i0:i1] 

########
# Triple points
########

sig_cor  = 0.4 / dt 

def corners_tangent_B(n_corner):

    p0, p1, p2 = triplet[n_corner, 0], triplet[n_corner, 1], triplet[n_corner, 2]
    i0, j0 = index_corners[n_corner, 0][0], index_corners[n_corner, 0][1]
    i1, j1 = index_corners[n_corner, 1][0], index_corners[n_corner, 1][1]
    i2, j2 = index_corners[n_corner, 2][0], index_corners[n_corner, 2][1]
    
    # Patch 0
    if (index_corners[n_corner, 0] == (-1, -1)):
        txi0, teta0 = 1, -1
        norm = N.sqrt(g11u[-1, -1, 0] * txi0**2 + g22u[-1, -1, 0] * teta0**2 + 2.0 * g12u[-1, -1, 0] * txi0 * teta0)
        txi0 /= norm
        teta0 /= norm
    elif (index_corners[n_corner, 0] == (-1, 0)):
        txi0, teta0 = 1, 1
        norm = N.sqrt(g11u[-1, 0, 0] * txi0**2 + g22u[-1, 0, 0] * teta0**2 + 2.0 * g12u[-1, 0, 0] * txi0 * teta0)
        txi0 /= norm
        teta0 /= norm
    elif (index_corners[n_corner, 0] == (0, 0)):
        txi0, teta0 = 1, -1
        norm = N.sqrt(g11u[0, 0, 0] * txi0**2 + g22u[0, 0, 0] * teta0**2 + 2.0 * g12u[0, 0, 0] * txi0 * teta0)
        txi0 /= norm
        teta0 /= norm
    elif (index_corners[n_corner, 0] == (0, -1)):
        txi0, teta0 = 1, 1
        norm = N.sqrt(g11u[0, -1, 0] * txi0**2 + g22u[0, -1, 0] * teta0**2 + 2.0 * g12u[0, -1, 0] * txi0 * teta0)
        txi0 /= norm
        teta0 /= norm
    
    txi1, teta1 = transform_form(p0, p1, xi_int[i0], eta_int[j0], txi0, teta0)
    txi2, teta2 = transform_form(p0, p2, xi_int[i0], eta_int[j0], txi0, teta0)
    Etan0 = E1u[p0, i0, j0] * txi0 + E2u[p0, i0, j0] * teta0
    Etan1 = E1u[p1, i1, j1] * txi1 + E2u[p1, i1, j1] * teta1
    Etan2 = E1u[p2, i2, j2] * txi2 + E2u[p2, i2, j2] * teta2
    deltac0 = Etan0 - 0.5 * (Etan1 + Etan2)

    # Patch 1
    if (index_corners[n_corner, 1] == (-1, -1)):
        txi1, teta1 = 1, -1
        norm = N.sqrt(g11u[-1, -1, 1] * txi1**2 + g22u[-1, -1, 0] * teta1**2 + 2.0 * g12u[-1, -1, 0] * txi1 * teta1)
        txi1 /= norm
        teta1 /= norm
    elif (index_corners[n_corner, 1] == (-1, 0)):
        txi1, teta1 = 1, 1
        norm = N.sqrt(g11u[-1, 0, 0] * txi1**2 + g22u[-1, 0, 0] * teta1**2 + 2.0 * g12u[-1, 0, 0] * txi1 * teta1)
        txi1 /= norm
        teta1 /= norm
    elif (index_corners[n_corner, 1] == (0, 0)):
        txi1, teta1 = 1, -1
        norm = N.sqrt(g11u[0, 0, 0] * txi1**2 + g22u[0, 0, 0] * teta1**2 + 2.0 * g12u[0, 0, 0] * txi1 * teta1)
        txi1 /= norm
        teta1 /= norm
    elif (index_corners[n_corner, 1] == (0, -1)):
        txi1, teta1 = 1, 1
        norm = N.sqrt(g11u[0, -1, 0] * txi1**2 + g22u[0, -1, 0] * teta1**2 + 2.0 * g12u[0, -1, 0] * txi1 * teta1)
        txi1 /= norm
        teta1 /= norm
    
    txi0, teta0 = transform_form(p1, p0, xi_int[i1], eta_int[j1], txi1, teta1)
    txi2, teta2 = transform_form(p1, p2, xi_int[i1], eta_int[j1], txi1, teta1)
    Etan0 = E1u[p0, i0, j0] * txi0 + E2u[p0, i0, j0] * teta0
    Etan1 = E1u[p1, i1, j1] * txi1 + E2u[p1, i1, j1] * teta1
    Etan2 = E1u[p2, i2, j2] * txi2 + E2u[p2, i2, j2] * teta2
    deltac1 = Etan1 - 0.5 * (Etan0 + Etan2)

    # Patch 2
    if (index_corners[n_corner, 2] == (-1, -1)):
        txi2, teta2 = 1, -1
        norm = N.sqrt(g11u[-1, -1, 1] * txi2**2 + g22u[-1, -1, 0] * teta2**2 + 2.0 * g12u[-1, -1, 0] * txi2 * teta2)
        txi2 /= norm
        teta2 /= norm
    elif (index_corners[n_corner, 2] == (-1, 0)):
        txi2, teta2 = 1, 1
        norm = N.sqrt(g11u[-1, 0, 0] * txi2**2 + g22u[-1, 0, 0] * teta2**2 + 2.0 * g12u[-1, 0, 0] * txi2 * teta2)
        txi2 /= norm
        teta2 /= norm
    elif (index_corners[n_corner, 2] == (0, 0)):
        txi2, teta2 = 1, -1
        norm = N.sqrt(g11u[0, 0, 0] * txi2**2 + g22u[0, 0, 0] * teta2**2 + 2.0 * g12u[0, 0, 0] * txi2 * teta2)
        txi2 /= norm
        teta2 /= norm
    elif (index_corners[n_corner, 2] == (0, -1)):
        txi2, teta2 = 1, 1
        norm = N.sqrt(g11u[0, -1, 0] * txi2**2 + g22u[0, -1, 0] * teta2**2 + 2.0 * g12u[0, -1, 0] * txi2 * teta2)
        txi2 /= norm
        teta2 /= norm
    
    txi0, teta0 = transform_form(p2, p0, xi_int[i2], eta_int[j2], txi2, teta2)
    txi1, teta1 = transform_form(p2, p1, xi_int[i2], eta_int[j2], txi2, teta2)
    Etan0 = E1u[p0, i0, j0] * txi0 + E2u[p0, i0, j0] * teta0
    Etan1 = E1u[p1, i1, j1] * txi1 + E2u[p1, i1, j1] * teta1
    Etan2 = E1u[p2, i2, j2] * txi2 + E2u[p2, i2, j2] * teta2
    deltac2 = Etan2 - 0.5 * (Etan0 + Etan1)
 
    # Final term
    delta = (deltac0 + deltac1 + deltac2) / 3.0

    diff_Br[p0, i0, j0] = sig_cor * dt * delta
    diff_Br[p1, i1, j1] = sig_cor * dt * delta
    diff_Br[p2, i2, j2] = sig_cor * dt * delta

def corners_tangent_E(n_corner):

    p0, p1, p2 = triplet[n_corner, 0], triplet[n_corner, 1], triplet[n_corner, 2]
    i0, j0 = index_corners[n_corner, 0][0], index_corners[n_corner, 0][1]
    i1, j1 = index_corners[n_corner, 1][0], index_corners[n_corner, 1][1]
    i2, j2 = index_corners[n_corner, 2][0], index_corners[n_corner, 2][1]
    
    Br0 = Br[p0, i0, j0]
    Br1 = Br[p1, i1, j1]
    Br2 = Br[p2, i2, j2]

    # Patch 0
    if (index_corners[n_corner, 0] == (-1, -1)):
        txi0, teta0 = 1, -1
        norm = N.sqrt(g11u[-1, -1, 0] * txi0**2 + g22u[-1, -1, 0] * teta0**2 + 2.0 * g12u[-1, -1, 0] * txi0 * teta0)
        txi0 /= norm
        teta0 /= norm
    elif (index_corners[n_corner, 0] == (-1, 0)):
        txi0, teta0 = 1, 1
        norm = N.sqrt(g11u[-1, 0, 0] * txi0**2 + g22u[-1, 0, 0] * teta0**2 + 2.0 * g12u[-1, 0, 0] * txi0 * teta0)
        txi0 /= norm
        teta0 /= norm
    elif (index_corners[n_corner, 0] == (0, 0)):
        txi0, teta0 = 1, -1
        norm = N.sqrt(g11u[0, 0, 0] * txi0**2 + g22u[0, 0, 0] * teta0**2 + 2.0 * g12u[0, 0, 0] * txi0 * teta0)
        txi0 /= norm
        teta0 /= norm
    elif (index_corners[n_corner, 0] == (0, -1)):
        txi0, teta0 = 1, 1
        norm = N.sqrt(g11u[0, -1, 0] * txi0**2 + g22u[0, -1, 0] * teta0**2 + 2.0 * g12u[0, -1, 0] * txi0 * teta0)
        txi0 /= norm
        teta0 /= norm
    
    deltaEt = sig_cor * dt * (Br0 - 0.5 * (Br1 + Br2))
    diff_E1[p0, i0, j0] = deltaEt * (g11u[i0, j0, 0] * txi0 + g12u[i0, j0, 0] * teta0)
    diff_E2[p0, i0, j0] = deltaEt * (g22u[i0, j0, 0] * teta0 + g12u[i0, j0, 0] * txi0)

    # Patch 1
    if (index_corners[n_corner, 1] == (-1, -1)):
        txi1, teta1 = 1, -1
        norm = N.sqrt(g11u[-1, -1, 0] * txi1**2 + g22u[-1, -1, 0] * teta1**2 + 2.0 * g12u[-1, -1, 0] * txi1 * teta1)
        txi1 /= norm
        teta1 /= norm
    elif (index_corners[n_corner, 1] == (-1, 0)):
        txi1, teta1 = 1, 1
        norm = N.sqrt(g11u[-1, 0, 0] * txi1**2 + g22u[-1, 0, 0] * teta1**2 + 2.0 * g12u[-1, 0, 0] * txi1 * teta1)
        txi1 /= norm
        teta1 /= norm
    elif (index_corners[n_corner, 1] == (0, 0)):
        txi1, teta1 = 1, -1
        norm = N.sqrt(g11u[0, 0, 0] * txi1**2 + g22u[0, 0, 0] * teta1**2 + 2.0 * g12u[0, 0, 0] * txi1 * teta1)
        txi1 /= norm
        teta1 /= norm
    elif (index_corners[n_corner, 1] == (0, -1)):
        txi1, teta1 = 1, 1
        norm = N.sqrt(g11u[0, -1, 0] * txi1**2 + g22u[0, -1, 0] * teta1**2 + 2.0 * g12u[0, -1, 0] * txi1 * teta1)
        txi1 /= norm
        teta1 /= norm
    
    deltaEt = sig_cor * dt * (Br1 - 0.5 * (Br0 + Br2))
    diff_E1[p1, i1, j1] = deltaEt * (g11u[i1, j1, 0] * txi1 + g12u[i1, j1, 0] * teta1)
    diff_E2[p1, i1, j1] = deltaEt * (g22u[i1, j1, 0] * teta1 + g12u[i1, j1, 0] * txi1)

    # Patch 2
    if (index_corners[n_corner, 2] == (-1, -1)):
        txi2, teta2 = 1, -1
        norm = N.sqrt(g11u[-1, -1, 0] * txi2**2 + g22u[-1, -1, 0] * teta2**2 + 2.0 * g12u[-1, -1, 0] * txi2 * teta2)
        txi2 /= norm
        teta2 /= norm
    elif (index_corners[n_corner, 2] == (-1, 0)):
        txi2, teta2 = 1, 1
        norm = N.sqrt(g11u[-1, 0, 0] * txi2**2 + g22u[-1, 0, 0] * teta2**2 + 2.0 * g12u[-1, 0, 0] * txi2 * teta2)
        txi2 /= norm
        teta2 /= norm
    elif (index_corners[n_corner, 2] == (0, 0)):
        txi2, teta2 = 1, -1
        norm = N.sqrt(g11u[0, 0, 0] * txi2**2 + g22u[0, 0, 0] * teta2**2 + 2.0 * g12u[0, 0, 0] * txi2 * teta2)
        txi2 /= norm
        teta2 /= norm
    elif (index_corners[n_corner, 2] == (0, -1)):
        txi2, teta2 = 1, 1
        norm = N.sqrt(g11u[0, -1, 0] * txi2**2 + g22u[0, -1, 0] * teta2**2 + 2.0 * g12u[0, -1, 0] * txi2 * teta2)
        txi2 /= norm
        teta2 /= norm
    
    deltaEt = sig_cor * dt * (Br2 - 0.5 * (Br0 + Br1))
    diff_E1[p2, i2, j2] = deltaEt * (g11u[i2, j2, 0] * txi2 + g12u[i2, j2, 0] * teta2)
    diff_E2[p2, i2, j2] = deltaEt * (g22u[i2, j2, 0] * teta2 + g12u[i2, j2, 0] * txi2)


# def corners(field):

#     #ABS
#     p0 = Sphere.A
#     p1 = Sphere.B 
#     p2 = Sphere.S
    
#     R0 = field[p0, -1, 0]
#     R1 = field[p1, 0, 0]
#     R2 = field[p2, -1, -1]
#     Rm = (R0 + R1 + R2) / 3.0
#     field[p0, -1, 0] -= dt * sig_cor * (R0 - 0.5 * (R1 + R2))
#     field[p1, 0, 0]  -= dt * sig_cor * (R1 - 0.5 * (R0 + R2))
#     field[p2, -1, -1]-= dt * sig_cor * (R2 - 0.5 * (R0 + R1))

#     #ABN
#     p0 = Sphere.A
#     p1 = Sphere.B 
#     p2 = Sphere.N
    
#     R0 = field[p0, -1, -1]
#     R1 = field[p1, 0, -1]
#     R2 = field[p2, 0, 0]
#     Rm = (R0 + R1 + R2) / 3.0
#     field[p0, -1, -1] -= dt * sig_cor * (R0 - 0.5 * (R1 + R2))
#     field[p1, 0, -1]  -= dt * sig_cor * (R1 - 0.5 * (R0 + R2))
#     field[p2, 0, 0]   -= dt * sig_cor * (R2 - 0.5 * (R0 + R1))

#     #ADN
#     p0 = Sphere.A
#     p1 = Sphere.D 
#     p2 = Sphere.N
    
#     R0 = field[p0, 0, -1]
#     R1 = field[p1, 0, -1]
#     R2 = field[p2, 0, -1]
#     Rm = (R0 + R1 + R2) / 3.0
#     field[p0, 0, -1] -= dt * sig_cor * (R0 - 0.5 * (R1 + R2))
#     field[p1, 0, -1] -= dt * sig_cor * (R1 - 0.5 * (R0 + R2))
#     field[p2, 0, -1] -= dt * sig_cor * (R2 - 0.5 * (R0 + R1))

#     #ADS
#     p0 = Sphere.A
#     p1 = Sphere.D 
#     p2 = Sphere.S
    
#     R0 = field[p0, 0, 0]
#     R1 = field[p1, -1, -1]
#     R2 = field[p2, 0, -1]
#     Rm = (R0 + R1 + R2) / 3.0
#     field[p0, 0, 0]   -= dt * sig_cor * (R0 - 0.5 * (R1 + R2))
#     field[p1, -1, -1] -= dt * sig_cor * (R1 - 0.5 * (R0 + R2))
#     field[p2, 0, -1]  -= dt * sig_cor * (R2 - 0.5 * (R0 + R1))

#     #BCN
#     p0 = Sphere.B
#     p1 = Sphere.C
#     p2 = Sphere.N
    
#     R0 = field[p0, -1, -1]
#     R1 = field[p1, 0, 0]
#     R2 = field[p2, -1, 0]
#     Rm = (R0 + R1 + R2) / 3.0
#     field[p0, -1, -1] -= dt * sig_cor * (R0 - 0.5 * (R1 + R2))
#     field[p1, 0, 0]   -= dt * sig_cor * (R1 - 0.5 * (R0 + R2))
#     field[p2, -1, 0]  -= dt * sig_cor * (R2 - 0.5 * (R0 + R1))

#     #BCS
#     p0 = Sphere.B
#     p1 = Sphere.C
#     p2 = Sphere.S
    
#     R0 = field[p0, -1, 0]
#     R1 = field[p1, -1, 0]
#     R2 = field[p2, -1, 0]
#     Rm = (R0 + R1 + R2) / 3.0
#     field[p0, -1, 0] -= dt * sig_cor * (R0 - 0.5 * (R1 + R2))
#     field[p1, -1, 0] -= dt * sig_cor * (R1 - 0.5 * (R0 + R2))
#     field[p2, -1, 0] -= dt * sig_cor * (R2 - 0.5 * (R0 + R1))

#     #CDN
#     p0 = Sphere.C
#     p1 = Sphere.D
#     p2 = Sphere.N
    
#     R0 = field[p0, 0, -1]
#     R1 = field[p1, 0, 0]
#     R2 = field[p2, -1, -1]
#     Rm = (R0 + R1 + R2) / 3.0
#     field[p0, 0, -1]  -= dt * sig_cor * (R0 - 0.5 * (R1 + R2))
#     field[p1, 0, 0]   -= dt * sig_cor * (R1 - 0.5 * (R0 + R2))
#     field[p2, -1, -1] -= dt * sig_cor * (R2 - 0.5 * (R0 + R1))

#     #CDS
#     p0 = Sphere.C
#     p1 = Sphere.D
#     p2 = Sphere.S
    
#     R0 = field[p0, -1, -1]
#     R1 = field[p1, -1, 0]
#     R2 = field[p2, 0, 0]
#     Rm = (R0 + R1 + R2) / 3.0
#     field[p0, -1, -1]-= dt * sig_cor * (R0 - 0.5 * (R1 + R2))
#     field[p1, -1, 0] -= dt * sig_cor * (R1 - 0.5 * (R0 + R2))
#     field[p2, 0, 0]  -= dt * sig_cor * (R2 - 0.5 * (R0 + R1))

# def corners_B(p0, p1, p2):

#     #ABS
#     p0 = Sphere.A
#     p1 = Sphere.B 
#     p2 = Sphere.S
    
#     # A -> B
#     xi0 = xi_int[-1]
#     eta0 = eta_int[0]
#     E_target = transform_vect(p0, p1, xi0, eta0, E1u[p0, -1, 0], E2u[p0, -1, 0])[1]
#     delta1 = dt * sig_in * (E2u[p1, 0, 0] - E_target)
#     # B -> A
#     xi1 = xi_int[0]
#     eta1 = eta_int[0]
#     E_target = transform_vect(p1, p0, xi1, eta1, E1u[p1, 0, 0], E2u[p1, 0, 0])[1]
#     delta2 = dt * sig_in * (E_target - E1u[p0, -1, :])
#     # A -> S
#     xi0 = xi_int[-1]
#     eta0 = eta_int[0]
#     E_target = transform_vect(p0, p2, xi0, eta0, E1u[p0, -1, 0], E2u[p0, -1, 0])[0]
#     delta3 = dt * sig_in * (E1u[p1, -1, -1] - E_target)
#     # S -> A
#     xi2 = xi_int[-1]
#     eta2 = eta_int[-1]
#     E_target = transform_vect(p2, p0, xi2, eta2, E1u[p2, -1, -1], E2u[p2, -1, -1])[0]
#     delta4 = dt * sig_in * (E_target - E1u[p0, -1, 0])
#     # B -> S
#     xi1 = xi_int[0]
#     eta1 = eta_int[0]
#     E_target = transform_vect(p1, p2, xi1, eta1, E1u[p1, 0, 0], E2u[p1, 0, 0])[1]
#     delta5 = dt * sig_in * (E1u[p2, -1, -1] - E_target)
            
########
# Initialization
########

amp = 1.0
n_mode = 2
wave = 2.0 * (xi_max - xi_min) / n_mode
# Br0 = amp * N.sin(2.0 * N.pi * (xBr_grid - xi_min) / wave) * N.sin(2.0 * N.pi * (yBr_grid - xi_min) / wave)
E1u0 = N.zeros_like(E1u)
E2u0 = N.zeros_like(E2u)

for patch in range(6):

    fvec = (globals()["vec_sph_to_" + sphere[patch]])
    fveci = (globals()["vec_"+sphere[patch]+"_to_sph"])
    fcoord = (globals()["coord_" + sphere[patch] + "_to_sph"])

    r0 = 1.0

    for i in range(Nxi_half):
        for j in range(Neta_half):

            th0, ph0 = fcoord(xBr_grid[i, j], yBr_grid[i, j])
            Br[patch, i, j] = amp * N.sin(th0)**3 * N.cos(3.0 * ph0)

    for i in range(Nxi_half):
        for j in range(Neta_int):

            th0, ph0 = fcoord(xE1_grid[i, j], yE1_grid[i, j])

            EtTMP = 0.0
            EpTMP = 0.0

            ECStmp = fvec(th0, ph0, EtTMP, EpTMP)

            E1u[patch, i, j] = ECStmp[0]
            E1u0[patch, i, j] = ECStmp[0]

    for i in range(Nxi_int):
        for j in range(Neta_half):

            th0, ph0 = fcoord(xE2_grid[i, j], yE2_grid[i, j])

            EtTMP = 0.0
            EpTMP = 0.0

            ECStmp = fvec(th0, ph0, EtTMP, EpTMP)

            E2u[patch, i, j] = ECStmp[1]
            E2u0[patch, i, j] = ECStmp[1]

########
# Visualization
########

ratio = 0.5

xi_grid_c, eta_grid_c = unflip_eq(xBr_grid, yBr_grid)
xi_grid_d, eta_grid_d = unflip_eq(xBr_grid, yBr_grid)
xi_grid_n, eta_grid_n = unflip_po(xBr_grid, yBr_grid)

xi_grid_c2, eta_grid_c2 = unflip_eq(xE1_grid, yE1_grid)
xi_grid_d2, eta_grid_d2 = unflip_eq(xE1_grid, yE1_grid)
xi_grid_n2, eta_grid_n2 = unflip_po(xE1_grid, yE1_grid)

def plot_fields_unfolded(it, vm):

    fig = P.figure(1, facecolor='w')
    ax = P.subplot(111)

    ax.pcolormesh(xBr_grid, yBr_grid, Br[Sphere.A, :, :], cmap = "RdBu_r", vmin = - vm, vmax = vm)
    ax.pcolormesh(xBr_grid + N.pi / 2.0, yBr_grid, Br[Sphere.B, :, :], cmap = "RdBu_r", vmin = - vm, vmax = vm)
    ax.pcolormesh(xBr_grid, yBr_grid - N.pi / 2.0, Br[Sphere.S, :, :], cmap = "RdBu_r", vmin = - vm, vmax = vm)

    ax.pcolormesh(xi_grid_c + N.pi, eta_grid_c, Br[Sphere.C, :, :], cmap = "RdBu_r", vmin = - vm, vmax = vm)
    ax.pcolormesh(xi_grid_d - N.pi / 2.0, eta_grid_d, Br[Sphere.D, :, :], cmap = "RdBu_r", vmin = - vm, vmax = vm)
    ax.pcolormesh(xi_grid_n, eta_grid_n + N.pi / 2.0, Br[Sphere.N, :, :], cmap = "RdBu_r", vmin = - vm, vmax = vm)
    
    figsave_png(fig, "snapshots_penalty/Br_" + str(it))

    P.close('all')

def plot_fields_unfolded_E1(it, vm):

    fig = P.figure(1, facecolor='w')
    ax = P.subplot(111)

    ax.pcolormesh(xE1_grid, yE1_grid, E1u[Sphere.A, :, :], cmap = "RdBu_r", vmin = - vm, vmax = vm)
    ax.pcolormesh(xE1_grid + N.pi / 2.0, yE1_grid, E1u[Sphere.B, :, :], cmap = "RdBu_r", vmin = - vm, vmax = vm)
    ax.pcolormesh(xE1_grid, yE1_grid - N.pi / 2.0, E1u[Sphere.S, :, :], cmap = "RdBu_r", vmin = - vm, vmax = vm)

    ax.pcolormesh(xi_grid_c2 + N.pi, eta_grid_c2, E1u[Sphere.C, :, :], cmap = "RdBu_r", vmin = - vm, vmax = vm)
    ax.pcolormesh(xi_grid_d2 - N.pi / 2.0, eta_grid_d2, E1u[Sphere.D, :, :], cmap = "RdBu_r", vmin = - vm, vmax = vm)
    ax.pcolormesh(xi_grid_n2, eta_grid_n2 + N.pi / 2.0, E1u[Sphere.N, :, :], cmap = "RdBu_r", vmin = - vm, vmax = vm)
    
    figsave_png(fig, "snapshots_penalty/E1u_" + str(it))

    P.close('all')

########
# Main routine
########

idump = 0

patch = range(n_patches)

from scipy.ndimage.filters import gaussian_filter

Nt = 50 # Number of iterations
FDUMP = 1 # Dump frequency
time = dt * N.arange(Nt)
energy = N.zeros((n_patches, Nt))

for it in tqdm(range(Nt), "Progression"):
    if ((it % FDUMP) == 0):
        plot_fields_unfolded(idump, 1.0)
        plot_fields_unfolded_E1(idump, 1.0)
        idump += 1
    
    diff_Br[:, :, :] = 0.0
    diff_E1[:, :, :] = 0.0
    diff_E2[:, :, :] = 0.0

    compute_diff_B(patch)

    push_E(patch, it, dt)

    for i in range(n_corners):
        corners_tangent_E(i)
        
    for i in range(n_zeros):
        p0, p1 = index_row[i], index_col[i]
        compute_delta_E(p0, p1, dt)
        interface_E(p0, p1)
    
    # for i in range(n_zeros):
    #     p0, p1 = index_row[i], index_col[i]
    #     interface_E(p0, p1)

    # corners_E(patch)

    contra_to_cov_E(patch)
    # contra_to_cov_E_weights(patch)

    compute_diff_E(patch)

    push_B(patch, dt)

    for i in range(n_corners):
        corners_tangent_B(i)

    for i in range(n_zeros):
        p0, p1 = index_row[i], index_col[i]
        compute_delta_B(p0, p1, dt)
        interface_B(p0, p1)

    # for i in range(n_zeros):
    #     p0, p1 = index_row[i], index_col[i]
    #     interface_B(p0, p1)
    
    for p in range(n_patches):
        energy[p, it] = dxi * deta * (N.sum(Br[p, :, :]**2) \
        + N.sum(E1u[p, :, :]**2) + N.sum(E2u[p, :, :]**2))
for p in range(n_patches):
    P.plot(time, energy[p, :])


