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
cfl = 0.1
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
sqrt_det_g_half = N.empty(Nxi_half)
g12d_half = N.empty(Nxi_half)
g11d_half = N.empty(Nxi_half)
g22d_half = N.empty(Nxi_half)

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

for i in range(Nxi_half):        
        X = N.tan(xi_half[i])
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
Br = N.zeros((n_patches, Nxi_half, Neta_half))
E1u = N.zeros((n_patches, Nxi_half, Neta_int))
E2u = N.zeros((n_patches, Nxi_int,  Neta_half))
E1d = N.zeros((n_patches, Nxi_half, Neta_int))
E2d = N.zeros((n_patches, Nxi_int,  Neta_half))
E1d0 = N.zeros((n_patches, Nxi_half, Neta_int))
E2d0 = N.zeros((n_patches, Nxi_int,  Neta_half))
Br0 = N.zeros((n_patches, Nxi_half, Neta_half))
dBrd1 = N.zeros((n_patches, Nxi_int,  Neta_half))
dBrd2 = N.zeros((n_patches, Nxi_half, Neta_int))
dE1d2 = N.zeros((n_patches, Nxi_half, Neta_half))
dE2d1 = N.zeros((n_patches, Nxi_half, Neta_half))

E1d0 = N.zeros((n_patches, Nxi_half, Neta_int))
E2d0 = N.zeros((n_patches, Nxi_int,  Neta_half))
Br0 = N.zeros((n_patches, Nxi_half, Neta_half))

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
# Dissipation
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

Dp = N.zeros((Nxi_int, Nxi_half))

Dp[0, 0:3] = N.array([-1, 1/2, 1/2])
Dp[1, 0:3] = N.array([-1/2, -1/4, 3/4])

Dp[-1, -3:] = N.array([-3/4, 1/4, 1/2])
Dp[-2, -3:] = N.array([-1/2, -1/2, 1])

for i in range(2, Nxi_int-2):
    Dp[i, i:(i+2)] = N.array([-1, 1])

Ap = N.zeros((Nxi_int, Nxi_int))

Ap[0, 0:3] = N.array([2/5, -4/5, 2/5]) / P_half_2[0]
Ap[1, 0:3] = N.array([3/5, -6/5, 3/5])
Ap[2, 0:4] = N.array([1/5, 3/5, -9/5, 1])
# Ap[0, 0:3] = N.array([1, -2, 1]) / P_int_2[0]
# Ap[1, 0:3] = N.array([1, -2, 1])
# Ap[2, 0:4] = N.array([0, 1, -2, 1])


# Ap[-1, -3:] = N.array([1, -2, 1]) / P_int_2[0]
# Ap[-2, -3:] = N.array([1, -2, 1])
# Ap[-3, -4:] = N.array([1, -2, 1, 0])
Ap[-1, -3:] = N.array([2/5, -4/5, 2/5]) / P_int_2[0]
Ap[-2, -3:] = N.array([3/5, -6/5, 3/5])
Ap[-3, -4:] = N.array([1, -9/5, 3/5,1/5])

for i in range(3, Nxi_int-3):
    Ap[i, (i-1):(i+2)] = N.array([1, -2, 1])

Am = N.zeros((Nxi_half, Nxi_half))

# Am[0, 0:3] = N.array([8/3, -4, 4/3]) / P_half_2[0]
# Am[1, 0:3] = N.array([8/3, -4, 4/3]) / P_half_2[1]
# Am[2, 0:4] = N.array([0, 1, -2, 1]) / P_half_2[2]
Am[0, 0:3] = N.array([1/2, -3/4, 1/4]) / P_half_2[0]
Am[1, 0:3] = N.array([1/2, -3/4, 1/4]) / P_half_2[1]
Am[2, 0:4] = N.array([1/2, 1/20, -27/20, 4/5]) / P_half_2[2]

Am[-1, -3:] = N.array([1/4, -3/4, 1/2]) / P_half_2[-1]
Am[-2, -3:] = N.array([1/4, -3/4, 1/2]) / P_half_2[-2]
Am[-3, -4:] = N.array([4/5, -27/20, 1/20, 1/2]) / P_half_2[-3]
# Am[-1, -3:] = N.array([4/3, -4, 8/3]) / P_half_2[-1]
# Am[-2, -3:] = N.array([4/3, -4, 8/3]) / P_half_2[-2]
# Am[-3, -4:] = N.array([1, -2, 1, 0]) / P_half_2[-3]

for i in range(3, Nxi_half-3):
    Am[i, (i-1):(i+2)] = N.array([1, -2, 1])

epsilon = 0.5

def dissipate_E(p, dtin, E1in, E2in):
    
    for j in range(Nxi_half):
        E1in[p, j, :] += dtin * epsilon * N.dot(Ap, E1in[p, j, :]) / dxi
        E2in[p, :, j] += dtin * epsilon * N.dot(Ap, E2in[p, :, j]) / dxi

    for j in range(Nxi_int):
        E1in[p, :, j] += dtin * epsilon * N.dot(Am, E1in[p, :, j]) / dxi
        E2in[p, j, :] += dtin * epsilon * N.dot(Am, E2in[p, j, :]) / dxi

def dissipate_B(p, dtin, Brin):
    
    for j in range(Nxi_half):
        Brin[p, j, :] += dtin * epsilon * N.dot(Am, Brin[p, j, :])
        Brin[p, :, j] += dtin * epsilon * N.dot(Am, Brin[p, :, j])
    
########
# Pushers
########

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

def compute_diff_B_order(p):
  
    dBrd1[p, 0, :] = (-323/378 * Br[p, 0, :] + 2783/3072 * Br[p, 1, :] -1085/27648 * Br[p, 2, :] -17/9216 * Br[p, 3, :] -667/64512 * Br[p, 4, :]) / dxi / P_int_4[0]
    dBrd1[p, 1, :] = (-5/21 * Br[p, 0, :] -847/1024 * Br[p, 1, :] + 1085/1024 * Br[p, 2, :] -37/1024 * Br[p, 3, :] +  899/21504 * Br[p, 4, :]) / dxi / P_int_4[1]
    dBrd1[p, 2, :] = (5/42 * Br[p, 0, :] -121/1024 * Br[p, 1, :] -1085/1024 * Br[p, 2, :] + 3575/3072 * Br[p, 3, :] -753/7168 * Br[p, 4, :]) / dxi / P_int_4[2]
    dBrd1[p, 3, :] = (-5/189 * Br[p, 0, :] + 121/3072 * Br[p, 1, :] + 1085/27648 * Br[p, 2, :] -10759/9216 * Br[p, 3, :] + 74635/64512 * Br[p, 4, :] - 1/24 * Br[p, 5, :]) / dxi / P_int_4[3]

    dBrd1[p, -1, :] = (323/378 * Br[p, -1, :] - 2783/3072 * Br[p, -2, :] + 1085/27648 * Br[p, -3, :] + 17/9216 * Br[p, -4, :] + 667/64512 * Br[p, -5, :]) / dxi / P_int_4[-1]
    dBrd1[p, -2, :] = (5/21 * Br[p, -1, :] + 847/1024 * Br[p, -2, :] - 1085/1024 * Br[p, -3, :] + 37/1024 * Br[p, -4, :] - 899/21504 * Br[p, -5, :]) / dxi / P_int_4[-2]
    dBrd1[p, -3, :] = (- 5/42 * Br[p, -1, :] + 121/1024 * Br[p, -2, :] + 1085/1024 * Br[p, -3, :] - 3575/3072 * Br[p, -4, :] + 753/7168 * Br[p, -5, :]) / dxi / P_int_4[-3]
    dBrd1[p, -4, :] = (5/189 * Br[p, -1, :] - 121/3072 * Br[p, -2, :] - 1085/27648 * Br[p, -3, :] + 10759/9216 * Br[p, -4, :] - 74635/64512 * Br[p, -5, :] + 1/24 * Br[p, -6, :]) / dxi / P_int_4[-4]

    dBrd1[p, 4:(Nxi_int - 4), :] = (N.roll(Br, 1, axis = 1)[p, 4:(Nxi_int - 4), :] - 27.0 * Br[p, 4:(Nxi_int - 4), :] + 27.0 * N.roll(Br, -1, axis = 1)[p, 4:(Nxi_int - 4), :] - N.roll(Br, -2, axis = 1)[p, 4:(Nxi_int - 4), :]) / (24.0 * dxi)

    dBrd2[p, :, 0] = (-323/378 * Br[p, :, 0] + 2783/3072 * Br[p, :, 1] -1085/27648 * Br[p, :, 2] -17/9216 * Br[p, :, 3] -667/64512 * Br[p, :, 4]) / deta / P_int_4[0]
    dBrd2[p, :, 1] = (-5/21 * Br[p, :, 0] -847/1024 * Br[p, :, 1] + 1085/1024 * Br[p, :, 2] -37/1024 * Br[p, :, 3] +  899/21504 * Br[p, :, 4]) / deta / P_int_4[1]
    dBrd2[p, :, 2] = (5/42 * Br[p, :, 0] -121/1024 * Br[p, :, 1] -1085/1024 * Br[p, :, 2] + 3575/3072 * Br[p, :, 3] -753/7168 * Br[p, :, 4]) / deta / P_int_4[2]
    dBrd2[p, :, 3] = (-5/189 * Br[p, :, 0] + 121/3072 * Br[p, :, 1] + 1085/27648 * Br[p, :, 2] -10759/9216 * Br[p, :, 3] + 74635/64512 * Br[p, :, 4] - 1/24 * Br[p, :, 5]) / deta / P_int_4[3]

    dBrd2[p, :, -1] = (323/378 * Br[p, :, -1] - 2783/3072 * Br[p, :, -2] + 1085/27648 * Br[p, :, -3] + 17/9216 * Br[p, :, -4] + 667/64512 * Br[p, :, -5]) / deta / P_int_4[-1]
    dBrd2[p, :, -2] = (5/21 * Br[p, :, -1] + 847/1024 * Br[p, :, -2] - 1085/1024 * Br[p, :, -3] + 37/1024 * Br[p, :, -4] - 899/21504 * Br[p, :, -5]) / deta / P_int_4[-2]
    dBrd2[p, :, -3] = (- 5/42 * Br[p, :, -1] + 121/1024 * Br[p, :, -2] + 1085/1024 * Br[p, :, -3] - 3575/3072 * Br[p, :, -4] + 753/7168 * Br[p, :, -5]) / deta / P_int_4[-3]
    dBrd2[p, :, -4] = (5/189 * Br[p, :, -1] - 121/3072 * Br[p, :, -2] - 1085/27648 * Br[p, :, -3] + 10759/9216 * Br[p, :, -4] - 74635/64512 * Br[p, :, -5] + 1/24 * Br[p, :, -6]) / deta / P_int_4[-4]

    dBrd2[p, :, 4:(Nxi_int - 4)] = (N.roll(Br, 1, axis = 2)[p, :, 4:(Neta_int - 4)] - 27.0 * Br[p, :, 4:(Neta_int - 4)] + 27.0 * N.roll(Br, -1, axis = 2)[p, :, 4:(Neta_int - 4)] - N.roll(Br, -2, axis = 2)[p, :, 4:(Nxi_int - 4)]) / (24.0 * deta)

def compute_diff_B_low(p):

    dBrd1[p, 0, :] = (- 2.0 * Br[p, 0, :] + 2.0 * Br[p, 1, :]) / dxi
    dBrd1[p, Nxi_int - 1, :] = (- 2.0 * Br[p, -2, :] + 2.0 * Br[p, -1, :]) / dxi
    
    dBrd1[p, 1:(Nxi_int - 1), :] = (N.roll(Br, -1, axis = 1)[p, 1:(Nxi_int - 1), :] - Br[p, 1:(Nxi_int - 1), :]) / dxi

    dBrd2[p, :, 0] = (- 2.0 * Br[p, :, 0] + 2.0 * Br[p, :, 1]) / deta
    dBrd2[p, :, Nxi_int - 1] = (- 2.0 * Br[p, :, -2] + 2.0 * Br[p, :, -1]) / deta
    
    dBrd2[p, :, 1:(Neta_int - 1)] = (N.roll(Br, -1, axis = 2)[p, :, 1:(Neta_int - 1)] - Br[p, :, 1:(Neta_int - 1)]) / deta


def compute_diff_E_order(p):
  
    dE2d1[p, 0, :] = (-55/378 * E2d[p, 0, :] + 5/21 * E2d[p, 1, :] -5/42 * E2d[p, 2, :] + 5/189 * E2d[p, 3, :]) / dxi / P_half_4[0]
    dE2d1[p, 1, :] = (-2783/3072 * E2d[p, 0, :] + 847/1024 * E2d[p, 1, :] + 121/1024 * E2d[p, 2, :] -121/3072 * E2d[p, 3, :]) / dxi / P_half_4[1]
    dE2d1[p, 2, :] = (1085/27648 * E2d[p, 0, :] -1085/1024 * E2d[p, 1, :] + 1085/1024 * E2d[p, 2, :] -1085/27648 * E2d[p, 3, :]) / dxi / P_half_4[2]
    dE2d1[p, 3, :] = (17/9216 * E2d[p, 0, :] + 37/1024 * E2d[p, 1, :] -3575/3072 * E2d[p, 2, :] +10759/9216 * E2d[p, 3, :] - 1/24 * E2d[p, 4, :]) / dxi / P_half_4[3]
    dE2d1[p, 4, :] = (667/64512 * E2d[p, 0, :] -899/21504 * E2d[p, 1, :] + 753/7168 * E2d[p, 2, :] -74635/64512 * E2d[p, 3, :] + 9/8 * E2d[p, 4, :] - 1/24 * E2d[p, 5, :]) / dxi / P_half_4[4]

    dE2d1[p, Nxi_half -1, :] = (55/378 * E2d[p, -1, :] - 5/21 * E2d[p, -2, :] + 5/42 * E2d[p, -3, :] - 5/189 * E2d[p, -4, :]) / dxi / P_half_4[-1]
    dE2d1[p, Nxi_half -2, :] = (2783/3072 * E2d[p, -1, :] - 847/1024 * E2d[p, -2, :] - 121/1024 * E2d[p, -3, :] + 121/3072 * E2d[p, -4, :]) / dxi / P_half_4[-2]
    dE2d1[p, Nxi_half -3, :] = (- 1085/27648 * E2d[p, -1, :] + 1085/1024 * E2d[p, -2, :] - 1085/1024 * E2d[p, -3, :] + 1085/27648 * E2d[p, -4, :]) / dxi / P_half_4[-3]
    dE2d1[p, Nxi_half -4, :] = (- 17/9216 * E2d[p, -1, :] - 37/1024 * E2d[p, -2, :] + 3575/3072 * E2d[p, -3, :] - 10759/9216 * E2d[p, -4, :] + 1/24 * E2d[p, -5, :]) / dxi / P_half_4[-4]
    dE2d1[p, Nxi_half -5, :] = (- 667/64512 * E2d[p, -1, :] + 899/21504 * E2d[p, -2, :] - 753/7168 * E2d[p, -3, :] + 74635/64512 * E2d[p, -4, :] - 9/8 * E2d[p, -5, :] + 1/24 * E2d[p, -6, :]) / dxi / P_half_4[-5]

    dE2d1[p, 5:(Nxi_half - 5), :] = (N.roll(E2d, 2, axis = 1)[p, 5:(Nxi_half - 5), :] - 27.0 * N.roll(E2d, 1, axis = 1)[p, 5:(Nxi_half - 5), :] + 27.0 * E2d[p, 5:(Nxi_half - 5), :] - N.roll(E2d, -1, axis = 1)[p, 5:(Nxi_half - 5), :]) / (24.0 * dxi)

    dE1d2[p, :, 0] = (-55/378 * E1d[p, :, 0] + 5/21 * E1d[p, :, 1] -5/42 * E1d[p, :, 2] + 5/189 * E1d[p, :, 3]) / deta / P_half_4[0]
    dE1d2[p, :, 1] = (-2783/3072 * E1d[p, :, 0] + 847/1024 * E1d[p, :, 1] + 121/1024 * E1d[p, :, 2] -121/3072 * E1d[p, :, 3]) / deta / P_half_4[1]
    dE1d2[p, :, 2] = (1085/27648 * E1d[p, :, 0] -1085/1024 * E1d[p, :, 1] + 1085/1024 * E1d[p, :, 2] -1085/27648 * E1d[p, :, 3]) / deta / P_half_4[2]
    dE1d2[p, :, 3] = (17/9216 * E1d[p, :, 0] + 37/1024 * E1d[p, :, 1] -3575/3072 * E1d[p, :, 2] +10759/9216 * E1d[p, :, 3] - 1/24 * E1d[p, :, 4]) / deta / P_half_4[3]
    dE1d2[p, :, 4] = (667/64512 * E1d[p, :, 0] -899/21504 * E1d[p, :, 1] + 753/7168 * E1d[p, :, 2] -74635/64512 * E1d[p, :, 3] + 9/8 * E1d[p, :, 4] - 1/24 * E1d[p, :, 5]) / deta / P_half_4[4]

    dE1d2[p, :, Neta_half -1] = (55/378 * E1d[p, :, -1] - 5/21 * E1d[p, :, -2] + 5/42 * E1d[p, :, -3] - 5/189 * E1d[p, :, -4]) / deta / P_half_4[-1]
    dE1d2[p, :, Neta_half -2] = (2783/3072 * E1d[p, :, -1] - 847/1024 * E1d[p, :, -2] - 121/1024 * E1d[p, :, -3] + 121/3072 * E1d[p, :, -4]) / deta / P_half_4[-2]
    dE1d2[p, :, Neta_half -3] = (- 1085/27648 * E1d[p, :, -1] + 1085/1024 * E1d[p, :, -2] - 1085/1024 * E1d[p, :, -3] + 1085/27648 * E1d[p, :, -4]) / deta / P_half_4[-3]
    dE1d2[p, :, Neta_half -4] = (- 17/9216 * E1d[p, :, -1] - 37/1024 * E1d[p, :, -2] + 3575/3072 * E1d[p, :, -3] -10759/9216 * E1d[p, :, -4] + 1/24 * E1d[p, :, -5]) / deta / P_half_4[-4]
    dE1d2[p, :, Neta_half -5] = (- 667/64512 * E1d[p, :, -1] + 899/21504 * E1d[p, :, -2] - 753/7168 * E1d[p, :, -3] + 74635/64512 * E1d[p, :, -4] - 9/8 * E1d[p, :, -5] + 1/24 * E1d[p, :, -6]) / deta / P_half_4[-5]

    dE1d2[p, :, 5:(Neta_half - 5)] = (N.roll(E1d, 2, axis = 2)[p, :, 5:(Neta_half - 5)] - 27.0 * N.roll(E1d, 1, axis = 2)[p, :, 5:(Neta_half - 5)] + 27.0 * E1d[p, :, 5:(Neta_half - 5)] - N.roll(E1d, -1, axis = 2)[p, :, 5:(Neta_half - 5)]) / (24.0 * deta)


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

    dE2d1[p, 0, :] = (- 0.5 * E2d[p, 0, :] + 0.5 * E2d[p, 1, :]) / dxi / P_half_2[0]
    dE2d1[p, 1, :] = (- 0.25 * E2d[p, 0, :] + 0.25 * E2d[p, 1, :]) / dxi / P_half_2[1]

    dE2d1[p, Nxi_half - 2, :] = (- 0.25 * E2d[p, -2, :] + 0.25 * E2d[p, -1, :]) / dxi / P_half_2[Nxi_half - 2]
    dE2d1[p, Nxi_half - 1, :] = (- 0.5 * E2d[p, -2, :] + 0.5 * E2d[p, -1, :]) / dxi / P_half_2[Nxi_half - 1]

    dE2d1[p, 2:(Nxi_half - 2), :] = (E2d[p, 2:(Nxi_half - 2), :] - N.roll(E2d, 1, axis = 1)[p, 2:(Nxi_half - 2), :]) / dxi

    dE1d2[p, :, 0] = (- 0.5 * E1d[p, :, 0] + 0.5 * E1d[p, :, 1]) / dxi / P_half_2[0]
    dE1d2[p, :, 1] = (- 0.25 * E1d[p, :, 0] + 0.25 * E1d[p, :, 1]) / dxi / P_half_2[1]

    dE1d2[p, :, Neta_half - 2] = (- 0.25 * E1d[p, :, -2] + 0.25 * E1d[p, :, -1]) / deta / P_half_2[Nxi_half - 2]
    dE1d2[p, :, Neta_half - 1] = (- 0.5 * E1d[p, :, -2] + 0.5 * E1d[p, :, -1]) / deta / P_half_2[Nxi_half - 1]

    dE1d2[p, :, 2:(Neta_half - 2)] = (E1d[p, :, 2:(Neta_half - 2)] - N.roll(E1d, 1, axis = 2)[p, :, 2:(Neta_half - 2)]) / deta

def compute_diff_E_low(p):

    dE2d1[p, 0, :] = (- E2d[p, 0, :] + E2d[p, 1, :]) / dxi
    dE2d1[p, 1, :] = (- E2d[p, 0, :] + E2d[p, 1, :]) / dxi

    dE2d1[p, Nxi_half - 2, :] = (- E2d[p, -2, :] + E2d[p, -1, :]) / dxi
    dE2d1[p, Nxi_half - 1, :] = (- E2d[p, -2, :] + E2d[p, -1, :]) / dxi

    dE2d1[p, 2:(Nxi_half - 2), :] = (E2d[p, 2:(Nxi_half - 2), :] - N.roll(E2d, 1, axis = 1)[p, 2:(Nxi_half - 2), :]) / dxi

    dE1d2[p, :, 0] = (- E1d[p, :, 0] + E1d[p, :, 1]) / dxi
    dE1d2[p, :, 1] = (- E1d[p, :, 0] + E1d[p, :, 1]) / dxi

    dE1d2[p, :, Neta_half - 2] = (- E1d[p, :, -2] + E1d[p, :, -1]) / deta
    dE1d2[p, :, Neta_half - 1] = (- E1d[p, :, -2] + E1d[p, :, -1]) / deta

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

def push_B(p, itime, dtin):
        
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
        Br[p, :, :] += dtin * Jz[p, :, :] * N.sin(20.0 * itime * dtin) * (1 + N.tanh(20 - itime/5.))/2.

def push_E(p, itime, dtin):

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

def compute_energy():
    
    E1u_int = N.zeros((n_patches, Nxi_int, Neta_int))
    E1u_int[:, 1:-1, :] = 0.5 * (E1u[:, 1:-2, :] + N.roll(E1u, -1, axis = 1)[:, 1:-2, :])
    E1u_int[:, 0, :] = E1u[:, 0, :]
    E1u_int[:, -1, :] = E1u[:, -1, :]

    E2u_int = N.zeros((n_patches, Nxi_int, Neta_int))
    E2u_int[:, :, 1:-1] = 0.5 * (E2u[:, :, 1:-2] + N.roll(E2u, -1, axis = 2)[:, :, 1:-2])
    E2u_int[:, :,  0] = E2u[:, :, 0]
    E2u_int[:, :, -1] = E2u[:, :, -1]

    E1d_int = N.zeros((n_patches, Nxi_int, Neta_int))
    E1d_int[:, 1:-1, :] = 0.5 * (E1d[:, 1:-2, :] + N.roll(E1d, -1, axis = 1)[:, 1:-2, :])
    E1d_int[:, 0, :] = E1d[:, 0, :]
    E1d_int[:, -1, :] = E1d[:, -1, :]

    E2d_int = N.zeros((n_patches, Nxi_int, Neta_int))
    E2d_int[:, :, 1:-1] = 0.5 * (E2d[:, :, 1:-2] + N.roll(E2d, -1, axis = 2)[:, :, 1:-2])
    E2d_int[:, :,  0] = E2d[:, :, 0]
    E2d_int[:, :, -1] = E2d[:, :, -1]

    energy_e = N.sum(E1d_int[:, 1:-1, :] * E1u_int[:, 1:-1, :] + E2d_int[:, 1:-1, :] * E2u_int[:, 1:-1, :], axis = 1)

    # energy_e = spi.simps(spi.simps(E1d_int * E1u_int + E2d_int * E2u_int, x = xi_int, axis = 1), x = eta_int, axis = 1)
    energy_b = spi.simps(spi.simps(Br * Br, x = xi_half, axis = 1), x = eta_half, axis = 1)
    return energy_e + energy_b
    
########
# Boundary conditions
########

def interp(arr_in, xA, xB):
    return N.interp(xB, xA, arr_in)

sqrt_gd_half = N.sqrt(g22d_half)

# Interface inner boundary 
sig_in  = 1.0
sig_cor = sig_in

def compute_delta_B(p0, p1, dtin, E1in, E2in, Brin):

    i0 = 0
    i1 = Nxi_half

    top = topology[p0, p1]
    
    if (top == 'xx'):

        Eeta_0 = E2in[p0, -1, :]
        Eeta_1 = E2in[p1, 0, :]
        Br_0 = Brin[p0, -1, :]
        Br_1 = Brin[p1, 0, :]

        diff_Br[p0, -1, i0:i1] += dtin * sig_in * 0.5 * ((- Eeta_0[i0:i1] + sqrt_gd_half[i0:i1] * Br_0[i0:i1]) - \
                                                         (- Eeta_1[i0:i1] + sqrt_gd_half[i0:i1] * Br_1[i0:i1])) / dxi / P_int_2[0]
        diff_Br[p1, 0, i0:i1]  += dtin * sig_in * 0.5 * ((  Eeta_1[i0:i1] + sqrt_gd_half[i0:i1] * Br_1[i0:i1]) - \
                                                         (  Eeta_0[i0:i1] + sqrt_gd_half[i0:i1] * Br_0[i0:i1])) / dxi / P_int_2[0]

        # diff_Br[p0, -1, i0:i1] += dtin * sig_in * 0.5 * ((- Eeta_0[i0:i1] + Br_0[i0:i1]) - \
        #                                                  (- Eeta_1[i0:i1] + Br_1[i0:i1])) / dxi / P_int_2[0]
        # diff_Br[p1, 0, i0:i1]  += dtin * sig_in * 0.5 * ((  Eeta_1[i0:i1] + Br_1[i0:i1]) - \
        #                                                  (  Eeta_0[i0:i1] + Br_0[i0:i1])) / dxi / P_int_2[0]


    if (top == 'xy'):

        Eeta_0 = E2in[p0, -1, :]
        Exi_1  = E1in[p1, :, 0]
        Br_0 = Br[p0, -1, :]
        Br_1 = Br[p1, :, 0]
        
        diff_Br[p0, -1, i0:i1] += dtin * sig_in * 0.5 * ((- Eeta_0[i0:i1] + sqrt_gd_half[i0:i1] * Br_0[i0:i1]) - \
                                                         ((Exi_1[::-1])[i0:i1] + sqrt_gd_half[i0:i1] * (Br_1[::-1])[i0:i1])) / dxi / P_int_2[0]
        diff_Br[p1, i0:i1, 0]  += dtin * sig_in * 0.5 * ((- Exi_1[i0:i1] + sqrt_gd_half[i0:i1] * Br_1[i0:i1]) - \
                                                         ((Eeta_0[::-1])[i0:i1] + sqrt_gd_half[i0:i1] * (Br_0[::-1])[i0:i1])) / dxi / P_int_2[0]

        # diff_Br[p0, -1, i0:i1] += dtin * sig_in * 0.5 * ((- Eeta_0[i0:i1] + Br_0[i0:i1]) - \
        #                                                  ((Exi_1[::-1])[i0:i1] + (Br_1[::-1])[i0:i1])) / dxi / P_int_2[0]
        # diff_Br[p1, i0:i1, 0]  += dtin * sig_in * 0.5 * ((- Exi_1[i0:i1] + Br_1[i0:i1]) - \
        #                                                  ((Eeta_0[::-1])[i0:i1] + (Br_0[::-1])[i0:i1])) / dxi / P_int_2[0]


    if (top == 'yy'):

        Exi_0 = E1in[p0, :, -1]
        Exi_1 = E1in[p1, :, 0]
        Br_0 = Brin[p0, :, -1]
        Br_1 = Brin[p1, :, 0]
        
        diff_Br[p0, i0:i1, -1] += dtin * sig_in * 0.5 * ((Exi_0[i0:i1] + sqrt_gd_half[i0:i1] * Br_0[i0:i1]) - \
                                                         (Exi_1[i0:i1] + sqrt_gd_half[i0:i1] * Br_1[i0:i1])) / deta / P_int_2[0]
        diff_Br[p1, i0:i1, 0]  += dtin * sig_in * 0.5 * ((- Exi_1[i0:i1] + sqrt_gd_half[i0:i1] * Br_1[i0:i1]) - \
                                                         (- Exi_0[i0:i1] + sqrt_gd_half[i0:i1] * Br_0[i0:i1])) / deta / P_int_2[0]

        # diff_Br[p0, i0:i1, -1] += dtin * sig_in * 0.5 * ((Exi_0[i0:i1] + Br_0[i0:i1]) - \
        #                                                  (Exi_1[i0:i1] + Br_1[i0:i1])) / deta / P_int_2[0]
        # diff_Br[p1, i0:i1, 0]  += dtin * sig_in * 0.5 * ((- Exi_1[i0:i1] + Br_1[i0:i1]) - \
        #                                                  (- Exi_0[i0:i1] + Br_0[i0:i1])) / deta / P_int_2[0]

    if (top == 'yx'):

        Exi_0 = E1in[p0, :, -1]
        Eeta_1 = E2in[p1, 0, :]
        Br_0 = Brin[p0, :, -1]
        Br_1 = Brin[p1, 0, :]

        diff_Br[p0, i0:i1, -1] += dtin * sig_in * 0.5 * ((Exi_0[i0:i1] + sqrt_gd_half[i0:i1] * Br_0[i0:i1]) - \
                                                         (- (Eeta_1[::-1])[i0:i1] + sqrt_gd_half[i0:i1] * (Br_1[::-1])[i0:i1])) / deta / P_int_2[0]
        diff_Br[p1, 0, i0:i1]  += dtin * sig_in * 0.5 * ((Eeta_1[i0:i1] + sqrt_gd_half[i0:i1] * Br_1[i0:i1]) - \
                                                         (- (Exi_0[::-1])[i0:i1] + sqrt_gd_half[i0:i1] * (Br_0[::-1])[i0:i1])) / deta / P_int_2[0]

        # diff_Br[p0, i0:i1, -1] += dtin * sig_in * 0.5 * ((Exi_0[i0:i1] + Br_0[i0:i1]) - \
        #                                                  (- (Eeta_1[::-1])[i0:i1] + (Br_1[::-1])[i0:i1])) / deta / P_int_2[0]
        # diff_Br[p1, 0, i0:i1]  += dtin * sig_in * 0.5 * ((Eeta_1[i0:i1] + Br_1[i0:i1]) - \
        #                                                  (- (Exi_0[::-1])[i0:i1] + (Br_0[::-1])[i0:i1])) / deta / P_int_2[0]


def interface_B(p0, p1):

    i0 = 1
    i1 = Nxi_half - 1

    top = topology[p0, p1]
    
    if (top == 'xx'):
        Br[p0, -1, i0:i1] -= diff_Br[p0, -1, i0:i1] / sqrt_det_g_half[i0:i1]
        Br[p1, 0, i0:i1]  -= diff_Br[p1, 0, i0:i1]  / sqrt_det_g_half[i0:i1]

    if (top == 'xy'):
        Br[p0, -1, i0:i1] -= diff_Br[p0, -1, i0:i1] / sqrt_det_g_half[i0:i1]
        Br[p1, i0:i1, 0]  -= diff_Br[p1, i0:i1, 0]  / sqrt_det_g_half[i0:i1]

    if (top == 'yy'):
        Br[p0, i0:i1, -1] -= diff_Br[p0, i0:i1, -1] / sqrt_det_g_half[i0:i1]
        Br[p1, i0:i1, 0]  -= diff_Br[p1, i0:i1, 0]  / sqrt_det_g_half[i0:i1]

    if (top == 'yx'):
        Br[p0, i0:i1, -1] -= diff_Br[p0, i0:i1, -1] / sqrt_det_g_half[i0:i1]
        Br[p1, 0, i0:i1]  -= diff_Br[p1, 0, i0:i1]  / sqrt_det_g_half[i0:i1]

def corners_B(p0):

    Br[p0, 0, 0]   -= diff_Br[p0, 0, 0] * sig_cor / sig_in / sqrt_det_g_half[0]
    Br[p0, -1, 0]  -= diff_Br[p0, -1, 0] * sig_cor / sig_in  / sqrt_det_g_half[0]
    Br[p0, 0, -1]  -= diff_Br[p0, 0, -1] * sig_cor / sig_in  / sqrt_det_g_half[0]
    Br[p0, -1, -1] -= diff_Br[p0, -1, -1] * sig_cor / sig_in  / sqrt_det_g_half[0]

def compute_delta_E(p0, p1, dtin, E1in, E2in, Brin):

    i0 = 0
    i1 = Nxi_half

    top = topology[p0, p1]
    
    if (top == 'xx'):

        Eeta_0 = E2in[p0, -1, :]
        Eeta_1 = E2in[p1, 0, :]
        Br_0 = Brin[p0, -1, :]
        Br_1 = Brin[p1, 0, :]
        
        diff_E2[p0, -1, i0:i1] += dtin * sig_in * 0.5 * ((Eeta_0[i0:i1] / sqrt_gd_half[i0:i1] - Br_0[i0:i1]) - \
                                                         (Eeta_1[i0:i1] / sqrt_gd_half[i0:i1] - Br_1[i0:i1])) / dxi / P_int_2[0]
        diff_E2[p1, 0, i0:i1]  += dtin * sig_in * 0.5 * ((Eeta_1[i0:i1] / sqrt_gd_half[i0:i1] + Br_1[i0:i1]) - \
                                                         (Eeta_0[i0:i1] / sqrt_gd_half[i0:i1] + Br_0[i0:i1])) / dxi / P_int_2[0]

        # diff_E2[p0, -1, i0:i1] += dtin * sig_in * 0.5 * ((Eeta_0[i0:i1] - Br_0[i0:i1]) - \
        #                                                  (Eeta_1[i0:i1] - Br_1[i0:i1])) / dxi / P_int_2[0]
        # diff_E2[p1, 0, i0:i1]  += dtin * sig_in * 0.5 * ((Eeta_1[i0:i1] + Br_1[i0:i1]) - \
        #                                                  (Eeta_0[i0:i1] + Br_0[i0:i1])) / dxi / P_int_2[0]


    if (top == 'xy'):

        Eeta_0 = E2in[p0, -1, :]
        Exi_1  = E1in[p1, :, 0]
        Br_0 = Br[p0, -1, :]
        Br_1 = Br[p1, :, 0]

        diff_E2[p0, -1, i0:i1] += dtin * sig_in * 0.5 * ((Eeta_0[i0:i1] / sqrt_gd_half[i0:i1] - Br_0[i0:i1]) - \
                                                         (- (Exi_1[::-1])[i0:i1] / sqrt_gd_half[i0:i1] - (Br_1[::-1])[i0:i1])) / dxi / P_int_2[0]
        diff_E1[p1, i0:i1, 0]  += dtin * sig_in * 0.5 * ((Exi_1[i0:i1] / sqrt_gd_half[i0:i1] - Br_1[i0:i1]) - \
                                                         (- (Eeta_0[::-1])[i0:i1] / sqrt_gd_half[i0:i1] - (Br_0[::-1])[i0:i1])) / dxi / P_int_2[0]

        # diff_E2[p0, -1, i0:i1] += dtin * sig_in * 0.5 * ((Eeta_0[i0:i1] - Br_0[i0:i1]) - \
        #                                                  (- (Exi_1[::-1])[i0:i1] - (Br_1[::-1])[i0:i1])) / dxi / P_int_2[0]
        # diff_E1[p1, i0:i1, 0]  += dtin * sig_in * 0.5 * ((Exi_1[i0:i1] - Br_1[i0:i1]) - \
        #                                                  (- (Eeta_0[::-1])[i0:i1] - (Br_0[::-1])[i0:i1])) / dxi / P_int_2[0]
        
    if (top == 'yy'):

        Exi_0 = E1in[p0, :, -1]
        Exi_1 = E1in[p1, :, 0]
        Br_0 = Brin[p0, :, -1]
        Br_1 = Brin[p1, :, 0]

        diff_E1[p0, i0:i1, -1] += dtin * sig_in * 0.5 * ((Exi_0[i0:i1] / sqrt_gd_half[i0:i1] + Br_0[i0:i1]) - \
                                                         (Exi_1[i0:i1] / sqrt_gd_half[i0:i1] + Br_1[i0:i1])) / deta / P_int_2[0]
        diff_E1[p1, i0:i1, 0]  += dtin * sig_in * 0.5 * ((Exi_1[i0:i1] / sqrt_gd_half[i0:i1] - Br_1[i0:i1]) - \
                                                         (Exi_0[i0:i1] / sqrt_gd_half[i0:i1] - Br_0[i0:i1])) / deta / P_int_2[0]

        # diff_E1[p0, i0:i1, -1] += dtin * sig_in * 0.5 * ((Exi_0[i0:i1] + Br_0[i0:i1]) - \
        #                                                  (Exi_1[i0:i1] + Br_1[i0:i1])) / deta / P_int_2[0]
        # diff_E1[p1, i0:i1, 0]  += dtin * sig_in * 0.5 * ((Exi_1[i0:i1] - Br_1[i0:i1]) - \
        #                                                  (Exi_0[i0:i1] - Br_0[i0:i1])) / deta / P_int_2[0]

    if (top == 'yx'):

        Exi_0 = E1in[p0, :, -1]
        Eeta_1 = E2in[p1, 0, :]
        Br_0 = Brin[p0, :, -1]
        Br_1 = Brin[p1, 0, :]

        diff_E1[p0, i0:i1, -1] += dtin * sig_in * 0.5 * ((Exi_0[i0:i1] / sqrt_gd_half[i0:i1] + Br_0[i0:i1]) - \
                                                         (- (Eeta_1[::-1])[i0:i1] / sqrt_gd_half[i0:i1] + (Br_1[::-1])[i0:i1])) / deta / P_int_2[0]
        diff_E2[p1, 0, i0:i1]  += dtin * sig_in * 0.5 * ((Eeta_1[i0:i1] / sqrt_gd_half[i0:i1] + Br_1[i0:i1]) - \
                                                         (- (Exi_0[::-1])[i0:i1] / sqrt_gd_half[i0:i1] + (Br_0[::-1])[i0:i1])) / deta / P_int_2[0]

        # diff_E1[p0, i0:i1, -1] += dtin * sig_in * 0.5 * ((Exi_0[i0:i1] + Br_0[i0:i1]) - \
        #                                                  (- (Eeta_1[::-1])[i0:i1] + (Br_1[::-1])[i0:i1])) / deta / P_int_2[0]
        # diff_E2[p1, 0, i0:i1]  += dtin * sig_in * 0.5 * ((Eeta_1[i0:i1] + Br_1[i0:i1]) - \
        #                                                  (- (Exi_0[::-1])[i0:i1] + (Br_0[::-1])[i0:i1])) / deta / P_int_2[0]

def interface_E(p0, p1):

    i0 =  1
    i1 = Nxi_half - 1

    top = topology[p0, p1]
    
    if (top == 'xx'):
        E2u[p0, -1, i0:i1] -= diff_E2[p0, -1, i0:i1] / sqrt_det_g_half[i0:i1]
        E2u[p1, 0, i0:i1]  -= diff_E2[p1, 0, i0:i1] / sqrt_det_g_half[i0:i1]

    if (top == 'xy'):
        E2u[p0, -1, i0:i1] -= diff_E2[p0, -1, i0:i1] / sqrt_det_g_half[i0:i1]
        E1u[p1, i0:i1, 0]  -= diff_E1[p1, i0:i1, 0] / sqrt_det_g_half[i0:i1] 

    if (top == 'yy'):
        E1u[p0, i0:i1, -1] -= diff_E1[p0, i0:i1, -1] / sqrt_det_g_half[i0:i1]
        E1u[p1, i0:i1, 0]  -= diff_E1[p1, i0:i1, 0] / sqrt_det_g_half[i0:i1] 

    if (top == 'yx'):
        E1u[p0, i0:i1, -1] -= diff_E1[p0, i0:i1, -1] / sqrt_det_g_half[i0:i1]
        E2u[p1, 0, i0:i1]  -= diff_E2[p1, 0, i0:i1] / sqrt_det_g_half[i0:i1]

def corners_E(p0):

    E1u[p0, 0, 0]   -= diff_E1[p0, 0, 0] * sig_cor / sig_in / sqrt_det_g_half[0]
    E1u[p0, -1, 0]  -= diff_E1[p0, -1, 0] * sig_cor / sig_in / sqrt_det_g_half[0]
    E1u[p0, 0, -1]  -= diff_E1[p0, 0, -1] * sig_cor / sig_in / sqrt_det_g_half[0]
    E1u[p0, -1, -1] -= diff_E1[p0, -1, -1] * sig_cor / sig_in / sqrt_det_g_half[0]

    E2u[p0, 0, 0]   -= diff_E2[p0, 0, 0] * sig_cor / sig_in / sqrt_det_g_half[0]
    E2u[p0, -1, 0]  -= diff_E2[p0, -1, 0] * sig_cor / sig_in / sqrt_det_g_half[0]
    E2u[p0, 0, -1]  -= diff_E2[p0, 0, -1] * sig_cor / sig_in / sqrt_det_g_half[0]
    E2u[p0, -1, -1] -= diff_E2[p0, -1, -1] * sig_cor / sig_in / sqrt_det_g_half[0]


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

    Br0 = Br[p0, i0, j0]
    Br1 = Br[p1, i1, j1]
    Br2 = Br[p2, i2, j2]
    mean0 = (Br0 + Br1 + Br2) / 3.0

    Bra = Br[p0, i0p, j0]
    Brb = Br[p0, i0, j0p]
    Brc = Br[p0, i0p, j0p]
    Brd = Br[p1, i1p, j1]
    Bre = Br[p1, i1, j1p]
    Brf = Br[p1, i1p, j1p]
    Brg = Br[p2, i2p, j2]
    Brh = Br[p2, i2, j2p]
    Bri = Br[p2, i2p, j2p]
    # mean = (0.5 * Bra + 0.5 * Brb + Brc + 0.5 * Brd + 0.5 * Bre + Brf + 0.5 * Brg + 0.5 * Brh + Bri) / 6.0
    mean = (Brc + Brf + Bri) / 3.0
    # mean = (Bra + Brb + Brd + Bre + Brg + Brh) / 6.0
    # mean = (Bra + Brb + Brd + Bre + Brg + Brh + Br0 + Br1 + Br2) / 9.0

    # Br[p0, i0, j0] -= dtin * sig_fil * (Br[p0, i0, j0] - mean0)
    # Br[p1, i1, j1] -= dtin * sig_fil * (Br[p1, i1, j1] - mean0)
    # Br[p2, i2, j2] -= dtin * sig_fil * (Br[p2, i2, j2] - mean0)
    
    Br[p0, i0, j0] -= dtin * sig_fil * (Br[p0, i0, j0] - mean)
    Br[p1, i1, j1] -= dtin * sig_fil * (Br[p1, i1, j1] - mean)
    Br[p2, i2, j2] -= dtin * sig_fil * (Br[p2, i2, j2] - mean)

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

    E1m = 0.5 * (E1u[p0, i0, j0] + E1u[p0, i0, j0p])    
    E2m = 0.5 * (E2u[p0, i0, j0] + E2u[p0, i0p, j0])
    Etha, Epha = fvec0(xi_half[i0p], eta_int[j0], E1u[p0, i0p, j0], E2m)
    Ethb, Ephb = fvec0(xi_int[i0], eta_half[j0p], E1m, E2u[p0, i0, j0p])
    E1m = 0.5 * (E1u[p1, i1, j1] + E1u[p1, i1, j1p])    
    E2m = 0.5 * (E2u[p1, i1, j1] + E2u[p1, i1p, j1])
    Ethc, Ephc = fvec1(xi_half[i1p], eta_int[j1], E1u[p1, i1p, j1], E2m)
    Ethd, Ephd = fvec1(xi_int[i1], eta_half[j1p], E1m, E2u[p1, i1, j1p])
    E1m = 0.5 * (E1u[p2, i2, j2] + E1u[p2, i2, j2p])    
    E2m = 0.5 * (E2u[p2, i2, j2] + E2u[p2, i2p, j2])
    Ethe, Ephe = fvec2(xi_half[i2p], eta_int[j2], E1u[p2, i2p, j2], E2m)
    Ethf, Ephf = fvec2(xi_int[i2], eta_half[j2p], E1m, E2u[p2, i2, j2p])

    Ethm = (Etha + Ethb + Ethc + Ethd + Ethe + Ethf) / 6.0
    Ephm = (Epha + Ephb + Ephc + Ephd + Ephe + Ephf) / 6.0

    Eth0, Eph0 = fvec0(xi_int[i0], eta_int[j0], E1u[p0, i0, j0], E2u[p0, i0, j0])
    Eth1, Eph1 = fvec1(xi_int[i1], eta_int[j1], E1u[p1, i1, j1], E2u[p1, i1, j1])
    Eth2, Eph2 = fvec2(xi_int[i2], eta_int[j2], E1u[p2, i2, j2], E2u[p2, i2, j2])
    Ethmm = (Eth0 + Eth1 + Eth2) / 3.0
    Ephmm = (Eph0 + Eph1 + Eph2) / 3.0
    E10, E20 = fvec0i(th0, ph0, Ethmm, Ephmm)
    E11, E21 = fvec1i(th0, ph0, Ethmm, Ephmm)
    E12, E22 = fvec2i(th0, ph0, Ethmm, Ephmm)

    # Ethm = (Etha + Ethb + Ethc + Ethd + Ethe + Ethf + Eth0 + Eth1 + Eth2) / 9.0
    # Ephm = (Epha + Ephb + Ephc + Ephd + Ephe + Ephf + Eph0 + Eph1 + Eph2) / 9.0
    
    Exi0, Eeta0 = fvec0i(th0, ph0, Ethm, Ephm)
    Exi1, Eeta1 = fvec1i(th0, ph0, Ethm, Ephm)
    Exi2, Eeta2 = fvec2i(th0, ph0, Ethm, Ephm)
    
    # Exi0, Eeta0 = fvec0i(th0, ph0, Ethmm, Ephmm)
    # Exi1, Eeta1 = fvec1i(th0, ph0, Ethmm, Ephmm)
    # Exi2, Eeta2 = fvec2i(th0, ph0, Ethmm, Ephmm)

    E1u[p0, i0, j0] -= sig_fil * dtin * (E1u[p0, i0, j0] - Exi0)
    E2u[p0, i0, j0] -= sig_fil * dtin * (E2u[p0, i0, j0] - Eeta0)
    E1u[p1, i1, j1] -= sig_fil * dtin * (E1u[p1, i1, j1] - Exi1)
    E2u[p1, i1, j1] -= sig_fil * dtin * (E2u[p1, i1, j1] - Eeta1)
    E1u[p2, i2, j2] -= sig_fil * dtin * (E1u[p2, i2, j2] - Exi2)
    E2u[p2, i2, j2] -= sig_fil * dtin * (E2u[p2, i2, j2] - Eeta2)


########
# Initialization
########

amp = 1.0
n_mode = 2
wave = 2.0 * (xi_max - xi_min) / n_mode
E1ui = N.zeros_like(E1u)
E2ui = N.zeros_like(E2u)

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

            EtTMP = 0.0 # amp * N.sin(th0)**3 * N.cos(3.0 * ph0)
            EpTMP = 0.0

            ECStmp = fvec(th0, ph0, EtTMP, EpTMP)

            E1u[patch, i, j] = ECStmp[0]
            E1ui[patch, i, j] = ECStmp[0]

    for i in range(Nxi_int):
        for j in range(Neta_half):

            th0, ph0 = fcoord(xE2_grid[i, j], yE2_grid[i, j])

            EtTMP = 0.0 # amp * N.sin(th0)**3 * N.cos(3.0 * ph0)
            EpTMP = 0.0

            ECStmp = fvec(th0, ph0, EtTMP, EpTMP)

            E2u[patch, i, j] = ECStmp[1]
            E2ui[patch, i, j] = ECStmp[1]

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

def plot_fields_unfolded_Br(it, vm):

    fig = P.figure(1, facecolor='w')
    ax = P.subplot(111)

    ax.pcolormesh(xBr_grid, yBr_grid, Br[Sphere.A, :, :], cmap = "RdBu_r", vmin = - vm, vmax = vm)
    ax.pcolormesh(xBr_grid + N.pi / 2.0, yBr_grid, Br[Sphere.B, :, :], cmap = "RdBu_r", vmin = - vm, vmax = vm)
    ax.pcolormesh(xBr_grid, yBr_grid - N.pi / 2.0, Br[Sphere.S, :, :], cmap = "RdBu_r", vmin = - vm, vmax = vm)

    ax.pcolormesh(xi_grid_c + N.pi, eta_grid_c, Br[Sphere.C, :, :], cmap = "RdBu_r", vmin = - vm, vmax = vm)
    ax.pcolormesh(xi_grid_d - N.pi / 2.0, eta_grid_d, Br[Sphere.D, :, :], cmap = "RdBu_r", vmin = - vm, vmax = vm)
    ax.pcolormesh(xi_grid_n, eta_grid_n + N.pi / 2.0, Br[Sphere.N, :, :], cmap = "RdBu_r", vmin = - vm, vmax = vm)
    
    P.title(r'$t={:.3f} R/c$'.format(FDUMP*it*dt))
    
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

Nt = 500000 # Number of iterations
FDUMP = 1000 # Dump frequency
time = dt * N.arange(Nt)
energy = N.zeros((n_patches, Nt))

# Initialize half time step

compute_diff_E(patch)
push_B(patch, 0, 0.5 * dt)
for i in range(n_zeros):
    p0, p1 = index_row[i], index_col[i]
    compute_delta_B(p0, p1, dt, E1d, E2d, Br)
for i in range(n_zeros):
    p0, p1 = index_row[i], index_col[i]
    interface_B(p0, p1)
corners_B(patch)

for it in tqdm(range(Nt), "Progression"):
    if ((it % FDUMP) == 0):

        # Dump data
        h5f = h5py.File('snapshots_penalty/Br_{}.h5'.format(idump), 'w')
        h5f.create_dataset('Br', data=Br[:,:,:])
        h5f.close()

        plot_fields_unfolded_Br(idump, 1.0)
        idump += 1

    diff_Br[:, :, :] = 0.0
    diff_E1[:, :, :] = 0.0
    diff_E2[:, :, :] = 0.0

    Br0[:, :, :] = Br[:, :, :]    
    E1d0[:, :, :] = E1d[:, :, :]
    E2d0[:, :, :] = E2d[:, :, :]

    compute_diff_B(patch)
    # compute_diff_B_alt(patch)
    # compute_diff_B_order(patch)

    push_E(patch, it, dt)
    
    for p0 in patch:
        dissipate_E(p0, dt, E1u, E2u)

    for i in range(n_zeros):
        p0, p1 = index_row[i], index_col[i]
        # compute_delta_E(p0, p1, dt, E1d0, E2d0, Br)
        compute_delta_E(p0, p1, dt, E1d, E2d, Br)

        interface_E(p0, p1)

    corners_E(patch)

    # ##### FILTERING
    # for i in range(n_corners):
    #     filter_E_corner(i, dt)

    contra_to_cov_E(patch)
    # contra_to_cov_E_weights(patch)

    compute_diff_E(patch)
    # compute_diff_E_alt(patch)
    # compute_diff_E_order(patch)
    
    push_B(patch, it, dt)

    for p0 in patch:
        dissipate_B(p0, dt, Br)

    for i in range(n_zeros):
        p0, p1 = index_row[i], index_col[i]
        compute_delta_B(p0, p1, dt, E1d, E2d, Br0)
        # compute_delta_B(p0, p1, dt, E1d, E2d, Br)

        interface_B(p0, p1)

    corners_B(patch)

    for p in range(n_patches):
        energy[p, it] = dxi * deta * (N.sum(Br[p, :, :]**2) \
        + N.sum(E1u[p, :, :]**2) + N.sum(E2u[p, :, :]**2))
        # energy[:, it] = compute_energy()
    
# for p in range(n_patches):
#     P.plot(time, energy[p, :])


