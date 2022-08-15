# Import modules
import numpy as N
import matplotlib.pyplot as P
import matplotlib
from skimage.measure import find_contours
from math import *
from tqdm import tqdm
import matplotlib.pyplot as plt
from scipy import interpolate
import scipy.integrate as spi

# Import my figure routines
from figure_module import *

import warnings
import matplotlib.cbook
warnings.filterwarnings("ignore",category=matplotlib.cbook.mplDeprecation)

# Connect patch indices and names
sphere = {0: "A", 1: "B", 2: "C", 3: "D", 4: "N", 5: "S"}

class Sphere:
    A = 0
    B = 1
    C = 2
    D = 3
    N = 4
    S = 5

# Parameters
cfl = 0.5
Nxi  = 128 # Number of cells in xi
Neta = 128 # Number of cells in eta

Nxi_int = Nxi + 1 # Number of integer points
Nxi_half = Nxi + 2 # NUmber of hlaf-step points
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

n_patches = 2

########
# Define metric tensor
########

g11d = N.empty((Nxi_int, Neta_int, 4))
g12d = N.empty((Nxi_int, Neta_int, 4))
g22d = N.empty((Nxi_int, Neta_int, 4))
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

g11u = N.empty((Nxi_int, Neta_int, 4))
g12u = N.empty((Nxi_int, Neta_int, 4))
g22u = N.empty((Nxi_int, Neta_int, 4))

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

dt = cfl * N.min(1.0 / N.sqrt(g11u / (dxi * dxi) + g22u / (deta * deta) + 2.0 * g12u / (dxi * deta)))
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

def compute_diff_B_low(p):

    dBrd1[p, 0, :] = (- 2.0 * Br[p, 0, :] + 2.0 * Br[p, 1, :]) / dxi
    dBrd1[p, Nxi_int - 1, :] = (- 2.0 * Br[p, -2, :] + 2.0 * Br[p, -1, :]) / dxi
    
    dBrd1[p, 1:(Nxi_int - 1), :] = (N.roll(Br, -1, axis = 1)[p, 1:(Nxi_int - 1), :] - Br[p, 1:(Nxi_int - 1), :]) / dxi

    dBrd2[p, :, 0] = (- 2.0 * Br[p, :, 0] + 2.0 * Br[p, :, 1]) / deta
    dBrd2[p, :, Nxi_int - 1] = (- 2.0 * Br[p, :, -2] + 2.0 * Br[p, :, -1]) / deta
    
    dBrd2[p, :, 1:(Neta_int - 1)] = (N.roll(Br, -1, axis = 2)[p, :, 1:(Neta_int - 1)] - Br[p, :, 1:(Neta_int - 1)]) / deta

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

def contra_to_cov_E_weights(p):

    ##### Exi

    # Interior
    w1 = sqrt_det_g[1:, 1:-1, 2]
    w2 = N.roll(N.roll(sqrt_det_g, 1, axis = 0), -1, axis = 1)[1:, 1:-1, 2]
    w3 = N.roll(sqrt_det_g, 1, axis = 0)[1:, 1:-1, 2]
    w4 = N.roll(sqrt_det_g, -1, axis = 1)[1:, 1:-1, 2]
    E1d[p, 1:-1, 1:-1] = g11d[0:-1, 1:-1, 1] * E1u[p, 1:-1, 1:-1] \
                       + g12d[0:-1, 1:-1, 1] * (w1 * E2u[p, 1:, 1:-2] + w2 * N.roll(N.roll(E2u, 1, axis = 1), -1, axis = 2)[p, 1:, 1:-2] \
                                           + w3 * N.roll(E2u, 1, axis = 1)[p, 1:, 1:-2] + w4 * N.roll(E2u, -1, axis = 2)[p, 1:, 1:-2]) / (w1 + w2 + w3 + w4)
    # Left edge
    w1 = sqrt_det_g[0, 1:-1, 2]
    w2 = N.roll(sqrt_det_g, -1, axis = 1)[0, 1:-1, 2]
    E1d[p, 0, 1:-1] = g11d[0, 1:-1, 0] * E1u[p, 0, 1:-1] + g12d[0, 1:-1, 0] * (w1 * E2u[p, 0, 1:-2] + w2 * N.roll(E2u, -1, axis = 2)[p, 0, 1:-2]) / (w1 + w2)
    # Right edge
    w1 = sqrt_det_g[-1, 1:-1, 2]
    w2 = N.roll(sqrt_det_g, -1, axis = 1)[-1, 1:-1, 2]
    E1d[p, -1, 1:-1] = g11d[-1, 1:-1, 0] * E1u[p, -1, 1:-1] + g12d[-1, 1:-1, 0] * (w1 * E2u[p, -1, 1:-2] + w2 * N.roll(E2u, -1, axis = 2)[p, -1, 1:-2]) / (w1 + w2)
    # Bottom edge
    w1 = sqrt_det_g[1:, 0, 0]
    w2 = N.roll(sqrt_det_g, 1, axis = 0)[1:, 0, 0]
    E1d[p, 1:-1, 0] = g11d[0:-1, 0, 1] * E1u[p, 1:-1, 0] + g12d[0:-1, 0, 1] * (w1 * E2u[p, 1:, 0] + w2 * N.roll(E2u, 1, axis = 1)[p, 1:, 0]) / (w1 + w2)
    # Top edge
    w1 = sqrt_det_g[1:, -1, 0]
    w2 = N.roll(sqrt_det_g, 1, axis = 0)[1:, -1, 0]
    E1d[p, 1:-1, -1] = g11d[0:-1, -1, 1] * E1u[p, 1:-1, -1] + g12d[0:-1, -1, 1] * (w1 * E2u[p, 1:, -1] + w2 * N.roll(E2u, 1, axis = 1)[p, 1:, -1]) / (w1 + w2)
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
    w1 = sqrt_det_g[1:-1, 1:, 1]
    w2 = N.roll(N.roll(sqrt_det_g, -1, axis = 0), 1, axis = 1)[1:-1, 1:, 1]
    w3 = N.roll(sqrt_det_g, -1, axis = 0)[1:-1, 1:, 1]
    w4 = N.roll(sqrt_det_g, 1, axis = 1)[1:-1, 1:, 1]
    E2d[p, 1:-1, 1:-1] = g22d[1:-1, 0:-1, 2] * E2u[p, 1:-1, 1:-1] \
                       + g12d[1:-1, 0:-1, 2] * (w1 * E1u[p, 1:-2, 1:] + w2 * N.roll(N.roll(E1u, -1, axis = 1), 1, axis = 2)[p, 1:-2, 1:] \
                                              + w3 *  N.roll(E1u, -1, axis = 1)[p, 1:-2, 1:] + w4 * N.roll(E1u, 1, axis = 2)[p, 1:-2, 1:]) / (w1 + w2 + w3 + w4)
    # Left edge
    w1 = sqrt_det_g[0, 1:, 2]
    w2 = N.roll(sqrt_det_g, 1, axis = 1)[0, 1:, 2]
    E2d[p, 0, 1:-1] = g22d[0, 0:-1, 2] * E2u[p, 0, 1:-1] + g12d[0, 0:-1, 2] * (w1 * E1u[p, 0, 1:] + w2 * N.roll(E1u, 1, axis = 2)[p, 0, 1:]) / (w1 + w2)
    # Right edge
    w1 = sqrt_det_g[-1, 1:, 2]
    w2 = N.roll(sqrt_det_g, 1, axis = 1)[-1, 1:, 2]
    E2d[p, -1, 1:-1] = g22d[-1, 0:-1, 2] * E2u[p, -1, 1:-1] + g12d[-1, 0:-1, 2] * (w1 * E1u[p, -1, 1:] + w2 * N.roll(E1u, 1, axis = 2)[p, -1, 1:]) / (w1 + w2)
    # Bottom edge
    w1 = sqrt_det_g[1:-1, 0, 1]
    w2 = N.roll(sqrt_det_g, -1, axis = 0)[1:-1, 0, 1]
    E2d[p, 1:-1, 0] = g22d[1:-1, 0, 0] * E2u[p, 1:-1, 0] + g12d[1:-1, 0, 0] * (w1 * E1u[p, 1:-2, 0] + w2 * N.roll(E1u, -1, axis = 1)[p, 1:-2, 0]) / (w1 + w2)
    # Top edge
    w1 = sqrt_det_g[1:-1, -1, 1]
    w2 = N.roll(sqrt_det_g, -1, axis = 0)[1:-1, -1, 1]
    E2d[p, 1:-1, -1] = g22d[1:-1, -1, 0] * E2u[p, 1:-1, -1] + g12d[1:-1, -1, 0] * (w1 * E1u[p, 1:-2, -1] + w2 * N.roll(E1u, -1, axis = 1)[p, 1:-2, -1]) / (w1 + w2)
    # Bottom left corner
    E2d[p, 0, 0] = g22d[0, 0, 0] * E2u[p, 0, 0] + g12d[0, 0, 0] * E1u[p, 0, 0]
    # Bottom right corner
    E2d[p, -1, 0] = g22d[-1, 0, 0] * E2u[p, -1, 0] + g12d[-1, 0, 0] * E1u[p, -1, 0]
    # Top left corner
    E2d[p, 0, -1] = g22d[0, -1, 0] * E2u[p, 0, -1] + g12d[0, -1, 0] * E1u[p, 0, -1]
    # Top right corner
    E2d[p, -1, -1] = g22d[-1, -1, 0] * E2u[p, -1, -1] + g12d[-1, -1, 0] * E1u[p, -1, -1]

def contra_to_cov_E_weights2(p):

    ##### Exi

    # Interior
    w1 = sqrt_det_g[1:, 1:-1, 2]
    w2 = N.roll(N.roll(sqrt_det_g, 1, axis = 0), -1, axis = 1)[1:, 1:-1, 2]
    w3 = N.roll(sqrt_det_g, 1, axis = 0)[1:, 1:-1, 2]
    w4 = N.roll(sqrt_det_g, -1, axis = 1)[1:, 1:-1, 2]
    g1 = g12d[1:, 1:-1, 2]
    g2 = N.roll(N.roll(g12d, 1, axis = 0), -1, axis = 1)[1:, 1:-1, 2]
    g3 = N.roll(g12d, 1, axis = 0)[1:, 1:-1, 2]
    g4 = N.roll(g12d, -1, axis = 1)[1:, 1:-1, 2]
    E1d[p, 1:-1, 1:-1] = g11d[0:-1, 1:-1, 1] * E1u[p, 1:-1, 1:-1] \
                       + (w1 * g1 * E2u[p, 1:, 1:-2] + w2 * g2 * N.roll(N.roll(E2u, 1, axis = 1), -1, axis = 2)[p, 1:, 1:-2] \
                        + w3 * g3 * N.roll(E2u, 1, axis = 1)[p, 1:, 1:-2] + w4 * g4 * N.roll(E2u, -1, axis = 2)[p, 1:, 1:-2]) / (w1 + w2 + w3 + w4)
    # Left edge
    w1 = sqrt_det_g[0, 1:-1, 2]
    w2 = N.roll(sqrt_det_g, -1, axis = 1)[0, 1:-1, 2]
    g1 = g12d[0, 1:-1, 2]
    g2 = N.roll(g12d, -1, axis = 1)[0, 1:-1, 2]
    E1d[p, 0, 1:-1] = g11d[0, 1:-1, 0] * E1u[p, 0, 1:-1] + (w1 * g1 * E2u[p, 0, 1:-2] + w2 * g2 * N.roll(E2u, -1, axis = 2)[p, 0, 1:-2]) / (w1 + w2)
    # Right edge
    w1 = sqrt_det_g[-1, 1:-1, 2]
    w2 = N.roll(sqrt_det_g, -1, axis = 1)[-1, 1:-1, 2]
    g1 = g12d[-1, 1:-1, 2]
    g2 = N.roll(g12d, -1, axis = 1)[-1, 1:-1, 2]
    E1d[p, -1, 1:-1] = g11d[-1, 1:-1, 0] * E1u[p, -1, 1:-1] + (w1 * g1 * E2u[p, -1, 1:-2] + w2 * g2 * N.roll(E2u, -1, axis = 2)[p, -1, 1:-2]) / (w1 + w2)
    # Bottom edge
    w1 = sqrt_det_g[1:, 0, 0]
    w2 = N.roll(sqrt_det_g, 1, axis = 0)[1:, 0, 0]
    g1 = g12d[1:, 0, 0]
    g2 = N.roll(g12d, 1, axis = 0)[1:, 0, 0]
    E1d[p, 1:-1, 0] = g11d[0:-1, 0, 1] * E1u[p, 1:-1, 0] + (w1 * g1 * E2u[p, 1:, 0] + w2 * g2 *  N.roll(E2u, 1, axis = 1)[p, 1:, 0]) / (w1 + w2)
    # Top edge
    w1 = sqrt_det_g[1:, -1, 0]
    w2 = N.roll(sqrt_det_g, 1, axis = 0)[1:, -1, 0]
    g1 = g12d[1:, -1, 0]
    g2 = N.roll(g12d, 1, axis = 0)[1:, -1, 0]
    E1d[p, 1:-1, -1] = g11d[0:-1, -1, 1] * E1u[p, 1:-1, -1] + (w1 * g1 * E2u[p, 1:, -1] + w2 * g2 * N.roll(E2u, 1, axis = 1)[p, 1:, -1]) / (w1 + w2)
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
    w1 = sqrt_det_g[1:-1, 1:, 1]
    w2 = N.roll(N.roll(sqrt_det_g, -1, axis = 0), 1, axis = 1)[1:-1, 1:, 1]
    w3 = N.roll(sqrt_det_g, -1, axis = 0)[1:-1, 1:, 1]
    w4 = N.roll(sqrt_det_g, 1, axis = 1)[1:-1, 1:, 1]
    g1 = g12d[1:-1, 1:, 1]
    g2 = N.roll(N.roll(g12d, -1, axis = 0), 1, axis = 1)[1:-1, 1:, 1]
    g3 = N.roll(g12d, -1, axis = 0)[1:-1, 1:, 1]
    g4 = N.roll(g12d, 1, axis = 1)[1:-1, 1:, 1]
    E2d[p, 1:-1, 1:-1] = g22d[1:-1, 0:-1, 2] * E2u[p, 1:-1, 1:-1] \
                       + (w1 * g1 * E1u[p, 1:-2, 1:] + w2 * g2 * N.roll(N.roll(E1u, -1, axis = 1), 1, axis = 2)[p, 1:-2, 1:] \
                        + w3 * g3 * N.roll(E1u, -1, axis = 1)[p, 1:-2, 1:] + w4 * g4 * N.roll(E1u, 1, axis = 2)[p, 1:-2, 1:]) / (w1 + w2 + w3 + w4)
    # Left edge
    w1 = sqrt_det_g[0, 1:, 2]
    w2 = N.roll(sqrt_det_g, 1, axis = 1)[0, 1:, 2]
    g1 = g12d[0, 1:, 2]
    g2 = N.roll(g12d, 1, axis = 1)[0, 1:, 2]
    E2d[p, 0, 1:-1] = g22d[0, 0:-1, 2] * E2u[p, 0, 1:-1] + (w1 * g1 * E1u[p, 0, 1:] + w2 * g2 * N.roll(E1u, 1, axis = 2)[p, 0, 1:]) / (w1 + w2)
    # Right edge
    w1 = sqrt_det_g[-1, 1:, 2]
    w2 = N.roll(sqrt_det_g, 1, axis = 1)[-1, 1:, 2]
    g1 = g12d[-1, 1:, 2]
    g2 = N.roll(g12d, 1, axis = 1)[-1, 1:, 2]
    E2d[p, -1, 1:-1] = g22d[-1, 0:-1, 2] * E2u[p, -1, 1:-1] + (w1 * g1 * E1u[p, -1, 1:] + w2 * g2 * N.roll(E1u, 1, axis = 2)[p, -1, 1:]) / (w1 + w2)
    # Bottom edge
    w1 = sqrt_det_g[1:-1, 0, 1]
    w2 = N.roll(sqrt_det_g, -1, axis = 0)[1:-1, 0, 1]
    g1 = g12d[1:-1, 0, 1]
    g2 = N.roll(g12d, -1, axis = 0)[1:-1, 0, 1]
    E2d[p, 1:-1, 0] = g22d[1:-1, 0, 0] * E2u[p, 1:-1, 0] + (w1 * g1 * E1u[p, 1:-2, 0] + w2 * g2 * N.roll(E1u, -1, axis = 1)[p, 1:-2, 0]) / (w1 + w2)
    # Top edge
    w1 = sqrt_det_g[1:-1, -1, 1]
    w2 = N.roll(sqrt_det_g, -1, axis = 0)[1:-1, -1, 1]
    g1 = g12d[1:-1, -1, 1]
    g2 = N.roll(g12d, -1, axis = 0)[1:-1, -1, 1]
    E2d[p, 1:-1, -1] = g22d[1:-1, -1, 0] * E2u[p, 1:-1, -1] + (w1 * g1 * E1u[p, 1:-2, -1] + w2 * g2 * N.roll(E1u, -1, axis = 1)[p, 1:-2, -1]) / (w1 + w2)
    # Bottom left corner
    E2d[p, 0, 0] = g22d[0, 0, 0] * E2u[p, 0, 0] + g12d[0, 0, 0] * E1u[p, 0, 0]
    # Bottom right corner
    E2d[p, -1, 0] = g22d[-1, 0, 0] * E2u[p, -1, 0] + g12d[-1, 0, 0] * E1u[p, -1, 0]
    # Top left corner
    E2d[p, 0, -1] = g22d[0, -1, 0] * E2u[p, 0, -1] + g12d[0, -1, 0] * E1u[p, 0, -1]
    # Top right corner
    E2d[p, -1, -1] = g22d[-1, -1, 0] * E2u[p, -1, -1] + g12d[-1, -1, 0] * E1u[p, -1, -1]


Jz = N.zeros_like(Br)
Jz[Sphere.A, :, :] = 50.0 * N.exp(- (xBr_grid**2 + yBr_grid**2) / 0.1**2)

def push_B(p):
        
        # Interior
        Br[p, 1:-1, 1:-1] += dt * (dE1d2[p, 1:-1, 1:-1] - dE2d1[p, 1:-1, 1:-1]) / sqrt_det_g[0:-1, 0:-1, 3] 
        # Left edge
        Br[p, 0, 1:-1] += dt * (dE1d2[p, 0, 1:-1] - dE2d1[p, 0, 1:-1]) / sqrt_det_g[0, 0:-1, 2] 
        # Right edge
        Br[p, -1, 1:-1] += dt * (dE1d2[p, -1, 1:-1] - dE2d1[p, -1, 1:-1]) / sqrt_det_g[-1, 0:-1, 2] 
        # Bottom edge
        Br[p, 1:-1, 0] += dt * (dE1d2[p, 1:-1, 0] - dE2d1[p, 1:-1, 0]) / sqrt_det_g[0:-1, 0, 1] 
        # Top edge
        Br[p, 1:-1, -1] += dt * (dE1d2[p, 1:-1, -1] - dE2d1[p, 1:-1, -1]) / sqrt_det_g[0:-1, -1, 1] 
        # Bottom left corner
        Br[p, 0, 0] += dt * (dE1d2[p, 0, 0] - dE2d1[p, 0, 0]) / sqrt_det_g[0, 0, 0] 
        # Bottom right corner
        Br[p, -1, 0] += dt * (dE1d2[p, -1, 0] - dE2d1[p, -1, 0]) / sqrt_det_g[-1, 0, 0] 
        # Top left corner
        Br[p, 0, -1] += dt * (dE1d2[p, 0, -1] - dE2d1[p, 0, -1]) / sqrt_det_g[0, -1, 0] 
        # Top right corner
        Br[p, -1, -1] += dt * (dE1d2[p, -1, -1] - dE2d1[p, -1, -1]) / sqrt_det_g[-1, -1, 0] 
        
        # Current
        Br[p, :, :] += dt * Jz[p, :, :] * N.sin(20.0 * it * dt) # * N.exp(- it * dt / 1.0)

def push_E(p, it):

        # Interior
        E1u[p, 1:-1, :] += dt * dBrd2[p, 1:-1, :] / sqrt_det_g[0:-1, :, 1] 
        # Left edge
        E1u[p, 0, :] += dt * dBrd2[p, 0, :] / sqrt_det_g[0, :, 0] 
        # Right edge
        E1u[p, -1, :] += dt * dBrd2[p, -1, :] / sqrt_det_g[-1, :, 0] 

        # Interior
        E2u[p, :, 1:-1] -= dt * dBrd1[p, :, 1:-1] / sqrt_det_g[:, 0:-1, 2]
        # Bottom edge
        E2u[p, :, 0] -= dt * dBrd1[p, :, 0] / sqrt_det_g[:, 0, 0]
        # Top edge
        E2u[p, :, -1] -= dt * dBrd1[p, :, -1] / sqrt_det_g[:, -1, 0]

########
# Boundary conditions
########

def interp(arr_in, xA, xB):
    return N.interp(xB, xA, arr_in)

# def interp(arr_in, xA, xB):
#     f = interpolate.interp1d(xA, arr_in, kind='quadratic', fill_value="extrapolate", bounds_error=False)
#     return f(xB)

# Vertical interface inner boundary 
sig_in  = 1.0 / dxi # 75.0 # 0.5 / dt # 200.0

normt=N.zeros_like(xi_half)

txi, teta = 0.0, -1.0
normt[1:-1] = N.sqrt(g11u[-1, 0:-1, 2]*txi**2 + g22u[-1, 0:-1, 2]*teta**2 + 2.0 * g12u[-1, 0:-1, 2]*txi*teta)
normt[0] = N.sqrt(g11u[-1, 0, 0]*txi**2 + g22u[-1, 0, 0]*teta**2 + 2.0 * g12u[-1, 0, 0]*txi*teta)
normt[-1] = N.sqrt(g11u[-1, -1, 0]*txi**2 + g22u[-1, -1, 0]*teta**2 + 2.0 * g12u[-1, -1, 0]*txi*teta)

# def compute_delta_E(dtin):

#     # A -> B
#     xi1 = xi_int[0]
#     eta1 = eta_half
#     xi0, eta0 = transform_coords(Sphere.B, Sphere.A, xi1, eta1)
#     Exi_0 = interp(E1d[Sphere.A, -1, :], eta_int, eta0)
#     Eeta_0 = interp(E2d[Sphere.A, -1, :], eta_half, eta0)
#     E_th_A, E_ph_A = vec_A_to_sph(xi0, eta0, Exi_0, Eeta_0)

#     txi0, teta0 = 0.0, -1.0
#     # tth0, tph0 = form_A_to_sph(xi0, eta0, txi0, teta0)
#     # th0, ph0 = coord_A_to_sph(xi0,eta0)
#     # norm = N.sqrt(tth0**2 + N.sin(th0)**2 * tph0**2)
#     # tth0 /= norm
#     # tph0 /= norm
        
#     # Etan0 = E_th_A * tth0 + E_ph_A * tph0
#     Etan0 = Eeta_0 * teta0

#     # B -> A
#     xi0 = xi_int[-1]
#     eta0 = eta_half
#     xi1, eta1 = transform_coords(Sphere.A, Sphere.B, xi0, eta0)
#     Exi_1 = interp(E1d[Sphere.B, 0, :], eta_int, eta1)
#     Eeta_1 = interp(E2d[Sphere.B, 0, :], eta_half, eta1)
#     E_th_B, E_ph_B = vec_B_to_sph(xi1, eta1, Exi_1, Eeta_1)

#     txi1, teta1 = 0.0, -1.0
#     # tth1, tph1 = form_B_to_sph(xi1, eta1, txi1, teta1)
#     # th1, ph1 = coord_B_to_sph(xi1,eta1)
#     # norm = N.sqrt(tth1**2 + N.sin(th1)**2 * tph1**2)
#     # tth1 /= norm
#     # tph1 /= norm

#     # Etan1 = E_th_B * tth1 + E_ph_B * tph1
#     Etan1 = Eeta_1 * teta1
    
#     # WORKS
#     # diff_E2[Sphere.B, 0, :] = dtin * sig_in * (E_th_A - E_th_B)
#     # diff_E2[Sphere.A, -1, :] = dtin * sig_in * (E_th_A - E_th_B)
    
#     diff_E2[Sphere.B, 0, :] = dtin * sig_in * (Etan0 - Etan1)
#     diff_E2[Sphere.A, -1, :] = dtin * sig_in * (Etan0 - Etan1)

#     # diff_E2[Sphere.B, 0, :] = dtin * sig_in * (Eeta_1 - Eeta_0)
#     # diff_E2[Sphere.A, -1, :] = dtin * sig_in * (Eeta_1 - Eeta_0)

def compute_delta_E(dtin, E1in, E2in, Brin):

    xi0 = xi_int[-1]
    eta0 = eta_half
    Exi_0 = interp(E1in[Sphere.A, -1, :], eta_int, eta0)
    Eeta_0 = interp(E2in[Sphere.A, -1, :], eta_half, eta0)

    xi1 = xi_int[0]
    eta1 = eta_half
    Exi_1 = interp(E1in[Sphere.B, 0, :], eta_int, eta1)
    Eeta_1 = interp(E2in[Sphere.B, 0, :], eta_half, eta1)

    diff_E2[Sphere.B, 0, :] = dtin * sig_in * (Eeta_1 - Eeta_0) 
    diff_E2[Sphere.A, -1, :] = dtin * sig_in * (Eeta_1 - Eeta_0)
    
    diff_E2[Sphere.A, -1, :] = dt * sig_in * 0.5 * (- E2in[0, -1, :] / N.sqrt(g22d_half) + Brin[0, -1, :] - (- E2in[1, 0, :] / N.sqrt(g22d_half) + Brin[1, 0, :]))
    diff_E2[Sphere.B, 0, :]  = dt * sig_in * 0.5 * (E2in[1, 0, :] / N.sqrt(g22d_half) + Brin[1, 0, :] - (E2in[0, -1, :] / N.sqrt(g22d_half) + Brin[0, -1, :] ))
    
def compute_delta_B(dtin, E1in, E2in, Brin):

    # diff_Br[Sphere.A, -1, :] = dtin * sig_in * (Br[Sphere.B, 0, :] - Br[Sphere.A, -1, :])
    # diff_Br[Sphere.B, 0, :] = dtin * sig_in * (Br[Sphere.B, 0, :] - Br[Sphere.A, -1, :])

    diff_Br[Sphere.A, -1, :] = dt * sig_in * 0.5 * (E2in[0, -1, :] / N.sqrt(g22d_half) - Brin[0, -1, :] - (E2in[1, 0, :] / N.sqrt(g22d_half) - Brin[1, 0, :]))
    diff_Br[Sphere.B, 0, :]  = dt * sig_in * 0.5 * (E2in[1, 0, :] / N.sqrt(g22d_half) + Brin[1, 0, :] - (E2in[0, -1, :] / N.sqrt(g22d_half) + Brin[0, -1, :]))

# def compute_delta_E(dtin):

#     # A -> B
#     xi1 = xi_int[0]
#     eta1 = eta_half
#     xi0, eta0 = transform_coords(Sphere.B, Sphere.A, xi1, eta1)
#     E1 = interp(E1u[Sphere.A, -1, :], eta_int, eta0)
#     E2 = interp(E2u[Sphere.A, -1, :], eta_half, eta0)
#     E_target = transform_vect(Sphere.A, Sphere.B, xi0, eta0, E1, E2)[1]
#     diff_E2[Sphere.B, 0, :] = dtin * sig_in * (E2u[Sphere.B, 0, :] - E_target)

#     # B -> A
#     xi0 = xi_int[-1]
#     eta0 = eta_half
#     xi1, eta1 = transform_coords(Sphere.A, Sphere.B, xi0, eta0)
#     E1 = interp(E1u[Sphere.B, 0, :], eta_int, eta1)
#     E2 = interp(E2u[Sphere.B, 0, :], eta_half, eta1)
#     E_target = transform_vect(Sphere.B, Sphere.A, xi1, eta1, E1, E2)[1]
#     diff_E2[Sphere.A, -1, :] = dtin * sig_in * (E_target - E2u[Sphere.A, -1, :])

# def compute_delta_B(dtin):

#     # A -> B
#     xi1 = xi_int[0]
#     eta1 = eta_half
#     xi0, eta0 = transform_coords(Sphere.B, Sphere.A, xi1, eta1)
#     B_target = interp(Br[Sphere.A, -1, :], eta_half, eta0)
#     diff_Br[Sphere.B, 0, :] = dtin * sig_in * (Br[Sphere.B, 0, :] - B_target)

#     # B -> A
#     xi0 = xi_int[-1]
#     eta0 = eta_half
#     xi1, eta1 = transform_coords(Sphere.A, Sphere.B, xi0, eta0)
#     B_target = interp(Br[Sphere.B, 0, :], eta_half, eta1)
#     diff_Br[Sphere.A, -1, :] = dtin * sig_in * (B_target - Br[Sphere.A, -1, :])


# def compute_delta_E(dtin):

#     # A -> B
#     xi1 = xi_int[0]
#     eta1 = eta_half

#     xi0, eta0 = transform_coords(Sphere.B, Sphere.A, xi1, eta1)
#     E1 = interp(E1d[Sphere.A, -1, :], eta_int, eta0)
#     E2 = interp(E2d[Sphere.A, -1, :], eta_half, eta0)
#     E_target = transform_form(Sphere.A, Sphere.B, xi0, eta0, E1, E2)[1]
#     diff_E2[Sphere.B, 0, 1:-1] = dtin * sig_in * (E2d[Sphere.B, 0, 1:-1] - E_target[1:-1]) / sqrt_det_g[0, 0:-1, 2]
#     diff_E2[Sphere.B, 0, 0] = dtin * sig_in * (E2d[Sphere.B, 0, 0] - E_target[0]) / sqrt_det_g[0, 0, 0]
#     diff_E2[Sphere.B, 0, -1] = dtin * sig_in * (E2d[Sphere.B, 0, -1] - E_target[-1]) / sqrt_det_g[0, -1, 0]

#     # B -> A
#     xi0 = xi_int[-1]
#     eta0 = eta_half
#     xi1, eta1 = transform_coords(Sphere.A, Sphere.B, xi0, eta0)
#     E1 = interp(E1d[Sphere.B, 0, :], eta_int, eta1)
#     E2 = interp(E2d[Sphere.B, 0, :], eta_half, eta1)
#     E_target = transform_form(Sphere.B, Sphere.A, xi1, eta1, E1, E2)[1]
#     diff_E2[Sphere.A, -1, 1:-1] = dtin * sig_in * (E_target[1:-1] - E2d[Sphere.A, -1, 1:-1])
#     diff_E2[Sphere.A, -1, 0] = dtin * sig_in * (E_target[0] - E2d[Sphere.A, -1, 0])
#     diff_E2[Sphere.A, -1, -1] = dtin * sig_in * (E_target[-1] - E2d[Sphere.A, -1, -1])
    
# def compute_delta_B(dtin):

#     # A -> B
#     xi1 = xi_int[0]
#     eta1 = eta_half
#     xi0, eta0 = transform_coords(Sphere.B, Sphere.A, xi1, eta1)
#     B_target = interp(Br[Sphere.A, -1, :], eta_half, eta0)
#     diff_Br[Sphere.B, 0, 1:-1] = dtin * sig_in * (Br[Sphere.B, 0, 1:-1] - B_target[1:-1]) / sqrt_det_g[0, 0:-1, 2]
#     diff_Br[Sphere.B, 0, 0] = dtin * sig_in * (Br[Sphere.B, 0, 0] - B_target[0]) / sqrt_det_g[0, 0, 0]
#     diff_Br[Sphere.B, 0, -1] = dtin * sig_in * (Br[Sphere.B, 0, -1] - B_target[-1]) / sqrt_det_g[0, -1, 0]

#     # B -> A
#     xi0 = xi_int[-1]
#     eta0 = eta_half
#     xi1, eta1 = transform_coords(Sphere.A, Sphere.B, xi0, eta0)
#     B_target = interp(Br[Sphere.B, 0, :], eta_half, eta1)
#     diff_Br[Sphere.A, -1, 1:-1] = dtin * sig_in * (B_target[1:-1] - Br[Sphere.A, -1, 1:-1]) / sqrt_det_g[-1, 0:-1, 2]
#     diff_Br[Sphere.A, -1, 0] = dtin * sig_in * (B_target[0] - Br[Sphere.A, -1, 0]) / sqrt_det_g[-1, 0, 0]
#     diff_Br[Sphere.A, -1, -1] = dtin * sig_in * (B_target[-1] - Br[Sphere.A, -1, -1]) / sqrt_det_g[-1, -1, 0]


def interface_B():

    # delta = 0.5 * (diff_E2[Sphere.A, -1, :] + diff_E2[Sphere.B, 0, :])
    # Br[Sphere.B, 0, :]  -= delta
    # Br[Sphere.A, -1, :] -= delta

    Br[Sphere.B, 0, :]  -= diff_E2[Sphere.B, 0, :]  / P_int_2[0]  / sqrt_det_g_half # / N.sqrt(g22d_half)
    Br[Sphere.A, -1, :] -= diff_E2[Sphere.A, -1, :] / P_int_2[-1] / sqrt_det_g_half # / N.sqrt(g22d_half)
# 
    # #### Third attempt
    # # A -> B
    # xi1 = xi_int[0]
    # eta1 = eta_half
    # xi0, eta0 = transform_coords(Sphere.B, Sphere.A, xi1, eta1)
    # E1 = interp(E1u[Sphere.A, -1, :], eta_int, eta0)
    # E2 = interp(E2u[Sphere.A, -1, :], eta_half, eta0)
    # E_target = transform_vect(Sphere.A, Sphere.B, xi0, eta0, E1, E2)[1]
    # delta1 = dt * sig_in * (E2u[Sphere.B, 0, :] - E_target)
    # # Br[Sphere.B, 0, :]  -= dt * sig_in * (E2u[Sphere.B, 0, :] - E_target) / P_half_2[0]
    # # This doesn't work, increment as to be the same on both signs of the edge

    # # B -> A
    # xi0 = xi_int[-1]
    # eta0 = eta_half
    # xi1, eta1 = transform_coords(Sphere.A, Sphere.B, xi0, eta0)
    # E1 = interp(E1u[Sphere.B, 0, :], eta_int, eta1)
    # E2 = interp(E2u[Sphere.B, 0, :], eta_half, eta1)
    # E_target = transform_vect(Sphere.B, Sphere.A, xi1, eta1, E1, E2)[1]
    # delta2 = dt * sig_in * (E_target - E2u[Sphere.A, -1, :])
    # # Br[Sphere.A, -1, :] -= dt * sig_in * (E_target - E2u[Sphere.A, -1, :]) / P_half_2[-1]

    # delta = 0.5 * (delta1 + delta2)
    # Br[Sphere.B, 0, :]  -= delta
    # Br[Sphere.A, -1, :] -= delta

    #### Second attempt, Etheta intermediate
    # xi1 = xi_int[0]
    # eta1 = eta_half
    # xi0, eta0 = transform_coords(Sphere.B, Sphere.A, xi1, eta1)

    # E1 = interp(E1u[Sphere.A, -1, :], eta_int, eta0)
    # E2 = interp(E2u[Sphere.A, -1, :], eta_half, eta0)
    # E_th_A, E_ph_A = vec_A_to_sph(xi0, eta0, E1, E2)

    # E1 = interp(E1u[Sphere.B, 0, :], eta_int, eta1)
    # E2 = interp(E2u[Sphere.B, 0, :], eta_half, eta1)
    # E_th_B, E_ph_B = vec_B_to_sph(xi1, eta1, E1, E2)

    # Br[1, 0, :]  -= dt * sig_in * (E_th_A - E_th_B) / P_half_2[-1]
    # Br[0, -1, :] -= dt * sig_in * (E_th_A - E_th_B) / P_half_2[-1]

    #### First naive attempt
    # Br[1, 0, :]  -= dt * sig_in * (E2u[1, 0, :] - E2u[0, -1, :]) / P_half_2[0]
    # Br[0, -1, :] -= dt * sig_in * (E2u[1, 0, :] - E2u[0, -1, :]) / P_half_2[-1]

def interface_E():

    # delta = 0.5 * (diff_Br[Sphere.A, -1, :] + diff_Br[Sphere.B, 0, :])
    # E2u[Sphere.B, 0, :]  -= delta
    # E2u[Sphere.A, -1, :] -= delta
    
    E2u[Sphere.B, 0, :]  -= diff_Br[Sphere.B, 0, :]  / P_int_2[0]  / sqrt_det_g_half / N.sqrt(g22d_half)
    E2u[Sphere.A, -1, :] -= diff_Br[Sphere.A, -1, :] / P_int_2[-1] / sqrt_det_g_half / N.sqrt(g22d_half)

    # #### Third attempt
    # # A -> B
    # xi1 = xi_int[0]
    # eta1 = eta_half
    # xi0, eta0 = transform_coords(Sphere.B, Sphere.A, xi1, eta1)
    # B_target = interp(Br[Sphere.A, -1, :], eta_half, eta0)
    # delta1 = dt * sig_in * (Br[Sphere.B, 0, :] - B_target)
    # # E2u[Sphere.B, 0, :]  -= dt * sig_in * (Br[Sphere.B, 0, :] - B_target) / P_half_2[-1]

    # # B -> A
    # xi0 = xi_int[-1]
    # eta0 = eta_half
    # xi1, eta1 = transform_coords(Sphere.A, Sphere.B, xi0, eta0)
    # B_target = interp(Br[Sphere.B, 0, :], eta_half, eta1)
    # delta2 = dt * sig_in * (B_target - Br[Sphere.A, -1, :])
    # # E2u[Sphere.A, -1, :] -= dt * sig_in * (B_target - Br[Sphere.A, -1, :]) / P_half_2[-1]

    # delta = 0.5 * (delta1 + delta2)
    # E2u[Sphere.B, 0, :]  -= delta
    # E2u[Sphere.A, -1, :] -= delta
    
    #### Second attempt, Etheta intermediate
    # xi1 = xi_int[0]
    # eta1 = eta_half
    # xi0, eta0 = transform_coords(Sphere.B, Sphere.A, xi1, eta1)

    # th0, ph0 = coord_A_to_sph(xi0, eta0)
    # E1 = interp(E1u[Sphere.A, -1, :], eta_int, eta0)
    # E2 = interp(E2u[Sphere.A, -1, :], eta_half, eta0)
    # Eth0, Eph0 = vec_A_to_sph(xi0, eta0, E1, E2)
    # E2new = vec_sph_to_A(th0, ph0, Eth0 + dt * sig_in * (Br[1, 0, :] - Br[0, -1, :]) / P_half_2[-1], Eph0)[1]
    # E2u[0, -1, :] = E2new

    # th1, ph1 = coord_B_to_sph(xi1, eta1)
    # E1 = interp(E1u[Sphere.B, 0, :], eta_int, eta1)
    # E2 = interp(E2u[Sphere.B, 0, :], eta_half, eta1)
    # Eth1, Eph1 = vec_B_to_sph(xi1, eta1, E1, E2)
    # E2new = vec_sph_to_B(th1, ph1, Eth1 + dt * sig_in * (Br[1, 0, :] - Br[0, -1, :]) / P_half_2[-1], Eph1)[1]
    # E2u[1, 0, :] = E2new

    #### First naive attempt
    # E2u[0, -1, :] -= dt * sig_in * (Br[1, 0, :] - Br[0, -1, :]) / P_half_2[-1]
    # E2u[1, 0, :]  -= dt * sig_in * (Br[1, 0, :] - Br[0, -1, :]) / P_half_2[-1]


# Absorbing outer boundaries
sig_abs = sig_in # 0.5 / dxi # 70.0 # 1.0 / dt # 500.0 

def BC_absorbing_B():

    Br[0, :, 0]  += dt * sig_abs * 0.5 * (E1d[0, :, 0] / N.sqrt(g11d_half) - Br[0, :, 0]) / P_half_2[0]
    Br[0, :, -1] -= dt * sig_abs * 0.5 * (E1d[0, :, -1]  / N.sqrt(g11d_half)+ Br[0, :, -1]) / P_half_2[-1]
    Br[0, 0, :]  -= dt * sig_abs * 0.5 * (E2d[0, 0, :] / N.sqrt(g22d_half) + Br[0, 0, :])/ P_half_2[0]
    
    # Br[0, -1, :] += dt * sig_abs * (E2d[0, -1, :] - Br[0, -1, :]) / P_half_2[-1]

    Br[1, :, 0]  += dt * sig_abs * 0.5 * (E1d[1, :, 0] / N.sqrt(g11d_half) - Br[1, :, 0])/ P_half_2[0]
    Br[1, :, -1] -= dt * sig_abs * 0.5 * (E1d[1, :, -1] / N.sqrt(g11d_half) + Br[1, :, -1]) / P_half_2[-1]
    Br[1, -1, :] += dt * sig_abs * 0.5 * (E2d[1, -1, :] / N.sqrt(g22d_half) - Br[1, -1, :]) / P_half_2[-1]
    
def BC_absorbing_E():

    E1u[0, :, 0]  -= dt * sig_abs * 0.5 * (E1d[0, :, 0] / N.sqrt(g11d_half) - Br[0, :, 0]) / P_int_2[0] / N.sqrt(g11d_half)
    E1u[0, :, -1] -= dt * sig_abs * 0.5 * (E1d[0, :, -1]  / N.sqrt(g11d_half)+ Br[0, :, -1])/ P_int_2[-1] / N.sqrt(g11d_half)
    E2u[0, 0, :]  -= dt * sig_abs * 0.5 * (E2d[0, 0, :] / N.sqrt(g22d_half) + Br[0, 0, :])/ P_int_2[0] / N.sqrt(g22d_half)
    
    # E2u[0, -1, :] -= dt * sig_abs * (E2d[0, -1, :] - Br[0, -1, :]) / P_half_2[-1]

    E1u[1, :, 0]  -= dt * sig_abs * 0.5 * (E1d[1, :, 0] / N.sqrt(g11d_half) - Br[1, :, 0])/ P_int_2[0] / N.sqrt(g11d_half)
    E1u[1, :, -1] -= dt * sig_abs * 0.5 * (E1d[1, :, -1]  / N.sqrt(g11d_half)+ Br[1, :, -1]) / P_int_2[-1] / N.sqrt(g11d_half)
    E2u[1, -1, :] -= dt * sig_abs * 0.5 * (E2d[1, -1, :]  / N.sqrt(g22d_half)- Br[1, -1, :]) / P_int_2[-1] / N.sqrt(g22d_half)
    
    return

# Perfectly conducting outer boundaries    
sig_cond = 0.1 / dxi # 80.0 500.0 

def BC_conducting_B():

    Br[0, :, 0]  += dt * sig_cond * E1u[0, :, 0] / P_half_2[0]
    Br[0, :, -1] -= dt * sig_cond * E1u[0, :, -1] / P_half_2[-1]
    Br[0, 0, :]  -= dt * sig_cond * E2u[0, 0, :] / P_half_2[0]

    # Br[0, -1, :] += dt * sig_cond * E2u[0, -1, :] / P_half_2[-1]

    Br[1, :, 0]  += dt * sig_cond * E1u[1, :, 0] / P_half_2[0]
    Br[1, :, -1] -= dt * sig_cond * E1u[1, :, -1] / P_half_2[-1]
    Br[1, -1, :] += dt * sig_cond * E2u[1, -1, :] / P_half_2[-1]
    
    return

def BC_conducting_E():

    # E1u[0, :, 0]  += dt * sig_cond * Br[0, :, 0] / P_half_2[0]
    # E1u[0, :, -1] -= dt * sig_cond * Br[0, :, -1] / P_half_2[-1]
    # E2u[0, 0, :]  -= dt * sig_cond * Br[0, 0, :] / P_half_2[0]

    # E2u[0, -1, :] += dt * sig_cond * Br[0, -1, :] / P_half_2[-1]

    return

########
# Initialization
########

amp = 0.0
n_mode = 2
wave = 2.0 * (xi_max - xi_min) / n_mode
Bri = amp * N.sin(2.0 * N.pi * (xBr_grid - xi_min) / wave) * N.sin(2.0 * N.pi * (yBr_grid - xi_min) / wave)
E1ui = N.zeros((Nxi_half, Neta_int))
E2ui = N.zeros((Nxi_int, Neta_half))

for p in range(n_patches):
    Br[p, :, :] = Bri[:, :]
    E1u[p, :, :] = E1ui[:, :]
    E2u[p, :, :] = E2ui[:, :]

########
# Visualization
########

ratio = 0.5

def plot_fields(it):

    fig = P.figure(1, facecolor='w')
    ax = P.subplot(111)

    P.pcolormesh(xBr_grid - N.pi/4.0, yBr_grid, Br[0, :, :], vmin = -1, vmax = 1, cmap = 'RdBu_r')

    P.pcolormesh(xBr_grid + N.pi/4.0, yBr_grid, Br[1, :, :], vmin = -1, vmax = 1, cmap = 'RdBu_r')
    
    # P.plot([0, 0],[-1.25, 1.25], color='k')
    
    P.colorbar()
    
    P.ylim((eta_min, eta_max))
    P.xlim((xi_min - 0.5, xi_max + 0.5))
    ax.set_aspect(1.0/ax.get_data_ratio()*ratio)
    
    P.title(r'$t={:.3f} R/c$'.format(FDUMP*it*dt))
    
    figsave_png(fig, "snapshots_penalty/fields_" + str(it))

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

for it in tqdm(range(Nt), "Progression"):
    if ((it % FDUMP) == 0):
        plot_fields(idump)
        idump += 1

    compute_diff_B(patch)
    # compute_diff_B_order(patch)
    # compute_diff_B_alt(patch)
    # compute_diff_B_low(patch)

    Br0[:,:,:] = Br[:,:,:]    
    E1d0[:,:,:] = E1d[:,:,:]
    E2d0[:,:,:] = E2d[:,:,:]

    push_E(patch, it)

    E1d0[:,:,:] = 0.5 * (E1d[:,:,:] + E1d0[:,:,:])
    E2d0[:,:,:] = 0.5 * (E2d[:,:,:] + E2d0[:,:,:])

    compute_delta_B(dt, E1d0, E2d0, Br)

    interface_E() 

    BC_absorbing_E()
    # BC_conducting_E()

    contra_to_cov_E_weights(patch)
    # contra_to_cov_E(patch)

    compute_diff_E(patch)
    # compute_diff_E_order(patch)
    # compute_diff_E_alt(patch)
    # compute_diff_E_low(patch)
    
    push_B(patch)

    Br0[:,:,:] = 0.5 * (Br[:,:,:] + Br0[:,:,:])

    compute_delta_E(dt, E1d, E2d, Br0)

    interface_B()

    BC_absorbing_B()
    # BC_conducting_B()

    # compute_delta_B(dt)
    # interface_B()

    for p in range(n_patches):
        energy[p, it] = dxi * deta * N.sum(Br[p, :, :]**2) \
        + N.sum(E1u[p, :, :]**2) + N.sum(E2u[p, :, :]**2)
for p in range(n_patches):
    P.plot(time, energy[p, :])

