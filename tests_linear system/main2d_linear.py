# %%
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

# %%
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
cfl = 0.7
Nxi =  64 #24
Neta = 64 #24
NG = 1 # Number of ghosts zones
xi_min, xi_max = - N.pi / 4.0, N.pi / 4.0
eta_min, eta_max = - N.pi / 4.0, N.pi / 4.0
dxi = (xi_max - xi_min) / Nxi
deta = (eta_max - eta_min) / Neta

# %%
# Define grids
xi  = N.arange(- NG - int(Nxi / 2), NG + int(Nxi / 2), 1) * dxi
eta = N.arange(- NG - int(Neta / 2), NG + int(Neta / 2), 1) * deta
eta_grid, xi_grid = N.meshgrid(eta, xi)
xi_yee  = xi  + 0.5 * dxi
eta_yee = eta + 0.5 * deta

# Define fields
Er  = N.zeros((6, Nxi + 2 * NG, Neta + 2 * NG))
E1u = N.zeros((6, Nxi + 2 * NG, Neta + 2 * NG))
E2u = N.zeros((6, Nxi + 2 * NG, Neta + 2 * NG))
Br  = N.zeros((6, Nxi + 2 * NG, Neta + 2 * NG))
B1u = N.zeros((6, Nxi + 2 * NG, Neta + 2 * NG))
B2u = N.zeros((6, Nxi + 2 * NG, Neta + 2 * NG))

E1d = N.zeros((6, Nxi + 2 * NG, Neta + 2 * NG))
E2d = N.zeros((6, Nxi + 2 * NG, Neta + 2 * NG))
B1d = N.zeros((6, Nxi + 2 * NG, Neta + 2 * NG))
B2d = N.zeros((6, Nxi + 2 * NG, Neta + 2 * NG))

# E1u[:, NG-1, Neta + NG] = 1000.0
# E2u[:, Nxi + NG, NG-1] = 1000.0

# %%
# Define metric tensor
########

g11d = N.empty((Nxi + 2 * NG, Neta + 2 * NG, 4))
g12d = N.empty((Nxi + 2 * NG, Neta + 2 * NG, 4))
g22d = N.empty((Nxi + 2 * NG, Neta + 2 * NG, 4))

for i in range(Nxi + 2 * NG):
    for j in range(Neta + 2 * NG):
        
        # 0 at (i, j)
        X = N.tan(xi[i])
        Y = N.tan(eta[j])
        C = N.sqrt(1.0 + X * X)
        D = N.sqrt(1.0 + Y * Y)
        delta = N.sqrt(1.0 + X * X + Y * Y)
        
        g11d[i, j, 0] = (C * C * D / (delta * delta))**2
        g22d[i, j, 0] = (C * D * D / (delta * delta))**2
        g12d[i, j, 0] = - X * Y * C * C * D * D / (delta)**4
        
        # 1 at (i + 1/2, j)
        X = N.tan(xi_yee[i])
        Y = N.tan(eta[j])
        C = N.sqrt(1.0 + X * X)
        D = N.sqrt(1.0 + Y * Y)
        delta = N.sqrt(1.0 + X * X + Y * Y)
        
        g11d[i, j, 1] = (C * C * D / (delta * delta))**2
        g22d[i, j, 1] = (C * D * D / (delta * delta))**2
        g12d[i, j, 1] = - X * Y * C * C * D * D / (delta)**4
        
        # 2 at (i, j + 1/2)
        X = N.tan(xi[i])
        Y = N.tan(eta_yee[j])
        C = N.sqrt(1.0 + X * X)
        D = N.sqrt(1.0 + Y * Y)
        delta = N.sqrt(1.0 + X * X + Y * Y)
        
        g11d[i, j, 2] = (C * C * D / (delta * delta))**2
        g22d[i, j, 2] = (C * D * D / (delta * delta))**2
        g12d[i, j, 2] = - X * Y * C * C * D * D / (delta)**4

        # 3 at (i + 1/2, j + 1/2)
        X = N.tan(xi_yee[i])
        Y = N.tan(eta_yee[j])
        C = N.sqrt(1.0 + X * X)
        D = N.sqrt(1.0 + Y * Y)
        delta = N.sqrt(1.0 + X * X + Y * Y)
        
        g11d[i, j, 3] = (C * C * D / (delta * delta))**2
        g22d[i, j, 3] = (C * D * D / (delta * delta))**2
        g12d[i, j, 3] = - X * Y * C * C * D * D / (delta)**4

sqrt_det_g = N.sqrt(g11d * g22d - g12d * g12d)

dt = cfl * N.min(1.0 / N.sqrt(g11d / (sqrt_det_g * sqrt_det_g) / (dxi * dxi) + g22d / (sqrt_det_g * sqrt_det_g) / (deta * deta) ))
print("delta t = {}".format(dt))

# %%
# Single-patch push routines
########

def contra_to_cov_E(patch):
    
    i0, i1 = NG, Nxi + NG
    j0, j1 = NG, Neta + NG
             
    # E1d[patch, i0:i1, j0:j1] = g11d[i0:i1, j0:j1, 1] * E1u[patch, i0:i1, j0:j1] + \
    #                      0.5 * g12d[i0:i1, j0:j1, 1] * (E2u[patch, i0:i1, j0:j1] + N.roll(N.roll(E2u, -1, axis = 1), 1, axis = 2)[patch, i0:i1, j0:j1])
    # E2d[patch, i0:i1, j0:j1] = g22d[i0:i1, j0:j1, 2] * E2u[patch, i0:i1, j0:j1] + \
    #                      0.5 * g12d[i0:i1, j0:j1, 2] * (E1u[patch, i0:i1, j0:j1] + N.roll(N.roll(E1u, 1, axis = 1), -1, axis = 2)[patch, i0:i1, j0:j1])

    E1d[patch, i0:i1, j0:j1] = g11d[i0:i1, j0:j1, 1] * E1u[patch, i0:i1, j0:j1] + \
                         0.25 * g12d[i0:i1, j0:j1, 1] * (E2u[patch, i0:i1, j0:j1] + N.roll(N.roll(E2u, -1, axis = 1), 1, axis = 2)[patch, i0:i1, j0:j1] \
                                                        + N.roll(E2u, -1, axis = 1)[patch, i0:i1, j0:j1] + N.roll(E2u, 1, axis = 2)[patch, i0:i1, j0:j1])
    E2d[patch, i0:i1, j0:j1] = g22d[i0:i1, j0:j1, 2] * E2u[patch, i0:i1, j0:j1] + \
                         0.25 * g12d[i0:i1, j0:j1, 2] * (E1u[patch, i0:i1, j0:j1] + N.roll(N.roll(E1u, 1, axis = 1), -1, axis = 2)[patch, i0:i1, j0:j1] + 
                                                         + N.roll(E1u, 1, axis = 1)[patch, i0:i1, j0:j1] + N.roll(E1u, -1, axis = 2)[patch, i0:i1, j0:j1])

    # w1 = sqrt_det_g[i0:i1, j0:j1, 2]
    # w2 = N.roll(N.roll(sqrt_det_g[:, :, 2], -1, axis = 0), 1, axis = 1)[i0:i1, j0:j1]
    # w3 = N.roll(sqrt_det_g[:, :, 2], -1, axis = 0)[i0:i1, j0:j1]
    # w4 = N.roll(sqrt_det_g[:, :, 2], 1, axis = 1)[i0:i1, j0:j1]
    
    # E1d[patch, i0:i1, j0:j1] = g11d[i0:i1, j0:j1, 1] * E1u[patch, i0:i1, j0:j1] \
    #                          + (w1 * g12d[i0:i1, j0:j1, 2] * E2u[patch, i0:i1, j0:j1] + w2 * N.roll(N.roll(g12d, -1, axis = 0), 1, axis = 1)[i0:i1, j0:j1, 2] * N.roll(N.roll(E2u, -1, axis = 1), 1, axis = 2)[patch, i0:i1, j0:j1] \
    #                          + w3 * N.roll(g12d, -1, axis = 0)[i0:i1, j0:j1, 2] * N.roll(E2u, -1, axis = 1)[patch, i0:i1, j0:j1] + w4 * N.roll(g12d, 1, axis = 1)[i0:i1, j0:j1, 2] * N.roll(E2u, 1, axis = 2)[patch, i0:i1, j0:j1]) / (w1 + w2 + w3 + w4)

    # w1 = sqrt_det_g[i0:i1, j0:j1, 1]
    # w2 = N.roll(N.roll(sqrt_det_g[:, :, 1], 1, axis = 0), -1, axis = 1)[i0:i1, j0:j1]
    # w3 = N.roll(sqrt_det_g[:, :, 1], 1, axis = 0)[i0:i1, j0:j1]
    # w4 = N.roll(sqrt_det_g[:, :, 1], -1, axis = 1)[i0:i1, j0:j1]
    
    # E2d[patch, i0:i1, j0:j1] = g22d[i0:i1, j0:j1, 2] * E2u[patch, i0:i1, j0:j1] \
    #                          + (w1 * g12d[i0:i1, j0:j1, 1] * E1u[patch, i0:i1, j0:j1] + w2 * N.roll(N.roll(g12d, 1, axis = 0), -1, axis = 1)[i0:i1, j0:j1, 1] * N.roll(N.roll(E1u, 1, axis = 1), -1, axis = 2)[patch, i0:i1, j0:j1] + 
    #                          + w3 * N.roll(g12d, 1, axis = 0)[i0:i1, j0:j1, 1] * N.roll(E1u, 1, axis = 1)[patch, i0:i1, j0:j1] + w4 * N.roll(g12d, -1, axis = 1)[i0:i1, j0:j1, 1] * N.roll(E1u, -1, axis = 2)[patch, i0:i1, j0:j1]) / (w1 + w2 + w3 + w4)


def contra_to_cov_B(patch):

    i0, i1 = NG, Nxi + NG 
    j0, j1 = NG, Neta + NG
    
    # B1d[patch, i0:i1, j0:j1] = g11d[i0:i1, j0:j1, 2] * B1u[patch, i0:i1, j0:j1] + \
    #                      0.5 * g12d[i0:i1, j0:j1, 2] * (B2u[patch, i0:i1, j0:j1] + N.roll(N.roll(B2u, 1, axis = 1), -1, axis = 2)[patch, i0:i1, j0:j1])            
    # B2d[patch, i0:i1, j0:j1] = g22d[i0:i1, j0:j1, 1] * B2u[patch, i0:i1, j0:j1] + \
    #                      0.5 * g12d[i0:i1, j0:j1, 1] * (B1u[patch, i0:i1, j0:j1] + N.roll(N.roll(B1u, -1, axis = 1), 1, axis = 2)[patch, i0:i1, j0:j1])

    B1d[patch, i0:i1, j0:j1] = g11d[i0:i1, j0:j1, 2] * B1u[patch, i0:i1, j0:j1] + \
                         0.25 * g12d[i0:i1, j0:j1, 2] * (B2u[patch, i0:i1, j0:j1] + N.roll(N.roll(B2u, 1, axis = 1), -1, axis = 2)[patch, i0:i1, j0:j1]
                                                         + N.roll(B2u, 1, axis = 1)[patch, i0:i1, j0:j1] + N.roll(B2u, -1, axis = 2)[patch, i0:i1, j0:j1])            
    B2d[patch, i0:i1, j0:j1] = g22d[i0:i1, j0:j1, 1] * B2u[patch, i0:i1, j0:j1] + \
                         0.25 * g12d[i0:i1, j0:j1, 1] * (B1u[patch, i0:i1, j0:j1] + N.roll(N.roll(B1u, -1, axis = 1), 1, axis = 2)[patch, i0:i1, j0:j1]
                                                         + N.roll(B1u, -1, axis = 1)[patch, i0:i1, j0:j1] + N.roll(B1u, 1, axis = 2)[patch, i0:i1, j0:j1])

def push_B(it, patch):
    
    i0, i1 = NG, Nxi + NG
    j0, j1 = NG, Neta + NG
    
    Br[patch, i0:i1, j0:j1]  -= ((N.roll(E2d, -1, axis = 1)[patch, i0:i1, j0:j1] - E2d[patch, i0:i1, j0:j1]) / dxi - \
                                 (N.roll(E1d, -1, axis = 2)[patch, i0:i1, j0:j1] - E1d[patch, i0:i1, j0:j1]) / deta) \
                                * dt / sqrt_det_g[i0:i1, j0:j1, 3]  - 4.0 * N.pi * dt * Jr(it, patch, i0, i1, j0, j1) 
    B1u[patch, i0:i1, j0:j1] -= ((N.roll(Er, -1, axis = 2)[patch, i0:i1, j0:j1] - Er[patch, i0:i1, j0:j1]) / deta) * dt / sqrt_det_g[i0:i1, j0:j1, 2]
    B2u[patch, i0:i1, j0:j1] += ((N.roll(Er, -1, axis = 1)[patch, i0:i1, j0:j1] - Er[patch, i0:i1, j0:j1]) / dxi)  * dt / sqrt_det_g[i0:i1, j0:j1, 1]

def push_E(it, patch):
    
    i0, i1 = NG, Nxi + NG
    j0, j1 = NG, Neta + NG
    
    Er[patch, i0:i1, j0:j1] += ((B2d[patch, i0:i1, j0:j1] - N.roll(B2d, 1, axis = 1)[patch, i0:i1, j0:j1]) / dxi - \
                                (B1d[patch, i0:i1, j0:j1] - N.roll(B1d, 1, axis = 2)[patch, i0:i1, j0:j1]) / deta) \
                               * dt / sqrt_det_g[i0:i1, j0:j1, 0] # - 4.0 * N.pi * dt * Jr(it, patch, i0, i1, j0, j1) 
    E1u[patch, i0:i1, j0:j1] += ((Br[patch, i0:i1, j0:j1] - N.roll(Br, 1, axis = 2)[patch, i0:i1, j0:j1]) / deta) * dt / sqrt_det_g[i0:i1, j0:j1, 1]
    E2u[patch, i0:i1, j0:j1] -= ((Br[patch, i0:i1, j0:j1] - N.roll(Br, 1, axis = 1)[patch, i0:i1, j0:j1]) / dxi)  * dt / sqrt_det_g[i0:i1, j0:j1, 2]

# def push_B(it, patch):
    
#     i0, i1 = NG, Nxi + NG
#     j0, j1 = NG, Neta + NG
    
#     Br[patch, i0:i1, j0:j1]  -= ((27.0 * N.roll(E2d, -1, axis = 1)[patch, i0:i1, j0:j1] - 27.0 * E2d[patch, i0:i1, j0:j1]
#                                   - N.roll(E2d, -2, axis = 1)[patch, i0:i1, j0:j1] + N.roll(E2d, 1, axis = 1)[patch, i0:i1, j0:j1]) / dxi - \
#                                  (27.0 * N.roll(E1d, -1, axis = 2)[patch, i0:i1, j0:j1] - 27.0 * E1d[patch, i0:i1, j0:j1]
#                                   + N.roll(E1d, -2, axis = 2)[patch, i0:i1, j0:j1] - N.roll(E1d, 1, axis = 2)[patch, i0:i1, j0:j1]) / deta) \
#                                 * dt / sqrt_det_g[i0:i1, j0:j1, 3] / 24.0 - 4.0 * N.pi * dt * Jr(it, patch, i0, i1, j0, j1) 

#     B1u[patch, i0:i1, j0:j1] -= ((N.roll(Er, -1, axis = 2)[patch, i0:i1, j0:j1] - Er[patch, i0:i1, j0:j1]) / deta) * dt / sqrt_det_g[i0:i1, j0:j1, 2]
#     B2u[patch, i0:i1, j0:j1] += ((N.roll(Er, -1, axis = 1)[patch, i0:i1, j0:j1] - Er[patch, i0:i1, j0:j1]) / dxi)  * dt / sqrt_det_g[i0:i1, j0:j1, 1]


# def push_E(it, patch):
    
#     i0, i1 = NG, Nxi + NG
#     j0, j1 = NG, Neta + NG 
    
#     Er[patch, i0:i1, j0:j1] += ((B2d[patch, i0:i1, j0:j1] - N.roll(B2d, 1, axis = 1)[patch, i0:i1, j0:j1]) / dxi - \
#                                 (B1d[patch, i0:i1, j0:j1] - N.roll(B1d, 1, axis = 2)[patch, i0:i1, j0:j1]) / deta) \
#                                * dt / sqrt_det_g[i0:i1, j0:j1, 0]

#     E1u[patch, i0:i1, j0:j1] += ((27.0 * Br[patch, i0:i1, j0:j1] - 27.0 * N.roll(Br, 1, axis = 2)[patch, i0:i1, j0:j1] \
#                                   + N.roll(Br, 2, axis = 2)[patch, i0:i1, j0:j1] - N.roll(Br, -1, axis = 2)[patch, i0:i1, j0:j1] \
#                                   ) / 24.0 / deta) * dt / sqrt_det_g[i0:i1, j0:j1, 1]

#     E2u[patch, i0:i1, j0:j1] -= ((27.0 * Br[patch, i0:i1, j0:j1] - 27.0 * N.roll(Br, 1, axis = 1)[patch, i0:i1, j0:j1] \
#                                   + N.roll(Br, 2, axis = 1)[patch, i0:i1, j0:j1] - N.roll(Br, -1, axis = 1)[patch, i0:i1, j0:j1] \
#                                   ) / 24.0 / dxi)  * dt / sqrt_det_g[i0:i1, j0:j1, 2]


divB=N.zeros_like(B1u)

# Compute div(B) scalar on whole domain
def compute_div_B(patch):
    
    for i in range(NG, Nxi + NG - 1):
        for j in range(NG, Neta + NG): 
               
            divB[patch, i, j] = ((sqrt_det_g[i + 1, j, 2] * B1u[patch, i + 1, j] - sqrt_det_g[i, j, 2] * B1u[patch, i, j]) / dxi + \
                            (sqrt_det_g[i, j + 1, 1] * B2u[patch, i, j + 1] - sqrt_det_g[i, j, 1] * B2u[patch, i, j]) / deta) / sqrt_det_g[i, j, 3]


# %%
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

# %%
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

from jacob_transformations_flip import *

def jacob(patch0, patch1, xi0, eta0):
    fcoord0 = (globals()["coord_" + sphere[patch0] + "_to_sph"])
    theta0, phi0 = fcoord0(xi0, eta0)
    fjac0 = (globals()["jacob_" + sphere[patch0] + "_to_sph"])
    fjac1 = (globals()["jacob_sph_to_" + sphere[patch1]])
    J1 = fjac0(xi0, eta0)
    J0 = fjac1(theta0, phi0)
    res=N.zeros((Nxi + 2 * NG, 2, 2))
    res[:, 0, 0] = J1[0,0,:] * J0[0,0,:] + J1[0,1,:] * J0[1,0,:]
    res[:, 0, 1] = J1[0,0,:] * J0[0,1,:] + J1[0,1,:] * J0[1,1,:]
    res[:, 1, 0] = J1[1,0,:] * J0[0,0,:] + J1[1,1,:] * J0[1,0,:]
    res[:, 1, 1] = J1[1,0,:] * J0[0,1,:] + J1[1,1,:] * J0[1,1,:]
    return res

from jacob_inv_transformations_flip import *

def jacob_inv(patch0, patch1, xi0, eta0):
    fcoord0 = (globals()["coord_" + sphere[patch0] + "_to_sph"])
    theta0, phi0 = fcoord0(xi0, eta0)
    fjac0 = (globals()["jacob_inv_" + sphere[patch0] + "_to_sph"])
    fjac1 = (globals()["jacob_inv_sph_to_" + sphere[patch1]])
    J0 = fjac0(xi0, eta0)
    J1 = fjac1(theta0, phi0)
    res=N.zeros((Nxi + 2 * NG, 2, 2))
    res[:, 0, 0] = J1[0,0,:] * J0[0,0,:] + J1[0,1,:] * J0[1,0,:]
    res[:, 0, 1] = J1[0,0,:] * J0[0,1,:] + J1[0,1,:] * J0[1,1,:]
    res[:, 1, 0] = J1[1,0,:] * J0[0,0,:] + J1[1,1,:] * J0[1,0,:]
    res[:, 1, 1] = J1[1,0,:] * J0[0,1,:] + J1[1,1,:] * J0[1,1,:]
    # res = N.sum(N.transpose(J1,(0,2,1)).reshape(Nxi + 2 * NG,2,2,1)*J0.reshape(Nxi + 2 * NG,2,1,2),-3)
    return res

# %%
# Interpolation routines
########

# def interp(arr_in, xA, xB):
#     f = interpolate.interp1d(xA, arr_in, kind='linear', bounds_error=True)
#     return f(xB)

# def interp(arr_in, xA, xB):
#     f = interpolate.interp1d(xA, arr_in, kind='quadratic', fill_value="extrapolate", bounds_error=False)
#     return f(xB)

def interp(arr_in, xA, xB):
    return N.interp(xB, xA, arr_in)

# %%
# Communication between two patches
########

def communicate_E_patch_contra(patch0, patch1):
    communicate_E_patch(patch0, patch1, "vect")

def communicate_E_patch_cov(patch0, patch1):
    communicate_E_patch(patch0, patch1, "form")

def communicate_B_patch_contra(patch0, patch1):
    communicate_B_patch(patch0, patch1, "vect")

def communicate_B_patch_cov(patch0, patch1):
    communicate_B_patch(patch0, patch1, "form")

E_edge_xi_0 = N.zeros(Nxi + 2 * NG)
E_edge_eta_0 = N.zeros(Nxi + 2 * NG)
E_edge_xi_1 = N.zeros(Neta + 2 * NG)
E_edge_eta_1 = N.zeros(Neta + 2 * NG)

def compute_E_edge(p0, p1, typ, top):
    
    if (typ == "vect"):
        field1 = E1u
        field2 = E2u
        jac = jacob
    elif (typ == "form"):
        field1 = E1d
        field2 = E2d
        jac = jacob_inv
    
    if (top == 'xx'):
    
        # E_xi_b at NG, j 
        
        xi1 = xi_grid[NG - 1, :] + 0.0 * dxi
        eta1 = eta_grid[NG - 1, :] + 0.5 * deta
        xi0, eta0 = transform_coords(p1, p0, xi1, eta1)
        E_eta_0 = interp(field2[p0, Nxi + NG - 1, :], eta_yee, eta0)

        xi1 = xi_grid[NG - 1, :] + 0.5 * dxi
        eta1 = eta_grid[NG - 1, :] + 0.5 * deta
        xi0, eta0 = transform_coords(p1, p0, xi1, eta1)
        E_xi_0 = interp(field1[p0, Nxi + NG - 1, :], eta, eta0)

        E_eta_1 = field2[p1, NG, :]

        xi1 = xi_grid[NG, :] + 0.5 * dxi
        eta1 = eta_grid[NG, :] + 0.5 * deta
        E_xi_1 = interp(field1[p1, NG, :], eta, eta1)
        
        xi1 = xi_grid[NG - 1, :] + 0.5 * dxi
        eta1 = eta_grid[NG - 1, :] + 0.5 * deta
        xi0, eta0 = transform_coords(p1, p0, xi1, eta1)
        J_0to1 = jac(p0, p1, xi0, eta0)
        
        xi1 = xi_grid[NG, :] + 0.0 * dxi
        eta1 = eta_grid[NG, :] + 0.5 * deta            
        J_1to0 = jac(p1, p0, xi1, eta1)

        denom = 4.0 - J_0to1[:, 1, 0] * J_0to1[:, 0, 1]      
        E_edge_xi_1[:] = (2.0 * E_xi_1 + 2.0 * J_0to1[:, 0, 0] * E_xi_0 + J_0to1[:, 1, 0] * (E_eta_0 + J_0to1[:, 1, 1] * E_eta_1)) / denom
        
        # E_eta_a at Nxi + NG - 1/2, j + 1/2
        
        E_xi_0 = field1[p0, Nxi + NG - 1, :]
        
        xi0 = xi_grid[NG - 1, :] + 0.0 * dxi
        eta0 = eta_grid[NG - 1, :] + 0.0 * deta
        E_eta_0 = interp(field2[p0, Nxi + NG - 1, :], eta_yee, eta0)
        
        xi0 = xi_grid[NG, :] + 0.5 * dxi
        eta0 = eta_grid[NG, :] + 0.0 * deta
        xi1, eta1 = transform_coords(p0, p1, xi0, eta0)
        E_xi_1 = interp(field1[p1, NG, :], eta, eta1)
        
        xi0 = xi_grid[NG, :] + 0.0 * dxi
        eta0 = eta_grid[NG, :] + 0.0 * deta
        xi1, eta1 = transform_coords(p0, p1, xi0, eta0)
        E_eta_1 = interp(field2[p1, NG, :], eta_yee, eta1)
        
        xi0 = xi_grid[NG - 1, :] + 0.5 * dxi
        eta0 = eta_grid[NG - 1, :] + 0.0 * deta
        J_0to1 = jac(p0, p1, xi0, eta0)
        
        xi0 = xi_grid[NG, :] + 0.0 * dxi
        eta0 = eta_grid[NG, :] + 0.0 * deta
        xi1, eta1 = transform_coords(p0, p1, xi0, eta0)
        J_1to0 = jac(p1, p0, xi1, eta1)
        
        denom = 4.0 - J_0to1[:, 1, 0] * J_1to0[:, 0, 1]      
        E_edge_eta_0[:] = (2.0 * E_eta_0 + 2.0 * J_1to0[:, 1, 1] * E_eta_1 + J_1to0[:, 0, 1] * (E_xi_1 + J_0to1[:, 0, 0] * E_xi_0)) / denom
            
    elif (top == 'xy'):

        # E_eta_b at i, NG 

        xi1 = xi_grid[:, NG - 1] + 0.5 * dxi
        eta1 = eta_grid[:, NG - 1] + 0.0 * deta
        xi0, eta0 = transform_coords(p1, p0, xi1, eta1)
        E_eta_0 = interp(field2[p0, Nxi + NG - 1, :], eta_yee, eta0)
        
        xi1 = xi_grid[:, NG - 1] + 0.5 * dxi
        eta1 = eta_grid[:, NG - 1] + 0.5 * deta
        xi0, eta0 = transform_coords(p1, p0, xi1, eta1)
        E_xi_0 = interp(field1[p0, Nxi + NG - 1, :], eta, eta0)

        E_xi_1 = field1[p1, :, NG]
                
        xi1 = xi_grid[NG, :] + 0.5 * dxi
        eta1 = eta_grid[NG, :] + 0.5 * deta
        E_eta_1 = interp(field2[p1, :, NG], xi, xi1)
                
        xi1 = xi_grid[:, NG - 1] + 0.5 * dxi
        eta1 = eta_grid[:, NG - 1] + 0.5 * deta
        xi0, eta0 = transform_coords(p1, p0, xi1, eta1)
        J_0to1 = jac(p0, p1, xi0, eta0)
        
        xi1 = xi_grid[:, NG] + 0.5 * dxi
        eta1 = eta_grid[:, NG] + 0.0 * deta            
        J_1to0 = jac(p1, p0, xi1, eta1)
        
        denom = 4.0 - J_0to1[:, 1, 1] * J_1to0[:, 1, 1]      
        E_edge_eta_1[:] = (2.0 * E_eta_1 + 2.0 * J_0to1[:, 1, 0] * E_xi_0 + J_0to1[:, 1, 1] * (E_eta_0 + J_1to0[:, 1, 0] * E_xi_0)) / denom

        # E_eta_a at Nxi + NG - 1/2, j + 1/2

        xi0 = xi_grid[NG - 1, :] + 0.0 * dxi
        eta0 = eta_grid[NG - 1, :] + 0.0 * deta
        E_eta_0 = interp(field2[p0, Nxi + NG - 1, :], eta_yee, eta0)        

        E_xi_0 = field1[p0, Nxi + NG - 1, :]
        
        xi0 = xi_grid[NG, :] + 0.0 * dxi
        eta0 = eta_grid[NG, :] + 0.0 * deta
        xi1, eta1 = transform_coords(p0, p1, xi0, eta0)
        E_xi_1 = interp(field1[p1, :, NG], xi_yee, xi1)
        
        xi0 = xi_grid[NG, :] + 0.5 * dxi
        eta0 = eta_grid[NG, :] + 0.0 * deta
        xi1, eta1 = transform_coords(p0, p1, xi0, eta0)
        E_eta_1 = interp(field2[p1, :, NG], xi, xi1)
        
        xi0 = xi_grid[NG - 1, :] + 0.5 * dxi
        eta0 = eta_grid[NG - 1, :] + 0.0 * deta
        J_0to1 = jac(p0, p1, xi0, eta0)
        
        xi0 = xi_grid[NG, :] + 0.0 * dxi
        eta0 = eta_grid[NG, :] + 0.0 * deta
        xi1, eta1 = transform_coords(p0, p1, xi0, eta0)
        J_1to0 = jac(p1, p0, xi1, eta1)
        
        denom = 4.0 - J_0to1[:, 1, 1] * J_1to0[:, 1, 1]      
        E_edge_eta_0[:] = (2.0 * E_eta_0 + 2.0 * J_1to0[:, 1, 0] * E_xi_1 + J_1to0[:, 1, 1] * (E_eta_1 + J_0to1[:, 1, 0] * E_xi_1)) / denom

    elif (top == 'yy'):
    
        for j in range(NG, Nxi + NG):

            E_xi_0  = 0.5 * (field1[p0, j, Neta + NG - 1] + field1[p0, j - 1, Neta + NG - 1])    
            E_eta_0 = field2[p0, j, Neta + NG - 1]
            E_xi_1  = 0.5 * (field1[p1, j, NG] + field1[p1, j - 1, NG])    
            E_eta_1 = field2[p1, j, NG] 
            
            J_0to1 = jac(p0, p1, xi[j], eta_yee[Neta + NG - 1])
            J_1to0 = jac(p1, p0, xi[j], eta[NG])

            denom = 4.0 - J_0to1[1, 0] * J_1to0[0, 1]
            E_edge_0[j] = (2.0 * E_xi_0 + 2.0 * J_1to0[0, 0] * E_xi_1 + J_1to0[0, 1] * (E_eta_1 + J_0to1[1, 1] * E_eta_0)) / denom
            E_edge_1[j] = (2.0 * E_eta_1 + 2.0 * J_0to1[1, 1] * E_eta_0 + J_0to1[1, 0] * (E_xi_0 + J_1to0[0, 0] * E_xi_1)) / denom

    elif (top == 'yx'):
    
        for j in range(NG, Nxi + NG):

            E_xi_0  = 0.5 * (field1[p0, j, Neta + NG - 1] + field1[p0, j - 1, Neta + NG - 1])    
            E_eta_0 = field2[p0, j, Neta + NG - 1]
            E_xi_1  = field1[p1, NG, j]
            E_eta_1 = 0.5 * (field2[p1, NG, j] + field2[p1, NG, j - 1]) 
            
            J_0to1 = jac(p0, p1, xi[j], eta_yee[Neta + NG - 1])
            J_1to0 = jac(p1, p0, xi[NG], eta[j])

            denom = 4.0 - J_0to1[1, 0] * J_1to0[0, 1]
            E_edge_0[j] = (2.0 * E_xi_0 + 2.0 * J_1to0[0, 0] * E_xi_1 + J_1to0[0, 1] * (E_eta_1 + J_0to1[1, 1] * E_eta_0)) / denom

            denom = 4.0 - J_0to1[0, 1] * J_1to0[1, 0]      
            E_edge_1[j] = (2.0 * E_xi_1 + 2.0 * J_0to1[0, 0] * E_xi_0 + J_0to1[0, 1] * (E_eta_0 + J_1to0[1, 1] * E_eta_1)) / denom

    else:
        return

########
# Communication of covariant or contravariant E
# patch0 has the open boundary
# patch1 has the closed boundary 
########  
        
def communicate_E_patch(patch0, patch1, typ):

    if (typ == "vect"):
        field1 = E1u
        field2 = E2u
    if (typ == "form"):
        field1 = E1d
        field2 = E2d
    
    transform = (globals()["transform_" + typ])

    top = topology[patch0, patch1]

    if (top == 'xx'):

        #########
        # Communicate fields from xi left edge of patch1 to xi right edge of patch0
        ########

        i0 = Nxi + NG # Last ghost cell of xi edge of patch0             
        i1 = NG  # First active cell of xi edge of patch1   
                
        xi0 = xi_grid[i0, :] + 0.0 * dxi
        eta0 = eta_grid[i0, :] + 0.5 * deta
        
        xi1, eta1 = transform_coords(patch0, patch1, xi0, eta0)
        
        E1 = interp(field1[patch1, i1, :], eta_yee, eta1)
        E2 = interp(field2[patch1, i1, :], eta_yee, eta1)

        # E2p = interp(field2[patch0, i0, :], eta_yee, eta1)
        # E2 = interp(0.5*(field2[patch1, i1, :]+E2p), eta_yee, eta1)

        field2[patch0, i0, NG:(Nxi + NG)] = transform(patch1, patch0, xi1, eta1, E1, E2)[1][NG:(Nxi + NG)]

        #########
        # Communicate fields from xi right edge of patch0 to xi left edge patch1
        ########
        
        i0 = Nxi + NG - 1 # Last active cell of xi edge of patch0                   
        i1 = NG - 1 # First ghost cell of xi edge of patch1   
                
        xi1 = xi_grid[i1, :] + 0.5 * dxi
        eta1 = eta_grid[i1, :] + 0.0 * deta
        
        xi0, eta0 = transform_coords(patch1, patch0, xi1, eta1)
        
        E1 = interp(field1[patch0, i0, :], eta, eta0)
        # E2 = (interp(field2[patch0, i0, :], eta_yee, eta0) + interp(field2[patch0, i0 + 1, :], eta_yee, eta0)) / 2.0
        E2 = interp(field2[patch0, i0, :], eta_yee, eta0)
        
        field1[patch1, i1, NG:(Nxi + NG)] = transform(patch0, patch1, xi0, eta0, E1, E2)[0][NG:(Nxi + NG)]
    
    # if (top == 'xx'):

    #     compute_E_edge(patch0, patch1, typ, top)

    #     #########
    #     # Communicate fields from xi left edge of patch1 to xi right edge of patch0
    #     ########

    #     i0 = Nxi + NG # Last ghost cell of xi edge of patch0             
    #     i1 = NG  # First active cell of xi edge of patch1   
                
    #     xi0 = xi_grid[i0, :] + 0.0 * dxi
    #     eta0 = eta_grid[i0, :] + 0.5 * deta
    #     xi1, eta1 = transform_coords(patch0, patch1, xi0, eta0)
        
    #     E1 = interp(E_edge_xi_1, eta_yee, eta1)
    #     E2 = interp(field2[patch1, i1, :], eta_yee, eta1)
    #     field2[patch0, i0, NG:(Nxi + NG)] = transform(patch1, patch0, xi1, eta1, E1, E2)[1][NG:(Nxi + NG)]

    #     #########
    #     # Communicate fields from xi right edge of patch0 to xi left edge patch1
    #     ########
        
    #     i0 = Nxi + NG - 1 # Last active cell of xi edge of patch0                   
    #     i1 = NG - 1 # First ghost cell of xi edge of patch1   
                
    #     xi1 = xi_grid[i1, :] + 0.5 * dxi
    #     eta1 = eta_grid[i1, :] + 0.0 * deta
    #     xi0, eta0 = transform_coords(patch1, patch0, xi1, eta1)
        
    #     E1 = interp(field1[patch0, i0, :], eta, eta0)
    #     E2 = interp(E_edge_eta_0, eta, eta0)
        
    #     field1[patch1, i1, NG:(Nxi + NG)] = transform(patch0, patch1, xi0, eta0, E1, E2)[0][NG:(Nxi + NG)]

        if (typ == "vect"):
        
            i0 = Nxi + NG # Last ghost cell of xi edge of patch0             
            i1 = NG  # First active cell of xi edge of patch1 
            
            xi0 = xi_grid[i0, :] + 0.0 * dxi
            eta0 = eta_grid[i0, :] + 0.0 * deta
            
            xi1, eta1 = transform_coords(patch0, patch1, xi0, eta0)
            
            Er1 = interp(Er[patch1, i1, :], eta, eta1)
            Er[patch0, i0, NG:(Nxi + NG)] = Er1[NG:(Nxi + NG)]

    elif (top == 'xy'):
                    
        #########
        # Communicate fields from eta left edge of patch1 to xi right edge of patch0
        ########
        
        i0 = Nxi + NG # Last ghost cell of xi edge of patch0             
        j1 = NG  # First active cell of eta edge of patch1 
                
        xi0 = xi_grid[i0, :] + 0.0 * dxi
        eta0 = eta_grid[i0, :] + 0.5 * deta
        
        xi1, eta1 = transform_coords(patch0, patch1, xi0, eta0)

        E1 = interp(field1[patch1, :, j1], xi_yee, xi1)
        E2 = interp(field2[patch1, :, j1], xi_yee, xi1)

        field2[patch0, i0, NG:(Nxi + NG)] = transform(patch1, patch0, xi1, eta1, E1, E2)[1][NG:(Nxi + NG)]

        #########
        # Communicate fields from xi right edge of patch0 to eta left edge patch1
        ########
        
        i0 = Nxi + NG - 1 # Last active cell of xi edge of patch0                   
        j1 = NG - 1 # First ghost cell on eta edge of patch1 
                
        xi1 = xi_grid[:, j1] + 0.0 * dxi
        eta1 = eta_grid[:, j1] + 0.5 * deta
        
        xi0, eta0 = transform_coords(patch1, patch0, xi1, eta1)
    
        E1 = interp(field1[patch0, i0, :], eta, eta0)
        E2 = interp(field2[patch0, i0, :], eta_yee, eta0)
    
        field2[patch1, NG:(Nxi + NG + 1), j1] = transform(patch0, patch1, xi0, eta0, E1, E2)[1][NG:(Nxi + NG + 1)]

        if (typ == "vect"):
        
            i0 = Nxi + NG # Last ghost cell of xi edge of patch0             
            j1 = NG  # First active cell of eta edge of patch1 
            
            xi0 = xi_grid[i0, :] + 0.0 * dxi
            eta0 = eta_grid[i0, :] + 0.0 * deta
            
            xi1, eta1 = transform_coords(patch0, patch1, xi0, eta0)
            
            Er1 = interp(Er[patch1, :, j1], xi, xi1)
            Er[patch0, i0, (NG + 1):(Nxi + NG)] = Er1[(NG + 1):(Nxi + NG)]

    # elif (top == 'xy'):

    #     compute_E_edge(patch0, patch1, typ, top)
 
    #     #########
    #     # Communicate fields from eta left edge of patch1 to xi right edge of patch0
    #     ########
        
    #     i0 = Nxi + NG # Last ghost cell of xi edge of patch0             
    #     j1 = NG  # First active cell of eta edge of patch1 
                
    #     xi0 = xi_grid[i0, :] + 0.0 * dxi
    #     eta0 = eta_grid[i0, :] + 0.5 * deta
    #     xi1, eta1 = transform_coords(patch0, patch1, xi0, eta0)

    #     E1 = interp(field1[patch1, :, j1], xi_yee, xi1)
    #     E2 = interp(E_edge_eta_1, xi_yee, xi1)

    #     field2[patch0, i0, NG:(Nxi + NG)] = transform(patch1, patch0, xi1, eta1, E1, E2)[1][NG:(Nxi + NG)]

    #     #########
    #     # Communicate fields from xi right edge of patch0 to eta left edge patch1
    #     ########
        
    #     i0 = Nxi + NG - 1 # Last active cell of xi edge of patch0                   
    #     j1 = NG - 1 # First ghost cell on eta edge of patch1 
                
    #     xi1 = xi_grid[:, j1] + 0.0 * dxi
    #     eta1 = eta_grid[:, j1] + 0.5 * deta
        
    #     xi0, eta0 = transform_coords(patch1, patch0, xi1, eta1)
    
    #     E1 = interp(field1[patch0, i0, :], eta, eta0)
    #     E2 = interp(E_edge_eta_0, eta, eta0)
    
    #     field2[patch1, NG:(Nxi + NG + 1), j1] = transform(patch0, patch1, xi0, eta0, E1, E2)[1][NG:(Nxi + NG + 1)]

    elif (top == 'yy'):

        #########
        # Communicate fields from eta left edge of patch1 to eta right edge of patch0
        ########
        
        j0 = Neta + NG # Last ghost cell of eta edge of patch0
        j1 = NG # First active cell of eta edge of patch1
                
        xi0 = xi_grid[:, j0] + 0.5 * dxi
        eta0 = eta_grid[:, j0] + 0.0 * deta
        
        xi1, eta1 = transform_coords(patch0, patch1, xi0, eta0)
        
        E1 = interp(field1[patch1, :, j1], xi_yee, xi1)
        E2 = interp(field2[patch1, :, j1], xi, xi1)
        
        field1[patch0, NG:(Nxi + NG), j0] = transform(patch1, patch0, xi1, eta1, E1, E2)[0][NG:(Nxi + NG)]

        #########
        # Communicate fields from eta right edge of patch0 to eta left edge patch1
        ########
        
        j0 = Neta + NG - 1 # Last active cell of eta edge of patch0
        j1 = NG - 1 # First ghost cell of eta edge of patch1
                
        xi1 = xi_grid[:, j1] + 0.0 * dxi
        eta1 = eta_grid[:, j1] + 0.5 * deta
        
        xi0, eta0 = transform_coords(patch1, patch0, xi1, eta1)
        
        E1 = (interp(field1[patch0, :, j0], xi_yee, xi0)+interp(field1[patch0, :, j0 + 1], xi_yee, xi0)) / 2.0
        E2 = interp(field2[patch0, :, j0], xi, xi0)
        
        field2[patch1, NG:(Nxi + NG), j1] = transform(patch0, patch1, xi0, eta0, E1, E2)[1][NG:(Nxi + NG)]            

        if (typ == "vect"):

            j0 = Neta + NG # Last ghost cell of eta edge of patch0
            j1 = NG # First active cell of eta edge of patch1
            
            xi0 = xi_grid[:, j0] + 0.0 * dxi
            eta0 = eta_grid[:,j0] + 0.0 * deta
            
            xi1, eta1 = transform_coords(patch0, patch1, xi0, eta0)
            
            Er1 = interp(Er[patch1, :, j1], xi, xi1)
            Er[patch0, NG:(Nxi + NG), j0] = Er1[NG:(Nxi + NG)]
        
    elif (top == 'yx'):

        #########
        # Communicate fields from eta left edge of patch1 to xi right edge of patch0
        ########
        
        j0 = Neta + NG # Last ghost cell of eta edge of patch0
        i1 = NG # First active cell of xi edge of patch1
                
        xi0 = xi_grid[:, j0] + 0.5 * dxi
        eta0 = eta_grid[:, j0] + 0.0 * deta
        
        xi1, eta1 = transform_coords(patch0, patch1, xi0, eta0)
        
        E1 = interp(field1[patch1, i1, :], eta, eta1)
        E2 = interp(field2[patch1, i1, :], eta_yee, eta1)
        
        field1[patch0, NG:(Nxi + NG), j0] = transform(patch1, patch0, xi1, eta1, E1, E2)[0][NG:(Nxi + NG)]

        #########
        # Communicate fields from eta right edge of patch0 to xi left edge patch1
        ########
        
        j0 = Neta + NG - 1 # Last active cell of eta edge of patch0
        i1 = NG - 1 # First ghost cell on xi edge of patch1
                
        xi1 = xi_grid[i1, :] + 0.5 * dxi
        eta1 = eta_grid[i1, :] + 0.0 * deta
        
        xi0, eta0 = transform_coords(patch1, patch0, xi1, eta1)
                
        E1 = interp(0.5 * (field1[patch0, :, j0] + field1[patch0, :, j0 + 1]), xi_yee, xi0)
        E2 = interp(field2[patch0, :, j0], xi, xi0)

        field1[patch1, i1, NG:(Nxi + NG + 1)] = transform(patch0, patch1, xi0, eta0, E1, E2)[0][NG:(Nxi + NG + 1)]

        if (typ == "vect"):
    
            j0 = Neta + NG # Last ghost cell of eta edge of patch0
            i1 = NG # First active cell of xi edge of patch1
            
            xi0 = xi_grid[:, j0] + 0.0 * dxi
            eta0 = eta_grid[:, j0] + 0.0 * deta
            
            xi1, eta1 = transform_coords(patch0, patch1, xi0, eta0)
            
            Er1 = interp(Er[patch1, i1, :], eta, eta1)
            Er[patch0, (NG + 1):(Nxi + NG), j0] = Er1[(NG + 1):(Nxi + NG)]

    else:
        return
            
########
# Communication of covariant or contravariant B
# patch0 has the open boundary
# patch1 has the closed boundary 
########

def communicate_B_patch(patch0, patch1, typ):

    if (typ == "vect"):
        field1 = B1u
        field2 = B2u
    if (typ == "form"):
        field1 = B1d
        field2 = B2d
    
    transform = (globals()["transform_" + typ])

    top = topology[patch0, patch1]
    
    if (top == 'xx'):
            
        #########
        # Communicate fields from xi left edge of patch1 to xi right edge of patch0
        ########
        
        i0 = Nxi + NG # Last ghost cell of xi edge of patch0             
        i1 = NG  # First active cell of xi edge of patch1 
                
        xi0 = xi_grid[i0, :] + 0.0 * dxi
        eta0 = eta_grid[i0, :] + 0.5 * deta
        
        xi1, eta1 = transform_coords(patch0, patch1, xi0, eta0)
        
        B1 = interp(field1[patch1, i1, :], eta_yee, eta1)
        B2 = interp(field2[patch1, i1, :], eta, eta1)
        
        field1[patch0, i0, NG:(Nxi + NG)] = transform(patch1, patch0, xi1, eta1, B1, B2)[0][NG:(Nxi + NG)]
        
        #########
        # Communicate fields from xi right edge of patch0 to xi left edge patch1
        ########
        
        i0 = Nxi + NG - 1 # Last active cell of xi edge of patch0                   
        i1 = NG - 1 # First ghost cell of xi edge of patch1   
                
        xi1 = xi_grid[i1, :] + 0.5 * dxi
        eta1 = eta_grid[i1, :] + 0.0 * deta
        
        xi0, eta0 = transform_coords(patch1, patch0, xi1, eta1)
        
        B1 = (interp(field1[patch0, i0, :], eta_yee, eta0) + interp(field1[patch0, i0 + 1, :], eta_yee, eta0)) / 2.0
        B2 = interp(field2[patch0, i0, :], eta, eta0)

        field2[patch1, i1, NG:(Nxi + NG)] = transform(patch0, patch1, xi0, eta0, B1, B2)[1][NG:(Nxi + NG)]
        
        if (typ == "vect"):

            i0 = Nxi + NG - 1 # Last active cell of xi edge of patch0                   
            i1 = NG - 1 # First ghost cell of xi edge of patch1   
            
            xi1 = xi_grid[i1,:] + 0.5 * dxi
            eta1 = eta_grid[i1,:] + 0.5 * deta
            
            xi0, eta0 = transform_coords(patch1, patch0, xi1, eta1)
            
            Br0 = interp(Br[patch0, i0, :], eta_yee, eta0)
            Br[patch1, i1, NG:(Nxi + NG)] = Br0[NG:(Nxi + NG)]
    
    # elif (top == 'xy'):

    if (top == 'xy'):
            
        #########
        # Communicate fields from eta left edge of patch1 to xi right edge of patch0
        ########
        
        i0 = Nxi + NG # Last ghost cell of xi edge of patch0             
        j1 = NG  # First active cell of eta edge of patch1  
                
        xi0 = xi_grid[i0, :] + 0.0 * dxi
        eta0 = eta_grid[i0, :] + 0.5 * deta
        
        xi1, eta1 = transform_coords(patch0, patch1, xi0, eta0)
        
        B1 = interp(field1[patch1, :, j1], xi, xi1)
        B2 = interp(field2[patch1, :, j1], xi_yee, xi1)
        
        field1[patch0, i0, NG:(Nxi + NG)] = transform(patch1, patch0, xi1, eta1, B1, B2)[0][NG:(Nxi + NG)]
        
        #########
        # Communicate fields from xi right edge of patch0 to eta left edge patch1
        ########
        
        i0 = Nxi + NG - 1 # Last active cell of xi edge of patch0                   
        j1 = NG - 1 # First ghost cell on eta edge of patch1
                
        xi1 = xi_grid[:, j1] + 0.0 * dxi
        eta1 = eta_grid[:, j1] + 0.5 * deta
        
        xi0, eta0 = transform_coords(patch1, patch0, xi1, eta1)
        
        B1 = (interp(field1[patch0, i0, :], eta_yee, eta0) + interp(field1[patch0, i0 + 1, :], eta_yee, eta0)) / 2.0
        B2 = interp(field2[patch0, i0, :], eta, eta0)
        
        field1[patch1, NG:(Nxi + NG + 1), j1] = transform(patch0, patch1, xi0, eta0, B1, B2)[0][NG:(Nxi + NG + 1)]
        
        if (typ == "vect"):

            i0 = Nxi + NG - 1 # Last active cell of xi edge of patch0                   
            j1 = NG - 1 # First ghost cell on eta edge of patch1
            
            xi1 = xi_grid[:,j1] + 0.5 * dxi
            eta1 = eta_grid[:,j1] + 0.5 * deta
            
            xi0, eta0 = transform_coords(patch1, patch0, xi1, eta1)
            
            Br0 = interp(Br[patch0, i0, :], eta_yee, eta0)
            Br[patch1, NG:(Nxi + NG), j1] = Br0[NG:(Nxi + NG)]
            
    # elif (top == 'yy'):

    if (top == 'yy'):        

        #########
        # Communicate fields from eta left edge of patch1 to eta right edge of patch0
        ########
        
        j0 = Neta + NG # Last ghost cell of eta edge of patch0
        j1 = NG # First active cell of eta edge of patch1
                
        xi0 = xi_grid[:, j0] + 0.5 * dxi
        eta0 = eta_grid[:, j0] + 0.0 * deta
        
        xi1, eta1 = transform_coords(patch0, patch1, xi0, eta0)
        
        B1 = interp(field1[patch1, :, j1], xi, xi1)
        B2 = interp(field2[patch1, :, j1], xi_yee, xi1)
        
        field2[patch0, NG:(Nxi + NG), j0] = transform(patch1, patch0, xi1, eta1, B1, B2)[1][NG:(Nxi + NG)]      

    #     #########
    #     # Communicate fields from eta right edge of patch0 to eta left edge patch1
    #     ########
        
        j0 = Neta + NG - 1 # Last active cell of eta edge of patch0
        j1 = NG - 1 # First ghost cell of eta edge of patch1
                
        xi1 = xi_grid[:, j1] + 0.0 * dxi
        eta1 = eta_grid[:, j1] + 0.5 * deta
        
        xi0, eta0 = transform_coords(patch1, patch0, xi1, eta1)
        
        B1 =  interp(field1[patch0, :, j0], xi, xi0)
        B2 = (interp(field2[patch0, :, j0], xi_yee, xi0) + interp(field2[patch0, :, j0 + 1], xi_yee, xi0)) / 2.0
        
        field1[patch1, NG:(Nxi + NG), j1] = transform(patch0, patch1, xi0, eta0, B1, B2)[0][NG:(Nxi + NG)]
        
        if (typ == "vect"):

            j0 = Neta + NG - 1 # Last active cell of eta edge of patch0
            j1 = NG - 1 # First ghost cell of eta edge of patch1
            
            xi1 = xi_grid[:,j1] + 0.5 * dxi
            eta1 = eta_grid[:,j1] + 0.5 * deta
            
            xi0, eta0 = transform_coords(patch1, patch0, xi1, eta1)
            
            Br0 = interp(Br[patch0, :, j0], xi_yee, xi0)
            Br[patch1, NG:(Nxi + NG), j1] = Br0[NG:(Nxi + NG)]
            
    # elif (top == 'yx'):

    if (top == 'yx'):

        #########
        # Communicate fields from eta left edge of patch1 to xi right edge of patch0
        ########
        
        j0 = Neta + NG # Last ghost cell of eta edge of patch0
        i1 = NG # First active cell of xi edge of patch1            
        
        xi0 = xi_grid[:, j0] + 0.5 * dxi
        eta0 = eta_grid[:, j0] + 0.0 * deta
        
        xi1, eta1 = transform_coords(patch0, patch1, xi0, eta0)
        
        B1 = interp(field1[patch1, i1, :], eta_yee, eta1)
        B2 = interp(field2[patch1, i1, :], eta, eta1)
        
        field2[patch0, NG:(Nxi + NG), j0] = transform(patch1, patch0, xi1, eta1, B1, B2)[1][NG:(Nxi + NG)]
        
        #########
        # Communicate fields from eta right edge of patch0 to xi left edge patch1
        ########
        
        j0 = Neta + NG - 1 # Last active cell of eta edge of patch0
        i1 = NG - 1 # First ghost cell on xi edge of patch1
                
        xi1 = xi_grid[i1, :] + 0.5 * dxi
        eta1 = eta_grid[i1, :] + 0.0 * deta
        
        xi0, eta0 = transform_coords(patch1, patch0, xi1, eta1)
        
        B1 =  interp(field1[patch0, :, j0], xi, xi0)
        B2 = (interp(field2[patch0, :, j0], xi_yee, xi0) + interp(field2[patch0, :, j0 + 1], xi_yee, xi0)) / 2.0
        
        field2[patch1, i1, NG:(Nxi + NG + 1)] = transform(patch0, patch1, xi0, eta0, B1, B2)[1][NG:(Nxi + NG + 1)]
        
        if (typ == "vect"):

            j0 = Neta + NG - 1 # Last active cell of eta edge of patch0
            i1 = NG - 1 # First ghost cell on xi edge of patch1
            
            xi1 = xi_grid[i1, :] + 0.5 * dxi
            eta1 = eta_grid[i1, :] + 0.5 * deta
            
            xi0, eta0 = transform_coords(patch1, patch0, xi1, eta1)
            
            Br0 = interp(Br[patch0, :, j0], xi_yee, xi0)
            Br[patch1, i1, NG:(Nxi + NG)] = Br0[NG:(Nxi + NG)]
            
    else:
        return

########
# Update Er at two special poles
# Inteprolated right now. 
# To do: self-consistent evolution.
########

def update_poles():

    Er_mean_1 = (Er[Sphere.B, Nxi + NG - 1, NG]  + Er[Sphere.C, Nxi + NG - 1, NG]  + Er[Sphere.S, Nxi + NG - 1, NG])  / 3.0
    Er_mean_2 = (Er[Sphere.A, NG, Neta + NG - 1] + Er[Sphere.D, NG, Neta + NG - 1] + Er[Sphere.N, NG, Neta + NG - 1]) / 3.0
    Er[Sphere.A, NG, Neta + NG] = Er_mean_2
    Er[Sphere.B, Nxi + NG, NG]  = Er_mean_1
    Er[Sphere.C, Nxi + NG, NG]  = Er_mean_1
    Er[Sphere.D, NG, Neta + NG] = Er_mean_2
    Er[Sphere.N, NG, Neta + NG] = Er_mean_2
    Er[Sphere.S, Nxi + NG, NG]  = Er_mean_1


def update_corner_cells(typ):

    if (typ == "vect"):
        field1 = B1u
        field2 = B2u
    if (typ == "form"):
        field1 = B1d
        field2 = B2d

    transform = (globals()["transform_" + typ])
    
    #########
    # BCS
    ########
    
    patch0, patch1 = Sphere.B, Sphere.C
    i0 = Nxi + NG # Last ghost cell of xi edge of patch0             
    j1 = NG  # First active cell of eta edge of patch1 
            
    xi0 = xi_grid[i0, :] + 0.0 * dxi
    eta0 = eta_grid[i0, :] + 0.5 * deta
    
    xi1, eta1 = transform_coords(patch0, patch1, xi0, eta0)
    
    E1 = interp(field1[patch1, :, j1], xi_yee, xi1)
    E2 = interp(field2[patch1, :, j1], xi, xi1)
    # E2 = interp(0.5 * (field2[patch1, :, j1] + field2[patch0, :, j1 - 1]), xi, xi1)
    
    E2_C = transform(patch1, patch0, xi1, eta1, E1, E2)[1][NG - 1]


    patch0, patch1 = Sphere.S, Sphere.B
    i0 = Nxi + NG - 1 # Last active cell of xi edge of patch0                   
    j1 = NG - 1 # First ghost cell on eta edge of patch1 
            
    xi1 = xi_grid[:, j1] + 0.0 * dxi
    eta1 = eta_grid[:, j1] + 0.5 * deta
    
    xi0, eta0 = transform_coords(patch1, patch0, xi1, eta1)

    E1 = interp(field1[patch0, i0, :], eta, eta0)
    E2 = interp(field2[patch0, i0, :], eta_yee, eta0)
    # E2 = (interp(field2[patch0, i0, :], eta_yee, eta0) + interp(field2[patch0, i0 + 1, :], eta_yee, eta0)) / 2.0

    # field2[patch1, NG:(Nxi + NG + 1), j1] = transform(patch0, patch1, xi0, eta0, E1, E2)[1][NG:(Nxi + NG + 1)]
    E2_S = transform(patch0, patch1, xi0, eta0, E1, E2)[1][Nxi + NG]
    
    field2[Sphere.B, Nxi + NG, NG - 1] = 0.5 * (E2_C + E2_S)

    #########
    # SBC
    ########
    
    patch0, patch1 = Sphere.S, Sphere.B
    i0 = Nxi + NG # Last ghost cell of xi edge of patch0             
    j1 = NG  # First active cell of eta edge of patch1 
            
    xi0 = xi_grid[i0, :] + 0.0 * dxi
    eta0 = eta_grid[i0, :] + 0.5 * deta
    
    xi1, eta1 = transform_coords(patch0, patch1, xi0, eta0)
    
    E1 = interp(field1[patch1, :, j1], xi_yee, xi1)
    E2 = interp(field2[patch1, :, j1], xi, xi1)
    # E2 = interp(0.5 * (field2[patch1, :, j1] + field2[patch0, :, j1 - 1]), xi, xi1)
    
    E2_B = transform(patch1, patch0, xi1, eta1, E1, E2)[1][NG - 1]


    patch0, patch1 = Sphere.C, Sphere.S
    i0 = Nxi + NG - 1 # Last active cell of xi edge of patch0                   
    j1 = NG - 1 # First ghost cell on eta edge of patch1 
            
    xi1 = xi_grid[:, j1] + 0.0 * dxi
    eta1 = eta_grid[:, j1] + 0.5 * deta
    
    xi0, eta0 = transform_coords(patch1, patch0, xi1, eta1)

    E1 = interp(field1[patch0, i0, :], eta, eta0)
    E2 = interp(field2[patch0, i0, :], eta_yee, eta0)
    # E2 = (interp(field2[patch0, i0, :], eta_yee, eta0) + interp(field2[patch0, i0 + 1, :], eta_yee, eta0)) / 2.0

    # field2[patch1, NG:(Nxi + NG + 1), j1] = transform(patch0, patch1, xi0, eta0, E1, E2)[1][NG:(Nxi + NG + 1)]
    E2_C = transform(patch0, patch1, xi0, eta0, E1, E2)[1][Nxi + NG]
    
    field2[Sphere.S, Nxi + NG, NG - 1] = 0.5 * (E2_B + E2_C)
        
    #########
    # CSB
    ########
    
    patch0, patch1 = Sphere.C, Sphere.S
    i0 = Nxi + NG # Last ghost cell of xi edge of patch0             
    j1 = NG  # First active cell of eta edge of patch1 
            
    xi0 = xi_grid[i0, :] + 0.0 * dxi
    eta0 = eta_grid[i0, :] + 0.5 * deta
    
    xi1, eta1 = transform_coords(patch0, patch1, xi0, eta0)
    
    E1 = interp(field1[patch1, :, j1], xi_yee, xi1)
    E2 = interp(field2[patch1, :, j1], xi, xi1)
    # E2 = interp(0.5 * (field2[patch1, :, j1] + field2[patch0, :, j1 - 1]), xi, xi1)
    
    E2_B = transform(patch1, patch0, xi1, eta1, E1, E2)[1][NG - 1]


    patch0, patch1 = Sphere.B, Sphere.C
    i0 = Nxi + NG - 1 # Last active cell of xi edge of patch0                   
    j1 = NG - 1 # First ghost cell on eta edge of patch1 
            
    xi1 = xi_grid[:, j1] + 0.0 * dxi
    eta1 = eta_grid[:, j1] + 0.5 * deta
    
    xi0, eta0 = transform_coords(patch1, patch0, xi1, eta1)

    E1 = interp(field1[patch0, i0, :], eta, eta0)
    E2 = interp(field2[patch0, i0, :], eta_yee, eta0)
    # E2 = (interp(field2[patch0, i0, :], eta_yee, eta0) + interp(field2[patch0, i0 + 1, :], eta_yee, eta0)) / 2.0

    # field2[patch1, NG:(Nxi + NG + 1), j1] = transform(patch0, patch1, xi0, eta0, E1, E2)[1][NG:(Nxi + NG + 1)]
    E2_C = transform(patch0, patch1, xi0, eta0, E1, E2)[1][Nxi + NG]
    
    field2[Sphere.C, Nxi + NG, NG - 1] = 0.5 * (E2_B + E2_S)


# %%
# Plotting fields on an unfolded sphere
########

# xi_grid_c, eta_grid_c = unflip_eq(xi_grid[NG:(Nxi + NG), NG:(Neta + NG)], eta_grid[NG:(Nxi + NG), NG:(Neta + NG)])
# xi_grid_d, eta_grid_d = unflip_eq(xi_grid[NG:(Nxi + NG), NG:(Neta + NG)], eta_grid[NG:(Nxi + NG), NG:(Neta + NG)])
# xi_grid_n, eta_grid_n = unflip_po(xi_grid[NG:(Nxi + NG), NG:(Neta + NG)], eta_grid[NG:(Nxi + NG), NG:(Neta + NG)])

xi_grid_c, eta_grid_c = unflip_eq(xi_grid[:, :], eta_grid[:, :])
xi_grid_d, eta_grid_d = unflip_eq(xi_grid[:, :], eta_grid[:, :])
xi_grid_n, eta_grid_n = unflip_po(xi_grid[:, :], eta_grid[:, :])

delta =xi[-1]-xi[0]

def plot_fields_unfolded(it, field, vm):

    fig = P.figure(1, facecolor='w')
    ax = P.subplot(111)

    tab = (globals()[field])
    
    ax.pcolor(xi_grid, eta_grid, tab[Sphere.A, :, :], cmap = "BrBG", vmin = - vm, vmax = vm)
    ax.pcolor(xi_grid + N.pi / 2.0 + 2.0 * dxi, eta_grid, tab[Sphere.B, :, :], cmap = "BrBG", vmin = - vm, vmax = vm)
    ax.pcolor(xi_grid, eta_grid - N.pi / 2.0 - 2.0 * dxi, tab[Sphere.S, :, :], cmap = "BrBG", vmin = - vm, vmax = vm)

    ax.pcolor(xi_grid_c + N.pi + 4.0 * dxi, eta_grid_c, tab[Sphere.C, :, :], cmap = "BrBG", vmin = - vm, vmax = vm)
    ax.pcolor(xi_grid_d - N.pi / 2.0 - .0 * dxi, eta_grid_d, tab[Sphere.D, :, :], cmap = "BrBG", vmin = - vm, vmax = vm)
    ax.pcolor(xi_grid_n, eta_grid_n + N.pi / 2.0 + 2.0 * dxi, tab[Sphere.N, :, :], cmap = "BrBG", vmin = - vm, vmax = vm)

    ax.pcolormesh(xi_grid[NG:(Nxi + NG), NG:(Neta + NG)], eta_grid[NG:(Nxi + NG), NG:(Neta + NG)], tab[Sphere.A, NG:(Nxi + NG), NG:(Neta + NG)], cmap = "RdBu_r", vmin = - vm, vmax = vm)
    ax.pcolormesh(xi_grid[NG:(Nxi + NG), NG:(Neta + NG)] + N.pi / 2.0 + 2.0 * dxi, eta_grid[NG:(Nxi + NG), NG:(Neta + NG)], tab[Sphere.B, NG:(Nxi + NG), NG:(Neta + NG)], cmap = "RdBu_r", vmin = - vm, vmax = vm)
    ax.pcolormesh(xi_grid[NG:(Nxi + NG), NG:(Neta + NG)], eta_grid[NG:(Nxi + NG), NG:(Neta + NG)] - N.pi / 2.0 - 2.0 * dxi, tab[Sphere.S, NG:(Nxi + NG), NG:(Neta + NG)], cmap = "RdBu_r", vmin = - vm, vmax = vm)

    ax.pcolormesh(xi_grid_c[NG:(Nxi + NG), NG:(Neta + NG)] + N.pi + 4.0 * dxi, eta_grid_c[NG:(Nxi + NG), NG:(Neta + NG)], tab[Sphere.C, NG:(Nxi + NG), NG:(Neta + NG)], cmap = "RdBu_r", vmin = - vm, vmax = vm)
    ax.pcolormesh(xi_grid_d[NG:(Nxi + NG), NG:(Neta + NG)] - N.pi / 2.0 - 2.0 * dxi, eta_grid_d[NG:(Nxi + NG), NG:(Neta + NG)], tab[Sphere.D, NG:(Nxi + NG), NG:(Neta + NG)], cmap = "RdBu_r", vmin = - vm, vmax = vm)
    ax.pcolormesh(xi_grid_n[NG:(Nxi + NG), NG:(Neta + NG)], eta_grid_n[NG:(Nxi + NG), NG:(Neta + NG)] + N.pi / 2.0 + 2.0 * dxi, tab[Sphere.N, NG:(Nxi + NG), NG:(Neta + NG)], cmap = "RdBu_r", vmin = - vm, vmax = vm)

    val=0.5
    ax.plot([xi[NG]-0.5*dxi,xi[NG]-0.5*dxi],[eta[NG]-0.5*deta,eta[Neta+NG-1]-0.5*deta],color='k',lw=val)
    ax.plot([xi[Nxi+NG-1]-0.5*dxi,xi[Nxi+NG-1]-0.5*dxi],[eta[NG]-0.5*deta,eta[Neta+NG-1]-0.5*deta],color='k',lw=val)
    ax.plot([xi[NG]-0.5*dxi,xi[Nxi+NG-1]-0.5*dxi],[eta[NG]-0.5*deta,eta[NG]-0.5*deta],color='k',lw=val)
    ax.plot([xi[NG]-0.5*dxi,xi[Nxi+NG-1]-0.5*dxi],[eta[Neta+NG-1]-0.5*deta,eta[Neta+NG-1]-0.5*deta],color='k',lw=val)

    ax.plot([xi[NG]-0.5*dxi+delta,xi[NG]-0.5*dxi+delta],[eta[NG]-0.5*deta,eta[Neta+NG-1]-0.5*deta],color='k',lw=val)
    ax.plot([xi[Nxi+NG-1]-0.5*dxi+delta,xi[Nxi+NG-1]-0.5*dxi+delta],[eta[NG]-0.5*deta,eta[Neta+NG-1]-0.5*deta],color='k',lw=val)
    ax.plot([xi[NG]-0.5*dxi+delta,xi[Nxi+NG-1]-0.5*dxi+delta],[eta[NG]-0.5*deta,eta[NG]-0.5*deta],color='k',lw=val)
    ax.plot([xi[NG]-0.5*dxi+delta,xi[Nxi+NG-1]-0.5*dxi+delta],[eta[Neta+NG-1]-0.5*deta,eta[Neta+NG-1]-0.5*deta],color='k',lw=val)

    ax.plot([xi[NG]-0.5*dxi,xi[NG]-0.5*dxi],[eta[NG]-0.5*deta-delta,eta[Neta+NG-1]-0.5*deta-delta],color='k',lw=val)
    ax.plot([xi[Nxi+NG-1]-0.5*dxi,xi[Nxi+NG-1]-0.5*dxi],[eta[NG]-0.5*deta-delta,eta[Neta+NG-1]-0.5*deta-delta],color='k',lw=val)
    ax.plot([xi[NG]-0.5*dxi,xi[Nxi+NG-1]-0.5*dxi],[eta[NG]-0.5*deta-delta,eta[NG]-0.5*deta-delta],color='k',lw=val)
    ax.plot([xi[NG]-0.5*dxi,xi[Nxi+NG-1]-0.5*dxi],[eta[Neta+NG-1]-0.5*deta-delta,eta[Neta+NG-1]-0.5*deta-delta],color='k',lw=val)

    ax.plot([xi_grid_c[0,NG]-0.5*dxi+2.0*delta,xi_grid_c[0,NG]-0.5*dxi+2.0*delta],[eta_grid_c[NG,0]+0.5*deta,eta_grid_c[Neta+NG-1,0]+0.5*deta],color='k',lw=val)
    ax.plot([xi_grid_c[0,Nxi+NG-1]-0.5*dxi+2.0*delta,xi_grid_c[0,Nxi+NG-1]-0.5*dxi+2.0*delta],[eta_grid_c[NG,0]+0.5*deta,eta_grid_c[Neta+NG-1,0]+0.5*deta],color='k',lw=val)
    ax.plot([xi_grid_c[0,NG]-0.5*dxi+2.0*delta,xi_grid_c[0,Nxi+NG-1]-0.5*dxi+2.0*delta],[eta_grid_c[NG,0]+0.5*deta,eta_grid_c[NG,0]+0.5*deta],color='k',lw=val)
    ax.plot([xi_grid_c[0,NG]-0.5*dxi+2.0*delta,xi_grid_c[0,Nxi+NG-1]-0.5*dxi+2.0*delta],[eta_grid_c[Neta+NG-1,0]+0.5*deta,eta_grid_c[Neta+NG-1,0]+0.5*deta],color='k',lw=val)

    ax.plot([xi_grid_d[0,NG]-0.5*dxi-delta,xi_grid_d[0,NG]-0.5*dxi-delta],[eta_grid_d[NG,0]+0.5*deta,eta_grid_d[Neta+NG-1,0]+0.5*deta],color='k',lw=val)
    ax.plot([xi_grid_d[0,Nxi+NG-1]-0.5*dxi-delta,xi_grid_d[0,Nxi+NG-1]-0.5*dxi-delta],[eta_grid_d[NG,0]+0.5*deta,eta_grid_d[Neta+NG-1,0]+0.5*deta],color='k',lw=val)
    ax.plot([xi_grid_d[0,NG]-0.5*dxi-delta,xi_grid_d[0,Nxi+NG-1]-0.5*dxi-delta],[eta_grid_d[NG,0]+0.5*deta,eta_grid_d[NG,0]+0.5*deta],color='k',lw=val)
    ax.plot([xi_grid_d[0,NG]-0.5*dxi-delta,xi_grid_d[0,Nxi+NG-1]-0.5*dxi-delta],[eta_grid_d[Neta+NG-1,0]+0.5*deta,eta_grid_d[Neta+NG-1,0]+0.5*deta],color='k',lw=val)

    ax.plot([xi_grid_n[0,NG]+0.5*dxi,xi_grid_n[0,NG]+0.5*dxi],[eta_grid_n[NG,0]-0.5*deta+delta,eta_grid_n[Neta+NG-1,0]-0.5*deta+delta],color='k',lw=val)
    ax.plot([xi_grid_n[0,Nxi+NG-1]+0.5*dxi,xi_grid_n[0,Nxi+NG-1]+0.5*dxi],[eta_grid_n[NG,0]-0.5*deta+delta,eta_grid_n[Neta+NG-1,0]-0.5*deta+delta],color='k',lw=val)
    ax.plot([xi_grid_n[0,NG]+0.5*dxi,xi_grid_n[0,Nxi+NG-1]+0.5*dxi],[eta_grid_n[NG,0]-0.5*deta+delta,eta_grid_n[NG,0]-0.5*deta+delta],color='k',lw=val)
    ax.plot([xi_grid_n[0,NG]+0.5*dxi,xi_grid_n[0,Nxi+NG-1]+0.5*dxi],[eta_grid_n[Neta+NG-1,0]-0.5*deta+delta,eta_grid_n[Neta+NG-1,0]-0.5*deta+delta],color='k',lw=val)

    ax.plot(xi[NG],eta[NG],'ro') 
    ax.plot(xi[NG] + delta, eta[NG],'ro') 
    ax.plot(xi[NG], eta[NG] - delta,'ro') 
    ax.plot(xi_grid_c[NG,0] + 2.0 * delta, eta_grid_c[0,NG],'ro') 
    ax.plot(xi_grid_d[NG,0] - delta,eta_grid_d[0,NG],'ro') 
    ax.plot(xi_grid_n[NG,0], eta_grid_n[0,NG] + delta,'ro') 

    figsave_png(fig, "snapshots_2d/" + field + "_" + str(it))

    P.close('all')
    # ax.cla()


# %%
# Plotting fields on a sphere
########

vmi = 0.2
xf = N.zeros_like(Er)
yf = N.zeros_like(Er)
zf = N.zeros_like(Er)

th0 = N.zeros_like(xi_grid)
ph0 = N.zeros_like(xi_grid)

phi_s = N.linspace(0, N.pi, 2*50)
theta_s = N.linspace(0, 2*N.pi, 2*50)
theta_s, phi_s = N.meshgrid(theta_s, phi_s)
x_s = 0.95 * N.sin(phi_s) * N.cos(theta_s)
y_s = 0.95 * N.sin(phi_s) * N.sin(theta_s)
z_s = 0.95 * N.cos(phi_s)

cmap_plot="RdBu_r"
norm_plot = matplotlib.colors.Normalize(vmin = - vmi, vmax = vmi)
m = matplotlib.cm.ScalarMappable(cmap = cmap_plot, norm = norm_plot)
m.set_array([])

for patch in range(6):
    
    fcoord = (globals()["coord_" + sphere[patch] + "_to_sph"])
    for i in range(Nxi + 2 * NG):
        for j in range(Neta + 2 * NG):
            th0[i, j], ph0[i, j] = fcoord(xi_grid[i, j], eta_grid[i, j])  
    xf[patch, :, :] = N.sin(th0) * N.cos(ph0)
    yf[patch, :, :] = N.sin(th0) * N.sin(ph0)
    zf[patch, :, :] = N.cos(th0)
        
def plot_fields_sphere(it, field, res, vmi):

    tab = (globals()[field])

    fig, ax = P.subplots(1,2, subplot_kw={"projection": "3d"}, figsize = fig_size, facecolor = 'w')
    
    for patch in range(6):
        for axs in ax:

            fcolors = m.to_rgba(tab[patch, NG:(Nxi + NG), NG:(Neta + NG)]) 
            axs.plot_surface(x_s, y_s, z_s, rstride=res, cstride=res, shade=False, color = 'black', zorder = 0)
            sf = axs.plot_surface(xf[patch, NG:(Nxi + NG), NG:(Neta + NG)], yf[patch, NG:(Nxi + NG), NG:(Neta + NG)],
                    zf[patch, NG:(Nxi + NG), NG:(Neta + NG)],
                    rstride = res, cstride = res, shade = False,
                    facecolors = fcolors, norm = norm_plot, zorder = 2)
            sf = axs.plot_surface(xf[patch, NG:(Nxi + NG), NG:(Neta + NG)], yf[patch, NG:(Nxi + NG), NG:(Neta + NG)],
                    zf[patch, NG:(Nxi + NG), NG:(Neta + NG)],
                    rstride = res, cstride = res, shade = False,
                    facecolors = fcolors, norm = norm_plot, zorder = 2)
            
    ax[0].set_box_aspect((1,1,1))
    ax[0].view_init(elev=40, azim=45)
    
    ax[1].set_box_aspect((1,1,1))
    ax[1].view_init(elev=-40, azim=180+45)
    
    fig.tight_layout(pad=1.0)
        
    figsave_png(fig, "snapshots_2d/sphere_" + field + "_" + str(it))

    P.close("all")

# %%
# Source current
########

theta0, phi0 = 90.0 / 360.0 * 2.0 * N.pi, 180.0 / 360.0 * 2.0 * N.pi # Center of the wave packet !60
x0 = N.sin(theta0) * N.cos(phi0)
y0 = N.sin(theta0) * N.sin(phi0)
z0 = N.cos(theta0)

def shape_packet(x, y, z, width):
    return N.exp(- y * y / (width * width)) * N.exp(- x * x / (width * width)) * N.exp(- z * z / (width * width)) 

w = 0.1        # Radius of wave packet
omega = 20.0   # Frequency of current
J0 = 1.0       # Current amplitude
p0 = Sphere.C  # Patch location of antenna

Jr_tot = N.zeros_like(Er)

for patch in range(6):

    fcoord0 = (globals()["coord_" + sphere[patch] + "_to_sph"])
    for i in range(Nxi + 2 * NG):
        for j in range(Neta + 2 * NG):
            th, ph = fcoord0(xi_grid[i, j], eta_grid[i, j])
            x = N.sin(th) * N.cos(ph)
            y = N.sin(th) * N.sin(ph)
            z = N.cos(th)        
     
            Jr_tot[patch, i, j] = J0 * shape_packet(x - x0, y - y0, z - z0, w) * int(patch == p0)

def Jr(it, patch, i0, i1, j0, j1):
    return 0.0 # Jr_tot[patch, i0:i1, j0:j1] * N.sin(omega * dt * it) * (1 + N.tanh(20 - it/5.))/2.


# %%
# Initialization
########

B1u0 = N.zeros_like(B1u)
B2u0 = N.zeros_like(B2u)
E1u0 = N.zeros_like(E1u)
E2u0 = N.zeros_like(E2u)

for patch in range(6):

    fvec = (globals()["vec_sph_to_" + sphere[patch]])
    fveci = (globals()["vec_"+sphere[patch]+"_to_sph"])
    fcoord = (globals()["coord_" + sphere[patch] + "_to_sph"])

    for i in range(Nxi + 2 * NG):
        for j in range(Neta + 2 * NG):

            r0 = 1.0
            th0, ph0 = fcoord(xi_grid[i, j] + 0.5 * dxi, eta_grid[i, j] + 0.5 * deta)
            Br[patch, i, j] = 0.0 #N.sin(th0)**3 * N.cos(3.0 * ph0)

            BtTMP = 0.0
            BpTMP = 0.0

            th0, ph0 = fcoord(xi_grid[i, j] + 0.5 * dxi, eta_grid[i, j])
            BCStmp = fvec(th0, ph0, BtTMP, BpTMP)
            B2u[patch, i, j] = BCStmp[1]
            B2u0[patch, i, j] = BCStmp[1]
            
            th0, ph0 = fcoord(xi_grid[i, j], eta_grid[i, j] + 0.5 * deta)
            BCStmp = fvec(th0, ph0, BtTMP, BpTMP)
            B1u[patch, i, j] = BCStmp[0]
            B1u0[patch, i, j] = BCStmp[0]

            Er[patch, i, j] = 0.0

            EtTMP = N.cos(ph0) # N.sin(th0)**3 * N.cos(3.0 * ph0) # 0.0
            EpTMP = N.sin(th0) # N.sin(th0)**3 * N.cos(3.0 * ph0) # 0.0

            th0, ph0 = fcoord(xi_grid[i, j], eta_grid[i, j] + 0.5 * deta)
            ECStmp = fvec(th0, ph0, EtTMP, EpTMP)
            E2u[patch, i, j] = ECStmp[1]
            E2u0[patch, i, j] = ECStmp[1]

            th0, ph0 = fcoord(xi_grid[i, j] + 0.5 * dxi, eta_grid[i, j])
            ECStmp = fvec(th0, ph0, EtTMP, EpTMP)
            E1u[patch, i, j] = ECStmp[0]
            E1u0[patch, i, j] = ECStmp[0]

idump = 0

Nt = 2 # Number of iterations
FDUMP = 1 # Dump frequency

# Figure parameters
scale, aspect = 2.0, 0.7
vm = 0.2
ratio = 2.0
fig_size=deffigsize(scale, aspect)
style = '2d' 

# Define figure
if (style == '2d'):
    fig = P.figure(1, facecolor='w')
    ax = P.subplot(111)
elif (style == '3d'):
    fig, ax = P.subplots(subplot_kw={"projection": "3d"}, figsize = fig_size, facecolor = 'w')
    ax.view_init(elev=45, azim=45)
    ax.plot_surface(x_s, y_s, z_s, rstride=1, cstride=1, shade=False, color = 'grey', zorder = 0)

# %%
# Main routine
########

# index= N.array([index_row, index_col])
# perm_index = N.random.permutation(index.T)

# index_row = perm_index[:, 0]
# index_col = perm_index[:, 1]

int_energy = N.zeros((6, Nt))

for it in tqdm(range(Nt), "Progression"):
    if ((it % FDUMP) == 0):
        plot_fields_unfolded(idump, "E1u", 2.0)
        plot_fields_unfolded(idump, "E2u", 2.0)
        # plot_fields_unfolded(idump, "E1u", 1.0)
        idump += 1

    # for p in range(6):
    #     contra_to_cov_B(p)

    # for i in range(n_zeros):
    #     p0, p1 = index_row[i], index_col[i]
    #     communicate_B_patch_cov(p0, p1)

    # for p in range(6):
    #     push_E(it, p)

    # update_poles()
        
    for i in range(n_zeros):
        p0, p1 = index_row[i], index_col[i]
        communicate_E_patch_contra(p0, p1)

    # for i in range(n_zeros):
    #     p0, p1 = index_row[i], index_col[i]
    #     communicate_E_corner(p0, p1, "vect")
    #     communicate_E_corner(p0, p1, "form")
    
    # for p in range(6):
    #     contra_to_cov_E(p)
        
    # for i in range(n_zeros):
    #     p0, p1 = index_row[i], index_col[i]
    #     communicate_E_patch_cov(p0, p1)
         
    # for p in range(6):        
    #     push_B(it, p)

    # for i in range(n_zeros):
    #     p0, p1 = index_row[i], index_col[i]
    #     communicate_B_patch_contra(p0, p1)

    # for p in range(6):
    #     dens_energy = Er[p, NG:Nxi+NG, NG:Neta+NG]**2 \
    #                       + E1u[p, NG:Nxi+NG, NG:Neta+NG] * E1d[p, NG:Nxi+NG, NG:Neta+NG] \
    #                       + E2u[p, NG:Nxi+NG, NG:Neta+NG] * E2d[p, NG:Nxi+NG, NG:Neta+NG] \
    #                       + Br[p, NG:Nxi+NG, NG:Neta+NG] * Br[p, NG:Nxi+NG, NG:Neta+NG]   \
    #                       + B1u[p, NG:Nxi+NG, NG:Neta+NG] * B1d[p, NG:Nxi+NG, NG:Neta+NG] \
    #                       + B2u[p, NG:Nxi+NG, NG:Neta+NG] * B2d[p, NG:Nxi+NG, NG:Neta+NG] \
    
    #     int_energy[p, it] = spi.simps(spi.simps(dens_energy, dx = dxi, axis = 0), dx = deta)
    

# %%
# Energy diagnostic
########

time = dt * N.arange(Nt)

# Figure parameters
scale, aspect = 1.0, 0.7
ratio = 2.0
fig_size=deffigsize(scale, aspect)

fig = P.figure(2, figsize = fig_size, facecolor='w')
ax = P.subplot(111)

# for p in range(6):
#     P.plot(time, int_energy[p, :], label = r"Patch {}".format(sphere[p]))

P.plot(time, N.sum(int_energy, axis=0))

P.xlabel(r"$t$")
P.ylabel(r"Electromagnetic energy")

# P.legend(loc = "upper right", ncol = 1)

P.grid(True, ls='--')

# figsave_png(fig, "./total_energy")