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
cfl = 0.7
Nxi = 256 # Number of cells
Nxi_int = Nxi + 1 # Number of integer points
Nxi_half = Nxi + 2 # NUmber of hlaf-step points
Neta = 256 # Number of cells
Neta_int = Neta + 1 # Number of integer points
Neta_half = Neta + 2 # NUmber of hlaf-step points

xi_min, xi_max = - 0.5, 0.5
dxi = (xi_max - xi_min) / Nxi
eta_min, eta_max = - 0.5, 0.5
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
    
    for i in range(2, Nxi_int - 2):
        dBrd1[p, i, :] = (Br[p, i + 1, :] - Br[p, i, :]) / dxi

    dBrd2[p, :, 0] = (- 0.5 * Br[p, :, 0] + 0.25 * Br[p, :, 1] + 0.25 * Br[p, :, 2]) / deta / P_int_2[0]
    dBrd2[p, :, 1] = (- 0.5 * Br[p, :, 0] - 0.25 * Br[p, :, 1] + 0.75 * Br[p, :, 2]) / deta / P_int_2[1]

    dBrd2[p, :, Nxi_int - 2] = (- 0.75 * Br[p, :, -3] + 0.25 * Br[p, :, -2] + 0.5 * Br[p, :, -1]) / deta / P_int_2[Nxi_int - 2]
    dBrd2[p, :, Nxi_int - 1] = (- 0.25 * Br[p, :, -3] - 0.25 * Br[p, :, -2] + 0.5 * Br[p, :, -1]) / deta / P_int_2[Nxi_int - 1]

    for j in range(2, Neta_int - 2):
        dBrd2[p, :, j] = (Br[p, :, j + 1] - Br[p, :, j]) / deta

def compute_diff_E(p):

    dE2d1[p, 0, :] = (- 0.5 * E2d[p, 0, :] + 0.5 * E2d[p, 1, :]) / dxi / P_half_2[0]
    dE2d1[p, 1, :] = (- 0.25 * E2d[p, 0, :] + 0.25 * E2d[p, 1, :]) / dxi / P_half_2[1]
    dE2d1[p, 2, :] = (- 0.25 * E2d[p, 0, :] - 0.75 * E2d[p, 1, :] + E2d[p, 2, :]) / dxi / P_half_2[2]

    dE2d1[p, Nxi_half - 3, :] = (- E2d[p, -3, :] + 0.75 * E2d[p, -2, :] + 0.25 * E2d[p, -1, :]) / dxi / P_half_2[Nxi_half - 3]
    dE2d1[p, Nxi_half - 2, :] = (- 0.25 * E2d[p, -2, :] + 0.25 * E2d[p, -1, :]) / dxi / P_half_2[Nxi_half - 2]
    dE2d1[p, Nxi_half - 1, :] = (- 0.5 * E2d[p, -2, :] + 0.5 * E2d[p, -1, :]) / dxi / P_half_2[Nxi_half - 1]

    for i in range(3, Nxi_half - 3):
        dE2d1[p, i, :] = (E2d[p, i, :] - E2d[p, i - 1, :]) / dxi

    dE1d2[p, :, 0] = (- 0.5 * E1d[p, :, 0] + 0.5 * E1d[p, :, 1]) / dxi / P_half_2[0]
    dE1d2[p, :, 1] = (- 0.25 * E1d[p, :, 0] + 0.25 * E1d[p, :, 1]) / dxi / P_half_2[1]
    dE1d2[p, :, 2] = (- 0.25 * E1d[p, :, 0] - 0.75 * E1d[p, :, 1] + E1d[p, :, 2]) / dxi / P_half_2[2]

    dE1d2[p, :, Neta_half - 3] = (- E1d[p, :, -3] + 0.75 * E1d[p, :, -2] + 0.25 * E1d[p, :, -1]) / deta / P_half_2[Nxi_half - 3]
    dE1d2[p, :, Neta_half - 2] = (- 0.25 * E1d[p, :, -2] + 0.25 * E1d[p, :, -1]) / deta / P_half_2[Nxi_half - 2]
    dE1d2[p, :, Neta_half - 1] = (- 0.5 * E1d[p, :, -2] + 0.5 * E1d[p, :, -1]) / deta / P_half_2[Nxi_half - 1]

    for j in range(3, Neta_half - 3):
        dE1d2[p, :, j] = (E1d[p, :, j] - E1d[p, :, j - 1]) / deta

Jz = N.zeros_like(Br)
Jz[0, :, :] = 0.0 * N.exp(- (xBr_grid**2 + yBr_grid**2) / 0.1**2)

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
        Br[p, :, :] += dt * Jz[p, :, :] * N.sin(10.0 * it * dt) * N.exp(- it * dt / 1.0)

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

sig_ref = 1.0 / dt # 200.0
sig_in  = 0.5 / dt # 200.0
sig_abs = 200.0 

# Vertical interface inner boundary 

def interface_B():

    Br[0, -1, :] -= dt * sig_in * (Ey[1, 0, :] - Ey[0, -1, :]) / P_half_2[-1]
    Br[1, 0, :]  -= dt * sig_in * (Ey[1, 0, :] - Ey[0, -1, :]) / P_half_2[0]

def interface_E():

    Ey[0, -1, :] -= dt * sig_in * (Br[1, 0, :] - Br[0, -1, :]) / P_half_2[-1]
    Ey[1, 0, :]  -= dt * sig_in * (Br[1, 0, :] - Br[0, -1, :]) / P_half_2[0]

# Perfectly conducting outer boundaries    

def BC_conducting_B():

    Br[0, :, 0]  += dt * sig_abs * E1u[0, :, 0] / P_half_2[0]
    Br[0, :, -1] -= dt * sig_abs * E1u[0, :, -1] / P_half_2[-1]
    Br[0, 0, :]  -= dt * sig_abs * E2u[0, 0, :] / P_half_2[0]

    Br[0, -1, :] += dt * sig_abs * E2u[0, -1, :] / P_half_2[-1]

    # Br[1, :, 0]  += dt * sig_abs * E1d[1, :, 0] / P_half_2[0]
    # Br[1, :, -1] -= dt * sig_abs * E1d[1, :, -1] / P_half_2[-1]
    # Br[1, -1, :] += dt * sig_abs * E2d[1, -1, :] / P_half_2[-1]
    
    return

def BC_conducting_E():
    return

# Absorbing outer boundaries

def BC_absorbing_B():

    Br[0, :, 0]  += dt * sig_abs * (E1u[0, :, 0] - Br[0, :, 0]) / P_half_2[0]
    Br[0, :, -1] -= dt * sig_abs * (E1u[0, :, -1] + Br[0, :, -1]) / P_half_2[-1]
    Br[0, 0, :]  -= dt * sig_abs * (E2u[0, 0, :] + Br[0, 0, :]) / P_half_2[0]
    
    Br[0, -1, :] += dt * sig_abs * (E2u[0, -1, :] - Br[0, -1, :]) / P_half_2[-1]

    # Br[1, :, 0]  += dt * sig_abs * (E1u[1, :, 0] - Br[1, :, 0]) / P_half_2[0]
    # Br[1, :, -1] -= dt * sig_abs * (E1u[1, :, -1] + Br[1, :, -1]) / P_half_2[-1]
    # Br[1, -1, :] += dt * sig_abs * (E2u[1, -1, :] - Br[1, -1, :]) / P_half_2[-1]
    
def BC_absorbing_E():

    E1u[0, :, 0]  -= dt * sig_abs * (E1u[0, :, 0] - Br[0, :, 0]) / P_half_2[0]
    E1u[0, :, -1] -= dt * sig_abs * (E1u[0, :, -1] + Br[0, :, -1]) / P_half_2[-1]    
    E2u[0, 0, :]  -= dt * sig_abs * (E2u[0, 0, :] + Br[0, 0, :]) / P_half_2[0]
    # Ey[0, -1, :] -= dt * sig_abs * (Ey[0, -1, :] - Br[0, -1, :]) / P_half_2[-1]

    # E1u[1, :, 0]  -= dt * sig_abs * (E1u[1, :, 0] - Br[1, :, 0]) / P_half_2[0]
    # E1u[1, :, -1] -= dt * sig_abs * (E1u[1, :, -1] + Br[1, :, -1]) / P_half_2[-1]    
    # E2u[1, -1, :]  -= dt * sig_abs * (E2u[1, -1, :] - Br[1, -1, :]) / P_half_2[-1]
    
    return

########
# Initialization
########

amp = 1.0
n_mode = 2
wave = 2.0 * (xi_max - xi_min) / n_mode
Br0 = amp * N.cos(2.0 * N.pi * (xBr_grid - xi_min) / wave) * N.cos(2.0 * N.pi * (yBr_grid - xi_min) / wave)
E1u0 = N.zeros((Nxi_half, Neta_int))
E2u0 = N.zeros((Nxi_int, Neta_half))

for p in range(n_patches):
    Br[p, :, :] = Br0[:, :]
    E1u[p, :, :] = E1u0[:, :]
    E2u[p, :, :] = E2u0[:, :]

########
# Visualization
########

ratio = 0.5

def plot_fields(it):

    fig = P.figure(1, facecolor='w')
    ax = P.subplot(111)

    # P.pcolormesh(xBr_grid - 0.5, yBr_grid, Br[0, :, :], vmin = -1, vmax = 1, cmap = 'RdBu_r')

    # P.pcolormesh(xBr_grid + 0.5, yBr_grid, Br[1, :, :], vmin = -1, vmax = 1, cmap = 'RdBu_r')

    P.pcolormesh(xE1_grid - 0.5, yE1_grid, E1u[0, :, :], vmin = -1, vmax = 1, cmap = 'RdBu_r')
    # P.pcolormesh(xEy_grid + 0.5, yEy_grid, Ey[1, :, :], vmin = -1, vmax = 1, cmap = 'RdBu_r')
    
    P.plot([0, 0],[-1.25, 1.25], color='k')
    
    P.colorbar()
    
    P.ylim((eta_min, eta_max))
    P.xlim((xi_min - 0.5, xi_max + 0.5))
    ax.set_aspect(1.0/ax.get_data_ratio()*ratio)
    
    figsave_png(fig, "snapshots_penalty/fields_" + str(it))

    P.close('all')

########
# Main routine
########

idump = 0

Nt = 10000 # Number of iterations
FDUMP = 100 # Dump frequency
time = dt * N.arange(Nt)
energy = N.zeros((n_patches, Nt))

for it in tqdm(range(Nt), "Progression"):
    if ((it % FDUMP) == 0):
        plot_fields(idump)
        idump += 1

    for p in range(n_patches):
        compute_diff_B(p)
        push_E(p, it)

    # interface_E()
    # BC_absorbing_E()
    BC_conducting_E()
    
    for p in range(n_patches):
        contra_to_cov_E(p)

    for p in range(n_patches):
        compute_diff_E(p)
        push_B(p)

    # interface_B()
    # BC_absorbing_B()
    BC_conducting_B()

    for p in range(n_patches):
        energy[p, it] = dxi * deta * N.sum(Br[p, :, :]**2) \
        + N.sum(E1u[p, :, :]**2) + N.sum(E2u[p, :, :]**2)
for p in range(n_patches):
    P.plot(time, energy[p, :])


