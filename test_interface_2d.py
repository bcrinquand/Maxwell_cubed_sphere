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

# Parameters
cfl = 0.7
Nx = 256 # Number of cells
Nx_int = Nx + 1 # Number of integer points
Nx_half = Nx + 2 # NUmber of hlaf-step points
Ny = 256 # Number of cells
Ny_int = Ny + 1 # Number of integer points
Ny_half = Ny + 2 # NUmber of hlaf-step points

x_min, x_max = - 0.5, 0.5
dx = (x_max - x_min) / Nx
y_min, y_max = - 0.5, 0.5
dy = (y_max - y_min) / Ny

dt = cfl / N.sqrt(1.0 / dx**2 + 1.0 / dy**2)

# Define grids
x_int  = N.linspace(x_min, x_max, Nx_int)
x_half  = N.zeros(Nx_half)
x_half[0] = x_int[0]
x_half[-1] = x_int[-1]
x_half[1:-1] = x_int[:-1] + 0.5 * dx

y_int  = N.linspace(y_min, y_max, Ny_int)
y_half  = N.zeros(Ny_half)
y_half[0] = y_int[0]
y_half[-1] = y_int[-1]
y_half[1:-1] = y_int[:-1] + 0.5 * dy

yBz_grid, xBz_grid = N.meshgrid(y_half, x_half)
yEx_grid, xEx_grid = N.meshgrid(y_int, x_half)
yEy_grid, xEy_grid = N.meshgrid(y_half, x_int)

n_patches = 2    # Ex[1, :, 0]  += dt * sig_abs * (Ex[1, :, 0] - Bz[1, :, 0]) / P_half_2[0]
    # Ex[1, :, -1] -= dt * sig_abs * (Ex[1, :, -1] + Bz[1, :, -1]) / P_half_2[-1]
    # Ey[1, -1, :] += dt * sig_abs * (Ey[1, -1, :] - Bz[1, -1, :]) / P_half_2[-1]


# Define fields
Bz = N.zeros((n_patches, Nx_half, Ny_half))
Ex = N.zeros((n_patches, Nx_half, Ny_int))
Ey = N.zeros((n_patches, Nx_int, Ny_half))
dBzdx = N.zeros((n_patches, Nx_int, Ny_half))
dBzdy = N.zeros((n_patches, Nx_half, Ny_int))
dExdy = N.zeros((n_patches, Nx_half, Ny_half))
dEydx = N.zeros((n_patches, Nx_half, Ny_half))

########
# Pushers
########

P_int_2 = N.ones(Nx_int)
P_int_2[0] = 0.5 
P_int_2[-1] = 0.5 

P_half_2 = N.ones(Nx_half)
P_half_2[0] = 0.5 
P_half_2[1] = 0.25 
P_half_2[2] = 1.25 
P_half_2[-3] = 1.25 
P_half_2[-2] = 0.25 
P_half_2[-1] = 0.5 

def compute_diff_B(p):
    
    dBzdx[p, 0, :] = (- 0.5 * Bz[p, 0, :] + 0.25 * Bz[p, 1, :] + 0.25 * Bz[p, 2, :]) / dx / P_int_2[0]
    dBzdx[p, 1, :] = (- 0.5 * Bz[p, 0, :] - 0.25 * Bz[p, 1, :] + 0.75 * Bz[p, 2, :]) / dx / P_int_2[1]
    
    dBzdx[p, Nx_int - 2, :] = (- 0.75 * Bz[p, -3, :] + 0.25 * Bz[p, -2, :] + 0.5 * Bz[p, -1, :]) / dx / P_int_2[Nx_int - 2]
    dBzdx[p, Nx_int - 1, :] = (- 0.25 * Bz[p, -3, :] - 0.25 * Bz[p, -2, :] + 0.5 * Bz[p, -1, :]) / dx / P_int_2[Nx_int - 1]
    
    for i in range(2, Nx_int - 2):
        dBzdx[p, i, :] = (Bz[p, i + 1, :] - Bz[p, i, :]) / dx

    dBzdy[p, :, 0] = (- 0.5 * Bz[p, :, 0] + 0.25 * Bz[p, :, 1] + 0.25 * Bz[p, :, 2]) / dy / P_int_2[0]
    dBzdy[p, :, 1] = (- 0.5 * Bz[p, :, 0] - 0.25 * Bz[p, :, 1] + 0.75 * Bz[p, :, 2]) / dy / P_int_2[1]

    dBzdy[p, :, Nx_int - 2] = (- 0.75 * Bz[p, :, -3] + 0.25 * Bz[p, :, -2] + 0.5 * Bz[p, :, -1]) / dy / P_int_2[Nx_int - 2]
    dBzdy[p, :, Nx_int - 1] = (- 0.25 * Bz[p, :, -3] - 0.25 * Bz[p, :, -2] + 0.5 * Bz[p, :, -1]) / dy / P_int_2[Nx_int - 1]

    for j in range(2, Ny_int - 2):
        dBzdy[p, :, j] = (Bz[p, :, j + 1] - Bz[p, :, j]) / dy

def compute_diff_E(p):

    dEydx[p, 0, :] = (- 0.5 * Ey[p, 0, :] + 0.5 * Ey[p, 1, :]) / dx / P_half_2[0]
    dEydx[p, 1, :] = (- 0.25 * Ey[p, 0, :] + 0.25 * Ey[p, 1, :]) / dx / P_half_2[1]
    dEydx[p, 2, :] = (- 0.25 * Ey[p, 0, :] - 0.75 * Ey[p, 1, :] + Ey[p, 2, :]) / dx / P_half_2[2]

    dEydx[p, Nx_half - 3, :] = (- Ey[p, -3, :] + 0.75 * Ey[p, -2, :] + 0.25 * Ey[p, -1, :]) / dx / P_half_2[Nx_half - 3]
    dEydx[p, Nx_half - 2, :] = (- 0.25 * Ey[p, -2, :] + 0.25 * Ey[p, -1, :]) / dx / P_half_2[Nx_half - 2]
    dEydx[p, Nx_half - 1, :] = (- 0.5 * Ey[p, -2, :] + 0.5 * Ey[p, -1, :]) / dx / P_half_2[Nx_half - 1]

    for i in range(3, Nx_half - 3):
        dEydx[p, i, :] = (Ey[p, i, :] - Ey[p, i - 1, :]) / dx

    dExdy[p, :, 0] = (- 0.5 * Ex[p, :, 0] + 0.5 * Ex[p, :, 1]) / dx / P_half_2[0]
    dExdy[p, :, 1] = (- 0.25 * Ex[p, :, 0] + 0.25 * Ex[p, :, 1]) / dx / P_half_2[1]
    dExdy[p, :, 2] = (- 0.25 * Ex[p, :, 0] - 0.75 * Ex[p, :, 1] + Ex[p, :, 2]) / dx / P_half_2[2]

    dExdy[p, :, Ny_half - 3] = (- Ex[p, :, -3] + 0.75 * Ex[p, :, -2] + 0.25 * Ex[p, :, -1]) / dy / P_half_2[Nx_half - 3]
    dExdy[p, :, Ny_half - 2] = (- 0.25 * Ex[p, :, -2] + 0.25 * Ex[p, :, -1]) / dy / P_half_2[Nx_half - 2]
    dExdy[p, :, Ny_half - 1] = (- 0.5 * Ex[p, :, -2] + 0.5 * Ex[p, :, -1]) / dy / P_half_2[Nx_half - 1]

    for j in range(3, Ny_half - 3):
        dExdy[p, :, j] = (Ex[p, :, j] - Ex[p, :, j - 1]) / dy

Jz = N.zeros_like(Bz)
Jz[0, :, :] = 100.0 * N.exp(- (xBz_grid**2 + yBz_grid**2) / 0.1**2)

def push_B(p):
        Bz[p, :, :] += dt * (dExdy[p, :, :] - dEydx[p, :, :]) + dt * Jz[p, :, :] * N.sin(10.0 * it * dt) * N.exp(- it * dt / 1.0)

def push_E(p, it):
        Ex[p, :, :] += dt * dBzdy[p, :, :]
        Ey[p, :, :] -= dt * dBzdx[p, :, :]

########
# Boundary conditions
########

sig_ref = 1.0 / dt # 200.0
sig_in  = 0.5 / dt # 200.0
sig_abs = 200.0 

# Vertical interface inner boundary 

def interface_B():

    Bz[0, -1, :] -= dt * sig_in * (Ey[1, 0, :] - Ey[0, -1, :]) / P_half_2[-1]
    Bz[1, 0, :]  -= dt * sig_in * (Ey[1, 0, :] - Ey[0, -1, :]) / P_half_2[0]

def interface_E():

    Ey[0, -1, :] -= dt * sig_in * (Bz[1, 0, :] - Bz[0, -1, :]) / P_half_2[-1]
    Ey[1, 0, :]  -= dt * sig_in * (Bz[1, 0, :] - Bz[0, -1, :]) / P_half_2[0]

# Perfectly conducting outer boundaries    

def BC_conducting_B():

    Bz[0, :, 0]  += dt * sig_abs * Ex[0, :, 0] / P_half_2[0]
    Bz[0, :, -1] -= dt * sig_abs * Ex[0, :, -1] / P_half_2[-1]
    Bz[0, 0, :]  -= dt * sig_abs * Ey[0, 0, :] / P_half_2[0]

    Bz[1, :, 0]  += dt * sig_abs * Ex[1, :, 0] / P_half_2[0]
    Bz[1, :, -1] -= dt * sig_abs * Ex[1, :, -1] / P_half_2[-1]
    Bz[1, -1, :] += dt * sig_abs * Ey[1, -1, :] / P_half_2[-1]
    
    return

def BC_conducting_E():
    return

# Absorbing outer boundaries

def BC_absorbing_B():

    Bz[0, :, 0]  += dt * sig_abs * (Ex[0, :, 0] - Bz[0, :, 0]) / P_half_2[0]
    Bz[0, :, -1] -= dt * sig_abs * (Ex[0, :, -1] + Bz[0, :, -1]) / P_half_2[-1]
    Bz[0, 0, :]  -= dt * sig_abs * (Ey[0, 0, :] + Bz[0, 0, :]) / P_half_2[0]
    # Bz[0, -1, :] += dt * sig_abs * (Ey[0, -1, :] - Bz[0, -1, :]) / P_half_2[-1]

    Bz[1, :, 0]  += dt * sig_abs * (Ex[1, :, 0] - Bz[1, :, 0]) / P_half_2[0]
    Bz[1, :, -1] -= dt * sig_abs * (Ex[1, :, -1] + Bz[1, :, -1]) / P_half_2[-1]
    Bz[1, -1, :] += dt * sig_abs * (Ey[1, -1, :] - Bz[1, -1, :]) / P_half_2[-1]
    # Bz[0, 0, :] -= dt * sig_abs * (Ey[0, 0, :] + Bz[0, 0, :]) / P_half_2[0]

def BC_absorbing_E():

    Ex[0, :, 0]  -= dt * sig_abs * (Ex[0, :, 0] - Bz[0, :, 0]) / P_half_2[0]
    Ex[0, :, -1] -= dt * sig_abs * (Ex[0, :, -1] + Bz[0, :, -1]) / P_half_2[-1]    
    Ey[0, 0, :]  -= dt * sig_abs * (Ey[0, 0, :] + Bz[0, 0, :]) / P_half_2[0]
    # Ey[0, -1, :] -= dt * sig_abs * (Ey[0, -1, :] - Bz[0, -1, :]) / P_half_2[-1]

    Ex[1, :, 0]  -= dt * sig_abs * (Ex[1, :, 0] - Bz[1, :, 0]) / P_half_2[0]
    Ex[1, :, -1] -= dt * sig_abs * (Ex[1, :, -1] + Bz[1, :, -1]) / P_half_2[-1]    
    Ey[1, -1, :]  -= dt * sig_abs * (Ey[1, -1, :] - Bz[1, -1, :]) / P_half_2[-1]
    # Ey[1, 0, :] -= dt * sig_abs * (Ey[1, 0, :] - Bz[1, 0, :]) / P_half_2[0]
    
    return

########
# Initialization
########

amp = 0.0
n_mode = 2
wave = 2.0 * (x_max - x_min) / n_mode
Bz0 = amp * N.cos(2.0 * N.pi * (xBz_grid - x_min) / wave) * N.cos(2.0 * N.pi * (yBz_grid - x_min) / wave)
Ex0 = N.zeros((Nx_half, Ny_int))
Ey0 = N.zeros((Nx_int, Ny_half))

for p in range(n_patches):
    Bz[p, :, :] = Bz0[:, :]
    Ex[p, :, :] = Ex0[:, :]
    Ey[p, :, :] = Ey0[:, :]

########
# Visualization
########

ratio = 0.5

def plot_fields(it):

    fig = P.figure(1, facecolor='w')
    ax = P.subplot(111)

    P.pcolormesh(xBz_grid - 0.5, yBz_grid, Bz[0, :, :], vmin = -1, vmax = 1, cmap = 'RdBu_r')
    P.pcolormesh(xBz_grid + 0.5, yBz_grid, Bz[1, :, :], vmin = -1, vmax = 1, cmap = 'RdBu_r')

    # P.pcolormesh(xEy_grid - 0.5, yEy_grid, Ey[0, :, :], vmin = -1, vmax = 1, cmap = 'RdBu_r')
    # P.pcolormesh(xEy_grid + 0.5, yEy_grid, Ey[1, :, :], vmin = -1, vmax = 1, cmap = 'RdBu_r')
    
    P.plot([0, 0],[-1.25, 1.25], color='k')
    
    P.colorbar()
    
    P.ylim((y_min, y_max))
    P.xlim((x_min - 0.5, x_max + 0.5))
    ax.set_aspect(1.0/ax.get_data_ratio()*ratio)
    
    figsave_png(fig, "snapshots_penalty/fields_" + str(it))

    P.close('all')

########
# Main routine
########

idump = 0

Nt = 2000 # Number of iterations
FDUMP = 50 # Dump frequency
time = dt * N.arange(Nt)
energy = N.zeros((n_patches, Nt))

for it in tqdm(range(Nt), "Progression"):
    if ((it % FDUMP) == 0):
        plot_fields(idump)
        idump += 1

    for p in range(n_patches):
        compute_diff_B(p)
        push_E(p, it)

    interface_E()
    BC_absorbing_E()
    # BC_conducting_E()

    for p in range(n_patches):
        compute_diff_E(p)
        push_B(p)

    interface_B()
    BC_absorbing_B()
    # BC_conducting_B()

    for p in range(n_patches):
        energy[p, it] = dx * dy * N.sum(Bz[p, :, :]**2) + N.sum(Ex[p, :, :]**2) + N.sum(Ey[p, :, :]**2)

for p in range(n_patches):
    P.plot(time, energy[p, :])


