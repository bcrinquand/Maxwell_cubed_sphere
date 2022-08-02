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
cfl = 0.1
Nx = 256 # Number of cells
Nx_int = Nx + 1 # Number of integer points
Nx_half = Nx + 2 # NUmber of hlaf-step points

x_min, x_max = - 0.5, 0.5
dx = (x_max - x_min) / Nx

dt = cfl * dx

# Define grids
x_int  = N.linspace(x_min, x_max, Nx_int)
x_half  = N.zeros(Nx_half)
x_half[0] = x_int[0]
x_half[-1] = x_int[-1]
x_half[1:-1] = x_int[:-1] + 0.5 * dx

n_patches = 2

# Define fields
Ez = N.zeros((n_patches, Nx_int))
By = N.zeros((n_patches, Nx_half))
dBdx = N.zeros((n_patches, Nx_int))
dEdx = N.zeros((n_patches, Nx_half))

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

def compute_diff_By_2(p):
    
    dBdx[p, 0] = (- 0.5 * By[p, 0] + 0.25 * By[p, 1] + 0.25 * By[p, 2]) / dx / P_int_2[0]
    dBdx[p, 1] = (- 0.5 * By[p, 0] - 0.25 * By[p, 1] + 0.75 * By[p, 2]) / dx / P_int_2[1]

    dBdx[p, Nx_int - 2] = (- 0.75 * By[p, -3] + 0.25 * By[p, -2] + 0.5 * By[p, -1]) / dx / P_int_2[Nx_int - 2]
    dBdx[p, Nx_int - 1] = (- 0.25 * By[p, -3] - 0.25 * By[p, -2] + 0.5 * By[p, -1]) / dx / P_int_2[Nx_int - 1]

    for i in range(2, Nx_int - 2):
        dBdx[p, i] = (By[p, i + 1] - By[p, i]) / dx

def compute_diff_Ez_2(p):
    
    dEdx[p, 0] = (- 0.5 * Ez[p, 0] + 0.5 * Ez[p, 1]) / dx / P_half_2[0]
    dEdx[p, 1] = (- 0.25 * Ez[p, 0] + 0.25 * Ez[p, 1]) / dx / P_half_2[1]
    dEdx[p, 2] = (- 0.25 * Ez[p, 0] - 0.75 * Ez[p, 1] + Ez[p, 2]) / dx / P_half_2[2]

    dEdx[p, Nx_half - 3] = (- Ez[p, -3] + 0.75 * Ez[p, -2] + 0.25 * Ez[p, -1]) / dx / P_half_2[Nx_half - 3]
    dEdx[p, Nx_half - 2] = (- 0.25 * Ez[p, -2] + 0.25 * Ez[p, -1]) / dx / P_half_2[Nx_half - 2]
    dEdx[p, Nx_half - 1] = (- 0.5 * Ez[p, -2] + 0.5 * Ez[p, -1]) / dx / P_half_2[Nx_half - 1]

    for i in range(3, Nx_half - 3):
        dEdx[p, i] = (Ez[p, i] - Ez[p, i - 1]) / dx

Jz = N.zeros_like(Ez)
Jz[1, :] = 0.0 * N.exp(-x_int*x_int/0.1/0.1)

def push_B(p):
        By[p, :] -= dt * dEdx[p, :]

def push_E(p, it):
        Ez[p, :] -= dt * dBdx[p, :] + dt * Jz[p, :] * N.sin(10.0 * it * dt) # * N.exp(- it * dt / 0.5)

########
# Boundary conditions
########

sig_ref = 1.0 / dt # 200.0
sig_in  = 0.5 / dt # 200.0
sig_abs = 200.0 

def push_BC_E():

    # Absorbing outer boundaries
    # Ez[0, 0]  -= dt * sig_abs * (Ez[0, 0] + By[0, 0]) / P_int_2[0]
    # Ez[1, -1] -= dt * sig_abs * (Ez[1, -1] - By[1, -1]) / P_int_2[-1]

    # Interface inner boundaries
    Ez[0, -1] -= dt * sig_in * (By[1, 0] - By[0, -1]) / P_int_2[-1]
    Ez[1, 0]  -= dt * sig_in * (By[1, 0] - By[0, -1]) / P_int_2[0]

    # Perfectly conducting outer boundaries
    return
    
def push_BC_B():
    
    # Absorbing outer boundaries
    # By[0, 0]  -= dt * sig_ref * (Ez[0, 0] + By[0, 0]) / P_half_2[0]
    # By[1, -1] -= dt * sig_ref * (By[1, -1] - Ez[1, -1]) / P_half_2[-1]

    # Interface inner boundaries
    By[0, -1] -= dt * sig_in * (Ez[1, 0] - Ez[0, -1]) / P_half_2[-1]
    By[1, 0]  -= dt * sig_in * (Ez[1, 0] - Ez[0, -1]) / P_half_2[0]

    # Perfectly conducting outer boundaries
    By[0, 0]  -= dt * sig_abs * Ez[0, 0] / P_half_2[0]
    By[1, -1] += dt * sig_abs * Ez[1, -1] / P_half_2[-1]
    
    return

########
# Initialization
########

amp = 1.0
n_mode = 2
wave = 2.0 * (x_max - x_min) / n_mode
Ez0 = amp * N.sin(2.0 * N.pi * (x_int - x_min) / wave)
By0 = N.zeros(Nx_half)
# By0 = amp * N.cos(2.0 * N.pi * (x_half - x_min) / wave)
# Ez0 = N.zeros_like(Ez)

for p in range(n_patches):
    Ez[p, :] = Ez0[:]
    By[p, :] = By0[:]

########
# Visualization
########

def plot_fields(it):

    fig = P.figure(1, facecolor='w')
    ax = P.subplot(111)

    P.plot(x_int - 0.5, Ez[0, :], color=colours[0], ls = styles[0])
    P.plot(x_half - 0.5, By[0, :], color=colours[1], ls = styles[0])

    P.plot(x_int + 0.5, Ez[1, :], color=colours[0], ls = styles[1])
    P.plot(x_half + 0.5, By[1, :], color=colours[1], ls = styles[1])
    
    P.plot([0, 0],[-1.25, 1.25], color='k')
    
    P.ylim((-1.25, 1.25))
    P.xlim((-1, 1))
    
    figsave_png(fig, "snapshots_1d/fields_" + str(it))

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
        compute_diff_By_2(p)
        push_E(p, it)

    push_BC_E()

    for p in range(n_patches):
        compute_diff_Ez_2(p)
        push_B(p)

    push_BC_B()

    for p in range(n_patches):
        energy[p, it] = dx * N.sum(P_int_2 * Ez[p, :] * Ez[p, :]) + dx * N.sum(P_half_2 * By[p, :] * By[p, :])

for p in range(n_patches):
    P.plot(time, energy[p, :])


