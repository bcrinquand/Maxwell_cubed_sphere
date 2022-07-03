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
Nx = 512 # Number of cells
Nx_int = Nx + 1 # Number of integer points
Nx_half = Nx + 2 # NUmber of hlaf-step points

x_min, x_max = - 1.0, 1.0
dx = (x_max - x_min) / Nx

dt = cfl * dx

# Define grids
x_int  = N.linspace(x_min, x_max, Nx_int)
x_half  = N.zeros(Nx_half)
x_half[0] = x_int[0]
x_half[-1] = x_int[-1]
x_half[1:-1] = x_int[:-1] + 0.5 * dx

# Define fields
Ez = N.zeros((Nx_int))
By = N.zeros((Nx_half))
dBdx = N.zeros((Nx_int))
dEdx = N.zeros((Nx_half))

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

P_int_4 = N.ones(Nx_int)
P_int_4[0] = 407/1152
P_int_4[1] = 473/384
P_int_4[2] = 343/384
P_int_4[3] = 1177/1152
P_int_4[-1] = 407/1152
P_int_4[-2] = 473/384
P_int_4[-3] = 343/384
P_int_4[-4] = 1177/1152

P_half_4 = N.ones(Nx_half)
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

def compute_diff_By_2():
    
    dBdx[0] = (- 0.5 * By[0] + 0.25 * By[1] + 0.25 * By[2]) / dx / P_int_2[0]
    dBdx[1] = (- 0.5 * By[0] - 0.25 * By[1] + 0.75 * By[2]) / dx / P_int_2[1]

    dBdx[Nx_int - 2] = (- 0.75 * By[-3] + 0.25 * By[-2] + 0.5 * By[-1]) / dx / P_int_2[Nx_int - 2]
    dBdx[Nx_int - 1] = (- 0.25 * By[-3] - 0.25 * By[-2] + 0.5 * By[-1]) / dx / P_int_2[Nx_int - 1]

    for i in range(2, Nx_int - 2):
        dBdx[i] = (By[i + 1] - By[i]) / dx

def compute_diff_Ez_2():
    
    dEdx[0] = (- 0.5 * Ez[0] + 0.5 * Ez[1]) / dx / P_half_2[0]
    dEdx[1] = (- 0.25 * Ez[0] + 0.25 * Ez[1]) / dx / P_half_2[1]
    dEdx[2] = (- 0.25 * Ez[0] - 0.75 * Ez[1] + Ez[2]) / dx / P_half_2[2]

    dEdx[Nx_half - 3] = (- Ez[-3] + 0.75 * Ez[-2] + 0.25 * Ez[-1]) / dx / P_half_2[Nx_half - 3]
    dEdx[Nx_half - 2] = (- 0.25 * Ez[-2] + 0.25 * Ez[-1]) / dx / P_half_2[Nx_half - 2]
    dEdx[Nx_half - 1] = (- 0.5 * Ez[-2] + 0.5 * Ez[-1]) / dx / P_half_2[Nx_half - 1]

    for i in range(3, Nx_half - 3):
        dEdx[i] = (Ez[i] - Ez[i - 1]) / dx

def compute_diff_By_4():
  
    dBdx[0] = (-55/378 * By[0] + 5/21 * By[1] -5/42 * By[2] + 5/189 * By[3]) / dx / P_int_4[0]
    dBdx[1] = (-2783/3072 * By[0] + 847/1024 * By[1] + 121/1024 * By[2] -121/3072 * By[3]) / dx / P_int_4[1]
    dBdx[2] = (1085/27648 * By[0] -1085/1024 * By[1] + 1085/1024 * By[2] -1085/27648 * By[3]) / dx / P_int_4[2]
    dBdx[3] = (17/9216 * By[0] + 37/1024 * By[1] -3575/3072 * By[2] -10759/9216 * By[3]) / dx / P_int_4[3]
    dBdx[4] = (667/64512 * By[0] -899/21504 * By[1] + 753/7168 * By[2] -74635/64512 * By[3]) / dx / P_int_4[4]

    dBdx[-1] = (55/378 * By[-1] - 5/21 * By[-2] + 5/42 * By[-3] - 5/189 * By[-4]) / dx / P_int_4[-1]
    dBdx[-2] = (2783/3072 * By[-1] - 847/1024 * By[-2] - 121/1024 * By[-3] + 121/3072 * By[-4]) / dx / P_int_4[-2]
    dBdx[-3] = (- 1085/27648 * By[-1] + 1085/1024 * By[-2] - 1085/1024 * By[-3] + 1085/27648 * By[-4]) / dx / P_int_4[-3]
    dBdx[-4] = (- 17/9216 * By[-1] - 37/1024 * By[-2] + 3575/3072 * By[-3] + 10759/9216 * By[-4]) / dx / P_int_4[-4]
    dBdx[-5] = (- 667/64512 * By[-1] + 899/21504 * By[-2] - 753/7168 * By[-3] + 74635/64512 * By[-4]) / dx / P_int_4[-5]

    for i in range(5, Nx_int - 5):
        dBdx[i] = (By[i - 1] - 27.0 * By[i] + 27.0 * By[i + 1] - By[i + 2]) / (24.0 * dx)

def compute_diff_Ez_4():
  
    dEdx[0] = (-323/378 * Ez[0] + 2783/3072 * Ez[1] -1085/27648 * Ez[2] -17/9216 * Ez[3] -667/64512 * Ez[4]) / dx / P_half_4[0]
    dEdx[1] = (-5/21 * Ez[0] -847/1024 * Ez[1] + 1085/1024 * Ez[2] -37/1024 * Ez[3] +  899/215040 * Ez[4]) / dx / P_half_4[1]
    dEdx[2] = (5/42 * Ez[0] -121/1024 * Ez[1] -1085/1024 * Ez[2] + 3575/3072 * Ez[3] -753/7168 * Ez[4]) / dx / P_half_4[2]
    dEdx[3] = (-5/189 * Ez[0] + 121/3072 * Ez[1] + 1085/27648 * Ez[2] -10759/9216 * Ez[3] + 74635/64512 * Ez[4]) / dx / P_half_4[3]

    dEdx[-1] = (323/378 * Ez[-1] - 2783/3072 * Ez[-2] + 1085/27648 * Ez[-3] + 17/9216 * Ez[-4] + 667/64512 * Ez[-5]) / dx / P_half_4[-1]
    dEdx[-2] = (5/21 * Ez[-1] + 847/1024 * Ez[-2] - 1085/1024 * Ez[-3] + 37/1024 * Ez[-4] - 899/215040 * Ez[-5]) / dx / P_half_4[-2]
    dEdx[-3] = (- 5/42 * Ez[-1] + 121/1024 * Ez[-2] + 1085/1024 * Ez[-3] - 3575/3072 * Ez[-4] + 753/7168 * Ez[-5]) / dx / P_half_4[-3]
    dEdx[-4] = (5/189 * Ez[-1] - 121/3072 * Ez[-2] - 1085/27648 * Ez[-3] + 10759/9216 * Ez[-4] - 74635/64512 * Ez[-5]) / dx / P_half_4[-4]

    for i in range(4, Nx_half - 4):
        dEdx[i] = (Ez[i - 2] - 27.0 * Ez[i - 1] + 27.0 * Ez[i] - Ez[i + 1]) / (24.0 * dx)

Jz = N.zeros_like(Ez)
Jz[:] = 2.0 * N.exp(-x_int*x_int/0.1/0.1)

def push_B():
        By[:] -= dt * dEdx

def push_E(it):
        Ez[:] -= dt * dBdx + dt * Jz[:] * N.sin(10.0 * it * dt) * N.exp(- it * dt / 0.5)

########
# Boundary conditions
########

sigmal = 100.0 # 200.0
sigmar = 100.0 # 200.0

def push_BC_E():
    Ez[0]  -= dt * sigmal * (Ez[0] + By[0]) / P_int_2[0]
    Ez[-1] -= dt * sigmar * (Ez[-1] - By[-1]) / P_int_2[-1]
    return
    
def push_BC_B():
    # By[0]  += dt * sigmal * Ez[0] / P_half_2[0]
    # By[-1] += dt * sigmar * Ez[-1] / P_half_2[-1]
    By[0]  -= dt * sigmal * (Ez[0] + By[0]) / P_half_2[0]
    By[-1] -= dt * sigmar * (By[-1] - Ez[-1]) / P_half_2[-1]
    return

########
# Initialization
########

amp = 0.0
n_mode = 2
wave = 2.0 * (x_max - x_min) / n_mode
Ez0 = amp * N.sin(2.0 * N.pi * (x_int - x_min) / wave)
By0 = N.zeros_like(By)
# By0 = amp * N.cos(2.0 * N.pi * (x_half - x_min) / wave)
# Ez0 = N.zeros_like(Ez)

Ez[:] = Ez0[:]
By[:] = By0[:]

########
# Visualization
########

def plot_fields(it):

    fig = P.figure(1, facecolor='w')
    ax = P.subplot(111)

    P.plot(x_int, Ez, color=colours[0], ls = styles[0])
    P.plot(x_half, By, color=colours[1], ls = styles[1])
    
    P.ylim((-1.25, 1.25))
    P.xlim((x_min, x_max))
    
    figsave_png(fig, "snapshots_1d/fields_" + str(it))

    P.close('all')

########
# Main routine
########

idump = 0

Nt = 10000 # Number of iterations
FDUMP = 50 # Dump frequency
time = dt * N.arange(Nt)
energy = N.zeros(Nt)

for it in tqdm(range(Nt), "Progression"):
    if ((it % FDUMP) == 0):
        plot_fields(idump)
        idump += 1

    compute_diff_By_2()
    push_E(it)
    push_BC_E()
    compute_diff_Ez_2()
    push_B()
    push_BC_B()
    
    energy[it] = dx * N.sum(P_int_2 * Ez * Ez) + dx * N.sum(P_half_2 * By * By)

P.plot(time, energy)


