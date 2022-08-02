# Import modules
import os

import numpy as N
import matplotlib.pyplot as P
import matplotlib
import time
import scipy.integrate as spi
from scipy import interpolate
from skimage.measure import find_contours
from scipy.interpolate import griddata
import matplotlib.pyplot as plt
from scipy.interpolate import RegularGridInterpolator

from math import *
import sys
import h5py
from tqdm import tqdm
from vec_transformations_flip import *
from coord_transformations_flip import *

# Import my figure routines
from figure_module import *

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

# Grids
Nxi  = 45 # Number of cells in xi
Neta = 45 # Number of cells in eta
FDUMP = 7000
dt = 0.008806157051359396

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

field = N.zeros((n_patches, Nxi_half, Neta_half))

# Reads hdf5 file
def ReadFieldHDF5(it):
    h5f = h5py.File('data/Br_{}.h5'.format(it), 'r')
    field[:, :, :] = N.array(h5f['Br'])
    h5f.close()

########    
# Plotting fields on a sphere
########

# Figure parameters
scale, aspect = 2.0, 0.7
ratio = 2.0
fig_size=deffigsize(scale, aspect)

vmi = 1.0
xf = N.zeros_like(field)
yf = N.zeros_like(field)
zf = N.zeros_like(field)

th0 = N.zeros_like(xBr_grid)
ph0 = N.zeros_like(xBr_grid)

phi_s = N.linspace(0, N.pi, 2*50)
theta_s = N.linspace(0, 2*N.pi, 2*50)
theta_s, phi_s = N.meshgrid(theta_s, phi_s)
x_s = 0.95 * N.sin(phi_s) * N.cos(theta_s)
y_s = 0.95 * N.sin(phi_s) * N.sin(theta_s)
z_s = 0.95 * N.cos(phi_s)

cmap_plot="PuOr_r"
norm_plot = matplotlib.colors.Normalize(vmin = - vmi, vmax = vmi)
m = matplotlib.cm.ScalarMappable(cmap = cmap_plot, norm = norm_plot)
m.set_array([])

for patch in range(6):
    
    fcoord = (globals()["coord_" + sphere[patch] + "_to_sph"])
    for i in range(Nxi_half):
        for j in range(Neta_half):
            th0[i, j], ph0[i, j] = fcoord(xBr_grid[i, j], yBr_grid[i, j])  
    xf[patch, :, :] = N.sin(th0) * N.cos(ph0)
    yf[patch, :, :] = N.sin(th0) * N.sin(ph0)
    zf[patch, :, :] = N.cos(th0)
    
# def plot_fields_sphere(it, res, vmi):

#     fig, ax = P.subplots(1,2, subplot_kw={"projection": "3d"}, figsize = fig_size, facecolor = 'w')
    
#     for patch in range(6):
#         for axs in ax:

#             fcolors = m.to_rgba(field[patch, :, :]) 
#             axs.plot_surface(x_s, y_s, z_s, rstride=res, cstride=res, shade=False, color = 'black', zorder = 0)
#             sf = axs.plot_surface(xf[patch, :, :], yf[patch, :, :],
#                     zf[patch, :, :],
#                     rstride = res, cstride = res, shade = False,
#                     facecolors = fcolors, norm = norm_plot, zorder = 2)
#             sf = axs.plot_surface(xf[patch, :, :], yf[patch, :, :],
#                     zf[patch, :, :],
#                     rstride = res, cstride = res, shade = False,
#                     facecolors = fcolors, norm = norm_plot, zorder = 2)
            
#     ax[0].set_box_aspect((1,1,1))
#     ax[0].view_init(elev=40, azim=45)
    
#     ax[1].set_box_aspect((1,1,1))
#     ax[1].view_init(elev=-40, azim=180+45)
    
#     fig.tight_layout(pad=1.0)
        
#     figsave_png(fig, "snapshots_penalty/sphere_Br_" + str(it))

#     P.close("all")

def plot_fields_sphere(it, res):

    fig, ax = P.subplots(1,1, subplot_kw={"projection": "3d"}, figsize = fig_size, facecolor = 'w')
    
    for patch in range(6):

        fcolors = m.to_rgba(field[patch, :, :]) 
        ax.plot_surface(x_s, y_s, z_s, rstride=res, cstride=res, shade=False, color = 'black', zorder = 0)
        sf = ax.plot_surface(xf[patch, :, :], yf[patch, :, :],
                zf[patch, :, :],
                rstride = res, cstride = res, shade = False,
                facecolors = fcolors, norm = norm_plot, zorder = 2)
            
    ax.set_box_aspect((1,1,1))
    ax.view_init(elev=40, azim=45)
    
    ax.set_title(r'$t={:.3f} R/c$'.format(FDUMP*it*dt))
    
    fig.tight_layout(pad=1.0)
        
    figsave_png(fig, "snapshots_penalty/sphere_Br_" + str(it))

    P.close("all")

list = os.listdir("data/")
num = len(list)

for it in tqdm(range(num), "Progression"):
    ReadFieldHDF5(it)
    plot_fields_sphere(it, 1)
