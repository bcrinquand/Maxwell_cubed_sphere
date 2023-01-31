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
import scipy.interpolate as sci

# Import my figure routines
import sys
sys.path.append('../')
sys.path.append('../transformations/')

from figure_module import *

import warnings
import matplotlib.cbook
warnings.filterwarnings("ignore",category=matplotlib.cbook.mplDeprecation)

# Parameters
cfl = 0.5
Nx = 256 # Number of cells
Nx_int = Nx + 1 # Number of integer points
Nx_half = Nx + 2 # NUmber of hlaf-step points
Ny = 64 # Number of cells
Ny_int = Ny + 1 # Number of integer points
Ny_half = Ny + 2 # NUmber of hlaf-step points

Nt = 10000 # Number of iterations
FDUMP = 10 # Dump frequency

x_min, x_max = 0.0, 1.0
dx = (x_max - x_min) / Nx
y_min, y_max = 0.0, 0.25
dy = (y_max - y_min) / Ny

dt = cfl / N.sqrt(1.0 / dx**2 + 1.0 / dy**2)

q = 1.0 # Absolute value of charge to mass ratio
lambdad = 10.0 # Ratio of skin depth to dx
n0 = 1.0 / (4.0 * N.pi * lambdad**2 * dx**2) 

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

yEz_grid, xEz_grid = N.meshgrid(y_int, x_int)
yBz_grid, xBz_grid = N.meshgrid(y_half, x_half)
yEx_grid, xEx_grid = N.meshgrid(y_int, x_half)
yEy_grid, xEy_grid = N.meshgrid(y_half, x_int)

n_patches = 2

# Define fields
Ex = N.zeros((n_patches, Nx_half, Ny_int))
Ey = N.zeros((n_patches, Nx_int, Ny_half))
Ez = N.zeros((n_patches, Nx_int, Ny_int))
Bx = N.zeros((n_patches, Nx_int, Ny_half))
By = N.zeros((n_patches, Nx_half, Ny_int))
Bz = N.zeros((n_patches, Nx_half, Ny_half))

Jx  = N.zeros((n_patches, Nx_half, Ny_int))
Jy  = N.zeros((n_patches, Nx_int, Ny_half))
rho0 = N.zeros((n_patches, Nx_int, Ny_int))
rho1 = N.zeros((n_patches, Nx_int, Ny_int))

dBzdx = N.zeros((n_patches, Nx_int, Ny_half))
dBzdy = N.zeros((n_patches, Nx_half, Ny_int))
dExdy = N.zeros((n_patches, Nx_half, Ny_half))
dEydx = N.zeros((n_patches, Nx_half, Ny_half))
dBxdy = N.zeros((n_patches, Nx_int, Ny_int))
dBydx = N.zeros((n_patches, Nx_int, Ny_int))
dEzdx = N.zeros((n_patches, Nx_half, Ny_int))
dEzdy = N.zeros((n_patches, Nx_int, Ny_half))

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

# def compute_diff_B(p):
    
#     dBzdx[p, 0, :] = (- 0.5 * Bz[p, 0, :] + 0.25 * Bz[p, 1, :] + 0.25 * Bz[p, 2, :]) / dx / P_int_2[0]
#     dBzdx[p, 1, :] = (- 0.5 * Bz[p, 0, :] - 0.25 * Bz[p, 1, :] + 0.75 * Bz[p, 2, :]) / dx / P_int_2[1]
#     dBzdx[p, Nx_int - 2, :] = (- 0.75 * Bz[p, -3, :] + 0.25 * Bz[p, -2, :] + 0.5 * Bz[p, -1, :]) / dx / P_int_2[Nx_int - 2]
#     dBzdx[p, Nx_int - 1, :] = (- 0.25 * Bz[p, -3, :] - 0.25 * Bz[p, -2, :] + 0.5 * Bz[p, -1, :]) / dx / P_int_2[Nx_int - 1]
#     dBzdx[p, 2:(Nx_int - 2), :] = (N.roll(Bz, -1, axis = 1)[p, 2:(Nx_int - 2), :] - Bz[p, 2:(Nx_int - 2), :]) / dx

#     dBzdy[p, :, 0] = (- 0.5 * Bz[p, :, 0] + 0.25 * Bz[p, :, 1] + 0.25 * Bz[p, :, 2]) / dy / P_int_2[0]
#     dBzdy[p, :, 1] = (- 0.5 * Bz[p, :, 0] - 0.25 * Bz[p, :, 1] + 0.75 * Bz[p, :, 2]) / dy / P_int_2[1]
#     dBzdy[p, :, Ny_int - 2] = (- 0.75 * Bz[p, :, -3] + 0.25 * Bz[p, :, -2] + 0.5 * Bz[p, :, -1]) / dy / P_int_2[Ny_int - 2]
#     dBzdy[p, :, Ny_int - 1] = (- 0.25 * Bz[p, :, -3] - 0.25 * Bz[p, :, -2] + 0.5 * Bz[p, :, -1]) / dy / P_int_2[Ny_int - 1]
#     dBzdy[p, :, 2:(Ny_int - 2)] = (N.roll(Bz, -1, axis = 2)[p, :, 2:(Ny_int - 2)] - Bz[p, :, 2:(Ny_int - 2)]) / dy

def compute_diff_B(p):
    
    dBzdx[p, 0, :] = (- Bz[p, 0, :] + Bz[p, 1, :]) / dx / 0.5
    dBzdx[p, Nx_int - 1, :] = (- Bz[p, -2, :] + Bz[p, -1, :]) / dx / 0.5
    dBzdx[p, 1:(Nx_int - 1), :] = (N.roll(Bz, -1, axis = 1)[p, 1:(Nx_int - 1), :] - Bz[p, 1:(Nx_int - 1), :]) / dx

    dBzdy[p, :, 0] = (- Bz[p, :, 0] + Bz[p, :, 1]) / dy / 0.5
    dBzdy[p, :, Ny_int - 1] = (- Bz[p, :, -2] + Bz[p, :, -1]) / dy / 0.5
    dBzdy[p, :, 1:(Ny_int - 1)] = (N.roll(Bz, -1, axis = 2)[p, :, 1:(Ny_int - 1)] - Bz[p, :, 1:(Ny_int - 1)]) / dy
    
def compute_diff_E(p):

    dEydx[p, 0, :] = (- 0.5 * Ey[p, 0, :] + 0.5 * Ey[p, 1, :]) / dx / P_half_2[0]
    dEydx[p, 1, :] = (- 0.25 * Ey[p, 0, :] + 0.25 * Ey[p, 1, :]) / dx / P_half_2[1]
    dEydx[p, 2, :] = (- 0.25 * Ey[p, 0, :] - 0.75 * Ey[p, 1, :] + Ey[p, 2, :]) / dx / P_half_2[2]
    dEydx[p, Nx_half - 3, :] = (- Ey[p, -3, :] + 0.75 * Ey[p, -2, :] + 0.25 * Ey[p, -1, :]) / dx / P_half_2[Nx_half - 3]
    dEydx[p, Nx_half - 2, :] = (- 0.25 * Ey[p, -2, :] + 0.25 * Ey[p, -1, :]) / dx / P_half_2[Nx_half - 2]
    dEydx[p, Nx_half - 1, :] = (- 0.5 * Ey[p, -2, :] + 0.5 * Ey[p, -1, :]) / dx / P_half_2[Nx_half - 1]
    dEydx[p, 3:(Nx_half - 3), :] = (Ey[p, 3:(Nx_half - 3), :] - N.roll(Ey, 1, axis = 1)[p, 3:(Nx_half - 3), :]) / dx

    dExdy[p, :, 0] = (- 0.5 * Ex[p, :, 0] + 0.5 * Ex[p, :, 1]) / dx / P_half_2[0]
    dExdy[p, :, 1] = (- 0.25 * Ex[p, :, 0] + 0.25 * Ex[p, :, 1]) / dx / P_half_2[1]
    dExdy[p, :, 2] = (- 0.25 * Ex[p, :, 0] - 0.75 * Ex[p, :, 1] + Ex[p, :, 2]) / dx / P_half_2[2]
    dExdy[p, :, Ny_half - 3] = (- Ex[p, :, -3] + 0.75 * Ex[p, :, -2] + 0.25 * Ex[p, :, -1]) / dy / P_half_2[Nx_half - 3]
    dExdy[p, :, Ny_half - 2] = (- 0.25 * Ex[p, :, -2] + 0.25 * Ex[p, :, -1]) / dy / P_half_2[Nx_half - 2]
    dExdy[p, :, Ny_half - 1] = (- 0.5 * Ex[p, :, -2] + 0.5 * Ex[p, :, -1]) / dy / P_half_2[Nx_half - 1]
    dExdy[p, :, 3:(Ny_half - 3)] = (Ex[p, :, 3:(Ny_half - 3)] - N.roll(Ex, 1, axis = 2)[p, :, 3:(Ny_half - 3)]) / dy

def push_B(p, it, dtin):
        Bz[p, :, :] += dtin * (dExdy[p, :, :] - dEydx[p, :, :])

def push_E(p, it, dtin):
        Ex[p, :, :] += dtin * (dBzdy[p, :, :] - 4.0 * N.pi * Jx[p, :, :])
        Ey[p, :, :] += dtin * (- dBzdx[p, :, :] - 4.0 * N.pi * Jy[p, :, :])

########
# Particles
########

PPC = 2
np = (2 * Nx) * Ny * PPC # 2 patches
theta = 0.001
np2 = int(np / 2)

xp  = N.zeros((np, 2))
yp  = N.zeros((np, 2))
uxp = N.zeros((np, 2))
uyp = N.zeros((np, 2))
wp  = N.zeros((np, 2)) # charge x weight (can be negative)
tag = N.zeros((np, 2), dtype='int') # Patch in which the partice is located
switch = N.empty((np, 2), dtype='int') # Did the particle cross patches?

######

from scipy.stats import rv_continuous
# import scipy.special as scp

# class Juttner(rv_continuous): 
#     def _pdf(self, x):
#         cst = scp.kn(2, 1.0 / theta) * 4.0 * N.pi * theta
#         return N.exp(- N.sqrt(1.0 + x**2) / theta) / cst

def initialize_part():

    ipart = 0
    for i in range(Nx):
        for j in range(Ny):
            for k in range(int(PPC)):
                
                sample = N.random.uniform(x_int[i], x_int[i + 1])
                xp[ipart, 0] = sample
                xp[ipart, 1] = sample
                sample = N.random.uniform(x_int[i], x_int[i + 1])
                xp[np2 + ipart - 1, 0] = sample
                xp[np2 + ipart - 1, 1] = sample
                sample = N.random.uniform(y_int[j], y_int[j + 1])
                yp[ipart, 0] = sample
                yp[ipart, 1] = sample
                sample = N.random.uniform(y_int[j], y_int[j + 1])
                yp[np2 + ipart - 1, 0] = sample
                yp[np2 + ipart - 1, 1] = sample
                sample = N.random.normal(0.0, theta)
                uxp[ipart, 0] = sample
                uxp[ipart, 1] = sample
                sample = N.random.normal(0.0, theta)
                uxp[np2 + ipart - 1, 0] = sample
                uxp[np2 + ipart - 1, 1] = sample
                sample = N.random.normal(0.0, theta)
                uyp[ipart, 0] = sample
                uyp[ipart, 1] = sample
                sample = N.random.normal(0.0, theta)
                uyp[np2 + ipart - 1, 0] = sample
                uyp[np2 + ipart - 1, 1] = sample
                
                ipart += 1
    
    # xp[0, 0] = 0.25
    # xp[0, 1] = 0.25
    # yp[0, 0] = 0.25
    # yp[0, 1] = 0.25
    # uxp[0, 0] = 0.8
    # uxp[0, 1] = - 0.8
    # uyp[0, 0] = -0.8
    # uyp[0, 1] = 0.8
    
    uxp[N.abs(uxp) > 3.0 * N.sqrt(theta)] = 3.0 * N.sqrt(theta)
    uyp[N.abs(uyp) > 3.0 * N.sqrt(theta)] = 3.0 * N.sqrt(theta)

    tag[:np2, :] = 0
    tag[np2:, :] = 1
    
    wp[:, 0] =   n0 * dx**2 / PPC
    wp[:, 1] = - n0 * dx**2 / PPC
    
    switch[:, :] = 0 

# Returns index of CELL
def i_from_pos(x0, y0):
    i0 = int(((x0 - x_min) / dx) // 1)
    j0 = int(((y0 - y_min) / dy) // 1)
    return i0, j0

def change_tag(tag):
    return int(1-tag)

# Particle periodic boundary conditions
def BC_part(ip, s):
    
    switch[ip, s] = 0
    
    if (yp[ip, s] > y_max):
        yp[ip, s] -= y_max
    elif (yp[ip, s] < 0.0):
        yp[ip, s] += y_max

    if ((xp[ip, s] > x_max)and(tag[ip, s]==1)):
        xp[ip, s] -= x_max
        tag[ip, s] = 0
        switch[ip, s] = -1
    elif ((xp[ip, s] < x_min)and(tag[ip, s]==0)):
        xp[ip, s] += x_max
        tag[ip, s] = 1
        switch[ip, s] = 1
    elif ((xp[ip, s] < x_min)and(tag[ip, s]==1)):
        xp[ip, s] += x_max
        tag[ip, s] = 0
        switch[ip, s] = -1
    elif ((xp[ip, s] > x_max)and(tag[ip, s]==0)):
        xp[ip, s] -= x_max
        tag[ip, s] = 1
        switch[ip, s] = 1
    
    return

Bztot = N.zeros((2 * Nx + 3, Ny_half))
Extot = N.zeros((2 * Nx + 3, Ny_int))
Eytot = N.zeros((2 * Nx + 1, Ny_half))

xtot_half = N.concatenate((x_half, x_half[1:] + x_max - x_min))
xtot_int  = N.concatenate((x_int, x_int[1:] + x_max - x_min))

def interpolate_field(x0, y0):
    
    Bztot[:(Nx_half - 1), :] = Bz[0, :-1, :]
    Bztot[Nx_half:, :] = Bz[1, 1:, :]
    Bztot[Nx_half, :] = 0.5 * (Bz[0, -1, :] + Bz[1, 0, :])

    Extot[:(Nx_half - 1), :] = Ex[0, :-1, :]
    Extot[Nx_half:, :] = Ex[1, 1:, :]
    Extot[Nx_half, :] = 0.5 * (Ex[0, -1, :] + Ex[1, 0, :])

    Eytot[:(Nx_int - 1), :] = Ey[0, :-1, :]
    Eytot[Nx_int:, :] = Ey[1, 1:, :]
    Eytot[Nx_int, :] = 0.5 * (Ey[0, -1, :] + Bz[1, 0, :])
    
    # Bz0 = (sci.RectBivariateSpline(xtot_half, y_half, Bztot[:, :]))(x0, y0)
    # Ex0 = (sci.RectBivariateSpline(xtot_half, y_int,  Extot[:, :]))(x0, y0)
    # Ey0 = (sci.RectBivariateSpline(xtot_int,  y_half, Eytot[:, :]))(x0, y0)
    
    Bz0 = (sci.RegularGridInterpolator((xtot_half, y_half), Bztot[:, :], method = 'linear', bounds_error = False))((x0, y0))
    Ex0 = (sci.RegularGridInterpolator((xtot_half, y_int ), Extot[:, :], method = 'linear', bounds_error = False))((x0, y0))
    Ey0 = (sci.RegularGridInterpolator((xtot_int,  y_half), Eytot[:, :], method = 'linear', bounds_error = False))((x0, y0))

    return Ex0, Ey0, Bz0

def push_x(s):
    gammap = N.sqrt(1.0 + uxp[:, s]**2 + uyp[:, s]**2)
    xp[:, s] += uxp[:, s] * dt / gammap
    yp[:, s] += uyp[:, s] * dt / gammap

# it has x_n, u_n+1/2
# it - 1 has x^n-1, u_n-1/2

def push_u(s):
    w0 = wp[:, s]
    x0 = N.zeros_like(xp)
    y0 = N.zeros_like(yp)
    
    x0[:, s] = xp[:, s]
    y0[:, s] = yp[:, s] 
    ux0, uy0 = uxp[:, s], uyp[:, s] 
    tag0 = tag[:, s]
    
    x0[tag0 == 1] = x0[tag0 == 1] + x_max

    Ex0, Ey0, Bz0 = interpolate_field(x0[:, s], y0[:, s])        
    Ex0 *= 0.5 * q * dt * N.sign(w0)
    Ey0 *= 0.5 * q * dt * N.sign(w0)
    Bz0 *= 0.5 * q * dt * N.sign(w0)
    
    uxe = ux0 + Ex0
    uye = uy0 + Ey0
    
    gammae = N.sqrt(1.0 + uxe**2 + uye**2)
    Bz1 = Bz0/gammae
    t = 2.0 / (1.0 + Bz1*Bz1)
    
    ux1 = (uxe + uye * Bz1) * t
    uy1 = (uye - uxe * Bz1) * t
    
    uxm = uxe + uy1 * Bz1
    uym = uye - ux1 * Bz1
    
    uxf = uxm + Ex0
    uyf = uym + Ey0
    
    uxp[:, s] = uxf
    uyp[:, s] = uyf
    return

########
# Current deposition
########

# Computes intermediate point in zig-zag algorithm
def compute_intermediate(x1, y1, x2, y2):

    i1, j1 = i_from_pos(x1, y1)
    i2, j2 = i_from_pos(x2, y2)
    
    x_mid = 0.5 * (x1 + x2)
    y_mid = 0.5 * (y1 + y2)
    
    x_r = min(min(i1*dx, i2*dx) + dx, max(max(i1*dx, i2*dx), x_mid))
    y_r = min(min(j1*dy, j2*dy) + dy, max(max(j1*dy, j2*dy), y_mid))

    return x_r, y_r

# Bunch of tests of particle location 

def test_edge(i):
    if (i==0)or(i==(Nx-1)):
        return True
    else:
        return False
    
def test_out(i):
    if (i<0)or(i>(Nx - 1)):
        return True
    else:
        return False

def test_inside(i):
    if (i>0)and(i<(Nx-1)):
        return True
    else:
        return False

# Loop on particles
def deposit_J(it):

    Jx[:, :, :] = 0.0
    Jy[:, :, :] = 0.0
    
    for ip in range(np):
        deposit_particle(it, ip)
                    
S0 = dx * dy

def fluxes_1(x0, y0, xr, yr, w0):
    
    # Indices of initial/final cell (from 0 to N-1)
    i0, j0 = i_from_pos(x0, y0)
    
    Fx = q * (xr - x0) / dt * w0
    Wx = 0.5 * (x0 + xr) / dx - i0
    Fy = q * (yr - y0) / dt * w0
    Wy = 0.5 * (y0 + yr) / dy - j0
    
    return Fx, Wx, Fy, Wy

def fluxes_2(x0, y0, xr, yr, w0):
    
    # Indices of initial/final cell (from 0 to N-1)
    i0, j0 = i_from_pos(x0, y0)
    
    Fx = q * (x0 - xr) / dt * w0
    Wx = 0.5 * (x0 + xr) / dx - i0
    Fy = q * (y0 - yr) / dt * w0
    Wy = 0.5 * (y0 + yr) / dy - j0
    
    return Fx, Wx, Fy, Wy

### NEW STENCIL
def subcell(m_tag0, m_i1, m_j1, m_Fx1, m_Fy1, m_Wx1, m_Wy1):

    # Shifted by 1 because of extra points at the edge
    ix1 = m_i1 + 1
    jy1 = m_j1 + 1
    
    tag1 = change_tag(m_tag0)
    deltax1 = 1.0
    deltay1 = 1.0

    # Bulk current deposition
    Jx[m_tag0, ix1, m_j1] += deltax1 * m_Fx1 * (1.0 - m_Wy1) / S0
    Jx[m_tag0, ix1, m_j1 + 1] += deltax1 * m_Fx1 * m_Wy1 / S0
    Jy[m_tag0, m_i1, jy1] += deltay1 * m_Fy1 * (1.0 - m_Wx1) / S0
    Jy[m_tag0, m_i1 + 1, jy1] += deltay1 * m_Fy1 * m_Wx1 / S0

##### Vertical interfaces

    # Current on left edge
    if (m_i1 == 0):
        Jx[m_tag0, 0, m_j1]     += 0.5 * m_Fx1 * (1.0 - m_Wy1) / S0
        Jx[m_tag0, 0, m_j1 + 1] += 0.5 * m_Fx1 * m_Wy1 / S0
        
        Jx[tag1, -1, m_j1] += 0.5 * m_Fx1 * (1.0 - m_Wy1) / S0
        Jx[tag1, -1, m_j1 + 1] += 0.5 * m_Fx1 * m_Wy1 / S0
        Jy[tag1, -1, jy1] += deltay1 * m_Fy1 * (1.0 - m_Wx1) / S0

    # Current on right edge            
    if (m_i1 == (Nx - 1)):
        Jx[m_tag0, -1, m_j1]     += 0.5 * m_Fx1 * (1.0 - m_Wy1) / S0
        Jx[m_tag0, -1, m_j1 + 1] += 0.5 * m_Fx1 * m_Wy1 / S0

        Jx[tag1, 0, m_j1] += 0.5 * m_Fx1 * (1.0 - m_Wy1) / S0
        Jx[tag1, 0, m_j1 + 1] += 0.5 * m_Fx1 * m_Wy1 / S0
        Jy[tag1, 0, jy1] += deltay1 * m_Fy1 * m_Wx1 / S0

##### Horizontal interfaces

    # Current on bottom edge
    if (m_j1 == 0):
        Jy[m_tag0, m_i1, 0]     += 0.5 * m_Fy1 * (1.0 - m_Wx1) / S0
        Jy[m_tag0, m_i1 + 1, 0] += 0.5 * m_Fy1 * m_Wx1 / S0

        Jy[m_tag0, m_i1, -1]     += 0.5 * m_Fy1 * (1.0 - m_Wx1) / S0
        Jy[m_tag0, m_i1 + 1, -1] += 0.5 * m_Fy1 * m_Wx1 / S0
        
        Jx[m_tag0, ix1, -1] += m_Fx1 * (1.0 - m_Wy1) / S0
        
    # Current on top edge            
    if (m_j1 == (Ny - 1)):
        Jy[m_tag0, m_i1, -1]     += 0.5 * m_Fy1 * (1.0 - m_Wx1) / S0
        Jy[m_tag0, m_i1 + 1, -1] += 0.5 * m_Fy1 * m_Wx1 / S0

        Jy[m_tag0, m_i1, 0]     += 0.5 * m_Fy1 * (1.0 - m_Wx1) / S0
        Jy[m_tag0, m_i1 + 1, 0] += 0.5 * m_Fy1 * m_Wx1 / S0

        Jx[m_tag0, ix1, 0] += m_Fx1 * m_Wy1 / S0

    return

# # Deposit current on a subcell
# def subcell(m_tag0, m_i1, m_j1, m_Fx1, m_Fy1, m_Wx1, m_Wy1):

#     # Shifted by 1 because of extra points at the edge
#     ix1 = m_i1 + 1
#     jy1 = m_j1 + 1
    
#     tag1 = int(1-m_tag0)
    
#     deltax1 = 1.0
#     deltay1 = 1.0
            
#     # Interior, no deposition on edge cell
#     if (test_inside(m_i1)==True):
#         pass
#     # Particle starts in edge and moves to interior
#     elif (test_edge(m_i1)==True):
#         deltax1 = 3.0

#     # Interior, no deposition on edge cell
#     if (test_inside(m_j1)==True):
#         pass
#     # Particle starts in edge and moves to interior
#     elif (test_edge(m_j1)==True):
#         deltay1 = 3.0

#     # Bulk current deposition
#     Jx[m_tag0, ix1, m_j1] += deltax1 * m_Fx1 * (1.0 - m_Wy1) / S0
#     Jx[m_tag0, ix1, m_j1 + 1] += deltax1 * m_Fx1 * m_Wy1 / S0
    
#     Jy[m_tag0, m_i1, jy1] += deltay1 * m_Fy1 * (1.0 - m_Wx1) / S0
#     Jy[m_tag0, m_i1 + 1, jy1] += deltay1 * m_Fy1 * m_Wx1 / S0

# ##### Vertical interfaces

#     # Current on left edge
#     if (m_i1 == 0):
#         Jx[m_tag0, 0, m_j1]     += 0.5 * m_Fx1 * (1.0 - m_Wy1) / S0
#         Jx[m_tag0, 0, m_j1 + 1] += 0.5 * m_Fx1 * m_Wy1 / S0
        
#     # Current on mid-left cell
#     if (m_i1 == 1):
#         Jx[m_tag0, 1, m_j1]     += - m_Fx1 * (1.0 - m_Wy1) / S0
#         Jx[m_tag0, 1, m_j1 + 1] += - m_Fx1 * m_Wy1 / S0
        
#     # Current on right edge            
#     if (m_i1 == (Nx - 1)):
#         Jx[m_tag0, -1, m_j1]     += 0.5 * m_Fx1 * (1.0 - m_Wy1) / S0
#         Jx[m_tag0, -1, m_j1 + 1] += 0.5 * m_Fx1 * m_Wy1 / S0
        
#     # Current on mid-right cell
#     if (m_i1 == (Nx - 2)):
#         Jx[m_tag0, -2, m_j1]     += - m_Fx1 * (1.0 - m_Wy1) / S0
#         Jx[m_tag0, -2, m_j1 + 1] += - m_Fx1 * m_Wy1 / S0
        
#     # Current from charge that's not into patch 1 yet
#     if ((m_tag0==0)and(test_edge(m_i1)==True)):
#         Jx[tag1, 0, m_j1] += 0.5 * m_Fx1 * (1.0 - m_Wy1) / S0
#         Jx[tag1, 0, m_j1 + 1] += 0.5 * m_Fx1 * m_Wy1 / S0
#         Jx[tag1, 1, m_j1] += -1.0 * m_Fx1 * (1.0 - m_Wy1) / S0
#         Jx[tag1, 1, m_j1 + 1] += - 1.0 * m_Fx1 * m_Wy1 / S0

#         Jy[tag1, 0, jy1] += m_Fy1 * m_Wx1 / S0
        
#     # Current from charge that's not into patch 0 yet
#     if ((m_tag0==1)and(test_edge(m_i1)==True)):
#         Jx[tag1, -1, m_j1] += 0.5 * m_Fx1 * (1.0 - m_Wy1) / S0
#         Jx[tag1, -1, m_j1 + 1] += 0.5 * m_Fx1 * m_Wy1 / S0
#         Jx[tag1, -2, m_j1] += -1.0 * m_Fx1 * (1.0 - m_Wy1) / S0
#         Jx[tag1, -2, m_j1 + 1] += - 1.0 * m_Fx1 * m_Wy1 / S0
        
#         Jy[tag1, -1, jy1] += m_Fy1 * (1.0 - m_Wx1) / S0

# ##### Horizontal interfaces

#     # Current on bottom edge
#     if (m_j1 == 0):
#         Jy[m_tag0, m_i1, 0]     += 0.5 * m_Fy1 * (1.0 - m_Wx1) / S0
#         Jy[m_tag0, m_i1 + 1, 0] += 0.5 * m_Fy1 * m_Wx1 / S0

#         Jy[m_tag0, m_i1, -1]     += 0.5 * m_Fy1 * (1.0 - m_Wx1) / S0
#         Jy[m_tag0, m_i1 + 1, -1] += 0.5 * m_Fy1 * m_Wx1 / S0
        
#         Jx[m_tag0, ix1, -1] += m_Fy1 * (1.0 - m_Wx1) / S0

#     # Current on mid-bottom cell
#     if (m_j1 == 1):
#         Jy[m_tag0, m_i1, 1]     += - m_Fy1 * (1.0 - m_Wx1) / S0
#         Jy[m_tag0, m_i1 + 1, 1] += - m_Fy1 * m_Wx1 / S0
        
#     # Current on top edge            
#     if (m_j1 == (Ny - 1)):
#         Jy[m_tag0, m_i1, -1]     += 0.5 * m_Fy1 * (1.0 - m_Wx1) / S0
#         Jy[m_tag0, m_i1 + 1, -1] += 0.5 * m_Fy1 * m_Wx1 / S0

#         Jy[m_tag0, m_i1, 0]     += 0.5 * m_Fy1 * (1.0 - m_Wx1) / S0
#         Jy[m_tag0, m_i1 + 1, 0] += 0.5 * m_Fy1 * m_Wx1 / S0

#         Jx[m_tag0, ix1, 0] += m_Fx1 * m_Wy1 / S0

#     # Current on mid-top cell
#     if (m_j1 == (Ny - 2)):
#         Jy[m_tag0, m_i1, -2]     += - m_Fy1 * (1.0 - m_Wx1) / S0
#         Jy[m_tag0, m_i1 + 1, -2] += - m_Fy1 * m_Wx1 / S0

#     return

# # Current deposition of a single particle
# # Note : only deals with x_min and x_max interfaces, no corners or top/bottom interfaces!!
# # Assumes y_min = 0.0 and x_min = 0.0
def deposit_particle(ip, s):
    
    gammap = N.sqrt(1.0 + uxp[ip, s]**2 + uyp[ip, s]**2)
    x2, y2 = xp[ip, s], yp[ip, s] 
    x1, y1 = x2 - uxp[ip, s] * dt / gammap, y2 - uyp[ip, s] * dt / gammap

    w0   = wp[ip, s]
    tag1 = tag[ip, s]

    if ((y1 > y_max)and(x1 > x_max)):
        y1 -= y_max
        x1 -= x_max
        tag0 = change_tag(tag1)

        yr = 0.0
        xr = 0.0
        Fx1, Wx1, Fy1, Wy1 = fluxes_1(x1, y1, xr, yr, w0)
        
        yr = y_max
        xr = x_max
        Fx2, Wx2, Fy2, Wy2 = fluxes_2(x2, y2, xr, yr, w0)
            
    elif ((y1 > y_max)and(x1 < 0.0)):
        y1 -= y_max
        x1 += x_max
        tag0 = change_tag(tag1)

        yr = 0.0
        xr = x_max
        Fx1, Wx1, Fy1, Wy1 = fluxes_1(x1, y1, xr, yr, w0)
        
        yr = y_max
        xr = 0.0
        Fx2, Wx2, Fy2, Wy2 = fluxes_2(x2, y2, xr, yr, w0)
            
    elif ((y1 < 0.0)and(x1 > x_max)):
        y1 += y_max
        x1 -= x_max
        tag0 = change_tag(tag1)

        yr = y_max
        xr = 0.0
        Fx1, Wx1, Fy1, Wy1 = fluxes_1(x1, y1, xr, yr, w0)
        
        yr = 0.0
        xr = x_max
        Fx2, Wx2, Fy2, Wy2 = fluxes_2(x2, y2, xr, yr, w0)
            
    elif ((y1 < 0.0)and(x1 < 0.0)):
        y1 += y_max
        x1 += x_max
        tag0 = change_tag(tag1)

        yr = 0.0
        xr = 0.0
        Fx1, Wx1, Fy1, Wy1 = fluxes_1(x1, y1, xr, yr, w0)
    
        yr = y_max
        xr = x_max
        Fx2, Wx2, Fy2, Wy2 = fluxes_2(x2, y2, xr, yr, w0)
            
    elif (y1 > y_max):
        tag0=tag1
        y1 -= y_max
        xr, yr = compute_intermediate(x1, y1, x2, y2)
        
        yr = 0.0
        Fx1, Wx1, Fy1, Wy1 = fluxes_1(x1, y1, xr, yr, w0)
        yr = y_max
        Fx2, Wx2, Fy2, Wy2 = fluxes_2(x2, y2, xr, yr, w0)
            
    elif (y1 < 0.0):
        tag0=tag1
        y1 += y_max
        xr, yr = compute_intermediate(x1, y1, x2, y2)
        
        yr = y_max
        Fx1, Wx1, Fy1, Wy1 = fluxes_1(x1, y1, xr, yr, w0)
        yr = 0.0
        Fx2, Wx2, Fy2, Wy2 = fluxes_2(x2, y2, xr, yr, w0)
            
    elif (x1 > x_max):
        x1 -= x_max
        tag0 = change_tag(tag1)
        xr, yr = compute_intermediate(x1, y1, x2, y2)
        
        xr = 0.0
        Fx1, Wx1, Fy1, Wy1 = fluxes_1(x1, y1, xr, yr, w0)
        xr = x_max    
        Fx2, Wx2, Fy2, Wy2 = fluxes_2(x2, y2, xr, yr, w0)
            
    elif (x1 < 0.0):
        x1 += x_max
        tag0 = change_tag(tag1)
        xr, yr = compute_intermediate(x1, y1, x2, y2)
        
        xr = x_max
        Fx1, Wx1, Fy1, Wy1 = fluxes_1(x1, y1, xr, yr, w0)
        xr = 0.0     
        Fx2, Wx2, Fy2, Wy2 = fluxes_2(x2, y2, xr, yr, w0)
            
    else:
        tag0=tag1
        xr, yr = compute_intermediate(x1, y1, x2, y2)
        
        Fx1, Wx1, Fy1, Wy1 = fluxes_1(x1, y1, xr, yr, w0)
        Fx2, Wx2, Fy2, Wy2 = fluxes_2(x2, y2, xr, yr, w0)

        # # if (y2<0.5)and(y2>0.4995):
        # if (Fx1>1e-4)or((Fy1>1e-4)):
        #     print(x1, y1, x2, y2, ip, s, Fx1, Fx2, Fy1, Fy2, 'YOFINAL', xr, yr)
                
    # Indices of initial/final cell (from 0 to N-1)
    i1, j1 = i_from_pos(x1, y1)
    i2, j2 = i_from_pos(x2, y2)

    # xr, yr = compute_intermediate(x1, y1, x2, y2)
    
    # if (i1>63)or(i2>63):
    #     print(tag0, tag1, i1, i2, x1, x2)
        
    # Particle does not leave patch
    if (tag1 == tag0):
        
        # Umeda notation

        # Fx1 = q * (xr - x1) / dt * w0
        # Fx2 = q * (x2 - xr) / dt * w0

        # Fy1 = q * (yr - y1) / dt * w0
        # Fy2 = q * (y2 - yr) / dt * w0
        
        # Wx1 = 0.5 * (x1 + xr) / dx - i1
        # Wy1 = 0.5 * (y1 + yr) / dy - j1

        # Wx2 = 0.5 * (x2 + xr) / dx - i2
        # Wy2 = 0.5 * (y2 + yr) / dy - j2

        # if (Fy1>1e-4):
        #     print(Fy1,Fy2,yr,y1,y2)

        # First part of trajectory
        subcell(tag0, i1, j1, Fx1, Fy1, Wx1, Wy1) 
          
        # Second part of trajectory    
        subcell(tag1, i2, j2, Fx2, Fy2, Wx2, Wy2) 

    # Particle leaves patch
    elif (tag1 != tag0):
        
        # First part of trajectory
        subcell(tag0, i1, j1, Fx1, Fy1, Wx1, Wy1) 
          
        # Second part of trajectory    
        subcell(tag1, i2, j2, Fx2, Fy2, Wx2, Wy2) 
    
    return

# def deposit_particle(ip, s):
    
#     gammap = N.sqrt(1.0 + uxp[ip, s]**2 + uyp[ip, s]**2)
#     x2, y2 = xp[ip, s], yp[ip, s] 
#     x1, y1 = xp[ip, s] - uxp[ip, s] * dt / gammap, yp[ip, s] - uyp[ip, s] * dt / gammap

#     # Indices of initial/final cell (from 0 to N-1)
#     i1, j1 = i_from_pos(x1, y1)
#     i2, j2 = i_from_pos(x2, y2)

#     xr, yr = compute_intermediate(x1, y1, x2, y2)

#     w0   = wp[ip, s]
#     tag0 = tag[ip, s]
        
#     # Coefficients of deposited Jx
#     deltax1 = 1.0
#     deltax2 = 1.0
        
#     deltay1 = 1.0
#     deltay2 = 1.0
    
#     sw = switch[ip, s]
    
#     # Particle does not leave patch
#     if (sw == 0):

#         # Umeda notation

#         Fx1 = q * (xr - x1) / dt * w0
#         Fx2 = q * (x2 - xr) / dt * w0

#         Fy1 = q * (yr - y1) / dt * w0
#         Fy2 = q * (y2 - yr) / dt * w0

#         Wx1 = 0.5 * (x1 + xr) / dx - i1
#         Wy1 = 0.5 * (y1 + yr) / dy - j1

#         Wx2 = 0.5 * (x2 + xr) / dx - i2
#         Wy2 = 0.5 * (y2 + yr) / dy - j2

#         # Shifted by 1 because of extra points at the edge
#         ix1 = i1 + 1
#         ix2 = i2 + 1

#         jy1 = j1 + 1
#         jy2 = j2 + 1

#         #######
#         # First part of trajectory    
#         #######

#         # Bulk current deposition
#         Jx[tag0, ix1, j1] += deltax1 * Fx1 * (1.0 - Wy1) / S0
#         Jx[tag0, ix1, j1 + 1] += deltax1 * Fx1 * Wy1 / S0

#         Jy[tag0, i1, jy1] += deltay1 * Fy1 * (1.0 - Wx1) / S0
#         Jy[tag0, i1 + 1, jy1] += deltay1 * Fy1 * Wx1 / S0

#         # Current on left edge
#         if (i1 == 0):
#             Jx[tag0, 0, j1]     += 0.5 * Fx1 * (1.0 - Wy1) / S0
#             Jx[tag0, 0, j1 + 1] += 0.5 * Fx1 * Wy1 / S0
            
#         # Current on right edge            
#         if (i1 == (Nx - 1)):
#             Jx[tag0, -1, j1]     += 0.5 * Fx1 * (1.0 - Wy1) / S0
#             Jx[tag0, -1, j1 + 1] += 0.5 * Fx1 * Wy1 / S0
            
#         # Current from charge that's not into patch 1 yet
#         if (tag0==0)and(i1==(Nx-1)):
#             Jx[1, 0, j1] += 0.5 * Fx1 * (1.0 - Wy1) / S0
#             Jx[1, 0, j1 + 1] += 0.5 * Fx1 * Wy1 / S0

#         # Current from charge that's not into patch 0 yet
#         if (tag0==1)and(i1==0):
#             Jx[0, -1, j1] += 0.5 * Fx1 * (1.0 - Wy1) / S0
#             Jx[0, -1, j1 + 1] += 0.5 * Fx1 * Wy1 / S0
            
#         #######
#         # Second part of trajectory    
#         #######
        
#         Jx[tag0, ix2, j2] += deltax2 * Fx2 * (1.0 - Wy2) / S0
#         Jx[tag0, ix2, j2 + 1] += deltax2 * Fx2 * Wy2 / S0

#         Jy[tag0, i2, jy2] += deltay2 * Fy2 * (1.0 - Wx2) / S0
#         Jy[tag0, i2 + 1, jy2] += deltay2 * Fy2 * Wx2 / S0
        
#         if (i2 == 0):
#             Jx[tag0, 0, j2]     += 0.5 * Fx2 * (1.0 - Wy2) / S0
#             Jx[tag0, 0, j2 + 1] += 0.5 * Fx2 * Wy2 / S0        
            
#         if (i2 == (Nx - 1)):
#             Jx[tag0, -1, j2]     += 0.5 * Fx2 * (1.0 - Wy2) / S0
#             Jx[tag0, -1, j2 + 1] += 0.5 * Fx2 * Wy2 / S0
            
#         if (tag0==0)and(i2==(Nx-1)):
#             Jx[1, 0, j2] += 0.5 * Fx2 * (1.0 - Wy2) / S0
#             Jx[1, 0, j2 + 1] += 0.5 * Fx2 * Wy2 / S0

#         if (tag0==1)and(i2==0):
#             Jx[0, -1, j2] += 0.5 * Fx2 * (1.0 - Wy2) / S0
#             Jx[0, -1, j2 + 1] += 0.5 * Fx2 * Wy2 / S0

#     # Particle leaves patch
#     elif (sw != 0):
            
#         # There is better way of doing this than having this dichotomy. Will do later
#         if (sw == 1):
#             tag1 = 1

#             # When particle leaves patch 0 to 1, x2 is already written in new patch coordinates, so must specify xr in both cases
#             xr = 1.0
#             Fx1 = q * (xr - x1) / dt * w0
#             Wx1 = 0.5 * (x1 + xr) / dx - i1
#             Fy1 = q * (yr - y1) / dt * w0
#             Wy1 = 0.5 * (y1 + yr) / dy - j1
            
#             xr = 0.0
#             Fx2 = q * (x2 - xr) / dt * w0
#             Wx2 = 0.5 * (x2 + xr) / dx - i2
#             Fy2 = q * (y2 - yr) / dt * w0
#             Wy2 = 0.5 * (y2 + yr) / dy - j2

#             ix1 = i1 + 1
#             ix2 = i2 + 1

#             jy1 = j1 + 1
#             jy2 = j2 + 1

#             # Current deposited during first part of trajectory: particle is in patch 0

#             Jy[tag0, i1, jy1] += deltay1 * Fy1 * (1.0 - Wx1) / S0
#             Jy[tag0, i1 + 1, jy1] += deltay1 * Fy1 * Wx1 / S0
            
#             # Current at mid-last cell of patch 0
#             Jx[tag0, ix1, j1] += deltax1 * Fx1 * (1.0 - Wy1) / S0
#             Jx[tag0, ix1, j1 + 1] += deltax1 * Fx1 * Wy1 / S0

#             # Current on the edge of patch 0
#             Jx[tag0, -1, j1] += 0.5 * Fx1 * (1.0 - Wy1) / S0
#             Jx[tag0, -1, j1 + 1] += 0.5 * Fx1 * Wy1 / S0

#             # Current on patch 1
#             Jx[tag1, 0, j1] += 0.5 * Fx1 * (1.0 - Wy1) / S0
#             Jx[tag1, 0, j1 + 1] += 0.5 * Fx1 * Wy1 / S0

#             # Current deposited during second part of trajectory: particle is in patch 1

#             Jy[tag1, i2, jy2] += deltay2 * Fy2 * (1.0 - Wx2) / S0
#             Jy[tag1, i2 + 1, jy2] += deltay2 * Fy2 * Wx2 / S0
            
#             # Current at mid-last cell of patch 1
#             Jx[tag1, ix2, j2] += deltax2 * Fx2 * (1.0 - Wy2) / S0
#             Jx[tag1, ix2, j2 + 1] += deltax2 * Fx2 * Wy2 / S0

#             # Current on the edge of patch 1
#             Jx[tag1, 0, j2] += 0.5 * Fx2 * (1.0 - Wy2) / S0
#             Jx[tag1, 0, j2 + 1] += 0.5 * Fx2 * Wy2 / S0

#             # Current on patch 0
#             Jx[tag0, -1, j1] += 0.5 * Fx2 * (1.0 - Wy2) / S0
#             Jx[tag0, -1, j1 + 1] += 0.5 * Fx2 * Wy2 / S0

#         # Same thing if particle starts in patch 1
#         elif (sw == -1):
#             tag1 = 0

#             xr = 0.0
#             Fx1 = q * (xr - x1) / dt * w0
#             Wx1 = 0.5 * (x1 + xr) / dx - i1
#             Fy1 = q * (yr - y1) / dt * w0
#             Wy1 = 0.5 * (y1 + yr) / dy - j1

#             xr = 1.0
#             Fx2 = q * (x2 - xr) / dt * w0
#             Wx2 = 0.5 * (x2 + xr) / dx - i2
#             Fy2 = q * (y2 - yr) / dt * w0
#             Wy2 = 0.5 * (y2 + yr) / dy - j2

#             ix1 = i1 + 1
#             ix2 = i2 + 1

#             jy1 = j1 + 1
#             jy2 = j2 + 1

#             Jx[tag0, ix1, j1] += deltax1 * Fx1 * (1.0 - Wy1) / S0
#             Jx[tag0, ix1, j1 + 1] += deltax1 * Fx1 * Wy1 / S0

#             Jx[tag0, 0, j1] += 0.5 * Fx1 * (1.0 - Wy1) / S0
#             Jx[tag0, 0, j1 + 1] += 0.5 * Fx1 * Wy1 / S0

#             Jy[tag0, i1, jy1] += deltay1 * Fy1 * (1.0 - Wx1) / S0
#             Jy[tag0, i1 + 1, jy1] += deltay1 * Fy1 * Wx1 / S0

#             Jx[tag1, -1, j1] += 0.5 * Fx1 * (1.0 - Wy1) / S0
#             Jx[tag1, -1, j1 + 1] += 0.5 * Fx1 * Wy1 / S0

#             Jx[tag1, ix2, j2] += deltax2 * Fx2 * (1.0 - Wy2) / S0
#             Jx[tag1, ix2, j2 + 1] += deltax2 * Fx2 * Wy2 / S0

#             Jx[tag1, -1, j2] += 0.5 * Fx2 * (1.0 - Wy2) / S0
#             Jx[tag1, -1, j2 + 1] += 0.5 * Fx2 * Wy2 / S0

#             Jy[tag1, i2, jy2] += deltay2 * Fy2 * (1.0 - Wx2) / S0
#             Jy[tag1, i2 + 1, jy2] += deltay2 * Fy2 * Wx2 / S0

#             Jx[tag0, 0, j1] += 0.5 * Fx2 * (1.0 - Wy2) / S0
#             Jx[tag0, 0, j1 + 1] += 0.5 * Fx2 * Wy2 / S0
    
#     return

Jbuffx = N.zeros_like(Jx)
Jbuffy = N.zeros_like(Jy)
Jintx = N.zeros_like(Jx)
Jinty = N.zeros_like(Jy)

def filter_current(iter):

    Jbuffx[:, :, :] = Jx[:, :, :]
    Jbuffy[:, :, :] = Jy[:, :, :]
    
    # Pretend information from other patch is in a ghost cell

    if (iter == 0):
        return

    for i in range(iter):
    
        #######
        # Jx
        #######
        
        Jintx[:, :, :] = Jbuffx[:, :, :]

        # Bulk, regular 1-2-1 stencil
        Jbuffx[:, 2:(Nx_half-2), 1:(Ny_int-1)] = 0.25 * Jintx[:, 2:(Nx_half-2), 1:(Ny_int-1)] \
                                            + 0.125 * N.roll(Jintx, 1, axis = 1)[:, 2:(Nx_half-2), 1:(Ny_int-1)] \
                                            + 0.125 * N.roll(Jintx, -1, axis = 1)[:, 2:(Nx_half-2), 1:(Ny_int-1)] \
                                            + 0.125 * N.roll(Jintx, 1, axis = 2)[:, 2:(Nx_half-2), 1:(Ny_int-1)] \
                                            + 0.125 * N.roll(Jintx, -1, axis = 2)[:, 2:(Nx_half-2), 1:(Ny_int-1)] \
                                            + 0.0625 * N.roll(N.roll(Jintx, 1, axis = 1), 1, axis = 2)[:, 2:(Nx_half-2), 1:(Ny_int-1)] \
                                            + 0.0625 * N.roll(N.roll(Jintx, -1, axis = 1), -1, axis = 2)[:, 2:(Nx_half-2), 1:(Ny_int-1)] \
                                            + 0.0625 * N.roll(N.roll(Jintx, 1, axis = 1), -1, axis = 2)[:, 2:(Nx_half-2), 1:(Ny_int-1)] \
                                            + 0.0625 * N.roll(N.roll(Jintx, -1, axis = 1), 1, axis = 2)[:, 2:(Nx_half-2), 1:(Ny_int-1)]

        # Top edge cell
        Jbuffx[:, 2:(Nx_half - 2), -1] = 0.25 * Jintx[:,  2:(Nx_half - 2), -1] \
                                            + 0.125 * Jintx[:,  2:(Nx_half - 2), -2] \
                                            + 0.125 * Jintx[:, 2:(Nx_half - 2), 1] \
                                            + 0.125 * N.roll(Jintx, 1, axis = 1)[:, 2:(Nx_half - 2), -1] \
                                            + 0.125 * N.roll(Jintx, -1, axis = 1)[:, 2:(Nx_half - 2), -1] \
                                            + 0.0625 * N.roll(Jintx, 1, axis = 1)[:, 2:(Nx_half - 2), -2] \
                                            + 0.0625 * N.roll(Jintx, -1, axis = 1)[:, 2:(Nx_half - 2), -2] \
                                            + 0.0625 * N.roll(Jintx, -1, axis = 1)[:, 2:(Nx_half - 2), 1] \
                                            + 0.0625 * N.roll(Jintx, 1, axis = 1)[:, 2:(Nx_half - 2), 1]

        # Bottom edge cell
        Jbuffx[:, 2:(Nx_half - 2), 0] = 0.25 * Jintx[:,  2:(Nx_half - 2), 0] \
                                            + 0.125 * Jintx[:,  2:(Nx_half - 2), 1] \
                                            + 0.125 * Jintx[:, 2:(Nx_half - 2), -2] \
                                            + 0.125 * N.roll(Jintx, 1, axis = 1)[:, 2:(Nx_half - 2), 0] \
                                            + 0.125 * N.roll(Jintx, -1, axis = 1)[:, 2:(Nx_half - 2), 0] \
                                            + 0.0625 * N.roll(Jintx, 1, axis = 1)[:, 2:(Nx_half - 2), 1] \
                                            + 0.0625 * N.roll(Jintx, -1, axis = 1)[:, 2:(Nx_half - 2), 1] \
                                            + 0.0625 * N.roll(Jintx, -1, axis = 1)[:, 2:(Nx_half - 2), -2] \
                                            + 0.0625 * N.roll(Jintx, 1, axis = 1)[:, 2:(Nx_half - 2), -2]

        # Left and right edges
        for tag0 in [0,1]:
            
            tag1 = change_tag(tag0)
        
            # Half-edge cell, right
            Jbuffx[tag0, -2, 1:(Ny_int-1)] = 0.25 * Jintx[tag0, -2, 1:(Ny_int-1)] \
                                                + 0.125 * Jintx[tag0, -3, 1:(Ny_int-1)] \
                                                + 0.125 * Jintx[tag1, 1, 1:(Ny_int-1)] \
                                                + 0.125 * N.roll(Jintx, 1, axis = 2)[tag0, -2, 1:(Ny_int-1)] \
                                                + 0.125 * N.roll(Jintx, -1, axis = 2)[tag0, -2, 1:(Ny_int-1)] \
                                                + 0.0625 * N.roll(Jintx, 1, axis = 2)[tag0, -3, 1:(Ny_int-1)] \
                                                + 0.0625 * N.roll(Jintx, -1, axis = 2)[tag0, -3, 1:(Ny_int-1)] \
                                                + 0.0625 * N.roll(Jintx, 1, axis = 2)[tag1, 1, 1:(Ny_int-1)] \
                                                + 0.0625 * N.roll(Jintx, -1, axis = 2)[tag1, 1, 1:(Ny_int-1)]

            # Edge cell, right
            Jbuffx[tag0, -1, 1:(Ny_int-1)] = 0.25 * Jintx[tag0, -1, 1:(Ny_int-1)] \
                                                + 0.125 * Jintx[tag0, -2, 1:(Ny_int-1)] \
                                                + 0.125 * Jintx[tag1, 1, 1:(Ny_int-1)] \
                                                + 0.125 * N.roll(Jintx, 1, axis = 2)[tag0, -1, 1:(Ny_int-1)] \
                                                + 0.125 * N.roll(Jintx, -1, axis = 2)[tag0, -1, 1:(Ny_int-1)] \
                                                + 0.0625 * N.roll(Jintx, 1, axis = 2)[tag0, -2, 1:(Ny_int-1)] \
                                                + 0.0625 * N.roll(Jintx, -1, axis = 2)[tag0, -2, 1:(Ny_int-1)] \
                                                + 0.0625 * N.roll(Jintx, 1, axis = 2)[tag1, 1, 1:(Ny_int-1)] \
                                                + 0.0625 * N.roll(Jintx, -1, axis = 2)[tag1, 1, 1:(Ny_int-1)]

            # Half-edge cell, left                          
            Jbuffx[tag0, 1, 1:(Ny_int-1)] = 0.25 * Jintx[tag0, 1, 1:(Ny_int-1)] \
                                                + 0.125 * Jintx[tag0, 2, 1:(Ny_int-1)] \
                                                + 0.125 * Jintx[tag1, -2, 1:(Ny_int-1)] \
                                                + 0.125 * N.roll(Jintx, 1, axis = 2)[tag0, 1, 1:(Ny_int-1)] \
                                                + 0.125 * N.roll(Jintx, -1, axis = 2)[tag0, 1, 1:(Ny_int-1)] \
                                                + 0.0625 * N.roll(Jintx, 1, axis = 2)[tag0, 2, 1:(Ny_int-1)] \
                                                + 0.0625 * N.roll(Jintx, -1, axis = 2)[tag0, 2, 1:(Ny_int-1)] \
                                                + 0.0625 * N.roll(Jintx, 1, axis = 2)[tag1, -2, 1:(Ny_int-1)] \
                                                + 0.0625 * N.roll(Jintx, -1, axis = 2)[tag1, -2, 1:(Ny_int-1)]

            # Edge cell, left
            Jbuffx[tag0, 0, 1:(Ny_int-1)] = 0.25 * Jintx[tag0, 0, 1:(Ny_int-1)] \
                                                + 0.125 * Jintx[tag0, 1, 1:(Ny_int-1)] \
                                                + 0.125 * Jintx[tag1, -2, 1:(Ny_int-1)] \
                                                + 0.125 * N.roll(Jintx, 1, axis = 2)[tag0, 0, 1:(Ny_int-1)] \
                                                + 0.125 * N.roll(Jintx, -1, axis = 2)[tag0, 0, 1:(Ny_int-1)] \
                                                + 0.0625 * N.roll(Jintx, 1, axis = 2)[tag0, 1, 1:(Ny_int-1)] \
                                                + 0.0625 * N.roll(Jintx, -1, axis = 2)[tag0, 1, 1:(Ny_int-1)] \
                                                + 0.0625 * N.roll(Jintx, 1, axis = 2)[tag1, -2, 1:(Ny_int-1)] \
                                                + 0.0625 * N.roll(Jintx, -1, axis = 2)[tag1, -2, 1:(Ny_int-1)]

            # Top right corner
            Jbuffx[tag0, -2, -1] = 0.25 * Jintx[tag0, -2, -1] \
                                                + 0.125 * Jintx[tag0, -3, -1] \
                                                + 0.125 * Jintx[tag1, 1, -1] \
                                                + 0.125 * Jintx[tag0, -2, -2] \
                                                + 0.125 * Jintx[tag0, -2, 1] \
                                                + 0.0625 * Jintx[tag0, -3, -2] \
                                                + 0.0625 * Jintx[tag0, -3, 1] \
                                                + 0.0625 * Jintx[tag1, 1, -2] \
                                                + 0.0625 * Jintx[tag1, 1, 1]
            Jbuffx[tag0, -1, -1] = 0.25 * Jintx[tag0, -1, -1] \
                                                + 0.125 * Jintx[tag0, -2, -1] \
                                                + 0.125 * Jintx[tag1, 1, -1] \
                                                + 0.125  * Jintx[tag0, -1, -2] \
                                                + 0.125  * Jintx[tag0, -1, 1] \
                                                + 0.0625 * Jintx[tag0, -2, -2] \
                                                + 0.0625 * Jintx[tag0, -2, 1] \
                                                + 0.0625 * Jintx[tag1, 1, -2] \
                                                + 0.0625 * Jintx[tag1, 1, 1]

            # Top left corner
            Jbuffx[tag0, 1, -1] = 0.25 * Jintx[tag0, 1, -1] \
                                                + 0.125 * Jintx[tag0, 2, -1] \
                                                + 0.125 * Jintx[tag1, -2, -1] \
                                                + 0.125 * Jintx[tag0, 1, -2] \
                                                + 0.125 * Jintx[tag0, 1, 1] \
                                                + 0.0625 * Jintx[tag0, 2, -2] \
                                                + 0.0625 * Jintx[tag0, 2, 1] \
                                                + 0.0625 * Jintx[tag1, -2, -2] \
                                                + 0.0625 * Jintx[tag1, -2, 1]
            Jbuffx[tag0, 0, -1] = 0.25 * Jintx[tag0, 0, -1] \
                                                + 0.125 * Jintx[tag0, 1, -1] \
                                                + 0.125 * Jintx[tag1, -2, -1] \
                                                + 0.125  * Jintx[tag0, 0, -2] \
                                                + 0.125  * Jintx[tag0, 0, 1] \
                                                + 0.0625 * Jintx[tag0, 1, -2] \
                                                + 0.0625 * Jintx[tag0, 1, 1] \
                                                + 0.0625 * Jintx[tag1, -2, -2] \
                                                + 0.0625 * Jintx[tag1, -2, 1]
    
            # Bottom right corner
            Jbuffx[tag0, -2, 0] = 0.25 * Jintx[tag0, -2, 0] \
                                                + 0.125 * Jintx[tag0, -3, 0] \
                                                + 0.125 * Jintx[tag1, 1, 0] \
                                                + 0.125 * Jintx[tag0, -2, 1] \
                                                + 0.125 * Jintx[tag0, -2, -2] \
                                                + 0.0625 * Jintx[tag0, -3, 1] \
                                                + 0.0625 * Jintx[tag0, -3, -2] \
                                                + 0.0625 * Jintx[tag1, 1, -1] \
                                                + 0.0625 * Jintx[tag1, 1, -2]
            Jbuffx[tag0, -1, 0] = 0.25 * Jintx[tag0, -1, 0] \
                                                + 0.125 * Jintx[tag0, -2, 0] \
                                                + 0.125 * Jintx[tag1, 1, 0] \
                                                + 0.125  * Jintx[tag0, -1, 1] \
                                                + 0.125  * Jintx[tag0, -1, -2] \
                                                + 0.0625 * Jintx[tag0, -2, 1] \
                                                + 0.0625 * Jintx[tag0, -2, -2] \
                                                + 0.0625 * Jintx[tag1, 1, 1] \
                                                + 0.0625 * Jintx[tag1, 1, -2]
    
            # Bottom left corner
            Jbuffx[tag0, 1, 0] = 0.25 * Jintx[tag0, 1, 0] \
                                                + 0.125 * Jintx[tag0, 2, 0] \
                                                + 0.125 * Jintx[tag1, -2, 0] \
                                                + 0.125 * Jintx[tag0, 1, 1] \
                                                + 0.125 * Jintx[tag0, 1, -2] \
                                                + 0.0625 * Jintx[tag0, 2, 1] \
                                                + 0.0625 * Jintx[tag0, 2, -2] \
                                                + 0.0625 * Jintx[tag1, -2, 1] \
                                                + 0.0625 * Jintx[tag1, -2, -2]
            Jbuffx[tag0, 0, 0] = 0.25 * Jintx[tag0, 0, 0] \
                                                + 0.125 * Jintx[tag0, 1, 0] \
                                                + 0.125 * Jintx[tag1, -2, 0] \
                                                + 0.125  * Jintx[tag0, 0, 1] \
                                                + 0.125  * Jintx[tag0, 0, -2] \
                                                + 0.0625 * Jintx[tag0, 1, 1] \
                                                + 0.0625 * Jintx[tag0, 1, -2] \
                                                + 0.0625 * Jintx[tag1, -2, 1] \
                                                + 0.0625 * Jintx[tag1, -2, -2]
                                                           
        #######
        # Jy
        #######

        Jinty[:, :, :] = Jbuffy[:, :, :]

        # Bulk
        Jbuffy[:, 1:(Nx_int-1), 2:(Ny_half-2)] = 0.25 * Jinty[:, 1:(Nx_int-1), 2:(Ny_half-2)] \
                                            + 0.125 * N.roll(Jinty, 1, axis = 1)[:, 1:(Nx_int-1), 2:(Ny_half-2)] \
                                            + 0.125 * N.roll(Jinty, -1, axis = 1)[:, 1:(Nx_int-1), 2:(Ny_half-2)] \
                                            + 0.125 * N.roll(Jinty, 1, axis = 2)[:, 1:(Nx_int-1), 2:(Ny_half-2)] \
                                            + 0.125 * N.roll(Jinty, -1, axis = 2)[:, 1:(Nx_int-1), 2:(Ny_half-2)] \
                                            + 0.0625 * N.roll(N.roll(Jinty, 1, axis = 1), 1, axis = 2)[:, 1:(Nx_int-1), 2:(Ny_half-2)] \
                                            + 0.0625 * N.roll(N.roll(Jinty, -1, axis = 1), -1, axis = 2)[:, 1:(Nx_int-1), 2:(Ny_half-2)] \
                                            + 0.0625 * N.roll(N.roll(Jinty, 1, axis = 1), -1, axis = 2)[:, 1:(Nx_int-1), 2:(Ny_half-2)] \
                                            + 0.0625 * N.roll(N.roll(Jinty, -1, axis = 1), 1, axis = 2)[:, 1:(Nx_int-1), 2:(Ny_half-2)]

        # Half-edge cell, top
        Jbuffy[:, 1:(Nx_int - 1), -2] = 0.25 * Jinty[:, 1:(Nx_int - 1), -2] \
                                            + 0.125 * Jinty[:, 1:(Nx_int - 1), -3] \
                                            + 0.125 * Jinty[:, 1:(Nx_int - 1), 1] \
                                            + 0.125 * N.roll(Jinty, 1, axis = 1)[:, 1:(Nx_int - 1), -2] \
                                            + 0.125 * N.roll(Jinty, -1, axis = 1)[:, 1:(Nx_int - 1), -2] \
                                            + 0.0625 * N.roll(Jinty, 1, axis = 1)[:, 1:(Nx_int - 1), -3] \
                                            + 0.0625 * N.roll(Jinty, -1, axis = 1)[:, 1:(Nx_int - 1), -3] \
                                            + 0.0625 * N.roll(Jinty, 1, axis = 1)[:, 1:(Nx_int - 1), 1] \
                                            + 0.0625 * N.roll(Jinty, -1, axis = 1)[:, 1:(Nx_int - 1), 1]

        # Edge cell, top                                            
        Jbuffy[:, 1:(Nx_int - 1), -1] = 0.25 * Jinty[:, 1:(Nx_int - 1), -1] \
                                            + 0.125 * Jinty[:, 1:(Nx_int - 1), -2] \
                                            + 0.125 * Jinty[:, 1:(Nx_int - 1), 1] \
                                            + 0.125 * N.roll(Jinty, 1, axis = 1)[:, 1:(Nx_int - 1), -1] \
                                            + 0.125 * N.roll(Jinty, -1, axis = 1)[:, 1:(Nx_int - 1), -1] \
                                            + 0.0625 * N.roll(Jinty, 1, axis = 1)[:, 1:(Nx_int - 1), -2] \
                                            + 0.0625 * N.roll(Jinty, -1, axis = 1)[:, 1:(Nx_int - 1), -2] \
                                            + 0.0625 * N.roll(Jinty, 1, axis = 1)[:, 1:(Nx_int - 1), 1] \
                                            + 0.0625 * N.roll(Jinty, -1, axis = 1)[:, 1:(Nx_int - 1), 1]

        # Half-edge cell, bottom                                 
        Jbuffy[:, 1:(Nx_int - 1), 1] = 0.25 * Jinty[:, 1:(Nx_int - 1), 1] \
                                            + 0.125 * Jinty[:, 1:(Nx_int - 1), 2] \
                                            + 0.125 * Jinty[:, 1:(Nx_int - 1), -2] \
                                            + 0.125 * N.roll(Jinty, 1, axis = 1)[:, 1:(Nx_int - 1), 1] \
                                            + 0.125 * N.roll(Jinty, -1, axis = 1)[:, 1:(Nx_int - 1), 1] \
                                            + 0.0625 * N.roll(Jinty, 1, axis = 1)[:, 1:(Nx_int - 1), 2] \
                                            + 0.0625 * N.roll(Jinty, -1, axis = 1)[:, 1:(Nx_int - 1), 2] \
                                            + 0.0625 * N.roll(Jinty, 1, axis = 1)[:, 1:(Nx_int - 1), -2] \
                                            + 0.0625 * N.roll(Jinty, -1, axis = 1)[:, 1:(Nx_int - 1), -2]

        # Edge cell, bottom                                 
        Jbuffy[:, 1:(Nx_int - 1), 0] = 0.25 * Jinty[:, 1:(Nx_int - 1), 0] \
                                            + 0.125 * Jinty[:, 1:(Nx_int - 1), 1] \
                                            + 0.125 * Jinty[:, 1:(Nx_int - 1), -2] \
                                            + 0.125 * N.roll(Jinty, 1, axis = 1)[:, 1:(Nx_int - 1), 0] \
                                            + 0.125 * N.roll(Jinty, -1, axis = 1)[:, 1:(Nx_int - 1), 0] \
                                            + 0.0625 * N.roll(Jinty, 1, axis = 1)[:, 1:(Nx_int - 1), 1] \
                                            + 0.0625 * N.roll(Jinty, -1, axis = 1)[:, 1:(Nx_int - 1), 1] \
                                            + 0.0625 * N.roll(Jinty, 1, axis = 1)[:, 1:(Nx_int - 1), -2] \
                                            + 0.0625 * N.roll(Jinty, -1, axis = 1)[:, 1:(Nx_int - 1), -2]

        # Left and right edges
        for tag0 in [0,1]:
            
            tag1 = change_tag(tag0)
            
            # Right edge
            Jbuffy[tag0, -1, 2:(Ny_half-2)] = 0.25 * Jinty[tag0, -1, 2:(Ny_half-2)] \
                                                + 0.125 * Jinty[tag0, -2, 2:(Ny_half-2)] \
                                                + 0.125 * Jinty[tag1, 1, 2:(Ny_half-2)] \
                                                + 0.125 * N.roll(Jinty, 1, axis = 2)[tag0, -1, 2:(Ny_half-2)] \
                                                + 0.125 * N.roll(Jinty, -1, axis = 2)[tag0, -1, 2:(Ny_half-2)] \
                                                + 0.0625 * N.roll(Jinty, 1, axis = 2)[tag0, -2, 2:(Ny_half-2)] \
                                                + 0.0625 * N.roll(Jinty, -1, axis = 2)[tag0, -2, 2:(Ny_half-2)] \
                                                + 0.0625 * N.roll(Jinty, -1, axis = 2)[tag1, 1, 2:(Ny_half-2)] \
                                                + 0.0625 * N.roll(Jinty, 1, axis = 2)[tag1, 1, 2:(Ny_half-2)]

            # Left edge
            Jbuffy[tag0, 0, 2:(Ny_half-2)] = 0.25 * Jinty[tag0, 0, 2:(Ny_half-2)] \
                                                + 0.125 * Jinty[tag0, 1, 2:(Ny_half-2)] \
                                                + 0.125 * Jinty[tag1, -2, 2:(Ny_half-2)] \
                                                + 0.125 * N.roll(Jinty, 1, axis = 2)[tag0, 0, 2:(Ny_half-2)] \
                                                + 0.125 * N.roll(Jinty, -1, axis = 2)[tag0, 0, 2:(Ny_half-2)] \
                                                + 0.0625 * N.roll(Jinty, 1, axis = 2)[tag0, 1, 2:(Ny_half-2)] \
                                                + 0.0625 * N.roll(Jinty, -1, axis = 2)[tag0, 1, 2:(Ny_half-2)] \
                                                + 0.0625 * N.roll(Jinty, -1, axis = 2)[tag1, -2, 2:(Ny_half-2)] \
                                                + 0.0625 * N.roll(Jinty, 1, axis = 2)[tag1, -2, 2:(Ny_half-2)]

            # Top right corner
            Jbuffy[tag0, -1, -2] = 0.25 * Jinty[tag0, -1, -2] \
                                                + 0.125 * Jinty[tag0, -2, -2] \
                                                + 0.125 * Jinty[tag1, 1, -2] \
                                                + 0.125  * Jinty[tag0, -1, -3] \
                                                + 0.125  * Jinty[tag0, -1, 1] \
                                                + 0.0625 * Jinty[tag0, -2, -3] \
                                                + 0.0625 * Jinty[tag0, -2, 1] \
                                                + 0.0625 * Jinty[tag1, 1, -3] \
                                                + 0.0625 * Jinty[tag1, 1, 1]
            Jbuffy[tag0, -1, -1] = 0.25 * Jinty[tag0, -1, -1] \
                                                + 0.125 * Jinty[tag0, -2, -1] \
                                                + 0.125 * Jinty[tag1, 1, -1] \
                                                + 0.125  * Jinty[tag0, -1, -2] \
                                                + 0.125  * Jinty[tag0, -1, 1] \
                                                + 0.0625 * Jinty[tag0, -2, -2] \
                                                + 0.0625 * Jinty[tag0, -2, 1] \
                                                + 0.0625 * Jinty[tag1, 1, -2] \
                                                + 0.0625 * Jinty[tag1, 1, 1]

            # Bottom right corner
            Jbuffy[tag0, -1, 1] = 0.25 * Jinty[tag0, -1, 1] \
                                                + 0.125 * Jinty[tag0, -2, 1] \
                                                + 0.125 * Jinty[tag1, 1, 1] \
                                                + 0.125  * Jinty[tag0, -1, 2] \
                                                + 0.125  * Jinty[tag0, -1, -2] \
                                                + 0.0625 * Jinty[tag0, -2, 2] \
                                                + 0.0625 * Jinty[tag0, -2, -2] \
                                                + 0.0625 * Jinty[tag1, 1, 2] \
                                                + 0.0625 * Jinty[tag1, 1, -2]
            Jbuffy[tag0, -1, 0] = 0.25 * Jinty[tag0, -1, 0] \
                                                + 0.125 * Jinty[tag0, -2, 0] \
                                                + 0.125 * Jinty[tag1, 1, 0] \
                                                + 0.125  * Jinty[tag0, -1, 1] \
                                                + 0.125  * Jinty[tag0, -1, -2] \
                                                + 0.0625 * Jinty[tag0, -2, 1] \
                                                + 0.0625 * Jinty[tag0, -2, -2] \
                                                + 0.0625 * Jinty[tag1, 1, 1] \
                                                + 0.0625 * Jinty[tag1, 1, -2]

            # Top left corner
            Jbuffy[tag0, 0, -2] = 0.25 * Jinty[tag0, 0, -2] \
                                                + 0.125 * Jinty[tag0, 1, -2] \
                                                + 0.125 * Jinty[tag1, -2, -2] \
                                                + 0.125  * Jinty[tag0, 0, -3] \
                                                + 0.125  * Jinty[tag0, 0, 1] \
                                                + 0.0625 * Jinty[tag0, 1, -3] \
                                                + 0.0625 * Jinty[tag0, 1, 1] \
                                                + 0.0625 * Jinty[tag1, -2, -3] \
                                                + 0.0625 * Jinty[tag1, -2, 1]
            Jbuffy[tag0, 0, -1] = 0.25 * Jinty[tag0, 0, -1] \
                                                + 0.125 * Jinty[tag0, 1, -1] \
                                                + 0.125 * Jinty[tag1, -2, -1] \
                                                + 0.125  * Jinty[tag0, 0, -2] \
                                                + 0.125  * Jinty[tag0, 0, 1] \
                                                + 0.0625 * Jinty[tag0, 1, -2] \
                                                + 0.0625 * Jinty[tag0, 1, 1] \
                                                + 0.0625 * Jinty[tag1, -2, -2] \
                                                + 0.0625 * Jinty[tag1, -2, 1]

            # Bottom left corner
            Jbuffy[tag0, 0, 1] = 0.25 * Jinty[tag0, 0, 1] \
                                                + 0.125 * Jinty[tag0, 1, 1] \
                                                + 0.125 * Jinty[tag1, -2, 1] \
                                                + 0.125  * Jinty[tag0, 0, 2] \
                                                + 0.125  * Jinty[tag0, 0, -2] \
                                                + 0.0625 * Jinty[tag0, 1, 2] \
                                                + 0.0625 * Jinty[tag0, 1, -2] \
                                                + 0.0625 * Jinty[tag1, -2, 2] \
                                                + 0.0625 * Jinty[tag1, -2, -2]
            Jbuffy[tag0, 0, 0] = 0.25 * Jinty[tag0, 0, 0] \
                                                + 0.125 * Jinty[tag0, 1, 0] \
                                                + 0.125 * Jinty[tag1, -2, 0] \
                                                + 0.125  * Jinty[tag0, 0, 1] \
                                                + 0.125  * Jinty[tag0, 0, -2] \
                                                + 0.0625 * Jinty[tag0, 1, 1] \
                                                + 0.0625 * Jinty[tag0, 1, -2] \
                                                + 0.0625 * Jinty[tag1, -2, 1] \
                                                + 0.0625 * Jinty[tag1, -2, -2]

    Jx[:, :, :] = Jbuffx[:, :, :]
    Jy[:, :, :] = Jbuffy[:, :, :]

    return

########
# Boundary conditions
########

sig_abs = 1.0

# Periodic outer boundaries

def BC_periodic_B(dtin, Exin, Eyin, Bzin):

    Bz[0, 0, :]  -= dtin * sig_abs * (Eyin[0, 0, :] + Bzin[0, 0, :] - (Eyin[1, -1, :] + Bzin[1, -1, :]))   / dx / P_half_2[0]
    Bz[1, -1, :] += dtin * sig_abs * (Eyin[1, -1, :] - Bzin[1, -1, :] - (Eyin[0, 0, :] - Bzin[0, 0, :])) / dx / P_half_2[-1]

    Bz[0, :, 0]  += dtin * sig_abs * (Exin[0, :, 0] - Bzin[0, :, 0] - (Exin[0, :, -1] - Bzin[0, :, -1]))   / dx / P_half_2[0]
    Bz[0, :, -1] -= dtin * sig_abs * (Exin[0, :, -1] + Bzin[0, :, -1] - (Exin[0, :, 0] + Bzin[0, :, 0])) / dx / P_half_2[-1]
    # Bz[0, -1, :] += dtin * sig_abs * (Eyin[0, -1, :] - Bzin[0, -1, :]) / dx / P_half_2[-1]

    Bz[1, :, 0]  += dtin * sig_abs * (Exin[1, :, 0] - Bzin[1, :, 0] - (Exin[1, :, -1] - Bzin[1, :, -1])) / dx / P_half_2[0]
    Bz[1, :, -1] -= dtin * sig_abs * (Exin[1, :, -1] + Bzin[1, :, -1] - (Exin[1, :, 0] + Bzin[1, :, 0])) / dx / P_half_2[-1]
    # Bz[1, 0, :] -= dtin * sig_abs * (Eyin[1, 0, :] + Bzin[1, 0, :]) / dx / P_half_2[0]

    return

def BC_periodic_E(dtin, Exin, Eyin, Bzin):

    Ey[0, 0, :]  -= dtin * sig_abs * (Eyin[0, 0, :] + Bzin[0, 0, :] - (Eyin[1, -1, :] + Bzin[1, -1, :]))   / dx / P_half_2[0]
    Ey[1, -1, :] -= dtin * sig_abs * (Eyin[1, -1, :] - Bzin[1, -1, :] - (Eyin[0, 0, :] - Bzin[0, 0, :])) / dx / P_half_2[-1]

    Ex[0, :, 0]  -= dtin * sig_abs * (Exin[0, :, 0] - Bzin[0, :, 0] - (Exin[0, :, -1] - Bzin[0, :, -1]))   / dx / P_half_2[0]
    Ex[0, :, -1] -= dtin * sig_abs * (Exin[0, :, -1] + Bzin[0, :, -1] - (Exin[0, :, 0] + Bzin[0, :, 0])) / dx / P_half_2[-1]    
    # Ey[0, -1, :] -= dtin * sig_abs * (Eyin[0, -1, :] - Bzin[0, -1, :]) / dx / P_half_2[-1]

    Ex[1, :, 0]  -= dtin * sig_abs * (Exin[1, :, 0] - Bzin[1, :, 0] - (Exin[1, :, -1] - Bzin[1, :, -1])) / dx / P_half_2[0]
    Ex[1, :, -1] -= dtin * sig_abs * (Exin[1, :, -1] + Bzin[1, :, -1] - (Exin[1, :, 0] + Bzin[1, :, 0])) / dx / P_half_2[-1]    
    # Ey[1, 0, :] -= dtin * sig_abs * (Eyin[1, 0, :] + Bzin[1, 0, :]) / dx / P_half_2[0]

    return

def BC_penalty_B(dtin, Exin, Eyin, Bzin):
    Bz[0, -1, :] -= dtin * sig_abs * (Bzin[0, -1, :] - Eyin[0, -1, :] - (Bzin[1, 0, :] - Eyin[1, 0, :])) / dx / P_half_2[-1]
    Bz[1, 0, :]  -= dtin * sig_abs * (Bzin[1, 0, :] + Eyin[1, 0, :] - (Bzin[0, -1, :] + Eyin[0, -1, :])) / dx / P_half_2[0]
    return

def BC_penalty_E(dtin, Exin, Eyin, Bzin):
    Ey[0, -1, :] -= dtin * sig_abs * (Eyin[0, -1, :] - Bzin[0, -1, :] - (Eyin[1, 0, :] - Bzin[1, 0, :])) / dx / P_half_2[-1]
    Ey[1, 0, :]  -= dtin * sig_abs * (Eyin[1, 0, :] + Bzin[1, 0, :] - (Eyin[0, -1, :] + Bzin[0, -1, :])) / dx / P_half_2[0]
    return

########
# Visualization
########

ratio2 = 0.125

vm = 0.001

from matplotlib.gridspec import GridSpec

def plot_fields(idump, it):

    fig = P.figure(1, facecolor='w', figsize=(60,10))
    gs = GridSpec(3, 1, figure=fig)
    
    ax = fig.add_subplot(gs[0, 0])

    P.pcolormesh(xEx_grid, yEx_grid, Ex[1, :, :], vmin = -vm, vmax = vm, cmap = 'RdBu_r')
    P.pcolormesh(xEx_grid + x_max, yEx_grid, Ex[0, :, :], vmin = -vm, vmax = vm, cmap = 'RdBu_r')
        
    P.title(r'$E_x$', fontsize=16)
        
    P.ylim((0.0, y_max))
    P.xlim((0.0, 2.0 * x_max))
    ax.set_aspect(1.0/ax.get_data_ratio()*ratio2)

    P.plot([1.0, 1.0],[0, 1.0], color='k')

    ax = fig.add_subplot(gs[1, 0])

    P.pcolormesh(xEy_grid, yEy_grid, Ey[1, :, :], vmin = -vm, vmax = vm, cmap = 'RdBu_r')
    P.pcolormesh(xEy_grid + x_max, yEy_grid, Ey[0, :, :], vmin = -vm, vmax = vm, cmap = 'RdBu_r')

    P.title(r'$E_y$', fontsize=16)
        
    P.ylim((0.0, y_max))
    P.xlim((0.0, 2.0 * x_max))
    ax.set_aspect(1.0/ax.get_data_ratio()*ratio2)

    P.plot([1.0, 1.0],[0, 1.0], color='k')

    ax = fig.add_subplot(gs[2, 0])

    P.pcolormesh(xBz_grid, yBz_grid, Bz[1, :, :], vmin = -vm, vmax = vm, cmap = 'RdBu_r')
    P.pcolormesh(xBz_grid + x_max, yBz_grid, Bz[0, :, :], vmin = -vm, vmax = vm, cmap = 'RdBu_r')

    P.title(r'$B_z$', fontsize=16)
        
    P.ylim((0.0, y_max))
    P.xlim((0.0, 2.0 * x_max))
    ax.set_aspect(1.0/ax.get_data_ratio()*ratio2)

    P.plot([1.0, 1.0],[0, 1.0], color='k')
        
    figsave_png(fig, "../snapshots_langmuir/fields_" + str(idump))
    
    P.close('all')

vmc = 1e-3

def plot_currents(idump, it):

    fig = P.figure(1, facecolor='w', figsize=(30,10))
    gs = GridSpec(2, 1, figure=fig)
    
    ax = fig.add_subplot(gs[0, 0])

    P.pcolormesh(xEx_grid, yEx_grid, Jx[0, :, :], vmin = -vmc, vmax = vmc, cmap = 'RdBu_r')
    P.pcolormesh(xEx_grid + x_max, yEx_grid, Jx[1, :, :], vmin = -vmc, vmax = vmc, cmap = 'RdBu_r')
    # P.pcolormesh(xEx_grid, yEx_grid, Jx[0, :, :], vmin = -1.0, vmax = 1.0, cmap = 'RdBu_r')
    # P.pcolormesh(xEx_grid + x_max + 2.0 * dx, yEx_grid, Jx[1, :, :], vmin = -1.0, vmax = 1.0, cmap = 'RdBu_r')
        
    P.title(r'$J_x$', fontsize=16)
        
    P.ylim((0.0, y_max))
    P.xlim((0.0, 2.0 * x_max))
    ax.set_aspect(1.0/ax.get_data_ratio()*ratio2)

    P.plot([1.0, 1.0],[0, 1.0], color='k')

    ax = fig.add_subplot(gs[1, 0])

    # P.pcolormesh(xEx_grid, yEx_grid, Ex[0, :, :], vmin = -vm, vmax = vm, cmap = 'RdBu_r')
    # P.pcolormesh(xEx_grid + x_max + 2.0 * dx, yEx_grid, Ex[1, :, :], vmin = -vm, vmax = vm, cmap = 'RdBu_r')
    P.pcolormesh(xEy_grid, yEy_grid, Jy[0, :, :], vmin = -vm, vmax = vm, cmap = 'RdBu_r')
    P.pcolormesh(xEy_grid + x_max, yEy_grid, Jy[1, :, :], vmin = -vm, vmax = vm, cmap = 'RdBu_r')

    P.title(r'$J_y$', fontsize=16)
        
    P.ylim((0.0, y_max))
    P.xlim((0.0, 2.0 * x_max))
    ax.set_aspect(1.0/ax.get_data_ratio()*ratio2)

    P.plot([1.0, 1.0],[0, 1.0], color='k')
    
    figsave_png(fig, "../snapshots_langmuir/currents_" + str(idump))
    
    P.close('all')

ratio = 0.5

# vm = 0.02

from matplotlib.gridspec import GridSpec

def plot_scatter(idump, it):
    
    xtemp = N.zeros_like(xp)
    xtemp[tag == 0] = xp[tag == 0]
    xtemp[tag == 1] = xp[tag == 1] + x_max

    fig = P.figure(1, facecolor='w', figsize=(30,10))    
    ax = fig.add_subplot(111)

    P.scatter(xtemp[:, 0], yp[:, 0], color='k', s=5) #Positrons
    P.scatter(xtemp[:, 1], yp[:, 0], color='b', s=5) #Electrons
        
    P.title(r'$E_x$', fontsize=16)
        
    P.ylim((0.0, 0.5))
    P.xlim((0.0, 2.0))
    ax.set_aspect(1.0/ax.get_data_ratio()*ratio)

    P.plot([1.0, 1.0],[0, 1.0], color='k')

    figsave_png(fig, "../snapshots_langmuir/scatter_" + str(idump))
    
    P.close('all')

########
# Initialization
########

amp_l = 0.0
amp_w = 0.001
n_mode = 4
n_iter = 0

wave = 2.0 * (2.0 * x_max - x_min) / n_mode
Bzin0 = amp_w * N.cos(2.0 * N.pi * (xBz_grid - x_min) / wave) #* N.exp(-((xBz_grid - 0.5)**2) / 0.1**2)
Exin0 = amp_l * N.sin(2.0 * N.pi * (xEx_grid - x_min) / wave)
Eyin0 = amp_w * N.cos(2.0 * N.pi * (xEy_grid - x_min) / wave) #* N.exp(-((xEy_grid - 0.5)**2) / 0.1**2)
Bzin1 = amp_w * N.cos(2.0 * N.pi * (xBz_grid - x_min + x_max) / wave) #* N.exp(-((xBz_grid - 0.5)**2) / 0.1**2)
Exin1 = amp_l * N.sin(2.0 * N.pi * (xEx_grid - x_min + x_max) / wave)
Eyin1 = amp_w * N.cos(2.0 * N.pi * (xEy_grid - x_min + x_max) / wave) #* N.exp(-((xEy_grid - 0.5)**2) / 0.1**2)

Bz[0, :, :] = Bzin0[:, :]
Ey[0, :, :] = Eyin0[:, :]

Bz[1, :, :] = Bzin1[:, :]
Ey[1, :, :] = Eyin1[:, :]

Ex[0, :, :] = Exin0[:, :]
Ex[1, :, :] = Exin1[:, :]

initialize_part()

########
# Main routine
########

idump = 0
time = dt * N.arange(Nt)
energy = N.zeros((n_patches, Nt))
patches = N.array(range(n_patches))

# Fields at previous time steps
Ex0 = N.zeros_like(Ex)
Ey0 = N.zeros_like(Ey)
Bz0 = N.zeros_like(Bz)

for it in tqdm(range(Nt), "Progression"):
    if ((it % FDUMP) == 0):
        plot_fields(idump, it)
        plot_currents(idump, it)
        # plot_scatter(idump, it)
        idump += 1

    # 1st Faraday substep, starting from B at n-1/2, E at n, finishing with B at n
    compute_diff_E(patches)
    push_B(patches, it, 0.5 * dt)
    
    # Here, Bz is defined at n, no need for averaging
    BC_periodic_B(0.5 * dt, Ex[:, :, :], Ey[:, :, :], Bz[:, :, :])
    BC_penalty_B(0.5 * dt, Ex[:, :, :], Ey[:, :, :], Bz[:, :, :])
    
    Bz0[:, :, :] = Bz[:, :, :]

    # Particle push
    push_u(0)
    push_u(1)
    push_x(0)
    push_x(1)

    Jx[:, :, :] = 0.0
    Jy[:, :, :] = 0.0

    # Current deposition and BC    
    for ip in range(np):
        BC_part(ip, 0)
        deposit_particle(ip, 0)
        BC_part(ip, 1)
        deposit_particle(ip, 1)
    
    filter_current(n_iter)

    # 2nd Faraday substep, starting with B at n, finishing with B at n + 1/2
    compute_diff_E(patches)
    push_B(patches, it, 0.5 * dt)
    
    # Use Bz0, defined at n, this time
    BC_periodic_B(0.5 * dt, Ex[:, :, :], Ey[:, :, :], Bz0[:, :, :])
    BC_penalty_B(0.5 * dt, Ex[:, :, :], Ey[:, :, :], Bz0[:, :, :])

    # Ampère step, starting with E at n, finishing with E at n + 1
    Ex0[:, :, :] = Ex[:, :, :]
    Ey0[:, :, :] = Ey[:, :, :]
    compute_diff_B(patches)
    push_E(patches, it, dt)
      
    # Use averaged E field, defined at n + 1/2  
    BC_periodic_E(dt, 0.5 * (Ex0[:, :, :] + Ex[:, :, :]), 0.5 * (Ey0[:, :, :] + Ey[:, :, :]), Bz[:, :, :])
    BC_penalty_E(dt, 0.5 * (Ex0[:, :, :] + Ex[:, :, :]), 0.5 * (Ey0[:, :, :] + Ey[:, :, :]), Bz[:, :, :])

    energy[0, it] = dx * dy * N.sum(Bz[0, :, :]**2) + N.sum(Ex[0, :, :]**2) + N.sum(Ey[0, :, :]**2)
    energy[1, it] = dx * dy * N.sum(Bz[1, :, :]**2) + N.sum(Ex[1, :, :]**2) + N.sum(Ey[1, :, :]**2)
