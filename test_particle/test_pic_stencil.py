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
Nx = 200 # Number of cells
Nx_int = Nx + 1 # Number of integer points
Nx_half = Nx + 2 # NUmber of hlaf-step points
Ny = 200 # Number of cells
Ny_int = Ny + 1 # Number of integer points
Ny_half = Ny + 2 # NUmber of hlaf-step points

q = 1e-2 # Absolute value of charge

Nt = 4000 #351 # Number of iterations
FDUMP = 5 # Dump frequency

x_min, x_max = 0.0, 1.0
dx = (x_max - x_min) / Nx
y_min, y_max = 0.0, 1.0
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
rho0f = N.zeros((n_patches, Nx_int, Ny_int))
rho1f = N.zeros((n_patches, Nx_int, Ny_int))

dBzdx = N.zeros((n_patches, Nx_int, Ny_half))
dBzdy = N.zeros((n_patches, Nx_half, Ny_int))
dExdy = N.zeros((n_patches, Nx_half, Ny_half))
dEydx = N.zeros((n_patches, Nx_half, Ny_half))
dBxdy = N.zeros((n_patches, Nx_int, Ny_int))
dBydx = N.zeros((n_patches, Nx_int, Ny_int))
dEzdx = N.zeros((n_patches, Nx_half, Ny_int))
dEzdy = N.zeros((n_patches, Nx_int, Ny_half))

divE0 = N.zeros((n_patches, Nx_int, Ny_int))
divE1 = N.zeros((n_patches, Nx_int, Ny_int))
divcharge = N.zeros((n_patches, Nx_int, Ny_int))
divJ = N.zeros_like(divE0)
divJf = N.zeros_like(divE0)
dxJx = N.zeros_like(divE0)

########
# Pushers
########

delta = 0.5
gamma = 0.75 #0.9 #1 - 0.5 * delta  #Condition for currents at interface to match when particle in last cell  #0.2

P_int_2 = N.ones(Nx_int)

P_int_2[0]  = (3.0 * gamma - 2.0) / 2.0 / (gamma - delta)
P_int_2[-1] = (3.0 * gamma - 2.0) / 2.0 / (gamma - delta) 

P_half_2 = N.ones(Nx_half)
# P_half_2[1] = 0.25 
# P_half_2[2] = 1.25 
# P_half_2[-3] = 1.25 
# P_half_2[-2] = 0.25 

P_half_2[0]  = (2.0 - 3.0 * delta) * (1.0 - gamma) / (gamma - delta) 
P_half_2[-1] = (2.0 - 3.0 * delta) * (1.0 - gamma) / (gamma - delta) 


def compute_diff_B(p):
    
    # dBzdx[p, 0, :] = (- Bz[p, 0, :] + Bz[p, 1, :]) / dx / 0.5
    dBzdx[p, 0, :] = (2.0 * (delta - 1.0) * Bz[p, 0, :] + (2.0 - 3.0 * delta) * Bz[p, 1, :] + delta * Bz[p, 2, :]) / dx
    dBzdx[p, 1, :] = (2.0 * (gamma - 1.0) * Bz[p, 0, :] + (2.0 - 3.0 * gamma) * Bz[p, 1, :] + gamma * Bz[p, 2, :]) / dx

    # dBzdx[p, Nx_int - 1, :] = (- Bz[p, -2, :] + Bz[p, -1, :]) / dx / 0.5
    dBzdx[p, Nx_int - 1, :] = (- delta * Bz[p, -3, :] - (2.0 - 3.0 * delta) * Bz[p, -2, :] - 2.0 * (delta - 1.0) * Bz[p, -1, :]) / dx
    dBzdx[p, Nx_int - 2, :] = (- gamma * Bz[p, -3, :] - (2.0 - 3.0 * gamma) * Bz[p, -2, :] - 2.0 * (gamma - 1.0) * Bz[p, -1, :]) / dx
    
    dBzdx[p, 2:(Nx_int - 2), :] = (N.roll(Bz, -1, axis = 1)[p, 2:(Nx_int - 2), :] - Bz[p, 2:(Nx_int - 2), :]) / dx

    # dBzdy[p, :, 0] = (- Bz[p, :, 0] + Bz[p, :, 1]) / dy / 0.5
    dBzdy[p, :, 0] = (2.0 * (delta - 1.0) * Bz[p, :, 0] + (2.0 - 3.0 * delta) * Bz[p, :, 1] + delta * Bz[p, :, 2]) / dy
    dBzdy[p, :, 1] = (2.0 * (gamma - 1.0) * Bz[p, :, 0] + (2.0 - 3.0 * gamma) * Bz[p, :, 1] + gamma * Bz[p, :, 2]) / dy
    
    # dBzdy[p, :, Ny_int - 1] = (- Bz[p, :, -2] + Bz[p, :, -1]) / dy / 0.5
    dBzdy[p, :, Ny_int - 1] = (- delta * Bz[p, :, -3] - (2.0 - 3.0 * delta) * Bz[p, :, -2] - 2.0 * (delta - 1.0) * Bz[p, :, -1]) / dy
    dBzdy[p, :, Ny_int - 2] = (- gamma * Bz[p, :, -3] - (2.0 - 3.0 * gamma) * Bz[p, :, -2] - 2.0 * (gamma - 1.0) * Bz[p, :, -1]) / dy

    dBzdy[p, :, 2:(Ny_int - 2)] = (N.roll(Bz, -1, axis = 2)[p, :, 2:(Ny_int - 2)] - Bz[p, :, 2:(Ny_int - 2)]) / dy

def compute_diff_E(p):

    dEydx[p, 0, :] = (- 0.5 * Ey[p, 0, :] + 0.5 * Ey[p, 1, :]) / dx / 0.5
    dEydx[p, 1, :] = (- 0.25 * Ey[p, 0, :] + 0.25 * Ey[p, 1, :]) / dx / 0.25
    dEydx[p, 2, :] = (- 0.25 * Ey[p, 0, :] - 0.75 * Ey[p, 1, :] + Ey[p, 2, :]) / dx / (5.0 / 4.0)
    dEydx[p, Nx_half - 3, :] = (- Ey[p, -3, :] + 0.75 * Ey[p, -2, :] + 0.25 * Ey[p, -1, :]) / dx / (5.0 / 4.0)
    dEydx[p, Nx_half - 2, :] = (- 0.25 * Ey[p, -2, :] + 0.25 * Ey[p, -1, :]) / dx / 0.25
    dEydx[p, Nx_half - 1, :] = (- 0.5 * Ey[p, -2, :] + 0.5 * Ey[p, -1, :]) / dx / 0.5
    dEydx[p, 3:(Nx_half - 3), :] = (Ey[p, 3:(Nx_half - 3), :] - N.roll(Ey, 1, axis = 1)[p, 3:(Nx_half - 3), :]) / dx

    dExdy[p, :, 0] = (- 0.5 * Ex[p, :, 0] + 0.5 * Ex[p, :, 1]) / dx / 0.5
    dExdy[p, :, 1] = (- 0.25 * Ex[p, :, 0] + 0.25 * Ex[p, :, 1]) / dx / 0.25
    dExdy[p, :, 2] = (- 0.25 * Ex[p, :, 0] - 0.75 * Ex[p, :, 1] + Ex[p, :, 2]) / dx / (5.0 / 4.0)
    dExdy[p, :, Ny_half - 3] = (- Ex[p, :, -3] + 0.75 * Ex[p, :, -2] + 0.25 * Ex[p, :, -1]) / dy / (5.0 / 4.0)
    dExdy[p, :, Ny_half - 2] = (- 0.25 * Ex[p, :, -2] + 0.25 * Ex[p, :, -1]) / dy / 0.25
    dExdy[p, :, Ny_half - 1] = (- 0.5 * Ex[p, :, -2] + 0.5 * Ex[p, :, -1]) / dy / 0.5
    dExdy[p, :, 3:(Ny_half - 3)] = (Ex[p, :, 3:(Ny_half - 3)] - N.roll(Ex, 1, axis = 2)[p, :, 3:(Ny_half - 3)]) / dy

def compute_divE(p, fieldx, fieldy):

    divE0[p, :, :] = divE1[p, :, :]    
    divE1[p, :, :] = 0.0
    
    divE1[p, 0, :] += (2.0 * (delta - 1.0) * fieldx[p, 0, :] + (2.0 - 3.0 * delta) * fieldx[p, 1, :] + delta * fieldx[p, 2, :]) / dx
    divE1[p, 1, :] += (2.0 * (gamma - 1.0) * fieldx[p, 0, :] + (2.0 - 3.0 * gamma) * fieldx[p, 1, :] + gamma * fieldx[p, 2, :]) / dx

    divE1[p, Nx_int - 1, :] += (- delta * fieldx[p, -3, :] - (2.0 - 3.0 * delta) * fieldx[p, -2, :] - 2.0 * (delta - 1.0) * fieldx[p, -1, :]) / dx  
    divE1[p, Nx_int - 2, :] += (- gamma * fieldx[p, -3, :] - (2.0 - 3.0 * gamma) * fieldx[p, -2, :] - 2.0 * (gamma - 1.0) * fieldx[p, -1, :]) / dx

    divE1[p, 2:(Nx_int - 2), :] += (N.roll(fieldx, -1, axis = 1)[p, 2:(Nx_int - 2), :] - fieldx[p, 2:(Nx_int - 2), :]) / dx

    divE1[p, :, 0] += (2.0 * (delta - 1.0) * fieldy[p, :, 0] + (2.0 - 3.0 * delta) * fieldy[p, :, 1] + delta * fieldy[p, :, 2]) / dy
    divE1[p, :, 1] += (2.0 * (gamma - 1.0) * fieldy[p, :, 0] + (2.0 - 3.0 * gamma) * fieldy[p, :, 1] + gamma * fieldy[p, :, 2]) / dy

    divE1[p, :, Ny_int - 1] += (- delta * fieldy[p, :, -3] - (2.0 - 3.0 * delta) * fieldy[p, :, -2] - 2.0 * (delta - 1.0) * fieldy[p, :, -1]) / dy 
    divE1[p, :, Ny_int - 2] += (- gamma * fieldy[p, :, -3] - (2.0 - 3.0 * gamma) * fieldy[p, :, -2] - 2.0 * (gamma - 1.0) * fieldy[p, :, -1]) / dy

    divE1[p, :, 2:(Ny_int - 2)] += (N.roll(fieldy, -1, axis = 2)[p, :, 2:(Ny_int - 2)] - fieldy[p, :, 2:(Ny_int - 2)]) / dy

def compute_divcharge(p, tab):
    
    tab[p, :, :] = 0.0
    
    tab[p, 0, :] += (2.0 * (delta - 1.0) * Jx[p, 0, :] + (2.0 - 3.0 * delta) * Jx[p, 1, :] + delta * Jx[p, 2, :]) / dx
    tab[p, 1, :] += (2.0 * (gamma - 1.0) * Jx[p, 0, :] + (2.0 - 3.0 * gamma) * Jx[p, 1, :] + gamma * Jx[p, 2, :]) / dx

    tab[p, Nx_int - 1, :] += (- delta * Jx[p, -3, :] - (2.0 - 3.0 * delta) * Jx[p, -2, :] - 2.0 * (delta - 1.0) * Jx[p, -1, :]) / dx
    tab[p, Nx_int - 2, :] += (- gamma * Jx[p, -3, :] - (2.0 - 3.0 * gamma) * Jx[p, -2, :] - 2.0 * (gamma - 1.0) * Jx[p, -1, :]) / dx

    tab[p, 2:(Nx_int - 2), :] += (N.roll(Jx, -1, axis = 1)[p, 2:(Nx_int - 2), :] - Jx[p, 2:(Nx_int - 2), :]) / dx

    tab[p, :, 0] += (2.0 * (delta - 1.0) * Jy[p, :, 0] + (2.0 - 3.0 * delta) * Jy[p, :, 1] + delta * Jy[p, :, 2]) / dy
    tab[p, :, 1] += (2.0 * (gamma - 1.0) * Jy[p, :, 0] + (2.0 - 3.0 * gamma) * Jy[p, :, 1] + gamma * Jy[p, :, 2]) / dy

    tab[p, :, Ny_int - 1] += (- delta * Jy[p, :, -3] - (2.0 - 3.0 * delta) * Jy[p, :, -2] - 2.0 * (delta - 1.0) * Jy[p, :, -1]) / dy
    tab[p, :, Ny_int - 2] += (- gamma * Jy[p, :, -3] - (2.0 - 3.0 * gamma) * Jy[p, :, -2] - 2.0 * (gamma - 1.0) * Jy[p, :, -1]) / dy

    tab[p, :, 2:(Ny_int - 2)] += (N.roll(Jy, -1, axis = 2)[p, :, 2:(Ny_int - 2)] - Jy[p, :, 2:(Ny_int - 2)]) / dy

    drho = (rho1f[p, :, :] - rho0f[p, :, :]) / dt

    divcharge[p, :, :] = drho[p, : ,:] + tab[p, :, :]
    return

# Bz antenna
# Jz = N.zeros_like(Bz)
# Jz[1, :, :] = 0.0 * N.exp(- ((xBz_grid-0.5)**2 + (yBz_grid-0.5)**2) / 0.02**2)

xout = 0.9
vout = 0.02
Rout = 0.005
ampout = - 0.5

def shape(x0, y0):
    # return N.heaviside(Rout - N.sqrt(x0**2 + y0**2), 0.0)
    return N.exp(- (x0**2 + y0**2) / Rout**2)
        
Jxout = N.zeros((n_patches, Nx_half, Ny_int, Nt))
# for it in range(Nt):
#     Jxout[0, :, :, it] = ampout * (shape(xEx_grid[:, :] - xout - vout * it * dt, yEx_grid[:, :] - 0.51) + shape(xEx_grid[:, :] - xout + vout * it * dt, yEx_grid[:, :] - 0.51))
#     Jxout[1, :, :, it] = ampout * (shape(xEx_grid[:, :] - xout - vout * it * dt + x_max - x_min, yEx_grid[:, :] - 0.51) + shape(xEx_grid[:, :] - xout + vout * it * dt + x_max - x_min, yEx_grid[:, :] - 0.51))

def push_B(p, it, dtin):
        Bz[p, :, :] += dtin * (dExdy[p, :, :] - dEydx[p, :, :]) # + dtin * Jz[p, :, :] * N.sin(20.0 * it * dt) * N.exp(- it * dt / 1.0)

def push_E(p, it, dtin):
        # Ex[p, :, :] += dtin * (dBzdy[p, :, :] - 4.0 * N.pi * (Jx[p, :, :] + Jxout[p, :, :, it]))
        # Ex[p, :, :] += dtin * (dBzdy[p, :, :] - 4.0 * N.pi * (Jx[p, :, :] + Jxout[p, :, :] * N.sin(20.0 * it * dt) * N.exp(- it * dt / 1.0)))
        Ex[p, :, :] += dtin * (dBzdy[p, :, :] - 4.0 * N.pi * Jx[p, :, :])

        Ey[p, :, :] += dtin * (- dBzdx[p, :, :] - 4.0 * N.pi * Jy[p, :, :])

########
# Particles
########

np = 2
xp  = N.zeros((Nt + 1, np))
yp  = N.zeros((Nt + 1, np))
uxp = N.zeros((Nt + 1, np))
uyp = N.zeros((Nt + 1, np))
wp  = N.zeros((Nt + 1, np)) # charge x weight (can be negative)
tag = N.zeros((Nt + 1, np), dtype='int') # Patch in which the partice is located

ux0 = 0.8
uy0 = 0.2
x0 = 1.0 - 30.5 * dx
y0 = 0.5

def initialize_part():

    xp[0, :] = x0
    # xp[0, 0] = x0
    # xp[0, 0] = 0.0
    tag[:, 0] = 0
    tag[:, 1] = 0

    for ip in range(0, np, 2):
        r = y_min + N.random.rand() * (y_max - y_min)
        yp[0, ip] = y0
        yp[0, ip + 1] = y0
        uxp[0, ip] = - ux0
        uxp[0, ip + 1] = - ux0

        uyp[0, ip] = uy0
        uyp[0, ip + 1] = -uy0

        # wp[:, ip] = 0.0
        # wp[:, ip + 1] = 0.0
        wp[:, ip] = - 1.0
        wp[:, ip + 1] = 1.0
        
# Impose velocity with tanh profile to avoid discontinuities
def impose_velocity_part(it):
    uxp[it, 0] =   ux0 * 0.5 * (1 - N.tanh((10 - it)/50))
    uxp[it, 1] = - ux0 * 0.5 * (1 - N.tanh((10 - it)/50))
    uyp[it, 0] =   uy0 * 0.5 * (1 - N.tanh((10 - it)/50))
    uyp[it, 1] = - uy0 * 0.5 * (1 - N.tanh((10 - it)/50))
    
    # if (it>480):
    #     uxp[it, 0] = 0.0
    
# Returns index of CELL
def i_from_pos(x0, y0):
    i0 = int(((x0 - x_min) / dx) // 1)
    j0 = int(((y0 - y_min) / dy) // 1)
    return i0, j0

# Particle boundary conditions
def BC_part(it, ip):
    
    # if (yp[it + 1, ip] > y_max)or(yp[it + 1, ip] < y_min):
    #     wp[(it+1):, ip] = 0.0
    
    if ((xp[it + 1, ip] < x_min)and(tag[it + 1, ip]==0)):
        wp[(it+1):, ip] = 0.0
    
    if ((xp[it + 1, ip] > x_max)and(tag[it + 1, ip]==1)):
        wp[(it+1):, ip] = 0.0
        
    if ((xp[it + 1, ip] > x_max)and(tag[it + 1, ip]==0)):
        tag[(it+1):, ip] = 1
        xp[it + 1, ip] -= (x_max - x_min) 

    if ((xp[it + 1, ip] < x_min)and(tag[it + 1, ip]==1)):
        tag[(it+1):, ip] = 0
        xp[it + 1, ip] += (x_max - x_min) 

    if (yp[it + 1, ip] > y_max):
        yp[it + 1, ip] -= y_max
    elif (yp[it + 1, ip] < 0.0):
        yp[it + 1, ip] += y_max
        
    return

def interpolate_field(x0, y0, p):
    
    Bz0 = (sci.RectBivariateSpline(x_half, y_half, Bz[p, :, :]))(x0, y0)
    Ex0 = (sci.RectBivariateSpline(x_half, y_int,  Ex[p, :, :]))(x0, y0)
    Ey0 = (sci.RectBivariateSpline(x_int,  y_half, Ey[p, :, :]))(x0, y0)
    
    return Ex0, Ey0, Bz0

def push_x(it, ip):
    gammap = N.sqrt(1.0 + uxp[it, ip]**2 + uyp[it, ip]**2)
    xp[it+1, ip] = xp[it, ip] + uxp[it, ip] * dt / gammap
    yp[it+1, ip] = yp[it, ip] + uyp[it, ip] * dt / gammap

# it has x_n, u_n+1/2
# it - 1 has x^n-1, u_n-1/2
def push_u(it, ip):
    w0 = wp[it, ip]
    
    if (N.abs(w0) > 0):
        x0, y0 = xp[it, ip], yp[it, ip] 
        ux0, uy0 = uxp[it, ip], uyp[it, ip] 

        # Ex0, Ey0, Bz0 = interpolate_field(x0, y0, p)
        Ex0, Ey0, Bz0 = 0.0, 0.0, 0.0
        
        Ex0 *= 0.5 * q * dt * w0
        Ey0 *= 0.5 * q * dt * w0
        Bz0 *= 0.5 * q * dt * w0

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
        
        uxp[it + 1, ip] = uxf
        uyp[it + 1, ip] = uyf
    else:
        return

########
# Current deposition
########

# Computes intermediate point in zig-zag algorithm
def compute_intermediate(it, ip):

    x1, y1 = xp[it, ip], yp[it, ip] 
    x2, y2 = xp[it + 1, ip], yp[it + 1, ip] 

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
    rho0[:, :, :] = rho1[:, :, :]
    rho1[:, :, :] = 0.0
    
    for ip in range(np):
        w0 = wp[it + 1, ip]
        if (N.abs(w0) > 0):
            deposit_particle(it, ip)
    
    return
                    
S0 = dx * dy

# ### NEW STENCIL
# def subcell(m_tag0, m_i1, m_j1, m_Fx1, m_Fy1, m_Wx1, m_Wy1):

#         # Shifted by 1 because of extra points at the edge
#         ix1 = m_i1 + 1
#         jy1 = m_j1 + 1
        
#         tag1 = int(1-m_tag0)

#         deltax1 = 1.0
#         deltay1 = 1.0

#         # Particle leaves simulation box: only first half deposit
#         if (test_out(m_i1)==True):
#             deltax1 = 0.0
#             deltay1 = 0.0
    
#         # Bulk current deposition
#         Jx[m_tag0, ix1, m_j1] += deltax1 * m_Fx1 * (1.0 - m_Wy1) / S0
#         Jx[m_tag0, ix1, m_j1 + 1] += deltax1 * m_Fx1 * m_Wy1 / S0

#         Jy[m_tag0, m_i1, jy1] += deltay1 * m_Fy1 * (1.0 - m_Wx1) / S0
#         Jy[m_tag0, m_i1 + 1, jy1] += deltay1 * m_Fy1 * m_Wx1 / S0

#         # Current on left edge
#         if (m_i1 == 0):
#             Jx[m_tag0, 0, m_j1]     += 0.5 * m_Fx1 * (1.0 - m_Wy1) / S0
#             Jx[m_tag0, 0, m_j1 + 1] += 0.5 * m_Fx1 * m_Wy1 / S0
            
#         # Current on right edge            
#         if (m_i1 == (Nx - 1)):
#             Jx[m_tag0, -1, m_j1]     += 0.5 * m_Fx1 * (1.0 - m_Wy1) / S0
#             Jx[m_tag0, -1, m_j1 + 1] += 0.5 * m_Fx1 * m_Wy1 / S0
            
#         # Current from charge that's not into patch 1 yet
#         if (m_tag0==0)and(m_i1==(Nx-1)):
#             Jx[tag1, 0, m_j1] += 0.5 * m_Fx1 * (1.0 - m_Wy1) / S0
#             Jx[tag1, 0, m_j1 + 1] += 0.5 * m_Fx1 * m_Wy1 / S0
#             Jy[tag1, 0, jy1] += deltay1 * m_Fy1 * m_Wx1 / S0

#         # Current from charge that's not into patch 0 yet
#         if (m_tag0==1)and(m_i1==0):
#             Jx[tag1, -1, m_j1] += 0.5 * m_Fx1 * (1.0 - m_Wy1) / S0
#             Jx[tag1, -1, m_j1 + 1] += 0.5 * m_Fx1 * m_Wy1 / S0
#             Jy[tag1, -1, jy1] += deltay1 * m_Fy1 * (1.0 - m_Wx1) / S0

# Deposit current on a subcell
def subcell(m_tag0, m_i1, m_j1, m_Fx1, m_Fy1, m_Wx1, m_Wy1):

    # Shifted by 1 because of extra points at the edge
    ix1 = m_i1 + 1
    jy1 = m_j1 + 1
    
    tag1 = int(1-m_tag0)
    
    deltax1 = 1.0
    deltay1 = 1.0
            
    # Interior, no deposition on edge cell
    if (test_inside(m_i1)==True):
        pass
    # Particle starts in edge and moves to interior
    elif (test_edge(m_i1)==True):
        deltax1 = (delta + gamma - 2.0) / (delta - gamma)

    # Interior, no deposition on edge cell
    if (test_inside(m_j1)==True):
        pass
    # Particle starts in edge and moves to interior
    elif (test_edge(m_j1)==True):
        deltay1 = (delta + gamma - 2.0) / (delta - gamma)

    # Bulk current deposition
    Jx[m_tag0, ix1, m_j1] += deltax1 * m_Fx1 * (1.0 - m_Wy1) / S0
    Jx[m_tag0, ix1, m_j1 + 1] += deltax1 * m_Fx1 * m_Wy1 / S0
    
    Jy[m_tag0, m_i1, jy1] += deltay1 * m_Fy1 * (1.0 - m_Wx1) / S0
    Jy[m_tag0, m_i1 + 1, jy1] += deltay1 * m_Fy1 * m_Wx1 / S0

##### Vertical interfaces

    # Current on left edge
    if (m_i1 == 0):
        Jx[m_tag0, 0, m_j1]     += (3.0 * delta + 3.0 * gamma - 4.0) / 2.0 / (delta - gamma) * m_Fx1 * (1.0 - m_Wy1) / S0
        Jx[m_tag0, 0, m_j1 + 1] += (3.0 * delta + 3.0 * gamma - 4.0) / 2.0 / (delta - gamma) * m_Fx1 * m_Wy1 / S0
        
    # Current on mid-left cell
    if (m_i1 == 1):

        Jx[m_tag0, 0, m_j1]     += (delta*(gamma-delta)+2*gamma-2-3*delta*gamma+3*delta) / 2 / (1-delta) / (gamma-delta) * m_Fx1 * (1.0 - m_Wy1) / S0
        Jx[m_tag0, 0, m_j1 + 1] += (delta*(gamma-delta)+2*gamma-2-3*delta*gamma+3*delta) / 2 / (1-delta) / (gamma-delta) * m_Fx1 * m_Wy1 / S0

        Jx[m_tag0, 1, m_j1]     += (gamma - 1.0) / (gamma - delta) * m_Fx1 * (1.0 - m_Wy1) / S0
        Jx[m_tag0, 1, m_j1 + 1] += (gamma - 1.0) / (gamma - delta) * m_Fx1 * m_Wy1 / S0
        
    # Current on right edge            
    if (m_i1 == (Nx - 1)):

        Jx[m_tag0, -1, m_j1]     += (3.0 * delta + 3.0 * gamma - 4.0) / 2.0 / (delta - gamma) * m_Fx1 * (1.0 - m_Wy1) / S0
        Jx[m_tag0, -1, m_j1 + 1] += (3.0 * delta + 3.0 * gamma - 4.0) / 2.0 / (delta - gamma) * m_Fx1 * m_Wy1 / S0
        
    # Current on mid-right cell
    if (m_i1 == (Nx - 2)):
        Jx[m_tag0, -1, m_j1]     += (delta*(gamma-delta)+2*gamma-2-3*delta*gamma+3*delta) / 2 / (1-delta) / (gamma-delta) * m_Fx1 * (1.0 - m_Wy1) / S0
        Jx[m_tag0, -1, m_j1 + 1] += (delta*(gamma-delta)+2*gamma-2-3*delta*gamma+3*delta) / 2 / (1-delta) / (gamma-delta) * m_Fx1 * m_Wy1 / S0

        Jx[m_tag0, -2, m_j1]     += (gamma - 1.0) / (gamma - delta) * m_Fx1 * (1.0 - m_Wy1) / S0
        Jx[m_tag0, -2, m_j1 + 1] += (gamma - 1.0) / (gamma - delta) * m_Fx1 * m_Wy1 / S0
        
    # Current from charge that's not into patch 1 yet
    if ((m_tag0==0)and(test_edge(m_i1)==True)):
        Jx[tag1, 0, m_j1]     += (3.0 * gamma - 2.0) / (2.0 * gamma - 2.0 * delta) * m_Fx1 * (1.0 - m_Wy1) / S0
        Jx[tag1, 0, m_j1 + 1] += (3.0 * gamma - 2.0) / (2.0 * gamma - 2.0 * delta) * m_Fx1 * m_Wy1 / S0
        
        Jx[tag1, 1, m_j1]     += (gamma - 1.0) / (gamma - delta) * m_Fx1 * (1.0 - m_Wy1) / S0
        Jx[tag1, 1, m_j1 + 1] += (gamma - 1.0) / (gamma - delta) * m_Fx1 * m_Wy1 / S0

        Jy[tag1, 0, jy1] += m_Fy1 * m_Wx1 / S0
        
    # Current from charge that's not into patch 0 yet
    if ((m_tag0==1)and(test_edge(m_i1)==True)):
        Jx[tag1, -1, m_j1]     += (3.0 * gamma - 2.0) / (2.0 * gamma - 2.0 * delta) * m_Fx1 * (1.0 - m_Wy1) / S0
        Jx[tag1, -1, m_j1 + 1] += (3.0 * gamma - 2.0) / (2.0 * gamma - 2.0 * delta) * m_Fx1 * m_Wy1 / S0
        
        Jx[tag1, -2, m_j1]     += (gamma - 1.0) / (gamma - delta) * m_Fx1 * (1.0 - m_Wy1) / S0
        Jx[tag1, -2, m_j1 + 1] += (gamma - 1.0) / (gamma - delta) * m_Fx1 * m_Wy1 / S0
        
        Jy[tag1, -1, jy1] += m_Fy1 * (1.0 - m_Wx1) / S0

##### Horizontal interfaces

    # Current on bottom edge
    if (m_j1 == 0):
        Jy[m_tag0, m_i1, 0]     += 0.5 * m_Fy1 * (1.0 - m_Wx1) / S0
        Jy[m_tag0, m_i1 + 1, 0] += 0.5 * m_Fy1 * m_Wx1 / S0

        Jy[m_tag0, m_i1, -1]     += 0.5 * m_Fy1 * (1.0 - m_Wx1) / S0
        Jy[m_tag0, m_i1 + 1, -1] += 0.5 * m_Fy1 * m_Wx1 / S0
        
        Jx[m_tag0, ix1, -1] += m_Fx1 * (1.0 - m_Wy1) / S0

    # Current on mid-bottom cell
    if (m_j1 == 1):
        Jy[m_tag0, m_i1, 1]     += - m_Fy1 * (1.0 - m_Wx1) / S0
        Jy[m_tag0, m_i1 + 1, 1] += - m_Fy1 * m_Wx1 / S0
        
    # Current on top edge            
    if (m_j1 == (Ny - 1)):
        Jy[m_tag0, m_i1, -1]     += 0.5 * m_Fy1 * (1.0 - m_Wx1) / S0
        Jy[m_tag0, m_i1 + 1, -1] += 0.5 * m_Fy1 * m_Wx1 / S0

        Jy[m_tag0, m_i1, 0]     += 0.5 * m_Fy1 * (1.0 - m_Wx1) / S0
        Jy[m_tag0, m_i1 + 1, 0] += 0.5 * m_Fy1 * m_Wx1 / S0

        Jx[m_tag0, ix1, 0] += m_Fx1 * m_Wy1 / S0

    # Current on mid-top cell
    if (m_j1 == (Ny - 2)):
        Jy[m_tag0, m_i1, -2]     += - m_Fy1 * (1.0 - m_Wx1) / S0
        Jy[m_tag0, m_i1 + 1, -2] += - m_Fy1 * m_Wx1 / S0

    return

# # Current deposition of a single particle
# # Note : only deals with x_min and x_max interfaces, no corners or top/bottom interfaces!!

def deposit_particle(it, ip):
    
    x1, y1 = xp[it, ip], yp[it, ip] # Initial coordinates
    x2, y2 = xp[it + 1, ip], yp[it + 1, ip] # Final coordinates

    # Indices of initial/final cell (from 0 to N-1)
    i1, j1 = i_from_pos(x1, y1)
    i2, j2 = i_from_pos(x2, y2)

    xr, yr = compute_intermediate(it, ip)

    w0   = wp[it, ip]
    tag0 = tag[it, ip]
    tag1 = tag[it + 1, ip]
    
    # if (x2 > 0.95):
    #     wp[(it+1):, ip]=0.0
    
    # Particle does not leave patch
    if (tag0==tag1):

        # Umeda notation

        Fx1 = q * (xr - x1) / dt * w0
        Fx2 = q * (x2 - xr) / dt * w0

        Fy1 = q * (yr - y1) / dt * w0
        Fy2 = q * (y2 - yr) / dt * w0

        Wx1 = 0.5 * (x1 + xr) / dx - i1
        Wy1 = 0.5 * (y1 + yr) / dy - j1

        Wx2 = 0.5 * (x2 + xr) / dx - i2
        Wy2 = 0.5 * (y2 + yr) / dy - j2

        # First part of trajectory
        subcell(tag0, i1, j1, Fx1, Fy1, Wx1, Wy1) 
          
        # Second part of trajectory    
        subcell(tag1, i2, j2, Fx2, Fy2, Wx2, Wy2) 

        x2r = (x2 - x_min) / dx - i2
        y2r = (y2 - y_min) / dy - j2

        # Charge deposited in current patch
        if (test_out(i2)==False):
            rho1[tag0, i2, j2] += q * w0 * (1.0 - x2r) * (1.0 - y2r) / S0
            rho1[tag0, i2 + 1, j2] += q * w0 * x2r * (1.0 - y2r) / S0
            rho1[tag0, i2, j2 + 1] += q * w0 * (1.0 - x2r) * y2r / S0
            rho1[tag0, i2 + 1, j2 + 1] += q * w0 * x2r * y2r / S0

        # Charge deposited in other patch
        if (tag0==0)and(i2==(Nx-1)):
            rho1[1, 0, j2] += q * w0 * x2r * (1.0 - y2r) / S0
            rho1[1, 0, j2 + 1] += q * w0 * x2r * y2r / S0
        if (tag0==1)and(i2==0):
            rho1[0, -1, j2] += q * w0 * (1.0 - x2r) * (1.0 - y2r) / S0
            rho1[0, -1, j2 + 1] += q * w0 * (1.0 - x2r) * y2r / S0

    # Particle leaves patch
    elif (tag1!=tag0):

        if (tag0==0):
            # When particle leaves patch 0 to 1, x2 is already written in new patch coordinates, so must specify xr in both cases
            xr = 1.0
            Fx1 = q * (xr - x1) / dt * w0
            Wx1 = 0.5 * (x1 + xr) / dx - i1
            Fy1 = q * (yr - y1) / dt * w0
            Wy1 = 0.5 * (y1 + yr) / dy - j1
            
            # wp[(it+1):, ip] = 0.0
            
            xr = 0.0     
            Fx2 = q * (x2 - xr) / dt * w0
            Wx2 = 0.5 * (x2 + xr) / dx - i2
            Fy2 = q * (y2 - yr) / dt * w0
            Wy2 = 0.5 * (y2 + yr) / dy - j2
        else:
            # When particle leaves patch 0 to 1, x2 is already written in new patch coordinates, so must specify xr in both cases
            xr = 0.0
            Fx1 = q * (xr - x1) / dt * w0
            Wx1 = 0.5 * (x1 + xr) / dx - i1
            Fy1 = q * (yr - y1) / dt * w0
            Wy1 = 0.5 * (y1 + yr) / dy - j1
            
            # wp[(it+1):, ip] = 0.0
            
            xr = 1.0     
            Fx2 = q * (x2 - xr) / dt * w0
            Wx2 = 0.5 * (x2 + xr) / dx - i2
            Fy2 = q * (y2 - yr) / dt * w0
            Wy2 = 0.5 * (y2 + yr) / dy - j2
        
        x2r = (x2 - x_min) / dx - i2
        y2r = (y2 - y_min) / dy - j2
        
        rho1[tag1, i2, j2] += q * w0 * (1.0 - x2r) * (1.0 - y2r) / S0
        rho1[tag1, i2, j2 + 1] += q * w0 * (1.0 - x2r) * y2r / S0
        rho1[tag1, i2 + 1, j2] += q * w0 * x2r * (1.0 - y2r) / S0
        rho1[tag1, i2 + 1, j2 + 1] += q * w0 * x2r * y2r / S0

        if (tag1==0)and(i2==(Nx-1)):
            rho1[1, 0, j2] += q * w0 * x2r * (1.0 - y2r) / S0
            rho1[1, 0, j2 + 1] += q * w0 * x2r * y2r / S0
        if (tag1==1)and(i2==0):
            rho1[0, -1, j2] += q * w0 * (1.0 - x2r) * (1.0 - y2r) / S0
            rho1[0, -1, j2 + 1] += q * w0 * (1.0 - x2r) * y2r / S0
        
        subcell(tag0, i1, j1, Fx1, Fy1, Wx1, Wy1) 
        subcell(tag1, i2, j2, Fx2, Fy2, Wx2, Wy2) 

    # If absorbing patch 1    
    # Jx[1, :, :] = 0.0
    # Jy[1, :, :] = 0.0
    # rho1[1, :, :] = 0.0
    
    return

#############
# NEW STENCIL
#############

# def deposit_particle(it, ip):
    
#     x1, y1 = xp[it, ip], yp[it, ip] # Initial coordinates
#     x2, y2 = xp[it + 1, ip], yp[it + 1, ip] # Final coordinates

#     # Indices of initial/final cell (from 0 to N-1)
#     i1, j1 = i_from_pos(x1, y1)
#     i2, j2 = i_from_pos(x2, y2)

#     xr, yr = compute_intermediate(it, ip)

#     w0   = wp[it, ip]
#     tag0 = tag[it, ip]
#     tag1 = tag[it + 1, ip]
        
#     # Coefficients of deposited Jx
#     deltax1 = 1.0
#     deltax2 = 1.0
        
#     deltay1 = 1.0
#     deltay2 = 1.0
    
#     # Particle does not leave patch

#     if (tag0==tag1):

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
#             Jy[1, 0, jy1] += deltay1 * Fy1 * Wx1 / S0

#         # Current from charge that's not into patch 0 yet
#         if (tag0==1)and(i1==0):
#             Jx[0, -1, j1] += 0.5 * Fx1 * (1.0 - Wy1) / S0
#             Jx[0, -1, j1 + 1] += 0.5 * Fx1 * Wy1 / S0
#             Jy[0, -1, jy1] += deltay1 * Fy1 * (1.0 - Wx1) / S0
            
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
#             Jy[1, 0, jy2] += deltay2 * Fy2 * Wx2 / S0

#         if (tag0==1)and(i2==0):
#             Jx[0, -1, j2] += 0.5 * Fx2 * (1.0 - Wy2) / S0
#             Jx[0, -1, j2 + 1] += 0.5 * Fx2 * Wy2 / S0
#             Jy[0, -1, jy2] += deltay2 * Fy2 * (1.0 - Wx2) / S0

#         x2r = (x2 - x_min) / dx - i2
#         y2r = (y2 - y_min) / dy - j2

#         # Charge deposited in current patch
#         if (test_out(i2)==False):
#             rho1[tag0, i2, j2] += q * w0 * (1.0 - x2r) * (1.0 - y2r) / S0
#             rho1[tag0, i2 + 1, j2] += q * w0 * x2r * (1.0 - y2r) / S0
#             rho1[tag0, i2, j2 + 1] += q * w0 * (1.0 - x2r) * y2r / S0
#             rho1[tag0, i2 + 1, j2 + 1] += q * w0 * x2r * y2r / S0

#         # Charge deposited in other patch
#         if (tag0==0)and(i2==(Nx-1)):
#             rho1[1, 0, j2] += q * w0 * x2r * (1.0 - y2r) / S0
#             rho1[1, 0, j2 + 1] += q * w0 * x2r * y2r / S0
#         if (tag0==1)and(i2==0):
#             rho1[0, -1, j2] += q * w0 * (1.0 - x2r) * (1.0 - y2r) / S0
#             rho1[0, -1, j2 + 1] += q * w0 * (1.0 - x2r) * y2r / S0

#     # Particle leaves patch
#     elif (tag1!=tag0):
            
#         # There is better way of doing this than having this dichotomy. Will do later
#         if (tag0 == 0):

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

#             x2r = (x2 - x_min) / dx - i2
#             y2r = (y2 - y_min) / dy - j2

#             rho1[tag1, i2, j2] += q * w0 * (1.0 - x2r) * (1.0 - y2r) / S0
#             rho1[tag1, i2, j2 + 1] += q * w0 * (1.0 - x2r) * y2r / S0
#             rho1[tag1, i2 + 1, j2] += q * w0 * x2r * (1.0 - y2r) / S0
#             rho1[tag1, i2 + 1, j2 + 1] += q * w0 * x2r * y2r / S0

#             #NEW
#             rho1[tag0, -1, j2] += q * w0 * (1.0 - x2r) * (1.0 - y2r) / S0
#             rho1[tag0, -1, j2 + 1] += q * w0 * (1.0 - x2r) * y2r / S0

#             # Current deposited during first part of trajectory: particle is in patch 0

#             Jy[tag0, i1, jy1] += deltay1 * Fy1 * (1.0 - Wx1) / S0
#             Jy[tag0, i1 + 1, jy1] += deltay1 * Fy1 * Wx1 / S0

#             ###NEW
#             Jy[tag1, 0, jy1] += deltay1 * Fy1 * Wx1 / S0
            
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

#             ###NEW
#             Jy[0, -1, jy2] += deltay2 * Fy2 * (1.0 - Wx2) / S0
            
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
#         elif (tag0 == 1):

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

#             x2r = (x2 - x_min) / dx - i2
#             y2r = (y2 - y_min) / dy - j2

#             rho1[tag1, i2, j2] += q * w0 * (1.0 - x2r) * (1.0 - y2r) / S0
#             rho1[tag1, i2, j2 + 1] += q * w0 * (1.0 - x2r) * y2r / S0
#             rho1[tag1, i2 + 1, j2] += q * w0 * x2r * (1.0 - y2r) / S0
#             rho1[tag1, i2 + 1, j2 + 1] += q * w0 * x2r * y2r / S0

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
rhobuff = N.zeros_like(rho0)
rhoint = N.zeros_like(rho0)

c00 = 0.25
c10 = 0.125
c01 = 0.50
c11 = 0.50
c12 = 0.00
c04 = 0.0
c50 = 0.25
c40 = 0.125

x00 = 0.75
x01 = 0.125
x02 = 0.125
x03 = 0.0
x52 = 0.125
x51 = 0.125
x50 = -0.25

x10 = 0.375
x11 = 0.1875
x12 = 0.6875
x13 = -0.25
x42 = -0.0
x41 = -0.0
x40 = 0.0

x20 = 0.125
x21 = 0.0625
x22 = 0.5625
x23 = 0.25

ym1 = 0.25
y00 = 0.50
yp1 = 0.25

def filter_current(iter):

    Jbuffx[:, :, :] = Jx[:, :, :]
    Jbuffy[:, :, :] = Jy[:, :, :]
    rhobuff[:, :, :] = rho1[:, :, :]
    rho0f[:, :, :] = rho1f[:, :, :]
    
    # Pretend information from other patch is in a ghost cell

    for i in range(iter):
        
        Jintx[:, :, :] = Jbuffx[:, :, :]

        # Jx in bulk, regular 1-2-1 stencil
        Jbuffx[:, 3:(Nx_half-3), 1:(Ny_int-1)] = 0.50 * y00 * Jintx[:, 3:(Nx_half-3), 1:(Ny_int-1)] \
                                               + 0.25 * y00 * N.roll(Jintx, 1, axis = 1)[:, 3:(Nx_half-3), 1:(Ny_int-1)] \
                                               + 0.25 * y00 * N.roll(Jintx, -1, axis = 1)[:, 3:(Nx_half-3), 1:(Ny_int-1)] \
                                               + 0.50 * ym1 * N.roll(Jintx, 1, axis = 2)[:, 3:(Nx_half-3), 1:(Ny_int-1)] \
                                               + 0.50 * yp1 * N.roll(Jintx, -1, axis = 2)[:, 3:(Nx_half-3), 1:(Ny_int-1)] \
                                               + 0.25 * ym1 * N.roll(N.roll(Jintx, 1, axis = 1), 1, axis = 2)[:, 3:(Nx_half-3), 1:(Ny_int-1)] \
                                               + 0.25 * yp1 * N.roll(N.roll(Jintx, -1, axis = 1), -1, axis = 2)[:, 3:(Nx_half-3), 1:(Ny_int-1)] \
                                               + 0.25 * yp1 * N.roll(N.roll(Jintx, 1, axis = 1), -1, axis = 2)[:, 3:(Nx_half-3), 1:(Ny_int-1)] \
                                               + 0.25 * ym1 * N.roll(N.roll(Jintx, -1, axis = 1), 1, axis = 2)[:, 3:(Nx_half-3), 1:(Ny_int-1)]

        Jbuffx[0, -1, 1:(Ny_int-1)] = x00 * y00 * Jintx[0, -1, 1:(Ny_int-1)] \
                                    + x01 * y00 * Jintx[0, -2, 1:(Ny_int-1)] \
                                    + x02 * y00 * Jintx[0, -3, 1:(Ny_int-1)] \
                                    + x03 * y00 * Jintx[0, -4, 1:(Ny_int-1)] \
                                    + x50 * y00 * Jintx[1, 0, 1:(Ny_int-1)] \
                                    + x51 * y00 * Jintx[1, 1, 1:(Ny_int-1)] \
                                    + x52 * y00 * Jintx[1, 2, 1:(Ny_int-1)] \
                                    + x00 * ym1 * N.roll(Jintx, 1, axis = 2)[0, -1, 1:(Ny_int-1)] \
                                    + x01 * ym1 * N.roll(Jintx, 1, axis = 2)[0, -2, 1:(Ny_int-1)] \
                                    + x02 * ym1 * N.roll(Jintx, 1, axis = 2)[0, -3, 1:(Ny_int-1)] \
                                    + x03 * ym1 * N.roll(Jintx, 1, axis = 2)[0, -4, 1:(Ny_int-1)] \
                                    + x50 * ym1 * N.roll(Jintx, 1, axis = 2)[1, 0, 1:(Ny_int-1)] \
                                    + x51 * ym1 * N.roll(Jintx, 1, axis = 2)[1, 1, 1:(Ny_int-1)] \
                                    + x52 * ym1 * N.roll(Jintx, 1, axis = 2)[1, 2, 1:(Ny_int-1)] \
                                    + x00 * yp1 * N.roll(Jintx, -1, axis = 2)[0, -1, 1:(Ny_int-1)] \
                                    + x01 * yp1 * N.roll(Jintx, -1, axis = 2)[0, -2, 1:(Ny_int-1)] \
                                    + x02 * yp1 * N.roll(Jintx, -1, axis = 2)[0, -3, 1:(Ny_int-1)] \
                                    + x03 * yp1 * N.roll(Jintx, -1, axis = 2)[0, -4, 1:(Ny_int-1)] \
                                    + x50 * yp1 * N.roll(Jintx, -1, axis = 2)[1, 0, 1:(Ny_int-1)] \
                                    + x51 * yp1 * N.roll(Jintx, -1, axis = 2)[1, 1, 1:(Ny_int-1)] \
                                    + x52 * yp1 * N.roll(Jintx, -1, axis = 2)[1, 2, 1:(Ny_int-1)] \
                                        
        Jbuffx[0, -2, 1:(Ny_int-1)] = x10 * y00 * Jintx[0, -1, 1:(Ny_int-1)] \
                                    + x11 * y00 * Jintx[0, -2, 1:(Ny_int-1)] \
                                    + x12 * y00 * Jintx[0, -3, 1:(Ny_int-1)] \
                                    + x13 * y00 * Jintx[0, -4, 1:(Ny_int-1)] \
                                    + x40 * y00 * Jintx[1, 0, 1:(Ny_int-1)] \
                                    + x41 * y00 * Jintx[1, 1, 1:(Ny_int-1)] \
                                    + x42 * y00 * Jintx[1, 2, 1:(Ny_int-1)] \
                                    + x10 * ym1 * N.roll(Jintx, 1, axis = 2)[0, -1, 1:(Ny_int-1)] \
                                    + x11 * ym1 * N.roll(Jintx, 1, axis = 2)[0, -2, 1:(Ny_int-1)] \
                                    + x12 * ym1 * N.roll(Jintx, 1, axis = 2)[0, -3, 1:(Ny_int-1)] \
                                    + x13 * ym1 * N.roll(Jintx, 1, axis = 2)[0, -4, 1:(Ny_int-1)] \
                                    + x40 * ym1 * N.roll(Jintx, 1, axis = 2)[1, 0, 1:(Ny_int-1)] \
                                    + x41 * ym1 * N.roll(Jintx, 1, axis = 2)[1, 1, 1:(Ny_int-1)] \
                                    + x42 * ym1 * N.roll(Jintx, 1, axis = 2)[1, 2, 1:(Ny_int-1)] \
                                    + x10 * yp1 * N.roll(Jintx, -1, axis = 2)[0, -1, 1:(Ny_int-1)] \
                                    + x11 * yp1 * N.roll(Jintx, -1, axis = 2)[0, -2, 1:(Ny_int-1)] \
                                    + x12 * yp1 * N.roll(Jintx, -1, axis = 2)[0, -3, 1:(Ny_int-1)] \
                                    + x13 * yp1 * N.roll(Jintx, -1, axis = 2)[0, -4, 1:(Ny_int-1)] \
                                    + x40 * yp1 * N.roll(Jintx, -1, axis = 2)[1, 0, 1:(Ny_int-1)] \
                                    + x41 * yp1 * N.roll(Jintx, -1, axis = 2)[1, 1, 1:(Ny_int-1)] \
                                    + x42 * yp1 * N.roll(Jintx, -1, axis = 2)[1, 2, 1:(Ny_int-1)] \

        Jbuffx[0, -3, 1:(Ny_int-1)] = x20 * y00 * Jintx[0, -1, 1:(Ny_int-1)] \
                                    + x21 * y00 * Jintx[0, -2, 1:(Ny_int-1)] \
                                    + x22 * y00 * Jintx[0, -3, 1:(Ny_int-1)] \
                                    + x23 * y00 * Jintx[0, -4, 1:(Ny_int-1)] \
                                    + x20 * ym1 * N.roll(Jintx, 1, axis = 2)[0, -1, 1:(Ny_int-1)] \
                                    + x21 * ym1 * N.roll(Jintx, 1, axis = 2)[0, -2, 1:(Ny_int-1)] \
                                    + x22 * ym1 * N.roll(Jintx, 1, axis = 2)[0, -3, 1:(Ny_int-1)] \
                                    + x23 * ym1 * N.roll(Jintx, 1, axis = 2)[0, -4, 1:(Ny_int-1)] \
                                    + x20 * yp1 * N.roll(Jintx, -1, axis = 2)[0, -1, 1:(Ny_int-1)] \
                                    + x21 * yp1 * N.roll(Jintx, -1, axis = 2)[0, -2, 1:(Ny_int-1)] \
                                    + x22 * yp1 * N.roll(Jintx, -1, axis = 2)[0, -3, 1:(Ny_int-1)] \
                                    + x23 * yp1 * N.roll(Jintx, -1, axis = 2)[0, -4, 1:(Ny_int-1)] \

        Jbuffx[1, 0, 1:(Ny_int-1)]  = x00 * y00 * Jintx[1, 0, 1:(Ny_int-1)] \
                                    + x01 * y00 * Jintx[1, 1, 1:(Ny_int-1)] \
                                    + x02 * y00 * Jintx[1, 2, 1:(Ny_int-1)] \
                                    + x03 * y00 * Jintx[1, 3, 1:(Ny_int-1)] \
                                    + x50 * y00 * Jintx[0, -1, 1:(Ny_int-1)] \
                                    + x51 * y00 * Jintx[0, -2, 1:(Ny_int-1)] \
                                    + x52 * y00 * Jintx[0, -3, 1:(Ny_int-1)] \
                                    + x00 * ym1 * N.roll(Jintx, 1, axis = 2)[1, 0, 1:(Ny_int-1)] \
                                    + x01 * ym1 * N.roll(Jintx, 1, axis = 2)[1, 1, 1:(Ny_int-1)] \
                                    + x02 * ym1 * N.roll(Jintx, 1, axis = 2)[1, 2, 1:(Ny_int-1)] \
                                    + x03 * ym1 * N.roll(Jintx, 1, axis = 2)[1, 3, 1:(Ny_int-1)] \
                                    + x50 * ym1 * N.roll(Jintx, 1, axis = 2)[0, -1, 1:(Ny_int-1)] \
                                    + x51 * ym1 * N.roll(Jintx, 1, axis = 2)[0, -2, 1:(Ny_int-1)] \
                                    + x52 * ym1 * N.roll(Jintx, 1, axis = 2)[0, -3, 1:(Ny_int-1)] \
                                    + x00 * yp1 * N.roll(Jintx, -1, axis = 2)[1, 0, 1:(Ny_int-1)] \
                                    + x01 * yp1 * N.roll(Jintx, -1, axis = 2)[1, 1, 1:(Ny_int-1)] \
                                    + x02 * yp1 * N.roll(Jintx, -1, axis = 2)[1, 2, 1:(Ny_int-1)] \
                                    + x03 * yp1 * N.roll(Jintx, -1, axis = 2)[1, 3, 1:(Ny_int-1)] \
                                    + x50 * yp1 * N.roll(Jintx, -1, axis = 2)[0, -1, 1:(Ny_int-1)] \
                                    + x51 * yp1 * N.roll(Jintx, -1, axis = 2)[0, -2, 1:(Ny_int-1)] \
                                    + x52 * yp1 * N.roll(Jintx, -1, axis = 2)[0, -3, 1:(Ny_int-1)] \
                                        
        Jbuffx[1, 1, 1:(Ny_int-1)] = x10 * y00 * Jintx[1, 0, 1:(Ny_int-1)] \
                                   + x11 * y00 * Jintx[1, 1, 1:(Ny_int-1)] \
                                   + x12 * y00 * Jintx[1, 2, 1:(Ny_int-1)] \
                                   + x13 * y00 * Jintx[1, 3, 1:(Ny_int-1)] \
                                   + x40 * y00 * Jintx[0, -1, 1:(Ny_int-1)] \
                                   + x41 * y00 * Jintx[0, -2, 1:(Ny_int-1)] \
                                   + x42 * y00 * Jintx[0, -3, 1:(Ny_int-1)] \
                                   + x10 * ym1 * N.roll(Jintx, 1, axis = 2)[1, 0, 1:(Ny_int-1)] \
                                   + x11 * ym1 * N.roll(Jintx, 1, axis = 2)[1, 1, 1:(Ny_int-1)] \
                                   + x12 * ym1 * N.roll(Jintx, 1, axis = 2)[1, 2, 1:(Ny_int-1)] \
                                   + x13 * ym1 * N.roll(Jintx, 1, axis = 2)[1, 3, 1:(Ny_int-1)] \
                                   + x40 * ym1 * N.roll(Jintx, 1, axis = 2)[0, -1, 1:(Ny_int-1)] \
                                   + x41 * ym1 * N.roll(Jintx, 1, axis = 2)[0, -2, 1:(Ny_int-1)] \
                                   + x42 * ym1 * N.roll(Jintx, 1, axis = 2)[0, -3, 1:(Ny_int-1)] \
                                   + x10 * yp1 * N.roll(Jintx, -1, axis = 2)[1, 0, 1:(Ny_int-1)] \
                                   + x11 * yp1 * N.roll(Jintx, -1, axis = 2)[1, 1, 1:(Ny_int-1)] \
                                   + x12 * yp1 * N.roll(Jintx, -1, axis = 2)[1, 2, 1:(Ny_int-1)] \
                                   + x13 * yp1 * N.roll(Jintx, -1, axis = 2)[1, 3, 1:(Ny_int-1)] \
                                   + x40 * yp1 * N.roll(Jintx, -1, axis = 2)[0, -1, 1:(Ny_int-1)] \
                                   + x41 * yp1 * N.roll(Jintx, -1, axis = 2)[0, -2, 1:(Ny_int-1)] \
                                   + x42 * yp1 * N.roll(Jintx, -1, axis = 2)[0, -3, 1:(Ny_int-1)] \

        Jbuffx[1, 2, 1:(Ny_int-1)] = x20 * y00 * Jintx[1, 0, 1:(Ny_int-1)] \
                                   + x21 * y00 * Jintx[1, 1, 1:(Ny_int-1)] \
                                   + x22 * y00 * Jintx[1, 2, 1:(Ny_int-1)] \
                                   + x23 * y00 * Jintx[1, 3, 1:(Ny_int-1)] \
                                   + x20 * ym1 * N.roll(Jintx, 1, axis = 2)[1, 0, 1:(Ny_int-1)] \
                                   + x21 * ym1 * N.roll(Jintx, 1, axis = 2)[1, 1, 1:(Ny_int-1)] \
                                   + x22 * ym1 * N.roll(Jintx, 1, axis = 2)[1, 2, 1:(Ny_int-1)] \
                                   + x23 * ym1 * N.roll(Jintx, 1, axis = 2)[1, 3, 1:(Ny_int-1)] \
                                   + x20 * yp1 * N.roll(Jintx, -1, axis = 2)[1, 0, 1:(Ny_int-1)] \
                                   + x21 * yp1 * N.roll(Jintx, -1, axis = 2)[1, 1, 1:(Ny_int-1)] \
                                   + x22 * yp1 * N.roll(Jintx, -1, axis = 2)[1, 2, 1:(Ny_int-1)] \
                                   + x23 * yp1 * N.roll(Jintx, -1, axis = 2)[1, 3, 1:(Ny_int-1)] \

        Jinty[:, :, :] = Jbuffy[:, :, :]

        # Jy in bulk
        Jbuffy[:, 1:(Nx_int-1), 2:(Ny_half-2)] = 0.25   * Jinty[:, 1:(Nx_int-1), 2:(Ny_half-2)] \
                                               + 0.125  * N.roll(Jinty, 1, axis = 1)[:, 1:(Nx_int-1), 2:(Ny_half-2)] \
                                               + 0.125  * N.roll(Jinty, -1, axis = 1)[:, 1:(Nx_int-1), 2:(Ny_half-2)] \
                                               + 0.125  * N.roll(Jinty, 1, axis = 2)[:, 1:(Nx_int-1), 2:(Ny_half-2)] \
                                               + 0.125  * N.roll(Jinty, -1, axis = 2)[:, 1:(Nx_int-1), 2:(Ny_half-2)] \
                                               + 0.0625 * N.roll(N.roll(Jinty, 1, axis = 1), 1, axis = 2)[:, 1:(Nx_int-1), 2:(Ny_half-2)] \
                                               + 0.0625 * N.roll(N.roll(Jinty, -1, axis = 1), -1, axis = 2)[:, 1:(Nx_int-1), 2:(Ny_half-2)] \
                                               + 0.0625 * N.roll(N.roll(Jinty, 1, axis = 1), -1, axis = 2)[:, 1:(Nx_int-1), 2:(Ny_half-2)] \
                                               + 0.0625 * N.roll(N.roll(Jinty, -1, axis = 1), 1, axis = 2)[:, 1:(Nx_int-1), 2:(Ny_half-2)]

        # Jy at edge cell in patch 0
        Jbuffy[0, -1, 2:(Ny_half-2)] = c00 * y00 * Jinty[0, -1, 2:(Ny_half-2)] \
                                     + c01 * y00 * Jinty[0, -2, 2:(Ny_half-2)] \
                                     + c50 * y00 * Jinty[1, 0, 2:(Ny_half-2)] \
                                     + c04 * y00 * Jinty[1, 1, 2:(Ny_half-2)] \
                                     + c00 * ym1 * N.roll(Jinty, 1, axis = 2)[0, -1, 2:(Ny_half-2)] \
                                     + c01 * ym1 * N.roll(Jinty, 1, axis = 2)[0, -2, 2:(Ny_half-2)] \
                                     + c50 * ym1 * N.roll(Jinty, 1, axis = 2)[1, 0, 2:(Ny_half-2)] \
                                     + c04 * ym1 * N.roll(Jinty, 1, axis = 2)[1, 1, 2:(Ny_half-2)] \
                                     + c00 * yp1 * N.roll(Jinty, -1, axis = 2)[0, -1, 2:(Ny_half-2)] \
                                     + c01 * yp1 * N.roll(Jinty, -1, axis = 2)[0, -2, 2:(Ny_half-2)] \
                                     + c50 * yp1 * N.roll(Jinty, -1, axis = 2)[1, 0, 2:(Ny_half-2)] \
                                     + c04 * yp1 * N.roll(Jinty, -1, axis = 2)[1, 1, 2:(Ny_half-2)] \

        Jbuffy[0, -2, 2:(Ny_half-2)] = c10  * y00 * Jinty[0, -1, 2:(Ny_half-2)] \
                                     + c11  * y00 * Jinty[0, -2, 2:(Ny_half-2)] \
                                     + 0.25 * y00 * Jinty[0, -3, 2:(Ny_half-2)] \
                                     + c40  * y00 * Jinty[1, 0, 2:(Ny_half-2)] \
                                     + c12  * y00 * Jinty[1, 1, 2:(Ny_half-2)] \
                                     + c10  * ym1 * N.roll(Jinty, 1, axis = 2)[0, -1, 2:(Ny_half-2)] \
                                     + c11  * ym1 * N.roll(Jinty, 1, axis = 2)[0, -2, 2:(Ny_half-2)] \
                                     + 0.25 * ym1 * N.roll(Jinty, 1, axis = 2)[0, -3, 2:(Ny_half-2)] \
                                     + c40  * ym1 * N.roll(Jinty, 1, axis = 2)[1, 0, 2:(Ny_half-2)] \
                                     + c12  * ym1 * N.roll(Jinty, 1, axis = 2)[1, 1, 2:(Ny_half-2)] \
                                     + c10  * yp1 * N.roll(Jinty, -1, axis = 2)[0, -1, 2:(Ny_half-2)] \
                                     + c11  * yp1 * N.roll(Jinty, -1, axis = 2)[0, -2, 2:(Ny_half-2)] \
                                     + 0.25 * yp1 * N.roll(Jinty, -1, axis = 2)[0, -3, 2:(Ny_half-2)] \
                                     + c40  * yp1 * N.roll(Jinty, -1, axis = 2)[1, 0, 2:(Ny_half-2)] \
                                     + c12  * yp1 * N.roll(Jinty, -1, axis = 2)[1, 1, 2:(Ny_half-2)] \
                                         
        # Jx at edge cell in patch 1
        Jbuffy[1, 0, 2:(Ny_half-2)] = c00 * y00 * Jinty[1, 0, 2:(Ny_half-2)] \
                                    + c01 * y00 * Jinty[1, 1, 2:(Ny_half-2)] \
                                    + c50 * y00 * Jinty[0, -1, 2:(Ny_half-2)] \
                                    + c04 * y00 * Jinty[0, -2, 2:(Ny_half-2)] \
                                    + c00 * ym1 * N.roll(Jinty, 1, axis = 2)[1, 0, 2:(Ny_half-2)] \
                                    + c01 * ym1 * N.roll(Jinty, 1, axis = 2)[1, 1, 2:(Ny_half-2)] \
                                    + c50 * ym1 * N.roll(Jinty, 1, axis = 2)[0, -1, 2:(Ny_half-2)] \
                                    + c04 * ym1 * N.roll(Jinty, 1, axis = 2)[0, -2, 2:(Ny_half-2)] \
                                    + c00 * yp1 * N.roll(Jinty, -1, axis = 2)[1, 0, 2:(Ny_half-2)] \
                                    + c01 * yp1 * N.roll(Jinty, -1, axis = 2)[1, 1, 2:(Ny_half-2)] \
                                    + c50 * yp1 * N.roll(Jinty, -1, axis = 2)[0, -1, 2:(Ny_half-2)] \
                                    + c04 * yp1 * N.roll(Jinty, -1, axis = 2)[0, -2, 2:(Ny_half-2)] \

        Jbuffy[1, 1, 2:(Ny_half-2)] = c10  * y00 * Jinty[1, 0, 2:(Ny_half-2)] \
                                    + c11  * y00 * Jinty[1, 1, 2:(Ny_half-2)] \
                                    + 0.25 * y00 * Jinty[1, 2, 2:(Ny_half-2)] \
                                    + c40  * y00 * Jinty[0, -1, 2:(Ny_half-2)] \
                                    + c12  * y00 * Jinty[0, -2, 2:(Ny_half-2)] \
                                    + c10  * ym1 * N.roll(Jinty, 1, axis = 2)[1, 0, 2:(Ny_half-2)] \
                                    + c11  * ym1 * N.roll(Jinty, 1, axis = 2)[1, 1, 2:(Ny_half-2)] \
                                    + 0.25 * ym1 * N.roll(Jinty, 1, axis = 2)[1, 2, 2:(Ny_half-2)] \
                                    + c40  * ym1 * N.roll(Jinty, 1, axis = 2)[0, -1, 2:(Ny_half-2)] \
                                    + c12  * ym1 * N.roll(Jinty, 1, axis = 2)[0, -2, 2:(Ny_half-2)] \
                                    + c10  * yp1 * N.roll(Jinty, -1, axis = 2)[1, 0, 2:(Ny_half-2)] \
                                    + c11  * yp1 * N.roll(Jinty, -1, axis = 2)[1, 1, 2:(Ny_half-2)] \
                                    + 0.25 * yp1 * N.roll(Jinty, -1, axis = 2)[1, 2, 2:(Ny_half-2)] \
                                    + c40  * yp1 * N.roll(Jinty, -1, axis = 2)[0, -1, 2:(Ny_half-2)] \
                                    + c12  * yp1 * N.roll(Jinty, -1, axis = 2)[0, -2, 2:(Ny_half-2)] \

        rhoint[:, :, :] = rhobuff[:, :, :]

        # rho in bulk
        rhobuff[:, 2:(Nx_int-2), 1:(Ny_int-1)] = 0.50 * y00 * rhoint[:, 2:(Nx_int-2), 1:(Ny_int-1)] \
                                               + 0.25 * y00 * N.roll(rhoint, 1, axis = 1)[:, 2:(Nx_int-2), 1:(Ny_int-1)] \
                                               + 0.25 * y00 * N.roll(rhoint, -1, axis = 1)[:, 2:(Nx_int-2), 1:(Ny_int-1)] \
                                               + 0.50 * ym1 * N.roll(rhoint, 1, axis = 2)[:, 2:(Nx_int-2), 1:(Ny_int-1)] \
                                               + 0.50 * yp1 * N.roll(rhoint, -1, axis = 2)[:, 2:(Nx_int-2), 1:(Ny_int-1)] \
                                               + 0.25 * ym1 * N.roll(N.roll(rhoint, 1, axis = 1), 1, axis = 2)[:, 2:(Nx_int-2), 1:(Ny_int-1)] \
                                               + 0.25 * yp1 * N.roll(N.roll(rhoint, -1, axis = 1), -1, axis = 2)[:, 2:(Nx_int-2), 1:(Ny_int-1)] \
                                               + 0.25 * yp1 * N.roll(N.roll(rhoint, 1, axis = 1), -1, axis = 2)[:, 2:(Nx_int-2), 1:(Ny_int-1)] \
                                               + 0.25 * ym1 * N.roll(N.roll(rhoint, -1, axis = 1), 1, axis = 2)[:, 2:(Nx_int-2), 1:(Ny_int-1)]

        rhobuff[0, -1, 1:(Ny_int-1)] = c00 * y00 * rhoint[0, -1, 1:(Ny_int-1)] \
                                     + c01 * y00 * rhoint[0, -2, 1:(Ny_int-1)] \
                                     + c50 * y00 * rhoint[1, 0, 1:(Ny_int-1)] \
                                     + c04 * y00 * rhoint[1, 1, 1:(Ny_int-1)] \
                                     + c00 * ym1 * N.roll(rhoint, 1, axis = 2)[0, -1, 1:(Ny_int-1)] \
                                     + c01 * ym1 * N.roll(rhoint, 1, axis = 2)[0, -2, 1:(Ny_int-1)] \
                                     + c50 * ym1 * N.roll(rhoint, 1, axis = 2)[1, 0, 1:(Ny_int-1)] \
                                     + c04 * ym1 * N.roll(rhoint, 1, axis = 2)[1, 1, 1:(Ny_int-1)] \
                                     + c00 * yp1 * N.roll(rhoint, -1, axis = 2)[0, -1, 1:(Ny_int-1)] \
                                     + c01 * yp1 * N.roll(rhoint, -1, axis = 2)[0, -2, 1:(Ny_int-1)] \
                                     + c50 * yp1 * N.roll(rhoint, -1, axis = 2)[1, 0, 1:(Ny_int-1)] \
                                     + c04 * yp1 * N.roll(rhoint, -1, axis = 2)[1, 1, 1:(Ny_int-1)] \
                                        
        rhobuff[0, -2, 1:(Ny_int-1)] = c10  * y00 * rhoint[0, -1, 1:(Ny_int-1)] \
                                     + c11  * y00 * rhoint[0, -2, 1:(Ny_int-1)] \
                                     + 0.25 * y00 * rhoint[0, -3, 1:(Ny_int-1)] \
                                     + c40  * y00 * rhoint[1, 0, 1:(Ny_int-1)] \
                                     + c12  * y00 * rhoint[1, 1, 1:(Ny_int-1)] \
                                     + c10  * ym1 * N.roll(rhoint, 1, axis = 2)[0, -1, 1:(Ny_int-1)] \
                                     + c11  * ym1 * N.roll(rhoint, 1, axis = 2)[0, -2, 1:(Ny_int-1)] \
                                     + 0.25 * ym1 * N.roll(rhoint, 1, axis = 2)[0, -3, 1:(Ny_int-1)] \
                                     + c40  * ym1 * N.roll(rhoint, 1, axis = 2)[1, 0, 1:(Ny_int-1)] \
                                     + c12  * ym1 * N.roll(rhoint, 1, axis = 2)[1, 1, 1:(Ny_int-1)] \
                                     + c10  * yp1 * N.roll(rhoint, -1, axis = 2)[0, -1, 1:(Ny_int-1)] \
                                     + c11  * yp1 * N.roll(rhoint, -1, axis = 2)[0, -2, 1:(Ny_int-1)] \
                                     + 0.25 * yp1 * N.roll(rhoint, -1, axis = 2)[0, -3, 1:(Ny_int-1)] \
                                     + c40  * yp1 * N.roll(rhoint, -1, axis = 2)[1, 0, 1:(Ny_int-1)] \
                                     + c12  * yp1 * N.roll(rhoint, -1, axis = 2)[1, 1, 1:(Ny_int-1)] \

        rhobuff[1, 0, 1:(Ny_int-1)] = c00 * y00 * rhoint[1, 0, 1:(Ny_int-1)] \
                                    + c01 * y00 * rhoint[1, 1, 1:(Ny_int-1)] \
                                    + c50 * y00 * rhoint[0, -1, 1:(Ny_int-1)] \
                                    + c04 * y00 * rhoint[0, -2, 1:(Ny_int-1)] \
                                    + c00 * ym1 * N.roll(rhoint, 1, axis = 2)[1, 0, 1:(Ny_int-1)] \
                                    + c01 * ym1 * N.roll(rhoint, 1, axis = 2)[1, 1, 1:(Ny_int-1)] \
                                    + c50 * ym1 * N.roll(rhoint, 1, axis = 2)[0, -1, 1:(Ny_int-1)] \
                                    + c04 * ym1 * N.roll(rhoint, 1, axis = 2)[0, -2, 1:(Ny_int-1)] \
                                    + c00 * yp1 * N.roll(rhoint, -1, axis = 2)[1, 0, 1:(Ny_int-1)] \
                                    + c01 * yp1 * N.roll(rhoint, -1, axis = 2)[1, 1, 1:(Ny_int-1)] \
                                    + c50 * yp1 * N.roll(rhoint, -1, axis = 2)[0, -1, 1:(Ny_int-1)] \
                                    + c04 * yp1 * N.roll(rhoint, -1, axis = 2)[0, -2, 1:(Ny_int-1)] \
                                        
        rhobuff[1, 1, 1:(Ny_int-1)] = c10  * y00 * rhoint[1, 0, 1:(Ny_int-1)] \
                                    + c11  * y00 * rhoint[1, 1, 1:(Ny_int-1)] \
                                    + 0.25 * y00 * rhoint[1, 2, 1:(Ny_int-1)] \
                                    + c40  * y00 * rhoint[0, -1, 1:(Ny_int-1)] \
                                    + c12  * y00 * rhoint[0, -2, 1:(Ny_int-1)] \
                                    + c10  * ym1 * N.roll(rhoint, 1, axis = 2)[1, 0, 1:(Ny_int-1)] \
                                    + c11  * ym1 * N.roll(rhoint, 1, axis = 2)[1, 1, 1:(Ny_int-1)] \
                                    + 0.25 * ym1 * N.roll(rhoint, 1, axis = 2)[1, 2, 1:(Ny_int-1)] \
                                    + c40  * ym1 * N.roll(rhoint, 1, axis = 2)[0, -1, 1:(Ny_int-1)] \
                                    + c12  * ym1 * N.roll(rhoint, 1, axis = 2)[0, -2, 1:(Ny_int-1)] \
                                    + c10  * yp1 * N.roll(rhoint, -1, axis = 2)[1, 0, 1:(Ny_int-1)] \
                                    + c11  * yp1 * N.roll(rhoint, -1, axis = 2)[1, 1, 1:(Ny_int-1)] \
                                    + 0.25 * yp1 * N.roll(rhoint, -1, axis = 2)[1, 2, 1:(Ny_int-1)] \
                                    + c40  * yp1 * N.roll(rhoint, -1, axis = 2)[0, -1, 1:(Ny_int-1)] \
                                    + c12  * yp1 * N.roll(rhoint, -1, axis = 2)[0, -2, 1:(Ny_int-1)] \

    Jx[:, :, :] = Jbuffx[:, :, :]
    Jy[:, :, :] = Jbuffy[:, :, :]
    rho1f[:, :, :] = rhobuff[:, :, :]

    return

flux0 = N.zeros(Nt)
flux1 = N.zeros(Nt)
fluxa = N.zeros(Nt)

# Boundaries of Gauss' theorem diagnostic in patch 0 and 1
x0l = 0.7
x0r = 0.85
x1l = 0.05
x1r = 0.2

i0l = N.argmin(N.abs(x_half - x0l))
i0r = N.argmin(N.abs(x_half - x0r))
i1l = N.argmin(N.abs(x_half - x1l))
i1r = N.argmin(N.abs(x_half - x1r))

# Computes enclosed charge
def compute_charge():
    flux0[it]  = spi.simps(Ex[0, i0r, :], x=y_int) - spi.simps(Ex[0, i0l, :], x=y_int) \
               + spi.simps(Ey[0, i0l:i0r, -1], x=x_int[i0l:i0r]) - spi.simps(Ey[0, i0l:i0r, 0], x=x_int[i0l:i0r])
    flux1[it]  = spi.simps(Ex[1, i1r, :], x=y_int) - spi.simps(Ex[1, i1l, :], x=y_int) \
               + spi.simps(Ey[1, i1l:i1r, -1], x=x_int[i1l:i1r]) - spi.simps(Ey[1, i1l:i1r, 0], x=x_int[i1l:i1r])
    return

# Boundaries of Gauss' theorem across interface
xal = 0.95
xar = 1.05

ial = N.argmin(N.abs(x_half - xal))
iar = N.argmin(N.abs(x_half - (xar - x_max + x_min)))

# Computes enclosed charge
def compute_charge_across():

    fluxa[it]  = spi.simps(Ex[1, iar, :], x=y_int) - spi.simps(Ex[0, ial, :], x=y_int) \
               + spi.simps(Ey[0, ial:-1, -1], x=x_int[ial:-1]) - spi.simps(Ey[0, ial:-1, 0], x=x_int[ial:-1]) \
               + spi.simps(Ey[1, 1:iar, -1], x=x_int[1:iar]) - spi.simps(Ey[1, 1:iar, 0], x=x_int[ial:-1])

    return

########
# Boundary conditions
########

sig_abs = 0.5

def BC_conducting_B(dtin, Exin, Eyin, Bzin):

    # Bz[0, :, 0]  += dtin * sig_abs * Exin[0, :, 0]  / dx / P_half_2[0]
    # Bz[0, :, -1] -= dtin * sig_abs * Exin[0, :, -1] / dx / P_half_2[-1]

    # Bz[1, :, 0]  += dtin * sig_abs * Exin[1, :, 0]  / dx / P_half_2[0]
    # Bz[1, :, -1] -= dtin * sig_abs * Exin[1, :, -1] / dx / P_half_2[-1]
    
    # Bz[0, 0, :]  -= dtin * sig_abs * Eyin[0, 0, :]  / dx / P_half_2[0]
    # Bz[0, -1, :] += dtin * sig_abs * Eyin[0, -1, :] / dx / P_half_2[-1]

    # Bz[0, 0, :]  = 0.0
    # Bz[0, -1, :] = 0.0

    # Bz[1, :, 0]  += dtin * sig_abs * Exin[1, :, 0] / P_half_2[0]
    # Bz[1, :, -1] -= dtin * sig_abs * Exin[1, :, -1] / P_half_2[-1]
    # Bz[1, -1, :] += dtin * sig_abs * Eyin[1, -1, :] / P_half_2[-1]
    
    return    

def BC_conducting_E(dtin, Exin, Eyin, Bzin):
    return

# Absorbing outer boundaries

def BC_absorbing_B(dtin, Exin, Eyin, Bzin):

    Bz[0, 0, :]  -= dtin * sig_abs * (Eyin[0, 0, :] + Bzin[0, 0, :])   / dx / P_half_2[0]
    Bz[1, -1, :] += dtin * sig_abs * (Eyin[1, -1, :] - Bzin[1, -1, :]) / dx / P_half_2[-1]

    Bz[0, :, 0]  += dtin * sig_abs * (Exin[0, :, 0] - Bzin[0, :, 0])   / dx / P_half_2[0]
    Bz[0, :, -1] -= dtin * sig_abs * (Exin[0, :, -1] + Bzin[0, :, -1]) / dx / P_half_2[-1]
    # Bz[0, -1, :] += dtin * sig_abs * (Eyin[0, -1, :] - Bzin[0, -1, :]) / dx / P_half_2[-1]

    Bz[1, :, 0]  += dtin * sig_abs * (Exin[1, :, 0] - Bzin[1, :, 0]) / dx / P_half_2[0]
    Bz[1, :, -1] -= dtin * sig_abs * (Exin[1, :, -1] + Bzin[1, :, -1]) / dx / P_half_2[-1]
    # Bz[1, 0, :] -= dtin * sig_abs * (Eyin[1, 0, :] + Bzin[1, 0, :]) / dx / P_half_2[0]

    return

def BC_absorbing_E(dtin, Exin, Eyin, Bzin):

    Ey[0, 0, :]  -= dtin * sig_abs * (Eyin[0, 0, :] + Bzin[0, 0, :])   / dx / P_int_2[0]
    Ey[1, -1, :] -= dtin * sig_abs * (Eyin[1, -1, :] - Bzin[1, -1, :]) / dx / P_int_2[-1]

    Ex[0, :, 0]  -= dtin * sig_abs * (Exin[0, :, 0] - Bzin[0, :, 0])   / dx / P_int_2[0]
    Ex[0, :, -1] -= dtin * sig_abs * (Exin[0, :, -1] + Bzin[0, :, -1]) / dx / P_int_2[-1]    
    # Ey[0, -1, :] -= dtin * sig_abs * (Eyin[0, -1, :] - Bzin[0, -1, :]) / dx / P_int_2[-1]

    Ex[1, :, 0]  -= dtin * sig_abs * (Exin[1, :, 0] - Bzin[1, :, 0]) / dx / P_int_2[0]
    Ex[1, :, -1] -= dtin * sig_abs * (Exin[1, :, -1] + Bzin[1, :, -1]) / dx / P_int_2[-1]    
    # Ey[1, 0, :] -= dtin * sig_abs * (Eyin[1, 0, :] + Bzin[1, 0, :]) / dx / P_int_2[0]

    return

def BC_penalty_B(dtin, Exin, Eyin, Bzin):
    Bz[0, -1, :] -= dtin * sig_abs * (Bzin[0, -1, :] - Eyin[0, -1, :] - (Bzin[1, 0, :] - Eyin[1, 0, :])) / dx / P_half_2[-1]
    Bz[1, 0, :]  -= dtin * sig_abs * (Bzin[1, 0, :] + Eyin[1, 0, :] - (Bzin[0, -1, :] + Eyin[0, -1, :])) / dx / P_half_2[0]
    return

sig = sig_abs

def BC_penalty_E(dtin, Exin, Eyin, Bzin):
    penalty0 = - dtin * sig * (Eyin[0, -1, :] - Bzin[0, -1, :] - (Eyin[1, 0, :] - Bzin[1, 0, :])) / dx / P_int_2[-1]
    penalty1 = - dtin * sig * (Eyin[1, 0, :] + Bzin[1, 0, :] - (Eyin[0, -1, :] + Bzin[0, -1, :])) / dx / P_int_2[0]
    Ey[0, -1, :] -= dtin * sig * (Eyin[0, -1, :] - Bzin[0, -1, :] - (Eyin[1, 0, :] - Bzin[1, 0, :])) / dx / P_int_2[-1]
    Ey[1, 0, :]  -= dtin * sig * (Eyin[1, 0, :] + Bzin[1, 0, :] - (Eyin[0, -1, :] + Bzin[0, -1, :])) / dx / P_int_2[0]
    
    ## CLEANING
    Ex[0, -1, 2:-2] += (3.0 * gamma - 2.0) / (2.0 * delta - 2.0 * gamma) * (dx/dy) *(N.roll(penalty0, -1)[2:-3] - penalty0[2:-3]) 
    Ex[0, -2, 2:-2] += (gamma - 1.0) / (delta - gamma) * (dx/dy) *(N.roll(penalty0, -1)[2:-3] - penalty0[2:-3])

    Ex[0, -1, 1] += (3.0 * gamma - 2.0) / (2.0 * delta - 2.0 * gamma) * (dx/dy) * (2.0 * (gamma - 1) * penalty0[-1] + (2.0 - 3.0 * gamma) * penalty0[-2] + penalty0[-3]) 
    Ex[0, -2, 1] += (gamma - 1.0) / (delta - gamma) * (dx/dy) * (2.0 * (gamma - 1) * penalty0[-1] + (2.0 - 3.0 * gamma) * penalty0[-2] + penalty0[-3]) 
    Ex[0, 0, 1] += (3.0 * gamma - 2.0) / (2.0 * delta - 2.0 * gamma) * (dx/dy) * (2.0 * (gamma - 1) * penalty0[0] + (2.0 - 3.0 * gamma) * penalty0[1] + penalty0[0]) 
    Ex[0, 1, 1] += (gamma - 1.0) / (delta - gamma) * (dx/dy) * (2.0 * (gamma - 1) * penalty0[0] + (2.0 - 3.0 * gamma) * penalty0[1] + penalty0[0]) 

    Ex[1, 0, 2:-2]  -= (3.0 * gamma - 2.0) / (2.0 * delta - 2.0 * gamma) * (dx/dy) *(N.roll(penalty1, -1)[2:-3] - penalty1[2:-3])
    Ex[1, 1, 2:-2]  -= (gamma - 1.0) / (delta - gamma) * (dx/dy) *(N.roll(penalty1, -1)[2:-3] - penalty1[2:-3])

    Ex[1, -1, 1] -= (3.0 * gamma - 2.0) / (2.0 * delta - 2.0 * gamma) * (dx/dy) * (2.0 * (gamma - 1) * penalty1[-1] + (2.0 - 3.0 * gamma) * penalty1[-2] + penalty1[-3]) 
    Ex[1, -2, 1] -= (gamma - 1.0) / (delta - gamma) * (dx/dy) * (2.0 * (gamma - 1) * penalty1[-1] + (2.0 - 3.0 * gamma) * penalty1[-2] + penalty1[-3]) 
    Ex[1, 0, 1] -= (3.0 * gamma - 2.0) / (2.0 * delta - 2.0 * gamma) * (dx/dy) * (2.0 * (gamma - 1) * penalty1[0] + (2.0 - 3.0 * gamma) * penalty1[1] + penalty1[0]) 
    Ex[1, 1, 1] -= (gamma - 1.0) / (delta - gamma) * (dx/dy) * (2.0 * (gamma - 1) * penalty1[0] + (2.0 - 3.0 * gamma) * penalty1[1] + penalty1[0]) 

    return

def BC_periodic_B(dtin, Exin, Eyin, Bzin):

    Bz[0, :, 0]  += dtin * sig_abs * (Exin[0, :, 0] - Bzin[0, :, 0] - (Exin[0, :, -1] - Bzin[0, :, -1]))   / dx / P_half_2[0]
    Bz[0, :, -1] -= dtin * sig_abs * (Exin[0, :, -1] + Bzin[0, :, -1] - (Exin[0, :, 0] + Bzin[0, :, 0])) / dx / P_half_2[-1]
    # Bz[0, -1, :] += dtin * sig_abs * (Eyin[0, -1, :] - Bzin[0, -1, :]) / dx / P_half_2[-1]

    Bz[1, :, 0]  += dtin * sig_abs * (Exin[1, :, 0] - Bzin[1, :, 0] - (Exin[1, :, -1] - Bzin[1, :, -1])) / dx / P_half_2[0]
    Bz[1, :, -1] -= dtin * sig_abs * (Exin[1, :, -1] + Bzin[1, :, -1] - (Exin[1, :, 0] + Bzin[1, :, 0])) / dx / P_half_2[-1]
    # Bz[1, 0, :] -= dtin * sig_abs * (Eyin[1, 0, :] + Bzin[1, 0, :]) / dx / P_half_2[0]

    return

def BC_periodic_E(dtin, Exin, Eyin, Bzin):

    Ex[0, :, 0]  -= dtin * sig_abs * (Exin[0, :, 0] - Bzin[0, :, 0] - (Exin[0, :, -1] - Bzin[0, :, -1])) / dx / P_int_2[0]
    Ex[0, :, -1] -= dtin * sig_abs * (Exin[0, :, -1] + Bzin[0, :, -1] - (Exin[0, :, 0] + Bzin[0, :, 0])) / dx / P_int_2[-1]    
    # Ey[0, -1, :] -= dtin * sig_abs * (Eyin[0, -1, :] - Bzin[0, -1, :]) / dx / P_half_2[-1]

    Ex[1, :, 0]  -= dtin * sig_abs * (Exin[1, :, 0] - Bzin[1, :, 0] - (Exin[1, :, -1] - Bzin[1, :, -1])) / dx / P_int_2[0]
    Ex[1, :, -1] -= dtin * sig_abs * (Exin[1, :, -1] + Bzin[1, :, -1] - (Exin[1, :, 0] + Bzin[1, :, 0])) / dx / P_int_2[-1]    
    # Ey[1, 0, :] -= dtin * sig_abs * (Eyin[1, 0, :] + Bzin[1, 0, :]) / dx / P_half_2[0]

    return

########
# Visualization
########

ratio = 0.5

vm = 0.4

from matplotlib.gridspec import GridSpec

def plot_fields(idump, it):

    fig = P.figure(1, facecolor='w', figsize=(30,10))
    gs = GridSpec(2, 3, figure=fig)
    
    ax = fig.add_subplot(gs[0, :3])
    
    P.pcolormesh(xEx_grid, yEx_grid, Ex[0, :, :], vmin = -vm, vmax = vm, cmap = 'RdBu_r')
    P.pcolormesh(xEx_grid + x_max, yEx_grid, Ex[1, :, :], vmin = -vm, vmax = vm, cmap = 'RdBu_r')
    
    # P.pcolormesh(xEx_grid, yEx_grid, Jx[0, :, :], vmin = -0.1, vmax = 0.1, cmap = 'RdBu_r')
    # P.pcolormesh(xEx_grid + x_max, yEx_grid, Jx[1, :, :], vmin = -0.1, vmax = 0.1, cmap = 'RdBu_r')

    # P.pcolormesh(xEx_grid, yEx_grid, Jx[0, :, :] + Jxout[0, :, :, it], vmin = -0.1, vmax = 0.1, cmap = 'RdBu_r')
    # P.pcolormesh(xEx_grid + x_max, yEx_grid, Jx[1, :, :] + Jxout[1, :, :, it], vmin = -0.1, vmax = 0.1, cmap = 'RdBu_r')
    
    for ip in range(np):
        if (tag[it, ip]==0):
            P.scatter(xp[it, ip], yp[it, ip], s=10)
        elif (tag[it, ip]==1):
            P.scatter(xp[it, ip] + x_max - x_min, yp[it, ip], s=10)
    
    P.title(r'$E_x$', fontsize=16)
    
    # P.colorbar()
    
    P.xlim((0.5, 1.5))
    P.ylim((0.25, 0.75))

    ax.set_aspect(1.0/ax.get_data_ratio()*ratio)

    P.plot([1.0, 1.0],[0, 1.0], color='k')

    # P.plot([x0l, x0l],[0, 1.0], color=[0,0.4,0.4],   lw=1.0, ls = '--')
    # P.plot([x0r, x0r],[0, 1.0], color=[0,0.4,0.4],   lw=1.0, ls = '--')
    # P.plot([x1l + x_max, x1l + x_max],[0, 1.0], color=[0.8,0.0,0.0], lw=1.0, ls = '--')
    # P.plot([x1r + x_max, x1r + x_max],[0, 1.0], color=[0.8,0.0,0.0], lw=1.0, ls = '--')

    P.plot([xal, xal],[0, 1.0], color=[0,0.4,0.4],   lw=1.0, ls = '--')
    P.plot([xar, xar],[0, 1.0], color=[0,0.4,0.4],   lw=1.0, ls = '--')

    ax = fig.add_subplot(gs[1, :3])

    P.pcolormesh(xEy_grid, yEy_grid, Ey[0, :, :], vmin = -vm, vmax = vm, cmap = 'RdBu_r')
    P.pcolormesh(xEy_grid + x_max, yEy_grid, Ey[1, :, :], vmin = -vm, vmax = vm, cmap = 'RdBu_r')

    # P.colorbar()

    for ip in range(np):
        if (tag[it, ip]==0):
            P.scatter(xp[it, ip], yp[it, ip], s=10)
        elif (tag[it, ip]==1):
            P.scatter(xp[it, ip] + x_max - x_min, yp[it, ip], s=10)

    P.title(r'$E_y$', fontsize=16)

    P.xlim((0.5, 1.5))
    P.ylim((0.25, 0.75))
        
    ax.set_aspect(1.0/ax.get_data_ratio()*ratio)

    P.plot([1.0, 1.0],[0, 1.0], color='k')

    ax = fig.add_subplot(gs[:, -1])

    # P.scatter(time[it-1], flux0[it-1] / (4.0 * N.pi * q), s=10, color=[0, 0.4, 0.4])
    # P.scatter(time[it-1], flux1[it-1] / (4.0 * N.pi * q), s=10, color=[0.8, 0.0, 0.0])
    # P.plot(time[:it], flux0[:it] / (4.0 * N.pi * q), color = [0, 0.4, 0.4], ls = '-')
    # P.plot(time[:it], flux1[:it] / (4.0 * N.pi * q), color = [0.8, 0.0, 0.0], ls = '-')

    P.scatter(time[it-1], fluxa[it-1] / (4.0 * N.pi * q), s=10, color=[0, 0.4, 0.4])
    P.plot(time[:it], fluxa[:it] / (4.0 * N.pi * q), color = [0, 0.4, 0.4], ls = '-')

    P.plot([0.0, Nt*dt],[1.0, 1.0], color='k',   lw=1.0, ls = '--')
    P.plot([0.0, Nt*dt],[-1.0, -1.0], color='k', lw=1.0, ls = '--')
    
    P.xlim((0.0, Nt * dt))
    P.ylim((-1.1, 1.1))
    P.xlabel(r'$t$')
    P.ylabel(r'$Q$')
    
    figsave_png(fig, "../snapshots_penalty/fields_" + str(idump))
    
    P.close('all')

def plot_fields_zoom(idump, it):

    fig = P.figure(1, facecolor='w', figsize=(30,10))
    gs = GridSpec(2, 1, figure=fig)
    
    ax = fig.add_subplot(gs[0, 0])

    # P.pcolormesh(xEx_grid, yEx_grid, Ex[0, :, :], vmin = -vm, vmax = vm, cmap = 'RdBu_r')
    # P.pcolormesh(xEx_grid + x_max + 2.0 * dx, yEx_grid, Ex[1, :, :], vmin = -vm, vmax = vm, cmap = 'RdBu_r')
    P.pcolormesh(xEx_grid, yEx_grid, Jx[0, :, :], vmin = -1.0, vmax = 1.0, cmap = 'RdBu_r')
    P.pcolormesh(xEx_grid + x_max + 2.0 * dx, yEx_grid, Jx[1, :, :], vmin = -1.0, vmax = 1.0, cmap = 'RdBu_r')
    
    for ip in range(np):
        if (tag[it, ip]==0):
            P.scatter(xp[it, ip], yp[it, ip], s=20, color='k')
        elif (tag[it, ip]==1):
            P.scatter(xp[it, ip] + x_max - x_min+ 2.0 * dx, yp[it, ip], s=20, color='k')
    
    P.title(r'$E_x$', fontsize=16)
        
    P.ylim((0.0, 0.5))
    P.xlim((0.8, 1.2))
    ax.set_aspect(1.0/ax.get_data_ratio()*ratio)

    P.plot([1.0, 1.0],[0, 1.0], color='k')

    # P.plot([x0l, x0l],[0, 1.0], color=[0,0.4,0.4],   lw=1.0, ls = '--')
    # P.plot([x0r, x0r],[0, 1.0], color=[0,0.4,0.4],   lw=1.0, ls = '--')
    # P.plot([x1l + x_max, x1l + x_max],[0, 1.0], color=[0.8,0.0,0.0], lw=1.0, ls = '--')
    # P.plot([x1r + x_max, x1r + x_max],[0, 1.0], color=[0.8,0.0,0.0], lw=1.0, ls = '--')


    ax = fig.add_subplot(gs[1, 0])

    # P.pcolormesh(xEx_grid, yEx_grid, Ex[0, :, :], vmin = -vm, vmax = vm, cmap = 'RdBu_r')
    # P.pcolormesh(xEx_grid + x_max + 2.0 * dx, yEx_grid, Ex[1, :, :], vmin = -vm, vmax = vm, cmap = 'RdBu_r')
    P.pcolormesh(xEy_grid, yEy_grid, Jy[0, :, :], vmin = -vm, vmax = vm, cmap = 'RdBu_r')
    P.pcolormesh(xEy_grid + x_max + 2.0 * dx, yEy_grid, Jy[1, :, :], vmin = -vm, vmax = vm, cmap = 'RdBu_r')

    for ip in range(np):
        if (tag[it, ip]==0):
            P.scatter(xp[it, ip], yp[it, ip], s=20, color='k')
        elif (tag[it, ip]==1):
            P.scatter(xp[it, ip] + x_max - x_min+ 2.0 * dx, yp[it, ip], s=20, color='k')

    P.title(r'$E_y$', fontsize=16)
        
    P.ylim((0.0, 0.5))
    P.xlim((0.8, 1.2))
    ax.set_aspect(1.0/ax.get_data_ratio()*ratio)

    P.plot([1.0, 1.0],[0, 1.0], color='k')
    
    figsave_png(fig, "../snapshots_penalty/fields_zoom_" + str(idump))
    
    P.close('all')


def plot_div(idump, it):

    fig = P.figure(2, facecolor='w', figsize=(30,10))
    ax = P.subplot(211)

    P.pcolormesh(xEz_grid, yEz_grid, q * N.abs(divE1[0, :, :] - 4.0 * N.pi * rho1f[0, :, :]), vmin = -0.1, vmax = 0.1, cmap = 'RdBu_r')
    P.pcolormesh(xEz_grid + x_max + 2.0 * dx, yEz_grid, q * N.abs(divE1[1, :, :] - 4.0 * N.pi * rho1f[1, :, :]), vmin = -0.1, vmax = 0.1, cmap = 'RdBu_r')
        
    # P.ylim((0.0, 0.5))
    # P.xlim((0.6, 0.9))
    P.ylim((y_min, y_max))
    P.xlim((0.9, 1.1))
    # ax.set_aspect(1.0/ax.get_data_ratio()*ratio)
    
    ax = P.subplot(212)

    P.pcolormesh(xEz_grid, yEz_grid, q * N.abs(divcharge[0, :, :]), vmin = -0.1, vmax = 0.1, cmap = 'RdBu_r')
    P.pcolormesh(xEz_grid + x_max + 2.0 * dx, yEz_grid, N.abs(divcharge[1, :, :]), vmin = -0.1, vmax = 0.1, cmap = 'RdBu_r')
        
    # P.ylim((0.0, 0.5))
    # P.xlim((0.6, 0.9))
    P.ylim((y_min, y_max))
    P.xlim((0.8, 1.2))
    # ax.set_aspect(1.0/ax.get_data_ratio()*ratio)

    figsave_png(fig, "../snapshots_penalty/divergence_" + str(idump))

    P.close('all')
    
def compute_energy():
    
    energy[0, it] = dx * dy * N.sum(Bz[0, :, :]**2) + N.sum(Ex[0, :, :]**2) + N.sum(Ey[0, :, :]**2)
    energy[1, it] = dx * dy * N.sum(Bz[1, :, :]**2) + N.sum(Ex[1, :, :]**2) + N.sum(Ey[1, :, :]**2)
    
    return

########
# Initialization
########

amp = 0.0
n_mode = 2
n_iter = 8

wave = 2.0 * (x_max - x_min) / n_mode
Bz0 = amp * N.cos(2.0 * N.pi * (xBz_grid - x_min) / wave) * N.cos(2.0 * N.pi * (yBz_grid - x_min) / wave)
Ex0 = N.zeros((Nx_half, Ny_int))
Ey0 = N.zeros((Nx_int, Ny_half))

for p in range(n_patches):
    Bz[p, :, :] = Bz0[:, :]
    Ex[p, :, :] = Ex0[:, :]
    Ey[p, :, :] = Ey0[:, :]

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
        # plot_fields_zoom(idump, it)
        # plot_div(idump, it)
        idump += 1

    # print(it, Nt)

    # 1st Faraday substep, starting from B at n-1/2, E at n, finishing with B at n
    compute_diff_E(patches)
    push_B(patches, it, 0.5 * dt)
    
    # Here, Bz is defined at n, no need for averaging
    # BC_conducting_B(0.5 * dt, Ex[:, :, :], Ey[:, :, :], Bz[:, :, :])
    BC_absorbing_B(0.5 * dt, Ex[:, :, :], Ey[:, :, :], Bz[:, :, :])
    # BC_periodic_B(0.5 * dt, Ex[:, :, :], Ey[:, :, :], Bz[:, :, :])
    BC_penalty_B(0.5 * dt, Ex[:, :, :], Ey[:, :, :], Bz[:, :, :])
    
    Bz0[:, :, :] = Bz[:, :, :]

    # Particle push
    for ip in range(np):
        # push_u(it, ip)
        impose_velocity_part(it)
        push_x(it, ip)
        BC_part(it, ip)

    # Current deposition
    deposit_J(it)

    compute_divcharge(patches, divJ)

    filter_current(n_iter)

    compute_divcharge(patches, divJf)
    
    # 2nd Faraday substep, starting with B at n, finishing with B at n + 1/2
    compute_diff_E(patches)
    push_B(patches, it, 0.5 * dt)
    
    # Use Bz0, defined at n, this time
    # BC_conducting_B(0.5 * dt, Ex[:, :, :], Ey[:, :, :], Bz0[:, :, :])
    BC_absorbing_B(0.5 * dt, Ex[:, :, :], Ey[:, :, :], Bz0[:, :, :])
    # BC_periodic_B(0.5 * dt, Ex[:, :, :], Ey[:, :, :], Bz0[:, :, :])
    BC_penalty_B(0.5 * dt, Ex[:, :, :], Ey[:, :, :], Bz0[:, :, :])

    # Ampère step, starting with E at n, finishing with E at n + 1
    Ex0[:, :, :] = Ex[:, :, :]
    Ey0[:, :, :] = Ey[:, :, :]
    compute_diff_B(patches)
    push_E(patches, it, dt)

    Ex00 = Ex[:, 0, 0]
    Ex10 = Ex[:, -1, 0]
    Ex01 = Ex[:, 0, -1]
    Ex11 = Ex[:, -1, -1]
    Ey00 = Ey[:, 0, 0]
    Ey01 = Ey[:, 0, -1]
    Ey10 = Ey[:, -1, 0]
    Ey11 = Ey[:, -1, -1]

    # Use averaged E field, defined at n + 1/2  
    # BC_conducting_E(dt, 0.5 * (Ex0[:, :, :] + Ex[:, :, :]), 0.5 * (Ey0[:, :, :] + Ey[:, :, :]), Bz[:, :, :])
    BC_absorbing_E(dt, 0.5 * (Ex0[:, :, :] + Ex[:, :, :]), 0.5 * (Ey0[:, :, :] + Ey[:, :, :]), Bz[:, :, :])
    # BC_periodic_E(dt, 0.5 * (Ex0[:, :, :] + Ex[:, :, :]), 0.5 * (Ey0[:, :, :] + Ey[:, :, :]), Bz[:, :, :])
    BC_penalty_E(dt, 0.5 * (Ex0[:, :, :] + Ex[:, :, :]), 0.5 * (Ey0[:, :, :] + Ey[:, :, :]), Bz[:, :, :])

    # Clean corners
    Ex[0, -1, 0]  -= 0.5 * (Ex[0, -1, 0]  - Ex10[0])
    Ey[0, -1, 0]  -= 0.5 * (Ey[0, -1, 0]  - Ey10[0])
    Ex[0, -1, -1] -= 0.5 * (Ex[0, -1, -1] - Ex11[0])
    Ey[0, -1, -1] -= 0.5 * (Ey[0, -1, -1] - Ey11[0])
    Ex[1, 0, 0]   -= 0.5 * (Ex[1, 0, 0]   - Ex00[1])
    Ey[1, 0, 0]   -= 0.5 * (Ey[1, 0, 0]   - Ey00[1])
    Ex[1, 0, -1]  -= 0.5 * (Ex[1, 0, -1]  - Ex01[1])
    Ey[1, 0, -1]  -= 0.5 * (Ey[1, 0, -1]  - Ey01[1])

    compute_divE(patches, Ex[:, :, :], Ey[:, :, :])

    energy[0, it] = dx * dy * N.sum(Bz[0, :, :]**2) + N.sum(Ex[0, :, :]**2) + N.sum(Ey[0, :, :]**2)
    energy[1, it] = dx * dy * N.sum(Bz[1, :, :]**2) + N.sum(Ex[1, :, :]**2) + N.sum(Ey[1, :, :]**2)
    compute_charge()
    compute_charge_across()
    
    compute_energy()
