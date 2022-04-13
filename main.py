# Import modules
import numpy as N
import matplotlib.pyplot as P
import matplotlib
import time
import gc
import scipy.integrate as spi
from skimage.measure import find_contours
from math import *
import sys
# Import my figure routines
from figure_module import *

# Connect patch indices and names
sphere = {0: "A", 1: "B", 2: "C", 3: "D", 4: "N", 5: "S"}

# Parameters
cfl = 0.4
Nxi = 32
Neta = 32
NG = 1 # Number of ghosts zones
xi_min, xi_max = - N.pi / 4.0, N.pi / 4.0
eta_min, eta_max = - N.pi / 4.0, N.pi / 4.0
dxi = (xi_max - xi_min) / Nxi
deta = (eta_max - eta_min) / Neta

# Define grids
xi  = N.arange(- NG - int(Nxi / 2), NG + int(Nxi / 2), 1) * dxi
eta = N.arange(- NG - int(Neta / 2), NG + int(Neta / 2), 1) * deta
eta_grid, xi_grid = N.meshgrid(eta, xi)
xi_yee  = xi  + 0.5 * dxi
eta_yee = eta + 0.5 * deta

# Initialize fields
Er  = N.zeros((Nxi + 2 * NG, Neta + 2 * NG, 6))
E1u = N.zeros((Nxi + 2 * NG, Neta + 2 * NG, 6))
E2u = N.zeros((Nxi + 2 * NG, Neta + 2 * NG, 6))
Br  = N.zeros((Nxi + 2 * NG, Neta + 2 * NG, 6))
B1u = N.zeros((Nxi + 2 * NG, Neta + 2 * NG, 6))
B2u = N.zeros((Nxi + 2 * NG, Neta + 2 * NG, 6))

E1d = N.zeros((Nxi + 2 * NG, Neta + 2 * NG, 6))
E2d = N.zeros((Nxi + 2 * NG, Neta + 2 * NG, 6))
B1d = N.zeros((Nxi + 2 * NG, Neta + 2 * NG, 6))
B2d = N.zeros((Nxi + 2 * NG, Neta + 2 * NG, 6))

divB = N.zeros((2 * Nxi + 2 * NG, Neta + 2 * NG, 6))
divE = N.zeros((2 * Nxi + 2 * NG, Neta + 2 * NG, 6))
Ar = N.zeros((2 * Nxi + 2 * NG, Neta + 2 * NG, 6))

########
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

########
# Single-patch push routines
########

def contra_to_cov_E(patch):
    
    i0, i1 = NG, Nxi + NG
    j0, j1 = NG, Neta + NG
             
    E1d[i0:i1, j0:j1, patch] = g11d[i0:i1, j0:j1, 1] * E1u[i0:i1, j0:j1, patch] + \
                         0.5 * g12d[i0:i1, j0:j1, 1] * (E2u[i0:i1, j0:j1, patch] + N.roll(N.roll(E2u, -1, axis = 0), 1, axis = 1)[i0:i1, j0:j1, patch])
    E2d[i0:i1, j0:j1, patch] = g22d[i0:i1, j0:j1, 2] * E2u[i0:i1, j0:j1] + \
                         0.5 * g12d[i0:i1, j0:j1, 2] * (E1u[i0:i1, j0:j1, patch] + N.roll(N.roll(E1u, 1, axis = 0), -1, axis = 1)[i0:i1, j0:j1, patch])

def contra_to_cov_B(patch):

    i0, i1 = NG, Nxi + NG 
    j0, j1 = NG, Neta + NG
    
    B1d[i0:i1, j0:j1, patch] = g11d[i0:i1, j0:j1, 2] * B1u[i0:i1, j0:j1, patch] + \
                         0.5 * g12d[i0:i1, j0:j1, 2] * (B2u[i0:i1, j0:j1, patch] + N.roll(N.roll(B2u, 1, axis = 0), -1, axis = 1)[i0:i1, j0:j1, patch])            
    B2d[i0:i1, j0:j1, patch] = g22d[i0:i1, j0:j1, 1] * B2u[i0:i1, j0:j1, patch] + \
                         0.5 * g12d[i0:i1, j0:j1, 1] * (B1u[i0:i1, j0:j1, patch] + N.roll(N.roll(B1u, -1, axis = 0), 1, axis = 1)[i0:i1, j0:j1, patch])

def push_B(patch):
    
    i0, i1 = NG, Nxi + NG
    j0, j1 = NG, Neta + NG
    
    Br[i0:i1, j0:j1, patch]  -= ((N.roll(E2d, -1, axis = 0)[i0:i1, j0:j1, patch] - E2d[i0:i1, j0:j1, patch]) / dxi - \
                                 (N.roll(E1d, -1, axis = 1)[i0:i1, j0:j1, patch] - E1d[i0:i1, j0:j1, patch]) / deta) \
                                * dt / sqrt_det_g[i0:i1, j0:j1, 3]
    B1u[i0:i1, j0:j1, patch] -= ((N.roll(Er, -1, axis = 1)[i0:i1, j0:j1, patch] - Er[i0:i1, j0:j1, patch]) / deta) * dt / sqrt_det_g[i0:i1, j0:j1, 2]
    B2u[i0:i1, j0:j1, patch] += ((N.roll(Er, -1, axis = 0)[i0:i1, j0:j1, patch] - Er[i0:i1, j0:j1, patch]) / dxi)  * dt / sqrt_det_g[i0:i1, j0:j1, 1]

def push_E(it, patch):
    
    i0, i1 = NG, Nxi + NG
    j0, j1 = NG, Neta + NG
    
    Er[i0:i1, j0:j1, patch] += ((B2d[i0:i1, j0:j1, patch] - N.roll(B2d, 1, axis = 0)[i0:i1, j0:j1, patch]) / dxi - \
                                (B1d[i0:i1, j0:j1, patch] - N.roll(B1d, 1, axis = 1)[i0:i1, j0:j1, patch]) / deta) \
                               * dt / sqrt_det_g[i0:i1, j0:j1, 0] - 4.0 * N.pi * dt * Jr(it, xi_grid[i0:i1, j0:j1, patch], eta_grid[i0:i1, j0:j1]) 
    E1u[i0:i1, j0:j1, patch] += ((Br[i0:i1, j0:j1, patch] - N.roll(Br, 1, axis = 1)[i0:i1, j0:j1, patch]) / deta) * dt / sqrt_det_g[i0:i1, j0:j1, 1]
    E2u[i0:i1, j0:j1, patch] -= ((Br[i0:i1, j0:j1, patch] - N.roll(Br, 1, axis = 0)[i0:i1, j0:j1, patch]) / dxi)  * dt / sqrt_det_g[i0:i1, j0:j1, 2]

########
# Topology of the patches
########

topology = N.array([
    [   0, 'xy',    0,    0, 'yx',    0],
    [   0,    0, 'xy',    0, 'yy',    0],
    [   0,    0,    0, 'yy',    0, 'xy'],
    ['yx',    0,    0,    0,    0, 'xx'],
    [   0,    0, 'xx', 'yx',    0,    0],
    ['yy', 'xy',    0,    0,    0,    0]
                ])

########
# Generic coordinate transformation
########

from coord_transformations import *

def transform_coords(patch0, patch1, xi0, eta0):
    fcoord0 = (globals()["coord_" + sphere[patch0] + "_to_sph"])
    fcoord1 = (globals()["coord_sph_to_" + sphere[patch1]])
    return fcoord1(fcoord0(xi0, eta0))

########
# Generic vector transformation 
########

from vec_transformations import *

def transform_vec(patch0, patch1, xi0, eta0, vxi0, veta0):
    fcoord0 = (globals()["coord_" + sphere[patch0] + "_to_sph"])
    theta0, phi0 = fcoord0(xi0, eta0)
    fvec0 = (globals()["vec_" + sphere[patch0] + "_to_sph"])
    fvec1 = (globals()["vec_sph_to_" + sphere[patch1]])
    return fvec1(theta0, phi0, fvec0(xi0, eta0, vxi0, veta0))

########
# Linear form transformations
########

from form_transformations import *

def transform_form(patch0, patch1, xi0, eta0, vxi0, veta0):
    fcoord0 = (globals()["coord_" + sphere[patch0] + "_to_sph"])
    theta0, phi0 = fcoord0(xi0, eta0)
    fform0 = (globals()["form_" + sphere[patch0] + "_to_sph"])
    fform1 = (globals()["form_sph_to_" + sphere[patch1]])
    return fform1(theta0, phi0, fform0(xi0, eta0, vxi0, veta0))

########
# Interpolation
########

# def interp_vec_A_to_B(xi_a, eta_b, loc, fieldr, field1, field2):
    
#     # Right edge of patch A
#     i0 = Nxi + NG - 1
#     j  = int((eta_b - eta_min) / deta) + 1
    
#     # Select closest neighbors of eta_b in patch b coordinates
#     if (eta_b > 0):
#         j_bot = j - 1
#         j_top = j
#     else:
#         j_bot = j
#         j_top = j + 1
        
#     if (loc == "half"):
#         y_bot = eta_yee[j_bot]
#         y_top = eta_yee[j_top]  
#     elif (loc == "integer"):
#         y_bot = eta[j_bot]
#         y_top = eta[j_top]  
    
#     # Define field components to be interpolated
#     field1_bot, field1_top = field1[i0, j_bot], field1[i0, j_top]
#     field2_bot, field2_top = field2[i0, j_bot], field2[i0, j_top]
#     fieldr_bot, fieldr_top = fieldr[i0, j_bot], fieldr[i0, j_top]

#     # Interpolate field components to center of cell
    
#     # Rotate fields
#     vxi_b_bot, veta_b_bot = vec_A_to_B(xi_a, y_bot, field1_bot, field2_bot)    
#     vxi_b_top, veta_b_top = vec_A_to_B(xi_a, y_top, field1_top, field2_top)    
#     eta_b_bot = coord_A_to_B(xi_a, y_bot)[1]
#     eta_b_top = coord_A_to_B(xi_a, y_top)[1]
            
#     # Define weight coefficients for interpolation
#     w1 = (eta_b - eta_b_bot) / (eta_b_top - eta_b_bot)
#     w2 = (eta_b_top - eta_b) / (eta_b_top - eta_b_bot)

#     return w1 * fieldr_top + w2 * fieldr_bot, w1 * vxi_b_top + w2 * vxi_b_bot, w1 * veta_b_top + w2 * veta_b_bot

########
# Initial values
########

theta0, phi0 = 90.0 / 360.0 * 2.0 * N.pi, 60.0 / 360.0 * 2.0 * N.pi # Center of the wave packet
x0 = N.sin(theta0) * N.cos(phi0)
y0 = N.sin(theta0) * N.sin(phi0)
z0 = N.cos(theta0)

w = 0.1 # Radius of wave packet
omega = 1.0 # Frequency of current
J0 = 1.0 # Current amplitude

def shape_packet(x, y, z, width):
    return N.exp(- y * y / (width * width)) * N.exp(- x * x / (width * width)) * N.exp(- z * z / (width * width)) 

########
# Source current
########

def Jr(it, xi, eta, patch):

    fcoord0 = (globals()["coord_" + sphere[patch] + "_to_sph"])
    theta, phi = fcoord0(xi, eta)
    x = N.sin(theta) * N.cos(phi)
    y = N.sin(theta) * N.sin(phi)
    z = N.cos(theta)        
    return J0 * N.sin(omega * dt * it) * shape_packet(x - x0, y - y0, z - z0, w)

########
# Plotting function on a sphere
########

# Define background sphere
phi_s = N.linspace(0, N.pi, 2*50)
theta_s = N.linspace(0, 2*N.pi, 2*50)
theta_s, phi_s = N.meshgrid(theta_s, phi_s)
x_s = 0.95 * N.sin(phi_s) * N.cos(theta_s)
y_s = 0.95 * N.sin(phi_s) * N.sin(theta_s)
z_s = 0.95 * N.cos(phi_s)

xf = N.zeros_like(Er)
yf = N.zeros_like(Er)
zf = N.zeros_like(Er)
harmonic = N.zeros_like(Er)

cmap_plot="RdBu_r"
norm_plot = matplotlib.colors.Normalize(vmin = - 1.0, vmax = 1.0)
m = matplotlib.cm.ScalarMappable(cmap = cmap_plot, norm = norm_plot)
m.set_array([])

for patch in range(6):
    
    fcoord = (globals()["coord_" + sphere[patch] + "_to_sph"])
    theta, phi = fcoord(xi_grid, eta_grid)
    # Remove singularity in spherical coordinates
    phi[theta == 0.0] = 0.0
    phi[theta == N.pi] = 0.0
    xf[:, :, patch] = N.sin(theta) * N.cos(phi)
    yf[:, :, patch] = N.sin(theta) * N.sin(phi)
    zf[:, :, patch] = N.cos(theta)
    
    harmonic[:, :, patch] = N.cos(3.0 * phi) * N.sin(theta)**3

def plot_fields_sphere(it, field, res, az):
            
    fig_size=deffigsize(scale, aspect)
    fig, ax = P.subplots(subplot_kw={"projection": "3d"}, figsize = fig_size, facecolor = 'w')
    ax.view_init(elev = 10, azim = az)
    
    for patch in range(6):

        fcolors = m.to_rgba(field[NG:(Nxi + NG), NG:(Neta + NG), patch]) 
        sf = ax.plot_surface(xf[NG:(Nxi + NG), NG:(Neta + NG), patch], yf[NG:(Nxi + NG), NG:(Neta + NG), patch], \
                zf[NG:(Nxi + NG), NG:(Neta + NG), patch], \
                rstride = res, cstride = res, shade = False, \
                facecolors = fcolors, norm = norm_plot, zorder = 1)

    ax.plot_surface(x_s, y_s, z_s, rstride=1, cstride=1, shade=False, color = 'grey', zorder = 0)

    # contours_a = find_contours(Ar,  0.001)
    # contours_b = find_contours(Ar, -0.001)
    # contours = contours_a + contours_b
    # if (len(contours) > 0):
    #     for ind in range(len(contours)):
    #         xy = contours[ind]
    #         xi_c, eta_c = xy.T
            
    #         phi_c = xi_c * dxi + xi[0]           
    #         theta_c = 0.5 * N.pi - N.arctan(N.tan(eta_c * deta + eta[0]) * N.cos(phi_c))
    #         theta_c[phi_c > N.pi/4.0] = 0.5 * N.pi - N.arctan(N.tan(eta_c[phi_c > N.pi/4.0] * deta + eta[0]) * N.sin(phi_c[phi_c > N.pi/4.0]))
            
    #         x_c = r * N.sin(theta_c) * N.cos(phi_c)
    #         y_c = r * N.sin(theta_c) * N.sin(phi_c)
    #         z_c = r * N.cos(theta_c)
            
    #         lines = ax.plot(x_c, y_c, z_c, color = 'black', zorder = 10)

    figsave_png(fig, "snapshots/harmonic_" + str(it))
    
    # if (len(contours) > 0):
    #     lines.pop(0).remove()

    # sf.remove()    
    # ax.cla()

# Figure parameters
scale, aspect = 1.5, 0.7
vm = 0.5
ratio = 2.0

# Define figure
plot_fields_sphere(0, harmonic, 1, 45)

# iter = 0
# idump = 0

# Nt = 1500 # Number of iterations
# FDUMP = 10 # Dump frequency

# # Figure parameters
# scale, aspect = 2.0, 0.7
# vm = 0.2 * amp
# ratio = 2.0

# # Define figure
# fig_size=deffigsize(scale, aspect)
# fig, ax = P.subplots(subplot_kw={"projection": "3d"}, figsize = fig_size, facecolor = 'w')
# ax.view_init(elev=10, azim=45)
# ax.plot_surface(x_s, y_s, z_s, rstride=1, cstride=1, shade=False, color = 'grey', zorder = 0)

# # fig_size=deffigsize(scale, aspect)
# # fig = P.figure(1, facecolor='w')
# # ax = P.subplot(111)

# # Run the simulation
# for it in range(Nt):
#     if ((it % FDUMP) == 0):
#         plot_fields_sphere(idump, "Er", fig, ax, 2)
#         idump += 1

#     print(it)
#     iter += 1
    
#     contra_to_cov_E_a()
#     contra_to_cov_E_b()
    
#     contra_to_cov_E_edge_a()
#     contra_to_cov_E_edge_b()
    
#     push_B_a()
#     push_B_edge_a()
#     push_B_b()
    
#     BC_B_metal_a()
#     # BC_B_absorb_a()
#     BC_B_metal_b()
#     # BC_B_absorb_b()

#     contra_to_cov_B_a()
#     contra_to_cov_B_b()
    
#     contra_to_cov_B_edge_a()
#     contra_to_cov_B_edge_b()

#     push_E_a(it)
#     push_E_edge_b()
#     push_E_b(it)
    
#     BC_E_metal_a()
#     # BC_E_absorb_a()
#     BC_E_metal_b()
#     # BC_E_absorb_b()

#     compute_potential()



