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

class Sphere:
    A = 0
    B = 1
    C = 2
    D = 3
    N = 4
    S = 5

# Parameters
cfl = 0.4
Nxi = 64
Neta = 64
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

# divB = N.zeros((2 * Nxi + 2 * NG, Neta + 2 * NG, 6))
# divE = N.zeros((2 * Nxi + 2 * NG, Neta + 2 * NG, 6))
# Ar = N.zeros((2 * Nxi + 2 * NG, Neta + 2 * NG, 6))

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
             
    E1d[patch, i0:i1, j0:j1] = g11d[i0:i1, j0:j1, 1] * E1u[patch, i0:i1, j0:j1] + \
                         0.5 * g12d[i0:i1, j0:j1, 1] * (E2u[patch, i0:i1, j0:j1] + N.roll(N.roll(E2u, -1, axis = 1), 1, axis = 2)[patch, i0:i1, j0:j1])
    E2d[patch, i0:i1, j0:j1] = g22d[i0:i1, j0:j1, 2] * E2u[patch, i0:i1, j0:j1] + \
                         0.5 * g12d[i0:i1, j0:j1, 2] * (E1u[patch, i0:i1, j0:j1] + N.roll(N.roll(E1u, 1, axis = 1), -1, axis = 2)[patch, i0:i1, j0:j1])

def contra_to_cov_B(patch):

    i0, i1 = NG, Nxi + NG 
    j0, j1 = NG, Neta + NG
    
    B1d[patch, i0:i1, j0:j1] = g11d[i0:i1, j0:j1, 2] * B1u[patch, i0:i1, j0:j1] + \
                         0.5 * g12d[i0:i1, j0:j1, 2] * (B2u[patch, i0:i1, j0:j1] + N.roll(N.roll(B2u, 1, axis = 1), -1, axis = 2)[patch, i0:i1, j0:j1])            
    B2d[patch, i0:i1, j0:j1] = g22d[i0:i1, j0:j1, 1] * B2u[patch, i0:i1, j0:j1] + \
                         0.5 * g12d[i0:i1, j0:j1, 1] * (B1u[patch, i0:i1, j0:j1] + N.roll(N.roll(B1u, -1, axis = 1), 1, axis = 2)[patch, i0:i1, j0:j1])

def push_B(patch):
    
    i0, i1 = NG, Nxi + NG
    j0, j1 = NG, Neta + NG
    
    Br[patch, i0:i1, j0:j1]  -= ((N.roll(E2d, -1, axis = 1)[patch, i0:i1, j0:j1] - E2d[patch, i0:i1, j0:j1]) / dxi - \
                                 (N.roll(E1d, -1, axis = 2)[patch, i0:i1, j0:j1] - E1d[patch, i0:i1, j0:j1]) / deta) \
                                * dt / sqrt_det_g[i0:i1, j0:j1, 3]
    B1u[patch, i0:i1, j0:j1] -= ((N.roll(Er, -1, axis = 2)[patch, i0:i1, j0:j1] - Er[patch, i0:i1, j0:j1]) / deta) * dt / sqrt_det_g[i0:i1, j0:j1, 2]
    B2u[patch, i0:i1, j0:j1] += ((N.roll(Er, -1, axis = 1)[patch, i0:i1, j0:j1] - Er[patch, i0:i1, j0:j1]) / dxi)  * dt / sqrt_det_g[i0:i1, j0:j1, 1]

def push_E(it, patch):
    
    i0, i1 = NG, Nxi + NG
    j0, j1 = NG, Neta + NG
    
    Er[patch, i0:i1, j0:j1] += ((B2d[patch, i0:i1, j0:j1] - N.roll(B2d, 1, axis = 1)[patch, i0:i1, j0:j1]) / dxi - \
                                (B1d[patch, i0:i1, j0:j1] - N.roll(B1d, 1, axis = 2)[patch, i0:i1, j0:j1]) / deta) \
                               * dt / sqrt_det_g[i0:i1, j0:j1, 0] - 4.0 * N.pi * dt * Jr(it, patch, i0, i1, j0, j1) 
    E1u[patch, i0:i1, j0:j1] += ((Br[patch, i0:i1, j0:j1] - N.roll(Br, 1, axis = 2)[patch, i0:i1, j0:j1]) / deta) * dt / sqrt_det_g[i0:i1, j0:j1, 1]
    E2u[patch, i0:i1, j0:j1] -= ((Br[patch, i0:i1, j0:j1] - N.roll(Br, 1, axis = 1)[patch, i0:i1, j0:j1]) / dxi)  * dt / sqrt_det_g[i0:i1, j0:j1, 2]

    # if (patch == Sphere.D):
    #     print(Er[patch, 16,16], B2d[patch, 16,16], B2d[patch, 15,16], B1d[patch, 16,16],  B1d[patch, 16,17])

########
# Topology of the patches
########

topology = N.array([
    [   0, 'xx',    0,    0, 'yx',    0],
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
# from coord_transformations_flip import *

def transform_coords(patch0, patch1, xi0, eta0):
    fcoord0 = (globals()["coord_" + sphere[patch0] + "_to_sph"])
    fcoord1 = (globals()["coord_sph_to_" + sphere[patch1]])
    return fcoord1(*fcoord0(xi0, eta0))

########
# Generic vector transformation 
########

from vec_transformations import *
# from vec_transformations_flip import *

def transform_vect(patch0, patch1, xi0, eta0, vxi0, veta0):
    fcoord0 = (globals()["coord_" + sphere[patch0] + "_to_sph"])
    theta0, phi0 = fcoord0(xi0, eta0)
    fvec0 = (globals()["vec_" + sphere[patch0] + "_to_sph"])
    fvec1 = (globals()["vec_sph_to_" + sphere[patch1]])
    return fvec1(theta0, phi0, *fvec0(xi0, eta0, vxi0, veta0))

########
# Linear form transformations
########

from form_transformations import *
# from form_transformations_flip import *

def transform_form(patch0, patch1, xi0, eta0, vxi0, veta0):
    fcoord0 = (globals()["coord_" + sphere[patch0] + "_to_sph"])
    theta0, phi0 = fcoord0(xi0, eta0)
    fform0 = (globals()["form_" + sphere[patch0] + "_to_sph"])
    fform1 = (globals()["form_sph_to_" + sphere[patch1]])
    return fform1(theta0, phi0, *fform0(xi0, eta0, vxi0, veta0))

########
# Interpolation routines
########

def interp(arr, x0, x):
    
    i0 = N.argmin(N.abs(x - x0))
    if (x0 > x[i0]):
        i_min = i0
        i_max = i0 + 1
    else:
        i_min = i0 - 1
        i_max = i0

    if (i_max == Nxi + 2 * NG):
        return arr[i_min]
    else:
        # Define weight coefficients for interpolation
        w1 = (x0 - x[i_min]) / (x[i_max] - x[i_min])
        w2 = (x[i_max] - x0) / (x[i_max] - x[i_min])

        return (w1 * arr[i_max] + w2 * arr[i_min]) / (w1 + w2)

########
# Communication at between two patches
########

# patch0 has the open boundary
# patch1 has the closed boundary 
          
def communicate_E_patch(patch0, patch1):
        
    top = topology[patch0, patch1]
    if (top == 0):
        return
    elif (top == 'xx'):

        for k in range(NG, Neta + NG):         
            
            #########
            # Communicate fields from xi right edge of patch0 to xi left edge patch1
            ########
                
            i0 = Nxi + NG - 1 # Last active cell of xi edge of patch0                   
            i1 = NG - 1 # First ghost cell of xi edge of patch1
            communicate_E_local(patch0, patch1, i0, i1, k, "a") 
                
            #########
            # Communicate fields from xi left edge of patch1 to xi right edge of patch0
            ########

            i0 = Nxi + NG # Last ghost cell of xi edge of patch0             
            i1 = NG  # First active cell of xi edge of patch1
            communicate_E_local(patch1, patch0, i1, i0, k, "a") 
                
    elif (top == 'xy'):

        for k in range(NG, Neta + NG): 

            #########
            # Communicate fields from xi right edge of patch0 to eta left edge of patch1
            ########                

            i0 = Nxi + NG - 1 # Last active cell of xi edge of patch0                   
            j1 = NG - 1 # First ghost cell on eta edge of patch1
            communicate_E_local(patch0, patch1, i0, k, j1, "a") 
                
    #         #########
    #         # Communicate fields from eta left edge of patch1 to xi right edge of patch0
    #         ########                

            i0 = Nxi + NG # Last ghost cell of xi edge of patch0             
            j1 = NG  # First active cell of eta edge of patch1
            communicate_E_local(patch1, patch0, j1, i0, k, "b") 

    elif (top == 'yy'):

        for k in range(NG, Nxi + NG):

            #########
            # Communicate fields from eta top edge of patch0 to eta bottom edge of patch1
            ########                

            j0 = Neta + NG - 1 # Last active cell of eta edge of patch0
            j1 = NG - 1 # First ghost cell of eta edge of patch1
            communicate_E_local(patch0, patch1, j0, k, j1, "b") 

            #########
            # Communicate fields from eta bottom edge of patch1 to eta top edge of patch0
            ########                

            j0 = Neta + NG # Last ghost cell of eta edge of patch0
            j1 = NG # First active cell of eta edge of patch1
            communicate_E_local(patch1, patch0, j1, k, j0, "b") 

    elif (top == 'yx'):

        for k in range(NG, Nxi + NG):

    #         #########
    #         # Communicate fields from eta top edge of patch0 to xi bottom edge of patch1
    #         ########                

            j0 = Neta + NG - 1 # Last active cell of eta edge of patch0
            i1 = NG - 1 # First ghost cell on xi edge of patch1
            communicate_E_local(patch0, patch1, j0, i1, k, "b") 

    #         #########
    #         # Communicate fields from xi bottom edge of patch1 to eta top edge of patch0
    #         ########                

            j0 = Neta + NG # Last ghost cell of eta edge of patch0
            i1 = NG # First active cell of xi edge of patch1
            communicate_E_local(patch1, patch0, i1, k, j0, "a")


def communicate_B_patch(patch0, patch1):
        
    top = topology[patch0, patch1]
    if (top == 0):
        return
    elif (top == 'xx'):

        for k in range(NG, Neta + NG):         
            
            #########
            # Communicate fields from xi right edge of patch0 to xi left edge patch1
            ########
                
            i0 = Nxi + NG - 1 # Last active cell of xi edge of patch0                   
            i1 = NG - 1 # First ghost cell of xi edge of patch1
            communicate_B_local(patch0, patch1, i0, i1, k, "a") 
                
            #########
            # Communicate fields from xi left edge of patch1 to xi right edge of patch0
            ########

            i0 = Nxi + NG # Last ghost cell of xi edge of patch0             
            i1 = NG  # First active cell of xi edge of patch1
            communicate_B_local(patch1, patch0, i1, i0, k, "a") 
                
    elif (top == 'xy'):

        for k in range(NG, Neta + NG): 

            #########
            # Communicate fields from xi right edge of patch0 to eta left edge of patch1
            ########                

            i0 = Nxi + NG - 1 # Last active cell of xi edge of patch0                   
            j1 = NG - 1 # First ghost cell on eta edge of patch1
            communicate_B_local(patch0, patch1, i0, k, j1, "a") 

    #         #########
    #         # Communicate fields from eta left edge of patch1 to xi right edge of patch0
    #         ########                

            i0 = Nxi + NG # Last ghost cell of xi edge of patch0             
            j1 = NG  # First active cell of eta edge of patch1
            communicate_B_local(patch1, patch0, j1, i0, k, "b") 

    elif (top == 'yy'):

        for k in range(NG, Nxi + NG):

            #########
            # Communicate fields from eta top edge of patch0 to eta bottom edge of patch1
            ########                

            j0 = Neta + NG - 1 # Last active cell of eta edge of patch0
            j1 = NG - 1 # First ghost cell of eta edge of patch1
            communicate_B_local(patch0, patch1, j0, k, j1, "b") 

            #########
            # Communicate fields from eta bottom edge of patch1 to eta top edge of patch0
            ########                

            j0 = Neta + NG # Last ghost cell of eta edge of patch0
            j1 = NG # First active cell of eta edge of patch1
            communicate_B_local(patch1, patch0, j1, k, j0, "b") 

    elif (top == 'yx'):

        for k in range(NG, Nxi + NG):

            #########
            # Communicate fields from eta top edge of patch0 to xi bottom edge of patch1
            ########                

            j0 = Neta + NG - 1 # Last active cell of eta edge of patch0
            i1 = NG - 1 # First ghost cell on xi edge of patch1
            communicate_B_local(patch0, patch1, j0, i1, k, "b") 

            #########
            # Communicate fields from xi bottom edge of patch1 to eta top edge of patch0
            ########                

            j0 = Neta + NG # Last ghost cell of eta edge of patch0
            i1 = NG # First active cell of xi edge of patch1
            communicate_B_local(patch1, patch0, i1, k, j0, "a")

#######
# Communication of a single value
########
 
# index0 is the index of the cell that is being communicated
# index1, index2 label the location of the point where the field is computed

def communicate_E_local(patch0, patch1, index0, index1, index2, loc):
    
    if (loc == "a"):
        ert  =  Er[patch0, index0, :]
        e1ut = E1u[patch0, index0, :]
        e2ut = E2u[patch0, index0, :]
        e1dt = E1d[patch0, index0, :]
        e2dt = E2d[patch0, index0, :]
    elif (loc == "b"):
        ert  =  Er[patch0, :, index0]
        e1ut = E1u[patch0, :, index0]
        e2ut = E2u[patch0, :, index0]
        e1dt = E1d[patch0, :, index0]
        e2dt = E2d[patch0, :, index0]
    
    # Interpolate E^r       
    xi0, eta0 = transform_coords(patch1, patch0, xi[index1], eta[index2])
    if (loc == "a"):
        tab0 = eta0
        tab = eta
    elif (loc == "b"):
        tab0 = xi0
        tab = xi
    Er[patch1, index1, index2] = interp(ert, tab0, tab)

    # Interpolate E^xi and E_xi     
    xi0, eta0 = transform_coords(patch1, patch0, xi_yee[index1], eta[index2])
    if (loc == "a"):
        tab0 = eta0
        tab = eta
    elif (loc == "b"):
        tab0 = xi0
        tab = xi 
    E1u_int = interp(e1ut, tab0, tab)
    E1d_int = interp(e1dt, tab0, tab)

    # Interpolate E^eta and E_eta     
    xi0, eta0 = transform_coords(patch1, patch0, xi[index1], eta_yee[index2])
    if (loc == "a"):
        tab0 = eta0
        tab = eta
    elif (loc == "b"):
        tab0 = xi0
        tab = xi
    E2u_int = interp(e2ut, tab0, tab)
    E2d_int = interp(e2dt, tab0, tab)

    xi0, eta0 = transform_coords(patch1, patch0, xi[index1], eta[index2])
    
    # Convert from patch0 to patch1 coordinates
    # E1u[patch1, index1, index2], E2u[patch1, index1, index2] = transform_vect(patch0, patch1, xi[index1], eta[index2], E1u_int, E2u_int)
    # E1d[patch1, index1, index2], E2d[patch1, index1, index2] = transform_form(patch0, patch1, xi[index1], eta[index2], E1d_int, E2d_int)
    E1u[patch1, index1, index2], E2u[patch1, index1, index2] = transform_vect(patch0, patch1, xi0, eta0, E1u_int, E2u_int)
    E1d[patch1, index1, index2], E2d[patch1, index1, index2] = transform_form(patch0, patch1, xi0, eta0, E1d_int, E2d_int)

def communicate_B_local(patch0, patch1, index0, index1, index2, loc):
    
    if (loc == "a"):
        brt  =  Br[patch0, index0, :]
        b1ut = B1u[patch0, index0, :]
        b2ut = B2u[patch0, index0, :]
        b1dt = B1d[patch0, index0, :]
        b2dt = B2d[patch0, index0, :]
    elif (loc == "b"):
        brt  =  Br[patch0, :, index0]
        b1ut = B1u[patch0, :, index0]
        b2ut = B2u[patch0, :, index0]
        b1dt = B1d[patch0, :, index0]
        b2dt = B2d[patch0, :, index0]
    
    # Interpolate B^r       
    xi0, eta0 = transform_coords(patch1, patch0, xi_yee[index1], eta_yee[index2])
    if (loc == "a"):
        tab0 = eta0
        tab = eta
    elif (loc == "b"):
        tab0 = xi0
        tab = xi
    Br[patch1, index1, index2] = interp(brt, tab0, tab)

    # Interpolate B^xi and B_xi     
    xi0, eta0 = transform_coords(patch1, patch0, xi[index1], eta_yee[index2])
    if (loc == "a"):
        tab0 = eta0
        tab = eta
    elif (loc == "b"):
        tab0 = xi0
        tab = xi 
    B1u_int = interp(b1ut, tab0, tab)
    B1d_int = interp(b1dt, tab0, tab)

    # Interpolate B^eta and B_eta     
    xi0, eta0 = transform_coords(patch1, patch0, xi_yee[index1], eta[index2])
    if (loc == "a"):
        tab0 = eta0
        tab = eta
    elif (loc == "b"):
        tab0 = xi0
        tab = xi
    B2u_int = interp(b2ut, tab0, tab)
    B2d_int = interp(b2dt, tab0, tab)

    xi0, eta0 = transform_coords(patch1, patch0, xi[index1], eta[index2])

    # Convert from patch0 to patch1 coordinates
    # B1u[patch1, index1, index2], B2u[patch1, index1, index2] = transform_vect(patch0, patch1, xi[index1], eta[index2], B1u_int, B2u_int)
    # B1d[patch1, index1, index2], B2d[patch1, index1, index2] = transform_form(patch0, patch1, xi[index1], eta[index2], B1d_int, B2d_int)

    B1u[patch1, index1, index2], B2u[patch1, index1, index2] = transform_vect(patch0, patch1, xi0, eta0, B1u_int, B2u_int)
    B1d[patch1, index1, index2], B2d[patch1, index1, index2] = transform_form(patch0, patch1, xi0, eta0, B1d_int, B2d_int)

    # if patch0==Sphere.S:
    #     print(B1u_int, B2u_int, B1u[patch1, index1, index2], B2u[patch1, index1, index2], xi0, eta0)

########
# Plotting fields on an unfolded sphere
########

def plot_fields_unfolded(it, field, fig, ax, vm):

    tab = (globals()[field])

    ax.pcolormesh(xi_grid[NG:(Nxi + NG), NG:(Neta + NG)], eta_grid[NG:(Nxi + NG), NG:(Neta + NG)], tab[Sphere.A, NG:(Nxi + NG), NG:(Neta + NG)], cmap = "RdBu_r", vmin = - vm, vmax = vm)
    ax.pcolormesh(xi_grid[NG:(Nxi + NG), NG:(Neta + NG)] + N.pi / 2.0, eta_grid[NG:(Nxi + NG), NG:(Neta + NG)], tab[Sphere.B, NG:(Nxi + NG), NG:(Neta + NG)], cmap = "RdBu_r", vmin = - vm, vmax = vm)
    ax.pcolormesh(xi_grid[NG:(Nxi + NG), NG:(Neta + NG)] + N.pi, eta_grid[NG:(Nxi + NG), NG:(Neta + NG)], tab[Sphere.C, NG:(Nxi + NG), NG:(Neta + NG)], cmap = "RdBu_r", vmin = - vm, vmax = vm)
    ax.pcolormesh(xi_grid[NG:(Nxi + NG), NG:(Neta + NG)] - N.pi / 2.0, eta_grid[NG:(Nxi + NG), NG:(Neta + NG)], tab[Sphere.D, NG:(Nxi + NG), NG:(Neta + NG)], cmap = "RdBu_r", vmin = - vm, vmax = vm)
    ax.pcolormesh(xi_grid[NG:(Nxi + NG), NG:(Neta + NG)], eta_grid[NG:(Nxi + NG), NG:(Neta + NG)] + N.pi / 2.0, tab[Sphere.N, NG:(Nxi + NG), NG:(Neta + NG)], cmap = "RdBu_r", vmin = - vm, vmax = vm)
    ax.pcolormesh(xi_grid[NG:(Nxi + NG), NG:(Neta + NG)], eta_grid[NG:(Nxi + NG), NG:(Neta + NG)] - N.pi / 2.0, tab[Sphere.S, NG:(Nxi + NG), NG:(Neta + NG)], cmap = "RdBu_r", vmin = - vm, vmax = vm)
    
    figsave_png(fig, "snapshots/" + field + "_" + str(it))

    ax.cla()

########
# Plotting fields on a sphere
########

# Define background sphere
phi_s = N.linspace(0, N.pi, 2*50)
theta_s = N.linspace(0, 2*N.pi, 2*50)
theta_s, phi_s = N.meshgrid(theta_s, phi_s)
x_s = 0.99 * N.sin(phi_s) * N.cos(theta_s)
y_s = 0.99 * N.sin(phi_s) * N.sin(theta_s)
z_s = 0.99 * N.cos(phi_s)

xf = N.zeros_like(Er)
yf = N.zeros_like(Er)
zf = N.zeros_like(Er)
harmonic = N.zeros_like(Er)

th0 = N.zeros_like(xi_grid)
ph0 = N.zeros_like(xi_grid)

cmap_plot="RdBu_r"
norm_plot = matplotlib.colors.Normalize(vmin = - 1.0, vmax = 1.0)
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
    
    harmonic[patch, :, :] = N.cos(3.0 * ph0) * N.sin(th0)**3

def plot_fields_sphere(it, fig, ax, field, res, az):

    tab = (globals()[field])
    ax.view_init(elev = 10, azim = az)
    
    # for patch in range(6):
    for patch in [Sphere.A, Sphere.B, Sphere.N]:

        fcolors = m.to_rgba(tab[patch, NG:(Nxi + NG), NG:(Neta + NG)]) 
        sf = ax.plot_surface(xf[patch, NG:(Nxi + NG), NG:(Neta + NG)], yf[patch, NG:(Nxi + NG), NG:(Neta + NG)], \
                zf[patch, NG:(Nxi + NG), NG:(Neta + NG)], \
                rstride = res, cstride = res, shade = False, \
                facecolors = fcolors, norm = norm_plot, zorder = 1)

    ax.plot_surface(x_s, y_s, z_s, rstride=1, cstride=1, shade=False, color = 'grey', zorder = 0)

# Need to compute Ar potential before plotting this

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
            
            # lines = ax.plot(x_c, y_c, z_c, color = 'black', zorder = 10)

    figsave_png(fig, "snapshots/" + field + "_" + str(it))

    sf.remove()    
    
    # if (len(contours) > 0):
    #     lines.pop(0).remove()

########
# Source current
########

theta0, phi0 = 90.0 / 360.0 * 2.0 * N.pi, 0.0 / 360.0 * 2.0 * N.pi # Center of the wave packet !60
x0 = N.sin(theta0) * N.cos(phi0)
y0 = N.sin(theta0) * N.sin(phi0)
z0 = N.cos(theta0)

def shape_packet(x, y, z, width):
    return N.exp(- y * y / (width * width)) * N.exp(- x * x / (width * width)) * N.exp(- z * z / (width * width)) 

w = 0.1 # Radius of wave packet
omega = 20.0 # Frequency of current
J0 = 1.0 # Current amplitude
p0 = Sphere.A # Patch location of antenna

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
    return Jr_tot[patch, i0:i1, j0:j1] * N.sin(omega * dt * it)

########
# Initialization
########

for p in range(6):
    pass # All fields to zero

iter = 0
idump = 0

Nt = 1500 # Number of iterations
FDUMP = 10 # Dump frequency

# Figure parameters
scale, aspect = 2.0, 0.7
vm = 0.2
ratio = 2.0

# Define figure

# fig_size=deffigsize(scale, aspect)
# fig, ax = P.subplots(subplot_kw={"projection": "3d"}, figsize = fig_size, facecolor = 'w')
# ax.view_init(elev=10, azim=45)
# ax.plot_surface(x_s, y_s, z_s, rstride=1, cstride=1, shade=False, color = 'grey', zorder = 0)

fig_size=deffigsize(scale, aspect)
fig = P.figure(1, facecolor='w')
ax = P.subplot(111)

########
# Main routine
########

for it in range(Nt):
    if ((it % FDUMP) == 0):
        # plot_fields_unfolded(idump, "Er", fig, ax, 0.5)
        plot_fields_unfolded(idump, "B1u", fig, ax, 0.5)
        plot_fields_unfolded(idump, "B2u", fig, ax, 0.5)

        idump += 1

    print(it)
    iter += 1
    
    for p in range(6):
        contra_to_cov_E(p)
        push_B(p)

    for p0 in range(6):
        for p1 in range(6):
            communicate_B_patch(p0, p1)

    for p in range(6):
        contra_to_cov_B(p)
        push_E(it, p)

    for p0 in range(6):
        for p1 in range(6):
            communicate_E_patch(p0, p1)
            
#     compute_potential()



