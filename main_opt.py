# Import modules
import numpy as N
import matplotlib.pyplot as P
import matplotlib
import time
import scipy.integrate as spi
from skimage.measure import find_contours
from math import *
import sys
from tqdm import tqdm
from scipy import interpolate

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
cfl = 0.7
Nxi = 128
Neta = 128
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

Ar = N.zeros((6, Nxi + 2 * NG, Neta + 2 * NG))
# divB = N.zeros((6, Nxi + 2 * NG, Neta + 2 * NG))
# divE = N.zeros((6, Nxi + 2 * NG, Neta + 2 * NG))

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

# Compute A_r
def push_potential(patch):

    i0, i1 = NG, Nxi + NG
    j0, j1 = NG, Neta + NG
    
    # Ar[Sphere.A, :, NG - 1] = Ar[Sphere.S, :, Neta + NG]
    # for j in range(NG + 1, Neta + NG): 
    #     Ar[Sphere.A, :(Nxi + NG), j] = spi.simps(sqrt_det_g[:(Nxi + NG), NG:j, 2] * B1u[Sphere.A, :(Nxi + NG), NG:j], axis = 2, x = eta[NG:j], dx = deta)               
    
    Ar[patch, i0:i1, j0:j1] -= dt * Er[patch, i0:i1, j0:j1]

########
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

########
# Generic coordinate transformation
########

from coord_transformations_flip import *

def transform_coords(patch0, patch1, xi0, eta0):
    fcoord0 = (globals()["coord_" + sphere[patch0] + "_to_sph"])
    fcoord1 = (globals()["coord_sph_to_" + sphere[patch1]])
    return fcoord1(*fcoord0(xi0, eta0))

########
# Generic vector transformation 
########

from vec_transformations_flip import *

def transform_vect(patch0, patch1, xi0, eta0, vxi0, veta0):
    fcoord0 = (globals()["coord_" + sphere[patch0] + "_to_sph"])
    theta0, phi0 = fcoord0(xi0, eta0)
    fvec0 = (globals()["vec_" + sphere[patch0] + "_to_sph"])
    fvec1 = (globals()["vec_sph_to_" + sphere[patch1]])
    return fvec1(theta0, phi0, *fvec0(xi0, eta0, vxi0, veta0))

########
# Linear form transformations
########

from form_transformations_flip import *

def transform_form(patch0, patch1, xi0, eta0, vxi0, veta0):
    fcoord0 = (globals()["coord_" + sphere[patch0] + "_to_sph"])
    theta0, phi0 = fcoord0(xi0, eta0)
    fform0 = (globals()["form_" + sphere[patch0] + "_to_sph"])
    fform1 = (globals()["form_sph_to_" + sphere[patch1]])
    return fform1(theta0, phi0, *fform0(xi0, eta0, vxi0, veta0))

########
# Interpolation routine
########

def interp(arr, x0, x):
    f = interpolate.interp1d(x, arr, kind='linear', fill_value=(0,0), bounds_error=False)
    return f(x0)

########
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

########
# Communication of covariant or contravariant E
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
    if (top == 0):
        return
    elif (top == 'xx'):

        k = range(NG, Neta + NG)   

        #########
        # Communicate fields from xi right edge of patch0 to xi left edge patch1
        ########
        
        i0 = Nxi + NG - 1 # Last active cell of xi edge of patch0                   
        i1 = NG - 1 # First ghost cell of xi edge of patch1   

        ert =  Er[patch0, i0, :]
        e1t = field1[patch0, i0, :]
        e2t = field2[patch0, i0, :]
        # e2t = 0.5 * (field2[patch0, i0, :] + N.roll(field2[patch0, i0 + 1, :], 1))

        # Interpolate E^xi    
        xi0, eta0 = transform_coords(patch1, patch0, xi_yee[i1], eta[k])
        E1_int = interp(e1t, eta0, eta)

        # Interpolate E^eta     
        xi0, eta0 = transform_coords(patch1, patch0, xi[i1], eta_yee[k])
        E2_int = interp(e2t, eta0, eta_yee)

        # Convert from patch0 to patch1 coordinates
        xi0, eta0 = transform_coords(patch1, patch0, xi_yee[i1], eta[k])
        field1[patch1, i1, k]= transform(patch0, patch1, xi0, eta0, E1_int, E2_int)[0]

        # field2[patch1, i1, k]= transform(patch0, patch1, xi0, eta0, E1_int, E2_int)[1]
        
        #########
        # Communicate fields from xi left edge of patch1 to xi right edge of patch0
        ########
        
        i0 = Nxi + NG # Last ghost cell of xi edge of patch0             
        i1 = NG  # First active cell of xi edge of patch1   

        ert =  Er[patch1, i1, :]
        e1t = field1[patch1, i1, :]
        # e1t = 0.5 * (field1[patch1, i1, :] + N.roll(field1[patch1, i1 - 1, :], -1))
        e2t = field2[patch1, i1, :]

        if (typ == "vect"):
            # Interpolate E^r       
            xi0, eta0 = transform_coords(patch0, patch1, xi[i0], eta[k])
            Er[patch0, i0, k] = interp(ert, eta0, eta)

        # Interpolate E^xi    
        xi0, eta0 = transform_coords(patch0, patch1, xi_yee[i0], eta[k])
        E1_int = interp(e1t, eta0, eta)

        # Interpolate E^eta     
        xi0, eta0 = transform_coords(patch0, patch1, xi[i0], eta_yee[k])
        E2_int = interp(e2t, eta0, eta_yee)

        # Convert from patch0 to patch1 coordinates
        xi0, eta0 = transform_coords(patch0, patch1, xi[i0], eta_yee[k])
        field2[patch0, i0, k] = transform(patch1, patch0, xi0, eta0, E1_int, E2_int)[1]

        # field1[patch0, i0, k] = transform(patch1, patch0, xi0, eta0, E1_int, E2_int)[0]
    
    elif (top == 'xy'):

        k = range(NG, Neta + NG)   

        # #########
        # # Communicate fields from xi right edge of patch0 to eta left edge of patch1
        # ########    
        
        i0 = Nxi + NG - 1 # Last active cell of xi edge of patch0                   
        j1 = NG - 1 # First ghost cell on eta edge of patch1 

        ert =  Er[patch0, i0, :]
        e1t = field1[patch0, i0, :]
        e2t = field2[patch0, i0, :]
        # e2t = 0.5 * (field2[patch0, i0, :] + N.roll(field2[patch0, i0 + 1, :], 1))

        # Interpolate E^xi    
        xi0, eta0 = transform_coords(patch1, patch0, xi_yee[k], eta[j1])
        E1_int = interp(e1t, eta0, eta)

        # Interpolate E^eta     
        xi0, eta0 = transform_coords(patch1, patch0, xi[k], eta_yee[j1])
        E2_int = interp(e2t, eta0, eta_yee)

        # Convert from patch0 to patch1 coordinates
        xi0, eta0 = transform_coords(patch1, patch0, xi_yee[k], eta[j1])
        field2[patch1, k, j1] = transform(patch0, patch1, xi0, eta0, E1_int, E2_int)[1]

        # field1[patch1, k, j1] = transform(patch0, patch1, xi0, eta0, E1_int, E2_int)[0]
                
        #########
        # Communicate fields from eta left edge of patch1 to xi right edge of patch0
        ########     
        
        i0 = Nxi + NG # Last ghost cell of xi edge of patch0             
        j1 = NG  # First active cell of eta edge of patch1   

        ert =  Er[patch1, :, j1]
        e1t = field1[patch1, :, j1]
        e2t = field2[patch1, :, j1]
        # e2t = 0.5 * (field2[patch1, :, j1] + N.roll(field2[patch1, :, j1 - 1], - 1))

        if (typ == "vect"):
            # Interpolate E^r       
            xi0, eta0 = transform_coords(patch0, patch1, xi[i0], eta[k])
            Er[patch0, i0, k] = interp(ert, xi0, xi)

        # Interpolate E^xi    
        xi0, eta0 = transform_coords(patch0, patch1, xi_yee[i0], eta[k])
        E1_int = interp(e1t, xi0, xi_yee)

        # Interpolate E^eta     
        xi0, eta0 = transform_coords(patch0, patch1, xi[i0], eta_yee[k])
        E2_int = interp(e2t, xi0, xi)

        # Convert from patch0 to patch1 coordinates
        xi0, eta0 = transform_coords(patch0, patch1, xi_yee[i0], eta[k])
        field2[patch0, i0, k] = transform(patch1, patch0, xi0, eta0, E1_int, E2_int)[1]

        # field1[patch0, i0, k] = transform(patch1, patch0, xi0, eta0, E1_int, E2_int)[0]
      
    elif (top == 'yy'):

        k = range(NG, Nxi + NG)       

        #########
        # Communicate fields from eta top edge of patch0 to eta bottom edge of patch1
        ########                
                            
        j0 = Neta + NG - 1 # Last active cell of eta edge of patch0
        j1 = NG - 1 # First ghost cell of eta edge of patch1

        ert =  Er[patch0, :, j0]
        e1t = field1[patch0, :, j0]
        # e1t = 0.5 * (field1[patch0, :, j0] + N.roll(field1[patch0, :, j0 + 1], 1))
        e2t = field2[patch0, :, j0]

        # Interpolate E^xi    
        xi0, eta0 = transform_coords(patch1, patch0, xi_yee[k], eta[j1])
        E1_int = interp(e1t, xi0, xi_yee)

        # Interpolate E^eta     
        xi0, eta0 = transform_coords(patch1, patch0, xi[k], eta_yee[j1])
        E2_int = interp(e2t, xi0, xi)

        # Convert from patch0 to patch1 coordinates
        xi0, eta0 = transform_coords(patch1, patch0, xi[k], eta_yee[j1])
        field2[patch1, k, j1] = transform(patch0, patch1, xi0, eta0, E1_int, E2_int)[1]

        # field1[patch1, k, j1] = transform(patch0, patch1, xi0, eta0, E1_int, E2_int)[0]
        
        #########
        # Communicate fields from eta bottom edge of patch1 to eta top edge of patch0
        ########   
          
        j0 = Neta + NG # Last ghost cell of eta edge of patch0
        j1 = NG # First active cell of eta edge of patch1

        ert =  Er[patch1, :, j1]
        e1t = field1[patch1, :, j1]
        e2t = field2[patch1, :, j1]
        # e2t = 0.5 * (field2[patch1, :, j1] + N.roll(field2[patch1, :, j1 - 1], -1))
    
        if (typ == "vect"):
            # Interpolate E^r       
            xi0, eta0 = transform_coords(patch0, patch1, xi[k], eta[j0])
            Er[patch0, k, j0] = interp(ert, xi0, xi)

        # Interpolate E^xi    
        xi0, eta0 = transform_coords(patch0, patch1, xi_yee[k], eta[j0])
        E1_int = interp(e1t, xi0, xi_yee)

        # Interpolate E^eta     
        xi0, eta0 = transform_coords(patch0, patch1, xi[k], eta_yee[j0])
        E2_int = interp(e2t, xi0, xi)

        # Convert from patch0 to patch1 coordinates
        xi0, eta0 = transform_coords(patch0, patch1, xi_yee[k], eta[j0])
        field1[patch0, k, j0] = transform(patch1, patch0, xi0, eta0, E1_int, E2_int)[0]

        # field2[patch0, k, j0] = transform(patch1, patch0, xi0, eta0, E1_int, E2_int)[1]

    elif (top == 'yx'):

        k = range(NG, Nxi + NG)       
            
        #########
        # Communicate fields from eta top edge of patch0 to xi bottom edge of patch1
        ########     
                
        j0 = Neta + NG - 1 # Last active cell of eta edge of patch0
        i1 = NG - 1 # First ghost cell on xi edge of patch1

        ert =  Er[patch0, :, j0]
        e1t = field1[patch0, :, j0]
        # e1t = 0.5 * (field1[patch0, :, j0] + N.roll(field1[patch0, :, j0 + 1], 1))
        e2t = field2[patch0, :, j0]

        # Interpolate E^xi    
        xi0, eta0 = transform_coords(patch1, patch0, xi_yee[i1], eta[k])
        E1_int = interp(e1t, xi0, xi_yee)

        # Interpolate E^eta     
        xi0, eta0 = transform_coords(patch1, patch0, xi[i1], eta_yee[k])
        E2_int = interp(e2t, xi0, xi)

        # Convert from patch0 to patch1 coordinates
        xi0, eta0 = transform_coords(patch1, patch0, xi[i1], eta_yee[k])
        field1[patch1, i1, k] = transform(patch0, patch1, xi0, eta0, E1_int, E2_int)[0]
        
        # field2[patch1, i1, k] = transform(patch0, patch1, xi0, eta0, E1_int, E2_int)[1]

        #########
        # Communicate fields from xi bottom edge of patch1 to eta top edge of patch0
        ########   
        
        j0 = Neta + NG # Last ghost cell of eta edge of patch0
        i1 = NG # First active cell of xi edge of patch1

        ert =  Er[patch1, i1, :]
        e1t = field1[patch1, i1, :]
        # e1t = 0.5 * (field1[patch1, i1, :] + N.roll(field1[patch1, i1 - 1, :], -1))
        e2t = field2[patch1, i1, :]

        if (typ == "vect"):
            # Interpolate E^r       
            xi0, eta0 = transform_coords(patch0, patch1, xi[k], eta[j0])
            Er[patch0, k, j0] = interp(ert, eta0, eta)

        # Interpolate E^xi    
        xi0, eta0 = transform_coords(patch0, patch1, xi_yee[k], eta[j0])
        E1_int = interp(e1t, eta0, eta)

        # Interpolate E^eta     
        xi0, eta0 = transform_coords(patch0, patch1, xi[k], eta_yee[j0])
        E2_int = interp(e2t, eta0, eta_yee)

        # Convert from patch0 to patch1 coordinates
        xi0, eta0 = transform_coords(patch0, patch1, xi[k], eta_yee[j0])
        field1[patch0, k, j0] = transform(patch1, patch0, xi0, eta0, E1_int, E2_int)[0]

        # field2[patch0, k, j0] = transform(patch1, patch0, xi0, eta0, E1_int, E2_int)[1]

########
# Communication of covariant or contravariant B
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
    if (top == 0):
        return
    elif (top == 'xx'):

        k = range(NG, Neta + NG)   

        #########
        # Communicate fields from xi right edge of patch0 to xi left edge patch1
        ########
        
        i0 = Nxi + NG - 1 # Last active cell of xi edge of patch0                   
        i1 = NG - 1 # First ghost cell of xi edge of patch1   

        brt =  Br[patch0, i0, :]
        b1t = field1[patch0, i0, :]
        # b1t = 0.5 * (field1[patch0, i0, :] + N.roll(field1[patch0, i0 + 1, :], 1))
        b2t = field2[patch0, i0, :]

        if (typ == "vect"):
            # Interpolate B^r       
            xi0, eta0 = transform_coords(patch1, patch0, xi_yee[i1], eta_yee[k])
            Br[patch1, i1, k] = interp(brt, eta0, eta_yee)

        # Interpolate B^xi     
        xi0, eta0 = transform_coords(patch1, patch0, xi[i1], eta_yee[k])
        B1_int = interp(b1t, eta0, eta_yee)

        # Interpolate B^eta     
        xi0, eta0 = transform_coords(patch1, patch0, xi_yee[i1], eta[k])
        B2_int = interp(b2t, eta0, eta)

        # Convert from patch0 to patch1 coordinates
        xi0, eta0 = transform_coords(patch1, patch0, xi_yee[i1], eta[k])
        field2[patch1, i1, k] = transform(patch0, patch1, xi0, eta0, B1_int, B2_int)[1]

        # field1[patch1, i1, k] = transform(patch0, patch1, xi0, eta0, B1_int, B2_int)[0]
        
        #########
        # Communicate fields from xi left edge of patch1 to xi right edge of patch0
        ########
        
        i0 = Nxi + NG # Last ghost cell of xi edge of patch0             
        i1 = NG  # First active cell of xi edge of patch1   

        brt = Br[patch1, i1, :]
        b1t = field1[patch1, i1, :]
        b2t = field2[patch1, i1, :]
        b2t = 0.5 * (field2[patch1, i1, :] + N.roll(field2[patch1, i1 - 1, :], -1))

        # Interpolate B^xi     
        xi0, eta0 = transform_coords(patch0, patch1, xi[i0], eta_yee[k])
        B1_int = interp(b1t, eta0, eta_yee)

        # Interpolate B^eta     
        xi0, eta0 = transform_coords(patch0, patch1, xi_yee[i0], eta[k])
        B2_int = interp(b2t, eta0, eta)

        # Convert from patch0 to patch1 coordinates
        xi0, eta0 = transform_coords(patch0, patch1, xi[i0], eta_yee[k])
        field1[patch0, i0, k] = transform(patch1, patch0, xi0, eta0, B1_int, B2_int)[0]

        # field2[patch0, i0, k] = transform(patch1, patch0, xi0, eta0, B1_int, B2_int)[1]
    
    elif (top == 'xy'):

        k = range(NG, Neta + NG)   

        # #########
        # # Communicate fields from xi right edge of patch0 to eta left edge of patch1
        # ########    
        
        i0 = Nxi + NG - 1 # Last active cell of xi edge of patch0                   
        j1 = NG - 1 # First ghost cell on eta edge of patch1 

        brt = Br[patch0, i0, :]
        b1t = field1[patch0, i0, :]
        # b1t = 0.5 * (field1[patch0, i0, :] + N.roll(field1[patch0, i0 + 1, :], 1))
        b2t = field2[patch0, i0, :]

        if (typ == "vect"):
            # Interpolate B^r       
            xi0, eta0 = transform_coords(patch1, patch0, xi_yee[k], eta_yee[j1])
            Br[patch1, k, j1] = interp(brt, eta0, eta_yee)

        # Interpolate B^xi     
        xi0, eta0 = transform_coords(patch1, patch0, xi[k], eta_yee[j1])
        B1_int = interp(b1t, eta0, eta_yee)

        # Interpolate B^eta     
        xi0, eta0 = transform_coords(patch1, patch0, xi_yee[k], eta[j1])
        B2_int = interp(b2t, eta0, eta)

        # Convert from patch0 to patch1 coordinates
        xi0, eta0 = transform_coords(patch1, patch0, xi_yee[k], eta[j1])
        field1[patch1, k, j1] = transform(patch0, patch1, xi0, eta0, B1_int, B2_int)[0]
                
        # field2[patch1, k, j1] = transform(patch0, patch1, xi0, eta0, B1_int, B2_int)[1]

        #########
        # Communicate fields from eta left edge of patch1 to xi right edge of patch0
        ########     
        
        i0 = Nxi + NG # Last ghost cell of xi edge of patch0             
        j1 = NG  # First active cell of eta edge of patch1   

        brt = Br[patch1, :, j1]
        b1t = field1[patch1, :, j1]
        # b1t = 0.5 * (field1[patch1, :, j1] + N.roll(field1[patch1, :, j1 - 1], -1))
        b2t = field2[patch1, :, j1]

        # Interpolate B^xi     
        xi0, eta0 = transform_coords(patch0, patch1, xi[i0], eta_yee[k])
        B1_int = interp(b1t, xi0, xi)

        # Interpolate B^eta     
        xi0, eta0 = transform_coords(patch0, patch1, xi_yee[i0], eta[k])
        B2_int = interp(b2t, xi0, xi_yee)

        # Convert from patch0 to patch1 coordinates
        xi0, eta0 = transform_coords(patch0, patch1, xi_yee[i0], eta[k])
        field1[patch0, i0, k] = transform(patch1, patch0, xi0, eta0, B1_int, B2_int)[0]
      
        # field2[patch0, i0, k] = transform(patch1, patch0, xi0, eta0, B1_int, B2_int)[1]

    elif (top == 'yy'):

        k = range(NG, Nxi + NG)       

        #########
        # Communicate fields from eta top edge of patch0 to eta bottom edge of patch1
        ########                
                            
        j0 = Neta + NG - 1 # Last active cell of eta edge of patch0
        j1 = NG - 1 # First ghost cell of eta edge of patch1

        brt = Br[patch0, :, j0]
        b1t = field1[patch0, :, j0]
        b2t = field2[patch0, :, j0]
        # b2t = 0.5 * (field2[patch0, :, j0] + N.roll(field2[patch0, :, j0 + 1], 1))

        if (typ == "vect"):
            # Interpolate B^r
            xi0, eta0 = transform_coords(patch1, patch0, xi_yee[k], eta_yee[j1])
            Br[patch1, k, j1] = interp(brt, xi0, xi_yee)

        # Interpolate B^xi     
        xi0, eta0 = transform_coords(patch1, patch0, xi[k], eta_yee[j1])
        B1_int = interp(b1t, xi0, xi)

        # Interpolate B^eta     
        xi0, eta0 = transform_coords(patch1, patch0, xi_yee[k], eta[j1])
        B2_int = interp(b2t, xi0, xi_yee)

        # Convert from patch0 to patch1 coordinates
        xi0, eta0 = transform_coords(patch1, patch0, xi[k], eta_yee[j1])
        field1[patch1, k, j1] = transform(patch0, patch1, xi0, eta0, B1_int, B2_int)[0]
        
        # field2[patch1, k, j1] = transform(patch0, patch1, xi0, eta0, B1_int, B2_int)[1]

        #########
        # Communicate fields from eta bottom edge of patch1 to eta top edge of patch0
        ########   
          
        j0 = Neta + NG # Last ghost cell of eta edge of patch0
        j1 = NG # First active cell of eta edge of patch1

        brt = Br[patch1, :, j1]
        b1t = field1[patch1, :, j1]
        # b1t = 0.5 * (field1[patch1, :, j1] + N.roll(field1[patch1, :, j1 - 1], -1))
        b2t = field2[patch1, :, j1]

        # Interpolate B^xi     
        xi0, eta0 = transform_coords(patch0, patch1, xi[k], eta_yee[j0])
        B1_int = interp(b1t, xi0, xi_yee)

        # Interpolate B^eta     
        xi0, eta0 = transform_coords(patch0, patch1, xi_yee[k], eta[j0])
        B2_int = interp(b2t, xi0, xi_yee)

        # Convert from patch0 to patch1 coordinates
        xi0, eta0 = transform_coords(patch0, patch1, xi_yee[k], eta[j0])
        field2[patch0, k, j0] = transform(patch1, patch0, xi0, eta0, B1_int, B2_int)[1]

        # field1[patch0, k, j0] = transform(patch1, patch0, xi0, eta0, B1_int, B2_int)[0]

    elif (top == 'yx'):

        k = range(NG, Nxi + NG)       
            
        #########
        # Communicate fields from eta top edge of patch0 to xi bottom edge of patch1
        ########     
                
        j0 = Neta + NG - 1 # Last active cell of eta edge of patch0
        i1 = NG - 1 # First ghost cell on xi edge of patch1

        brt = Br[patch0, :, j0]
        b1t = field1[patch0, :, j0]
        b2t = field2[patch0, :, j0]
        # b2t = 0.5 * (field2[patch0, :, j0] + N.roll(field2[patch0, :, j0 + 1], 1))
        
        if (typ == "vect"):
            # Interpolate B^r 
            xi0, eta0 = transform_coords(patch1, patch0, xi_yee[i1], eta_yee[k])
            Br[patch1, i1, k] = interp(brt, xi0, xi_yee)

        # Interpolate B^xi     
        xi0, eta0 = transform_coords(patch1, patch0, xi[i1], eta_yee[k])
        B1_int = interp(b1t, xi0, xi)
        
        # Interpolate B^eta     
        xi0, eta0 = transform_coords(patch1, patch0, xi_yee[i1], eta[k])
        B2_int = interp(b2t, xi0, xi_yee)

        # Convert from patch0 to patch1 coordinates
        xi0, eta0 = transform_coords(patch1, patch0, xi[i1], eta_yee[k])
        field2[patch1, i1, k] = transform(patch0, patch1, xi0, eta0, B1_int, B2_int)[1]
        
        # field1[patch1, i1, k] = transform(patch0, patch1, xi0, eta0, B1_int, B2_int)[0]

        #########
        # Communicate fields from xi bottom edge of patch1 to eta top edge of patch0
        ########   
        
        j0 = Neta + NG # Last ghost cell of eta edge of patch0
        i1 = NG # First active cell of xi edge of patch1

        brt = Br[patch1, i1, :]
        b1t = field1[patch1, i1, :]
        b2t = field2[patch1, i1, :]
        # b2t = 0.5 * (field2[patch1, i1, :] + N.roll(field2[patch1, i1 - 1, :], -1))
        
        # Interpolate B^xi     
        xi0, eta0 = transform_coords(patch0, patch1, xi[k], eta_yee[j0])
        B1_int = interp(b1t, eta0, eta_yee)

        # Interpolate B^eta
        xi0, eta0 = transform_coords(patch0, patch1, xi_yee[k], eta[j0])
        B2_int = interp(b2t, eta0, eta)

        # Convert from patch0 to patch1 coordinates
        xi0, eta0 = transform_coords(patch0, patch1, xi[k], eta_yee[j0])
        field2[patch0, k, j0] = transform(patch1, patch0, xi0, eta0, B1_int, B2_int)[1]

        # field1[patch0, k, j0] = transform(patch1, patch0, xi0, eta0, B1_int, B2_int)[0]

########
# Special treatment of two triple points
########

def update_poles():

    #BCS open triple point
    Er_mean = (Er[Sphere.B, Nxi + NG - 1, NG]  \
             + Er[Sphere.C, Nxi + NG - 1, NG]  \
             + Er[Sphere.S, Nxi + NG - 1, NG])  / 3.0
    Er[Sphere.B, Nxi + NG, NG]  = Er_mean
    Er[Sphere.C, Nxi + NG, NG]  = Er_mean
    Er[Sphere.S, Nxi + NG, NG]  = Er_mean

    #ADN open triple point
    Er_mean = (Er[Sphere.A, NG, Neta + NG - 1] \
             + Er[Sphere.D, NG, Neta + NG - 1] \
             + Er[Sphere.N, NG, Neta + NG - 1]) / 3.0
    Er[Sphere.A, NG, Neta + NG] = Er_mean
    Er[Sphere.D, NG, Neta + NG] = Er_mean
    Er[Sphere.N, NG, Neta + NG] = Er_mean

########
# Plotting fields on an unfolded sphere
########

xi_grid_c, eta_grid_c = unflip_eq(xi_grid[NG:(Nxi + NG), NG:(Neta + NG)], eta_grid[NG:(Nxi + NG), NG:(Neta + NG)])
xi_grid_d, eta_grid_d = unflip_eq(xi_grid[NG:(Nxi + NG), NG:(Neta + NG)], eta_grid[NG:(Nxi + NG), NG:(Neta + NG)])
xi_grid_n, eta_grid_n = unflip_po(xi_grid[NG:(Nxi + NG), NG:(Neta + NG)], eta_grid[NG:(Nxi + NG), NG:(Neta + NG)])

def plot_fields_unfolded(it, field, fig, ax, vm):

    tab = (globals()[field])

    ax.pcolormesh(xi_grid[NG:(Nxi + NG), NG:(Neta + NG)], eta_grid[NG:(Nxi + NG), NG:(Neta + NG)], tab[Sphere.A, NG:(Nxi + NG), NG:(Neta + NG)], cmap = "RdBu_r", vmin = - vm, vmax = vm)
    ax.pcolormesh(xi_grid[NG:(Nxi + NG), NG:(Neta + NG)] + N.pi / 2.0, eta_grid[NG:(Nxi + NG), NG:(Neta + NG)], tab[Sphere.B, NG:(Nxi + NG), NG:(Neta + NG)], cmap = "RdBu_r", vmin = - vm, vmax = vm)
    ax.pcolormesh(xi_grid[NG:(Nxi + NG), NG:(Neta + NG)], eta_grid[NG:(Nxi + NG), NG:(Neta + NG)] - N.pi / 2.0, tab[Sphere.S, NG:(Nxi + NG), NG:(Neta + NG)], cmap = "RdBu_r", vmin = - vm, vmax = vm)

    if (field == "B1u"):
        ax.pcolormesh(xi_grid_c + N.pi, eta_grid_c, B2u[Sphere.C, NG:(Nxi + NG), NG:(Neta + NG)], cmap = "RdBu_r", vmin = - vm, vmax = vm)
        ax.pcolormesh(xi_grid_d - N.pi / 2.0, eta_grid_d, B2u[Sphere.D, NG:(Nxi + NG), NG:(Neta + NG)], cmap = "RdBu_r", vmin = - vm, vmax = vm)
        ax.pcolormesh(xi_grid_n, eta_grid_n + N.pi / 2.0, - B2u[Sphere.N, NG:(Nxi + NG), NG:(Neta + NG)], cmap = "RdBu_r", vmin = - vm, vmax = vm)
    elif (field == "B2u"):
        ax.pcolormesh(xi_grid_c + N.pi, eta_grid_c, - B1u[Sphere.C, NG:(Nxi + NG), NG:(Neta + NG)], cmap = "RdBu_r", vmin = - vm, vmax = vm)
        ax.pcolormesh(xi_grid_d - N.pi / 2.0, eta_grid_d, - B1u[Sphere.D, NG:(Nxi + NG), NG:(Neta + NG)], cmap = "RdBu_r", vmin = - vm, vmax = vm)
        ax.pcolormesh(xi_grid_n, eta_grid_n + N.pi / 2.0, B1u[Sphere.N, NG:(Nxi + NG), NG:(Neta + NG)], cmap = "RdBu_r", vmin = - vm, vmax = vm)
    elif (field == "E2u"):
        ax.pcolormesh(xi_grid_c + N.pi, eta_grid_c, - E1u[Sphere.C, NG:(Nxi + NG), NG:(Neta + NG)], cmap = "RdBu_r", vmin = - vm, vmax = vm)
        ax.pcolormesh(xi_grid_d - N.pi / 2.0, eta_grid_d, - E1u[Sphere.D, NG:(Nxi + NG), NG:(Neta + NG)], cmap = "RdBu_r", vmin = - vm, vmax = vm)
        ax.pcolormesh(xi_grid_n, eta_grid_n + N.pi / 2.0, E1u[Sphere.N, NG:(Nxi + NG), NG:(Neta + NG)], cmap = "RdBu_r", vmin = - vm, vmax = vm)
    elif (field == "E1u"):
        ax.pcolormesh(xi_grid_c + N.pi, eta_grid_c, E2u[Sphere.C, NG:(Nxi + NG), NG:(Neta + NG)], cmap = "RdBu_r", vmin = - vm, vmax = vm)
        ax.pcolormesh(xi_grid_d - N.pi / 2.0, eta_grid_d, E2u[Sphere.D, NG:(Nxi + NG), NG:(Neta + NG)], cmap = "RdBu_r", vmin = - vm, vmax = vm)
        ax.pcolormesh(xi_grid_n, eta_grid_n + N.pi / 2.0, - E2u[Sphere.N, NG:(Nxi + NG), NG:(Neta + NG)], cmap = "RdBu_r", vmin = - vm, vmax = vm)
    else:
        ax.pcolormesh(xi_grid_c + N.pi, eta_grid_c, tab[Sphere.C, NG:(Nxi + NG), NG:(Neta + NG)], cmap = "RdBu_r", vmin = - vm, vmax = vm)
        ax.pcolormesh(xi_grid_d - N.pi / 2.0, eta_grid_d, tab[Sphere.D, NG:(Nxi + NG), NG:(Neta + NG)], cmap = "RdBu_r", vmin = - vm, vmax = vm)
        ax.pcolormesh(xi_grid_n, eta_grid_n + N.pi / 2.0, tab[Sphere.N, NG:(Nxi + NG), NG:(Neta + NG)], cmap = "RdBu_r", vmin = - vm, vmax = vm)
    
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

th0 = N.zeros_like(xi_grid)
ph0 = N.zeros_like(xi_grid)

cmap_plot="RdBu_r"
norm_plot = matplotlib.colors.Normalize(vmin = - 0.2, vmax = 0.2)
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
    
def plot_fields_sphere(it, field, res):

    tab = (globals()[field])

    fig, ax = P.subplots(subplot_kw={"projection": "3d"}, figsize = fig_size, facecolor = 'w')
    ax.view_init(elev=45, azim=45)
    ax.plot_surface(x_s, y_s, z_s, rstride=1, cstride=1, shade=False, color = 'grey', zorder = 0)
    
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

    P.close("all")    

########
# Source current
########

theta0, phi0 = 90.0 / 360.0 * 2.0 * N.pi, 0.0 / 360.0 * 2.0 * N.pi # Center of the wave packet
x0 = N.sin(theta0) * N.cos(phi0) 
y0 = N.sin(theta0) * N.sin(phi0)
z0 = N.cos(theta0)

def shape_packet(x, y, z, width):
    return N.exp(- y * y / (width * width)) * N.exp(- x * x / (width * width)) * N.exp(- z * z / (width * width)) 

w = 0.1         # Radius of wave packet
omega = 20.0    # Frequency of current
J0 = 1.0        # Current amplitude
p0 = Sphere.A   # Patch location of antenna

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
    return Jr_tot[patch, i0:i1, j0:j1] * N.sin(omega * dt * it) * (1 + N.tanh(20 - it/5.))/2.
#    return Jr_tot[patch, i0:i1, j0:j1] * N.sin(omega * dt * it)

########
# Initialization
########

for p in range(6):
    pass # All fields to zero

pinit = Sphere.A

for i in range(Nxi + 2 * NG):
    for j in range(Neta + 2 * NG):

        fcoord0 = (globals()["coord_" + sphere[pinit] + "_to_sph"])
        th, ph = fcoord0(xi_grid[i, j], eta_grid[i, j])
        x = N.sin(th) * N.cos(ph)
        y = N.sin(th) * N.sin(ph)
        z = N.cos(th)  
        
        x0 = N.sin(theta0) * N.cos(phi0)
        y0 = N.sin(theta0) * N.sin(phi0)
        z0 = N.cos(theta0)
        
        Er[Sphere.A, i, j] = 0.0 #shape_packet(x - x0, y - y0, z - z0, w) 
        B2u[Sphere.A, i, j] = 0.0 #- shape_packet(x - x0, y - y0, z - z0, w)

iter = 0
idump = 0

Nt = 2500 # Number of iterations
FDUMP = 10 # Dump frequency

time = dt * N.arange(Nt)
int_energy = N.zeros((6, Nt))

# Figure parameters
scale, aspect = 2.0, 0.7
vm = 0.2
ratio = 2.0
fig_size=deffigsize(scale, aspect)
style = '2d' 

# Define figure
if (style == '2d'):
    fig = P.figure(1, figsize = fig_size, facecolor='w')
    ax = P.subplot(111)
elif (style == '3d'):
    fig, ax = P.subplots(subplot_kw={"projection": "3d"}, figsize = fig_size, facecolor = 'w')
    ax.view_init(elev=45, azim=45)
    ax.plot_surface(x_s, y_s, z_s, rstride=1, cstride=1, shade=False, color = 'grey', zorder = 0)

########
# Main routine
########

# for it in range(Nt):
for it in tqdm(range(Nt), "Progression"):
    if ((it % FDUMP) == 0):
        plot_fields_unfolded(idump, "Er", fig, ax, 0.2)
        # plot_fields_unfolded(idump, "B1u", fig, ax, 0.2)
        # plot_fields_unfolded(idump, "B2u", fig, ax, 0.2)
        # plot_fields_sphere(idump, "Er", 2)
        idump += 1

    # print(it)
    iter += 1
    
    for p in range(6):
        contra_to_cov_E(p)

    for i in range(n_zeros):
        p0, p1 = index_row[i], index_col[i]
        communicate_E_patch_cov(p0, p1)

    update_poles()

    for p in range(6):
        push_B(p)

    for i in range(n_zeros):
        p0, p1 = index_row[i], index_col[i]
        communicate_B_patch_contra(p0, p1)

    for p in range(6):
        contra_to_cov_B(p)

    for i in range(n_zeros):
        p0, p1 = index_row[i], index_col[i]
        communicate_B_patch_cov(p0, p1)

    for p in range(6):
        push_E(it, p)

    for i in range(n_zeros):
        p0, p1 = index_row[i], index_col[i]
        communicate_E_patch_contra(p0, p1)

    for p in range(6):
        push_potential(p)

    energy = (Er * Er + B1d * B1u + B2d * B2u) * sqrt_det_g[:, :, 0]
    
    int_energy[:, it] = N.array([spi.simps(spi.simps(energy[p, :, :], axis = 1, dx = dxi)) for p in range(6)])

# Figure parameters
scale, aspect = 1.0, 0.7
ratio = 2.0
fig_size=deffigsize(scale, aspect)

fig = P.figure(2, figsize = fig_size, facecolor='w')
ax = P.subplot(111)

for p in range(6):
    P.plot(time, int_energy[p, :], label = r"Patch {}".format(sphere[p]))

P.xlabel(r"$t$")
P.ylabel(r"Electromagnetic energy")

P.legend(loc = "upper right", ncol = 1)

P.grid(True, ls='--')

figsave_png(fig, "./total_energy")

fig = P.figure(3, figsize = fig_size, facecolor='w')
ax = P.subplot(111)

for p in range(6):
    P.semilogy(time, int_energy[p, :], label = r"Patch {}".format(sphere[p]))

P.xlabel(r"$t$")
P.ylabel(r"Electromagnetic energy")
P.ylim((1e-5, 5.0))

P.legend(loc = "lower right", ncol = 1)

P.grid(True, ls='--')

figsave_png(fig, "./total_energy_log")
