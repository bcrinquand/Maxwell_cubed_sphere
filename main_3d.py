# Import modules
import numpy as N
import matplotlib.pyplot as P
import matplotlib
import time
import scipy.integrate as spi
from scipy import interpolate
from skimage.measure import find_contours
from math import *
import sys
import h5py
from tqdm import tqdm

# Import my figure routines
from figure_module import *

# Connect patch indices and names
sphere = {0: "A", 1: "B", 2: "C", 3: "D", 4: "N", 5: "S"}

outdir = '/Users/jensmahlmann/Documents/Projects/Development/CubedSphere/'

class Sphere:
    A = 0
    B = 1
    C = 2
    D = 3
    N = 4
    S = 5

# Parameters
cfl = 0.6
Nr = 70
Nxi = 64
Neta = 64
NG = 1 # Number of ghosts zones
r_min, r_max = 1.0, 10.0
xi_min, xi_max = - N.pi / 4.0, N.pi / 4.0
eta_min, eta_max = - N.pi / 4.0, N.pi / 4.0
dr = (r_max - r_min) / Nr
dxi = (xi_max - xi_min) / Nxi
deta = (eta_max - eta_min) / Neta

# Define grids
r = N.linspace(r_min, r_max, Nr)
xi  = N.arange(- NG - int(Nxi / 2), NG + int(Nxi / 2), 1) * dxi
eta = N.arange(- NG - int(Neta / 2), NG + int(Neta / 2), 1) * deta
eta_grid, xi_grid = N.meshgrid(eta, xi)
r_yee = r + 0.5 * dr
xi_yee  = xi  + 0.5 * dxi
eta_yee = eta + 0.5 * deta

# Initialize fields
Er  = N.zeros((6, Nr, Nxi + 2 * NG, Neta + 2 * NG,))
E1u = N.zeros((6, Nr, Nxi + 2 * NG, Neta + 2 * NG,))
E2u = N.zeros((6, Nr, Nxi + 2 * NG, Neta + 2 * NG,))
Br  = N.zeros((6, Nr, Nxi + 2 * NG, Neta + 2 * NG,))
B1u = N.zeros((6, Nr, Nxi + 2 * NG, Neta + 2 * NG,))
B2u = N.zeros((6, Nr, Nxi + 2 * NG, Neta + 2 * NG,))

E1d = N.zeros((6, Nr, Nxi + 2 * NG, Neta + 2 * NG))
E2d = N.zeros((6, Nr, Nxi + 2 * NG, Neta + 2 * NG))
B1d = N.zeros((6, Nr, Nxi + 2 * NG, Neta + 2 * NG))
B2d = N.zeros((6, Nr, Nxi + 2 * NG, Neta + 2 * NG))

Ar = N.zeros((6, Nxi + 2 * NG, Neta + 2 * NG))

########
# Define initial data
########

def InitialData():

    B0 = 1.0

    for patch in range(6):

        fvec = (globals()["vec_sph_to_" + sphere[patch]])
        fcoord = (globals()["coord_" + sphere[patch] + "_to_sph"])

        for h in tqdm(range(Nr)):
            for i in range(Nxi + 2 * NG):
                for j in range(Neta + 2 * NG):

                    r0 = r[h]
                    th0, ph0 = fcoord(xi_grid[i, j], eta_grid[i, j])

                    BrTMP = 2.0 * (cos(th0)+cos(-th0))/2.0 / ((r0**3.0)) * B0
                    BtTMP = (sin(th0)-sin(-th0))/2.0 / ((r0**4.0)) * B0
                    BpTMP = 0.0

                    BCStmp = fvec(th0, ph0, BtTMP, BpTMP)

                    Br[patch, h, i, j] = BrTMP
                    B1u[patch, h, i, j] = BCStmp[0]
                    B2u[patch, h, i, j] = BCStmp[1]

                    Er[patch, h, i, j] = 0.0
                    E1u[patch, h, i, j] = 0.0
                    E2u[patch, h, i, j] = 0.0


########
# Dump HDF5 output
########

def WriteFieldHDF5(it, field):

    outvec = (globals()[field])
    h5f = h5py.File(outdir+field+'.'+ str(it).rjust(5, '0') +'.h5', 'w')

    for patch in range(6):
        h5f.create_dataset(field+str(patch), data=outvec[patch, :, :, :])

    h5f.close()

def WriteAllFieldsHDF5(idump):

    WriteFieldHDF5(idump, "Br")
    WriteFieldHDF5(idump, "B1u")
    WriteFieldHDF5(idump, "B2u")
    WriteFieldHDF5(idump, "Er")
    WriteFieldHDF5(idump, "E1u")
    WriteFieldHDF5(idump, "E2u")

def WriteCoordsHDF5():

    h5f = h5py.File(outdir+'grid'+'.'+ str(0).rjust(5, '0') +'.h5', 'w')

    h5f.create_dataset('r', data=r)
    h5f.create_dataset('xi', data=xi)
    h5f.create_dataset('eta', data=eta)

    h5f.close()

########
# Define metric tensor
########

g11d = N.empty((Nr, Nxi + 2 * NG, Neta + 2 * NG, 6))
g12d = N.empty((Nr, Nxi + 2 * NG, Neta + 2 * NG, 6))
g22d = N.empty((Nr, Nxi + 2 * NG, Neta + 2 * NG, 6))

for i in range(Nxi + 2 * NG):
    for j in range(Neta + 2 * NG):
        for k in range(Nr):

            # 0 at (k, i + 1/2, j)
            r0 = r[k]
            X = N.tan(xi_yee[i])
            Y = N.tan(eta[j])
            C = N.sqrt(1.0 + X * X)
            D = N.sqrt(1.0 + Y * Y)
            delta = N.sqrt(1.0 + X * X + Y * Y)

            g11d[k, i, j, 0] = (r0 * r0 * C * C * D / (delta * delta))**2
            g22d[k, i, j, 0] = (r0 * r0 * C * D * D / (delta * delta))**2
            g12d[k, i, j, 0] = - r0 * r0 * X * Y * C * C * D * D / (delta)**4

            # 1 at (k, i, j + 1/2)
            r0 = r[k]
            X = N.tan(xi[i])
            Y = N.tan(eta_yee[j])
            C = N.sqrt(1.0 + X * X)
            D = N.sqrt(1.0 + Y * Y)
            delta = N.sqrt(1.0 + X * X + Y * Y)

            g11d[k, i, j, 1] = (r0 * r0 * C * C * D / (delta * delta))**2
            g22d[k, i, j, 1] = (r0 * r0 * C * D * D / (delta * delta))**2
            g12d[k, i, j, 1] = - r0 * r0 * X * Y * C * C * D * D / (delta)**4

            # 2 at (k, i + 1/2, j + 1/2)
            r0 = r[k]
            X = N.tan(xi_yee[i])
            Y = N.tan(eta_yee[j])
            C = N.sqrt(1.0 + X * X)
            D = N.sqrt(1.0 + Y * Y)
            delta = N.sqrt(1.0 + X * X + Y * Y)

            g11d[k, i, j, 2] = (r0 * r0 * C * C * D / (delta * delta))**2
            g22d[k, i, j, 2] = (r0 * r0 * C * D * D / (delta * delta))**2
            g12d[k, i, j, 2] = - r0 * r0 * X * Y * C * C * D * D / (delta)**4

            # 3 at (k + 1/2, i, j)
            r0 = r_yee[k]
            X = N.tan(xi[i])
            Y = N.tan(eta[j])
            C = N.sqrt(1.0 + X * X)
            D = N.sqrt(1.0 + Y * Y)
            delta = N.sqrt(1.0 + X * X + Y * Y)

            g11d[k, i, j, 3] = (r0 * r0 * C * C * D / (delta * delta))**2
            g22d[k, i, j, 3] = (r0 * r0 * C * D * D / (delta * delta))**2
            g12d[k, i, j, 3] = - r0 * r0 * X * Y * C * C * D * D / (delta)**4

            # 4 at (k + 1/2, i, j + 1/2)
            r0 = r_yee[k]
            X = N.tan(xi[i])
            Y = N.tan(eta_yee[j])
            C = N.sqrt(1.0 + X * X)
            D = N.sqrt(1.0 + Y * Y)
            delta = N.sqrt(1.0 + X * X + Y * Y)

            g11d[k, i, j, 4] = (r0 * r0 * C * C * D / (delta * delta))**2
            g22d[k, i, j, 4] = (r0 * r0 * C * D * D / (delta * delta))**2
            g12d[k, i, j, 4] = - r0 * r0 * X * Y * C * C * D * D / (delta)**4

            # 5 at (k + 1/2, i + 1/2, j)
            r0 = r_yee[k]
            X = N.tan(xi_yee[i])
            Y = N.tan(eta[j])
            C = N.sqrt(1.0 + X * X)
            D = N.sqrt(1.0 + Y * Y)
            delta = N.sqrt(1.0 + X * X + Y * Y)

            g11d[k, i, j, 5] = (r0 * r0 * C * C * D / (delta * delta))**2
            g22d[k, i, j, 5] = (r0 * r0 * C * D * D / (delta * delta))**2
            g12d[k, i, j, 5] = - r0 * r0 * X * Y * C * C * D * D / (delta)**4

sqrt_det_g = N.sqrt(g11d * g22d - g12d * g12d)

dt = cfl * N.min(1.0 / N.sqrt(1.0 / (dr * dr) + g11d / (sqrt_det_g * sqrt_det_g) / (dxi * dxi) + g22d / (sqrt_det_g * sqrt_det_g) / (deta * deta) ))
print("delta t = {}".format(dt))

########
# Single-patch push routines
########

def contra_to_cov_E(patch):

    i0, i1 = NG, Nxi + NG
    j0, j1 = NG, Neta + NG

    E1d[patch, :, i0:i1, j0:j1] = g11d[:, i0:i1, j0:j1, 0] * E1u[patch, :, i0:i1, j0:j1] + \
                         0.5 * g12d[:, i0:i1, j0:j1, 0] * (E2u[patch, :, i0:i1, j0:j1] + N.roll(N.roll(E2u, -1, axis = 2), 1, axis = 3)[patch, :, i0:i1, j0:j1])
    E2d[patch, :, i0:i1, j0:j1] = g22d[:, i0:i1, j0:j1, 1] * E2u[patch, :, i0:i1, j0:j1] + \
                         0.5 * g12d[:, i0:i1, j0:j1, 1] * (E1u[patch, :, i0:i1, j0:j1] + N.roll(N.roll(E1u, 1, axis = 2), -1, axis = 3)[patch, :, i0:i1, j0:j1])

def contra_to_cov_B(patch):

    i0, i1 = NG, Nxi + NG
    j0, j1 = NG, Neta + NG

    B1d[patch, :, i0:i1, j0:j1] = g11d[:, i0:i1, j0:j1, 4] * B1u[patch, :, i0:i1, j0:j1] + \
                         0.5 * g12d[:, i0:i1, j0:j1, 4] * (B2u[patch, :, i0:i1, j0:j1] + N.roll(N.roll(B2u, 1, axis = 2), -1, axis = 3)[patch, :, i0:i1, j0:j1])
    B2d[patch, :, i0:i1, j0:j1] = g22d[:, i0:i1, j0:j1, 5] * B2u[patch, :, i0:i1, j0:j1] + \
                         0.5 * g12d[:, i0:i1, j0:j1, 5] * (B1u[patch, :, i0:i1, j0:j1] + N.roll(N.roll(B1u, -1, axis = 2), 1, axis = 3)[patch, :, i0:i1, j0:j1])

def push_B(patch):

    i0, i1 = NG, Nxi + NG
    j0, j1 = NG, Neta + NG

    Br[patch, :, i0:i1, j0:j1]  -= ((N.roll(E2d, -1, axis = 2)[patch, :, i0:i1, j0:j1] - E2d[patch, :, i0:i1, j0:j1]) / dxi \
                                  - (N.roll(E1d, -1, axis = 3)[patch, :, i0:i1, j0:j1] - E1d[patch, :, i0:i1, j0:j1]) / deta) \
                                  * dt / sqrt_det_g[:, i0:i1, j0:j1, 2]
    B1u[patch, :, i0:i1, j0:j1] -= ((N.roll(Er, -1, axis = 3)[patch, :, i0:i1, j0:j1] - Er[patch, :, i0:i1, j0:j1]) / deta \
                                 - (N.roll(E2d, -1, axis = 1)[patch, :, i0:i1, j0:j1] - E2d[patch, :, i0:i1, j0:j1]) / dr) \
                                  * dt / sqrt_det_g[:, i0:i1, j0:j1, 4]
    B2u[patch, :, i0:i1, j0:j1] += ((N.roll(Er, -1, axis = 2)[patch, :, i0:i1, j0:j1] - Er[patch, :, i0:i1, j0:j1]) / dxi \
                                 - (N.roll(E1d, -1, axis = 1)[patch, :, i0:i1, j0:j1] - E1d[patch, :, i0:i1, j0:j1]) / dr) \
                                  * dt / sqrt_det_g[:, i0:i1, j0:j1, 5]

def push_E(patch):

    i0, i1 = NG, Nxi + NG
    j0, j1 = NG, Neta + NG

    Er[patch, :, i0:i1, j0:j1]  += ((B2d[patch, :, i0:i1, j0:j1] - N.roll(B2d, 1, axis = 2)[patch, :, i0:i1, j0:j1]) / dxi \
                                 - (B1d[patch, :, i0:i1, j0:j1] - N.roll(B1d, 1, axis = 3)[patch, :, i0:i1, j0:j1]) / deta) \
                                 * dt / sqrt_det_g[:, i0:i1, j0:j1, 3] # - 4.0 * N.pi * dt * Jr(it, patch, i0, i1, j0, j1)
    E1u[patch, :, i0:i1, j0:j1] += ((Br[patch, :, i0:i1, j0:j1] - N.roll(Br, 1, axis = 3)[patch, :, i0:i1, j0:j1]) / deta \
                                 - (B2d[patch, :, i0:i1, j0:j1] - N.roll(B2d, 1, axis = 1)[patch, :, i0:i1, j0:j1]) / dr) \
                                 * dt / sqrt_det_g[:, i0:i1, j0:j1, 0]
    E2u[patch, :, i0:i1, j0:j1] -= ((Br[patch, :, i0:i1, j0:j1] - N.roll(Br, 1, axis = 2)[patch, :, i0:i1, j0:j1]) / dxi \
                                 - (B2d[patch, :, i0:i1, j0:j1] - N.roll(B2d, 1, axis = 1)[patch, :, i0:i1, j0:j1]) / dr) \
                                 * dt / sqrt_det_g[:, i0:i1, j0:j1, 1]

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
# Interpolation routines
########

def interp(arr_in, xA, xB):
    f = interpolate.interp1d(xA, arr_in, axis = 1, kind='linear', fill_value=(0,0), bounds_error=False)
    return f(xB)

# %%
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
# patch0 has the open boundary
# patch1 has the closed boundary
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

    if (top == 'xx'):

        #########
        # Communicate fields from xi left edge of patch1 to xi right edge of patch0
        ########

        i0 = Nxi + NG # Last ghost cell of xi edge of patch0
        i1 = NG  # First active cell of xi edge of patch1

        xi0 = xi_grid[i0, :] + 0.0 * dxi
        eta0 = eta_grid[i0, :] + 0.5 * deta

        xi1, eta1 = transform_coords(patch0, patch1, xi0, eta0)

        E1 = interp(field1[patch1, :, i1, :], eta, eta1)
        E2 = interp(field2[patch1, :, i1, :], eta_yee, eta1)

        field2[patch0, :, i0, NG:(Nxi + NG)] = transform(patch1, patch0, xi1, eta1, E1, E2)[1][:, NG:(Nxi + NG)]

        if (typ == "vect"):

            i0 = Nxi + NG # Last ghost cell of xi edge of patch0
            i1 = NG  # First active cell of xi edge of patch1

            xi0 = xi_grid[i0, :] + 0.0 * dxi
            eta0 = eta_grid[i0, :] + 0.0 * deta

            xi1, eta1 = transform_coords(patch0, patch1, xi0, eta0)

            Er1 = interp(Er[patch1, :, i1, :], eta, eta1)
            Er[patch0, :, i0, NG:(Nxi + NG)] = Er1[:, NG:(Nxi + NG)]

        #########
        # Communicate fields from xi right edge of patch0 to xi left edge patch1
        ########

        i0 = Nxi + NG - 1 # Last active cell of xi edge of patch0
        i1 = NG - 1 # First ghost cell of xi edge of patch1

        xi1 = xi_grid[i1, :] + 0.5 * dxi
        eta1 = eta_grid[i1, :] + 0.0 * deta

        xi0, eta0 = transform_coords(patch1, patch0, xi1, eta1)

        E1 =  interp(field1[patch0, :, i0, :], eta, eta0)
        E2 = (interp(field2[patch0, :, i0, :], eta_yee, eta0) + interp(field2[patch0, :, i0 + 1, :], eta_yee, eta0)) / 2.0

        field1[patch1, :, i1, NG:(Nxi + NG)] = transform(patch0, patch1, xi0, eta0, E1, E2)[0][:, NG:(Nxi + NG)]

    elif (top == 'xy'):

        #########
        # Communicate fields from eta left edge of patch1 to xi right edge of patch0
        ########

        i0 = Nxi + NG # Last ghost cell of xi edge of patch0
        j1 = NG  # First active cell of eta edge of patch1

        xi0 = xi_grid[i0, :] + 0.0 * dxi
        eta0 = eta_grid[i0, :] + 0.5 * deta

        xi1, eta1 = transform_coords(patch0, patch1, xi0, eta0)

        E1 = interp(field1[patch1, :, :, j1], xi_yee, xi1)
        E2 = interp(field2[patch1, :, :, j1], xi, xi1)

        field2[patch0, :, i0, NG:(Nxi + NG)] = transform(patch1, patch0, xi1, eta1, E1, E2)[1][:, NG:(Nxi + NG)]

        if (typ == "vect"):

            i0 = Nxi + NG # Last ghost cell of xi edge of patch0
            j1 = NG  # First active cell of eta edge of patch1

            xi0 = xi_grid[i0, :] + 0.0 * dxi
            eta0 = eta_grid[i0, :] + 0.0 * deta

            xi1, eta1 = transform_coords(patch0, patch1, xi0, eta0)

            Er1 = interp(Er[patch1, :, :, j1], xi, xi1)
            Er[patch0, :, i0, (NG + 1):(Nxi + NG)] = Er1[:, (NG + 1):(Nxi + NG)]

        #########
        # Communicate fields from xi right edge of patch0 to eta left edge patch1
        ########

        i0 = Nxi + NG - 1 # Last active cell of xi edge of patch0
        j1 = NG - 1 # First ghost cell on eta edge of patch1

        xi1 = xi_grid[:, j1] + 0.0 * dxi
        eta1 = eta_grid[:, j1] + 0.5 * deta

        xi0, eta0 = transform_coords(patch1, patch0, xi1, eta1)

        E1 =  interp(field1[patch0, :, i0, :], eta, eta0)
        E2 = (interp(field2[patch0, :, i0, :], eta_yee, eta0) + interp(field2[patch0, :, i0 + 1, :], eta_yee, eta0)) / 2.0

        field2[patch1, :, NG:(Nxi + NG + 1), j1] = transform(patch0, patch1, xi0, eta0, E1, E2)[1][:, NG:(Nxi + NG + 1)]

    elif (top == 'yy'):

        #########
        # Communicate fields from eta left edge of patch1 to eta right edge of patch0
        ########

        j0 = Neta + NG # Last ghost cell of eta edge of patch0
        j1 = NG # First active cell of eta edge of patch1

        xi0 = xi_grid[:, j0] + 0.5 * dxi
        eta0 = eta_grid[:, j0] + 0.0 * deta

        xi1, eta1 = transform_coords(patch0, patch1, xi0, eta0)

        E1 = interp(field1[patch1, :, :, j1], xi_yee, xi1)
        E2 = interp(field2[patch1, :, :, j1], xi, xi1)

        field1[patch0, :, NG:(Nxi + NG), j0] = transform(patch1, patch0, xi1, eta1, E1, E2)[0][:, NG:(Nxi + NG)]

        if (typ == "vect"):

            j0 = Neta + NG # Last ghost cell of eta edge of patch0
            j1 = NG # First active cell of eta edge of patch1

            xi0 = xi_grid[:, j0] + 0.0 * dxi
            eta0 = eta_grid[:,j0] + 0.0 * deta

            xi1, eta1 = transform_coords(patch0, patch1, xi0, eta0)

            Er1 = interp(Er[patch1, :, :, j1], xi, xi1)
            Er[patch0, :, NG:(Nxi + NG), j0] = Er1[:, NG:(Nxi + NG)]

        #########
        # Communicate fields from eta right edge of patch0 to eta left edge patch1
        ########

        j0 = Neta + NG - 1 # Last active cell of eta edge of patch0
        j1 = NG - 1 # First ghost cell of eta edge of patch1

        xi1 = xi_grid[:, j1] + 0.0 * dxi
        eta1 = eta_grid[:, j1] + 0.5 * deta

        xi0, eta0 = transform_coords(patch1, patch0, xi1, eta1)

        E1 = (interp(field1[patch0, :, :, j0], xi_yee, xi0)+interp(field1[patch0, :, :, j0 + 1], xi_yee, xi0)) / 2.0
        E2 =  interp(field2[patch0, :, :, j0], xi, xi0)

        field2[patch1, :, NG:(Nxi + NG), j1] = transform(patch0, patch1, xi0, eta0, E1, E2)[1][:, NG:(Nxi + NG)]

    elif (top == 'yx'):

        #########
        # Communicate fields from eta left edge of patch1 to xi right edge of patch0
        ########

        j0 = Neta + NG # Last ghost cell of eta edge of patch0
        i1 = NG # First active cell of xi edge of patch1

        xi0 = xi_grid[:, j0] + 0.5 * dxi
        eta0 = eta_grid[:, j0] + 0.0 * deta

        xi1, eta1 = transform_coords(patch0, patch1, xi0, eta0)

        E1 = interp(field1[patch1, :, i1, :], eta, eta1)
        E2 = interp(field2[patch1, :, i1, :], eta_yee, eta1)

        field1[patch0, :, NG:(Nxi + NG), j0] = transform(patch1, patch0, xi1, eta1, E1, E2)[0][:, NG:(Nxi + NG)]

        if (typ == "vect"):

            j0 = Neta + NG # Last ghost cell of eta edge of patch0
            i1 = NG # First active cell of xi edge of patch1

            xi0 = xi_grid[:, j0] + 0.0 * dxi
            eta0 = eta_grid[:, j0] + 0.0 * deta

            xi1, eta1 = transform_coords(patch0, patch1, xi0, eta0)

            Er1 = interp(Er[patch1, :, i1, :], eta, eta1)
            Er[patch0, :, (NG + 1):(Nxi + NG), j0] = Er1[:, (NG + 1):(Nxi + NG)]

        #########
        # Communicate fields from eta right edge of patch0 to xi left edge patch1
        ########

        j0 = Neta + NG - 1 # Last active cell of eta edge of patch0
        i1 = NG - 1 # First ghost cell on xi edge of patch1

        xi1 = xi_grid[i1, :] + 0.5 * dxi
        eta1 = eta_grid[i1, :] + 0.0 * deta

        xi0, eta0 = transform_coords(patch1, patch0, xi1, eta1)

        E1 = (interp(field1[patch0, :, :, j0], xi_yee, xi0) + interp(field1[patch0, :, :, j0 + 1], xi_yee, xi0)) / 2.0
        E2 =  interp(field2[patch0, :, :, j0], xi, xi0)

        field1[patch1, :, i1, NG:(Nxi + NG + 1)] = transform(patch0, patch1, xi0, eta0, E1, E2)[0][:, NG:(Nxi + NG + 1)]

    else:
        return

########
# Communication of covariant or contravariant B
# patch0 has the open boundary
# patch1 has the closed boundary
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

    if (top == 'xx'):

        #########
        # Communicate fields from xi left edge of patch1 to xi right edge of patch0
        ########

        i0 = Nxi + NG # Last ghost cell of xi edge of patch0
        i1 = NG  # First active cell of xi edge of patch1

        xi0 = xi_grid[i0, :] + 0.0 * dxi
        eta0 = eta_grid[i0, :] + 0.5 * deta

        xi1, eta1 = transform_coords(patch0, patch1, xi0, eta0)

        B1 = interp(field1[patch1, :, i1, :], eta_yee, eta1)
        B2 = interp(field2[patch1, :, i1, :], eta, eta1)

        field1[patch0, :, i0, NG:(Nxi + NG)] = transform(patch1, patch0, xi1, eta1, B1, B2)[0][:, NG:(Nxi + NG)]

        #########
        # Communicate fields from xi right edge of patch0 to xi left edge patch1
        ########

        i0 = Nxi + NG - 1 # Last active cell of xi edge of patch0
        i1 = NG - 1 # First ghost cell of xi edge of patch1

        xi1 = xi_grid[i1, :] + 0.5 * dxi
        eta1 = eta_grid[i1, :] + 0.0 * deta

        xi0, eta0 = transform_coords(patch1, patch0, xi1, eta1)

        B1 = (interp(field1[patch0, :, i0, :], eta_yee, eta0) + interp(field1[patch0, :, i0 + 1, :], eta_yee, eta0)) / 2.0
        B2 =  interp(field2[patch0, :, i0, :], eta, eta0)

        field2[patch1, :, i1, NG:(Nxi + NG)] = transform(patch0, patch1, xi0, eta0, B1, B2)[1][:, NG:(Nxi + NG)]

        if (typ == "vect"):

            i0 = Nxi + NG - 1 # Last active cell of xi edge of patch0
            i1 = NG - 1 # First ghost cell of xi edge of patch1

            xi1 = xi_grid[i1,:] + 0.5 * dxi
            eta1 = eta_grid[i1,:] + 0.5 * deta

            xi0, eta0 = transform_coords(patch1, patch0, xi1, eta1)

            Br0 = interp(Br[patch0, :, i0, :], eta_yee, eta0)
            Br[patch1, :, i1, NG:(Nxi + NG)] = Br0[:, NG:(Nxi + NG)]

    elif (top == 'xy'):

        #########
        # Communicate fields from eta left edge of patch1 to xi right edge of patch0
        ########

        i0 = Nxi + NG # Last ghost cell of xi edge of patch0
        j1 = NG  # First active cell of eta edge of patch1

        xi0 = xi_grid[i0, :] + 0.0 * dxi
        eta0 = eta_grid[i0, :] + 0.5 * deta

        xi1, eta1 = transform_coords(patch0, patch1, xi0, eta0)

        B1 = interp(field1[patch1, :, :, j1], xi, xi1)
        B2 = interp(field2[patch1, :, :, j1], xi_yee, xi1)

        field1[patch0, :, i0, NG:(Nxi + NG)] = transform(patch1, patch0, xi1, eta1, B1, B2)[0][:, NG:(Nxi + NG)]

        #########
        # Communicate fields from xi right edge of patch0 to eta left edge patch1
        ########

        i0 = Nxi + NG - 1 # Last active cell of xi edge of patch0
        j1 = NG - 1 # First ghost cell on eta edge of patch1

        xi1 = xi_grid[:, j1] + 0.0 * dxi
        eta1 = eta_grid[:, j1] + 0.5 * deta

        xi0, eta0 = transform_coords(patch1, patch0, xi1, eta1)

        B1 = (interp(field1[patch0, :, i0, :], eta_yee, eta0) + interp(field1[patch0, :, i0 + 1, :], eta_yee, eta0)) / 2.0
        B2 =  interp(field2[patch0, :, i0, :], eta, eta0)

        field1[patch1, :, NG:(Nxi + NG + 1), j1] = transform(patch0, patch1, xi0, eta0, B1, B2)[0][:, NG:(Nxi + NG + 1)]

        if (typ == "vect"):

            i0 = Nxi + NG - 1 # Last active cell of xi edge of patch0
            j1 = NG - 1 # First ghost cell on eta edge of patch1

            xi1 = xi_grid[:,j1] + 0.5 * dxi
            eta1 = eta_grid[:,j1] + 0.5 * deta

            xi0, eta0 = transform_coords(patch1, patch0, xi1, eta1)

            Br0 = interp(Br[patch0, :, i0, :], eta_yee, eta0)
            Br[patch1, :, NG:(Nxi + NG), j1] = Br0[:, NG:(Nxi + NG)]

    elif (top == 'yy'):

        #########
        # Communicate fields from eta left edge of patch1 to eta right edge of patch0
        ########

        j0 = Neta + NG # Last ghost cell of eta edge of patch0
        j1 = NG # First active cell of eta edge of patch1

        xi0 = xi_grid[:, j0] + 0.5 * dxi
        eta0 = eta_grid[:, j0] + 0.0 * deta

        xi1, eta1 = transform_coords(patch0, patch1, xi0, eta0)

        B1 = interp(field1[patch1, :, :, j1], xi, xi1)
        B2 = interp(field2[patch1, :, :, j1], xi_yee, xi1)

        field2[patch0, :, NG:(Nxi + NG), j0] = transform(patch1, patch0, xi1, eta1, B1, B2)[1][:, NG:(Nxi + NG)]

        #########
        # Communicate fields from eta right edge of patch0 to eta left edge patch1
        ########

        j0 = Neta + NG - 1 # Last active cell of eta edge of patch0
        j1 = NG - 1 # First ghost cell of eta edge of patch1

        xi1 = xi_grid[:, j1] + 0.0 * dxi
        eta1 = eta_grid[:, j1] + 0.5 * deta

        xi0, eta0 = transform_coords(patch1, patch0, xi1, eta1)

        B1 =  interp(field1[patch0, :, :, j0], xi, xi0)
        B2 = (interp(field2[patch0, :, :, j0], xi_yee, xi0) + interp(field2[patch0, :, :, j0 + 1], xi_yee, xi0)) / 2.0

        field1[patch1, :, NG:(Nxi + NG), j1] = transform(patch0, patch1, xi0, eta0, B1, B2)[0][:, NG:(Nxi + NG)]

        if (typ == "vect"):

            j0 = Neta + NG - 1 # Last active cell of eta edge of patch0
            j1 = NG - 1 # First ghost cell of eta edge of patch1

            xi1 = xi_grid[:,j1] + 0.5 * dxi
            eta1 = eta_grid[:,j1] + 0.5 * deta

            xi0, eta0 = transform_coords(patch1, patch0, xi1, eta1)

            Br0 = interp(Br[patch0, :, :, j0], xi_yee, xi0)
            Br[patch1, :, NG:(Nxi + NG), j1] = Br0[:, NG:(Nxi + NG)]

    elif (top == 'yx'):

        #########
        # Communicate fields from eta left edge of patch1 to xi right edge of patch0
        ########

        j0 = Neta + NG # Last ghost cell of eta edge of patch0
        i1 = NG # First active cell of xi edge of patch1

        xi0 = xi_grid[:, j0] + 0.5 * dxi
        eta0 = eta_grid[:, j0] + 0.0 * deta

        xi1, eta1 = transform_coords(patch0, patch1, xi0, eta0)

        B1 = interp(field1[patch1, :, i1, :], eta_yee, eta1)
        B2 = interp(field2[patch1, :, i1, :], eta, eta1)

        field2[patch0, :, NG:(Nxi + NG), j0] = transform(patch1, patch0, xi1, eta1, B1, B2)[1][:, NG:(Nxi + NG)]

        #########
        # Communicate fields from eta right edge of patch0 to xi left edge patch1
        ########

        j0 = Neta + NG - 1 # Last active cell of eta edge of patch0
        i1 = NG - 1 # First ghost cell on xi edge of patch1

        xi1 = xi_grid[i1, :] + 0.5 * dxi
        eta1 = eta_grid[i1, :] + 0.0 * deta

        xi0, eta0 = transform_coords(patch1, patch0, xi1, eta1)

        B1 =  interp(field1[patch0, :, :, j0], xi, xi0)
        B2 = (interp(field2[patch0, :, :, j0], xi_yee, xi0) + interp(field2[patch0, :, :, j0 + 1], xi_yee, xi0)) / 2.0

        field2[patch1, :, i1, NG:(Nxi + NG + 1)] = transform(patch0, patch1, xi0, eta0, B1, B2)[1][:, NG:(Nxi + NG + 1)]

        if (typ == "vect"):

            j0 = Neta + NG - 1 # Last active cell of eta edge of patch0
            i1 = NG - 1 # First ghost cell on xi edge of patch1

            xi1 = xi_grid[i1, :] + 0.5 * dxi
            eta1 = eta_grid[i1, :] + 0.5 * deta

            xi0, eta0 = transform_coords(patch1, patch0, xi1, eta1)

            Br0 = interp(Br[patch0, :, :, j0], xi_yee, xi0)
            Br[patch1, :, i1, NG:(Nxi + NG)] = Br0[:, NG:(Nxi + NG)]

    else:
        return

########
# Update Er at two special poles
# Inteprolated right now.
# To do: self-consistent evolution.
########

def update_poles():

    Er_mean_1 = (Er[Sphere.B, :, Nxi + NG - 1, NG]  + Er[Sphere.C, :, Nxi + NG - 1, NG]  + Er[Sphere.S, :, Nxi + NG - 1, NG])  / 3.0
    Er_mean_2 = (Er[Sphere.A, :, NG, Neta + NG - 1] + Er[Sphere.D, :, NG, Neta + NG - 1] + Er[Sphere.N, :, NG, Neta + NG - 1]) / 3.0
    Er[Sphere.A, :, NG, Neta + NG] = Er_mean_2
    Er[Sphere.B, :, Nxi + NG, NG]  = Er_mean_1
    Er[Sphere.C, :, Nxi + NG, NG]  = Er_mean_1
    Er[Sphere.D, :, NG, Neta + NG] = Er_mean_2
    Er[Sphere.N, :, NG, Neta + NG] = Er_mean_2
    Er[Sphere.S, :, Nxi + NG, NG]  = Er_mean_1

########
# Boundary conditions at r_min
########

omega = 0.0 # Angular velocity of the conductor at r_min

# Fields at r_min
E1_surf = N.zeros((6, Nxi + 2 * NG, Neta + 2 * NG))
E2_surf = N.zeros((6, Nxi + 2 * NG, Neta + 2 * NG))
Br_surf = N.zeros((6, Nxi + 2 * NG, Neta + 2 * NG))

for patch in range(6):

    fvec = (globals()["vec_sph_to_" + sphere[patch]])
    fcoord = (globals()["coord_" + sphere[patch] + "_to_sph"])

    for i in range(Nxi + 2 * NG):
        for j in range(Neta + 2 * NG):

            r0 = r[0]
            th0, ph0 = fcoord(xi_grid[i, j], eta_grid[i, j])
            x0, y0, z0 = coord_sph_to_cart(r0, th0, ph0)
            omega_sph = vec_cart_to_sph(x0, y0, z0, 0.0, 0.0, omega)
            omega_patch = fvec(th0, ph0, omega_sph[1], omega_sph[2])

            Br_surf[patch, i, j] = 1.0 / (r[0]**2) # 2.0 * N.cos(th0) / (r[0]**3)
            E1_surf[patch, i, j] = - omega_patch[0] * Br_surf[patch, i, j] / sqrt_det_g[0, i, j, 0]
            E2_surf[patch, i, j] =   omega_patch[1] * Br_surf[patch, i, j] / sqrt_det_g[0, i, j, 1]

            ########
            # Initial nonzero field
            ########

            Br[patch, :, i, j] = 1.0 / (r**2)

##### WARNING: what was just defined is actually covariant E, not contravariant
##### TO DO: switch from covariant to contravariant before assigning!

def BC_E_metal_rmin(patch):
    E1u[patch, 0, :, :] = E1_surf[patch, :, :]
    E2u[patch, 0, :, :] = E2_surf[patch, :, :]

def BC_B_metal_rmin(patch):
    Br[patch, 0, :, :] = Br_surf[patch, :, :]
    Br[patch, 0, :, :] = Br_surf[patch, :, :]

########
# Absorbing boundary conditions at r_max
########

i_abs = 5 # Thickness of absorbing layer in number of cells
r_abs = r[Nr - i_abs]
delta = ((r - r_abs) / (r_max - r_abs)) * N.heaviside(r - r_abs, 0.0)
sigma = N.exp(- 10.0 * delta**3)

delta = ((r_yee - r_abs) / (r_max - r_abs)) * N.heaviside(r_yee - r_abs, 0.0)
sigma_yee = N.exp(- 10.0 * delta**3)

def BC_B_absorb(patch):
    for k in range(Nr - i_abs, Nr):
        Br[patch, k, :, :]   *= sigma[k]
        B1u[patch, k, :, :] *= sigma_yee[k]
        B2u[patch, k, :, :] *= sigma_yee[k]

def BC_E_absorb(patch):
    for k in range(Nr - i_abs, Nr):
        Er[patch, k, :, :]  *= sigma_yee[k]
        E1u[patch, k, :, :] *= sigma[k]
        E2u[patch, k, :, :] *= sigma[k]

########
# Boundary conditions at r_max
########

def BC_E_metal_rmax(patch):
    E1u[patch, -1, :, :] = 0.0
    E2u[patch, -1, :, :] = 0.0

def BC_B_metal_rmax(patch):
    Br[patch, -1, :, :] = 0.0

########
# Plotting fields on an unfolded sphere
########

# Figure parameters
scale, aspect = 2.0, 0.7
vm = 0.2
ratio = 2.0
fig_size=deffigsize(scale, aspect)

xi_grid_c, eta_grid_c = unflip_eq(xi_grid, eta_grid)
xi_grid_d, eta_grid_d = unflip_eq(xi_grid, eta_grid)
xi_grid_n, eta_grid_n = unflip_po(xi_grid, eta_grid)

def plot_fields_unfolded(it, field, vm, index):

    fig = P.figure(1, facecolor='w', figsize=fig_size)
    ax = P.subplot(111)

    tab = (globals()[field])

    ax.pcolormesh(xi_grid[NG:(Nxi + NG), NG:(Neta + NG)], eta_grid[NG:(Nxi + NG), NG:(Neta + NG)], tab[Sphere.A, index, NG:(Nxi + NG), NG:(Neta + NG)], cmap = "RdBu_r", vmin = - vm, vmax = vm)
    ax.pcolormesh(xi_grid[NG:(Nxi + NG), NG:(Neta + NG)] + N.pi / 2.0, eta_grid[NG:(Nxi + NG), NG:(Neta + NG)], tab[Sphere.B, index, NG:(Nxi + NG), NG:(Neta + NG)], cmap = "RdBu_r", vmin = - vm, vmax = vm)
    ax.pcolormesh(xi_grid[NG:(Nxi + NG), NG:(Neta + NG)], eta_grid[NG:(Nxi + NG), NG:(Neta + NG)] - N.pi / 2.0, tab[Sphere.S, index, NG:(Nxi + NG), NG:(Neta + NG)], cmap = "RdBu_r", vmin = - vm, vmax = vm)

    ax.pcolormesh(xi_grid_c[NG:(Nxi + NG), NG:(Neta + NG)] + N.pi, eta_grid_c[NG:(Nxi + NG), NG:(Neta + NG)], tab[Sphere.C, index, NG:(Nxi + NG), NG:(Neta + NG)], cmap = "RdBu_r", vmin = - vm, vmax = vm)
    ax.pcolormesh(xi_grid_d[NG:(Nxi + NG), NG:(Neta + NG)] - N.pi / 2.0, eta_grid_d[NG:(Nxi + NG), NG:(Neta + NG)], tab[Sphere.D, index, NG:(Nxi + NG), NG:(Neta + NG)], cmap = "RdBu_r", vmin = - vm, vmax = vm)
    ax.pcolormesh(xi_grid_n[NG:(Nxi + NG), NG:(Neta + NG)], eta_grid_n[NG:(Nxi + NG), NG:(Neta + NG)] + N.pi / 2.0, tab[Sphere.N, index, NG:(Nxi + NG), NG:(Neta + NG)], cmap = "RdBu_r", vmin = - vm, vmax = vm)

    figsave_png(fig, outdir + field + "_" + str(it))

    P.close("all")

########
# Initialization
########

for p in range(6):
    pass # All fields to zero

iter = 0
idump = 0

Nt = 700 # Number of iterations
FDUMP = 100 # Dump frequency

InitialData()
WriteCoordsHDF5()

########
# Main routine
########

for it in tqdm(range(Nt), "Progression"):
    if ((it % FDUMP) == 0):
        plot_fields_unfolded(idump, "B2u", 0.05, 10)
        WriteAllFieldsHDF5(idump)
        idump += 1

    for p in range(6):
        contra_to_cov_E(p)

    for i in range(n_zeros):
        p0, p1 = index_row[i], index_col[i]
        communicate_E_patch_cov(p0, p1)

    for p in range(6):
        push_B(p)

    for p in range(6):
        BC_B_metal_rmin(p)
        BC_B_metal_rmax(p)
        BC_B_absorb(p)

    for i in range(n_zeros):
        p0, p1 = index_row[i], index_col[i]
        communicate_B_patch_contra(p0, p1)

    for p in range(6):
        contra_to_cov_B(p)

    for i in range(n_zeros):
        p0, p1 = index_row[i], index_col[i]
        communicate_B_patch_cov(p0, p1)

    for p in range(6):
        push_E(p)

    for p in range(6):
        BC_E_metal_rmin(p)
        BC_E_metal_rmax(p)
        BC_E_absorb(p)

    update_poles()

    for i in range(n_zeros):
        p0, p1 = index_row[i], index_col[i]
        communicate_E_patch_contra(p0, p1)
