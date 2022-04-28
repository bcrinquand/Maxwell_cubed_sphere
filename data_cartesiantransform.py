# Import modules
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

outdir = '/Users/jensmahlmann/Documents/Projects/Development/CubedSphere/'
NG = 1

ONt = 50
ONp = 100
Nxyz = 80

# Connect patch indices and names
sphere = {0: "A", 1: "B", 2: "C", 3: "D", 4: "N", 5: "S"}

########
# Read grid dimensions
########

h5f = h5py.File(outdir+'grid'+'.'+ str(0).rjust(5, '0') +'.h5', 'r')

r = N.array(h5f['r'])
xi = N.array(h5f['xi'])
eta = N.array(h5f['eta'])

h5f.close()

Nr = len(r)
Nxi = len(xi) - 2*NG
Neta = len(eta) - 2*NG

print('Loaded grid with Nr x Nxi x Neta = '+str(Nr)+' x '+str(Nxi)+' x '+str(Neta))

########
# Establish grid infrastructure
########

r_min, r_max = r[0], r[-1]
xi_min, xi_max = - N.pi / 4.0, N.pi / 4.0
eta_min, eta_max = - N.pi / 4.0, N.pi / 4.0
dr = (r_max - r_min) / Nr
dxi = (xi_max - xi_min) / Nxi
deta = (eta_max - eta_min) / Neta

eta_grid, xi_grid = N.meshgrid(eta, xi)
r_yee = r + 0.5 * dr
xi_yee  = xi  + 0.5 * dxi
eta_yee = eta + 0.5 * deta

# Initialize fields
Br  = N.zeros((6, Nr, Nxi + 2 * NG, Neta + 2 * NG,))
B1u = N.zeros((6, Nr, Nxi + 2 * NG, Neta + 2 * NG,))
B2u = N.zeros((6, Nr, Nxi + 2 * NG, Neta + 2 * NG,))

Ot = N.linspace(0, N.pi, ONt)
Op = N.linspace(- N.pi, N.pi, ONp, endpoint=False)

BrS = N.zeros((Nr-1, ONt, ONp,))
BtS = N.zeros((Nr-1, ONt, ONp,))
BpS = N.zeros((Nr-1, ONt, ONp,))


def ReadFieldHDF5(it, field):

    invec = (globals()[field])
    h5f = h5py.File(outdir+field+'.'+ str(it).rjust(5, '0') +'.h5', 'r')

    for patch in range(6):
        fieldin = N.array(h5f[field+str(patch)])
        invec[patch, :, :, :] = fieldin

    h5f.close()

########
# Read some fields in cubed sphere coordinetas
########

ReadFieldHDF5(6, "Br")
ReadFieldHDF5(6, "B1u")
ReadFieldHDF5(6, "B2u")

########
# Interpolate Br from CS to Spherical positions
print('Interpolate Br from CS to Spherical positions')
########

Pt1D = []
Pp1D = []

for patch in range(6):

    fcoord = (globals()["coord_" + sphere[patch] + "_to_sph"])
    for i in range(NG, Nxi + 1*NG):
        for j in range(NG, Neta + 1*NG):

            th0, ph0 = fcoord(xi_grid[i, j] + 0.5 * dxi, eta_grid[i, j] + 0.5 * dxi)
            Pt1D.append(th0)
            Pp1D.append(ph0)

Pt1D = N.array(Pt1D)
Pp1D = N.array(Pp1D)

for h in tqdm(range(Nr-1)):

    PB1D = []

    for patch in range(6):

        for i in range(NG, Nxi + 1*NG):
            for j in range(NG, Neta + 1*NG):

                    PB1D.append((Br[patch, h, i, j]+Br[patch, h+1, i, j])/2.0)

    PB1D = N.array(PB1D)

    BrS[h] = griddata( (Pt1D.flatten(), Pp1D.flatten()), PB1D.flatten(), (Ot[None,:], Op[:,None]), method='nearest').transpose()

########
# Interpolate B1u/B2u from CS to Spherical positions and rotate
print('Interpolate B1u/B2u from CS to Spherical positions and rotate')
########

for h in tqdm(range(Nr-1)):

    PB1D = []
    PB2D = []

    for patch in range(6):

        fvec = (globals()["vec_" + sphere[patch] + "_to_sph"])

        for i in range(NG, Nxi + 1*NG):
            for j in range(NG, Neta + 1*NG):

                    B1uC = (B1u[patch, h, i, j] + B1u[patch, h, i+1, j])/2.0
                    B2uC = (B2u[patch, h, i, j] + B2u[patch, h, i, j+1])/2.0

                    Btu, Bpu = fvec(xi_grid[i, j] + 0.5 * dxi, eta_grid[i, j] + 0.5 * dxi, B1uC, B2uC)

                    PB1D.append(Btu)
                    PB2D.append(Bpu)

    PB1D = N.array(PB1D)
    PB2D = N.array(PB2D)

    BtS[h] = griddata( (Pt1D.flatten(), Pp1D.flatten()), PB1D.flatten(), (Ot[None,:], Op[:,None]), method='nearest').transpose()
    BpS[h] = griddata( (Pt1D.flatten(), Pp1D.flatten()), PB2D.flatten(), (Ot[None,:], Op[:,None]), method='nearest').transpose()

########
# Interpolate to Cartesian coordinates
print('Interpolate to Cartesian coordinates with Nx x Ny x Nz = '+str(Nxyz)+' x '+str(Nxyz)+' x '+str(Nxyz))
########

R, THETA, PHI = N.meshgrid(r[0:Nr-1], Ot, Op)

R = N.transpose(R, (1,0,2))
THETA = N.transpose(THETA, (1,0,2))
PHI = N.transpose(PHI, (1,0,2))

X = R * N.sin(THETA) * N.cos(PHI)
Y = R * N.sin(THETA) * N.sin(PHI)
Z = R * N.cos(THETA)

Txr = N.divide(X, N.sqrt(X*X+Y*Y+Z*Z), out=N.zeros_like(X), where=N.sqrt(X*X+Y*Y+Z*Z)!=0)
Txt = N.divide(X * Z, N.sqrt(X*X+Y*Y), out=N.zeros_like(X * Z), where=N.sqrt(X*X+Y*Y)!=0)
Txp = -Y
Tyr = N.divide(Y, N.sqrt(X*X+Y*Y+Z*Z), out=N.zeros_like(X), where=N.sqrt(X*X+Y*Y+Z*Z)!=0)
Tyt = N.divide(Y * Z, N.sqrt(X*X+Y*Y), out=N.zeros_like(X * Z), where=N.sqrt(X*X+Y*Y)!=0)
Typ = X
Tzr = N.divide(Z, N.sqrt(X*X+Y*Y+Z*Z), out=N.zeros_like(X), where=N.sqrt(X*X+Y*Y+Z*Z)!=0)
Tzt = - N.sqrt(X*X+Y*Y)
Tzp = 0

data_xc = Txr * BrS + Txt * BtS + Txp * BpS
data_yc = Tyr * BrS + Tyt * BtS + Typ * BpS
data_zc = Tzr * BrS + Tzt * BtS + Tzp * BpS

R1d = N.transpose(R,(2,1,0))[0][0]
Theta1d = N.transpose(THETA,(0,2,1))[0][0]
Phi1d = N.transpose(PHI,(0,1,2))[0][0]

my_interpolating_function_x = RegularGridInterpolator((R1d, Theta1d, Phi1d), data_xc, bounds_error=False, fill_value = 0.0, method='linear')
my_interpolating_function_y = RegularGridInterpolator((R1d, Theta1d, Phi1d), data_yc, bounds_error=False, fill_value = 0.0, method='linear')
my_interpolating_function_z = RegularGridInterpolator((R1d, Theta1d, Phi1d), data_zc, bounds_error=False, fill_value = 0.0, method='linear')

yi = N.linspace(-r_max,r_max,Nxyz)
xi = N.linspace(-r_max,r_max,Nxyz)
zi = N.linspace(-r_max,r_max,Nxyz)
xv, yv, zv = N.meshgrid(xi, yi, zi)
datax_out = N.zeros_like(xv)
datay_out = N.zeros_like(xv)
dataz_out = N.zeros_like(xv)

for i in tqdm(range(0, N.shape(datax_out)[0])):
    for j in range(0, N.shape(datax_out)[1]):
        for k in range(0, N.shape(datax_out)[2]):
            rtmp = N.sqrt(xv[i,j,k]**2 + yv[i,j,k]**2 + zv[i,j,k]**2)
            thetatmp = N.arccos(zv[i,j,k] / N.sqrt(xv[i,j,k]**2 + yv[i,j,k]**2 + zv[i,j,k]**2))
            phitmp = N.arctan2(yv[i,j,k], xv[i,j,k])
            datax_out[i,j,k] = my_interpolating_function_x([rtmp,thetatmp,phitmp])
            datay_out[i,j,k] = my_interpolating_function_y([rtmp,thetatmp,phitmp])
            dataz_out[i,j,k] = my_interpolating_function_z([rtmp,thetatmp,phitmp])

h5f = h5py.File(outdir+'CARTfield'+'.'+ str(0).rjust(5, '0') +'.h5', 'w')
h5f.create_dataset('bx', data=datax_out)
h5f.create_dataset('by', data=datay_out)
h5f.create_dataset('bz', data=dataz_out)
h5f.create_dataset('xpos', data=xv)
h5f.create_dataset('ypos', data=yv)
h5f.create_dataset('zpos', data=zv)
h5f.close()
