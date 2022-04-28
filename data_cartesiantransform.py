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

outdir = '/Users/jensmahlmann/Documents/Projects/Development/CubedSphere/'
NG = 1

h5f = h5py.File(outdir+'grid'+'.'+ str(0).rjust(5, '0') +'.h5', 'r')

r = N.array(h5f['r'])
xi = N.array(h5f['xi'])
eta = N.array(h5f['eta'])

h5f.close()

Nr = len(r)
Nxi = len(xi) - 2*NG
Neta = len(eta) - 2*NG

print('Loaded grid with Nr x Nxi x Neta = '+str(Nr)+' x '+str(Nxi)+' x '+str(Neta))

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
Er  = N.zeros((6, Nr, Nxi + 2 * NG, Neta + 2 * NG,))
E1u = N.zeros((6, Nr, Nxi + 2 * NG, Neta + 2 * NG,))
E2u = N.zeros((6, Nr, Nxi + 2 * NG, Neta + 2 * NG,))
Br  = N.zeros((6, Nr, Nxi + 2 * NG, Neta + 2 * NG,))
B1u = N.zeros((6, Nr, Nxi + 2 * NG, Neta + 2 * NG,))
B2u = N.zeros((6, Nr, Nxi + 2 * NG, Neta + 2 * NG,))

def ReadFieldHDF5(it, field):

    invec = (globals()[field])
    h5f = h5py.File(outdir+field+'.'+ str(it).rjust(5, '0') +'.h5', 'r')

    for patch in range(6):
        fieldin = N.array(h5f[field+str(patch)])
        invec[patch, :, :, :] = fieldin

    h5f.close()

ReadFieldHDF5(0, "Br")
ReadFieldHDF5(0, "B1u")
ReadFieldHDF5(0, "B2u")
