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

import sys
sys.path.append('../')
sys.path.append('../transformations/')

from vec_transformations_flip import *
from coord_transformations_flip import *

outdir = '/home/bcrinqua/GitHub/Maxwell_cubed_sphere/data_3d/'

num = sys.argv[1]

ONt = 100
ONp = 200
Nxyz = 100

n_patches = 6
# Connect patch indices and names
sphere = {0: "A", 1: "B", 2: "C", 3: "D", 4: "N", 5: "S"}

########
# Read grid dimensions
########

h5f = h5py.File(outdir+'grid.h5', 'r')

r = N.array(h5f['r'])
xi_int = N.array(h5f['xi_int'])
eta_int = N.array(h5f['eta_int'])
xi_half = N.array(h5f['xi_half'])
eta_half = N.array(h5f['eta_half'])

h5f.close()

NG = 1

Nr  = len(r)
Nr0 = len(r) - 2 * NG
Nxi_int = len(xi_int)
Neta_int = len(eta_int)
Nxi_half = len(xi_half)
Neta_half = len(eta_half)
Nxi = Nxi_int - 1
Neta = Neta_int - 1

# print('Loaded grid with Nr x Nxi x Neta = '+str(Nr0)+' x '+str(Nxi_int)+' x '+str(Neta_int))
print('File number {}'.format(num))
########
# Establish grid infrastructure
########


r_min, r_max = r[1], r[-1]
xi_min, xi_max = - N.pi / 4.0, N.pi / 4.0
eta_min, eta_max = - N.pi / 4.0, N.pi / 4.0
dr = (r_max - r_min) / Nr0
dxi = (xi_max - xi_min) / Nxi
deta = (eta_max - eta_min) / Neta
r_yee = r + 0.5 * dr

# Initialize fields
Er  = N.zeros((n_patches, Nr, Nxi_int, Neta_int))
E1u = N.zeros((n_patches, Nr, Nxi_half, Neta_int))
E2u = N.zeros((n_patches, Nr, Nxi_int, Neta_half))

Er_c = N.zeros((n_patches, Nr0, Nxi, Neta))
E1_c = N.zeros((n_patches, Nr0, Nxi, Neta))
E2_c = N.zeros((n_patches, Nr0, Nxi, Neta))

Ot = N.linspace(0, N.pi, ONt)
Op = N.linspace(- N.pi, N.pi, ONp)

ErS = N.zeros((Nr0, ONt, ONp))
EtS = N.zeros((Nr0, ONt, ONp))
EpS = N.zeros((Nr0, ONt, ONp))

def ReadFieldHDF5(it, field):

    invec = (globals()[field])
    h5f = h5py.File(outdir + field + '_' + str(it).rjust(5, '0') + '.h5', 'r')

    for patch in range(6):
        fieldin = N.array(h5f[field + str(patch)])
        invec[patch, :, :, :] = fieldin

    h5f.close()
    
########
# Read some fields in cubed sphere coordinates
########

ReadFieldHDF5(num, "Er")
ReadFieldHDF5(num, "E1u")
ReadFieldHDF5(num, "E2u")

########
# Interpolate to center of cells
########

def compute_E_nodes(p):

    Er_c[p, :, :, :] = 0.25 * (Er[p, NG:(Nr0 + NG), :-1, :-1] + N.roll(Er, -1, axis = 2)[p, NG:(Nr0 + NG), :-1, :-1] + N.roll(Er, -1, axis = 3)[p, NG:(Nr0 + NG), :-1, :-1] \
                             + N.roll(N.roll(Er, -1, axis = 2), -1, axis = 3)[p, NG:(Nr0 + NG), :-1, :-1])    
    
    E1_c[p, :, :, :] = 0.25 * (E1u[p, NG:(Nr0 + NG), 1:-1, :-1] + N.roll(E1u, -1, axis = 1)[p, NG:(Nr0 + NG), 1:-1, :-1] + N.roll(E1u, -1, axis = 3)[p, NG:(Nr0 + NG), 1:-1, :-1] \
                             + N.roll(N.roll(E1u, -1, axis = 1), -1, axis = 3)[p, NG:(Nr0 + NG), 1:-1, :-1])

    E2_c[p, :, :, :] = 0.25 * (E2u[p, NG:(Nr0 + NG), :-1, 1:-1] + N.roll(E2u, -1, axis = 1)[p, NG:(Nr0 + NG), :-1, 1:-1] + N.roll(E2u, -1, axis = 2)[p, NG:(Nr0 + NG), :-1, 1:-1] \
                             + N.roll(N.roll(E2u, -1, axis = 1), -1, axis = 2)[p, NG:(Nr0 + NG), :-1, 1:-1])

########
# Interpolate Er from CS to Spherical positions
print('Interpolate Er from CS to Spherical positions')
########

compute_E_nodes(range(n_patches))

Pt1D = []
Pp1D = []

eta_grid, xi_grid = N.meshgrid(eta_half[1:-1], xi_half[1:-1])

for patch in range(6):

    fcoord = (globals()["coord_" + sphere[patch] + "_to_sph"])
    for i in range(Nxi):
        for j in range(Neta):

            th0, ph0 = fcoord(xi_int[i] + 0.5 * dxi, eta_int[j] + 0.5 * deta)
            Pt1D.append(th0)
            Pp1D.append(ph0)

Pt1D = N.array(Pt1D)
Pp1D = N.array(Pp1D)

for h in tqdm(range(Nr0)):

    PB1D = []

    for patch in range(6):

        # for i in range(Nxi):
        #     for j in range(Neta):

        #             PB1D.append(Br_c[patch, h, i, j])

        PB1D = PB1D + (Er_c[patch, h, :, :]).flatten().tolist()


    PB1D = N.array(PB1D)

    ErS[h] = griddata( (Pt1D.flatten(), Pp1D.flatten()), PB1D.flatten(), (Ot[None,:], Op[:,None]), method='nearest').transpose()

########
# Interpolate E1u/E2u from CS to Spherical positions and rotate
print('Interpolate E1u/E2u from CS to Spherical positions and rotate')
########

for h in tqdm(range(Nr0)):

    PB1D = []
    PB2D = []

    # for patch in range(6):

    #     fvec = (globals()["vec_" + sphere[patch] + "_to_sph"])

    #     for i in range(Nxi):
    #         for j in range(Neta):

    #                 B1uC = B1_c[patch, h, i, j]
    #                 B2uC = B2_c[patch, h, i, j]

    #                 Btu, Bpu = fvec(xi_int[i] + 0.5 * dxi, eta_int[j] + 0.5 * dxi, B1uC, B2uC)

    #                 PB1D.append(Btu)
    #                 PB2D.append(Bpu)



    for patch in range(6):

        fvec = (globals()["vec_" + sphere[patch] + "_to_sph"])

        E1uC = E1_c[patch, h, :, :]
        E2uC = E2_c[patch, h, :, :]

        Etu, Epu = fvec(xi_grid[:, :], eta_grid[:, :], E1uC, E2uC)

        PB1D = PB1D + Etu.flatten().tolist()
        PB2D = PB2D + Epu.flatten().tolist()
        
    PB1D = N.array(PB1D)
    PB2D = N.array(PB2D)

    EtS[h] = griddata( (Pt1D.flatten(), Pp1D.flatten()), PB1D.flatten(), (Ot[None,:], Op[:,None]), method='nearest').transpose()
    EpS[h] = griddata( (Pt1D.flatten(), Pp1D.flatten()), PB2D.flatten(), (Ot[None,:], Op[:,None]), method='nearest').transpose()

########
# Interpolate to Cartesian coordinates
print('Interpolate to Cartesian coordinates with Nx x Ny x Nz = '+str(Nxyz)+' x '+str(Nxyz)+' x '+str(Nxyz))
########

R, THETA, PHI = N.meshgrid(r_yee[NG:(Nr0 + NG)], Ot, Op)

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
Tyt = N.divide(Y * Z, N.sqrt(X*X+Y*Y), out=N.zeros_like(Y * Z), where=N.sqrt(X*X+Y*Y)!=0)
Typ = X
Tzr = N.divide(Z, N.sqrt(X*X+Y*Y+Z*Z), out=N.zeros_like(X), where=N.sqrt(X*X+Y*Y+Z*Z)!=0)
Tzt = - N.sqrt(X*X+Y*Y)
Tzp = 0

data_xc = Txr * ErS + Txt * EtS + Txp * EpS
data_yc = Tyr * ErS + Tyt * EtS + Typ * EpS
data_zc = Tzr * ErS + Tzt * EtS + Tzp * EpS

R1d = N.transpose(R,(2,1,0))[0][0]
Theta1d = N.transpose(THETA,(0,2,1))[0][0]
Phi1d = N.transpose(PHI,(0,1,2))[0][0]

my_interpolating_function_x = RegularGridInterpolator((R1d, Theta1d, Phi1d), data_xc, bounds_error=False, fill_value = 0.0, method='linear')
my_interpolating_function_y = RegularGridInterpolator((R1d, Theta1d, Phi1d), data_yc, bounds_error=False, fill_value = 0.0, method='linear')
my_interpolating_function_z = RegularGridInterpolator((R1d, Theta1d, Phi1d), data_zc, bounds_error=False, fill_value = 0.0, method='linear')

my_interpolating_function_r = RegularGridInterpolator((R1d, Theta1d, Phi1d), ErS, bounds_error=False, fill_value = 0.0, method='linear')
my_interpolating_function_t = RegularGridInterpolator((R1d, Theta1d, Phi1d), EtS, bounds_error=False, fill_value = 0.0, method='linear')
my_interpolating_function_p = RegularGridInterpolator((R1d, Theta1d, Phi1d), EpS, bounds_error=False, fill_value = 0.0, method='linear')

yi = N.linspace(-r_max,r_max,Nxyz)
xi = N.linspace(-r_max,r_max,Nxyz)
zi = N.linspace(-r_max,r_max,Nxyz)
xv, yv, zv = N.meshgrid(xi, yi, zi)
datax_out = N.zeros_like(xv)
datay_out = N.zeros_like(xv)
dataz_out = N.zeros_like(xv)

datar_out = N.zeros_like(xv)
datat_out = N.zeros_like(xv)
datap_out = N.zeros_like(xv)

rtmp = N.sqrt(xv**2 + yv**2 + zv**2)
thetatmp = N.arccos(zv / N.sqrt(xv**2 + yv**2 + zv**2))
phitmp = N.arctan2(yv, xv)

rf = rtmp.flatten()
thf = thetatmp.flatten()
phf = phitmp.flatten()
Ntot=Nxyz**3
flat = N.array([[rf[i] for i in range(Ntot)],[thf[i] for i in range(Ntot)],[phf[i] for i in range(Ntot)]])

datax_out = my_interpolating_function_x(flat.T)
datay_out = my_interpolating_function_y(flat.T)
dataz_out = my_interpolating_function_z(flat.T)

datax = datax_out.reshape((Nxyz,Nxyz,Nxyz))
datay = datay_out.reshape((Nxyz,Nxyz,Nxyz))
dataz = dataz_out.reshape((Nxyz,Nxyz,Nxyz))

# datar_out = my_interpolating_function_r(flat.T)
# datat_out = my_interpolating_function_t(flat.T)
# datap_out = my_interpolating_function_p(flat.T)

# datar = datar_out.reshape((Nxyz,Nxyz,Nxyz))
# datat = datat_out.reshape((Nxyz,Nxyz,Nxyz))
# datap = datap_out.reshape((Nxyz,Nxyz,Nxyz))

# for i in tqdm(range(0, N.shape(datax_out)[0])):
#     for j in range(0, N.shape(datax_out)[1]):
#         for k in range(0, N.shape(datax_out)[2]):
#             rtmp = N.sqrt(xv[i,j,k]**2 + yv[i,j,k]**2 + zv[i,j,k]**2)
#             thetatmp = N.arccos(zv[i,j,k] / N.sqrt(xv[i,j,k]**2 + yv[i,j,k]**2 + zv[i,j,k]**2))
#             phitmp = N.arctan2(yv[i,j,k], xv[i,j,k])

#             datax_out[i,j,k] = my_interpolating_function_x([rtmp,thetatmp,phitmp])
#             datay_out[i,j,k] = my_interpolating_function_y([rtmp,thetatmp,phitmp])
#             dataz_out[i,j,k] = my_interpolating_function_z([rtmp,thetatmp,phitmp])

#             # datar_out[i,j,k] = my_interpolating_function_r([rtmp,thetatmp,phitmp])
#             # datat_out[i,j,k] = my_interpolating_function_t([rtmp,thetatmp,phitmp])
#             # datap_out[i,j,k] = my_interpolating_function_p([rtmp,thetatmp,phitmp])

Esq = N.sqrt(datax**2 + datay**2 + dataz**2)

Er_avg  = N.mean(ErS, axis = 2)
Eth_avg = N.mean(EtS, axis = 2)
Eph_avg = N.mean(EpS, axis = 2)

Er_tor  = ErS[:, int(ONt/2), :]
Eth_tor = EtS[:, int(ONt/2), :]
Eph_tor = EpS[:, int(ONt/2), :]

h5f = h5py.File(outdir+'Esq'+'_'+ str(num).rjust(5, '0') +'.h5', 'w')
h5f.create_dataset('Bsq', data=Esq)
h5f.close()

h5f = h5py.File(outdir+'Exyz'+'_'+ str(num).rjust(5, '0') +'.h5', 'w')
h5f.create_dataset('Ex', data=datax)
h5f.create_dataset('Ey', data=datay)
h5f.create_dataset('Ez', data=dataz)
h5f.close()

h5f = h5py.File(outdir+'Erthph'+'_'+ str(num).rjust(5, '0') +'.h5', 'w')
h5f.create_dataset('Er_pol',  data=Er_avg)
h5f.create_dataset('Eth_pol', data=Eth_avg)
h5f.create_dataset('Eph_pol', data=Eph_avg)
h5f.create_dataset('Er_tor',  data=Er_tor)
h5f.create_dataset('Eth_tor', data=Eth_tor)
h5f.create_dataset('Eph_tor', data=Eph_tor)
h5f.close()

# h5f = h5py.File(outdir+'Erthph_cart'+'_'+ str(num).rjust(5, '0') +'.h5', 'w')
# h5f.create_dataset('Er', data=datar)
# h5f.create_dataset('Eth', data=datat)
# h5f.create_dataset('Eph', data=datap)
# h5f.close()
