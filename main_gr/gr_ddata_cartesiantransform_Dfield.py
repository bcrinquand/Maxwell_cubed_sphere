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

from gr_metric import *

outdir = '/home/bcrinqua/GitHub/Maxwell_cubed_sphere/data_3d_gr/'

num = sys.argv[1]

ONt = 60
ONp = 60
Nxyz = 60

a = 0.5

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
Dru = N.zeros((n_patches, Nr, Nxi_int, Neta_int))
D1u = N.zeros((n_patches, Nr, Nxi_half, Neta_int))
D2u = N.zeros((n_patches, Nr, Nxi_int, Neta_half))

Dr_c = N.zeros((n_patches, Nr0, Nxi, Neta))
D1_c = N.zeros((n_patches, Nr0, Nxi, Neta))
D2_c = N.zeros((n_patches, Nr0, Nxi, Neta))

dth = N.pi / ONt
dph = N.pi / ONp

Ot = N.linspace(0, N.pi, ONt)
Op = N.linspace(- N.pi, N.pi, ONp)

# Ot = dth * N.arange(ONt)
# Op = -N.pi + dph * N.arange(ONp)

DrS = N.zeros((Nr0, ONt, ONp))
DtS = N.zeros((Nr0, ONt, ONp))
DpS = N.zeros((Nr0, ONt, ONp))

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

ReadFieldHDF5(num, "Dru")
ReadFieldHDF5(num, "D1u")
ReadFieldHDF5(num, "D2u")

########
# Interpolate to center of cells
########

def compute_D_nodes(p):

    Dr_c[p, :, :, :] = 0.25 * (Dru[p, NG:(Nr0 + NG), :-1, :-1] + N.roll(Dru, -1, axis = 2)[p, NG:(Nr0 + NG), :-1, :-1] + N.roll(Dru, -1, axis = 3)[p, NG:(Nr0 + NG), :-1, :-1] \
                            +  N.roll(N.roll(Dru, -1, axis = 2), -1, axis = 3)[p, NG:(Nr0 + NG), :-1, :-1])    
    
    D1_c[p, :, :, :] = 0.25 * (D1u[p, NG:(Nr0 + NG), 1:-1, :-1] + N.roll(D1u, -1, axis = 1)[p, NG:(Nr0 + NG), 1:-1, :-1] + N.roll(D1u, -1, axis = 3)[p, NG:(Nr0 + NG), 1:-1, :-1] \
                            +  N.roll(N.roll(D1u, -1, axis = 1), -1, axis = 3)[p, NG:(Nr0 + NG), 1:-1, :-1])

    D2_c[p, :, :, :] = 0.25 * (D2u[p, NG:(Nr0 + NG), :-1, 1:-1] + N.roll(D2u, -1, axis = 1)[p, NG:(Nr0 + NG), :-1, 1:-1] + N.roll(D2u, -1, axis = 2)[p, NG:(Nr0 + NG), :-1, 1:-1] \
                            +  N.roll(N.roll(D2u, -1, axis = 1), -1, axis = 2)[p, NG:(Nr0 + NG), :-1, 1:-1])

########
# Interpolate Dr from CS to Spherical KS positions
print('Interpolate Dr from CS to Spherical positions')
########

compute_D_nodes(range(n_patches))

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

        PB1D = PB1D + (Dr_c[patch, h, :, :] * r_yee[NG + h]).flatten().tolist()


    PB1D = N.array(PB1D)

    DrS[h] = griddata( (Pt1D.flatten(), Pp1D.flatten()), PB1D.flatten(), (Ot[None,:], Op[:,None]), method='nearest').transpose()

########
# Interpolate D1u/D2u from CS to Spherical KS positions and rotate
print('Interpolate D1u/D2u from CS to Spherical positions and rotate')
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

        D1uC = D1_c[patch, h, :, :]
        D2uC = D2_c[patch, h, :, :]

        Dtu, Dpu = fvec(xi_grid[:, :], eta_grid[:, :], D1uC, D2uC)

        PB1D = PB1D + Dtu.flatten().tolist()
        PB2D = PB2D + Dpu.flatten().tolist()
        
    PB1D = N.array(PB1D)
    PB2D = N.array(PB2D)

    DtS[h] = griddata( (Pt1D.flatten(), Pp1D.flatten()), PB1D.flatten(), (Ot[None,:], Op[:,None]), method='nearest').transpose()
    DpS[h] = griddata( (Pt1D.flatten(), Pp1D.flatten()), PB2D.flatten(), (Ot[None,:], Op[:,None]), method='nearest').transpose()

########
# Interpolate to Cartesian KS coordinates
print('Interpolate to Cartesian KS coordinates with Nx x Ny x Nz = '+str(Nxyz)+' x '+str(Nxyz)+' x '+str(Nxyz))
########

R, THETA, PHI = N.meshgrid(r_yee[NG:(Nr0 + NG)], Ot, Op)

R = N.transpose(R, (1,0,2))
THETA = N.transpose(THETA, (1,0,2))
PHI = N.transpose(PHI, (1,0,2))

X = (R * N.cos(PHI) - a * N.sin(PHI)) * N.sin(THETA)
Y = (R * N.sin(PHI) + a * N.cos(PHI)) * N.sin(THETA)
Z =  R * N.cos(THETA)

Txr = (X * R + a * Y) / (R * R + a * a)
Txt = N.divide(X * Z * N.sqrt(R * R + a * a), R * N.sqrt(X * X + Y * Y), out=N.zeros_like(X * Z), where=N.sqrt(X * X + Y * Y)!=0)
Txp = -Y
Tyr = (Y * R - a * X) / (R * R + a * a)
Tyt = N.divide(Y * Z * N.sqrt(R * R + a * a), R * N.sqrt(X * X + Y * Y), out=N.zeros_like(X * Z), where=N.sqrt(X * X + Y * Y)!=0)
Typ = X
Tzr = Z / R
Tzt = - R * N.sqrt(X * X + Y * Y) / N.sqrt(R * R + a * a)
Tzp = 0

data_xc = Txr * DrS + Txt * DtS + Txp * DpS
data_yc = Tyr * DrS + Tyt * DtS + Typ * DpS
data_zc = Tzr * DrS + Tzt * DtS + Tzp * DpS

R1d = N.transpose(R,(2,1,0))[0][0]
Theta1d = N.transpose(THETA,(0,2,1))[0][0]
Phi1d = N.transpose(PHI,(0,1,2))[0][0]

my_interpolating_function_x = RegularGridInterpolator((R1d, Theta1d, Phi1d), data_xc, bounds_error=False, fill_value = 0.0, method='linear')
my_interpolating_function_y = RegularGridInterpolator((R1d, Theta1d, Phi1d), data_yc, bounds_error=False, fill_value = 0.0, method='linear')
my_interpolating_function_z = RegularGridInterpolator((R1d, Theta1d, Phi1d), data_zc, bounds_error=False, fill_value = 0.0, method='linear')

my_interpolating_function_r = RegularGridInterpolator((R1d, Theta1d, Phi1d), DrS, bounds_error=False, fill_value = 0.0, method='linear')
my_interpolating_function_t = RegularGridInterpolator((R1d, Theta1d, Phi1d), DtS, bounds_error=False, fill_value = 0.0, method='linear')
my_interpolating_function_p = RegularGridInterpolator((R1d, Theta1d, Phi1d), DpS, bounds_error=False, fill_value = 0.0, method='linear')

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

Dsq = N.sqrt(datax**2 + datay**2 + dataz**2)

Dr_pol  = N.mean(DrS, axis = 2)
Dth_pol = N.mean(DtS, axis = 2)
Dph_pol = N.mean(DpS, axis = 2)

Dr_tor  = DrS[:, int(ONt/2), :]
Dth_tor = DtS[:, int(ONt/2), :]
Dph_tor = DpS[:, int(ONt/2), :]

h5f = h5py.File(outdir+'Dsq'+'_'+ str(num).rjust(5, '0') +'.h5', 'w')
h5f.create_dataset('Dsq', data=Dsq)
h5f.close()

h5f = h5py.File(outdir+'Dxyz'+'_'+ str(num).rjust(5, '0') +'.h5', 'w')
h5f.create_dataset('Dx', data=datax)
h5f.create_dataset('Dy', data=datay)
h5f.create_dataset('Dz', data=dataz)
h5f.close()

h5f = h5py.File(outdir+'Drthph'+'_'+ str(num).rjust(5, '0') +'.h5', 'w')
h5f.create_dataset('Dr_pol',  data=Dr_pol)
h5f.create_dataset('Dth_pol', data=Dth_pol)
h5f.create_dataset('Dph_pol', data=Dph_pol)
h5f.create_dataset('Dr_tor',  data=Dr_tor)
h5f.create_dataset('Dth_tor', data=Dth_tor)
h5f.create_dataset('Dph_tor', data=Dph_tor)
h5f.close()

# h5f = h5py.File(outdir+'Brthph_cart'+'_'+ str(num).rjust(5, '0') +'.h5', 'w')
# h5f.create_dataset('Br', data=datar)
# h5f.create_dataset('Bth', data=datat)
# h5f.create_dataset('Bph', data=datap)
# h5f.close()

h5f = h5py.File(outdir+'grid_xyz.h5', 'w')
h5f.create_dataset('xpos', data=xi)
h5f.create_dataset('ypos', data=yi)
h5f.create_dataset('zpos', data=zi)
h5f.close()

h5f = h5py.File(outdir+'grid_rthph.h5', 'w')
h5f.create_dataset('rpos',  data=r_yee[NG:(Nr0 + NG)])
h5f.create_dataset('thpos', data=Ot)
h5f.create_dataset('phpos', data=Op)
h5f.close()
