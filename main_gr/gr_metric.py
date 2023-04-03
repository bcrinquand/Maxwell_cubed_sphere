import numpy as N

n_patches = 6
# Connect patch indices and names
sphere = {0: "A", 1: "B", 2: "C", 3: "D", 4: "N", 5: "S"}

import sys
sys.path.append('../')
sys.path.append('../transformations/')

from coord_transformations_flip import *
from jacob_transformations_flip import *

########
# Define metric in spherical coordinates
########

def grrd(patch, r, xi, eta, spin):
    fcoord = (globals()["coord_" + sphere[patch] + "_to_sph"])
    theta = fcoord(xi, eta)[0]
    return 1.0 + 2.0 * r / (r * r + spin * spin * N.cos(theta)**2)

def gththd(patch, r, xi, eta, spin):
    fcoord = (globals()["coord_" + sphere[patch] + "_to_sph"])
    theta = fcoord(xi, eta)[0]
    return r * r + spin * spin * N.cos(theta)**2

def gphphd(patch, r, xi, eta, spin):
    fcoord = (globals()["coord_" + sphere[patch] + "_to_sph"])
    theta = fcoord(xi, eta)[0]
    delta = r * r - 2.0 * r + spin * spin
    As = (r * r + spin *spin) * (r * r + spin * spin) - spin * spin * delta * N.sin(theta)**2
    return As * N.sin(theta)**2 / (r * r + spin * spin * N.cos(theta)**2)

def grphd(patch, r, xi, eta, spin):
    fcoord = (globals()["coord_" + sphere[patch] + "_to_sph"])
    theta = fcoord(xi, eta)[0]
    return - spin * N.sin(theta)**2 * (1.0 + 2.0 * r / (r * r + spin * spin * N.cos(theta)**2))

def alphas(patch, r, xi, eta, spin):
    fcoord = (globals()["coord_" + sphere[patch] + "_to_sph"])
    theta = fcoord(xi, eta)[0]
    z = 2.0 * r / (r * r + spin * spin * N.cos(theta)**2)
    return 1.0 / N.sqrt(1.0 + z)

def betaru(patch, r, xi, eta, spin):
    fcoord = (globals()["coord_" + sphere[patch] + "_to_sph"])
    theta = fcoord(xi, eta)[0]
    z = 2.0 * r / (r * r + spin * spin * N.cos(theta)**2)
    return z / (1.0 + z)

########
# Define metric in patch coordinates
########

def g11d(patch, r, xi, eta, spin):
    fjac = (globals()["jacob_" + sphere[patch] + "_to_sph"])
    if (N.abs(xi) < 1e-10)and(N.abs(eta) < 1e-10)and((sphere[patch] == 'N')or(sphere[patch] == 'S')):
        return r * r + spin * spin
    else:
        return gththd(patch, r, xi, eta, spin) * (fjac(xi, eta)[0, 0])**2 + gphphd(patch, r, xi, eta, spin) * (fjac(xi, eta)[1, 0])**2 

def g22d(patch, r, xi, eta, spin):
    fjac = (globals()["jacob_" + sphere[patch] + "_to_sph"])
    if (N.abs(xi) < 1e-10)and(N.abs(eta) < 1e-10)and((sphere[patch] == 'N')or(sphere[patch] == 'S')):
        return r * r + spin * spin
    else:
        return gththd(patch, r, xi, eta, spin) * (fjac(xi, eta)[0, 1])**2 + gphphd(patch, r, xi, eta, spin) * (fjac(xi, eta)[1, 1])**2 

def g12d(patch, r, xi, eta, spin):
    fjac = (globals()["jacob_" + sphere[patch] + "_to_sph"])
    return gththd(patch, r, xi, eta, spin) * fjac(xi, eta)[0, 0] * fjac(xi, eta)[0, 1] \
         + gphphd(patch, r, xi, eta, spin) * fjac(xi, eta)[1, 0] * fjac(xi, eta)[1, 1] 

def gr1d(patch, r, xi, eta, spin):
    fjac = (globals()["jacob_" + sphere[patch] + "_to_sph"])
    return grphd(patch, r, xi, eta, spin) * fjac(xi, eta)[1, 0]

def gr2d(patch, r, xi, eta, spin):
    fjac = (globals()["jacob_" + sphere[patch] + "_to_sph"])
    return grphd(patch, r, xi, eta, spin) * fjac(xi, eta)[1, 1]

########
# Define metric in cartesian coordinates
########

# Cartesian KS coordinates according to Carrasco et al. (2018)
def KSCart(patch, r, xi, eta, spin):
    mass = 1.0
    fcoord = (globals()["coord_" + sphere[patch] + "_to_Cart"])
    x, y, z = fcoord(r, xi, eta)
    rKS = N.sqrt(0.5 * (r**2 - spin**2) + N.sqrt(0.25 * (r**2 - spin**2)**2 + spin**2 * z**2))
    HKS = 2 * mass / (rKS**2 + spin**2 * z**2 / rKS**2)
    lKSt = 1.0
    lKSx = (rKS * x + spin * y) / (rKS**2 + spin**2)
    lKSy = (rKS * y - spin * x) / (rKS**2 + spin**2)
    lKSz = z / rKS
    etaMink = [[-1, 0, 0, 0],[0, 1, 0, 0],[0, 0, 1, 0],[0, 0, 1, 0]]
    
    gKStt = etaMink[0, 0] + HKS * lKSt * lKSt
    gKStx = etaMink[0, 1] + HKS * lKSt * lKSx
    gKSty = etaMink[0, 2] + HKS * lKSt * lKSy
    gKStz = etaMink[0, 3] + HKS * lKSt * lKSz
    
    gKSxt = etaMink[1, 0] + HKS * lKSx * lKSt
    gKSxx = etaMink[1, 1] + HKS * lKSx * lKSx
    gKSxy = etaMink[1, 2] + HKS * lKSx * lKSy
    gKSxz = etaMink[1, 3] + HKS * lKSx * lKSz
    
    gKSyt = etaMink[2, 0] + HKS * lKSy * lKSt
    gKSyx = etaMink[2, 1] + HKS * lKSy * lKSx
    gKSyy = etaMink[2, 2] + HKS * lKSy * lKSy
    gKSyz = etaMink[2, 3] + HKS * lKSy * lKSz
    
    gKSzt = etaMink[3, 0] + HKS * lKSz * lKSt
    gKSzx = etaMink[3, 1] + HKS * lKSz * lKSx
    gKSzy = etaMink[3, 2] + HKS * lKSz * lKSy
    gKSzz = etaMink[3, 3] + HKS * lKSz * lKSz
    
    return [[gKStt, gKStx, gKSty, gKStz], [gKSxt, gKSxx, gKSxy, gKSxz], [gKSyt, gKSyx, gKSyy, gKSyz], [gKSzt, gKSzx, gKSzy, gKSzz]]

########
# Define metric in patch coordinates from Cartesian KS, according to (jacob_patch_to_Cart)^T.gKS.jacob_patch_to_Cart
########

def grrd(patch, r, xi, eta, spin):
    fjac = (globals()["jacob_" + sphere[patch] + "_to_Cart"])
    fjacmat = fjac(r, xi, eta)
    gKSmat = KSCart(r, xi, eta)
    
    fjacxx = fjacmat[0,0]
    fjacxy = fjacmat[0,1]
    fjacxz = fjacmat[0,2]
        
    fjacyx = fjacmat[1,0]
    fjacyy = fjacmat[1,1]
    fjacyz = fjacmat[1,2]
        
    fjaczx = fjacmat[2,0]
    fjaczy = fjacmat[2,1]
    fjaczz = fjacmat[2,2]
    
    gKSxx = gKSmat[1,1]
    gKSxy = gKSmat[1,2]
    gKSxz = gKSmat[1,3]

    gKSyx = gKSmat[2,1]
    gKSyy = gKSmat[2,2]
    gKSyz = gKSmat[2,3]
    
    gKSzx = gKSmat[3,1]
    gKSzy = gKSmat[3,2]
    gKSzz = gKSmat[3,3]
    
    return fjacxx**2*gKSxx + fjacxx*fjacyx*(gKSxy + gKSyx) + fjacyx**2*gKSyy + fjacxx*fjaczx*(gKSxz + gKSzx) + fjacyx*fjaczx*(gKSyz + gKSzy) + fjaczx**2*gKSzz
    
def gr1d(patch, r, xi, eta, spin):
    fjac = (globals()["jacob_" + sphere[patch] + "_to_Cart"])
    fjacmat = fjac(r, xi, eta)
    gKSmat = KSCart(r, xi, eta)
    
    fjacxx = fjacmat[0,0]
    fjacxy = fjacmat[0,1]
    fjacxz = fjacmat[0,2]
        
    fjacyx = fjacmat[1,0]
    fjacyy = fjacmat[1,1]
    fjacyz = fjacmat[1,2]
        
    fjaczx = fjacmat[2,0]
    fjaczy = fjacmat[2,1]
    fjaczz = fjacmat[2,2]
    
    gKSxx = gKSmat[1,1]
    gKSxy = gKSmat[1,2]
    gKSxz = gKSmat[1,3]

    gKSyx = gKSmat[2,1]
    gKSyy = gKSmat[2,2]
    gKSyz = gKSmat[2,3]
    
    gKSzx = gKSmat[3,1]
    gKSzy = gKSmat[3,2]
    gKSzz = gKSmat[3,3]
    
    return fjacxx*(fjacxy*gKSxx + fjacyy*gKSxy + fjaczy*gKSxz) + fjacxy*fjacyx*gKSyx + fjacyx*fjacyy*gKSyy + fjacyx*fjaczy*gKSyz + fjacxy*fjaczx*gKSzx + fjacyy*fjaczx*gKSzy + fjaczx*fjaczy*gKSzz
    

def gr2d(patch, r, xi, eta, spin):
    fjac = (globals()["jacob_" + sphere[patch] + "_to_Cart"])
    fjacmat = fjac(r, xi, eta)
    gKSmat = KSCart(r, xi, eta)
    
    fjacxx = fjacmat[0,0]
    fjacxy = fjacmat[0,1]
    fjacxz = fjacmat[0,2]
        
    fjacyx = fjacmat[1,0]
    fjacyy = fjacmat[1,1]
    fjacyz = fjacmat[1,2]
        
    fjaczx = fjacmat[2,0]
    fjaczy = fjacmat[2,1]
    fjaczz = fjacmat[2,2]
    
    gKSxx = gKSmat[1,1]
    gKSxy = gKSmat[1,2]
    gKSxz = gKSmat[1,3]

    gKSyx = gKSmat[2,1]
    gKSyy = gKSmat[2,2]
    gKSyz = gKSmat[2,3]
    
    gKSzx = gKSmat[3,1]
    gKSzy = gKSmat[3,2]
    gKSzz = gKSmat[3,3]
    
    return fjacxx*(fjacxz*gKSxx + fjacyz*gKSxy + fjaczz*gKSxz) + fjacxz*fjacyx*gKSyx + fjacyx*fjacyz*gKSyy + fjacyx*fjaczz*gKSyz + fjacxz*fjaczx*gKSzx + fjacyz*fjaczx*gKSzy + fjaczx*fjaczz*gKSzz
    

def g1rd(patch, r, xi, eta, spin):
    fjac = (globals()["jacob_" + sphere[patch] + "_to_Cart"])
    fjacmat = fjac(r, xi, eta)
    gKSmat = KSCart(r, xi, eta)
    
    fjacxx = fjacmat[0,0]
    fjacxy = fjacmat[0,1]
    fjacxz = fjacmat[0,2]
        
    fjacyx = fjacmat[1,0]
    fjacyy = fjacmat[1,1]
    fjacyz = fjacmat[1,2]
        
    fjaczx = fjacmat[2,0]
    fjaczy = fjacmat[2,1]
    fjaczz = fjacmat[2,2]
    
    gKSxx = gKSmat[1,1]
    gKSxy = gKSmat[1,2]
    gKSxz = gKSmat[1,3]

    gKSyx = gKSmat[2,1]
    gKSyy = gKSmat[2,2]
    gKSyz = gKSmat[2,3]
    
    gKSzx = gKSmat[3,1]
    gKSzy = gKSmat[3,2]
    gKSzz = gKSmat[3,3]
    
    return fjacxy*fjacyx*gKSxy + fjacxy*fjaczx*gKSxz + fjacyx*fjacyy*gKSyy + fjacyy*fjaczx*gKSyz + fjacxx*(fjacxy*gKSxx + fjacyy*gKSyx + fjaczy*gKSzx) + fjacyx*fjaczy*gKSzy + fjaczx*fjaczy*gKSzz
     
def g11d(patch, r, xi, eta, spin):
    fjac = (globals()["jacob_" + sphere[patch] + "_to_Cart"])
    fjacmat = fjac(r, xi, eta)
    gKSmat = KSCart(r, xi, eta)
    
    fjacxx = fjacmat[0,0]
    fjacxy = fjacmat[0,1]
    fjacxz = fjacmat[0,2]
        
    fjacyx = fjacmat[1,0]
    fjacyy = fjacmat[1,1]
    fjacyz = fjacmat[1,2]
        
    fjaczx = fjacmat[2,0]
    fjaczy = fjacmat[2,1]
    fjaczz = fjacmat[2,2]
    
    gKSxx = gKSmat[1,1]
    gKSxy = gKSmat[1,2]
    gKSxz = gKSmat[1,3]

    gKSyx = gKSmat[2,1]
    gKSyy = gKSmat[2,2]
    gKSyz = gKSmat[2,3]
    
    gKSzx = gKSmat[3,1]
    gKSzy = gKSmat[3,2]
    gKSzz = gKSmat[3,3]
    
    return fjacxy**2*gKSxx + fjacxy*fjacyy*(gKSxy + gKSyx) + fjacyy**2*gKSyy + fjacxy*fjaczy*(gKSxz + gKSzx) + fjacyy*fjaczy*(gKSyz + gKSzy) + fjaczy**2*gKSzz

def g12d(patch, r, xi, eta, spin):
    fjac = (globals()["jacob_" + sphere[patch] + "_to_Cart"])
    fjacmat = fjac(r, xi, eta)
    gKSmat = KSCart(r, xi, eta)
    
    fjacxx = fjacmat[0,0]
    fjacxy = fjacmat[0,1]
    fjacxz = fjacmat[0,2]
        
    fjacyx = fjacmat[1,0]
    fjacyy = fjacmat[1,1]
    fjacyz = fjacmat[1,2]
        
    fjaczx = fjacmat[2,0]
    fjaczy = fjacmat[2,1]
    fjaczz = fjacmat[2,2]
    
    gKSxx = gKSmat[1,1]
    gKSxy = gKSmat[1,2]
    gKSxz = gKSmat[1,3]

    gKSyx = gKSmat[2,1]
    gKSyy = gKSmat[2,2]
    gKSyz = gKSmat[2,3]
    
    gKSzx = gKSmat[3,1]
    gKSzy = gKSmat[3,2]
    gKSzz = gKSmat[3,3]
    
    return fjacxy*(fjacxz*gKSxx + fjacyz*gKSxy + fjaczz*gKSxz) + fjacxz*fjacyy*gKSyx + fjacyy*fjacyz*gKSyy + fjacyy*fjaczz*gKSyz + fjacxz*fjaczy*gKSzx + fjacyz*fjaczy*gKSzy + fjaczy*fjaczz*gKSzz

def g2rd(patch, r, xi, eta, spin):
    fjac = (globals()["jacob_" + sphere[patch] + "_to_Cart"])
    fjacmat = fjac(r, xi, eta)
    gKSmat = KSCart(r, xi, eta)
    
    fjacxx = fjacmat[0,0]
    fjacxy = fjacmat[0,1]
    fjacxz = fjacmat[0,2]
        
    fjacyx = fjacmat[1,0]
    fjacyy = fjacmat[1,1]
    fjacyz = fjacmat[1,2]
        
    fjaczx = fjacmat[2,0]
    fjaczy = fjacmat[2,1]
    fjaczz = fjacmat[2,2]
    
    gKSxx = gKSmat[1,1]
    gKSxy = gKSmat[1,2]
    gKSxz = gKSmat[1,3]

    gKSyx = gKSmat[2,1]
    gKSyy = gKSmat[2,2]
    gKSyz = gKSmat[2,3]
    
    gKSzx = gKSmat[3,1]
    gKSzy = gKSmat[3,2]
    gKSzz = gKSmat[3,3]
    
    return fjacxz*fjacyx*gKSxy + fjacxz*fjaczx*gKSxz + fjacyx*fjacyz*gKSyy + fjacyz*fjaczx*gKSyz + fjacxx*(fjacxz*gKSxx + fjacyz*gKSyx + fjaczz*gKSzx) + fjacyx*fjaczz*gKSzy + fjaczx*fjaczz*gKSzz

def g21d(patch, r, xi, eta, spin):
    fjac = (globals()["jacob_" + sphere[patch] + "_to_Cart"])
    fjacmat = fjac(r, xi, eta)
    gKSmat = KSCart(r, xi, eta)
    
    fjacxx = fjacmat[0,0]
    fjacxy = fjacmat[0,1]
    fjacxz = fjacmat[0,2]
        
    fjacyx = fjacmat[1,0]
    fjacyy = fjacmat[1,1]
    fjacyz = fjacmat[1,2]
        
    fjaczx = fjacmat[2,0]
    fjaczy = fjacmat[2,1]
    fjaczz = fjacmat[2,2]
    
    gKSxx = gKSmat[1,1]
    gKSxy = gKSmat[1,2]
    gKSxz = gKSmat[1,3]

    gKSyx = gKSmat[2,1]
    gKSyy = gKSmat[2,2]
    gKSyz = gKSmat[2,3]
    
    gKSzx = gKSmat[3,1]
    gKSzy = gKSmat[3,2]
    gKSzz = gKSmat[3,3]
    
    return fjacxz*fjacyy*gKSxy + fjacxz*fjaczy*gKSxz + fjacyy*fjacyz*gKSyy + fjacyz*fjaczy*gKSyz + fjacxy*(fjacxz*gKSxx + fjacyz*gKSyx + fjaczz*gKSzx) + fjacyy*fjaczz*gKSzy + fjaczy*fjaczz*gKSzz
    
def g22d(patch, r, xi, eta, spin):
    fjac = (globals()["jacob_" + sphere[patch] + "_to_Cart"])
    fjacmat = fjac(r, xi, eta)
    gKSmat = KSCart(r, xi, eta)
    
    fjacxx = fjacmat[0,0]
    fjacxy = fjacmat[0,1]
    fjacxz = fjacmat[0,2]
        
    fjacyx = fjacmat[1,0]
    fjacyy = fjacmat[1,1]
    fjacyz = fjacmat[1,2]
        
    fjaczx = fjacmat[2,0]
    fjaczy = fjacmat[2,1]
    fjaczz = fjacmat[2,2]
    
    gKSxx = gKSmat[1,1]
    gKSxy = gKSmat[1,2]
    gKSxz = gKSmat[1,3]

    gKSyx = gKSmat[2,1]
    gKSyy = gKSmat[2,2]
    gKSyz = gKSmat[2,3]
    
    gKSzx = gKSmat[3,1]
    gKSzy = gKSmat[3,2]
    gKSzz = gKSmat[3,3]
    
    return fjacxz**2*gKSxx + fjacxz*fjacyz*(gKSxy + gKSyx) + fjacyz**2*gKSyy + fjacxz*fjaczz*(gKSxz + gKSzx) + fjacyz*fjaczz*(gKSyz + gKSzy) + fjaczz**2*gKSzz
    