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
    return 0.9 * z / (1.0 + z)

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

