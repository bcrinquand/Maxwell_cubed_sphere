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

def glld(patch, l, xi, eta, spin):
    fcoord = (globals()["coord_" + sphere[patch] + "_to_sph"])
    theta = fcoord(xi, eta)[0]
    r = N.exp(l)
    return (1.0 + 2.0 * r / (r * r + spin * spin * N.cos(theta)**2)) * r * r

def gththd(patch, l, xi, eta, spin):
    fcoord = (globals()["coord_" + sphere[patch] + "_to_sph"])
    theta = fcoord(xi, eta)[0]
    r = N.exp(l)
    return r * r + spin * spin * N.cos(theta)**2

def gphphd(patch, l, xi, eta, spin):
    fcoord = (globals()["coord_" + sphere[patch] + "_to_sph"])
    theta = fcoord(xi, eta)[0]
    r = N.exp(l)
    delta = r * r - 2.0 * r + spin * spin
    As = (r * r + spin *spin) * (r * r + spin * spin) - spin * spin * delta * N.sin(theta)**2
    return As * N.sin(theta)**2 / (r * r + spin * spin * N.cos(theta)**2)

def glphd(patch, l, xi, eta, spin):
    fcoord = (globals()["coord_" + sphere[patch] + "_to_sph"])
    theta = fcoord(xi, eta)[0]
    r = N.exp(l)
    return - r * spin * N.sin(theta)**2 * (1.0 + 2.0 * r / (r * r + spin * spin * N.cos(theta)**2))

def alphas(patch, l, xi, eta, spin):
    fcoord = (globals()["coord_" + sphere[patch] + "_to_sph"])
    theta = fcoord(xi, eta)[0]
    r = N.exp(l)
    z = 2.0 * r / (r * r + spin * spin * N.cos(theta)**2)
    return 1.0 / N.sqrt(1.0 + z)

def betalu(patch, l, xi, eta, spin):
    fcoord = (globals()["coord_" + sphere[patch] + "_to_sph"])
    theta = fcoord(xi, eta)[0]
    r = N.exp(l)
    z = 2.0 * r / (r * r + spin * spin * N.cos(theta)**2)
    return 0.8 * z / (1.0 + z) / r

def betald(patch, l, xi, eta, spin):
    fcoord = (globals()["coord_" + sphere[patch] + "_to_sph"])
    theta = fcoord(xi, eta)[0]
    r = N.exp(l)
    z = 2.0 * r / (r * r + spin * spin * N.cos(theta)**2)
    return z * r

def betaphd(patch, l, xi, eta, spin):
    fcoord = (globals()["coord_" + sphere[patch] + "_to_sph"])
    theta = fcoord(xi, eta)[0]
    r = N.exp(l)
    z = 2.0 * r / (r * r + spin * spin * N.cos(theta)**2)
    return - spin * z * N.sin(theta)**2

########
# Define metric in patch coordinates
########

def g11d(patch, l, xi, eta, spin):
    fjac = (globals()["jacob_" + sphere[patch] + "_to_sph"])
    r = N.exp(l)
    if (N.abs(xi) < 1e-10)and(N.abs(eta) < 1e-10)and((sphere[patch] == 'N')or(sphere[patch] == 'S')):
        return r * r + spin * spin
    else:
        return gththd(patch, l, xi, eta, spin) * (fjac(xi, eta)[0, 0])**2 + gphphd(patch, l, xi, eta, spin) * (fjac(xi, eta)[1, 0])**2 

def g22d(patch, l, xi, eta, spin):
    fjac = (globals()["jacob_" + sphere[patch] + "_to_sph"])
    r = N.exp(l)
    if (N.abs(xi) < 1e-10)and(N.abs(eta) < 1e-10)and((sphere[patch] == 'N')or(sphere[patch] == 'S')):
        return r * r + spin * spin
    else:
        return gththd(patch, l, xi, eta, spin) * (fjac(xi, eta)[0, 1])**2 + gphphd(patch, l, xi, eta, spin) * (fjac(xi, eta)[1, 1])**2 

def g12d(patch, l, xi, eta, spin):
    fjac = (globals()["jacob_" + sphere[patch] + "_to_sph"])
    return gththd(patch, l, xi, eta, spin) * fjac(xi, eta)[0, 0] * fjac(xi, eta)[0, 1] \
         + gphphd(patch, l, xi, eta, spin) * fjac(xi, eta)[1, 0] * fjac(xi, eta)[1, 1] 

def gl1d(patch, l, xi, eta, spin):
    fjac = (globals()["jacob_" + sphere[patch] + "_to_sph"])
    return glphd(patch, l, xi, eta, spin) * fjac(xi, eta)[1, 0]

def gl2d(patch, l, xi, eta, spin):
    fjac = (globals()["jacob_" + sphere[patch] + "_to_sph"])
    return glphd(patch, l, xi, eta, spin) * fjac(xi, eta)[1, 1]

def beta2d(patch, l, xi, eta, spin):
    fjac = (globals()["jacob_" + sphere[patch] + "_to_sph"])
    return betaphd(patch, l, xi, eta, spin) * fjac(xi, eta)[1, 1]
