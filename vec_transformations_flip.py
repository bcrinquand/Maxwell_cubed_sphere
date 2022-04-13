import numpy as N

def vec_A_to_sph(xi, eta, vxi, veta):
    X = N.tan(xi)
    Y = N.tan(eta)
    C = N.sqrt(1.0 + X * X)
    D = N.sqrt(1.0 + Y * Y)
    delta = N.sqrt(1.0 + X * X + Y * Y)
    return (C * X * Y / delta**2) * vxi - (C * D * D / delta**2) * veta, vxi

def vec_sph_to_A(theta, phi, vth, vph):
    X = N.tan(phi)
    Y = 1.0 / N.tan(theta) / N.cos(phi)
    C = N.sqrt(1.0 + X * X)
    D = N.sqrt(1.0 + Y * Y)
    delta = N.sqrt(1.0 + X * X + Y * Y)
    return vph, - (delta**2 / (C * D * D)) * vth + (X * Y / D**2) * vph

def vec_B_to_sph(xi, eta, vxi, veta):
    X = N.tan(xi)
    Y = N.tan(eta)
    C = N.sqrt(1.0 + X * X)
    D = N.sqrt(1.0 + Y * Y)
    delta = N.sqrt(1.0 + X * X + Y * Y)
    return (C * X * Y / delta**2) * vxi - (C * D * D / delta**2) * veta, vxi

def vec_sph_to_B(theta, phi, vth, vph):
    X = - 1.0 / N.tan(phi)
    Y = 1.0 / N.tan(theta) / N.sin(phi)
    C = N.sqrt(1.0 + X * X)
    D = N.sqrt(1.0 + Y * Y)
    delta = N.sqrt(1.0 + X * X + Y * Y)
    return vph, - (delta**2 / (C * D * D)) * vth + (X * Y / D**2) * vph

# Patch C is flipped as xi_C = eta, eta_C = -xi

def vec_C_to_sph(xi, eta, vxi, veta):
    xi_flip  = eta
    eta_flip = - xi
    vxi_flip = veta
    veta_flip = - vxi
    X = N.tan(xi_flip)
    Y = N.tan(eta_flip)
    C = N.sqrt(1.0 + X * X)
    D = N.sqrt(1.0 + Y * Y)
    delta = N.sqrt(1.0 + X * X + Y * Y)
    return (C * X * Y / delta**2) * vxi_flip - (C * D * D / delta**2) * veta_flip, vxi_flip

def vec_sph_to_C(theta, phi, vth, vph):
    X = N.tan(phi)
    Y = - 1.0 / N.tan(theta) / N.cos(phi)
    C = N.sqrt(1.0 + X * X)
    D = N.sqrt(1.0 + Y * Y)
    delta = N.sqrt(1.0 + X * X + Y * Y)
    vxi_flip = vph
    veta_flip = - (delta**2 / (C * D * D)) * vth + (X * Y / D**2) * vph
    vxi = - veta_flip
    veta = vxi_flip
    return vxi, veta

# Patch D is flipped as xi_D = eta, eta_D = -xi

def vec_D_to_sph(xi, eta, vxi, veta):
    xi_flip  = eta
    eta_flip = - xi
    vxi_flip = veta
    veta_flip = - vxi
    X = N.tan(xi_flip)
    Y = N.tan(eta_flip)
    C = N.sqrt(1.0 + X * X)
    D = N.sqrt(1.0 + Y * Y)
    delta = N.sqrt(1.0 + X * X + Y * Y)
    return (C * X * Y / delta**2) * vxi_flip - (C * D * D / delta**2) * veta_flip, vxi_flip

def vec_sph_to_D(theta, phi, vth, vph):
    X = - 1.0 / N.tan(phi)
    Y = - 1.0 / N.tan(theta) / N.sin(phi)
    C = N.sqrt(1.0 + X * X)
    D = N.sqrt(1.0 + Y * Y)
    delta = N.sqrt(1.0 + X * X + Y * Y)
    vxi_flip = vph
    veta_flip = - (delta**2 / (C * D * D)) * vth + (X * Y / D**2) * vph
    vxi = - veta_flip
    veta = vxi_flip
    return vxi, veta

# Patch N is flipped as xi_N = - eta, eta_N = -xi

def vec_N_to_sph(xi, eta, vxi, veta):
    xi_flip  = - eta
    eta_flip = xi
    vxi_flip = - veta
    veta_flip = vxi
    X = N.tan(xi_flip)
    Y = N.tan(eta_flip)
    C = N.sqrt(1.0 + X * X)
    D = N.sqrt(1.0 + Y * Y)
    delta = N.sqrt(1.0 + X * X + Y * Y)
    vth = (X * C**2 / (delta**2 * N.sqrt(delta**2 - 1.0))) * vxi_flip + \
          (Y * D**2 / (delta**2 * N.sqrt(delta**2 - 1.0))) * veta_flip
    vph = - (Y * C**2 / (delta**2 - 1.0)) * vxi_flip \
          + (X * D**2 / (delta**2 - 1.0)) * veta_flip
    return vth, vph

def vec_sph_to_N(theta, phi, vth, vph):
    X = N.tan(theta) * N.sin(phi)
    Y = - N.tan(theta) * N.cos(phi)
    C = N.sqrt(1.0 + X * X)
    D = N.sqrt(1.0 + Y * Y)
    delta = N.sqrt(1.0 + X * X + Y * Y)
    vxi_flip = (X * delta**2 / (C**2 * N.sqrt(delta**2 - 1.0))) * vth - \
          (Y / (C**2)) * vph
    veta_flip = (Y * delta**2 / (D**2 * N.sqrt(delta**2 - 1.0))) * vth - \
           (X / (D**2)) * vph
    vxi = veta_flip
    veta = - vxi_flip
    return vxi, veta

def vec_S_to_sph(xi, eta, vxi, veta):
    X = N.tan(xi)
    Y = N.tan(eta)
    C = N.sqrt(1.0 + X * X)
    D = N.sqrt(1.0 + Y * Y)
    delta = N.sqrt(1.0 + X * X + Y * Y)
    vth = - (X * C**2 / (delta**2 * N.sqrt(delta**2 - 1.0))) * vxi \
          - (Y * D**2 / (delta**2 * N.sqrt(delta**2 - 1.0))) * veta
    vph = (Y * C**2 / (delta**2 - 1.0)) * vxi - \
          (X * D**2 / (delta**2 - 1.0)) * veta
    return vth, vph

def vec_sph_to_S(theta, phi, vth, vph):
    X = - N.tan(theta) * N.sin(phi)
    Y = - N.tan(theta) * N.cos(phi)
    C = N.sqrt(1.0 + X * X)
    D = N.sqrt(1.0 + Y * Y)
    delta = N.sqrt(1.0 + X * X + Y * Y)
    vxi = - (X * delta**2 / (C**2 * N.sqrt(delta**2 - 1.0))) * vth \
          - (Y / (C**2)) * vph
    veta = - (Y * delta**2 / (D**2 * N.sqrt(delta**2 - 1.0))) * vth \
           + (X / (D**2)) * vph
    return vxi, veta
