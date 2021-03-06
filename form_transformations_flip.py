import numpy as N

def form_A_to_sph(xi, eta, vxi, veta):
    X = N.tan(xi)
    Y = N.tan(eta)
    C = N.sqrt(1.0 + X * X)
    D = N.sqrt(1.0 + Y * Y)
    delta = N.sqrt(1.0 + X * X + Y * Y)
    return - (delta**2 / (C * D * D)) * veta, vxi + (X * Y / D**2) * veta

def form_sph_to_A(theta, phi, vth, vph):
    X = N.tan(phi)
    Y = 1.0 / N.tan(theta) / N.cos(phi)
    C = N.sqrt(1.0 + X * X)
    D = N.sqrt(1.0 + Y * Y)
    delta = N.sqrt(1.0 + X * X + Y * Y)
    return (C * X * Y / delta**2) * vth + vph, - (C * D * D / delta**2) * vth

def form_B_to_sph(xi, eta, vxi, veta):
    X = N.tan(xi)
    Y = N.tan(eta)
    C = N.sqrt(1.0 + X * X)
    D = N.sqrt(1.0 + Y * Y)
    delta = N.sqrt(1.0 + X * X + Y * Y)
    return - (delta**2 / (C * D * D)) * veta, vxi + (X * Y / D**2) * veta

def form_sph_to_B(theta, phi, vth, vph):
    X = - 1.0 / N.tan(phi)
    Y = 1.0 / N.tan(theta) / N.sin(phi)
    C = N.sqrt(1.0 + X * X)
    D = N.sqrt(1.0 + Y * Y)
    delta = N.sqrt(1.0 + X * X + Y * Y)
    return (C * X * Y / delta**2) * vth + vph, - (C * D * D / delta**2) * vth

# Patch C is flipped as xi_C = eta, eta_C = -xi

def form_C_to_sph(xi, eta, vxi, veta):
    xi_flip  = eta
    eta_flip = - xi
    vxi_flip = veta
    veta_flip = - vxi
    X = N.tan(xi_flip)
    Y = N.tan(eta_flip)
    C = N.sqrt(1.0 + X * X)
    D = N.sqrt(1.0 + Y * Y)
    delta = N.sqrt(1.0 + X * X + Y * Y)
    return - (delta**2 / (C * D * D)) * veta_flip, vxi_flip + (X * Y / D**2) * veta_flip

def form_sph_to_C(theta, phi, vth, vph):
    X = N.tan(phi)
    Y = - 1.0 / N.tan(theta) / N.cos(phi)
    C = N.sqrt(1.0 + X * X)
    D = N.sqrt(1.0 + Y * Y)
    delta = N.sqrt(1.0 + X * X + Y * Y)
    vxi_flip = (C * X * Y / delta**2) * vth + vph
    veta_flip = - (C * D * D / delta**2) * vth
    vxi = - veta_flip
    veta = vxi_flip
    return vxi, veta

# Patch D is flipped as xi_D = eta, eta_D = -xi

def form_D_to_sph(xi, eta, vxi, veta):
    xi_flip  = eta
    eta_flip = - xi
    vxi_flip = veta
    veta_flip = - vxi
    X = N.tan(xi_flip)
    Y = N.tan(eta_flip)
    C = N.sqrt(1.0 + X * X)
    D = N.sqrt(1.0 + Y * Y)
    delta = N.sqrt(1.0 + X * X + Y * Y)
    return - (delta**2 / (C * D * D)) * veta_flip, vxi_flip + (X * Y / D**2) * veta_flip

def form_sph_to_D(theta, phi, vth, vph):
    X = - 1.0 / N.tan(phi)
    Y = - 1.0 / N.tan(theta) / N.sin(phi)
    C = N.sqrt(1.0 + X * X)
    D = N.sqrt(1.0 + Y * Y)
    delta = N.sqrt(1.0 + X * X + Y * Y)
    vxi_flip = (C * X * Y / delta**2) * vth + vph
    veta_flip = - (C * D * D / delta**2) * vth
    vxi = - veta_flip
    veta = vxi_flip
    return vxi, veta

# Patch N is flipped as xi_N = - eta, eta_N = -xi

def form_N_to_sph(xi, eta, vxi, veta):
    xi_flip  = - eta
    eta_flip = xi
    vxi_flip = - veta
    veta_flip = vxi
    X = N.tan(xi_flip)
    Y = N.tan(eta_flip)
    C = N.sqrt(1.0 + X * X)
    D = N.sqrt(1.0 + Y * Y)
    delta = N.sqrt(1.0 + X * X + Y * Y)
    vth = (X * delta**2 / (C**2 * N.sqrt(delta**2 - 1.0))) * vxi_flip \
        + (Y * delta**2 / (D**2 * N.sqrt(delta**2 - 1.0))) * veta_flip
    vph = - (Y / (C**2)) * vxi_flip \
          + (X / (D**2)) * veta_flip
    return vth, vph

def form_sph_to_N(theta, phi, vth, vph):
    X = N.tan(theta) * N.sin(phi)
    Y = - N.tan(theta) * N.cos(phi)
    C = N.sqrt(1.0 + X * X)
    D = N.sqrt(1.0 + Y * Y)
    delta = N.sqrt(1.0 + X * X + Y * Y)
    vxi_flip = (X * C**2 / (delta**2 * N.sqrt(delta**2 - 1.0))) * vth \
             - (Y * C**2 / (delta**2 - 1.0)) * vph
    veta_flip = (Y * D**2 / (delta**2 * N.sqrt(delta**2 - 1.0))) * vth \
              + (X * D**2 / (delta**2 - 1.0)) * vph
    vxi = veta_flip
    veta = - vxi_flip
    return vxi, veta

def form_S_to_sph(xi, eta, vxi, veta):
    X = N.tan(xi)
    Y = N.tan(eta)
    C = N.sqrt(1.0 + X * X)
    D = N.sqrt(1.0 + Y * Y)
    delta = N.sqrt(1.0 + X * X + Y * Y)
    vth = - (X * delta**2 / (C**2 * N.sqrt(delta**2 - 1.0))) * vxi \
          - (Y * delta**2 / (D**2 * N.sqrt(delta**2 - 1.0))) * veta
    vph = (Y / (C**2)) * vxi \
        - (X / (D**2)) * veta
    return vth, vph

def form_sph_to_S(theta, phi, vth, vph):
    X = - N.tan(theta) * N.sin(phi)
    Y = - N.tan(theta) * N.cos(phi)
    C = N.sqrt(1.0 + X * X)
    D = N.sqrt(1.0 + Y * Y)
    delta = N.sqrt(1.0 + X * X + Y * Y)
    vxi = - (X * C**2 / (delta**2 * N.sqrt(delta**2 - 1.0))) * vth \
          + (Y * C**2 / (delta**2 - 1.0)) * vph
    veta = - (Y * D**2 / (delta**2 * N.sqrt(delta**2 - 1.0))) * vth \
           - (X * D**2 / (delta**2 - 1.0)) * vph
    return vxi, veta

def unflip_form_eq(vxi, veta):
    return - veta, vxi
def unflip_form_po(vxi, veta):
    return veta, - vxi