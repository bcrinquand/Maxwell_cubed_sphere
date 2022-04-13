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

def form_C_to_sph(xi, eta, vxi, veta):
    X = N.tan(xi)
    Y = N.tan(eta)
    C = N.sqrt(1.0 + X * X)
    D = N.sqrt(1.0 + Y * Y)
    delta = N.sqrt(1.0 + X * X + Y * Y)
    return - (delta**2 / (C * D * D)) * veta, vxi + (X * Y / D**2) * veta

def form_sph_to_C(theta, phi, vth, vph):
    X = N.tan(phi)
    Y = - 1.0 / N.tan(theta) / N.cos(phi)
    C = N.sqrt(1.0 + X * X)
    D = N.sqrt(1.0 + Y * Y)
    delta = N.sqrt(1.0 + X * X + Y * Y)
    return (C * X * Y / delta**2) * vth + vph, - (C * D * D / delta**2) * vth

def form_D_to_sph(xi, eta, vxi, veta):
    X = N.tan(xi)
    Y = N.tan(eta)
    C = N.sqrt(1.0 + X * X)
    D = N.sqrt(1.0 + Y * Y)
    delta = N.sqrt(1.0 + X * X + Y * Y)
    return - (delta**2 / (C * D * D)) * veta, vxi + (X * Y / D**2) * veta

def form_sph_to_D(theta, phi, vth, vph):
    X = - 1.0 / N.tan(phi)
    Y = - 1.0 / N.tan(theta) / N.sin(phi)
    C = N.sqrt(1.0 + X * X)
    D = N.sqrt(1.0 + Y * Y)
    delta = N.sqrt(1.0 + X * X + Y * Y)
    return (C * X * Y / delta**2) * vth + vph, - (C * D * D / delta**2) * vth

def form_N_to_sph(xi, eta, vxi, veta):
    X = N.tan(xi)
    Y = N.tan(eta)
    C = N.sqrt(1.0 + X * X)
    D = N.sqrt(1.0 + Y * Y)
    delta = N.sqrt(1.0 + X * X + Y * Y)
    vth = (X * delta**2 / (C**2 * N.sqrt(delta**2 - 1.0))) * vxi + \
          (Y * delta**2 / (D**2 * N.sqrt(delta**2 - 1.0))) * veta
    vph = - (Y / (C**2)) * vxi \
          + (X / (D**2)) * veta
    return vth, vph

def form_sph_to_N(theta, phi, vth, vph):
    X = N.tan(theta) * N.sin(phi)
    Y = - N.tan(theta) * N.cos(phi)
    C = N.sqrt(1.0 + X * X)
    D = N.sqrt(1.0 + Y * Y)
    delta = N.sqrt(1.0 + X * X + Y * Y)
    vxi = (X * C**2 / (delta**2 * N.sqrt(delta**2 - 1.0))) * vth - \
          (Y * C**2 / (delta**2 - 1.0)) * vph
    veta = (Y * D**2 / (delta**2 * N.sqrt(delta**2 - 1.0))) * vth \
         + (X * D**2 / (delta**2 - 1.0)) * vph
    return vxi, veta

def form_S_to_sph(xi, eta, vxi, veta):
    X = N.tan(xi)
    Y = N.tan(eta)
    C = N.sqrt(1.0 + X * X)
    D = N.sqrt(1.0 + Y * Y)
    delta = N.sqrt(1.0 + X * X + Y * Y)
    vth = - (X * delta**2 / (C**2 * N.sqrt(delta**2 - 1.0))) * vxi \
          - (Y * delta**2 / (D**2 * N.sqrt(delta**2 - 1.0))) * veta
    vph = (Y / (C**2)) * vxi - \
          (X / (D**2)) * veta
    return vth, vph

def form_sph_to_S(theta, phi, vth, vph):
    X = - N.tan(theta) * N.sin(phi)
    Y = - N.tan(theta) * N.cos(phi)
    C = N.sqrt(1.0 + X * X)
    D = N.sqrt(1.0 + Y * Y)
    delta = N.sqrt(1.0 + X * X + Y * Y)
    vxi = - (X * C**2 / (delta**2 * N.sqrt(delta**2 - 1.0))) * vth \
          + (Y * C**2 / (delta**2 - 1.0)) * vph
    veta = - (Y * D**2 / (delta**2 * N.sqrt(delta**2 - 1.0))) * vth - \
           - (X * D**2 / (delta**2 - 1.0)) * vph
    return vxi, veta
