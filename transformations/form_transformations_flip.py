import numpy as N

from coord_transformations_flip import coord_cart_to_sph, coord_sph_to_cart

########
# Spherical <-> Cartesian
########

def form_cart_to_sph(x, y, z, vx, vy, vz):
    r, theta, phi = coord_sph_to_cart(x, y, z)
    vr = N.sin(theta) * N.cos(phi) * vx + N.sin(theta) * N.sin(phi) * vy + N.cos(theta) * vz
    vth = r * (N.cos(theta) * N.cos(phi) * vx + N.cos(theta) * N.sin(phi) * vy  - N.sin(theta) * vz)
    vph = r * N.sin(theta) * (- N.sin(phi) * vx + N.cos(phi) * vy)
    return vr, vth, vph

def form_sph_to_cart(r, theta, phi, vr, vth, vph):
    x, y, z = coord_sph_to_cart(r, theta, phi)
    rho = N.sqrt(x * x + y * y)
    if (rho > 0.0):
        vx = (x / r) * vr + (x * z / (rho * r**2)) * vth - (y / rho**2) * vph
        vy = (y / r) * vr + (y * z / (rho * r**2)) * vth + (x / rho**2) * vph
    else:
        vx = 0.0
        vy = 0.0
    vz = (z / r) * vr - (rho / r**2) * vth
    return vx, vy, vz

########
# Spherical <-> Patches
########

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

def form_A_to_Cart(r, xi, eta, vr, vxi, veta):
    x, y, z, = coord_A_to_Cart(r, xi, eta)
    Jac = N.transpose(jacob_cart_to_A(x, y, z))
    vx = Jac[0,0] * vr + Jac[0,1] * vxi + Jac[0,2] * veta
    vy = Jac[1,0] * vr + Jac[1,1] * vxi + Jac[1,2] * veta
    vz = Jac[2,0] * vr + Jac[2,1] * vxi + Jac[2,2] * veta
    return vx, vy, vz

def form_Cart_to_A(x, y, z, vx, vy, vz):
    r, xi, eta, = coord_Cart_to_A(x, y, z)
    Jac = N.transpose(jacob_A_to_Cart(r, xi, eta))
    vr = Jac[0,0] * vx + Jac[0,1] * vy + Jac[0,2] * vz
    vxi = Jac[1,0] * vx + Jac[1,1] * vy + Jac[1,2] * vz
    veta = Jac[2,0] * vx + Jac[2,1] * vy + Jac[2,2] * vz
    return vr, vxi, veta