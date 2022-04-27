import numpy as N

from coord_transformations_flip import coord_cart_to_sph, coord_sph_to_cart

########
# Spherical <-> Cartesian
########

def vec_cart_to_sph(x, y, z, vx, vy, vz):
    r = N.sqrt(x * x + y * y + z * z)
    rho = N.sqrt(x * x + y * y)
    vr = (x * vx + y * vy + z * vz) / r
    if (rho > 0.0):
        vth = ((x * z / rho) * vx + (y * z / rho) * vy - rho * vz) / r**2
        vph = - (y / rho**2) * vx + (x / rho**2) * vy
    else:
        vth = 0.0
        vph = 0.0
    return vr, vth, vph

def vec_sph_to_cart(r, theta, phi, vr, vth, vph):
    vx = N.sin(theta) * N.cos(phi) * vr + r * N.cos(theta) * N.cos(phi) * vth - r * N.sin(theta) * N.sin(phi) * vph
    vy = N.sin(theta) * N.sin(phi) * vr + r * N.cos(theta) * N.sin(phi) * vth + r * N.sin(theta) * N.cos(phi) * vph
    vz = N.cos(theta) * vr - r * N.sin(theta) * vth
    return vx, vy, vz

########
# Spherical <-> Patches
########

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
    
    if N.isscalar(delta):
        if (delta > 1.0):
            vth = (X * C**2 / (delta**2 * N.sqrt(delta**2 - 1.0))) * vxi_flip + \
                (Y * D**2 / (delta**2 * N.sqrt(delta**2 - 1.0))) * veta_flip
            vph = - (Y * C**2 / (delta**2 - 1.0)) * vxi_flip \
                + (X * D**2 / (delta**2 - 1.0)) * veta_flip
        elif (delta == 1.0):
            vth = 0.0
            vph = 0.0
    else:
        vth = (X * C**2 / (delta**2 * N.sqrt(delta**2 - 1.0))) * vxi_flip + \
              (Y * D**2 / (delta**2 * N.sqrt(delta**2 - 1.0))) * veta_flip
        vph = - (Y * C**2 / (delta**2 - 1.0)) * vxi_flip \
              + (X * D**2 / (delta**2 - 1.0)) * veta_flip
        ma = N.broadcast_to(delta == 1.0, vxi.shape)
        vth[ma] = 0.0
        vph[ma] = 0.0
        
    return vth, vph

def vec_sph_to_N(theta, phi, vth, vph):
    X = N.tan(theta) * N.sin(phi)
    Y = - N.tan(theta) * N.cos(phi)
    C = N.sqrt(1.0 + X * X)
    D = N.sqrt(1.0 + Y * Y)
    delta = N.sqrt(1.0 + X * X + Y * Y)
    
    if N.isscalar(delta):
        if (delta > 1.0):
            vxi_flip = (X * delta**2 / (C**2 * N.sqrt(delta**2 - 1.0))) * vth \
                    - (Y / C**2) * vph
            veta_flip = (Y * delta**2 / (D**2 * N.sqrt(delta**2 - 1.0))) * vth \
                    + (X / D**2) * vph
        elif (delta == 1.0):
            vxi_flip = 0.0
            veta_flip = 0.0
    else:
        vxi_flip = (X * delta**2 / (C**2 * N.sqrt(delta**2 - 1.0))) * vth \
                - (Y / C**2) * vph
        veta_flip = (Y * delta**2 / (D**2 * N.sqrt(delta**2 - 1.0))) * vth \
                + (X / D**2) * vph
        ma = N.broadcast_to(delta == 1.0, vth.shape)
        vxi_flip[ma] = 0.0
        veta_flip[ma] = 0.0
    
    vxi = veta_flip
    veta = - vxi_flip
    return vxi, veta

def vec_S_to_sph(xi, eta, vxi, veta):
    X = N.tan(xi)
    Y = N.tan(eta)
    C = N.sqrt(1.0 + X * X)
    D = N.sqrt(1.0 + Y * Y)
    delta = N.sqrt(1.0 + X * X + Y * Y)
    
    if N.isscalar(delta):
        if (delta > 1.0):
            vth = - (X * C**2 / (delta**2 * N.sqrt(delta**2 - 1.0))) * vxi \
                  - (Y * D**2 / (delta**2 * N.sqrt(delta**2 - 1.0))) * veta
            vph = (Y * C**2 / (delta**2 - 1.0)) * vxi \
                - (X * D**2 / (delta**2 - 1.0)) * veta
        elif (delta == 1.0):
            vth = 0.0
            vph = 0.0
    else:
        vth = - (X * C**2 / (delta**2 * N.sqrt(delta**2 - 1.0))) * vxi \
              - (Y * D**2 / (delta**2 * N.sqrt(delta**2 - 1.0))) * veta
        vph = (Y * C**2 / (delta**2 - 1.0)) * vxi \
            - (X * D**2 / (delta**2 - 1.0)) * veta
        ma = N.broadcast_to(delta == 1.0, vxi.shape)
        vth[ma] = 0.0
        vph[ma] = 0.0
        
    return vth, vph

def vec_sph_to_S(theta, phi, vth, vph):
    X = - N.tan(theta) * N.sin(phi)
    Y = - N.tan(theta) * N.cos(phi)
    C = N.sqrt(1.0 + X * X)
    D = N.sqrt(1.0 + Y * Y)
    delta = N.sqrt(1.0 + X * X + Y * Y)

    if N.isscalar(delta):
        if (delta > 1.0):
            vxi = - (X * delta**2 / (C**2 * N.sqrt(delta**2 - 1.0))) * vth \
                  + (Y / C**2) * vph
            veta = - (Y * delta**2 / (D**2 * N.sqrt(delta**2 - 1.0))) * vth \
                   - (X / D**2) * vph
        elif (delta == 1.0):
            vxi = 0.0
            veta = 0.0
    else:
        vxi = - (X * delta**2 / (C**2 * N.sqrt(delta**2 - 1.0))) * vth \
              + (Y / C**2) * vph
        veta = - (Y * delta**2 / (D**2 * N.sqrt(delta**2 - 1.0))) * vth \
               - (X / D**2) * vph
        ma = N.broadcast_to(delta == 1.0, vth.shape)
        vxi[ma] = 0.0
        veta[ma] = 0.0
        
    return vxi, veta

def unflip_vec_eq(vxi, veta):
    return veta, - vxi
def unflip_vec_po(vxi, veta):
    return - veta, vxi
