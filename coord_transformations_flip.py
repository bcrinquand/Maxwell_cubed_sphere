import numpy as N

def coord_A_to_sph(xi, eta):
    X = N.tan(xi)
    Y = N.tan(eta)
    theta = 0.5 * N.pi - N.arctan(Y / N.sqrt(1.0 + X * X))
    phi = N.arctan2(X / N.sqrt(1.0 + X * X), 1.0 / N.sqrt(1.0 + X * X))
    return theta, phi

def coord_sph_to_A(theta, phi):
    xi = N.arctan(N.tan(phi))
    eta = N.arctan(1.0 / N.tan(theta) / N.cos(phi))
    return xi, eta

def coord_B_to_sph(xi, eta):
    X = N.tan(xi)
    Y = N.tan(eta)
    theta = 0.5 * N.pi - N.arctan(Y / N.sqrt(1.0 + X * X))
    phi = N.arctan2(1.0 / N.sqrt(1.0 + X * X), - X / N.sqrt(1.0 + X * X))
    return theta, phi

def coord_sph_to_B(theta, phi):
    xi  = N.arctan(- 1.0 / N.tan(phi))
    eta = N.arctan(1.0 / N.tan(theta) / N.sin(phi))
    return xi, eta

# Patch C is flipped as xi_C = eta, eta_C = -xi

def coord_C_to_sph(xi, eta):
    xi_flip  = eta
    eta_flip = - xi
    X = N.tan(xi_flip)
    Y = N.tan(eta_flip)
    theta = 0.5 * N.pi - N.arctan(Y / N.sqrt(1.0 + X * X))
    phi = N.arctan2(- X / N.sqrt(1.0 + X * X), - 1.0 / N.sqrt(1.0 + X * X))
    return theta, phi

def coord_sph_to_C(theta, phi):
    xi_flip  = N.arctan(N.tan(phi))
    eta_flip = N.arctan(- 1.0 / N.tan(theta) / N.cos(phi))
    xi = - eta_flip
    eta = xi_flip
    return xi, eta

# Patch D is flipped as xi_D = eta, eta_D = -xi

def coord_D_to_sph(xi, eta):
    xi_flip  = eta
    eta_flip = - xi
    X = N.tan(xi_flip)
    Y = N.tan(eta_flip)
    theta = 0.5 * N.pi - N.arctan(Y / N.sqrt(1.0 + X * X))
    phi = N.arctan2(- 1.0 / N.sqrt(1.0 + X * X), X / N.sqrt(1.0 + X * X))
    return theta, phi

def coord_sph_to_D(theta, phi):
    xi_flip  = N.arctan(- 1.0 / N.tan(phi))
    eta_flip = N.arctan(- 1.0 / N.tan(theta) / N.sin(phi))
    xi = - eta_flip
    eta = xi_flip
    return xi, eta

# Patch N is flipped as xi_N = - eta, eta_N = -xi

def coord_N_to_sph(xi, eta):
    global theta
    global phi
    xi_flip  = - eta
    eta_flip = xi
    X = N.tan(xi_flip)
    Y = N.tan(eta_flip)
    delta = N.sqrt(X * X + Y * Y)
    if (delta > 0):
        theta = N.pi / 2.0 - N.arctan(1.0 / delta)
        phi = N.arctan2(X / delta, - Y / delta)
    elif (delta == 0.0):
        theta = 0.0
        phi = 0.0
    return theta, phi

def coord_sph_to_N(theta, phi):
    xi_flip  = N.arctan(N.tan(theta) * N.sin(phi))
    eta_flip = N.arctan(- N.tan(theta) * N.cos(phi))
    xi = eta_flip
    eta = - xi_flip
    return xi, eta

def coord_S_to_sph(xi, eta):
    global theta
    global phi
    X = N.tan(xi)
    Y = N.tan(eta)
    delta = N.sqrt(X * X + Y * Y)
    if (delta > 0):
        theta = N.pi / 2.0 - N.arctan(- 1.0 /  delta)
        phi = N.arctan2(X / delta, Y / delta)
    elif (delta == 0.0):
        theta == N.pi
        phi = 0.0
    return theta, phi

def coord_sph_to_S(theta, phi):
    xi  = N.arctan(- N.tan(theta) * N.sin(phi))
    eta = N.arctan(- N.tan(theta) * N.cos(phi))
    return xi, eta

def unflip_eq(xi, eta):
    return eta, - xi
def unflip_po(xi, eta):
    return - eta, xi