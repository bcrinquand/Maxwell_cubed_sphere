import numpy as N

########
# Spherical <-> Cartesian
########

def coord_cart_to_sph(x, y, z):
    r = N.sqrt(x * x + y * y + z * z)
    theta = N.arccos(z / r)
    phi = N.arctan2(y, x)
    return r, theta, phi

def coord_sph_to_cart(r, theta, phi):
    x = r * N.sin(theta) * N.cos(phi)
    y = r * N.sin(theta) * N.sin(phi)
    z = r * N.cos(theta)
    return x, y, z

########
# Spherical <-> Patches
########

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
    xi_flip  = - eta
    eta_flip = xi
    X = N.tan(xi_flip)
    Y = N.tan(eta_flip)
    delta = N.sqrt(1.0 + X * X + Y * Y)
    
    if N.isscalar(delta):
        if (delta > 1.0):
            theta = N.pi / 2.0 - N.arctan(1.0 / N.sqrt(delta**2 - 1.0))
            phi = N.arctan2(X / N.sqrt(delta**2 - 1.0), - Y / N.sqrt(delta**2 - 1.0))
        elif (delta == 1.0):
            theta = 0.0
            phi = 0.0
    else:
        theta = N.pi / 2.0 - N.arctan(1.0 / N.sqrt(delta**2 - 1.0))
        phi = N.arctan2(X / N.sqrt(delta**2 - 1.0), - Y / N.sqrt(delta**2 - 1.0))
        theta[delta == 1.0] = 0.0
        phi[delta == 1.0] = 0.0
        
    return theta, phi

def coord_sph_to_N(theta, phi):
    xi_flip  = N.arctan(N.tan(theta) * N.sin(phi))
    eta_flip = N.arctan(- N.tan(theta) * N.cos(phi))
    xi = eta_flip
    eta = - xi_flip
    return xi, eta

def coord_S_to_sph(xi, eta):
    X = N.tan(xi)
    Y = N.tan(eta)
    delta = N.sqrt(1.0 + X * X + Y * Y)
    
    if N.isscalar(delta):
        if (delta > 1.0):
            theta = N.pi / 2.0 - N.arctan(- 1.0 /  N.sqrt(delta**2 - 1.0))
            phi = N.arctan2(X / N.sqrt(delta**2 - 1.0), Y / N.sqrt(delta**2 - 1.0))
        elif (delta == 1.0):
            theta = N.pi
            phi = 0.0
    else:
        theta = N.pi / 2.0 - N.arctan(- 1.0 / N.sqrt(delta**2 - 1.0))
        phi = N.arctan2(X / N.sqrt(delta**2 - 1.0), Y / N.sqrt(delta**2 - 1.0))
        theta[delta == 1.0] = N.pi
        phi[delta == 1.0] = 0.0
        
    return theta, phi

def coord_sph_to_S(theta, phi):
    xi  = N.arctan(- N.tan(theta) * N.sin(phi))
    eta = N.arctan(- N.tan(theta) * N.cos(phi))
    return xi, eta

def unflip_eq(xi, eta):
    return eta, - xi
def unflip_po(xi, eta):
    return - eta, xi


########
# Cartesian <-> Patches
########

def coord_A_to_Cart(r, xi, eta):
    X = N.tan(xi)
    Y = N.tan(eta)
    delta = N.sqrt(1 + X**2 + Y**2)
    x = r / delta
    y = r * X / delta
    z = r * Y / delta
    return x, y, z

def coord_Cart_to_A(x, y, z):
    xi = N.arctan(y / x)
    eta = N.arctan(z / x)
    r = N.sqrt(x**2 + y**2 + z**2)
    return r, xi, eta

def coord_B_to_Cart(r, xi, eta):
    X = N.tan(xi)
    Y = N.tan(eta)
    delta = N.sqrt(1 + X**2 + Y**2)
    x = - r * X / delta
    y = r / delta
    z = r * Y / delta
    return x, y, z

def coord_Cart_to_B(x, y, z):
    xi = N.arctan(- x / y)
    eta = N.arctan(z / y)
    r = N.sqrt(x**2 + y**2 + z**2)
    return r, xi, eta

def coord_C_to_Cart(r, xi, eta):
    X = N.tan(xi)
    Y = N.tan(eta)
    delta = N.sqrt(1 + X**2 + Y**2)
    x = - r / delta
    y = - r * Y / delta
    z = r * X / delta
    return x, y, z

def coord_Cart_to_C(x, y, z):
    xi = N.arctan(z / x)
    eta = N.arctan(y / x)
    r = N.sqrt(x**2 + y**2 + z**2)
    return r, xi, eta

def coord_D_to_Cart(r, xi, eta):
    X = N.tan(xi)
    Y = N.tan(eta)
    delta = N.sqrt(1 + X**2 + Y**2)
    x = r * Y / delta
    y = - r / delta
    z = - r * X / delta
    return x, y, z

def coord_Cart_to_D(x, y, z):
    xi = N.arctan(z / y)
    eta = N.arctan(- x / y)
    r = N.sqrt(x**2 + y**2 + z**2)
    return r, xi, eta

def coord_N_to_Cart(r, xi, eta):
    X = N.tan(xi)
    Y = N.tan(eta)
    delta = N.sqrt(1 + X**2 + Y**2)
    x = - r * X / delta
    y = - r * Y / delta
    z = r / delta
    return x, y, z

def coord_Cart_to_N(x, y, z):
    xi = N.arctan(- x / z)
    eta = N.arctan(- y / z)
    r = N.sqrt(x**2 + y**2 + z**2)
    return r, xi, eta

def coord_S_to_Cart(r, xi, eta):
    X = N.tan(xi)
    Y = N.tan(eta)
    delta = N.sqrt(1 + X**2 + Y**2)
    x = r * Y / delta
    y = r * X / delta
    z = - r / delta
    return x, y, z

def coord_Cart_to_S(x, y, z):
    xi = N.arctan(- y / z)
    eta = N.arctan(- x / z)
    r = N.sqrt(x**2 + y**2 + z**2)
    return r, xi, eta
