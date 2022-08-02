import numpy as N

def jacob_inv_A_to_sph(xi, eta):
    X = N.tan(xi)
    Y = N.tan(eta)
    C = N.sqrt(1.0 + X * X)
    D = N.sqrt(1.0 + Y * Y)
    delta = N.sqrt(1.0 + X * X + Y * Y)
    return N.array([[0.0 * xi, - delta**2 / (C * D * D)], [1.0 + 0.0 * xi, X * Y / D**2]])

def jacob_inv_sph_to_A(theta, phi):
    X = N.tan(phi)
    Y = 1.0 / N.tan(theta) / N.cos(phi)
    C = N.sqrt(1.0 + X * X)
    D = N.sqrt(1.0 + Y * Y)
    delta = N.sqrt(1.0 + X * X + Y * Y)
    return N.array([[C * X * Y / delta**2, 1.0 + 0.0 * theta], [- C * D * D / delta**2, 0.0 * theta]])

########################

def jacob_inv_B_to_sph(xi, eta):
    X = N.tan(xi)
    Y = N.tan(eta)
    C = N.sqrt(1.0 + X * X)
    D = N.sqrt(1.0 + Y * Y)
    delta = N.sqrt(1.0 + X * X + Y * Y)
    return N.array([[0.0 * xi, - delta**2 / (C * D * D)], [1.0 + 0.0 * xi, X * Y / D**2]])

def jacob_inv_sph_to_B(theta, phi):
    X = - 1.0 / N.tan(phi)
    Y = 1.0 / N.tan(theta) / N.sin(phi)
    C = N.sqrt(1.0 + X * X)
    D = N.sqrt(1.0 + Y * Y)
    delta = N.sqrt(1.0 + X * X + Y * Y)
    return N.array([[C * X * Y / delta**2, 1.0 + 0.0 * theta], [- C * D * D / delta**2, 0.0 * theta]])

########################

def jacob_inv_C_to_sph(xi, eta):
    X = N.tan(xi)
    Y = N.tan(eta)
    C = N.sqrt(1.0 + X * X)
    D = N.sqrt(1.0 + Y * Y)
    delta = N.sqrt(1.0 + X * X + Y * Y)
    return N.array([[delta**2 / (C * C * D), 0.0 * xi], [X * Y / C**2, 1.0 + 0.0 * xi]])

def jacob_inv_sph_to_C(theta, phi):
    X = 1.0 / N.tan(theta) / N.cos(phi)
    Y = N.tan(phi)
    C = N.sqrt(1.0 + X * X)
    D = N.sqrt(1.0 + Y * Y)
    delta = N.sqrt(1.0 + X * X + Y * Y)
    return N.array([[C * C * D / delta**2, 0.0 * theta], [- D * X * Y / delta**2, 1.0 + 0.0 * theta]])

########################

def jacob_inv_D_to_sph(xi, eta):
    X = N.tan(xi)
    Y = N.tan(eta)
    C = N.sqrt(1.0 + X * X)
    D = N.sqrt(1.0 + Y * Y)
    delta = N.sqrt(1.0 + X * X + Y * Y)
    return N.array([[delta**2 / (C * C * D), 0.0 * xi], [X * Y / C**2, 1.0 + 0.0 * xi]])

def jacob_inv_sph_to_D(theta, phi):
    X = 1.0 / N.tan(theta) / N.sin(phi)
    Y = - 1.0 / N.tan(phi) 
    C = N.sqrt(1.0 + X * X)
    D = N.sqrt(1.0 + Y * Y)
    delta = N.sqrt(1.0 + X * X + Y * Y)
    return N.array([[C * C * D / delta**2, 0.0 * theta], [- D * X * Y / delta**2, 1.0 + 0.0 * theta]])

########################

def jacob_inv_N_to_sph(xi, eta):
    X = N.tan(xi)
    Y = N.tan(eta)
    C = N.sqrt(1.0 + X * X)
    D = N.sqrt(1.0 + Y * Y)
    delta = N.sqrt(1.0 + X * X + Y * Y)
    return N.array([[X * delta * delta / (C * C * N.sqrt(delta**2 - 1.0)), Y * delta * delta / (D * D * N.sqrt(delta**2 - 1.0))], \
                    [- Y / (C * C), X / (D * D)]])

def jacob_inv_sph_to_N(theta, phi):
    X = - N.tan(theta) * N.cos(phi)
    Y = - N.tan(theta) * N.sin(phi)
    C = N.sqrt(1.0 + X * X)
    D = N.sqrt(1.0 + Y * Y)
    delta = N.sqrt(1.0 + X * X + Y * Y)
    return N.array([[X * C * C / (delta**2 * N.sqrt(delta**2 - 1.0)), - C * C * Y / (delta**2 - 1.0)], \
                    [Y * D * D / (delta**2 * N.sqrt(delta**2 - 1.0)), D * D * X / (delta**2 - 1.0)]])
    
########################

def jacob_inv_S_to_sph(xi, eta):
    X = N.tan(xi)
    Y = N.tan(eta)
    C = N.sqrt(1.0 + X * X)
    D = N.sqrt(1.0 + Y * Y)
    delta = N.sqrt(1.0 + X * X + Y * Y)
    return N.array([[- X * delta * delta / (C * C * N.sqrt(delta**2 - 1.0)), - Y * delta * delta / (D * D * N.sqrt(delta**2 - 1.0))], \
                    [Y / (C * C), - X / (D * D)]])

def jacob_inv_sph_to_S(theta, phi):
    X = - N.tan(theta) * N.sin(phi)
    Y = - N.tan(theta) * N.cos(phi)
    C = N.sqrt(1.0 + X * X)
    D = N.sqrt(1.0 + Y * Y)
    delta = N.sqrt(1.0 + X * X + Y * Y)
    return N.array([[- X * C * C / (N.sqrt(delta**2 - 1.0) * delta * delta), C * C * Y / (delta**2 - 1.0)], \
                    [- Y * D * D / (N.sqrt(delta**2 - 1.0) * delta * delta), - D * D * X / (delta**2 - 1.0)]])

