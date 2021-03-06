import numpy as N

def jacob_A_to_sph(xi, eta):
    X = N.tan(xi)
    Y = N.tan(eta)
    C = N.sqrt(1.0 + X * X)
    D = N.sqrt(1.0 + Y * Y)
    delta = N.sqrt(1.0 + X * X + Y * Y)
    return N.array([[C * X * Y / delta**2, - C * D * D / delta**2], [1.0, 0.0]])

def jacob_sph_to_A(theta, phi):
    X = N.tan(phi)
    Y = 1.0 / N.tan(theta) / N.cos(phi)
    C = N.sqrt(1.0 + X * X)
    D = N.sqrt(1.0 + Y * Y)
    delta = N.sqrt(1.0 + X * X + Y * Y)
    return N.array([[0.0, 1.0], [- delta**2 / (C * D * D) , X * Y / D**2]])

########################

def jacob_B_to_sph(xi, eta):
    X = N.tan(xi)
    Y = N.tan(eta)
    C = N.sqrt(1.0 + X * X)
    D = N.sqrt(1.0 + Y * Y)
    delta = N.sqrt(1.0 + X * X + Y * Y)
    return N.array([[C * X * Y / delta**2, - C * D * D / delta**2], [1.0, 0.0]])

def jacob_sph_to_B(theta, phi):
    X = - 1.0 / N.tan(phi)
    Y = 1.0 / N.tan(theta) / N.sin(phi)
    C = N.sqrt(1.0 + X * X)
    D = N.sqrt(1.0 + Y * Y)
    delta = N.sqrt(1.0 + X * X + Y * Y)
    return N.array([[0.0, 1.0], [- delta**2 / (C * D * D) , X * Y / D**2]])

########################

def vec_C_to_sph(xi, eta, vxi, veta):
    X = N.tan(xi)
    Y = N.tan(eta)
    C = N.sqrt(1.0 + X * X)
    D = N.sqrt(1.0 + Y * Y)
    delta = N.sqrt(1.0 + X * X + Y * Y)
    return N.array([[C * C * D / delta**2, - (D * X * Y / delta**2)], [0.0, 1.0]])

def vec_sph_to_C(theta, phi, vth, vph):
    X = 1.0 / N.tan(theta) / N.cos(phi)
    Y = N.tan(phi)
    C = N.sqrt(1.0 + X * X)
    D = N.sqrt(1.0 + Y * Y)
    delta = N.sqrt(1.0 + X * X + Y * Y)
    return N.array([[delta**2 / (C * C * D), X * Y / C**2], [0.0, 1.0]])

########################

def vec_D_to_sph(xi, eta, vxi, veta):
    X = N.tan(xi)
    Y = N.tan(eta)
    C = N.sqrt(1.0 + X * X)
    D = N.sqrt(1.0 + Y * Y)
    delta = N.sqrt(1.0 + X * X + Y * Y)
    return N.array([[C * C * D / delta**2, - (D * X * Y / delta**2)], [0.0, 1.0]])

def vec_sph_to_D(theta, phi, vth, vph):
    X = 1.0 / N.tan(theta) / N.sin(phi)
    Y = - 1.0 / N.tan(phi) 
    C = N.sqrt(1.0 + X * X)
    D = N.sqrt(1.0 + Y * Y)
    delta = N.sqrt(1.0 + X * X + Y * Y)
    return N.array([[delta**2 / (C * C * D), X * Y / C**2], [0.0, 1.0]])

########################

def jacob_N_to_sph(xi, eta):
    X = N.tan(xi)
    Y = N.tan(eta)
    C = N.sqrt(1.0 + X * X)
    D = N.sqrt(1.0 + Y * Y)
    delta = N.sqrt(1.0 + X * X + Y * Y)
    return N.array([[X * C**2 / (delta**2 * N.sqrt(delta**2 - 1.0)), Y * D**2 / (delta**2 * N.sqrt(delta**2 - 1.0))], \
                    [- (Y * C**2 / (delta**2 - 1.0)), X * D**2 / (delta**2 - 1.0)]])

def jacob_sph_to_N(theta, phi):
    X = - N.tan(theta) * N.cos(phi)
    Y = - N.tan(theta) * N.sin(phi)
    C = N.sqrt(1.0 + X * X)
    D = N.sqrt(1.0 + Y * Y)
    delta = N.sqrt(1.0 + X * X + Y * Y)
    return N.array([[X * delta**2 / (C**2 * N.sqrt(delta**2 - 1.0)), - (Y / (C**2))], \
                    [Y * delta**2 / (D**2 * N.sqrt(delta**2 - 1.0)), X / (D**2)]])
    
########################

def jacob_S_to_sph(xi, eta):
    X = N.tan(xi)
    Y = N.tan(eta)
    C = N.sqrt(1.0 + X * X)
    D = N.sqrt(1.0 + Y * Y)
    delta = N.sqrt(1.0 + X * X + Y * Y)
    return N.array([[- X * C**2 / (delta**2 * N.sqrt(delta**2 - 1.0)), - Y * D**2 / (delta**2 * N.sqrt(delta**2 - 1.0))], \
                    [(Y * C**2 / (delta**2 - 1.0)), - X * D**2 / (delta**2 - 1.0)]])

def jacob_sph_to_S(theta, phi):
    X = - N.tan(theta) * N.sin(phi)
    Y = - N.tan(theta) * N.cos(phi)
    C = N.sqrt(1.0 + X * X)
    D = N.sqrt(1.0 + Y * Y)
    delta = N.sqrt(1.0 + X * X + Y * Y)
    return N.array([[- X * delta**2 / (C**2 * N.sqrt(delta**2 - 1.0)), (Y / (C**2))], \
                    [- Y * delta**2 / (D**2 * N.sqrt(delta**2 - 1.0)), - X / (D**2)]])

