import numpy as N

def jacob_A_to_sph(xi, eta):
    X = N.tan(xi)
    Y = N.tan(eta)
    C = N.sqrt(1.0 + X * X)
    D = N.sqrt(1.0 + Y * Y)
    delta = N.sqrt(1.0 + X * X + Y * Y)
    return N.array([[C * X * Y / delta**2, - C * D * D / delta**2], [1.0 + 0.0 * xi, 0.0 * xi]])

def jacob_sph_to_A(theta, phi):
    X = N.tan(phi)
    Y = 1.0 / N.tan(theta) / N.cos(phi)
    C = N.sqrt(1.0 + X * X)
    D = N.sqrt(1.0 + Y * Y)
    delta = N.sqrt(1.0 + X * X + Y * Y)
    return N.array([[0.0 * theta, 1.0 + 0.0 * theta], [- delta**2 / (C * D * D) , X * Y / D**2]])

########################

def jacob_B_to_sph(xi, eta):
    X = N.tan(xi)
    Y = N.tan(eta)
    C = N.sqrt(1.0 + X * X)
    D = N.sqrt(1.0 + Y * Y)
    delta = N.sqrt(1.0 + X * X + Y * Y)
    return N.array([[C * X * Y / delta**2, - C * D * D / delta**2], [1.0 + 0.0 * xi, 0.0 * xi]])

def jacob_sph_to_B(theta, phi):
    X = - 1.0 / N.tan(phi)
    Y = 1.0 / N.tan(theta) / N.sin(phi)
    C = N.sqrt(1.0 + X * X)
    D = N.sqrt(1.0 + Y * Y)
    delta = N.sqrt(1.0 + X * X + Y * Y)
    return N.array([[0.0 * theta, 1.0 + 0.0 * theta], [- delta**2 / (C * D * D) , X * Y / D**2]])

########################

def jacob_C_to_sph(xi, eta):
    X = N.tan(xi)
    Y = N.tan(eta)
    C = N.sqrt(1.0 + X * X)
    D = N.sqrt(1.0 + Y * Y)
    delta = N.sqrt(1.0 + X * X + Y * Y)
    return N.array([[C * C * D / delta**2, - (D * X * Y / delta**2)], [0.0 * xi, 1.0 + 0.0 * xi]])

def jacob_sph_to_C(theta, phi):
    X = 1.0 / N.tan(theta) / N.cos(phi)
    Y = N.tan(phi)
    C = N.sqrt(1.0 + X * X)
    D = N.sqrt(1.0 + Y * Y)
    delta = N.sqrt(1.0 + X * X + Y * Y)
    return N.array([[delta**2 / (C * C * D), X * Y / C**2], [0.0 * theta, 1.0 + 0.0 * theta]])

########################

def jacob_D_to_sph(xi, eta):
    X = N.tan(xi)
    Y = N.tan(eta)
    C = N.sqrt(1.0 + X * X)
    D = N.sqrt(1.0 + Y * Y)
    delta = N.sqrt(1.0 + X * X + Y * Y)
    return N.array([[C * C * D / delta**2, - (D * X * Y / delta**2)], [0.0 * xi, 1.0 + 0.0 * xi]])

def jacob_sph_to_D(theta, phi):
    X = 1.0 / N.tan(theta) / N.sin(phi)
    Y = - 1.0 / N.tan(phi) 
    C = N.sqrt(1.0 + X * X)
    D = N.sqrt(1.0 + Y * Y)
    delta = N.sqrt(1.0 + X * X + Y * Y)
    return N.array([[delta**2 / (C * C * D), X * Y / C**2], [0.0 * theta, 1.0 + 0.0 * theta]])

########################

def jacob_N_to_sph(xi, eta):
    X = N.tan(xi)
    Y = N.tan(eta)
    C = N.sqrt(1.0 + X * X)
    D = N.sqrt(1.0 + Y * Y)
    delta = N.sqrt(1.0 + X * X + Y * Y)
    
    if (delta > 1.0):
        return N.array([[X * C**2 / (delta**2 * N.sqrt(delta**2 - 1.0)), Y * D**2 / (delta**2 * N.sqrt(delta**2 - 1.0))], \
                        [- (Y * C**2 / (delta**2 - 1.0)), X * D**2 / (delta**2 - 1.0)]])
    elif (delta == 1.0):
        return N.array([[0.0, 0.0], \
                        [- 0.0, 0.0]])
    else:
        return

def jacob_sph_to_N(theta, phi):
    X = - N.tan(theta) * N.cos(phi)
    Y = - N.tan(theta) * N.sin(phi)
    C = N.sqrt(1.0 + X * X)
    D = N.sqrt(1.0 + Y * Y)
    delta = N.sqrt(1.0 + X * X + Y * Y)

    if (delta > 1.0):
        return N.array([[X * delta**2 / (C**2 * N.sqrt(delta**2 - 1.0)), - (Y / (C**2))], \
                    [Y * delta**2 / (D**2 * N.sqrt(delta**2 - 1.0)), X / (D**2)]])
    elif (delta == 1.0):
        return N.array([[0.0, 0.0], \
                        [- 0.0, 0.0]])
    else:
        return
    
########################

def jacob_S_to_sph(xi, eta):
    X = N.tan(xi)
    Y = N.tan(eta)
    C = N.sqrt(1.0 + X * X)
    D = N.sqrt(1.0 + Y * Y)
    delta = N.sqrt(1.0 + X * X + Y * Y)

    if (delta > 1.0):
        return N.array([[- X * C**2 / (delta**2 * N.sqrt(delta**2 - 1.0)), - Y * D**2 / (delta**2 * N.sqrt(delta**2 - 1.0))], \
                    [(Y * C**2 / (delta**2 - 1.0)), - X * D**2 / (delta**2 - 1.0)]])
    elif (delta == 1.0):
        return N.array([[0.0, 0.0], \
                        [- 0.0, 0.0]])
    else:
        return

def jacob_sph_to_S(theta, phi):
    X = - N.tan(theta) * N.sin(phi)
    Y = - N.tan(theta) * N.cos(phi)
    C = N.sqrt(1.0 + X * X)
    D = N.sqrt(1.0 + Y * Y)
    delta = N.sqrt(1.0 + X * X + Y * Y)

    if (delta > 1.0):
        return N.array([[- X * delta**2 / (C**2 * N.sqrt(delta**2 - 1.0)), (Y / (C**2))], \
                    [- Y * delta**2 / (D**2 * N.sqrt(delta**2 - 1.0)), - X / (D**2)]])
    elif (delta == 1.0):
        return N.array([[0.0, 0.0], \
                        [- 0.0, 0.0]])
    else:
        return
    
########################
  
def jacob_A_to_cart(r, xi, eta):
    X = N.tan(xi)
    Y = N.tan(eta)
    C = N.sqrt(1.0 + X * X)
    D = N.sqrt(1.0 + Y * Y)
    delta = N.sqrt(1.0 + X * X + Y * Y)
    return N.array([[1 / delta, - r * X * C**2 / delta**3, - r * Y * D**2 / delta**3], [X / delta, r * C**2 * D**2 / delta**3, - r * X * Y * D**2 / delta**3], [Y / delta, - r * X * Y * C**2 / delta**3, r * C**2 * D**2 / delta**3]])

def jacob_cart_to_A(x, y, z):
    r = np.sqrt(x**2 + y**2 + z**2)
    X = y / x
    Y = z / x
    C = N.sqrt(1.0 + X * X)
    D = N.sqrt(1.0 + Y * Y)
    delta = N.sqrt(1.0 + X * X + Y * Y)
    return N.array([[1 / delta, X / delta, Y / delta], [- X * delta / (r * C**2), delta / (r * C**2), 0 * X], [- Y * delta / (r * D**2), 0 * X, delta / (r * D**2)]])

########################
  
def jacob_B_to_cart(r, xi, eta):
    X = N.tan(xi)
    Y = N.tan(eta)
    C = N.sqrt(1.0 + X * X)
    D = N.sqrt(1.0 + Y * Y)
    delta = N.sqrt(1.0 + X * X + Y * Y)
    return N.array([[- X / delta, - r * C**2 * D**2 / delta**3, r * X * Y * D**2 / delta**3], [1 / delta, - r * X * C**2 / delta**3, - r * Y * D**2 / delta**3], [Y / delta, - r * X * Y * C**2 / delta**3, r * C**2 * D**2 / delta**3]])

def jacob_cart_to_B(x, y, z):
    r = np.sqrt(x**2 + y**2 + z**2)
    X = - x / y
    Y = z / y
    C = N.sqrt(1.0 + X * X)
    D = N.sqrt(1.0 + Y * Y)
    delta = N.sqrt(1.0 + X * X + Y * Y)
    return N.array([[- X / delta, 1 / delta, Y / delta], [- delta / (r * C**2), - X * delta / (r * C**2), 0 * X], [0 * X, - Y * delta / (r * D**2), delta / (r * D**2)]])

########################
  
def jacob_C_to_cart(r, xi, eta):
    X = N.tan(xi)
    Y = N.tan(eta)
    C = N.sqrt(1.0 + X * X)
    D = N.sqrt(1.0 + Y * Y)
    delta = N.sqrt(1.0 + X * X + Y * Y)
    return N.array([[- 1 / delta, r * X * C**2 / delta**3, r * Y * D**2 / delta**3], [- Y / delta, r * X * Y * C**2 / delta**3, - r * C**2 * D**2 / delta**3], [X / delta, r * C**2 * D**2 / delta**3, - r * X * Y * D**2 / delta**3]])

def jacob_cart_to_C(x, y, z):
    r = np.sqrt(x**2 + y**2 + z**2)
    X = z / x
    Y = y / x
    C = N.sqrt(1.0 + X * X)
    D = N.sqrt(1.0 + Y * Y)
    delta = N.sqrt(1.0 + X * X + Y * Y)
    return N.array([[- 1 / delta, - Y / delta, X / delta], [X * delta / (r * C**2), 0 * X, delta / (r * C**2)], [Y * delta / (r * D**2), - delta / (r * D**2), 0 * X]])

########################
  
def jacob_D_to_cart(r, xi, eta):
    X = N.tan(xi)
    Y = N.tan(eta)
    C = N.sqrt(1.0 + X * X)
    D = N.sqrt(1.0 + Y * Y)
    delta = N.sqrt(1.0 + X * X + Y * Y)
    return N.array([[Y / delta, - r * X * Y * C**2 / delta**3, r * C**2 * D**2 / delta**3], [- 1 / delta, r * X * C**2 / delta**3, r * Y * D**2 / delta**3], [- X / delta, - r * C**2 * D**2 / delta**3, r * X * Y * D**2 / delta**3]])

def jacob_cart_to_D(x, y, z):
    r = np.sqrt(x**2 + y**2 + z**2)
    X = z / y
    Y = - x / y
    C = N.sqrt(1.0 + X * X)
    D = N.sqrt(1.0 + Y * Y)
    delta = N.sqrt(1.0 + X * X + Y * Y)
    return N.array([[Y / delta, - 1 / delta, - X / delta], [0 * X, X * delta / (r * C**2), - delta / (r * C**2)], [delta / (r * D**2), Y * delta / (r * D**2), , 0 * X]])

########################
  
def jacob_N_to_cart(r, xi, eta):
    X = N.tan(xi)
    Y = N.tan(eta)
    C = N.sqrt(1.0 + X * X)
    D = N.sqrt(1.0 + Y * Y)
    delta = N.sqrt(1.0 + X * X + Y * Y)
    return N.array([[- X / delta, - r * C**2 * D**2 / delta**3, r * X * Y * D**2 / delta**3], [- Y / delta, r * X * Y * C**2 / delta**3, - r * C**2 * D**2 / delta**3], [1 / delta, - r * X * C**2 / delta**3, - r * Y * D**2 / delta**3]])

def jacob_cart_to_N(x, y, z):
    r = np.sqrt(x**2 + y**2 + z**2)
    X = - x / z
    Y = - y / z
    C = N.sqrt(1.0 + X * X)
    D = N.sqrt(1.0 + Y * Y)
    delta = N.sqrt(1.0 + X * X + Y * Y)
    return N.array([[- X / delta, - Y / delta, 1 / delta], [- delta / (r * C**2), 0 * X, - X * delta / (r * C**2)], [0 * X, - delta / (r * D**2), - Y * delta / (r * D**2)]])

########################
  
def jacob_S_to_cart(r, xi, eta):
    X = N.tan(xi)
    Y = N.tan(eta)
    C = N.sqrt(1.0 + X * X)
    D = N.sqrt(1.0 + Y * Y)
    delta = N.sqrt(1.0 + X * X + Y * Y)
    return N.array([[Y / delta, - r * X * Y * C**2 / delta**3, r * C**2 * D**2 / delta**3], [X / delta, r * C**2 * D**2 / delta**3, - r * X * Y * D**2 / delta**3], [- 1 / delta, r * X * C**2 / delta**3, r * Y * D**2 / delta**3]])

def jacob_cart_to_S(x, y, z):
    r = np.sqrt(x**2 + y**2 + z**2)
    X = - y / z
    Y = - x / z
    C = N.sqrt(1.0 + X * X)
    D = N.sqrt(1.0 + Y * Y)
    delta = N.sqrt(1.0 + X * X + Y * Y)
    return N.array([[Y / delta, X / delta, - 1 / delta], [0 * X, delta / (r * C**2), X * delta / (r * C**2)], [delta / (r * D**2), 0 * X, Y * delta / (r * D**2)]])
