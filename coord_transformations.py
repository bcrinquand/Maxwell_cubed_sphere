import numpy as N

def coord_A_to_sph(xi, eta):
    X = N.tan(xi)
    Y = N.tan(eta)
    theta = 0.5 * N.pi - N.arctan(Y / N.sqrt(1.0 + X * X))
    phi = N.arctan2(X / N.sqrt(1.0 + X * X), 1.0 / N.sqrt(1.0 + X * X))
    return theta, phi

def coord_sph_to_A(theta, phi):
    eta = N.arctan(1.0 / N.tan(theta) / N.cos(phi))
    xi = N.arctan(N.tan(phi))
    return xi, eta

def coord_B_to_sph(xi, eta):
    X = N.tan(xi)
    Y = N.tan(eta)
    theta = 0.5 * N.pi - N.arctan(Y / N.sqrt(1.0 + X * X))
    phi = N.arctan2(1.0 / N.sqrt(1.0 + X * X), - X / N.sqrt(1.0 + X * X))
    return theta, phi

def coord_sph_to_B(theta, phi):
    eta = N.arctan(1.0 / N.tan(theta) / N.cos(phi))
    xi  = N.arctan(- 1.0 / N.tan(phi))
    return xi, eta

def coord_C_to_sph(xi, eta):
    X = N.tan(xi)
    Y = N.tan(eta)
    theta = 0.5 * N.pi - N.arctan(Y / N.sqrt(1.0 + X * X))
    phi = N.arctan2(- X / N.sqrt(1.0 + X * X), - 1.0 / N.sqrt(1.0 + X * X))
    return theta, phi

def coord_sph_to_C(theta, phi):
    eta = N.arctan(- 1.0 / N.tan(theta) / N.cos(phi))
    xi  = N.arctan(N.tan(phi))
    return xi, eta

def coord_D_to_sph(xi, eta):
    X = N.tan(xi)
    Y = N.tan(eta)
    theta = 0.5 * N.pi - N.arctan(Y / N.sqrt(1.0 + X * X))
    phi = N.arctan2(- 1.0 / N.sqrt(1.0 + X * X), X / N.sqrt(1.0 + X * X))
    return theta, phi

def coord_sph_to_D(theta, phi):
    eta = N.arctan(- 1.0 / N.tan(theta) / N.cos(phi))
    xi  = N.arctan(- 1.0 / N.tan(phi))
    return xi, eta

def coord_N_to_sph(xi, eta):
    X = N.tan(xi)
    Y = N.tan(eta)
    delta = N.sqrt(X * X + Y * Y)
    theta = N.pi / 2.0 - N.arctan(1.0 / delta)
    phi = N.arctan2(X / delta, - Y / delta)
    return theta, phi

def coord_sph_to_N(theta, phi):
    eta = N.arctan(N.tan(theta) * N.sin(phi))
    xi  = N.arctan(- N.tan(theta) * N.cos(phi))
    return xi, eta

def coord_S_to_sph(xi, eta):
    X = N.tan(xi)
    Y = N.tan(eta)
    delta = N.sqrt(X * X + Y * Y)
    theta = N.pi / 2.0 - N.arctan(-1.0 /  delta)
    phi = N.arctan2(X / delta, Y / delta)
    return theta, phi

def coord_sph_to_S(theta, phi):
    eta = N.arctan(- N.tan(theta) * N.sin(phi))
    xi  = N.arctan(- N.tan(theta) * N.cos(phi))
    return xi, eta