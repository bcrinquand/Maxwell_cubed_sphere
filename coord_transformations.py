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
    global theta
    global phi
    X = N.tan(xi)
    Y = N.tan(eta)
    delta = N.sqrt(X * X + Y * Y)
    if (delta > 0):
        theta = N.pi / 2.0 - N.arctan(1.0 / delta)
        phi = N.arctan2(X / delta, - Y / delta)
    elif (delta == 0.0):
        theta = 0.0
        phi = 0.0
    return theta, phi

def coord_sph_to_N(theta, phi):
    eta = N.arctan(N.tan(theta) * N.sin(phi))
    xi  = N.arctan(- N.tan(theta) * N.cos(phi))
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
    eta = N.arctan(- N.tan(theta) * N.sin(phi))
    xi  = N.arctan(- N.tan(theta) * N.cos(phi))
    return xi, eta


# def communicate_E_patch(patch0, patch1):
        
#     top = topology[patch0, patch1]
#     if (top == 0):
#         return
#     elif (top[0] == 'x'):

#         if (top[1] == 'x'):

#             for k in range(NG, Neta + NG):         
            
#                 #########
#                 # Communicate fields from xi right edge of patch0 to xi left edge patch1
#                 ########

#                 i0 = Nxi + NG - 1 # Last active cell of xi edge of patch0                   
#                 i1 = NG - 1 # First ghost cell of xi edge of patch1
                    
#                 # Interpolate E^r       
#                 xi0, eta0 = transform_coords(patch1, patch0, xi[i1], eta[k])
#                 Er[i1, k, patch1] = interp(Er[i0, :, patch0], eta0, eta)
                
#                 # Interpolate E^xi and E_xi     
#                 xi0, eta0 = transform_coords(patch1, patch0, xi_yee[i1], eta[k])
#                 E1u_int = interp(E1u[i0, :, patch0], eta0, eta)
#                 E1d_int = interp(E1d[i0, :, patch0], eta0, eta)
                
#                 # Interpolate E^eta and E_eta     
#                 xi0, eta0 = transform_coords(patch1, patch0, xi[i1], eta_yee[k])
#                 E2u_int = interp(E2u[i0, :, patch0], eta0, eta)
#                 E2d_int = interp(E2d[i0, :, patch0], eta0, eta)
                
#                 # Convert from patch0 to patch1 coordinates
#                 E1u[i1, k, patch1], E2u[i1, k, patch1] = transform_vect(patch0, patch1, xi[i1], eta[k], E1u_int, E2u_int)
#                 E1d[i1, k, patch1], E2d[i1, k, patch1] = transform_form(patch0, patch1, xi[i1], eta[k], E1d_int, E2d_int)

#                 #########
#                 # Communicate fields from xi left edge of patch1 to xi right edge of patch0
#                 ########

#                 i0 = Nxi + NG # Last ghost cell of xi edge of patch0             
#                 i1 = NG  # First active cell of xi edge of patch1
                    
#                 # Interpolate E^r       
#                 xi1, eta1 = transform_coords(patch0, patch1, xi[i0], eta[k])
#                 Er[i0, k, patch0] = interp(Er[i1, :, patch1], eta1, eta)

#                 # Interpolate E^xi and E_xi     
#                 xi1, eta1 = transform_coords(patch0, patch1, xi_yee[i0], eta[k])
#                 E1u_int = interp(E1u[i1, :, patch1], eta1, eta)
#                 E1d_int = interp(E1d[i1, :, patch1], eta1, eta)

#                 # Interpolate E^eta and E_eta     
#                 xi0, eta0 = transform_coords(patch1, patch0, xi[i0], eta_yee[k])
#                 E2u_int = interp(E2u[i1, :, patch1], eta1, eta)
#                 E2d_int = interp(E2d[i1, :, patch1], eta1, eta)

#                 # Convert from patch1 to patch0 coordinates
#                 E1u[i0, k, patch0], E2u[i0, k, patch0] = transform_vect(patch1, patch0, xi[i0], eta[k], E1u_int, E2u_int)
#                 E1d[i0, k, patch0], E2d[i0, k, patch0] = transform_form(patch1, patch0, xi[i0], eta[k], E1d_int, E2d_int)

#         elif (top[1] == 'y'):

#             for k in range(NG, Neta + NG): 

#                 #########
#                 # Communicate fields from xi right edge of patch0 to eta left edge of patch1
#                 ########                

#                 i0 = Nxi + NG - 1 # Last active cell of xi edge of patch0                   
#                 j1 = NG - 1 # First ghost cell on eta edge of patch1
                    
#                 # Communicate E^r       
#                 xi0, eta0 = transform_coords(patch1, patch0, xi[k], eta[j1])
#                 Er[k, j1, patch1] = interp(Er[i0, :, patch0], eta0, eta)

#                 # Communicate E^xi and E_xi     
#                 xi0, eta0 = transform_coords(patch1, patch0, xi_yee[k], eta[j1])
#                 E1u_int = interp(E1u[i0, :, patch0], eta0, eta)
#                 E1d_int = interp(E1d[i0, :, patch0], eta0, eta)

#                 # Communicate E^eta and E_eta     
#                 xi0, eta0 = transform_coords(patch1, patch0, xi[k], eta_yee[j1])
#                 E2u_int = interp(E2u[i0, :, patch0], eta0, eta)
#                 E2d_int = interp(E2d[i0, :, patch0], eta0, eta)

#                 # Convert from patch0 to patch1 coordinates
#                 E1u[k, j1, patch1], E2u[k, j1, patch1] = transform_vect(patch0, patch1, xi[k], eta[j1], E1u_int, E2u_int)
#                 E1d[k, j1, patch1], E2d[k, j1, patch1] = transform_form(patch0, patch1, xi[k], eta[j1], E1d_int, E2d_int)

#                 #########
#                 # Communicate fields from eta left edge of patch1 to xi right edge of patch0
#                 ########                

#                 i0 = Nxi + NG # Last ghost cell of xi edge of patch0             
#                 j1 = NG  # First active cell of eta edge of patch1

#                 # Communicate E^r       
#                 xi1, eta1 = transform_coords(patch0, patch1, xi[i0], eta[k])
#                 Er[i0, k, patch0] = interp(Er[:, j1, patch1], eta1, eta)

#                 # Communicate E^xi and E_xi     
#                 xi1, eta1 = transform_coords(patch1, patch0, xi_yee[i0], eta[k])
#                 E1u_int = interp(E1u[:, j1, patch1], eta1, eta)
#                 E1d_int = interp(E1d[:, j1, patch1], eta1, eta)

#                 # Communicate E^eta and E_eta     
#                 xi1, eta1 = transform_coords(patch1, patch0, xi[i0], eta_yee[k])
#                 E2u_int = interp(E2u[:, j1, patch1], eta1, eta)
#                 E2d_int = interp(E2d[:, j1, patch1], eta1, eta)
        
#                 # Convert from patch1 to patch0 coordinates
#                 E1u[i0, k, patch0], E2u[i0, k, patch0] = transform_vect(patch1, patch0, xi[i0], eta[k], E1u_int, E2u_int)
#                 E1d[i0, k, patch0], E2d[i0, k, patch0] = transform_form(patch1, patch0, xi[i0], eta[k], E1d_int, E2d_int)

#     elif (top[0] == 'y'):

#         if (top[1] == 'y'):
            
#             for k in range(NG, Nxi + NG):

#                 #########
#                 # Communicate fields from eta top edge of patch0 to eta bottom edge of patch1
#                 ########                

#                 j0 = Neta + NG -1 # Last active cell of eta edge of patch0
#                 j1 = NG - 1 # First ghost cell of eta edge of patch1

#                 # Communicate E^r       
#                 xi0, eta0 = transform_coords(patch1, patch0, xi[k], eta[j1])
#                 Er[k, j1, patch1] = interp(Er[:, j0, patch0], xi0, xi)

#                 # Communicate E^xi and E_xi     
#                 xi0, eta0 = transform_coords(patch1, patch0, xi_yee[k], eta[j1])
#                 E1u_int = interp(E1u[:, j0, patch0], xi0, xi)
#                 E1d_int = interp(E1d[:, j0, patch0], xi0, xi)

#                 # Communicate E^eta and E_eta     
#                 xi0, eta0 = transform_coords(patch1, patch0, xi[k], eta_yee[j1])
#                 E2u_int = interp(E2u[:, j0, patch0], xi0, xi)
#                 E2d_int = interp(E2d[:, j0, patch0], xi0, xi)
                
#                 # Convert from patch0 to patch1 coordinates
#                 E1u[k, j1, patch1], E2u[k, j1, patch1] = transform_vect(patch0, patch1, xi[k], eta[j1], E1u_int, E2u_int)
#                 E1d[k, j1, patch1], E2d[k, j1, patch1] = transform_form(patch0, patch1, xi[k], eta[j1], E1d_int, E2d_int)

#                 #########
#                 # Communicate fields from eta bottom edge of patch1 to eta top edge of patch0
#                 ########                

#                 j0 = Neta + NG # Last ghost cell of eta edge of patch0
#                 j1 = NG # First active cell of eta edge of patch1
#                 # Communicate E^r       
#                 xi1, eta1 = transform_coords(patch0, patch1, xi[k], eta[j0])
#                 Er[k, j0, patch0] = interp(Er[:, j1, patch1], xi1, xi)

#                 # Communicate E^xi and E_xi     
#                 xi1, eta1 = transform_coords(patch0, patch1, xi_yee[k], eta[j0])
#                 E1u_int = interp(E1u[:, j1, patch1], xi1, xi)
#                 E1d_int = interp(E1d[:, j1, patch1], xi1, xi)

#                 # Communicate E^eta and E_eta     
#                 xi1, eta1 = transform_coords(patch0, patch1, xi[k], eta_yee[j0])
#                 E2u_int = interp(E2u[:, j1, patch1], xi1, xi)
#                 E2d_int = interp(E2d[:, j1, patch1], xi1, xi)

#                 # Convert from patch1 to patch0 coordinates
#                 E1u[k, j0, patch0], E2u[k, j0, patch0] = transform_vect(patch1, patch0, xi[k], eta[j0], E1u_int, E2u_int)
#                 E1d[k, j0, patch0], E2d[k, j0, patch0] = transform_form(patch1, patch0, xi[k], eta[j0], E1d_int, E2d_int)

#         elif (top[1] == 'x'):

#             for k in range(NG, Nxi + NG):

#                 #########
#                 # Communicate fields from eta top edge of patch0 to xi bottom edge of patch1
#                 ########                

#                 j0 = Neta + NG -1 # Last active cell of eta edge of patch0
#                 i1 = NG - 1 # First ghost cell on xi edge of patch1
#                 # Communicate E^r       
#                 xi0, eta0 = transform_coords(patch1, patch0, xi[i1], eta[k])
#                 Er[i1, k, patch1] = interp(Er[:, j0, patch0], xi0, xi)

#                 # Communicate E^xi and E_xi     
#                 xi0, eta0 = transform_coords(patch1, patch0, xi_yee[i1], eta[k])
#                 E1u_int = interp(E1u[:, j0, patch0], xi0, xi)
#                 E1d_int = interp(E1d[:, j0, patch0], xi0, xi)

#                 # Communicate E^eta and E_eta     
#                 xi0, eta0 = transform_coords(patch1, patch0, xi[i1], eta_yee[k])
#                 E2u_int = interp(E2u[:, j0, patch0], xi0, xi)
#                 E2d_int = interp(E2d[:, j0, patch0], xi0, xi)

#                 # Convert from patch0 to patch1 coordinates
#                 E1u[i1, k, patch1], E2u[i1, k, patch1] = transform_vect(patch0, patch1, xi[i1], eta[k], E1u_int, E2u_int)
#                 E1d[i1, k, patch1], E2d[i1, k, patch1] = transform_form(patch0, patch1, xi[i1], eta[k], E1d_int, E2d_int)

#                 #########
#                 # Communicate fields from xi bottom edge of patch1 to eta top edge of patch0
#                 ########                

#                 j0 = Neta + NG # Last ghost cell of eta edge of patch0
#                 j1 = NG # First active cell of eta edge of patch1
#                 # Communicate E^r       
#                 xi1, eta1 = transform_coords(patch0, patch1, xi[k], eta[j0])
#                 Er[k, j0, patch0] = interp(Er[:, j1, patch1], xi1, xi)

#                 # Communicate E^xi and E_xi     
#                 xi0, eta0 = transform_coords(patch0, patch1, xi_yee[k], eta[j0])
#                 E1u_int = interp(E1u[:, j1, patch1], xi1, xi)
#                 E1d_int = interp(E1d[:, j1, patch1], xi1, xi)

#                 # Communicate E^eta and E_eta     
#                 xi0, eta0 = transform_coords(patch0, patch1, xi[k], eta_yee[j0])
#                 E2u_int = interp(E2u[:, j1, patch1], xi1, xi)
#                 E2d_int = interp(E2d[:, j1, patch1], xi1, xi)        
        
#                 # Convert from patch1 to patch0 coordinates
#                 E1u[k, j0, patch0], E2u[k, j0, patch0] = transform_vect(patch1, patch0, xi[k], eta[j0], E1u_int, E2u_int)
#                 E1d[k, j0, patch0], E2d[k, j0, patch0] = transform_form(patch1, patch0, xi[k], eta[j0], E1d_int, E2d_int)
