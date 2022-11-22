import numpy as N
from math import *
from tqdm import tqdm
import copy

from cubed_sphere import *
from vis import newplot

amp = 0.0
n_mode = 2
n_iter = 10

wave = 2.0 * (x_max - x_min) / n_mode
Bz0 = amp * N.cos(2.0 * N.pi * (xBz_grid - x_min) / wave) * \
    N.cos(2.0 * N.pi * (yBz_grid - x_min) / wave)
Ex0 = N.zeros((Nx_half, Ny_int))
Ey0 = N.zeros((Nx_int, Ny_half))

for p in range(n_patches):
    Bz[p, :, :] = Bz0[:, :]
    Ex[p, :, :] = Ex0[:, :]
    Ey[p, :, :] = Ey0[:, :]

initialize_part()

########
# Main routine
########

idump = 0
energy = N.zeros((n_patches, Nt))
patches = N.array(range(n_patches))

# Fields at previous time steps
Ex0 = N.zeros_like(Ex)
Ey0 = N.zeros_like(Ey)
Bz0 = N.zeros_like(Bz)

# clear directory `dir_dump`
output_dir = 'snapshots_penalty'


def clear_dir(output_dir):
    import os
    import shutil
    if os.path.exists(output_dir):
        shutil.rmtree(output_dir)
    os.makedirs(output_dir)


clear_dir(output_dir)

for it in tqdm(range(Nt), "Progression"):
    if ((it % FDUMP) == 0):
        newplot(output_dir, idump, it,
                x_right=x_max,
                dx=dx, dy=dy,
                Ex=Ex, xEx_grid=xEx_grid, yEx_grid=yEx_grid)
        idump += 1

    # print(it, Nt)

    # 1st Faraday substep, starting from B at n-1/2, E at n, finishing with B at n
    compute_diff_E(patches)
    Bz[patches, :, :] += 0.5 * dt * \
        (dExdy[patches, :, :] - dEydx[patches, :, :])

    # Here, Bz is defined at n, no need for averaging
    # BC_conducting_B(0.5 * dt, Ex[:, :, :], Ey[:, :, :], Bz[:, :, :])
    BC_absorbing_B(0.5 * dt, Ex, Ey, copy.deepcopy(Bz))
    BC_penalty_B(0.5 * dt, Ex, Ey, copy.deepcopy(Bz))

    Bz0 = copy.deepcopy(Bz)

    # Particle push
    for ip in range(np):
        # push_u(it, ip)
        # impose_velocity_part(it)
        uxp[it, 0] = ux0
        uxp[it, 1] = - ux0
        push_x(it, ip)
        BC_part(it, ip)

    # Current deposition
    deposit_J(it)

    compute_divcharge(patches)

    # filter_current(0, n_iter)

    # 2nd Faraday substep, starting with B at n, finishing with B at n + 1/2
    compute_diff_E(patches)
    Bz[patches, :, :] += 0.5 * dt * \
        (dExdy[patches, :, :] - dEydx[patches, :, :])

    # Use Bz0, defined at n, this time
    # BC_conducting_B(0.5 * dt, Ex[:, :, :], Ey[:, :, :], Bz0[:, :, :])
    BC_absorbing_B(0.5 * dt, Ex, Ey, Bz0)
    BC_penalty_B(0.5 * dt, Ex, Ey, Bz0)

    # Ampère step, starting with E at n, finishing with E at n + 1
    Ex0 = copy.deepcopy(Ex)
    Ey0 = copy.deepcopy(Ey)
    compute_diff_B(patches)
    # ! TODO: this should not include first/last cells
    # Ex[patches, :, :] += dt * \
    #     (dBzdy[patches, :, :] - 4.0 * N.pi * Jx[patches, :, :])
    # Ey[patches, :, :] += dt * \
    #     (-dBzdx[patches, :, :] - 4.0 * N.pi * Jy[patches, :, :])
    Ex[patches, :, :] += dt * \
        (dBzdy[patches, :, :] - 4.0 * N.pi * Jx[patches, :, :])
    Ey[0, :-1, :] += dt * (-dBzdx[0, :-1, :] - 4.0 * N.pi * Jy[0, :-1, :])
    Ey[1, 1:, :] += dt * (-dBzdx[1, 1:, :] - 4.0 * N.pi * Jy[1, 1:, :])

    # Use averaged E field, defined at n + 1/2
    # BC_conducting_E(dt, 0.5 * (Ex0[:, :, :] + Ex[:, :, :]),
    #                 0.5 * (Ey0[:, :, :] + Ey[:, :, :]), Bz[:, :, :])
    BC_absorbing_E(dt, 0.5 * (Ex0 + Ex), 0.5 * (Ey0 + Ey), Bz)
    # BC_penalty_E(dt, 0.5 * (Ex0 + Ex), 0.5 * (Ey0 + Ey), Bz)

    sigma = 1.0
    Ey1 = copy.deepcopy(Ey)
    dx1 = sigma * dt
    delta = dx + 2 * dx1
    SY_A = (-dBzdx[0, -1, :] - 4.0 * N.pi * Jy[0, -1, :])
    SY_B = (-dBzdx[1, 0, :] - 4.0 * N.pi * Jy[1, 0, :])
    deltaBZ_AB = (Bz[0, -1, :] - Bz[1, 0, :])

    Ey[0, -1, :] = \
        (1 / delta) * (dx * Ey1[0, -1, :] + 2 * dx1 * Ey1[1, 0, :]) +\
        (dt / delta) * ((dx + dx1) * SY_A + dx1 * SY_B) +\
        + (2 * dx1 / dx) * deltaBZ_AB

    Ey[1, 0, :] = \
        (1 / delta) * (2 * dx1 * Ey1[0, -1, :] + dx * Ey1[1, 0, :]) +\
        (dt / delta) * (dx1 * SY_A + (dx + dx1) * SY_B) +\
        + (2 * dx1 / dx) * deltaBZ_AB

    compute_divE(patches, Ex, Ey)

    energy[0, it] = dx * dy * \
        N.sum(Bz[0, :, :]**2) + N.sum(Ex[0, :, :]**2) + N.sum(Ey[0, :, :]**2)
    energy[1, it] = dx * dy * \
        N.sum(Bz[1, :, :]**2) + N.sum(Ex[1, :, :]**2) + N.sum(Ey[1, :, :]**2)
    compute_charge(it)
