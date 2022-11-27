import numpy as N
from math import *
from tqdm import tqdm
import copy

from cubed_sphere import *
from vis import newplot

initialize_part()

idump = 0
energy = N.zeros((n_patches, Nt))
patches = N.array(range(n_patches))

PARAMS = {"use_implicit_E": True, "use_implicit_B": True, "sigma": 1}

# Fields at previous time steps
Ex0 = N.zeros_like(Ex)
Ey0 = N.zeros_like(Ey)
Bz0 = N.zeros_like(Bz)

output_dir = "frames_implEB"


def clear_dir(output_dir):
    import os
    import shutil

    if os.path.exists(output_dir):
        shutil.rmtree(output_dir)
    os.makedirs(output_dir)


clear_dir(output_dir)


def write_output(it):
    global idump
    region_A = (0.97, 0.98, 0.7, 0.74)
    indices_A = (
        N.argmin(N.abs(x_half - region_A[0])),
        N.argmin(N.abs(x_half - region_A[1])),
        N.argmin(N.abs(y_int - region_A[2])),
        N.argmin(N.abs(y_int - region_A[3])),
    )
    x_slice = slice(indices_A[0], indices_A[1])
    y_slice = slice(indices_A[2], indices_A[3])
    flux0[it] = (
        spi.simps(Ex[0, indices_A[0], y_slice], x=y_int[y_slice])
        - spi.simps(Ex[0, indices_A[1], y_slice], x=y_int[y_slice])
        + spi.simps(Ey[0, x_slice, indices_A[2]], x=x_half[x_slice])
        - spi.simps(Ey[0, x_slice, indices_A[3]], x=x_half[x_slice])
    )
    # flux1[it] = spi.simps(Ex[1, i1r, :], x=y_int) - spi.simps(Ex[1, i1l, :], x=y_int) \
    #     + spi.simps(Ey[1, i1l:i1r, -1], x=x_int[i1l:i1r]) - \
    #     spi.simps(Ey[1, i1l:i1r, 0], x=x_int[i1l:i1r])
    actual_region_A = (
        x_half[indices_A[0]],
        x_half[indices_A[1]],
        y_int[indices_A[2]],
        y_int[indices_A[3]],
    )
    newplot(
        output_dir,
        idump,
        it,
        x_right=x_max,
        dx=dx,
        dy=dy,
        Ex=Ex,
        prtls=(tag, xp, yp, wp, np),
        fluxes=(actual_region_A, time, flux0, flux1),
        charge=q,
        xEx_grid=xEx_grid,
        yEx_grid=yEx_grid,
    )
    idump += 1


def push_particles(it):
    for ip in range(np):
        uxp[it, 0] = ux0
        uxp[it, 1] = -ux0
        push_x(it, ip)
        BC_part(it, ip)


for it in tqdm(N.arange(Nt)):
    if (it % FDUMP) == 0:
        write_output(it)

    compute_diff_E(patches)

    if PARAMS["use_implicit_B"]:
        # implicit B penalty terms
        Bz1 = copy.deepcopy(Bz)
        dx1 = PARAMS["sigma"] * dt
        delta = dx + 2 * dx1
        SZ_A = dExdy[0, -1, :]
        SZ_B = dExdy[1, 0, :]
        deltaEY_AB = Ey[0, -1, :] - Ey[1, 0, :]

        Bz[0, -1, :] = (
            (1 / delta) * (dx * Bz1[0, -1, :] + 2 * dx1 * Bz1[1, 0, :])
            + (dt / delta) * ((dx + dx1) * SZ_A + dx1 * SZ_B)
            + (2 * dx1 / dx) * deltaEY_AB
        )

        Bz[1, 0, :] = (
            (1 / delta) * (2 * dx1 * Bz1[0, -1, :] + dx * Bz1[1, 0, :])
            + (dt / delta) * (dx1 * SZ_A + (dx + dx1) * SZ_B)
            + (2 * dx1 / dx) * deltaEY_AB
        )
        
        Bz[0, :-1, :] += 0.5 * dt * (dExdy[0, :-1, :] - dEydx[0, :-1, :])
        Bz[1, 1:, :] += 0.5 * dt * (dExdy[1, 1:, :] - dEydx[1, 1:, :])
        BC_absorbing_B(0.5 * dt, Ex, Ey, copy.deepcopy(Bz))
    else:
        # explicit B penalty terms
        Bz[patches] += 0.5 * dt * (dExdy[patches] - dEydx[patches])
        BC_absorbing_B(0.5 * dt, Ex, Ey, copy.deepcopy(Bz))
        BC_penalty_B(0.5 * dt, PARAMS["sigma"], Ex, Ey, copy.deepcopy(Bz))
        Bz0 = copy.deepcopy(Bz)
        Bz[patches] += 0.5 * dt * (dExdy[patches] - dEydx[patches])
        BC_absorbing_B(0.5 * dt, Ex, Ey, Bz0)
        BC_penalty_B(0.5 * dt, PARAMS["sigma"], Ex, Ey, Bz0)

    compute_diff_B(patches)
    Ex0 = copy.deepcopy(Ex)
    Ey0 = copy.deepcopy(Ey)
    push_particles(it)
    deposit_J(it)

    Ex[patches] += dt * (dBzdy[patches] - 4.0 * N.pi * Jx[patches])
    if PARAMS["use_implicit_E"]:
        Ey[0, :-1, :] += dt * (-dBzdx[0, :-1, :] - 4.0 * N.pi * Jy[0, :-1, :])
        Ey[1, 1:, :] += dt * (-dBzdx[1, 1:, :] - 4.0 * N.pi * Jy[1, 1:, :])
    else:
        Ey[patches] += dt * (-dBzdx[patches] - 4.0 * N.pi * Jy[patches])

    BC_absorbing_E(dt, 0.5 * (Ex0 + Ex), 0.5 * (Ey0 + Ey), Bz)

    if PARAMS["use_implicit_E"]:
        # implicit E penalty terms
        Ey1 = copy.deepcopy(Ey)
        dx1 = PARAMS["sigma"] * dt
        delta = dx + 2 * dx1
        SY_A = -dBzdx[0, -1, :] - 4.0 * N.pi * Jy[0, -1, :]
        SY_B = -dBzdx[1, 0, :] - 4.0 * N.pi * Jy[1, 0, :]
        deltaBZ_AB = Bz[0, -1, :] - Bz[1, 0, :]

        Ey[0, -1, :] = (
            (1 / delta) * (dx * Ey1[0, -1, :] + 2 * dx1 * Ey1[1, 0, :])
            + (dt / delta) * ((dx + dx1) * SY_A + dx1 * SY_B)
            + (2 * dx1 / dx) * deltaBZ_AB
        )

        Ey[1, 0, :] = (
            (1 / delta) * (2 * dx1 * Ey1[0, -1, :] + dx * Ey1[1, 0, :])
            + (dt / delta) * (dx1 * SY_A + (dx + dx1) * SY_B)
            + (2 * dx1 / dx) * deltaBZ_AB
        )
    else:
        # explicit E penalty terms
        BC_penalty_E(dt, PARAMS["sigma"], 0.5 * (Ex0 + Ex), 0.5 * (Ey0 + Ey), Bz)
