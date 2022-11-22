import matplotlib.pyplot as plt
import matplotlib as mpl
import numpy as np


def fieldplot(ax, field, xlim, ylim, **kwargs):
    xmin, xmax = xlim
    ymin, ymax = ylim
    ax.imshow(field,
              vmin=kwargs.get("vmin", -0.1), vmax=kwargs.get("vmax", 0.1),
              cmap=kwargs.get("cmap", 'RdBu_r'),
              origin='lower', interpolation='nearest',
              extent=[xmin, xmax, ymin, ymax])
    zoom = kwargs.get('zoom', None)
    if zoom is not None:
        center, del_s = zoom
        ax.set(xlim=center[0] + del_s * np.array([-1, 1]),
               ylim=center[1] + del_s * np.array([-1, 1]))
    else:
        ax.set(xlim=[xmin, xmax], ylim=[ymin, ymax])
    ax.set(title=kwargs.get('title', None))


def QLOG(value):
    return np.sign(value) * np.abs(value)**0.25


def newplot(path, index, time_it, **kwargs):
    Ex = kwargs.get('Ex', None)
    xEx_grid = kwargs.get('xEx_grid', None)
    yEx_grid = kwargs.get('yEx_grid', None)
    dx = kwargs.get('dx', None)
    dy = kwargs.get('dy', None)
    x_right = kwargs.get('x_right', None)
    particles = kwargs.get('prtls', None)
    # xmin, xmax = xEx_grid.min(), xEx_grid.max() + dx
    # ymin, ymax = yEx_grid.min(), yEx_grid.max() + dy

    fig = plt.figure(figsize=(12, 12), dpi=200)
    gs = mpl.gridspec.GridSpec(2, 2, width_ratios=[1, 1], height_ratios=[1, 1])
    ax1 = fig.add_subplot(gs[0, 0])
    ax2 = fig.add_subplot(gs[1, 0])
    ax3 = fig.add_subplot(gs[:, -1])

    fieldplot(ax1, QLOG(Ex)[0].swapaxes(1, 0),
              xlim=(xEx_grid[:, 0].min() - dx, xEx_grid[:, 0].max() + dx),
              ylim=(yEx_grid[0].min(), yEx_grid[0].max() + dy),
              title=r"$E_x^A$ (zoom)",
              vmin=-1, vmax=1,
              zoom=((0.98, 0.72), 0.05)
              )
    ax1.axvline(xEx_grid[-1, 0], c='k', lw=0.5)

    fieldplot(ax2, QLOG(Ex)[1].swapaxes(1, 0),
              xlim=(xEx_grid[:, 0].min() + x_right - dx,
                    xEx_grid[:, 0].max() + x_right + dx),
              ylim=(yEx_grid[0].min(), yEx_grid[0].max() + dy),
              title=r"$E_x^B$ (zoom)",
              vmin=-1, vmax=1,
              zoom=((1.02, 0.72), 0.05)
              )
    ax2.axvline(xEx_grid[0, 0] + x_right, c='k', lw=0.5)

    fieldplot(ax3, np.hstack((QLOG(Ex)[0].swapaxes(1, 0)[:, :-1], QLOG(Ex)[1].swapaxes(1, 0)[:, 1:])),
              xlim=(xEx_grid[:, 0].min() - dx,
                    xEx_grid[:, 0].max() + x_right + dx),
              ylim=(yEx_grid[0].min(), yEx_grid[0].max() + dy),
              title=r"$E_x$ (at i+1/2 positions)",
              vmin=-1, vmax=1,
              zoom=((1.0, 0.72), 0.1)
              )
    ax3.axvline(xEx_grid[-1, 0], c='k', lw=0.5)

    if particles is not None:
        tags, xs, ys, wei, nprtl = particles
        for ip in range(nprtl):
            if (tags[time_it, ip] == 0):
                ax1.scatter(xs[time_it, ip], ys[time_it, ip],
                            s=10, c=f"C{int(wei[time_it, ip] * 0.5 + 0.5)}")
            elif (tags[time_it, ip] == 1):
                ax2.scatter(xs[time_it, ip] + x_right, ys[time_it, ip],
                            s=10, c=f"C{int(wei[time_it, ip] * 0.5 + 0.5)}")

    # center = (1.0, 0.72)
    # del_s = 0.05
    # ax.set(xlim=center[0] + del_s * np.array([-1, 1]),
    #        ylim=center[1] + del_s * np.array([-1, 1]),
    #        title=r'$E_x$ (zoom)')

    fig.savefig(f"{path}/fields_{index}.png", bbox_inches='tight', dpi=200)
    plt.close()


# from matplotlib.gridspec import GridSpec
# import numpy as N
# import matplotlib.pyplot as P

# from cubed_sphere import *

# # VIS
# ratio = 0.5


# # Saves figure
# def figsave_png(fig, filename, res=200):
#     fig.savefig('{}.png'.format(filename),
#                 bbox_inches='tight', dpi=res)


# def plot_fields(path, idump, it):
#     fig = P.figure(1, facecolor='w', figsize=(12, 6), dpi=200)
#     gs = GridSpec(2, 2, figure=fig, width_ratios=[
#                   1.75, 1], wspace=0.05, hspace=0.3)

#     ax = fig.add_subplot(gs[0, 0])

#     xmin, xmax = xEx_grid.min(), xEx_grid.max() + dx
#     ymin, ymax = yEx_grid.min(), yEx_grid.max() + dy


#     # print (xmin, xmax, ymin, ymax)
#     # P.pcolormesh(xEx_grid, yEx_grid,
#     #              Ex[0, :, :], vmin=-0.1, vmax=0.1, cmap='RdBu_r')
#     for i in range(Ex[0].shape[0]):
#         ax.axhline(i * dx, lw=0.25, color='k', zorder=5)
#         ax.axvline(i * dx, lw=0.25, color='k', zorder=5)

#     ax.imshow(Ex[0].swapaxes(1, 0), vmin=-0.1, vmax=0.1,
#               cmap='RdBu_r', origin='lower', interpolation='nearest',
#               extent=[ymin, ymax, xmin, xmax])
#     ax.imshow(Ex[1].swapaxes(1, 0), vmin=-0.1, vmax=0.1,
#               cmap='RdBu_r', origin='lower', interpolation='nearest',
#               extent=[xmin + x_max, xmax + x_max, ymin, ymax])

#     for ip in range(np):
#         if (tag[it, ip] == 0):
#             P.scatter(xp[it, ip], yp[it, ip], s=10)
#         elif (tag[it, ip] == 1):
#             P.scatter(xp[it, ip] + x_max - x_min + 2.0 * dx, yp[it, ip], s=10)

#     center = (1.0, 0.5)
#     del_s = 0.05
#     ax.set(xlim=center[0] + del_s * N.array([-1, 1]),
#            ylim=center[1] + del_s * N.array([-1, 1]), title=r'$E_x$')
#     # ax.set(ylim=(ymin, ymax), xlim=(xmin, xmax + x_max), title=r'$E_x$')

#     ax.axvline(1., color='k')

#     ax.axvline(x0l, color=[0, 0.4, 0.4], lw=1.0, ls='--')
#     ax.axvline(x0r, color=[0, 0.4, 0.4], lw=1.0, ls='--')
#     ax.axvline(x1l + x_max, color=[0.8, 0.0, 0.0], lw=1.0, ls='--')
#     ax.axvline(x1r + x_max, color=[0.8, 0.0, 0.0], lw=1.0, ls='--')

#     ax = fig.add_subplot(gs[1, 0])

#     ax.imshow(N.rot90(Ey[0])[::-1], vmin=-0.1, vmax=0.1,
#               cmap='RdBu_r', origin='lower', interpolation='nearest',
#               extent=[xmin, xmax, ymin, ymax])
#     ax.imshow(N.rot90(Ey[1])[::-1], vmin=-0.1, vmax=0.1,
#               cmap='RdBu_r', origin='lower', interpolation='nearest',
#               extent=[xmin + x_max, xmax + x_max, ymin, ymax])

#     for ip in range(np):
#         if (tag[it, ip] == 0):
#             ax.scatter(xp[it, ip], yp[it, ip], s=10)
#         elif (tag[it, ip] == 1):
#             ax.scatter(xp[it, ip] + x_max - x_min + 2.0 * dx, yp[it, ip], s=10)

#     ax.set(ylim=(y_min, y_max), xlim=(0.0, 2.0*x_max), title=r'$E_y$')

#     ax.plot([1.0, 1.0], [0, 1.0], color='k')

#     ax = fig.add_subplot(gs[:, -1])

#     ax.scatter(time[it-1], flux0[it-1] / (4.0 * N.pi * q),
#                s=10, color=[0, 0.4, 0.4])
#     ax.scatter(time[it-1], flux1[it-1] / (4.0 * N.pi * q),
#                s=10, color=[0.8, 0.0, 0.0])
#     ax.plot(time[:it], flux0[:it] / (4.0 * N.pi * q),
#             color=[0, 0.4, 0.4], ls='-')
#     ax.plot(time[:it], flux1[:it] / (4.0 * N.pi * q),
#             color=[0.8, 0.0, 0.0], ls='-')

#     ax.axhline(1.0, color='k', lw=1.0, ls='--')
#     ax.axhline(-1.0, color='k', lw=1.0, ls='--')

#     ax.set(xlim=(0.0, Nt * dt), ylim=(-1.1, 1.1), xlabel=r'$t$', ylabel=r'$Q$')
#     figsave_png(fig, f"{path}/fields_{idump}")
#     P.close(fig)


# vm = 1.0


# def plot_fields_zoom(path, idump, it):

#     fig = P.figure(1, facecolor='w', figsize=(8, 4), dpi=200)
#     gs = GridSpec(2, 1, figure=fig)

#     ax = fig.add_subplot(gs[0, 0])

#     P.pcolormesh(xEx_grid, yEx_grid, Jx[0, :, :],
#                  vmin=-1.0, vmax=1.0, cmap='RdBu_r')
#     P.pcolormesh(xEx_grid + x_max + 2.0 * dx, yEx_grid,
#                  Jx[1, :, :], vmin=-1.0, vmax=1.0, cmap='RdBu_r')

#     for ip in range(np):
#         if (tag[it, ip] == 0):
#             P.scatter(xp[it, ip], yp[it, ip], s=20, color='k')
#         elif (tag[it, ip] == 1):
#             P.scatter(xp[it, ip] + x_max - x_min + 2.0 *
#                       dx, yp[it, ip], s=20, color='k')

#     P.title(r'$J_x$')

#     P.ylim((0.4, 0.6))
#     P.xlim((0.8, 1.2))
#     ax.set_aspect(1.0/ax.get_data_ratio()*ratio)

#     P.plot([1.0, 1.0], [0, 1.0], color='k')

#     P.plot([x0l, x0l], [0, 1.0], color=[0, 0.4, 0.4],   lw=1.0, ls='--')
#     P.plot([x0r, x0r], [0, 1.0], color=[0, 0.4, 0.4],   lw=1.0, ls='--')
#     P.plot([x1l + x_max, x1l + x_max], [0, 1.0],
#            color=[0.8, 0.0, 0.0], lw=1.0, ls='--')
#     P.plot([x1r + x_max, x1r + x_max], [0, 1.0],
#            color=[0.8, 0.0, 0.0], lw=1.0, ls='--')

#     ax = fig.add_subplot(gs[1, 0])

#     P.pcolormesh(xEx_grid, yEx_grid, Ex[0, :, :],
#                  vmin=-vm, vmax=vm, cmap='RdBu_r')
#     P.pcolormesh(xEx_grid + x_max + 2.0 * dx, yEx_grid,
#                  Ex[1, :, :], vmin=-vm, vmax=vm, cmap='RdBu_r')
#     # P.pcolormesh(xEy_grid, yEy_grid, Ey[0, :, :], vmin = -vm, vmax = vm, cmap = 'RdBu_r')
#     # P.pcolormesh(xEy_grid + x_max + 2.0 * dx, yEy_grid, Ey[1, :, :], vmin = -vm, vmax = vm, cmap = 'RdBu_r')

#     for ip in range(np):
#         if (tag[it, ip] == 0):
#             P.scatter(xp[it, ip], yp[it, ip], s=20, color='k')
#         elif (tag[it, ip] == 1):
#             P.scatter(xp[it, ip] + x_max - x_min + 2.0 *
#                       dx, yp[it, ip], s=20, color='k')

#     P.title(r'$E_y$')

#     P.ylim((0.4, 0.6))
#     P.xlim((0.8, 1.2))
#     ax.set_aspect(1.0/ax.get_data_ratio()*ratio)

#     P.plot([1.0, 1.0], [0, 1.0], color='k')

#     figsave_png(fig, f"{path}/fields_zoom_{idump}")

#     P.close('all')


# def plot_div(idump, it):

#     fig = P.figure(2, facecolor='w', figsize=(30, 10))
#     ax = P.subplot(211)

#     P.pcolormesh(xEz_grid, yEz_grid, q * N.abs(
#         divE1[0, :, :] - 4.0 * N.pi * rho1[0, :, :]), vmin=-0.1, vmax=0.1, cmap='RdBu_r')
#     P.pcolormesh(xEz_grid + x_max + 2.0 * dx, yEz_grid, q * N.abs(
#         divE0[1, :, :] - 4.0 * N.pi * rho1[1, :, :]), vmin=-0.1, vmax=0.1, cmap='RdBu_r')

#     P.ylim((0.4, 0.6))
#     P.xlim((0.95, 1.05))
#     # ax.set_aspect(1.0/ax.get_data_ratio()*ratio)

#     ax = P.subplot(212)

#     P.pcolormesh(xEz_grid, yEz_grid, q *
#                  N.abs(divcharge[0, :, :]), vmin=-0.1, vmax=0.1, cmap='RdBu_r')
#     P.pcolormesh(xEz_grid + x_max + 2.0 * dx, yEz_grid,
#                  N.abs(divcharge[0, :, :]), vmin=-0.1, vmax=0.1, cmap='RdBu_r')

#     P.ylim((0.4, 0.6))
#     P.xlim((0.95, 1.05))
#     # ax.set_aspect(1.0/ax.get_data_ratio()*ratio)

#     figsave_png(fig, "snapshots_penalty/div_" + str(idump))

#     P.close('all')
