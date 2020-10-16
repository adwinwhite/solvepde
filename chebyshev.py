import numpy as np
import scipy.special
import cmath
import math


import matplotlib
matplotlib.use("Agg")
import matplotlib.animation as animation
import matplotlib.pyplot as plt

from mpl_toolkits.mplot3d.axes3d import Axes3D
from mpl_toolkits.mplot3d import proj3d
from matplotlib.animation import FuncAnimation
from scipy.sparse.linalg import eigsh
from scipy import sparse
# from sparse_dot_mkl import dot_product_mkl





grid_size = (400, 200)
packet_width_x = 0.05
packet_width_y = 0.1
direction_vector = 100
time_step = 0.08
num_of_frames = 360
max_order_of_chebyshev_poly = 100000
allowed_error = 10**(-13)
mesh_step = 1
display_size_step = 2

free_plane = (0, 0, grid_size[0] * mesh_step, grid_size[1] * mesh_step)
free_plane_length = (free_plane[2] - free_plane[0], free_plane[3] - free_plane[1])
operator_size = (grid_size[0] + 1) * (grid_size[1] + 1)


def get_potential(x, y):
    if x > 0.5 * free_plane[2] and x < 0.55 * free_plane[2] and (y < 0.45 * free_plane[3] or y > 0.55 * free_plane[3]):
        return 6
    else:
        return 0

def initial_wave_unnormalized(x, y):
    if x < free_plane[0] or x > free_plane[2] or y < free_plane[1] or y > free_plane[3]:
        return 0
    return cmath.exp(-(x - 0.25 * free_plane[2])**2/4/(packet_width_x * free_plane[2])**2 -(y - 0.5 * free_plane[3])**2/4/(packet_width_y * free_plane[3])**2 + x*direction_vector*1j)




def get_discretized_init_wave_function():
    results = []
    for j in range(grid_size[1] + 1):
        for i in range(grid_size[0] + 1):
            results.append(initial_wave_unnormalized(i * mesh_step, j * mesh_step))
    return np.array(results)




def flatten_hamiltionian(i, j):
    rowH = np.zeros((grid_size[1] + 1, grid_size[0] + 1))
    rowH[j][i] = -4
    if i + 1 <= grid_size[0]:
        rowH[j][i + 1] = 1
    if i - 1 >= 0:
        rowH[j][i - 1] = 1
    if j + 1 <= grid_size[1]:
        rowH[j + 1][i] = 1
    if j - 1 >= 0:
        rowH[j - 1][i] = 1
    rowH = rowH / mesh_step**2
    rowH[j][i] += get_potential(i * mesh_step, j * mesh_step)
    return sparse.csr_matrix(rowH.flatten())


def get_hamiltonian():
    hamiltonian = []
    for j in range(grid_size[1] + 1):
        for i in range(grid_size[0] + 1):
            hamiltonian.append(flatten_hamiltionian(i, j))
        print(j)
    return sparse.vstack(hamiltonian)


T_tilde_matrices = [None, sparse.identity(operator_size)]
def next_T_tilde_matrix(B):
    if T_tilde_matrices[0] is None:
        T_tilde_matrices[0] = T_tilde_matrices[1]
        T_tilde_matrices[1] = B
        return T_tilde_matrices[1]
    else:
        next_T_tilde = B * 2 * T_tilde_matrices[1] - T_tilde_matrices[0]
        T_tilde_matrices[0] = T_tilde_matrices[1]
        T_tilde_matrices[1] = next_T_tilde
        return next_T_tilde


H = get_hamiltonian()
# using recursion formula for chebyshev polynomial. x's range is R rather than [-1, 1]
def get_evolution_operator_one_timestep():
    print(H.shape)
    eigen_factor = 64
    # print("det(H) : {}".format(scipy.linalg.det(H)))
    # max_eigenvalue = eigsh(H, k=1, which="LA")[0][0]
    # min_eigenvalue = eigsh(H, k=1, which="SA")[0][0]
    max_eigenvalue = 0
    min_eigenvalue = -8
    print("max_eigenvalue : {}".format(max_eigenvalue))
    print("min_eigenvalue : {}".format(min_eigenvalue))
    z = (max_eigenvalue - min_eigenvalue) * time_step / eigen_factor
    B = ((H - sparse.identity(operator_size) * (max_eigenvalue + min_eigenvalue) / 2) / (max_eigenvalue - min_eigenvalue)) * (-1j) * eigen_factor
    # print("det(B) : {}".format(scipy.linalg.det(B)))
    evolution_operator = sparse.csr_matrix((operator_size, operator_size), dtype=np.complex128)
    jv = 1
    i = 1
    while abs(jv) > allowed_error and i <= max_order_of_chebyshev_poly:
        jv = scipy.special.jv(i, z)
        # evolution_operator += jv * next_T_tilde_matrix(B)
        tmpT = jv * next_T_tilde_matrix(B)
        print(i)
        # if sparse.linalg.det(tmpT) == 0:
        #     print("{}".format(i))
        evolution_operator += tmpT
        i += 1
    evolution_operator = (evolution_operator * 2 + sparse.identity(operator_size, dtype=np.complex128) * scipy.special.jv(0, z)) * np.exp((max_eigenvalue + min_eigenvalue) * time_step * (-0.5j))
    print("{} : {}".format(i, abs(jv)))
    # detm = scipy.linalg.det(evolution_operator * evolution_operator.transpose().conj())
    # factor = pow(1 / detm, 1 / operator_size)
    # evolution_operator *= factor
    # print("{}".format(scipy.linalg.det(evolution_operator * evolution_operator.transpose().conj())))
    return sparse.csr_matrix(evolution_operator)


def fake_border(wave):
    for i in range(1, grid_size[0]):
        wave[grid_size[1] * (grid_size[0] + 1) + i] = wave[(grid_size[1] - 1) * (grid_size[0] + 1) + i]
        wave[i] = wave[grid_size[0] + 1 + i]
    for j in range(1, grid_size[1]):
        wave[j * (grid_size[0] + 1)] = wave[j * (grid_size[0] + 1) + 1]
        wave[j * (grid_size[0] + 1) + grid_size[0]] = wave[j * (grid_size[0] + 1) + grid_size[0] - 1]
    return wave

def normalize_wave(wave):
    # integral = sum([abs(x)**2 for x in wave]) * mesh_step**2
    integral = 0
    for j in range(1, grid_size[1]):
        for i in range(1, grid_size[0]):
            integral += abs(wave[j * (grid_size[0] + 1) + i])**2
    integral *= mesh_step**2
    factor = math.sqrt(1/integral)
    return wave * factor

def apply_damping(wave, damping_factor=1.0, border_size=0):
    # factors
    factors = []
    for i in range(border_size):
        # factors.append(1.0 - damping_factor * time_step * (1 - math.sin(math.pi * i / 2 / border_size)))
        factors.append(damping_factor * math.sin(math.pi * i / 2 / border_size))
    # bottom
    for factor_index, j in enumerate(range(border_size)):
        for i in range(grid_size[0] + 1):
            wave[j * (grid_size[0] + 1) + i] *= factors[factor_index]
    # top
    for factor_index, j in enumerate(range(grid_size[1], grid_size[1] - border_size, -1)):
        for i in range(grid_size[0] + 1):
            wave[j * (grid_size[0] + 1) + i] *= factors[factor_index]
    # left
    for factor_index, i in enumerate(range(border_size)):
        for j in range(grid_size[1] + 1):
            wave[j * (grid_size[0] + 1) + i] *= factors[factor_index]
    # right
    for factor_index, i in enumerate(range(grid_size[0], grid_size[0] - border_size, -1)):
        for j in range(grid_size[1] + 1):
            wave[j * (grid_size[0] + 1) + i] *= factors[factor_index]
    return wave


evolution_operator = get_evolution_operator_one_timestep()
current_wave = normalize_wave(get_discretized_init_wave_function())
def propagate_wave(steps=1):
    global current_wave
    # current_wave = evolution_operator.dot(fake_border(current_wave))
    for i in range(steps):
        # current_wave = normalize_wave(evolution_operator.dot(fake_border(current_wave)))
    # current_wave = normalize_wave(apply_damping(evolution_operator.dot(current_wave), damping_factor=0.9, border_size=6))
        current_wave = normalize_wave(evolution_operator.dot(current_wave))
    return current_wave



xs = np.linspace(free_plane[0], free_plane[2], int(grid_size[0] / display_size_step) + 1)
ys = np.linspace(free_plane[1], free_plane[3], int(grid_size[1] / display_size_step) + 1)
xs, ys = np.meshgrid(xs, ys)
# ps = np.array([[get_potential(i * mesh_step, j * mesh_step) / 1000 for i in range(grid_size[0])] for j in range(grid_size[1])])

# draw the figure
def update_plot(frame_number):
    ax.clear()
    ax.set_zlim(0, 0.32 / grid_size[0])
    ax.set_xlim(free_plane[0], free_plane[2])
    ax.set_ylim(free_plane[1], free_plane[3])
    ax.set_xlabel("x")
    ax.set_ylabel("y")
    ax.invert_xaxis()
    # ax.plot_surface(xs, ys, ps, cmap="Dark2")
    # dis = np.reshape([abs(x)**2 for x in propagate_wave(steps=20)], (grid_size[1] + 1, grid_size[0] + 1))
    dis = np.abs(np.reshape(propagate_wave(steps=10), (grid_size[1] + 1, grid_size[0] + 1))[::display_size_step, ::display_size_step])**2
    # dis = np.reshape(propagate_wave(steps=20), (grid_size[1] + 1, grid_size[0] + 1))
    # propagate_wave(steps=20)
    # dis = np.reshape(current_wave, (grid_size[1] + 1, grid_size[0] + 1))[0:-1:grid_size]
    ax.plot_surface(xs, ys, dis, cmap="coolwarm")
    print(frame_number)

Writer = animation.writers['ffmpeg']
writer = Writer(fps=15, metadata=dict(artist='Me'), bitrate=1800)

fig = plt.figure()
ax = fig.add_subplot(111, projection='3d')

x_scale=2
y_scale=1
z_scale=1

scale=np.diag([x_scale, y_scale, z_scale, 1.0])
scale=scale*(1.0/scale.max())
scale[3,3]=1.0

def short_proj():
  return np.dot(Axes3D.get_proj(ax), scale)

ax.get_proj=short_proj

ani = FuncAnimation(fig, update_plot, num_of_frames, interval=1)
ani.save('wave.mp4', writer=writer)

# plt.show()
