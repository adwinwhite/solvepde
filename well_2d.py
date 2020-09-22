import numpy as np
import scipy.special
import cmath
import math

import matplotlib.pyplot as plt
from matplotlib.animation import FuncAnimation



free_plane = (0, 0, 1, 1)
grid_size = (20, 20)
packet_width_x = 0.1
packet_width_y = 0.2
direction_vector = 10000
time_step = 0.1
num_of_frames = 100
num_of_partitions_H = 800
order_of_chebyshev_poly = 64

free_plane_length = (free_plane[2] - free_plane[0], free_plane[3] - free_plane[1])
mesh_step = free_plane_length[0] / grid_size[0]
mesh_step_reciprocal = grid_size[0] / free_plane_length[0]
operator_size = grid_size[0] * grid_size[1]

def initial_wave_unnormalized(x, y):
    # with width 0.01 and direction vector k=1
    if x < free_plane[0] or x > free_plane[2] or y < free_plane[1] or y > free_plane[3]:
        return 0
    return cmath.exp(-(x-0.5)**2/4/packet_width_x**2 -(y - 0.5)**2/4/packet_width_y**2 + x*direction_vector*1j)

def get_normalization_factor():
    integral = 0
    for i in range(grid_size[0]):
        for j in range(grid_size[1]):
            integral += abs(initial_wave_unnormalized(i * mesh_step, j * mesh_step))**2
    integral *= mesh_step
    return math.sqrt(1/integral)

normalization_factor = get_normalization_factor()
def initial_wave_normalized(x, y):
    return initial_wave_unnormalized(x, y) * normalization_factor


def get_discretized_init_wave_function():
    results = []
    for i in range(grid_size[0]):
        for j in range(grid_size[1]):
            results.append(initial_wave_normalized(i * mesh_step, j * mesh_step))
    # results = np.transpose(np.array(results))
    # print([abs(x)**2 for x in list(results)])
    return np.array(results)

def flatten_hamiltionian(i, j):
    rowH = np.zeros((grid_size[0], grid_size[1]))
    rowH[i][j] = -4
    if i + 1 < grid_size[0]:
        rowH[i + 1][j] = 1
    if i - 1 >= 0:
        rowH[i - 1][j] = 1
    if j + 1 < grid_size[1]:
        rowH[i][j + 1] = 1
    if j - 1 >= 0:
        rowH[i][j - 1] = 1
    return rowH.flatten()


def get_hamiltonian():
    hamiltonian = []
    for i in range(grid_size[0]):
        for j in range(grid_size[1]):
            hamiltonian.append(flatten_hamiltionian(i, j))
    return np.array(hamiltonian)

T_tilde_matrices = {}
def T_tilde_matrix(order, B):
    if order in T_tilde_matrices:
        return T_tilde_matrices[order]
    result = None
    if order == 0:
        result = np.identity(operator_size)
    elif order == 1:
        result = B * 1j
    else:
        result = B * 2j * T_tilde_matrix(order - 1, B) + T_tilde_matrix(order - 2, B)
    T_tilde_matrices.update({order:result})
    return result

def get_evolution_operator(t):
    H = get_hamiltonian()
    max_entry = np.amax(H) * 16
    z = -t * max_entry
    B = H / max_entry
    evolution_operator = np.identity(operator_size) * scipy.special.jv(0, z) + 2 * sum([scipy.special.jv(i, z) * T_tilde_matrix(i, B) for i in range(1, order_of_chebyshev_poly)])
    return evolution_operator

def verify_evolution_operator(U):
    print(np.linalg.det(U * U.conj().transpose()))

def get_wave_distribution(t):
    U = get_evolution_operator(t)
    verify_evolution_operator(U)
    return np.reshape(U.dot(get_discretized_init_wave_function()), (grid_size[0], grid_size[1]))

def get_probability_distribution(t):
    W = get_wave_distribution(t)
    return np.abs(W)**2


xs = np.linspace(free_plane[0], free_plane[2], grid_size[0])
ys = np.linspace(free_plane[1], free_plane[3], grid_size[1])
xs, ys = np.meshgrid(xs, ys)

# draw the figure
def update_plot(frame_number):
    ax.plot_surface(xs, ys, get_probability_distribution(frame_number * time_step), cmap="coolwarm")

fig = plt.figure()
ax = fig.add_subplot(111, projection='3d')

# plot = [ax.plot_surface(xs, ys, get_probability_distribution(0), color='0.75', rstride=1, cstride=1)]
ax.plot_surface(xs, ys, get_probability_distribution(0), color='0.75', rstride=1, cstride=1, cmap="coolwarm")
ani = FuncAnimation(fig, update_plot, num_of_frames, interval=1000/60)

plt.show()
