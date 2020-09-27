import numpy as np
import scipy.special
import cmath
import math


import matplotlib
matplotlib.use("Agg")
import matplotlib.animation as animation
import matplotlib.pyplot as plt
from matplotlib.animation import FuncAnimation




grid_size = (64, 64)
packet_width_x = 0.05
packet_width_y = 0.05
direction_vector = 100000
time_step = 0.00001
mesh_step = 1 / grid_size[0]
num_of_frames = 1000
max_order_of_chebyshev_poly = 100000
allowed_error = 10**(-13)

free_plane = (0, 0, grid_size[0] * mesh_step, grid_size[1] * mesh_step)
free_plane_length = (free_plane[2] - free_plane[0], free_plane[3] - free_plane[1])
mesh_step_reciprocal = grid_size[0] / free_plane_length[0]
operator_size = grid_size[0] * grid_size[1]


def get_potential(x, y):
    # return 10 * ((x-0.5)**2 + (y-0.5)**2 + 0.0001)
    # if (x > 0.5 and x < 0.7) and (y > 0.55 or y < 0.45):
    #     return -10**5
    # else:
    #     return 0
    # if (x > 0.6 * free_plane[2] or x < 0.7 * free_plane[2]) and (y < 0.4 * free_plane[3] or y > 0.6 * free_plane[3]):
    #     return 10000
    # return 0
    # if (x - (free_plane[0] + free_plane_length[0] * 0.5))**2 + (x - (free_plane[1] + free_plane_length[1] * 0.5))**2 > 0.04 and (x - (free_plane[0] + free_plane_length[0] * 0.5))**2 + (x - (free_plane[1] + free_plane_length[1] * 0.5))**2 < 0.09:
    #     return -10000
    # return 0
    # if (x - 0.5)**2 + (y - 0.5)**2 < 0.04:
    #     return 5000
    # else:
    #     return 0
    # return x * 10000
    # if x > 0.5 and y > 0.5:
    #     return -10000
    # return x * 100000
    if (x > 0.5 and x < 0.7) and (y < 0.4 or y > 0.6):
        return 10000
    return 0

def initial_wave_unnormalized(x, y):
    # with width 0.01 and direction vector k=1
    if x < free_plane[0] or x > free_plane[2] or y < free_plane[1] or y > free_plane[3] or get_potential(x, y) >= 10000:
        return 0
    return cmath.exp(-(x - 0.2)**2/4/packet_width_x**2 -(y - 0.5)**2/4/packet_width_y**2 + x*direction_vector*1j)

def get_normalization_factor():
    integral = 0
    for j in range(grid_size[1]):
        for i in range(grid_size[0]):
            integral += abs(initial_wave_unnormalized(i * mesh_step, j * mesh_step))**2
    integral *= mesh_step**2
    return math.sqrt(1/integral)

normalization_factor = get_normalization_factor()
def initial_wave_normalized(x, y):
    return initial_wave_unnormalized(x, y) * normalization_factor


def get_discretized_init_wave_function():
    results = []
    for j in range(grid_size[1]):
        for i in range(grid_size[0]):
            results.append(initial_wave_normalized(i * mesh_step, j * mesh_step))
    return np.array(results)



def flatten_hamiltionian(i, j):
    rowH = np.zeros((grid_size[1], grid_size[0]))
    rowH[j][i] = -4
    if i + 1 < grid_size[0]:
        rowH[j][i + 1] = 1
    if i - 1 >= 0:
        rowH[j][i - 1] = 1
    if j + 1 < grid_size[1]:
        rowH[j + 1][i] = 1
    if j - 1 >= 0:
        rowH[j - 1][i] = 1
    rowH = rowH * mesh_step_reciprocal**2
    rowH[j][i] += get_potential(i * mesh_step, j * mesh_step)
    return rowH.flatten()


def get_hamiltonian():
    hamiltonian = []
    for j in range(grid_size[1]):
        for i in range(grid_size[0]):
            hamiltonian.append(flatten_hamiltionian(i, j))
    return np.array(hamiltonian)


T_tilde_matrices = [None, np.identity(operator_size)]
def next_T_tilde_matrix(B):
    if T_tilde_matrices[0] is None:
        T_tilde_matrices[0] = T_tilde_matrices[1]
        T_tilde_matrices[1] = B * 1j
        return T_tilde_matrices[1]
    else:
        next_T_tilde = B * 2j * T_tilde_matrices[1] + T_tilde_matrices[0]
        T_tilde_matrices[0] = T_tilde_matrices[1]
        T_tilde_matrices[1] = next_T_tilde
        return next_T_tilde


H = get_hamiltonian()
max_entry = np.amax(np.abs(H))
def get_evolution_operator_one_timestep():
    z = -time_step * max_entry
    B = H / max_entry
    # evolution_operator = np.identity(operator_size) * scipy.special.jv(0, z) + 2 * sum([scipy.special.jv(i, z) * T_tilde_matrix(i, B) for i in range(1, order_of_chebyshev_poly)])
    evolution_operator = np.zeros((operator_size, operator_size), dtype=np.complex128)
    jv = 1
    i = 1
    while abs(jv) > allowed_error and i <= max_order_of_chebyshev_poly:
        jv = scipy.special.jv(i, z)
        evolution_operator += jv * next_T_tilde_matrix(B)
        i += 1
    evolution_operator = evolution_operator * 2 + np.identity(operator_size, dtype=np.complex128) * scipy.special.jv(0, z)
    print("{} : {}".format(i, abs(jv)))
    return evolution_operator

def normalized_wave(wave):
    integral = sum([abs(x)**2 for x in wave]) * mesh_step**2
    factor = math.sqrt(1/integral)
    return wave * factor


evolution_operator = get_evolution_operator_one_timestep()
current_wave = get_discretized_init_wave_function()
def propagate_wave():
    global current_wave
    current_wave = normalized_wave(evolution_operator.dot(current_wave))
    return current_wave



xs = np.linspace(free_plane[0], free_plane[2], grid_size[0])
ys = np.linspace(free_plane[1], free_plane[3], grid_size[1])
xs, ys = np.meshgrid(xs, ys)
ps = np.array([[get_potential(i * mesh_step, j * mesh_step) / 1000 for i in range(grid_size[0])] for j in range(grid_size[1])])

# draw the figure
def update_plot(frame_number):
    ax.clear()
    ax.set_zlim(0, 30)
    ax.set_xlim(free_plane[0], free_plane[2])
    ax.set_ylim(free_plane[1], free_plane[3])
    ax.set_xlabel("x")
    ax.set_ylabel("y")
    ax.invert_xaxis()
    # ax.plot_surface(xs, ys, ps, cmap="Dark2")
    dis = np.reshape([abs(x)**2 for x in propagate_wave()], (grid_size[1], grid_size[0]))
    ax.plot_surface(xs, ys, dis, cmap="coolwarm")

Writer = animation.writers['ffmpeg']
writer = Writer(fps=15, metadata=dict(artist='Me'), bitrate=1800)

fig = plt.figure()
ax = fig.add_subplot(111, projection='3d')

ani = FuncAnimation(fig, update_plot, num_of_frames, interval=1000/60)
ani.save('wave.mp4', writer=writer)

plt.show()
