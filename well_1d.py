import numpy as np
import scipy.linalg
import scipy.special
import math
import cmath
import matplotlib.pyplot as plt
from matplotlib.animation import FuncAnimation
import time



# free line [0, 1].
# initial wave function is a 1d Gaussian wave packet
free_line = (0, 1)
grid_size = 256
packet_width = 0.04
direction_vector = 0
time_step = 0.000001
num_of_frames = 10000
num_of_partitions_H = 800
max_order_of_chebyshev_poly = 1000000
allowed_error = 0

free_line_length = free_line[1] - free_line[0]
mesh_step = free_line_length / grid_size
mesh_step_reciprocal = grid_size / free_line_length

def initial_wave_unnormalized(x):
    # with width 0.01 and direction vector k=1
    if x < free_line[0] or x > free_line[1]:
        return 0
    return cmath.exp(-(x-0.5)**2/4/packet_width**2 + x*direction_vector*1j)

def get_normalization_factor():
    integral = 0
    for i in range(grid_size):
        integral += abs(initial_wave_unnormalized(i * mesh_step))**2
    integral *= mesh_step
    return math.sqrt(1/integral)

normalization_factor = get_normalization_factor()
def initial_wave_normalized(x):
    return initial_wave_unnormalized(x) * normalization_factor


def get_discretized_init_wave_function():
    results = []
    for i in range(grid_size):
        results.append(initial_wave_normalized(i * mesh_step))
    results = np.array(results)
    return results



# the operator is not unitary.
# def get_evolution_operator(t):
#     exph0 = np.identity(grid_size) * 2
#     for i in range(grid_size - 1):
#         exph0[i][i + 1] = -1
#         exph0[i + 1][i] = -1
#     evolution_operator = scipy.linalg.expm(exph0 * t *(-1j) * mesh_step_reciprocal**2 / num_of_partitions_H)**num_of_partitions_H
#     print(np.linalg.det(evolution_operator * scipy.linalg.expm(exph0 * t * 1j * mesh_step_reciprocal**2)))
#
#     # exph0 = np.identity(grid_size) * (-2j) * t * mesh_step_reciprocal**2
#     # exph0 = scipy.linalg.expm(exph0)
#     #
#     # exph1 = np.zeros((grid_size, grid_size), dtype=np.complex128);
#     # temp2d = np.array([[0, 1], [1, 0]], dtype=np.complex128) * (-1j) * t * mesh_step_reciprocal**2
#     # temp2d = scipy.linalg.expm(temp2d)
#     # for i in range(0, grid_size - 1, 2):
#     #     exph1[np.ix_([i, i + 1], [i, i + 1])] = temp2d
#     #
#     # # zeros here
#     # exph2 = np.zeros((grid_size, grid_size), dtype=np.complex128);
#     # for i in range(1, grid_size - 2, 2):
#     #     exph2[np.ix_([i, i + 1], [i, i + 1])] = temp2d
#     # exph2[0][0] = 1
#     # exph2[grid_size - 1][grid_size - 1] = 1
#     #
#     # evolution_operator = exph0 * exph1 * exph2
#     # print(np.linalg.det(evolution_operator * evolution_operator.conj().transpose()))
#
#     return evolution_operator

T_tilde_matrices = {}
def T_tilde_matrix(order, B):
    if order in T_tilde_matrices:
        return T_tilde_matrices[order]
    result = None
    if order == 0:
        result = np.identity(grid_size)
    elif order == 1:
        result = B * 1j
    else:
        result = B * 2j * T_tilde_matrix(order - 1, B) + T_tilde_matrix(order - 2, B)
    T_tilde_matrices.update({order:result})
    return result


def get_potential(x):
    # if x < 0.4 or x > 0.6:
    #     return 10000
    # else:
    #     return 0
    # return (1-x) * 100000
    # return 100000 / ((x - 0.5)**2 + 0.01)
    return x * 100000

def get_hamiltonian():
    exph0 = np.identity(grid_size) * (-2)
    for i in range(grid_size - 1):
        exph0[i][i + 1] = 1
        exph0[i + 1][i] = 1
    exph0 = exph0 * mesh_step_reciprocal**2
    for i in range(grid_size):
        exph0[i][i] += get_potential(i * mesh_step)
    return exph0

H = get_hamiltonian()
max_entry = np.amax(np.abs(H))
def get_evolution_operator_chebyshev_one_timestep():
    z = -time_step * max_entry
    B = H / max_entry
    # evolution_operator = np.identity(operator_size) * scipy.special.jv(0, z) + 2 * sum([scipy.special.jv(i, z) * T_tilde_matrix(i, B) for i in range(1, order_of_chebyshev_poly)])
    evolution_operator = np.zeros((grid_size, grid_size), dtype=np.complex128)
    jv = 1
    i = 1
    while abs(jv) > allowed_error and i <= max_order_of_chebyshev_poly:
        jv = scipy.special.jv(i, z)
        evolution_operator += jv * T_tilde_matrix(i, B)
        i += 1
    evolution_operator = evolution_operator * 2 + np.identity(grid_size, dtype=np.complex128) * scipy.special.jv(0, z)
    print("{} : {}".format(i, abs(jv)))
    return evolution_operator

evolution_operator = get_evolution_operator_chebyshev_one_timestep()
current_wave = get_discretized_init_wave_function()

def propagate():
    global current_wave
    current_wave = evolution_operator.dot(current_wave)
    current_wave = normalized_wave(current_wave)
    return current_wave

def get_evolution_operator_chebyshev(t):
    z = -t * max_entry
    B = H / max_entry
    # evolution_operator = np.identity(operator_size) * scipy.special.jv(0, z) + 2 * sum([scipy.special.jv(i, z) * T_tilde_matrix(i, B) for i in range(1, order_of_chebyshev_poly)])
    evolution_operator = np.zeros((grid_size, grid_size), dtype=np.complex128)
    jv = 1
    i = 1
    while abs(jv) > allowed_error and i <= max_order_of_chebyshev_poly:
        jv = scipy.special.jv(i, z)
        evolution_operator += jv * T_tilde_matrix(i, B)
        i += 1
    evolution_operator = evolution_operator * 2 + np.identity(grid_size, dtype=np.complex128) * scipy.special.jv(0, z)
    print("{} : {}".format(i, abs(jv)))
    return evolution_operator

def get_discretized_wave_function(t):
    res = get_evolution_operator_chebyshev(t).dot(get_discretized_init_wave_function())
    return res

def next_probability_distribution():
    wave = propagate()
    ps = [abs(x)**2 for x in wave]
    res = [x.real for x in wave]
    ims = [x.imag for x in wave]
    return ps, res, ims

def verify_normalization(dis):
    integral = sum(dis) * mesh_step
    print(integral)
    return integral

def normalized_wave(wave):
    integral = sum([abs(x)**2 for x in wave]) * mesh_step
    factor = math.sqrt(1/integral)
    return wave * factor


# draw the figure
fig = plt.figure()
ax = fig.add_subplot(111)
xs = np.linspace(free_line[0], free_line[1], grid_size)

def update(frame):
    ax.clear()
    prob_dis, real_dis, imag_dis = next_probability_distribution()
    verify_normalization(prob_dis)
    ax.set_xlim(free_line[0], free_line[1])
    ax.set_ylim(0, 10)
    ax.plot(xs, prob_dis, 'g')
    ax.plot(xs, real_dis, 'r')
    ax.plot(xs, imag_dis, 'b')

ani = FuncAnimation(fig, update, frames=num_of_frames, interval=1000/60)
plt.show()
