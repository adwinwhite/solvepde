import numpy as np
import scipy.linalg
import math
import cmath
import matplotlib.pyplot as plt
from matplotlib.animation import FuncAnimation
import time



# free line [0, 1].
# initial wave function is a 1d Gaussian wave packet
free_line = (0, 1)
grid_size = 10
packet_width = 0.1
direction_vector = 10000
time_step = 0.0001
num_of_frames = 100
num_of_partitions_H = 800

free_line_length = free_line[1] - free_line[0]
mesh_step = free_line_length / grid_size
mesh_step_reciprocal = grid_size / free_line_length

def initial_wave_unnormalized(x):
    # with width 0.01 and direction vector k=1
    if x < free_line[0] or x > free_line[1]:
        return 0
    return cmath.exp(-(x-0.5)**2/4/packet_width**2 + x*direction_vector*1j)
    # return cmath.exp(-(x-0.5)**2/4/packet_width**2)

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
    results = np.transpose(np.array(results))
    # print([abs(x)**2 for x in list(results)])
    return results

# the operator is not unitary.
def get_evolution_operator(t):
    exph0 = np.identity(grid_size) * 2
    for i in range(grid_size - 1):
        exph0[i][i + 1] = -1
        exph0[i + 1][i] = -1
    evolution_operator = scipy.linalg.expm(exph0 * t *(-1j) * mesh_step_reciprocal**2 / num_of_partitions_H)**num_of_partitions_H
    print(np.linalg.det(evolution_operator * scipy.linalg.expm(exph0 * t * 1j * mesh_step_reciprocal**2)))

    # exph0 = np.identity(grid_size) * (-2j) * t * mesh_step_reciprocal**2
    # exph0 = scipy.linalg.expm(exph0)
    #
    # exph1 = np.zeros((grid_size, grid_size), dtype=np.complex128);
    # temp2d = np.array([[0, 1], [1, 0]], dtype=np.complex128) * (-1j) * t * mesh_step_reciprocal**2
    # temp2d = scipy.linalg.expm(temp2d)
    # for i in range(0, grid_size - 1, 2):
    #     exph1[np.ix_([i, i + 1], [i, i + 1])] = temp2d
    #
    # # zeros here
    # exph2 = np.zeros((grid_size, grid_size), dtype=np.complex128);
    # for i in range(1, grid_size - 2, 2):
    #     exph2[np.ix_([i, i + 1], [i, i + 1])] = temp2d
    # exph2[0][0] = 1
    # exph2[grid_size - 1][grid_size - 1] = 1
    #
    # evolution_operator = exph0 * exph1 * exph2
    # print(np.linalg.det(evolution_operator * evolution_operator.conj().transpose()))

    return evolution_operator

def get_discretized_wave_function(t):
    return get_evolution_operator(t).dot(get_discretized_init_wave_function())

def get_probability_distribution(t):
    ys = [abs(x)**2 for x in list(get_discretized_wave_function(t))]
    # print(sum(ys))
    xs = np.linspace(0, 1, grid_size)
    return (xs, ys)

for i in range(num_of_frames):
    get_probability_distribution(i * time_step)

# draw the figure
# fig, ax = plt.subplots()
# xdata, ydata = [], []
# line, = ax.plot([], [], 'o')
#
# def init():
#     ax.set_xlim(free_line[0], free_line[1])
#     ax.set_ylim(0, 5)
#     return line,
#
# def update(frame):
#     dis = get_probability_distribution(frame * time_step)
#     line.set_data(dis[0], dis[1])
#     return line,
#
# ani = FuncAnimation(fig, update, frames=num_of_frames,
#                     init_func=init, blit=True, repeat=False, interval=0.0001)
# plt.show()
