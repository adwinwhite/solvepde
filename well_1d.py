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
grid_size = 100
packet_width = 0.05
direction_vector = -1
time_step = 0.01
num_of_frames = 10000

free_line_length = free_line[1] - free_line[0]
mesh_step = free_line_length / grid_size

def initial_wave_unnormalized(x):
    # with width 0.01 and direction vector k=1
    if x < free_line[0] or x > free_line[1]:
        return 0
    return cmath.exp(-x**2/4/packet_width**2 + x*direction_vector*1j)

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

def get_evolution_operator(t):
    exph0 = np.identity(grid_size) * t / mesh_step**2 * (-2j)
    exph0 = scipy.linalg.expm(exph0)

    exph1 = np.zeros((grid_size, grid_size));
    for i in range(0, grid_size - 1, 2):
        exph1[i][i + 1] = 1
        exph1[i + 1][i] = 1
    exph1 = exph1 * 1j * t / mesh_step**2
    exph1 = scipy.linalg.expm(exph1)

    exph2 = np.zeros((grid_size, grid_size));
    for i in range(1, grid_size - 1, 2):
        exph2[i][i + 1] = 1
        exph2[i + 1][i] = 1
    exph2 = exph2 * 1j * t / mesh_step**2
    exph2 = scipy.linalg.expm(exph2)

    return exph0 * exph1 * exph2

def get_discretized_wave_function(t):
    return get_evolution_operator(t).dot(get_discretized_init_wave_function())

def get_probability_distribution(t):
    ys = [abs(x)**2 for x in list(get_discretized_wave_function(t))]
    xs = np.linspace(0, 1, grid_size)
    return (xs, ys)

# draw the figure
fig, ax = plt.subplots()
xdata, ydata = [], []
line, = ax.plot([], [], 'o')

def init():
    ax.set_xlim(free_line[0], free_line[1])
    ax.set_ylim(0, 0.5)
    return line,

def update(frame):
    dis = get_probability_distribution(frame)
    line.set_data(dis[0], dis[1])
    return line,

ani = FuncAnimation(fig, update, frames=np.linspace(0, num_of_frames * time_step, num_of_frames),
                    init_func=init, blit=True)
plt.show()
