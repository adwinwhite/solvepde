import numpy as np
import scipy.special
import cmath
import math


import matplotlib
# matplotlib.use("Agg")
import matplotlib.animation as animation
import matplotlib.pyplot as plt
from matplotlib.animation import FuncAnimation
from scipy.sparse.linalg import eigsh




grid_size = (100,)
packet_width_x = 5
direction_vector = 100
time_step = 0.01
free_line = (0, 100)
num_of_frames = 1000000
max_order_of_chebyshev_poly = 100000
allowed_error = 10**(-18)

free_line_length = (free_line[1] - free_line[0])
mesh_step = free_line_length / grid_size[0]

operator_size = (grid_size[0] + 1)


def get_potential(x):
    # return 10 * ((x-0.5)**2 + (y-0.5)**2 + 0.0001)
    # if (x > 0.5 and x < 0.7) and (y > 0.55 or y < 0.45):
    #     return -10**5
    # else:
    #     return 0
    # if (x > 0.6 * free_line[2] or x < 0.7 * free_line[2]) and (y < 0.4 * free_line[3] or y > 0.6 * free_line[3]):
    #     return 10000
    # return 0
    # if (x - (free_line[0] + free_line_length[0] * 0.5))**2 + (x - (free_line[1] + free_line_length[1] * 0.5))**2 > 0.04 and (x - (free_line[0] + free_line_length[0] * 0.5))**2 + (x - (free_line[1] + free_line_length[1] * 0.5))**2 < 0.09:
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
    # if (x > 0.5 and x < 0.7) and (y < 0.4 or y > 0.6):
    #     return 10000
    # return 0
    if x > 70:
        return 10
    else:
        return 0

def initial_wave_unnormalized(x):
    # with width 0.01 and direction vector k=1
    if x < free_line[0] or x > free_line[1]:
        return 0
    return cmath.exp(-(x - 30)**2/4/packet_width_x**2 + x*direction_vector*1j)




def get_discretized_init_wave_function():
    results = []
    for i in range(grid_size[0] + 1):
        results.append(initial_wave_unnormalized(i * mesh_step))
    return np.array(results)



def get_hamiltonian():
    hamiltonian = np.zeros((operator_size, operator_size))
    for i in range(operator_size):
        if i - 1 >= 0:
            hamiltonian[i][i - 1] = 1
        if i + 1 < operator_size:
            hamiltonian[i][i + 1] = 1
        hamiltonian[i][i] = -2
    hamiltonian /= mesh_step**2
    for i in range(operator_size):
        hamiltonian[i][i] += get_potential(i * mesh_step)
    return hamiltonian


T_tilde_matrices = [None, np.identity(operator_size)]
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
# max_entry = np.amax(np.abs(H))
def get_evolution_operator_one_timestep():
    max_eigenvalue = eigsh(H, k=1, which="LA")[0][0]
    min_eigenvalue = eigsh(H, k=1, which="SA")[0][0]
    z = (max_eigenvalue - min_eigenvalue) * time_step / 2
    B = ((H * 2  - np.identity(operator_size) * (max_eigenvalue + min_eigenvalue)) / (max_eigenvalue - min_eigenvalue)) * (-1j)
    evolution_operator = np.zeros((operator_size, operator_size), dtype=np.complex128)
    jv = 1
    i = 1
    while abs(jv) > allowed_error and i <= max_order_of_chebyshev_poly:
        jv = scipy.special.jv(i, z)
        evolution_operator += jv * next_T_tilde_matrix(B)
        i += 1
    evolution_operator = (evolution_operator * 2 + np.identity(operator_size, dtype=np.complex128) * scipy.special.jv(0, z)) * np.exp((max_eigenvalue + min_eigenvalue) * time_step / 2 * (-1j))
    print("{} : {}".format(i, abs(jv)))
    # detm = scipy.linalg.det(evolution_operator * evolution_operator.transpose().conj())
    # factor = pow(1 / detm, 1 / operator_size)
    # evolution_operator *= factor
    print("{}".format(scipy.linalg.det(evolution_operator * evolution_operator.transpose().conj())))
    return evolution_operator


def fake_border(wave):
    wave[0] = wave[1]
    wave[operator_size - 1] = wave[operator_size - 2]
    return wave

def normalize_wave(wave):
    # integral = sum([abs(x)**2 for x in wave]) * mesh_step**2
    integral = 0
    for i in range(1, grid_size[0]):
        integral += abs(wave[i])**2
    integral *= mesh_step
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
        current_wave = normalize_wave(evolution_operator.dot(fake_border(current_wave)))
    # current_wave = normalize_wave(apply_damping(evolution_operator.dot(current_wave), damping_factor=0.9, border_size=6))
    # current_wave = evolution_operator.dot(current_wave)
    # current_wave = normalize_wave(evolution_operator.dot(current_wave))
    return current_wave



xs = np.linspace(free_line[0], free_line[1], grid_size[0] + 1)
# ps = np.array([[get_potential(i * mesh_step, j * mesh_step) / 1000 for i in range(grid_size[0])] for j in range(grid_size[1])])

# draw the figure
def update_plot(frame_number):
    ax.clear()
    ax.set_xlim(free_line[0], free_line[1])
    ax.set_ylim(0, 0.2)
    ax.set_xlabel("x")
    ax.set_ylabel("y")
    propagate_wave(steps=10)
    ps = [abs(x)**2 for x in current_wave]
    # res = [x.real for x in current_wave]
    # ims = [x.imag for x in current_wave]
    ax.plot(xs, ps)
    # ax.plot(xs, res)
    # ax.plot(xs, ims)


fig = plt.figure()
ax = fig.add_subplot(111)

ani = FuncAnimation(fig, update_plot, num_of_frames, interval=10)
# ani.save('wave.mp4', writer=writer)

plt.show()
