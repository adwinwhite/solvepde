import numpy as np
import scipy.special
import cmath
import math


import matplotlib
# matplotlib.use("Agg")
import matplotlib.animation as animation
import matplotlib.pyplot as plt

from mpl_toolkits.mplot3d.axes3d import Axes3D
from mpl_toolkits.mplot3d import proj3d
from matplotlib.animation import FuncAnimation
from scipy.sparse.linalg import eigsh
from scipy import sparse
# from sparse_dot_mkl import dot_product_mkl





grid_size = (100, 100)
light_speed = 1
electric_constant = 1
magnetic_constant = 1
mesh_step = 1
num_of_frames = 10000
time_step = 0.1
electric_direction = np.array((0, 1, 0))
k = np.array((10, 0, 0))
E_0 = 10
allowed_error = 10**(-13)
max_order_of_chebyshev_poly = 100000
display_size_step = 2


free_plane = np.array([-grid_size[0] / 2, -grid_size[1] / 2, grid_size[0] / 2, grid_size[1] /2]) * mesh_step
free_plane_length = (grid_size[0] * mesh_step, grid_size[1] * mesh_step)
packet_center = (0, 0)
packet_size = (0.1 * free_plane_length[0], 0.1 * free_plane_length[1], 0.1 * free_plane_length[1])
operator_size = (grid_size[0] + 1) * (grid_size[1] + 1) * 3

electric_coeff_sqrt = math.sqrt(electric_constant)
magnetic_coeff_sqrt = math.sqrt(magnetic_constant)


electric_direction = electric_direction / np.linalg.norm(electric_direction)
k_direction = k / np.linalg.norm(k)
# print(electric_direction)
# print(k_direction)

def E(x, y, z):
    return np.array([-electric_direction[1] * y / 2 / packet_size[1]**2 - z / 2 / packet_size[2], -electric_direction[1] * (-x / 2 / packet_size[0]**2 + k[0] * 1j), -electric_direction[2] * (-x / 2 / packet_size[0]**2 + k[0] * 1j)]) * 1j * E_0 * math.pi**(3/2) / (packet_size[0] * packet_size[1] * packet_size[2]) * cmath.exp(1j * k[0] * x) * cmath.exp(-(x/2/packet_size[0])**2-(y/2/packet_size[1])**2-(z/2/packet_size[2])**2)

def B(x, y, z):
    return np.cross(k_direction, E(x, y, z)) / light_speed

def energy_density(t):
    xs = np.linspace(free_plane[0], free_plane[2], grid_size[0] + 1)
    ys = np.linspace(free_plane[1], free_plane[3], grid_size[1] + 1)
    # es = [[((electric_constant * E_z(x, y, t)**2 + magnetic_constant * (H_x(x, y, t)**2 + H_y((x, y, t)**2))) / 2) for x in xs] for y in ys]
    es = []
    for y in ys:
        row = []
        for x in xs:
            # curr_E = [x.real for x in E(x, y, 0)]
            # curr_B = [x.real for x in B(x, y, 0)]
            curr_E = E(x, y, 0)
            curr_B = np.cross(k_direction, curr_E) / light_speed
            curr_E = [x.real for x in curr_E]
            curr_B = [x.real for x in curr_B]
            row.append((electric_constant * (curr_E[0]**2 + curr_E[1]**2)+ curr_B[2]**2 / magnetic_constant) / 2)
        es.append(row)
    return np.array(es)


# def initial_wave_unnormalized(x, y):
#     if x < free_plane[0] or x > free_plane[2] or y < free_plane[1] or y > free_plane[3]:
#         return 0
#     return cmath.exp(-(x - 0.25 * free_plane[2])**2/4/(packet_width_x * free_plane[2])**2 -(y - 0.5 * free_plane[3])**2/4/(packet_width_y * free_plane[3])**2 + x*direction_vector*1j)
#
#
#
#
def get_discretized_init_wave_function():
    xs = np.linspace(free_plane[0], free_plane[2], grid_size[0] + 1)
    ys = np.linspace(free_plane[1], free_plane[3], grid_size[1] + 1)
    es = np.empty((operator_size))
    i = 0
    for y in ys:
        for x in xs:
            curr_E = E(x, y, 0)
            curr_B = np.cross(k_direction, curr_E) / light_speed
            curr_Y = [x.real * electric_coeff_sqrt for x in curr_E]
            curr_X = [x.real / magnetic_coeff_sqrt for x in curr_B]
            es[i * 3] = curr_Y[0]
            es[i * 3 + 1] = curr_Y[1]
            es[i * 3 + 2] = curr_X[2]
            i += 1
    return es




def flatten_hamiltionian(i, j):
    rows = [np.zeros((grid_size[1] + 1, grid_size[0] + 1, 3)) for m in range(3)]
    if j + 1 <= grid_size[1]:
        rows[0][j + 1][i][2] = 1
    if j - 1 >= 0:
        rows[0][j - 1][i][2] = -1
    rows[0] /= electric_coeff_sqrt * magnetic_coeff_sqrt * 2 * mesh_step

    if i + 1 <= grid_size[0]:
        rows[1][j][i + 1][2] = 1
    if i - 1 >= 0:
        rows[1][j][i - 1][2] = -1
    rows[1] /= electric_coeff_sqrt * magnetic_coeff_sqrt * 2 * mesh_step

    if j + 1 <= grid_size[1]:
        rows[2][j + 1][i][0] = 1
    if j - 1 >= 0:
        rows[2][j - 1][i][0] = -1
    if i + 1 <= grid_size[0]:
        rows[2][j][i + 1][1] = -1
    if i - 1 >= 0:
        rows[2][j][i - 1][1] = 1
    rows[2] /= electric_coeff_sqrt * magnetic_coeff_sqrt * 2 * mesh_step

    return sparse.vstack([sparse.csr_matrix(row.flatten()) for row in rows])



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
        tmpT = jv * next_T_tilde_matrix(B) * (1j)**i
        print(i)
        # if sparse.linalg.det(tmpT) == 0:
        #     print("{}".format(i))
        evolution_operator += tmpT
        i += 1
    evolution_operator = (evolution_operator * 2 + sparse.identity(operator_size, dtype=np.complex128) * scipy.special.jv(0, z)) * np.exp((max_eigenvalue + min_eigenvalue) * time_step * 0.5)
    print("{} : {}".format(i, abs(jv)))
    # detm = scipy.linalg.det(evolution_operator * evolution_operator.transpose().conj())
    # factor = pow(1 / detm, 1 / operator_size)
    # evolution_operator *= factor
    # print("{}".format(scipy.linalg.det(evolution_operator * evolution_operator.transpose().conj())))
    return sparse.csr_matrix(evolution_operator)
#
#
# def fake_border(wave):
#     for i in range(1, grid_size[0]):
#         wave[grid_size[1] * (grid_size[0] + 1) + i] = wave[(grid_size[1] - 1) * (grid_size[0] + 1) + i]
#         wave[i] = wave[grid_size[0] + 1 + i]
#     for j in range(1, grid_size[1]):
#         wave[j * (grid_size[0] + 1)] = wave[j * (grid_size[0] + 1) + 1]
#         wave[j * (grid_size[0] + 1) + grid_size[0]] = wave[j * (grid_size[0] + 1) + grid_size[0] - 1]
#     return wave
#
# def normalize_wave(wave):
#     # integral = sum([abs(x)**2 for x in wave]) * mesh_step**2
#     integral = 0
#     for j in range(1, grid_size[1]):
#         for i in range(1, grid_size[0]):
#             integral += abs(wave[j * (grid_size[0] + 1) + i])**2
#     integral *= mesh_step**2
#     factor = math.sqrt(1/integral)
#     return wave * factor
#
# def apply_damping(wave, damping_factor=1.0, border_size=0):
#     # factors
#     factors = []
#     for i in range(border_size):
#         # factors.append(1.0 - damping_factor * time_step * (1 - math.sin(math.pi * i / 2 / border_size)))
#         factors.append(damping_factor * math.sin(math.pi * i / 2 / border_size))
#     # bottom
#     for factor_index, j in enumerate(range(border_size)):
#         for i in range(grid_size[0] + 1):
#             wave[j * (grid_size[0] + 1) + i] *= factors[factor_index]
#     # top
#     for factor_index, j in enumerate(range(grid_size[1], grid_size[1] - border_size, -1)):
#         for i in range(grid_size[0] + 1):
#             wave[j * (grid_size[0] + 1) + i] *= factors[factor_index]
#     # left
#     for factor_index, i in enumerate(range(border_size)):
#         for j in range(grid_size[1] + 1):
#             wave[j * (grid_size[0] + 1) + i] *= factors[factor_index]
#     # right
#     for factor_index, i in enumerate(range(grid_size[0], grid_size[0] - border_size, -1)):
#         for j in range(grid_size[1] + 1):
#             wave[j * (grid_size[0] + 1) + i] *= factors[factor_index]
#     return wave
#
#
evolution_operator = get_evolution_operator_one_timestep()
current_wave = get_discretized_init_wave_function()
def propagate_wave(steps=1):
    global current_wave
    # current_wave = evolution_operator.dot(fake_border(current_wave))
    for i in range(steps):
        # current_wave = normalize_wave(evolution_operator.dot(fake_border(current_wave)))
    # current_wave = normalize_wave(apply_damping(evolution_operator.dot(current_wave), damping_factor=0.9, border_size=6))
        current_wave = evolution_operator.dot(current_wave)
    return current_wave

def wave2energe(wave):
    unflattened = np.reshape(wave, (grid_size[1] + 1, grid_size[0] + 1, 3))[::display_size_step, ::display_size_step, ::]
    # print(unflattened.shape)
    dis = np.array([[np.linalg.norm(v)**2 / 2 for v in unflattened[j]] for j in range(int(grid_size[1] / display_size_step) + 1)])
    return dis



xs = np.linspace(free_plane[0], free_plane[2], int(grid_size[0] / display_size_step) + 1)
ys = np.linspace(free_plane[1], free_plane[3], int(grid_size[1] / display_size_step) + 1)
xs, ys = np.meshgrid(xs, ys)
# ps = np.array([[get_potential(i * mesh_step, j * mesh_step) / 1000 for i in range(grid_size[0])] for j in range(grid_size[1])])

# draw the figure
def update_plot(frame_number):
    ax.clear()
    # ax.set_zlim(0, 1)
    # ax.set_xlim(0, free_plane_length[0])
    # ax.set_ylim(0, free_plane_length[1])
    ax.set_xlabel("x")
    ax.set_ylabel("y")
    ax.invert_xaxis()
    propagate_wave(steps=1)
    dis = wave2energe(current_wave)
    ax.plot_surface(xs, ys, dis, cmap="coolwarm")
    print(frame_number)

# Writer = animation.writers['ffmpeg']
# writer = Writer(fps=15, metadata=dict(artist='Me'), bitrate=1800)

fig = plt.figure()
ax = fig.add_subplot(111, projection='3d')

x_scale=1
y_scale=1
z_scale=1

scale=np.diag([x_scale, y_scale, z_scale, 1.0])
scale=scale*(1.0/scale.max())
scale[3,3]=1.0

def short_proj():
  return np.dot(Axes3D.get_proj(ax), scale)

ax.get_proj=short_proj

ani = FuncAnimation(fig, update_plot, num_of_frames, interval=1, repeat=False)
# ani.save('wave.mp4', writer=writer)

plt.show()
