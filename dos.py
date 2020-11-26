import numpy as np
import scipy.special
import cmath
import math
import copy


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





grid_size = (128, 128)
time_step = 0.01
max_order_of_chebyshev_poly = 100000
allowed_error = 10**(-13)
operator_size = grid_size[0] * grid_size[1]
t_n = -2.7
t_nn = -0.4
on_site_potential = 2.7

sample_size = 512
num_of_evolution_steps = 1
dt = time_step * num_of_evolution_steps

def get_initial_state():
    state = np.random.random((operator_size, ))
    total = sum([c**2 for c in state])
    return state / math.sqrt(total)

def is_A_site(i, j):
    if (j % 2 == 0 and i % 2 == 1) or (j % 2 == 1 and i % 2 == 0):
        return True
    return False



def flatten_hamiltionian(i, j):
    rowH = np.zeros((grid_size[1], grid_size[0]))
    if is_A_site(i, j):
        # nearest neighbors
        rowH[j][(i + 1) % grid_size[0]] = -t_n
        rowH[j][(i - 1) % grid_size[0]] = -t_n
        rowH[(j - 1) % grid_size[1]][i] = -t_n

        # # next nearest neighbors
        # rowH[j][(i + 2) % grid_size[0]] = -t_nn
        # rowH[(j + 1) % grid_size[1]][(i + 1) % grid_size[0]] = -t_nn
        # rowH[j][(i - 2) % grid_size[0]] = -t_nn
        # rowH[(j + 1) % grid_size[1]][(i - 1) % grid_size[0]] = -t_nn
        # rowH[(j - 1) % grid_size[1]][(i + 1) % grid_size[0]] = -t_nn
        # rowH[(j - 1) % grid_size[1]][(i - 1) % grid_size[0]] = -t_nn
        #
        # # on-site potential
        # rowH[j][i] = on_site_potential
    else:
        # nearest neighbors
        rowH[j][(i + 1) % grid_size[0]] = -t_n
        rowH[j][(i - 1) % grid_size[0]] = -t_n
        rowH[(j + 1) % grid_size[1]][i] = -t_n

        # # next nearest neighbors
        # rowH[j][(i + 2) % grid_size[0]] = -t_nn
        # rowH[(j - 1) % grid_size[1]][(i + 1) % grid_size[0]] = -t_nn
        # rowH[j][(i - 2) % grid_size[0]] = -t_nn
        # rowH[(j - 1) % grid_size[1]][(i - 1) % grid_size[0]] = -t_nn
        # rowH[(j + 1) % grid_size[1]][(i + 1) % grid_size[0]] = -t_nn
        # rowH[(j + 1) % grid_size[1]][(i - 1) % grid_size[0]] = -t_nn
        #
        # # on-site potential
        # rowH[j][i] = on_site_potential
    return sparse.csr_matrix(rowH.flatten())


def get_hamiltonian():
    hamiltonian = []
    for j in range(grid_size[1]):
        for i in range(grid_size[0]):
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
    max_eigenvalue = eigsh(H, k=1, which="LA")[0][0]
    min_eigenvalue = eigsh(H, k=1, which="SA")[0][0]
    # max_eigenvalue = 0
    # min_eigenvalue = -8
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



def normalize_state(state):
    total = sum([abs(c)**2 for c in state])
    return state / math.sqrt(total)



evolution_operator = get_evolution_operator_one_timestep()
initial_state = get_initial_state()
current_state = copy.deepcopy(initial_state)
def propagate_wave(steps=1):
    global current_state
    for i in range(steps):
        current_state = normalize_state(evolution_operator.dot(current_state))
    return current_state


def get_expectation_samples():
    samples = [None] * sample_size
    for i in range(sample_size):
        current_state = propagate_wave(steps=num_of_evolution_steps)
        samples[i] = initial_state.dot(current_state)
        print(i)
    return samples

def get_energy_distribution():
    samples = get_expectation_samples()
    g = np.fft.fft(samples)
    w = np.fft.fftfreq(sample_size)*2*np.pi/dt
    plt.scatter(w, np.abs(g))
    plt.show()

get_energy_distribution()
