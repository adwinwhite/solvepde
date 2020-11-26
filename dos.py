import numpy as np
import scipy.special
import cmath
import math
import copy
import multiprocessing as mp


import matplotlib
import matplotlib.pyplot as plt

from scipy.sparse.linalg import eigsh
from scipy import sparse
from scipy.signal import windows
# from sparse_dot_mkl import dot_product_mkl


# grid size, timestep and sample size are irrelevant to the shape of the output figure.


grid_size = (128, 128)
time_step = 0.001
max_order_of_chebyshev_poly = 100000
allowed_error = 10**(-13)
operator_size = grid_size[0] * grid_size[1]
t_n = -2.77
t_nn = -0.1
on_site_potential = -0.3

sample_size = 1024
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
    # pool = mp.Pool(mp.cpu_count())
    # hamiltonian = [pool.apply(flatten_hamiltionian, args=(i, j)) for j in range(grid_size[1]) for i in range(grid_size[0])]
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
def get_evolution_operator_one_timestep(dt):
    print(H.shape)
    eigen_factor = 2
    # print("det(H) : {}".format(scipy.linalg.det(H)))
    # max_eigenvalue = eigsh(H, k=1, which="LA")[0][0]
    # min_eigenvalue = eigsh(H, k=1, which="SA")[0][0]
    max_eigenvalue = 10
    min_eigenvalue = -10
    print("max_eigenvalue : {}".format(max_eigenvalue))
    print("min_eigenvalue : {}".format(min_eigenvalue))
    z = (max_eigenvalue - min_eigenvalue) * dt / eigen_factor
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
    evolution_operator = (evolution_operator * 2 + sparse.identity(operator_size, dtype=np.complex128) * scipy.special.jv(0, z)) * np.exp((max_eigenvalue + min_eigenvalue) * dt * (-0.5j))
    print("{} : {}".format(i, abs(jv)))
    # detm = scipy.linalg.det(evolution_operator * evolution_operator.transpose().conj())
    # factor = pow(1 / detm, 1 / operator_size)
    # evolution_operator *= factor
    # print("{}".format(scipy.linalg.det(evolution_operator * evolution_operator.transpose().conj())))
    return sparse.csr_matrix(evolution_operator)



def normalize_state(state):
    total = sum([abs(c)**2 for c in state])
    return state / math.sqrt(total)



evolution_operator = get_evolution_operator_one_timestep(time_step)
# inverse_evolution_operator = get_evolution_operator_one_timestep(-time_step)
# print(sparse.linalg.norm(evolution_operator.multiply(inverse_evolution_operator)))
initial_state = get_initial_state()
def evolve(state, steps=1, backwards=False):
    tmp_state = state
    # if backwards:
    #     for i in range(steps):
    #         tmp_state = normalize_state(inverse_evolution_operator.dot(tmp_state))
    # else:
    for i in range(steps):
        tmp_state = normalize_state(evolution_operator.dot(tmp_state))
    return tmp_state


def get_expectation_samples():
    samples = [None] * sample_size
    current_state = copy.deepcopy(initial_state)
    for i in range(sample_size):
        current_state = evolve(current_state, steps=num_of_evolution_steps)
        samples[i] = initial_state.dot(current_state)
        print(i)
    return samples

def get_dos():
    samples = get_expectation_samples()
    gaussian_window = windows.gaussian(sample_size, sample_size * 0.1)
    samples *= gaussian_window
    g = np.fft.fft(samples) * dt
    w = np.fft.fftfreq(sample_size)*2*np.pi/dt
    plt.scatter(w, np.real(g))
    plt.scatter(w, np.imag(g))

    plt.gca().set_xlim(-100, 100)
    plt.show()

get_dos()
