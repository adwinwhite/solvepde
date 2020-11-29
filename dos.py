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



t_n = -2.7
t_nn = -0.1
on_site_potential = -0.3


def get_random_state(size):
    state = np.random.random((size, ))
    total = sum([c**2 for c in state])
    return state / math.sqrt(total)

def is_A_site(i, j):
    if (j % 2 == 0 and i % 2 == 1) or (j % 2 == 1 and i % 2 == 0):
        return True
    return False



def flatten_hamiltonian(i, j, horizontal_size, vertical_size):
    rowH = np.zeros((vertical_size, horizontal_size))
    if is_A_site(i, j):
        # nearest neighbors
        rowH[j][(i + 1) % horizontal_size] = -t_n
        rowH[j][(i - 1) % horizontal_size] = -t_n
        rowH[(j - 1) % vertical_size][i] = -t_n

        # # next nearest neighbors
        # rowH[j][(i + 2) % horizontal_size] = -t_nn
        # rowH[(j + 1) % vertical_size][(i + 1) % horizontal_size] = -t_nn
        # rowH[j][(i - 2) % horizontal_size] = -t_nn
        # rowH[(j + 1) % vertical_size][(i - 1) % horizontal_size] = -t_nn
        # rowH[(j - 1) % vertical_size][(i + 1) % horizontal_size] = -t_nn
        # rowH[(j - 1) % vertical_size][(i - 1) % horizontal_size] = -t_nn
        #
        # # on-site potential
        # rowH[j][i] = on_site_potential
    else:
        # nearest neighbors
        rowH[j][(i + 1) % horizontal_size] = -t_n
        rowH[j][(i - 1) % horizontal_size] = -t_n
        rowH[(j + 1) % vertical_size][i] = -t_n

        # # next nearest neighbors
        # rowH[j][(i + 2) % horizontal_size] = -t_nn
        # rowH[(j - 1) % vertical_size][(i + 1) % horizontal_size] = -t_nn
        # rowH[j][(i - 2) % horizontal_size] = -t_nn
        # rowH[(j - 1) % vertical_size][(i - 1) % horizontal_size] = -t_nn
        # rowH[(j + 1) % vertical_size][(i + 1) % horizontal_size] = -t_nn
        # rowH[(j + 1) % vertical_size][(i - 1) % horizontal_size] = -t_nn
        #
        # # on-site potential
        # rowH[j][i] = on_site_potential
    return sparse.csr_matrix(rowH.flatten())


def get_hamiltonian(horizontal_size, vertical_size):
    # pool = mp.Pool(mp.cpu_count())
    # hamiltonian = [pool.apply(flatten_hamiltionian, args=(i, j)) for j in range(vertical_size) for i in range(horizontal_size)]
    hamiltonian = []
    for j in range(vertical_size):
        for i in range(horizontal_size):
            hamiltonian.append(flatten_hamiltonian(i, j, horizontal_size, vertical_size))
        print(j)
    return sparse.vstack(hamiltonian)



# using recursion formula for chebyshev polynomial. x's range is R rather than [-1, 1]
def get_evolution_operator_one_timestep(H, timestep, allowed_error=10**(-13), max_order_of_chebyshev_poly=10000):
    max_eigenvalue = eigsh(H, k=1, which="LA")[0][0]
    min_eigenvalue = eigsh(H, k=1, which="SA")[0][0]
    # max_eigenvalue = 10
    # min_eigenvalue = -10
    print("max_eigenvalue : {}".format(max_eigenvalue))
    print("min_eigenvalue : {}".format(min_eigenvalue))

    operator_size = H.shape[0]
    z = (max_eigenvalue - min_eigenvalue) * timestep / 2
    B = ((H - sparse.identity(operator_size) * (max_eigenvalue + min_eigenvalue) / 2) / (max_eigenvalue - min_eigenvalue)) * (-1j) * 2

    evolution_operator = sparse.csr_matrix((operator_size, operator_size), dtype=np.complex128)
    T_tilde1 = sparse.identity(operator_size)
    T_tilde2 = B
    jv = 1
    i = 1
    while abs(jv) > allowed_error and i <= max_order_of_chebyshev_poly:
        jv = scipy.special.jv(i, z)
        evolution_operator += jv * T_tilde2

        next_T_tilde = B * 2 * T_tilde2 - T_tilde1
        T_tilde1 = T_tilde2
        T_tilde2 = next_T_tilde
        i += 1
        print(i)

    evolution_operator = (evolution_operator * 2 + sparse.identity(operator_size, dtype=np.complex128) * scipy.special.jv(0, z)) * np.exp((max_eigenvalue + min_eigenvalue) * timestep * (-0.5j))
    print("{} : {}".format(i, abs(jv)))
    return sparse.csr_matrix(evolution_operator)



def normalize_state(state):
    total = sum([abs(c)**2 for c in state])
    return state / math.sqrt(total)



def get_correlation(initial_state, evolution_operator, sample_size):
    current_state = copy.deepcopy(initial_state)
    correlations = np.empty((sample_size + 1, ))
    correlations[0] = current_state.dot(current_state)
    for i in range(1, sample_size + 1):
        current_state = normalize_state(evolution_operator.dot(current_state))
        correlations[i] = initial_state.dot(current_state)
        print(i)
    return correlations

def half_hanning(i, N):
    return 0.5 * (1 + np.cos(np.pi * i / N))

def get_dos(correlations, energy_range, timestep):
    # windowed_corr = correlations * windows.hann(correlations.size, sym=False)
    # g = np.fft.fft(windowed_corr) * timestep
    # w = np.fft.fftfreq(windowed_corr.size)*2*np.pi/timestep/abs(t_n)
    # return w, g

    sample_size = correlations.size - 1
    energies = [energy_range / 2 * (-1 + i / sample_size) for i in range(sample_size * 2 + 1)]

    corr_symmetric = np.empty(sample_size * 2 + 1, dtype=np.complex128)
    corr_symmetric[sample_size] = correlations[0]
    for i in range(1, sample_size + 1):
        corr_symmetric[sample_size + i] = half_hanning(i, sample_size) * correlations[i]
        corr_symmetric[sample_size - i] = half_hanning(i, sample_size) * np.conjugate(correlations[i])

    # Fourier transform
    corr_fft = np.fft.fft(corr_symmetric)
    dos = np.empty(sample_size * 2 + 1, dtype=np.complex128)
    for i in range(sample_size * 2 + 1):
        dos[i - sample_size] = corr_fft[i]

    # Normalise
    dos = dos / np.sum(dos)
    return np.array(energies), dos

def draw_dos(horizontal_size, vertical_size, timestep, sample_size):
    random_state = get_random_state(horizontal_size * vertical_size)
    H = get_hamiltonian(horizontal_size, vertical_size)
    evolution_operator = get_evolution_operator_one_timestep(H, timestep)
    corrs = get_correlation(random_state, evolution_operator, sample_size)
    w, g = get_dos(corrs, 6., timestep)
    plt.scatter(w, np.abs(g))
    plt.show()
