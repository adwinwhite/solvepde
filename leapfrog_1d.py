import numpy as np
import scipy.linalg
import scipy.special
import math
import cmath
import matplotlib.pyplot as plt
from matplotlib.animation import FuncAnimation
import time


free_line = (0, 1)
grid_size = 256
packet_width = 0.04
momentum = 0
time_step = 0.000001
num_of_frames = 10000
num_of_partitions_H = 800
max_order_of_chebyshev_poly = 1000000
allowed_error = 0

free_line_length = free_line[1] - free_line[0]
mesh_step = free_line_length / grid_size
mesh_step_reciprocal = grid_size / free_line_length

def get_potential(x):
    return x * 100

def initial_wave_unnormalized(x):
    # with width 0.01 and direction vector k=1
    if x < free_line[0] or x > free_line[1]:
        return 0
    return cmath.exp(-(x-0.5)**2/4/packet_width**2 + x*momentum*1j)

def get_discretized_init_wave_function():
    results = []
    for i in range(grid_size + 1):
        results.append(initial_wave_unnormalized(i * mesh_step))
    results = np.array(results)
    return results

def normalized_wave(wave):
    integral = sum([abs(x)**2 for x in wave]) * mesh_step
    factor = math.sqrt(1/integral)
    return wave * factor

def get_hamiltonian():
    
