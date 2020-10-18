import math

import numpy as np

import functools

import matplotlib
# matplotlib.use("Agg")
import matplotlib.animation as animation
import matplotlib.pyplot as plt

from mpl_toolkits.mplot3d.axes3d import Axes3D
from mpl_toolkits.mplot3d import proj3d
from matplotlib.animation import FuncAnimation

grid_size = (10, 10)
light_speed = 1
electric_constant = 1
magnetic_constant = 1
free_plane_length = (50, 50)
num_of_frames = 10000
time_step = 0.1
k=10
packet_center = (0.5 * free_plane_length[0], 0.5 * free_plane_length[1])
packet_size = (0.1 * free_plane_length[0], 0.1 * free_plane_length[1])



def f(x, y, t=0):
    return math.sin(k * (x - packet_center[0] - light_speed * t)) * math.exp(-((x - packet_center[0] - light_speed * t) / packet_size[0])**2 - ((y - packet_center[1]) / packet_size[1])**2)

def partial_f(x, y, t=0):
    return math.cos(k * (x - packet_center[0] - light_speed * t)) * (-light_speed * k) * math.exp(-((x - packet_center[0] - light_speed * t) / packet_size[0])**2 - ((y - packet_center[1]) / packet_size[1])**2) + math.sin(k * (x - packet_center[0] - light_speed * k)) * math.exp(-((x - packet_center[0] - light_speed * t) / packet_size[0])**2 - ((y - packet_center[1]) / packet_size[1])**2) * 2 * light_speed / (packet_size[0])

@functools.lru_cache
def k_n(n):
    print(n)
    return n * math.pi / free_plane_length[0]

@functools.lru_cache
def k_m(m):
    return m * math.pi / free_plane_length[1]

@functools.lru_cache
def omega(n, m):
    return light_speed * math.sqrt(k_n(n)**2 + k_m(m)**2)

def a_nm(n, m, step_x=1, step_y=1):
    integral = 0
    for x in np.arange(0, free_plane_length[0], step_x):
        for y in np.arange(0, free_plane_length[1], step_y):
            print(y)
            integral += math.sin(k_n(n) * x) * math.sin(k_m(m) * y) * partial_f(x, y)
    integral *= step_x * step_y * 4 / (omega(n, m) * free_plane_length[0] * free_plane_length[1])
    return integral

def b_nm(n, m, step_x=1, step_y=1):
    integral = 0
    for x in np.arange(0, free_plane_length[0], step_x):
        for y in np.arange(0, free_plane_length[1], step_y):
            integral += math.sin(k_n(n) * x) * math.sin(k_m(m) * y) * f(x, y)
    integral *= step_x * step_y * 4 / (free_plane_length[0] * free_plane_length[1])
    return integral

def E_z(x, y, t=0, n=50, m=50):
    print(x)
    summation = 0
    for n in range(1, n + 1):
        for m in range(1, m + 1):
            print(m)
            summation += math.sin(k_n(n) * x) * math.sin(k_m(m) * y) * (a_nm(n, m) * math.sin(omega(n, m) * t) + b_nm(n, m) * math.cos(omega(n, m) * t))
    return summation

def H_x(x, y, t=0, n=50, m=50):
    summation = 0
    for n in range(1, n + 1):
        for m in range(1, m + 1):
            summation += light_speed * k_m(m) / omega(n, m) * math.sin(k_n(n) * x) * math.cos(k_m(m) * y) * (a_nm * math.cos(omega(n, m) * t) - b_nm(n, m) * math.sin(omega(n, m) * t))
    return summation

def H_y(x, y, t=0, n=50, m=50):
    summation = 0
    for n in range(1, n + 1):
        for m in range(1, m + 1):
            summation += -light_speed * k_n(n) / omega(n, m) * math.cos(k_n(n) * x) * math.sin(k_m(m) * y) * (a_nm(n, m) * math.cos(omega(n, m) * t) - b_nm(n, m) * math.sin(omega(n, m) * t))
    return summation

def energy_density(t):
    xs = np.linspace(0, free_plane_length[0], grid_size[0] + 1)
    ys = np.linspace(0, free_plane_length[1], grid_size[1] + 1)
    es = [[((electric_constant * E_z(x, y, t)**2 + magnetic_constant * (H_x(x, y, t)**2 + H_y((x, y, t)**2))) / 2) for x in xs] for y in ys]
    # for y in ys:
    #     row = []
    #     for x in xs:
    #         row.append((electric_constant * E_z(x, y, t=t)**2 + magnetic_constant * (H_x(x, y, t=t)**2 + H_y((x, y, t=t)**2))) / 2)
    #     es.append(row)
    return es


xs = np.linspace(0, free_plane_length[0], grid_size[0] + 1)
ys = np.linspace(0, free_plane_length[1], grid_size[1] + 1)
xs, ys = np.meshgrid(xs, ys)
# ps = np.array([[get_potential(i * mesh_step, j * mesh_step) / 1000 for i in range(grid_size[0])] for j in range(grid_size[1])])

# draw the figure
def update_plot(frame_number):
    ax.clear()
    ax.set_zlim(0, 1)
    ax.set_xlim(0, free_plane_length[0])
    ax.set_ylim(0, free_plane_length[1])
    ax.set_xlabel("x")
    ax.set_ylabel("y")
    ax.invert_xaxis()
    es = energy_density(frame_number * time_step)
    ax.plot_surface(xs, ys, es, cmap="coolwarm")
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

ani = FuncAnimation(fig, update_plot, num_of_frames, interval=1)
# ani.save('wave.mp4', writer=writer)

plt.show()
