import numpy as np

import matplotlib
# matplotlib.use("Agg")
import matplotlib.animation as animation
import matplotlib.pyplot as plt
from matplotlib.animation import FuncAnimation

# create the x grid
dx = 0.1
x = np.arange(-10, 10+dx, dx)
N = len(x)
# create the initial wave packet
psi0 = np.exp(-x**2/4, dtype=np.complex128)/((2*np.pi)**(1/4))
# the potential is zero in this case
V = np.zeros([N])
# construct the 4th order FD matrix
g = -5j/(4*dx**2) - 1j*V
a = 1j/(24*dx**2)
diag = np.diag(g)
off_diag1 = np.diag([16*a]*(N-1), 1) + np.diag([16*a]*(N-1), -1)
off_diag2 = np.diag([-a]*(N-2), 2) + np.diag([-a]*(N-2), -2)
M = diag + off_diag1 + off_diag2
# create the time grid
dt = 0.01
t = np.arange(0, 20+dt, dt)
steps = len(t)
# create an array containing wavefunctions for each step
y = psi0
# the RK4 method
def propagate():
    global y
    k1 = np.dot(M, y)
    k2 = np.dot(M, y + k1*dt/2)
    k3 = np.dot(M, y + k2*dt/2)
    k4 = np.dot(M, y + k3*dt)
    y = y + dt*(k1 + 2*k2 + 2*k3 + k4)/6


# draw the figure
def update_plot(frame_number):
    ax.clear()
    ax.set_ylim(0, 0.5)
    ax.set_xlabel("x")
    ax.set_ylabel("y")
    # ax.plot_surface(xs, ys, ps, cmap="Dark2")
    propagate()
    ys = [abs(yy)**2 for yy in y]
    ax.plot(x, ys)


fig = plt.figure()
ax = fig.add_subplot(111)

ani = FuncAnimation(fig, update_plot, steps, interval=10)


plt.show()
