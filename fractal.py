import numpy as np
import matplotlib.pyplot as plt
from matplotlib.animation import FuncAnimation
from scipy.integrate import solve_ivp

# Constants (assuming l1=l2=m1=m2=1)
g = 9.81
L1 = L2 = 1.0
M1 = M2 = 1.0

# System of ODEs
def double_pendulum(t, y):
    θ1, z1, θ2, z2 = y
    Δ = θ2 - θ1

    denom1 = (M1 + M2) * L1 - M2 * L1 * np.cos(Δ) ** 2
    denom2 = (L2 / L1) * denom1

    dθ1_dt = z1
    dθ2_dt = z2

    dz1_dt = (M2 * L1 * z1 ** 2 * np.sin(Δ) * np.cos(Δ) +
              M2 * g * np.sin(θ2) * np.cos(Δ) +
              M2 * L2 * z2 ** 2 * np.sin(Δ) -
              (M1 + M2) * g * np.sin(θ1)) / denom1

    dz2_dt = (-M2 * L2 * z2 ** 2 * np.sin(Δ) * np.cos(Δ) +
              (M1 + M2) * g * np.sin(θ1) * np.cos(Δ) -
              (M1 + M2) * L1 * z1 ** 2 * np.sin(Δ) -
              (M1 + M2) * g * np.sin(θ2)) / denom2

    return [dθ1_dt, dz1_dt, dθ2_dt, dz2_dt]

# Initial conditions: [θ1, θ1_dot, θ2, θ2_dot]
y0 = [np.pi / 2, 0, np.pi / 2 + 0.01, 0]  # Slight offset to see chaos

# Time array
t_span = (0, 20)
t_eval = np.linspace(*t_span, 1000)

# Solve ODE
sol = solve_ivp(double_pendulum, t_span, y0, t_eval=t_eval)

θ1 = sol.y[0]
θ2 = sol.y[2]

# Convert to (x, y) coordinates
x1 = L1 * np.sin(θ1)
y1 = -L1 * np.cos(θ1)
x2 = x1 + L2 * np.sin(θ2)
y2 = y1 - L2 * np.cos(θ2)

# Set up animation
fig, ax = plt.subplots(figsize=(6, 6))
ax.set_xlim(-2.2, 2.2)
ax.set_ylim(-2.2, 2.2)
ax.set_aspect('equal')
ax.axis('off')

line, = ax.plot([], [], 'o-', lw=2, color='black')
trace, = ax.plot([], [], lw=1, alpha=0.6, color='blue')
trail_x, trail_y = [], []

def init():
    line.set_data([], [])
    trace.set_data([], [])
    return line, trace

def update(i):
    x = [0, x1[i], x2[i]]
    y = [0, y1[i], y2[i]]
    line.set_data(x, y)

    trail_x.append(x2[i])
    trail_y.append(y2[i])
    trace.set_data(trail_x, trail_y)

    return line, trace

ani = FuncAnimation(fig, update, frames=len(t_eval), init_func=init, blit=True, interval=20)
plt.show()