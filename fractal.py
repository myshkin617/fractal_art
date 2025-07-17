import numpy as np
import matplotlib.pyplot as plt
from matplotlib.animation import FuncAnimation
from scipy.integrate import solve_ivp
import matplotlib.cm as cm

# Parameters
g = 9.81
L1 = L2 = M1 = M2 = 1.0
t_span = (0, 20)
t_eval = np.linspace(*t_span, 1000)
N = 10  # number of pendulums

# Generate slightly different initial θ2s
initial_angles = np.linspace(np.pi/2, np.pi/2 + 0.1, N)

# Store solutions
trajectories = []

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

# Solve for each initial condition
for θ2_0 in initial_angles:
    y0 = [np.pi/2, 0, θ2_0, 0]
    sol = solve_ivp(double_pendulum, t_span, y0, t_eval=t_eval)
    θ1 = sol.y[0]
    θ2 = sol.y[2]
    x1 = L1 * np.sin(θ1)
    y1 = -L1 * np.cos(θ1)
    x2 = x1 + L2 * np.sin(θ2)
    y2 = y1 - L2 * np.cos(θ2)
    trajectories.append((x2, y2))

# Plotting setup
fig, ax = plt.subplots(figsize=(8, 8))
ax.set_xlim(-2.5, 2.5)
ax.set_ylim(-2.5, 2.5)
ax.set_aspect('equal')
ax.axis('off')

colors = cm.hsv(np.linspace(0, 1, N))  # rainbow colors
lines = [ax.plot([], [], lw=1, color=colors[i])[0] for i in range(N)]
trails_x = [[] for _ in range(N)]
trails_y = [[] for _ in range(N)]

def init():
    for line in lines:
        line.set_data([], [])
    return lines

def update(frame):
    for i in range(N):
        x, y = trajectories[i]
        trails_x[i].append(x[frame])
        trails_y[i].append(y[frame])
        lines[i].set_data(trails_x[i], trails_y[i])
    return lines

ani = FuncAnimation(fig, update, frames=len(t_eval), init_func=init, blit=True, interval=20)
plt.show()