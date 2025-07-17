import numpy as np
import matplotlib.pyplot as plt
from matplotlib.animation import FuncAnimation, FFMpegWriter
from tqdm import tqdm

# Parameters
g = 9.81
L1 = L2 = 1.0
m1 = m2 = 1.0
dt = 0.01
steps = 300
resolution = 300

# Fractal grid
theta1_range = np.linspace(0, 2*np.pi, resolution)
theta2_range = np.linspace(0, 2*np.pi, resolution)

# Set up figure
fig, ax = plt.subplots(figsize=(6,6))
im = ax.imshow(np.zeros((resolution, resolution, 3)), origin='lower',
               extent=[0, 2*np.pi, 0, 2*np.pi])
ax.axis('off')

# Simulation function
def simulate(theta1, theta2, steps):
    omega1 = omega2 = 0.0
    for _ in range(steps):
        delta = theta2 - theta1
        den1 = (m1 + m2) * L1 - m2 * L1 * np.cos(delta)**2
        den2 = (L2 / L1) * den1
        a1 = (m2 * L1 * omega1**2 * np.sin(delta) * np.cos(delta) +
              m2 * g * np.sin(theta2) * np.cos(delta) +
              m2 * L2 * omega2**2 * np.sin(delta) -
              (m1 + m2) * g * np.sin(theta1)) / den1
        a2 = (-m2 * L2 * omega2**2 * np.sin(delta) * np.cos(delta) +
              (m1 + m2) * g * np.sin(theta1) * np.cos(delta) -
              (m1 + m2) * L1 * omega1**2 * np.sin(delta) -
              (m1 + m2) * g * np.sin(theta2)) / den2
        omega1 += a1 * dt
        omega2 += a2 * dt
        theta1 += omega1 * dt
        theta2 += omega2 * dt
    theta1 = np.mod(theta1, 2*np.pi)
    theta2 = np.mod(theta2, 2*np.pi)
    return theta1, theta2

# Wrapping update in tqdm manually since FuncAnimation hides it
frame_count = 30

def render_frames():
    for frame in tqdm(range(frame_count), desc="Rendering frames"):
        finals = np.zeros((resolution, resolution, 3))
        cur_steps = steps + frame * 5
        for i, t1 in enumerate(theta1_range):
            for j, t2 in enumerate(theta2_range):
                th1, th2 = simulate(t1, t2, cur_steps)
                color = plt.cm.hsv((th1 + th2) / (4 * np.pi))
                finals[j, i, :] = color[:3]
        im.set_data(finals)
        yield [im]

# Animate with manual frame generator
anim = FuncAnimation(fig, lambda f: f, frames=render_frames, blit=True, repeat=False)

# Save video with writer
writer = FFMpegWriter(fps=10)
anim.save("chaotic_butterfly.mp4", writer=writer)