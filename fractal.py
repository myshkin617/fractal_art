import numpy as np
import matplotlib.pyplot as plt

# Grid of initial values
theta1_vals = np.linspace(-np.pi, np.pi, 400)
theta2_vals = np.linspace(-np.pi, np.pi, 400)
image = np.zeros((400, 400))

def simulate(theta1_0, theta2_0, t_max=20, dt=0.01):
    # Initial angular velocities
    omega1 = 0
    omega2 = 0
    g = 9.81

    # Preallocate arrays
    theta1, theta2 = theta1_0, theta2_0

    for _ in range(int(t_max / dt)):
        # Super simplified acceleration (not accurate, but chaotic af)
        domega1 = -g * np.sin(theta1)
        domega2 = -g * np.sin(theta2)
        
        omega1 += domega1 * dt
        omega2 += domega2 * dt
        
        theta1 += omega1 * dt
        theta2 += omega2 * dt
    
    return theta2 % (2 * np.pi)  # Use mod to make color cycle

# Iterate over grid
for i, t1 in enumerate(theta1_vals):
    for j, t2 in enumerate(theta2_vals):
        result = simulate(t1, t2)
        image[i, j] = result  # This becomes pixel intensity

# Show it as an image (fractal basin)
plt.imshow(image, cmap='hsv', extent=[-np.pi, np.pi, -np.pi, np.pi])
plt.title("Fractal Basin of Double Pendulum (chaotic coloring)")
plt.xlabel("Initial θ1")
plt.ylabel("Initial θ2")
plt.colorbar(label="Final θ2 mod 2π")
plt.show()