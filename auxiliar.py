import numpy as np
import matplotlib.pyplot as plt

# Parameters
L = 0.01  # Length of the sheet in meters (10 mm)
h = 0.001  # Spatial step in meters (1 mm)
dt = 0.1  # Time step
time = 100  # Total simulation time in seconds
alpha = 1  # Thermal diffusivity (example value, adjust as needed)

def stencil(n):
    A = -2 * np.eye(n)
    for j in range(n - 1):
        A[j, j + 1] = 1
        A[j + 1, j] = 1
    return A

def boundary_condition_right(t):
    if t < 10:
        return -10 + (95 * t / 10)
    else:
        return 85

def explicit(h, dt, time, L):
    t = 0.0
    n = int(L / h) + 1  # Number of sections within the 1D sheet
    x = np.linspace(0, L, n)
    T = np.full(n, -10.0)  # Initial temperature in °C
    sol = []
    sol.append(T.copy())
    mat = alpha * dt * stencil(n) / h / h + np.eye(n)
    while t < time:
        T = np.dot(mat, T)
        T[0] = T[1]  # Left boundary condition (insulated)
        T[-1] = boundary_condition_right(t)  # Right boundary condition (changing temperature)
        sol.append(T.copy())
        t += dt
    return sol

# Run the simulation
sol = explicit(h, dt, time, L)

# Plot the results
plt.ion()
plt.xlabel('Position (m)')
plt.ylabel('Temperature (°C)')
plt.title('Temperature Distribution Over Time')
for T in sol:
    plt.plot(np.linspace(0, L, len(T)), T)
    plt.pause(0.1)
plt.show()