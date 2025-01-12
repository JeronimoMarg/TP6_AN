import numpy as np
import matplotlib.pyplot as plt

# Boundary condition on the right side
def boundary_condition_right(t):
    if t < 10:
        aux = -10 + (95 * t / 10)  # Linear increase from -10 to 85
        return aux
    else:
        return 85

def stencil_2d(n, m):
    """Constructs a 2D finite difference stencil matrix for heat diffusion."""
    A = np.eye(n * m) * -4  # Main diagonal
    for i in range(n * m):
        if i % n != n - 1:  # Right neighbor
            A[i, i + 1] = 1
        if i % n != 0:  # Left neighbor
            A[i, i - 1] = 1
        if i >= n:  # Top neighbor
            A[i, i - n] = 1
        if i < n * (m - 1):  # Bottom neighbor
            A[i, i + n] = 1
    return A

def explicit_phase_change(h, dt, time, Lx, Ly, latent_heat, k, rho, c_ice, c_water):
    """Simulates temperature evolution with phase change using an explicit finite difference scheme."""
    t = 0.0
    nx, ny = int(Lx / h), int(Ly / h)
    print(f"nx: {nx}, ny: {ny}")  # Debug print to check values
    x = np.linspace(0, Lx, nx)
    y = np.linspace(0, Ly, ny)
    T = np.full((ny, nx), -10.0)  # Initial temperature in °C
    sol = [T.copy()]

    # Stencil matrix for 2D heat equation
    A = stencil_2d(nx, ny)
    mat = np.eye(nx * ny) + (dt * k / (rho * c_ice)) * A / h**2

    while t < time:
        T_flat = T.flatten()

        # Update temperature field
        T_flat = np.dot(mat, T_flat)

        # Reshape back to 2D
        T = T_flat.reshape((ny, nx))

        # Handle phase change
        for i in range(ny):
            for j in range(nx):
                if T[i, j] == 0:  # At phase change temperature
                    T[i, j] += latent_heat / (rho * ((c_ice + c_water) / 2))

        # Apply boundary conditions
        if ny > 1:
            T[-1, :] = boundary_condition_right(t)  # Right (changing temperature)
            T[0, :] = T[1, :]  # Left (insulated)
        if nx > 1:
            T[:, 0] = T[:, 1]  # Bottom (insulated)
            T[:, -1] = T[:, -2]  # Top (insulated)

        sol.append(T.copy())
        t += dt

    return sol, x, y

# Parameters
Lx, Ly = 0.01, 0.001  # Domain size in meters (10mm x 1mm)
h = 0.001  # Grid spacing (1mm)
dt = 0.01  # Time step
time = 100.0  # Total simulation time
latent_heat = 334000  # Latent heat of fusion (J/kg)
k = 2.22  # Thermal conductivity (W/mK)
rho = 917  # Density of ice (kg/m^3)
c_ice = 2100  # Specific heat of ice (J/kgK)
c_water = 4187  # Specific heat of water (J/kgK)

# Run the simulation
sol, x, y = explicit_phase_change(h, dt, time, Lx, Ly, latent_heat, k, rho, c_ice, c_water)

# Visualization
final_T = sol[-1]
plt.imshow(final_T, extent=[0, Lx * 1000, 0, Ly * 1000], origin='lower', cmap='coolwarm', aspect='auto')
plt.colorbar(label='Temperature (°C)')
plt.xlabel('X (mm)')
plt.ylabel('Y (mm)')
plt.title('Temperature Distribution at t = 100s')
plt.show()
