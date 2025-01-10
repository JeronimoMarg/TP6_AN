import numpy as np
import matplotlib.pyplot as plt

def aKelvin(T):
    return T + 273.15

def aCelsius(T):
    return T - 273.15

# Parameters
dx = 0.00025  # spatial step (0.1 mm)
dy = 0.00025
dt = 0.1    # time step (s)
total_time = 100  # total simulation time (s)
x_length = 0.01  # domain length in x (10 mm)
y_length = 0.001  # domain length in y (1 mm)

# Thermal properties
rho_ice = 917  # density of ice (kg/m^3)
rho_water = 997  # density of water (kg/m^3)
c_ice = 2100  # specific heat of ice (J/kg*K)
c_water = 4186  # specific heat of water (J/kg*K)
k_ice = 1.6  # thermal conductivity of ice (W/m*K)
k_water = 0.6  # thermal conductivity of water (W/m*K)
h_fusion = 334  # latent heat of fusion (J/kg)

alpha_ice = k_ice / (rho_ice * c_ice)
alpha_water = k_water / (rho_water * c_water)

# Discretize the domain
nx = int(x_length / dx) + 1
ny = int(y_length / dy) + 1
nt = int(total_time / dt)

# Initialize temperature field (in Celsius)
T = np.full((nx, ny), aKelvin(-10))

# Enthalpy field
H = np.full((nx, ny), rho_ice * c_ice * (aKelvin(-10)))

# Boundary condition on the right side
def boundary_condition_right(t):
    if t <= aKelvin(-10):
        aux = -10 + (95 * t / 10)  # Linear increase from -10 to 85
        return aKelvin(aux)
    else:
        return aKelvin(85)

# Main simulation loop
for n in range(nt):
    t = n * dt
    T_old = T.copy()

    if t % 1 == 0:

        total_cells = nx * ny
        ice_cells = np.sum(T < aKelvin(0))
        water_cells = np.sum(T > aKelvin(0))
        ice_percentage = (ice_cells / total_cells) * 100
        water_percentage = (water_cells / total_cells) * 100
        print(f"Time: {t:.1f} s - Ice: {ice_percentage:.2f}%, Water: {water_percentage:.2f}%")

        plt.imshow(T.T, cmap="coolwarm", origin="lower", extent=[0, x_length * 1000, 0, y_length * 1000], vmin=aKelvin(-10), vmax=aKelvin(85))
        plt.colorbar(label="Temperature (K)")
        plt.title("Temperature Distribution After Seconds")
        plt.xlabel("x (mm)")
        plt.ylabel("y (mm)")
        plt.show()

    for i in range(1, nx - 1):
        for j in range(1, ny - 1):
            # Thermal diffusivity based on phase
            if T_old[i, j] < aKelvin(0):
                alpha = alpha_ice
                k = k_ice
                c = c_ice
                rho = rho_ice
            else:
                alpha = alpha_water
                k = k_water
                c = c_water
                rho = rho_water

            # Finite difference update
            d2T_dx2 = (T_old[i+1, j] - 2 * T_old[i, j] + T_old[i-1, j]) / dx**2
            d2T_dy2 = (T_old[i, j+1] - 2 * T_old[i, j] + T_old[i, j-1]) / dy**2

            T[i,j] = T_old[i,j] + alpha * dt * (d2T_dx2 + d2T_dy2)
            
    # Apply boundary conditions
    T[0, :] = T[1, :]  # Left (insulated)
    T[-1, :] = boundary_condition_right(t)  # Right (changing temperature)
    T[:, 0] = T[:, 1]  # Bottom (insulated)
    T[:, -1] = T[:, -2]  # Top (insulated)

# Plot final temperature distribution
plt.imshow(T.T, cmap="coolwarm", origin="lower", extent=[0, x_length * 1000, 0, y_length * 1000], vmin=aKelvin(-10), vmax=aKelvin(85))
plt.colorbar(label="Temperature (K)")
plt.title("Temperature Distribution After 100 Seconds")
plt.xlabel("x (mm)")
plt.ylabel("y (mm)")
plt.show()