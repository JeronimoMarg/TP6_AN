import numpy as np
import matplotlib.pyplot as plt

''' Este codigo realiza el punto 2 pero mostrando los porcentajes de hielo y agua en el dominio a traves del tiempo de simulacion. '''

dx = dy = 0.00025  # Paso espacial (0.25 mm)
dt = 0.01  # Paso temporal
total_time = 100  # Tiempo total de simulación en segundos
x_length = 0.01  # Longitud del dominio en x (10 mm)
y_length = 0.001  # Longitud del dominio en y (1 mm)

rho_ice = 918  # Densidad del hielo (kg/m^3)
rho_water = 1000  # Densidad del agua (kg/m^3)
c_ice = 2090  # Capacidad calorífica del hielo (J/kg*K)
c_water = 4186  # Capacidad calorífica del agua (J/kg*K)
k_ice = 1.6  # Conductividad térmica del hielo (W/m*K)
k_water = 0.6  # Conductividad térmica del agua (W/m*K)
h_fusion = 334000  # Calor latente de fusión (J/kg)

# calculo del alfa
alpha_ice = k_ice / (rho_ice * c_ice)
alpha_water = k_water / (rho_water * c_water)

# creacion de la malla
nx, ny = int(x_length / dx) + 1, int(y_length / dy) + 1
nt = int(total_time / dt)

def aKelvin(T):
    return T + 273.15

def aCelsius(T):
    return T - 273.15

# temperatura inicial (todo hielo)
T = np.full((nx, ny), aKelvin(-10))

# entalpia inicial (todo hielo)
H = np.where(T < aKelvin(0), rho_ice * c_ice * (T - aKelvin(0)), 0)

# funcion que determina las condiciones de frontera
def boundary_condition_right(t):
    return aKelvin(-10 + (95 * min(t / 10, 1)))

# se grafica la temperatura a lo largo de la línea central
def plot_temperature_line(T, t, x_length, y_length):
    middle_y = ny // 2
    temperature_profile = T[:, middle_y]
    plt.plot(np.linspace(0, x_length * 1000, len(temperature_profile)), aCelsius(temperature_profile), label=f"T = {t:.1f}s")

# se calculan los porcentajes de agua y hielo
def calculate_percentages(T):
    total_cells = T.size
    ice_cells = np.sum(T < aKelvin(0))
    water_cells = total_cells - ice_cells
    return (ice_cells/total_cells*100, water_cells/total_cells*100)

# simulacion propiamente dicha
for n in range(nt):
    t = n * dt
    T_old = T.copy()

    for i in range(1, nx - 1):
        for j in range(1, ny - 1):
            if T_old[i, j] < aKelvin(0):
                alpha, rho, c = alpha_ice, rho_ice, c_ice
            else:
                alpha, rho, c = alpha_water, rho_water, c_water

            # aproximacion de las derivadas segundas
            d2T_dx2 = (T_old[i+1, j] - 2 * T_old[i, j] + T_old[i-1, j]) / dx**2
            d2T_dy2 = (T_old[i, j+1] - 2 * T_old[i, j] + T_old[i, j-1]) / dy**2

            H[i, j] += rho * c * alpha * (d2T_dx2 + d2T_dy2) * dt

            # se convierte la entalpia a la temperatura
            if H[i, j] < 0:
                T[i, j] = H[i, j] / (rho_ice * c_ice) + aKelvin(0)
            elif H[i, j] > rho_water * h_fusion:
                T[i, j] = (H[i, j] - rho_water * h_fusion) / (rho_water * c_water) + aKelvin(0)
            else:
                T[i, j] = aKelvin(0)

    T[0, :] = T[1, :]
    T[-1, :] = boundary_condition_right(t)
    T[:, 0] = T[:, 1]
    T[:, -1] = T[:, -2]

    # se  muestran porcentajes cada 10 segundos
    if n % int(10/dt) == 0:
        ice_percent, water_percent = calculate_percentages(T)
        print(f"Tiempo: {t:.1f}s")
        print(f"Hielo: {ice_percent:.1f}%")
        print(f"Agua: {water_percent:.1f}%")
        print("-" * 20)
