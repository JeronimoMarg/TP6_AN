import numpy as np
import matplotlib.pyplot as plt

def aKelvin(T):
    return T + 273.15

def aCelsius(T):
    return T - 273.15

# Parámetros físicos y simulación
dx = dy = 0.00025  # Paso espacial (0.25 mm)
dt = 0.01  # Paso temporal reducido para estabilidad
total_time = 1200  # Tiempo total de simulación (s)
x_length = 0.01  # Longitud del dominio en x (10 mm)
y_length = 0.001  # Longitud del dominio en y (1 mm)

# Propiedades del material
rho_ice = 918  # Densidad del hielo (kg/m^3)
rho_water = 1000  # Densidad del agua (kg/m^3)
c_ice = 2090  # Capacidad calorífica del hielo (J/kg*K)
c_water = 4186  # Capacidad calorífica del agua (J/kg*K)
k_ice = 1.6  # Conductividad térmica del hielo (W/m*K)
k_water = 0.6  # Conductividad térmica del agua (W/m*K)
h_fusion = 334000  # Calor latente de fusión (J/kg)

alpha_ice = k_ice / (rho_ice * c_ice)
alpha_water = k_water / (rho_water * c_water)

# Discretización del dominio
nx, ny = int(x_length / dx) + 1, int(y_length / dy) + 1
nt = int(total_time / dt)

# Campo de temperatura inicial (en Kelvin)
T = np.full((nx, ny), aKelvin(-10))

# Campo de entalpía inicial
H = np.where(T < aKelvin(0), rho_ice * c_ice * (T - aKelvin(0)), 0)  # Inicializar entalpía del hielo

# Función de condición de frontera
def boundary_condition_right(t):
    return aKelvin(-10 + (95 * min(t / 10, 1)))  # De -10°C a 85°C en 10 segundos

# Graficar la temperatura a lo largo de la línea central
def plot_temperature_line(T, t, x_length, y_length):
    middle_y = ny // 2
    temperature_profile = T[:, middle_y]
    plt.plot(np.linspace(0, x_length * 1000, len(temperature_profile)), aCelsius(temperature_profile), label=f"T = {t:.1f}s")

# Simulación
for n in range(nt):
    t = n * dt
    T_old = T.copy()

    # Actualización del campo de entalpía
    for i in range(1, nx - 1):
        for j in range(1, ny - 1):
            # Selección de propiedades térmicas
            if T_old[i, j] < aKelvin(0):  # Hielo
                alpha, rho, c = alpha_ice, rho_ice, c_ice
            else:  # Agua
                alpha, rho, c = alpha_water, rho_water, c_water

            # Derivadas espaciales
            d2T_dx2 = (T_old[i+1, j] - 2 * T_old[i, j] + T_old[i-1, j]) / dx**2
            d2T_dy2 = (T_old[i, j+1] - 2 * T_old[i, j] + T_old[i, j-1]) / dy**2

            # Actualización de entalpía
            H[i, j] += rho * c * alpha * (d2T_dx2 + d2T_dy2) * dt

            # Conversión de entalpía a temperatura
            if H[i, j] < 0:  # Hielo
                T[i, j] = H[i, j] / (rho_ice * c_ice) + aKelvin(0)
            elif H[i, j] > rho_water * h_fusion:  # Agua
                T[i, j] = (H[i, j] - rho_water * h_fusion) / (rho_water * c_water) + aKelvin(0)
            else:  # Cambio de fase (0°C)
                T[i, j] = aKelvin(0)

    # Aplicar condiciones de frontera
    T[0, :] = T[1, :]
    T[-1, :] = boundary_condition_right(t)
    T[:, 0] = T[:, 1]
    T[:, -1] = T[:, -2]

    # Graficar cada 50 segundos
    if n % int(50 / dt) == 0:
        plot_temperature_line(T, t, x_length, y_length)

# Ajustar gráfico
plt.xlabel("Distancia (mm)")
plt.ylabel("Temperatura (°C)")
plt.title("Distribución de temperatura con cambio de fase")
plt.legend()
plt.grid(True)
plt.show()
