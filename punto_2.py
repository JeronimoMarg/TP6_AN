import numpy as np
import matplotlib.pyplot as plt

''' Este codigo realiza el punto 2, graficando la temperatura a traves del dominio en tiempo de simulacion. '''

def aKelvin(T):
    return T + 273.15

def aCelsius(T):
    return T - 273.15

dx = dy = 0.00025  # Paso espacial (0.25 mm)
dt = 0.01  # Paso temporal reducido para estabilidad
total_time = 100  # Tiempo total de simulación (s)
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

temperaturas_promedio = []
fraccion_hielo_historial = []

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

    # se manejan los graficos para la temperatura promedio y fraccion de fase
    temperaturas_promedio.append(np.mean(aCelsius(T)))
    total_temps = nx * ny
    total_temps_bajo0 = np.sum(T < aKelvin(0))
    fraccion = total_temps_bajo0 / total_temps
    fraccion_hielo_historial.append(fraccion)

    # Graficar cada 50 segundos
    if n % int(10 / dt) == 0:
        plot_temperature_line(T, t, x_length, y_length)

plt.xlabel("Distancia (mm)")
plt.ylabel("Temperatura (°C)")
plt.title("Distribución de temperatura con cambio de fase")
plt.legend()
plt.grid(True)
plt.show()
plt.figure(figsize=(8, 6))
tiempos = np.arange(len(temperaturas_promedio)) * dt
plt.plot(tiempos, temperaturas_promedio, label='Temperatura promedio.')
plt.xlabel('Tiempo (s)')
plt.ylabel('Temperatura promedio (°C)')
plt.title('Evolucion de la temperatura promedio del dominio')
plt.legend()
plt.grid()
plt.tight_layout()
plt.show()
plt.figure(figsize=(8, 6))
tiempos = np.arange(len(fraccion_hielo_historial)) * dt
plt.plot(tiempos, fraccion_hielo_historial, label='Fraccion de hielo.')
plt.xlabel('Tiempo (s)')
plt.ylabel('Fraccion de hielo')
plt.title('Evolución de la fraccion de hielo del corazon de hielo')
plt.legend()
plt.grid()
plt.tight_layout()
plt.show()