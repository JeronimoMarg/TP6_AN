import numpy as np
import matplotlib.pyplot as plt

# Constantes físicas del hielo y parámetros iniciales
k = 2.2  # Conductividad térmica (W/m·K)
c_p = 2100  # Calor específico (J/kg·K)
rho = 917  # Densidad (kg/m^3)
L_f = 334000  # Calor latente de fusión (J/kg)

# Dimensiones de la lámina
total_length = 0.01  # Longitud (m)
total_width = 0.001  # Ancho (m)
dx = 0.0001  # Paso espacial (m)
Nx = int(total_length / dx)  # Número de nodos espaciales

# Condiciones de frontera
T_initial = -10  # Temperatura inicial (°C)
T_right_start = -10  # Temperatura inicial en el borde derecho (°C)
T_right_end = 85  # Temperatura final en el borde derecho (°C)
heating_time = 10  # Tiempo de calentamiento en el borde derecho (s)

# Propiedades temporales
dt = 0.1  # Paso temporal (s)
time_limit = 300  # Tiempo máximo de simulación (s)

# Inicialización de variables
T = np.ones(Nx) * T_initial  # Perfil de temperatura inicial (°C)
x = np.linspace(0, total_length, Nx)  # Eje espacial

# Calcular la masa de cada elemento de la lámina
volume_per_node = total_width * dx * 1  # Volumen de cada nodo (m^3)
mass_per_node = rho * volume_per_node  # Masa de cada nodo (kg)

# Variables para el cálculo
time = 0  # Tiempo inicial
melted = np.zeros(Nx, dtype=bool)  # Estado de fusión de cada nodo

# Función para calcular la temperatura del borde derecho en función del tiempo
def T_right(t):
    if t <= heating_time:
        return T_right_start + (T_right_end - T_right_start) * (t / heating_time)
    else:
        return T_right_end

# Simulación por diferencias finitas
while time < time_limit:
    # Crear una copia para actualizar la temperatura
    T_new = T.copy()

    # Aplicar condiciones de frontera
    T_new[-1] = T_right(time)  # Borde derecho

    # Iterar sobre los nodos internos
    for i in range(1, Nx - 1):
        if not melted[i]:  # Solo actualizar nodos que no se han derretido
            d2T_dx2 = (T[i + 1] - 2 * T[i] + T[i - 1]) / dx**2
            dT = (k / (rho * c_p)) * d2T_dx2 * dt
            T_new[i] += dT

            # Verificar si el nodo alcanza 0 °C
            if T_new[i] >= 0:
                # Añadir el calor latente necesario para derretir
                energy_to_melt = L_f * mass_per_node
                heat_absorbed = (T_new[i] - 0) * c_p * mass_per_node

                if heat_absorbed >= energy_to_melt:
                    melted[i] = True  # Nodo completamente derretido

    # Actualizar el perfil de temperatura
    T = T_new
    time += dt

    # Verificar si todos los nodos se han derretido
    if np.all(melted):
        break

# Imprimir el tiempo necesario para derretir
print(f"Tiempo total para derretir la lámina: {time:.2f} segundos")

# Visualización final del perfil de temperatura
plt.plot(x, T, label="Temperatura final")
plt.xlabel("Posición (m)")
plt.ylabel("Temperatura (°C)")
plt.title("Perfil de temperatura al final de la simulación")
plt.legend()
plt.grid()
plt.show()
