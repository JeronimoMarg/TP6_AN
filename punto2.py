import numpy as np
import matplotlib.pyplot as plt

# Parámetros físicos
L = 10e-3  # Largo del dominio en metros (10 mm)
H = 1e-3   # Altura del dominio en metros (1 mm)
T_izq = -10.0  # Borde izquierdo, superior e inferior: aislados (no importa la temperatura)
T_derecha_ini = -10.0  # Temperatura inicial en el borde derecho (en °C)
T_derecha_fin = 85.0   # Temperatura final en el borde derecho después de 10 segundos

T_inicial = -10.0  # Temperatura inicial en todo el dominio (en °C)
k = 2.2  # Conductividad térmica del hielo (W/m·K)
c_p = 2000  # Capacidad calorífica específica (J/kg·K)
rho = 920  # Densidad del hielo (kg/m^3)
alfa = k / (rho * c_p)  # Difusividad térmica (m^2/s)

# Parámetros numéricos
Nx = 100  # Número de puntos en el eje x
Ny = 10   # Número de puntos en el eje y
dx = L / (Nx - 1)  # Tamaño del paso espacial en x
dy = H / (Ny - 1)  # Tamaño del paso espacial en y
dt = 0.25 * min(dx, dy) ** 2 / alfa  # Paso temporal (criterio de estabilidad de Courant)
T_total = 100  # Tiempo total de simulación (en segundos)
n_steps = int(T_total / dt)  # Número de pasos temporales

# Inicialización de la malla
T = np.full((Ny, Nx), T_inicial)  # Temperatura inicial en el dominio

# Condiciones de frontera
# Bordes superior e inferior: aislados (gradiente de temperatura = 0)
# Borde izquierdo: aislado (gradiente de temperatura = 0)
def actualizar_condiciones_de_frontera(T, t):
    # Borde derecho: rampa lineal de temperatura
    if t <= 10:
        T[:, -1] = T_derecha_ini + (T_derecha_fin - T_derecha_ini) * (t / 10)
    else:
        T[:, -1] = T_derecha_fin

# Simulación por diferencias finitas
for step in range(n_steps):
    t = step * dt  # Tiempo actual
    T_old = T.copy()

    # Actualizar el interior del dominio (difusión del calor)
    for i in range(1, Ny - 1):
        for j in range(1, Nx - 1):
            T[i, j] = T_old[i, j] + alfa * dt * (
                (T_old[i + 1, j] - 2 * T_old[i, j] + T_old[i - 1, j]) / dy ** 2 +
                (T_old[i, j + 1] - 2 * T_old[i, j] + T_old[i, j - 1]) / dx ** 2
            )

    # Actualizar condiciones de frontera
    actualizar_condiciones_de_frontera(T, t)

# Gráfica de la temperatura en la línea central
x = np.linspace(0, L, Nx)
plt.plot(x, T[Ny // 2, :], label=f"t = {T_total:.1f} s")
plt.xlabel("Largo del dominio (m)")
plt.ylabel("Temperatura (°C)")
plt.title("Distribución de temperatura en la línea central")
plt.legend()
plt.grid()
plt.show()