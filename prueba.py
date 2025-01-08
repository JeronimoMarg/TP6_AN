import numpy as np
import matplotlib.pyplot as plt

# Parámetros del problema
Lx = 10e-3  # Largo del dominio (10 mm)
Ly = 1e-3   # Ancho del dominio (1 mm)
Nx = 100    # Número de nodos en x
Ny = 10     # Número de nodos en y
dx = Lx / (Nx - 1)
dy = Ly / (Ny - 1)
alpha = 1.14e-6  # Difusividad térmica del hielo (m^2/s)
dt = 0.1  # Paso de tiempo (s)
total_time = 100  # Tiempo total de simulación (s)
Nt = int(total_time / dt)  # Número de pasos de tiempo

# Condiciones iniciales
T = np.full((Ny, Nx), -10.0)  # Temperatura inicial en todo el dominio (-10°C)
f = np.zeros((Ny, Nx))  # Fracción de fase inicial (todo hielo)

# Función para actualizar la temperatura en el borde derecho
def update_right_boundary(T, t):
    if t <= 10:
        T[:, -1] = -10 + (95 / 10) * t  # Rampa lineal de -10°C a 85°C en 10 s
    else:
        T[:, -1] = 85  # Mantener 85°C después de 10 s

# Simulación
for n in range(Nt):
    T_new = T.copy()
    for i in range(1, Ny-1):
        for j in range(1, Nx-1):
            T_new[i, j] = T[i, j] + alpha * dt * (
                (T[i+1, j] - 2*T[i, j] + T[i-1, j]) / dx**2 +
                (T[i, j+1] - 2*T[i, j] + T[i, j-1]) / dy**2
            )
    update_right_boundary(T_new, n*dt)
    T = T_new

# Graficar la temperatura en el dominio
plt.imshow(T, extent=[0, Lx, 0, Ly], origin='lower', aspect='auto', cmap='coolwarm')
plt.colorbar(label='Temperatura (°C)')
plt.xlabel('Largo del dominio (m)')
plt.ylabel('Ancho del dominio (m)')
plt.title('Distribución de temperatura en el dominio')
plt.show()