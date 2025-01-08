import numpy as np
import matplotlib.pyplot as plt

# Parámetros físicos
alpha = 1e-2  # difusividad térmica en mm^2/s (aumentado para mayor propagación de calor)
L_f = 334000  # calor latente de fusión en J/kg
rho = 917  # densidad del hielo en kg/m^3
c_p = 2.1e3  # capacidad calorífica del hielo en J/kg°C
k = 2.1  # conductividad térmica en W/m°C

# Dimensiones del dominio
Lx = 10e-3  # largo en metros
Ly = 1e-3  # ancho en metros

# Número de puntos espaciales
Nx = 100  # número de puntos en el eje x
Ny = 10  # número de puntos en el eje y

# Discretización espacial
dx = Lx / (Nx - 1)
dy = Ly / (Ny - 1)

# Parámetros temporales
dt = 1e-3  # paso de tiempo en segundos (ajustado)
t_max = 100  # tiempo máximo de simulación en segundos
nt = int(t_max / dt)  # número de pasos de tiempo

# Inicialización de temperatura y fracción de fase
T = np.full((Nx, Ny), -10.0)  # temperatura inicial en °C
phi = np.zeros((Nx, Ny))  # fracción de fase, inicialmente todo hielo

# Condiciones de frontera
def aplicar_condiciones_de_borde(T, t):
    # Borde izquierdo, superior e inferior: aislados térmicamente (condición de Neumann)
    T[0, :] = T[1, :]  # frontera izquierda
    T[:, 0] = T[:, 1]  # frontera inferior
    T[:, -1] = T[:, -2]  # frontera superior
    
    # Borde derecho: rampa lineal de temperatura de -10°C a 85°C en 10s
    T[-1, :] = -10 + 95 * (t / 10)  # Rampa suave de -10°C a 85°C

# Método de diferencias finitas con control de fase
temperaturas_centro = []

for t in range(1, nt):
    T_old = T.copy()
    # Resolver la ecuación de conducción de calor
    for i in range(1, Nx-1):
        for j in range(1, Ny-1):
            # Actualización de temperatura
            d2Tdx2 = (T_old[i+1, j] - 2*T_old[i, j] + T_old[i-1, j]) / dx**2
            d2Tdy2 = (T_old[i, j+1] - 2*T_old[i, j] + T_old[i, j-1]) / dy**2
            T[i, j] = T_old[i, j] + alpha * dt * (d2Tdx2 + d2Tdy2)
    
    # Aplicar condiciones de frontera
    aplicar_condiciones_de_borde(T, t * dt)
    
    # Control de la fusión: no permitir que la temperatura supere los 0°C en la fracción de fase
    phi = np.where(T <= 0, 0, np.where(T >= 0, 1, phi))
    T = np.where(phi == 0, T, np.where(T < 0, 0, T))  # Mantener la temperatura en 0°C cuando se funde el hielo
    
    # Asegurarse de que no haya NaN o valores infinitos
    T = np.nan_to_num(T, nan=0.0, posinf=0.0, neginf=0.0)
    
    # Limitar la temperatura entre -10°C y 85°C
    T = np.clip(T, -10, 85)
    
    # Registrar temperaturas a lo largo de la línea central
    temperaturas_centro.append(T[Nx//2, Ny//2])  # Tomando el valor en la línea central (al centro)

    # Visualización de la evolución de la temperatura en el dominio cada 1000 pasos
    if t % 1000 == 0:
        plt.imshow(T, cmap='hot', interpolation='nearest')
        plt.colorbar(label='Temperatura (°C)')
        plt.title(f'Temperatura en t = {t*dt} s')
        plt.show()

# Graficar los resultados de la temperatura en el centro del dominio
plt.plot(np.linspace(0, t_max, nt), temperaturas_centro)
plt.xlabel('Tiempo (s)')
plt.ylabel('Temperatura en el centro (°C)')
plt.title('Evolución de la temperatura en el centro del dominio')
plt.grid(True)
plt.show()