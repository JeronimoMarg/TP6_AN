import numpy as np
from numba import jit
import matplotlib.pyplot as plt

# --- Parámetros del problema ---
L_horizontal = 20e-3  # Longitud horizontal de la parte superior de la "T" (m)
H_vertical = 10e-3    # Altura de la parte vertical de la "T" (m)
W_vertical = 1e-3    # Ancho de la parte vertical de la "T" (m)
W_horizontal = 1e-3   # Espesor de la parte horizontal de la "T" (m)

TEMP_INITIAL_ICE = -30
TEMP_INITIAL_WATER = 20
TEMP_MELT = 0
CALOR_LATENTE_FUSION = 333000
RHO_WATER = 1000
RHO_ICE = 917
C_WATER = 4186
C_ICE = 2090

points_per_mm = 8
Nx = int(L_horizontal * 1000 * points_per_mm)
Ny = int(H_vertical * 1000 * points_per_mm)
vertical_width_points = int(W_vertical * 1000 * points_per_mm)

Dx = L_horizontal / Nx
Dy = H_vertical / Ny

k_water = 0.58
alpha_water = k_water / (RHO_WATER * C_WATER)
k_ice = 2.24
alpha_ice = k_ice / (RHO_ICE * C_ICE)
alpha_max = max(alpha_water, alpha_ice)
safety_factor = 0.1
Dt = safety_factor * min(Dx, Dy) ** 2 / alpha_max
TOTAL_TIME = 50  # Tiempo total en segundos
steps = int(TOTAL_TIME / Dt)

@jit(nopython=True)
def obtener_propiedades_termicas(T, ice_fraction=None):
    """Función para obtener propiedades térmicas dependiendo de la temperatura y fracción de hielo."""
    if ice_fraction is None:
        if T < TEMP_MELT:
            return k_ice, RHO_ICE, C_ICE
        else:
            return k_water, RHO_WATER, C_WATER
    else:
        k = k_ice * ice_fraction + k_water * (1 - ice_fraction)
        rho = RHO_ICE * ice_fraction + RHO_WATER * (1 - ice_fraction)
        c = C_ICE * ice_fraction + C_WATER * (1 - ice_fraction)
        return k, rho, c

@jit(nopython=True)
def calcular_paso(temp, energy, mask, Dx, Dy, Dt, T_melt, L_fusion):
    """Función para actualizar las temperaturas y energías en cada paso de tiempo."""
    Nx, Ny = temp.shape
    temp_new = np.copy(temp)
    energy_new = np.copy(energy)
    ice_fraction = np.ones_like(temp)

    E_ice_max = RHO_ICE * C_ICE * T_melt
    E_water_min = RHO_WATER * (C_WATER * T_melt + L_fusion)

    for i in range(1, Nx-1):
        for j in range(1, Ny-1):
            if mask[i, j]:
                # Calculando la fracción de hielo
                if energy[i, j] >= E_water_min:
                    ice_fraction[i, j] = 0.0  # Agua
                elif energy[i, j] <= E_ice_max:
                    ice_fraction[i, j] = 1.0  # Hielo
                else:
                    ice_fraction[i, j] = 1.0 - (energy[i, j] - E_ice_max) / (E_water_min - E_ice_max)

                # Obtener propiedades térmicas
                k, rho, c = obtener_propiedades_termicas(temp[i, j], ice_fraction[i, j])

                # Derivadas espaciales para el cálculo de la energía
                d2T_dx2 = (temp[i+1, j] - 2*temp[i, j] + temp[i-1, j]) / Dx**2
                d2T_dy2 = (temp[i, j+1] - 2*temp[i, j] + temp[i, j-1]) / Dy**2

                # Cálculo de la energía
                energy_new[i, j] = energy[i, j] + (k * (d2T_dx2 + d2T_dy2) * Dt)

                # Actualización de la temperatura
                temp_new[i, j] = energy_new[i, j] / (rho * c)

    return temp_new, energy_new, ice_fraction

def simular_cambio_fase_T():
    """Función principal para simular el cambio de fase en un dominio en forma de 'T'."""
    # Crear máscara para la forma de la "T"
    mask = np.zeros((Nx, Ny), dtype=bool)

    # Parte vertical de la "T"
    vertical_x_start = Nx // 2 - vertical_width_points // 2
    vertical_x_end = Nx // 2 + vertical_width_points // 2
    mask[vertical_x_start:vertical_x_end, :] = True

    # Parte horizontal de la "T"
    horizontal_y_start = Ny - int(W_horizontal * 1000 * points_per_mm)
    mask[:, horizontal_y_start:] = True

    # Inicializar condiciones de temperatura y energía
    temp = np.full((Nx, Ny), TEMP_INITIAL_WATER)
    energy = np.zeros((Nx, Ny))

    # Región de hielo en la parte superior central
    ice_x_start = Nx // 2 - vertical_width_points // 2
    ice_x_end = Nx // 2 + vertical_width_points // 2
    temp[ice_x_start:ice_x_end, horizontal_y_start:] = TEMP_INITIAL_ICE

    for i in range(Nx):
        for j in range(Ny):
            if mask[i, j]:
                if ice_x_start <= i < ice_x_end and j >= horizontal_y_start:
                    k, rho, c = obtener_propiedades_termicas(TEMP_INITIAL_ICE)
                    energy[i, j] = rho * c * TEMP_INITIAL_ICE  # Energía inicial en hielo
                else:
                    k, rho, c = obtener_propiedades_termicas(TEMP_INITIAL_WATER)
                    energy[i, j] = rho * c * TEMP_INITIAL_WATER  # Energía inicial en agua

    # Simulación paso a paso
    tiempo_demora = None
    for step in range(steps):
        temp, energy, ice_fraction = calcular_paso(temp, energy, mask, Dx, Dy, Dt, TEMP_MELT, CALOR_LATENTE_FUSION)

        # Medir la fracción de hielo en la región
        fraccion_hielo = np.mean(ice_fraction[ice_x_start:ice_x_end, horizontal_y_start:])

        # Verificar si se alcanzó el 50% de fusión
        if fraccion_hielo <= 0.5 and tiempo_demora is None:
            tiempo_demora = step * Dt

        # Mostrar la distribución de temperatura cada 10 pasos
        if step % 10000 == 0:
            plt.imshow(temp.T, cmap='hot', interpolation='nearest', origin='lower', extent=[0, L_horizontal, 0, H_vertical])
            plt.colorbar()
            plt.title(f'Temperature Distribution at Step {step}')
            plt.xlabel('X (m)')
            plt.ylabel('Y (m)')
            plt.show()

    return temp, ice_fraction, tiempo_demora

# Ejecutar la simulación para el dominio en forma de "T"
temp_final, ice_fraction_final, tiempo_demora = simular_cambio_fase_T()

# Mostrar el tiempo de demora
if tiempo_demora is not None:
    print(f"El tiempo de demora para alcanzar el 50% de la sección útil es: {tiempo_demora:.2f} segundos")
else:
    print("No se alcanzó el 50% de la sección útil durante la simulación.")
