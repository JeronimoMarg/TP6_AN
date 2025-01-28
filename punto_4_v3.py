import numpy as np
from numba import jit
import matplotlib.pyplot as plt

# --- Parámetros del problema ---
L = 21e-3  # Longitud total (10mm + 1mm + 10mm)
W = 1e-3   # Ancho de cada segmento
H = 11e-3  # Altura total (10mm + 1mm)

TEMP_INITIAL_ICE = -10
TEMP_INITIAL_WATER = 20
TEMP_MELT = 0
CALOR_LATENTE_FUSION = 334000
RHO_WATER = 1000
RHO_ICE = 918
C_WATER = 4186
C_ICE = 2090

points_per_mm = 8
Nx = int(L * 1000 * points_per_mm)
Ny = int(H * 1000 * points_per_mm)
Dx = L / Nx
Dy = H / Ny

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
def calcular_paso(temp, energy, mask, Dx, Dy, Dt, T_melt, L_fusion):
    Nx, Ny = temp.shape
    temp_new = np.copy(temp)
    energy_new = np.copy(energy)
    ice_fraction = np.ones_like(temp)

    for i in range(1, Nx - 1):
        for j in range(1, Ny - 1):
            if mask[i, j]:
                # Calcular energía disponible relativa al punto de fusión
                energia_disponible = energy[i, j]

                # Etapa 1: Calentamiento del hielo
                E_sensible_max = RHO_ICE * C_ICE * (TEMP_MELT - TEMP_INITIAL_ICE)
                E_latente_max = RHO_ICE * CALOR_LATENTE_FUSION

                if energia_disponible < E_sensible_max:
                    temp_new[i, j] = TEMP_INITIAL_ICE + (energia_disponible / (RHO_ICE * C_ICE))
                    ice_fraction[i, j] = 1.0
                elif energia_disponible < E_sensible_max + E_latente_max:
                    temp_new[i, j] = TEMP_MELT
                    ice_fraction[i, j] = 1.0 - (energia_disponible - E_sensible_max) / E_latente_max
                else:
                    temp_new[i, j] = TEMP_MELT + (energia_disponible - E_sensible_max - E_latente_max) / (RHO_WATER * C_WATER)
                    ice_fraction[i, j] = 0.0

                # Obtener propiedades térmicas según la fracción de hielo actual
                k = k_ice * ice_fraction[i, j] + k_water * (1 - ice_fraction[i, j])
                rho = RHO_ICE * ice_fraction[i, j] + RHO_WATER * (1 - ice_fraction[i, j])
                c = C_ICE * ice_fraction[i, j] + C_WATER * (1 - ice_fraction[i, j])

                # Actualizar energía con difusión térmica
                alpha = k / (rho * c)
                d2T_dx2 = (temp[i + 1, j] - 2 * temp[i, j] + temp[i - 1, j]) / Dx**2
                d2T_dy2 = (temp[i, j + 1] - 2 * temp[i, j] + temp[i, j - 1]) / Dy**2

                delta_energy = alpha * (d2T_dx2 + d2T_dy2) * Dt * rho * c

                energy_new[i, j] = energy[i, j] + delta_energy

    return temp_new, energy_new, ice_fraction

def simular_sin_flujo():
    mask = np.zeros((Nx, Ny), dtype=bool)

    mm_to_points = points_per_mm
    brazo_largo = 10 * mm_to_points
    brazo_ancho = 1 * mm_to_points
    corazon = 1 * mm_to_points

    centro_x = Nx // 2
    centro_y = Ny // 2 + brazo_largo // 2

    corazon_left = centro_x - corazon // 2
    corazon_right = centro_x + corazon // 2
    corazon_top = centro_y + corazon // 2
    corazon_bottom = centro_y - corazon // 2

    mask[corazon_left-brazo_largo:corazon_left, centro_y-brazo_ancho//2:centro_y+brazo_ancho//2] = True
    mask[corazon_right:corazon_right+brazo_largo, centro_y-brazo_ancho//2:centro_y+brazo_ancho//2] = True
    mask[centro_x-brazo_ancho//2:centro_x+brazo_ancho//2, centro_y-brazo_largo-corazon//2:centro_y-corazon//2] = True
    mask[corazon_left:corazon_right, corazon_bottom:corazon_top] = True

    temp = np.full((Nx, Ny), TEMP_INITIAL_WATER)
    energy = np.zeros((Nx, Ny))

    temperaturas_promedio = []

    for i in range(Nx):
        for j in range(Ny):
            if mask[i, j]:
                if corazon_left <= i < corazon_right and corazon_bottom <= j < corazon_top:
                    temp[i, j] = TEMP_INITIAL_ICE
                    energy[i, j] = RHO_ICE * C_ICE * (TEMP_INITIAL_ICE - TEMP_MELT)
                else:
                    temp[i, j] = TEMP_INITIAL_WATER
                    energy[i, j] = RHO_WATER * C_WATER * TEMP_INITIAL_WATER

    tiempo_demora = None

    for step in range(steps):
        temp, energy, ice_fraction = calcular_paso(
            temp, energy, mask, Dx, Dy, Dt, TEMP_MELT, CALOR_LATENTE_FUSION
        )

        fraccion_hielo = np.mean(ice_fraction[corazon_left:corazon_right, corazon_bottom:corazon_top])

        if fraccion_hielo <= 0.5 and tiempo_demora is None:
            tiempo_demora = step * Dt

        temp_corazon = temp[corazon_left:corazon_right, corazon_bottom:corazon_top]
        temp_promedio_corazon = np.mean(temp_corazon)
        temperaturas_promedio.append(temp_promedio_corazon)
        if step == 0 or step == 1:
            print(temp_promedio_corazon)

    return temp, ice_fraction, tiempo_demora, temperaturas_promedio

temp_final, ice_fraction_final, tiempo_demora, temperaturas_promedio = simular_sin_flujo()

if tiempo_demora is not None:
    print(f"El tiempo de demora para alcanzar el 50% de la sección útil es: {tiempo_demora:.2f} segundos")
else:
    print("No se alcanzó el 50% de la sección útil durante la simulación.")

# Graficar la evolución de la temperatura promedio del corazón de hielo
plt.figure(figsize=(8, 6))
tiempos = np.arange(len(temperaturas_promedio)) * Dt
plt.plot(tiempos, temperaturas_promedio, label='Temperatura promedio del corazón de hielo')
plt.xlabel('Tiempo (s)')
plt.ylabel('Temperatura promedio (°C)')
plt.title('Evolución de la temperatura promedio del corazón de hielo')
plt.legend()
plt.grid()
plt.tight_layout()
plt.show()