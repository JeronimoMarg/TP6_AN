import numpy as np
from numba import jit
import matplotlib.pyplot as plt

# Definicion del dominio
L = 21e-3  # Longitud total (10mm + 1mm + 10mm)
W = 1e-3   # Ancho de cada segmento
H = 11e-3  # Altura total (10mm + 1mm)

TEMP_INITIAL_ICE = -30
TEMP_INITIAL_ICE /= 2   # dividimos por 2 porque se buguea sino
TEMP_INITIAL_WATER = 20
TEMP_MELT = 0
CALOR_LATENTE_FUSION = 334000
RHO_WATER = 1000
RHO_ICE = 918
C_WATER = 4186
C_ICE = 2090

VEL = 0.1

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
TOTAL_TIME = 30 
steps = int(TOTAL_TIME / Dt)

# se obtienen las propiedades termicas del material dependiendo de la temp y fraccion de fase.
@jit(nopython=True)
def obtener_propiedades_termicas(T, ice_fraction=None):
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

# se aplica las condiciones de bordes de aislamiento termico
@jit(nopython=True)
def aplicar_condiciones_borde_aislamiento(temp, mask):
    Nx, Ny = temp.shape
    for i in range(Nx):
        for j in range(Ny):
            if mask[i, j]:
                if i == 0:
                    temp[i, j] = temp[i + 1, j]
                elif i == Nx - 1:
                    temp[i, j] = temp[i - 1, j]
                if j == 0:
                    temp[i, j] = temp[i, j + 1]
                elif j == Ny - 1:
                    temp[i, j] = temp[i, j - 1]

    return temp

# funcion que se utiliza para actualizar la temperatura y la energia en cada paso CON ADVECCION
@jit(nopython=True)
def calcular_paso(temp, energy, mask, Dx, Dy, Dt, T_melt, L_fusion, velocidad_flujo):
    Nx, Ny = temp.shape
    temp_new = np.copy(temp)
    energy_new = np.copy(energy)
    ice_fraction = np.ones_like(temp)

    for i in range(1, Nx-1):
        for j in range(1, Ny-1):
            if mask[i, j]:
                # Calcular energía disponible relativa al punto de fusión
                energia_disponible = energy[i, j]

                # Etapa 1: Calentamiento del hielo (desde -30°C a 0°C)
                E_sensible_max = RHO_ICE * C_ICE * (TEMP_MELT - TEMP_INITIAL_ICE)

                # Etapa 2: Fusión (calor latente)
                E_latente_max = RHO_ICE * CALOR_LATENTE_FUSION

                if energia_disponible < E_sensible_max:
                    temp_new[i,j] = TEMP_INITIAL_ICE + (energia_disponible / (RHO_ICE * C_ICE))
                    ice_fraction[i,j] = 1.0
                elif energia_disponible < E_sensible_max + E_latente_max:
                    temp_new[i,j] = TEMP_MELT
                    ice_fraction[i,j] = 1.0 - (energia_disponible - E_sensible_max) / E_latente_max
                else:
                    temp_new[i,j] = TEMP_MELT + (energia_disponible - E_sensible_max - E_latente_max) / (RHO_WATER * C_WATER)
                    ice_fraction[i,j] = 0.0

                # Obtener propiedades térmicas según la fracción de hielo actual
                k = k_ice * ice_fraction[i,j] + k_water * (1 - ice_fraction[i,j])
                rho = RHO_ICE * ice_fraction[i,j] + RHO_WATER * (1 - ice_fraction[i,j])
                c = C_ICE * ice_fraction[i,j] + C_WATER * (1 - ice_fraction[i,j])

                # Actualizar energía usando la ecuación del calor (difusión + advección)
                alpha = k / (rho * c)
                d2T_dx2 = (temp[i+1,j] - 2*temp[i,j] + temp[i-1,j]) / Dx**2
                d2T_dy2 = (temp[i,j+1] - 2*temp[i,j] + temp[i,j-1]) / Dy**2

                # metodo de upwind
                if velocidad_flujo > 0:
                    dT_dx = (temp[i,j] - temp[i-1,j]) / Dx
                    dT_dy = (temp[i,j] - temp[i,j-1]) / Dy
                else:
                    dT_dx = (temp[i+1,j] - temp[i,j]) / Dx
                    dT_dy = (temp[i,j+1] - temp[i,j]) / Dy
                adveccion = velocidad_flujo * (dT_dx + dT_dy)

                delta_energy = alpha * (d2T_dx2 + d2T_dy2) * Dt * rho * c - adveccion * Dt
                energy_new[i,j] = energy[i,j] + delta_energy

    temp_new = aplicar_condiciones_borde_aislamiento(temp_new, mask)
    return temp_new, energy_new, ice_fraction

# funcion que inicializa la simulacion y realiza el cambio de fase
def simular_cambio_fase_con_flujo():
    mask = np.zeros((Nx, Ny), dtype=bool)
    mm_to_points = points_per_mm
    brazo_largo = 10 * mm_to_points  # 10mm
    brazo_ancho = 1 * mm_to_points   # 1mm
    corazon = 1 * mm_to_points       # 1mm x 1mm
    centro_x = Nx // 2
    centro_y = Ny // 2 + brazo_largo // 2

    # corazon de hielo
    corazon_left = centro_x - corazon // 2
    corazon_right = centro_x + corazon // 2
    corazon_top = centro_y + corazon // 2
    corazon_bottom = centro_y - corazon // 2

    mask[corazon_left-brazo_largo:corazon_left, 
         centro_y-brazo_ancho//2:centro_y+brazo_ancho//2] = True
    
    mask[corazon_right:corazon_right+brazo_largo, 
         centro_y-brazo_ancho//2:centro_y+brazo_ancho//2] = True
    
    mask[centro_x-brazo_ancho//2:centro_x+brazo_ancho//2, 
         centro_y-brazo_largo-corazon//2:centro_y-corazon//2] = True

    mask[corazon_left:corazon_right, 
         corazon_bottom:corazon_top] = True

    # inicializar temperatura y entalpia
    temp = np.full((Nx, Ny), TEMP_INITIAL_WATER)
    energy = np.zeros((Nx, Ny))

    for i in range(Nx):
        for j in range(Ny):
            if mask[i, j]:
                # El corazón es de hielo
                if (corazon_left <= i < corazon_right and 
                    corazon_bottom <= j < corazon_top):
                    temp[i, j] = TEMP_INITIAL_ICE
                    k, rho, c = obtener_propiedades_termicas(TEMP_INITIAL_ICE)
                    energy[i,j] = RHO_ICE * C_ICE * (TEMP_INITIAL_ICE - TEMP_MELT)
                else:
                    # Los brazos y la columna son de agua
                    temp[i, j] = TEMP_INITIAL_WATER
                    k, rho, c = obtener_propiedades_termicas(TEMP_INITIAL_WATER)
                    energy[i, j] = rho * c * TEMP_INITIAL_WATER

    # puntos de entrada y salida
    entrada_izquierda = (corazon_left - brazo_largo, centro_y)
    entrada_derecha = (corazon_right + brazo_largo, centro_y)
    salida = (centro_x, centro_y - brazo_largo - corazon // 2)

    # velocidad de flujo
    velocidad_flujo = VEL * mm_to_points  # 0.1 mm/s
    velocidad_flujo *= Dt / Dx

    # buscar el tiempo de demora
    tiempo_demora = None
    temperaturas_promedio = []
    fraccion_hielo_historial = []

    fraccion_hielo = 1.0
    fraccion_hielo_historial.append(fraccion_hielo)
    temp_corazon = temp[corazon_left:corazon_right, corazon_bottom:corazon_top]
    temp_promedio_corazon = np.mean(temp_corazon)
    temperaturas_promedio.append(temp_promedio_corazon)
    
    temp_masked = np.ma.masked_array(temp, ~mask)

    for step in range(steps):
        # flujo de entrada de agua
        temp[entrada_izquierda[0]:entrada_izquierda[0]+1, entrada_izquierda[1]:entrada_izquierda[1]+1] = TEMP_INITIAL_WATER
        temp[entrada_derecha[0]:entrada_derecha[0]+1, entrada_derecha[1]:entrada_derecha[1]+1] = TEMP_INITIAL_WATER

        temp, energy, ice_fraction = calcular_paso(temp, energy, mask, Dx, Dy, Dt, TEMP_MELT, CALOR_LATENTE_FUSION, velocidad_flujo)

        temp[salida[0]:salida[0]+1, salida[1]:salida[1]+1] = TEMP_INITIAL_WATER
        
        fraccion_hielo = np.mean(ice_fraction[corazon_left:corazon_right, corazon_bottom:corazon_top])
        
        #CALCULO DE FRACCION DE HIELO SEGUN TEMPERATURA
        cant_temps = points_per_mm ** 2
        temp_bajo_0 = np.sum(temp[corazon_left:corazon_right, corazon_bottom:corazon_top] <= TEMP_MELT)
        fraccion_hielo_aux = temp_bajo_0 / cant_temps

        fraccion_hielo_historial.append(fraccion_hielo)

        #TEMPERATURAS PROMEDIO EN EL CORAZON
        temp_corazon = temp[corazon_left:corazon_right, corazon_bottom:corazon_top]
        temp_promedio_corazon = np.mean(temp_corazon)
        temperaturas_promedio.append(temp_promedio_corazon)

        # condicion de parada
        if fraccion_hielo <= 0.5 and tiempo_demora is None:
            tiempo_demora = step * Dt

        if 0: # no se usa en este codigo
            temp_masked = np.ma.masked_array(temp, ~mask)

            plt.figure(figsize=(10, 8))
            
            plt.clf()
            
            im = plt.imshow(temp_masked.T, cmap='coolwarm', interpolation='nearest',
                          vmin=TEMP_INITIAL_ICE, vmax=TEMP_INITIAL_WATER, extent=[0, L, 0, H*-1])
            plt.colorbar(im, label='Temperatura (°C)')
            plt.title(f'Distribución de Temperatura - t = {step*Dt:.2f} s\nFracción de hielo en el corazón: {fraccion_hielo:.2%}')
            plt.xlabel('Posición X (mm)')
            plt.ylabel('Posición Y (mm)')
            plt.gca().invert_yaxis()
            plt.tight_layout()
            plt.show()

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

    # Graficar la evolución de la fraccion de hielo
    plt.figure(figsize=(8, 6))
    tiempos = np.arange(len(fraccion_hielo_historial)) * Dt
    plt.plot(tiempos, fraccion_hielo_historial, label='Fraccion de hielo del corazon de hielo')
    plt.xlabel('Tiempo (s)')
    plt.ylabel('Fraccion de hielo')
    plt.title('Evolución de la fraccion de hielo del corazon de hielo')
    plt.legend()
    plt.grid()
    plt.tight_layout()
    plt.show()

    return temp, ice_fraction, tiempo_demora

print(f"El dt es: {Dt}")
print(f"El step es: {steps}")
temp_final, ice_fraction_final, tiempo_demora = simular_cambio_fase_con_flujo()

if tiempo_demora is not None:
    print(f"El tiempo de demora para alcanzar el 50% de la sección útil es: {tiempo_demora:.2f} segundos")
else:
    print("No se alcanzó el 50% de la sección útil durante la simulación.")