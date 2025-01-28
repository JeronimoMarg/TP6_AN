import numpy as np
from numba import jit
import matplotlib.pyplot as plt

# --- Parámetros del problema ---
L = 21e-3  # Longitud total (10mm + 1mm + 10mm)
W = 1e-3   # Ancho de cada segmento
H = 11e-3  # Altura total (10mm + 1mm)

temperatura_inicial_hielo = -30
TEMP_INITIAL_ICE = temperatura_inicial_hielo / 2 #pasa algo raro pero el codigo siempre multiplica la temeperatura inicial * 2 en la primera iteracion
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
Nw = int(W * 1000 * points_per_mm)
Dx = L / Nx
Dy = H / Ny

k_water = 0.58
alpha_water = k_water / (RHO_WATER * C_WATER)
k_ice = 2.24
alpha_ice = k_ice / (RHO_ICE * C_ICE)
alpha_max = max(alpha_water, alpha_ice)

safety_factor = 0.1
Dt = safety_factor * min(Dx, Dy) ** 2 / alpha_max
TOTAL_TIME = 10  # Tiempo total en segundos
steps = int(TOTAL_TIME / Dt)

# Velocidad de flujo de agua en mm/s
flow_velocity = 0.1  # 0.1 mm/s
flow_velocity_mps = flow_velocity * 1e-3  # Convertir a m/s

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
def aplicar_condiciones_borde_aislamiento(temp, mask):
    """
    Aplica la condición de borde de aislamiento térmico (flujo de calor cero) en los bordes de la forma de "T".
    """
    Nx, Ny = temp.shape

    # Aplicar condición de flujo de calor cero en los bordes de la forma de "T"
    for i in range(Nx):
        for j in range(Ny):
            if mask[i, j]:  # Solo aplicamos la condición en los puntos dentro de la forma de "T"
                # Borde izquierdo
                if i == 0:
                    temp[i, j] = temp[i + 1, j]
                # Borde derecho
                elif i == Nx - 1:
                    temp[i, j] = temp[i - 1, j]
                # Borde inferior
                if j == 0:
                    temp[i, j] = temp[i, j + 1]
                # Borde superior
                elif j == Ny - 1:
                    temp[i, j] = temp[i, j - 1]

    return temp

@jit(nopython=True)
def calcular_paso(temp, energy, mask, Dx, Dy, Dt, T_melt, L_fusion):
    Nx, Ny = temp.shape
    temp_new = np.copy(temp)
    energy_new = np.copy(energy)
    ice_fraction = np.ones_like(temp)

    for i in range(1, Nx-1):
        for j in range(1, Ny-1):
            if mask[i, j]:
                # Calcular energía disponible relativa al punto de fusión
                energia_disponible = energy[i, j]  # Ya está ajustada en la inicialización

                # Etapa 1: Calentamiento del hielo (desde -30°C a 0°C)
                E_sensible_max = RHO_ICE * C_ICE * (TEMP_MELT - TEMP_INITIAL_ICE)  # 30°C * capacidad calorífica

                # Etapa 2: Fusión (calor latente)
                E_latente_max = RHO_ICE * CALOR_LATENTE_FUSION  # Energía para fundir todo el hielo

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

                # Actualizar energía usando la ecuación del calor (difusión)
                alpha = k / (rho * c)
                d2T_dx2 = (temp[i+1,j] - 2*temp[i,j] + temp[i-1,j]) / Dx**2
                d2T_dy2 = (temp[i,j+1] - 2*temp[i,j] + temp[i,j-1]) / Dy**2
                delta_energy = alpha * (d2T_dx2 + d2T_dy2) * Dt * rho * c

                '''
                # Advección de calor debido al flujo de agua
                if i < Nx // 2:  # Flujo de entrada por el brazo izquierdo
                    delta_energy += flow_velocity_mps * (temp[i+1,j] - temp[i,j]) / Dx * Dt * rho * c
                elif i > Nx // 2:  # Flujo de salida por el brazo derecho
                    delta_energy += flow_velocity_mps * (temp[i-1,j] - temp[i,j]) / Dx * Dt * rho * c

                energy_new[i,j] = energy[i,j] + delta_energy
                '''
                # Término de convección (flujo horizontal de entrada)
                term_conveccion = -flow_velocity_mps * (temp[i, j] - temp[i - 1, j]) / Dx

                delta_energy = (alpha * (d2T_dx2 + d2T_dy2) + term_conveccion) * Dt * rho * c

                energy_new[i, j] = energy[i, j] + delta_energy

    # Aplicar condiciones de borde
    temp_new = aplicar_condiciones_borde_aislamiento(temp_new, mask)
    return temp_new, energy_new, ice_fraction

def simular_cambio_fase():
    """Función principal para simular el cambio de fase en el dominio con forma de T."""
    mask = np.zeros((Nx, Ny), dtype=bool)

    # Convertir dimensiones físicas a puntos de la malla
    mm_to_points = points_per_mm
    
    # Dimensiones en puntos
    brazo_largo = 10 * mm_to_points  # 10mm
    brazo_ancho = 1 * mm_to_points   # 1mm
    corazon = 1 * mm_to_points       # 1mm x 1mm

    # Posiciones centrales
    centro_x = Nx // 2
    centro_y = Ny // 2 + brazo_largo // 2  # Ajustado para mover el centro hacia arriba

    # Definir el corazón de hielo (cuadrado central)
    corazon_left = centro_x - corazon // 2
    corazon_right = centro_x + corazon // 2
    corazon_top = centro_y + corazon // 2
    corazon_bottom = centro_y - corazon // 2

    # Definir los brazos horizontales y la columna vertical
    # Brazo izquierdo
    mask[corazon_left-brazo_largo:corazon_left, 
         centro_y-brazo_ancho//2:centro_y+brazo_ancho//2] = True
    
    # Brazo derecho
    mask[corazon_right:corazon_right+brazo_largo, 
         centro_y-brazo_ancho//2:centro_y+brazo_ancho//2] = True
    
    # Columna vertical (ajustada para extenderse hacia abajo)
    mask[centro_x-brazo_ancho//2:centro_x+brazo_ancho//2, 
         centro_y-brazo_largo-corazon//2:centro_y-corazon//2] = True
    
    # Corazón de hielo
    mask[corazon_left:corazon_right, 
         corazon_bottom:corazon_top] = True

    # Inicializar las condiciones de temperatura y energía
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

    # Realizar la simulación paso a paso y buscar el tiempo de demora
    tiempo_demora = None
    temperaturas_promedio = []
    fraccion_hielo_historial = []

    fraccion_hielo = 1.0
    fraccion_hielo_historial.append(fraccion_hielo)

    ''' #comentamos esta parte porque agrega al grafico la mitad de la temperatura
    temp_corazon = temp[corazon_left:corazon_right, corazon_bottom:corazon_top]
    temp_promedio_corazon = np.mean(temp_corazon)
    temperaturas_promedio.append(temp_promedio_corazon)
    '''

    # Crear una máscara para los valores que no queremos mostrar
    temp_masked = np.ma.masked_array(temp, ~mask)

    for step in range(steps):
        temp, energy, ice_fraction = calcular_paso(temp, energy, mask, Dx, Dy, Dt, TEMP_MELT, CALOR_LATENTE_FUSION)
        
        # Medir la fracción de hielo en el corazón
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

        # Verificar si la fracción de hielo ha alcanzado el 50%
        if fraccion_hielo <= 0.5 and tiempo_demora is None:
            tiempo_demora = step * Dt

        # Mostrar el cambio de fase en la simulación cada cierto número de pasos
        if 0:  # Aumentar o disminuir este número para ver más o menos frames
            # Actualizar la máscara de temperatura
            temp_masked = np.ma.masked_array(temp, ~mask)

            # Crear una figura más grande para mejor visualización
            plt.figure(figsize=(10, 8))
            
            # Limpiar la figura anterior
            plt.clf()
            
            # Graficar la nueva distribución de temperatura
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
# Ejecutar la simulación
temp_final, ice_fraction_final, tiempo_demora = simular_cambio_fase()

# Mostrar el tiempo de demora
if tiempo_demora is not None:
    print(f"El tiempo de demora para alcanzar el 50% de la sección útil es: {tiempo_demora:.2f} segundos")
else:
    print("No se alcanzó el 50% de la sección útil durante la simulación.")