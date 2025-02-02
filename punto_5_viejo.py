import numpy as np
from numba import jit
import matplotlib.pyplot as plt

# --- Problem parameters ---
L = 21e-3  # Total length (10mm + 1mm + 10mm)
W = 1e-3   # Width of each segment
H = 11e-3  # Total height (10mm + 1mm)

TEMP_INITIAL_ICE = -30
TEMP_INITIAL_WATER = 20
TEMP_MELT = 0
CALOR_LATENTE_FUSION = 334000
RHO_WATER = 1000
RHO_ICE = 918
C_WATER = 4186
C_ICE = 2090

# Flow parameters
INLET_VELOCITY = 0.1e-3  # 0.1 mm/s converted to m/s
MU_WATER = 1.002e-3  # Water dynamic viscosity at 20°C

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
TOTAL_TIME = 30  # Total time in seconds
steps = int(TOTAL_TIME / Dt)

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

@jit(nopython=True)
def calcular_velocidades(temp, ice_fraction, mask, inlet_velocity):
    """Calculate velocity field based on ice fraction and inlet conditions."""
    Nx, Ny = temp.shape
    u = np.zeros((Nx+1, Ny))  # x-direction velocity
    v = np.zeros((Nx, Ny+1))  # y-direction velocity
    
    # Set inlet velocities at the ends of horizontal arms
    for j in range(Ny):
        # Left inlet
        if mask[1, j]:
            u[0, j] = inlet_velocity
        # Right inlet
        if mask[Nx-2, j]:
            u[Nx-1, j] = -inlet_velocity
    
    # Calculate internal velocities based on continuity and ice fraction
    for i in range(1, Nx):
        for j in range(1, Ny-1):
            if mask[i, j]:
                # Velocity affected by ice presence
                flow_factor = 1.0 - ice_fraction[i, j]
                u[i, j] = u[i-1, j] * flow_factor
                
                # Calculate vertical velocity component based on mass conservation
                div_u = (u[i, j] - u[i-1, j]) / Dx
                v[i, j] = v[i, j-1] - div_u * Dy
    
    return u, v

@jit(nopython=True)
def calcular_paso(temp, energy, mask, Dx, Dy, Dt, T_melt, L_fusion, inlet_velocity):
    """Update temperatures and energies at each time step considering water flow."""
    Nx, Ny = temp.shape
    temp_new = np.copy(temp)
    energy_new = np.copy(energy)
    ice_fraction = np.ones_like(temp)

    E_ice_max = RHO_ICE * C_ICE * T_melt
    E_water_min = RHO_WATER * (C_WATER * T_melt + L_fusion)

    # Calculate ice fraction and velocities
    for i in range(Nx):
        for j in range(Ny):
            if mask[i, j]:
                if energy[i,j] >= E_water_min:
                    ice_fraction[i,j] = 0.0
                elif energy[i,j] <= E_ice_max:
                    ice_fraction[i,j] = 1.0
                else:
                    ice_fraction[i,j] = 1.0 - (energy[i,j] - E_ice_max) / (E_water_min - E_ice_max)

    # Calculate velocity field
    u, v = calcular_velocidades(temp, ice_fraction, mask, inlet_velocity)

    # Update temperature and energy considering both conduction and advection
    for i in range(1, Nx-1):
        for j in range(1, Ny-1):
            if mask[i, j]:
                k, rho, c = obtener_propiedades_termicas(temp[i,j], ice_fraction[i,j])
                alpha = k / (rho * c)

                # Conduction terms
                d2T_dx2 = (temp[i+1,j] - 2*temp[i,j] + temp[i-1,j]) / Dx**2
                d2T_dy2 = (temp[i,j+1] - 2*temp[i,j] + temp[i,j-1]) / Dy**2

                # Advection terms
                u_avg = (u[i,j] + u[i+1,j]) / 2
                v_avg = (v[i,j] + v[i,j+1]) / 2
                dT_dx = (temp[i+1,j] - temp[i-1,j]) / (2*Dx)
                dT_dy = (temp[i,j+1] - temp[i,j-1]) / (2*Dy)

                # Combined energy equation with advection
                energy_new[i,j] = energy[i,j] + (
                    rho * c * alpha * (d2T_dx2 + d2T_dy2) * Dt -
                    rho * c * (u_avg * dT_dx + v_avg * dT_dy) * Dt
                )

                # Update temperature based on energy
                if energy_new[i,j] < 0:
                    temp_new[i,j] = energy_new[i,j] / (RHO_ICE * C_ICE) + T_melt
                elif energy_new[i,j] > RHO_WATER * CALOR_LATENTE_FUSION:
                    temp_new[i,j] = (energy_new[i,j] - RHO_WATER * CALOR_LATENTE_FUSION) / (RHO_WATER * C_WATER) + T_melt
                else:
                    temp_new[i,j] = T_melt

                # Apply inlet temperature condition
                if i == 0 or i == Nx-1:
                    if abs(u[i,j]) > 0:
                        temp_new[i,j] = TEMP_INITIAL_WATER

    return temp_new, energy_new, ice_fraction

def simular_cambio_fase():
    """Main function to simulate phase change with water flow."""
    # [Rest of the setup code remains the same until the simulation loop]
    
    # Initialize masks and temperatures as before
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

    # Define T shape
    mask[corazon_left-brazo_largo:corazon_left, 
         centro_y-brazo_ancho//2:centro_y+brazo_ancho//2] = True
    mask[corazon_right:corazon_right+brazo_largo, 
         centro_y-brazo_ancho//2:centro_y+brazo_ancho//2] = True
    mask[centro_x-brazo_ancho//2:centro_x+brazo_ancho//2, 
         centro_y-brazo_largo-corazon//2:centro_y-corazon//2] = True
    mask[corazon_left:corazon_right, 
         corazon_bottom:corazon_top] = True

    # Initialize temperature and energy
    temp = np.full((Nx, Ny), TEMP_INITIAL_WATER)
    energy = np.zeros((Nx, Ny))

    for i in range(Nx):
        for j in range(Ny):
            if mask[i, j]:
                if (corazon_left <= i < corazon_right and 
                    corazon_bottom <= j < corazon_top):
                    temp[i, j] = TEMP_INITIAL_ICE
                    energy[i,j] = RHO_ICE * C_ICE * (TEMP_INITIAL_ICE - TEMP_MELT)
                else:
                    temp[i, j] = TEMP_INITIAL_WATER
                    energy[i, j] = RHO_WATER * C_WATER * TEMP_INITIAL_WATER

    # Simulation loop
    tiempo_demora = None
    temperaturas_promedio = []
    fraccion_hielo_historial = []

    fraccion_hielo = 1.0
    fraccion_hielo_historial.append(fraccion_hielo)
    
    for step in range(steps):
        temp, energy, ice_fraction = calcular_paso(temp, energy, mask, Dx, Dy, Dt, 
                                                 TEMP_MELT, CALOR_LATENTE_FUSION, INLET_VELOCITY)
        
        # Calculate ice fraction in the core
        fraccion_hielo = np.mean(ice_fraction[corazon_left:corazon_right, 
                                            corazon_bottom:corazon_top])
        fraccion_hielo_historial.append(fraccion_hielo)

        # Calculate average temperature in the core
        temp_corazon = temp[corazon_left:corazon_right, corazon_bottom:corazon_top]
        temp_promedio_corazon = np.mean(temp_corazon)
        temperaturas_promedio.append(temp_promedio_corazon)

        # Check if ice fraction has reached 50%
        if fraccion_hielo <= 0.5 and tiempo_demora is None:
            tiempo_demora = step * Dt

    # Plot temperature evolution
    plt.figure(figsize=(8, 6))
    tiempos = np.arange(len(temperaturas_promedio)) * Dt
    plt.plot(tiempos, temperaturas_promedio, label='Average core temperature')
    plt.xlabel('Time (s)')
    plt.ylabel('Average temperature (°C)')
    plt.title('Core temperature evolution')
    plt.legend()
    plt.grid()
    plt.show()

    # Plot ice fraction evolution
    plt.figure(figsize=(8, 6))
    tiempos = np.arange(len(fraccion_hielo_historial)) * Dt
    plt.plot(tiempos, fraccion_hielo_historial, label='Core ice fraction')
    plt.xlabel('Time (s)')
    plt.ylabel('Ice fraction')
    plt.title('Core ice fraction evolution')
    plt.legend()
    plt.grid()
    plt.show()

    return temp, ice_fraction, tiempo_demora

# Run simulation
print(f"Time step (dt): {Dt}")
print(f"Number of steps: {steps}")
temp_final, ice_fraction_final, tiempo_demora = simular_cambio_fase()

if tiempo_demora is not None:
    print(f"Time to reach 50% ice fraction: {tiempo_demora:.2f} seconds")
else:
    print("50% ice fraction was not reached during the simulation.")