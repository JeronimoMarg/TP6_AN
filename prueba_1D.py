import numpy as np
import matplotlib.pyplot as plt

L=10 #en milimetros
h=0.01 #en milimetros
#dt=1.*h**2
#dt=0.5*h**2
dt=0.05
time=100.
alpha=0.5

# Thermal properties
rho_ice = 917  # density of ice (kg/m^3)
rho_water = 997  # density of water (kg/m^3)
c_ice = 2100  # specific heat of ice (J/kg*K)
c_water = 4186  # specific heat of water (J/kg*K)
k_ice = 1.6  # thermal conductivity of ice (W/m*K)
k_water = 0.6  # thermal conductivity of water (W/m*K)
h_fusion = 334000  # latent heat of fusion (J/kg)

alpha_ice = k_ice / (rho_ice * c_ice)
alpha_water = k_water / (rho_water * c_water)

n=int(L/h) + 1 #cantidad de secciones dentro del alambre 1d
x = np.linspace(0, L, n)

T = np.full(n, -10.0)

#Funciones del profe

def stencil(n):
    A = -2 * np.eye(n)
    for j in range (n-1):
        A[j,j+1]=1
        A[j+1,j]=1
    return A

# Condicion de borde de la derecha.
def boundary_condition_right(t):
    if t < 10:
        aux = -10 + (95 * t / 10)
        return aux
    else:
        return 85

def explicit(h,dt,time,L):
    t=0.0
    #n=int(L/h)
    #x=np.arange(0,L,h)
    #T=np.sin(np.pi*x/L)

    n=int(L/h) + 1 #cantidad de secciones dentro del alambre 1d
    x = np.arange(0, L, n)
    T = np.full(n, -10.0)
    #print(T)
    
    sol=[]
    sol.append(T)
    mat=dt*stencil(n)/h/h+np.eye(n)
    while t<time:
        T=np.dot(mat,T)
        T[0]=T[1]                           #condicion de borde de la izquierda
        T[-1]=boundary_condition_right(t)   #condicion de borde de la derecha
        sol.append(T)
        print(T)
        t+=dt
    return sol

def implicit(h,dt,time,L):
    t=0.0
    #n=int(L/h)
    #x=np.arange(0,L,h)
    #T=np.sin(np.pi*x/L)

    n=int(L/h) + 1 #cantidad de secciones dentro del alambre 1d
    x = np.arange(0, L, n)
    T = np.full(n, -10.0)
    #print(T)

    sol=[]
    sol.append(T)
    aux=np.eye(n)-dt*stencil(n)/h/h
    mat=np.linalg.inv(aux)
    while t<time:
        T1=np.dot(mat,T)
        T[0]=T[1]                           #condicion de borde de la izquierda
        T[-1]=boundary_condition_right(t)   #condicion de borde de la derecha
        sol.append(T)
        t+=dt
    return sol

sol1=explicit(h,dt,time,L)
sol2=implicit(h,dt,time,L)

plt.xlabel('Position')
plt.ylabel('Temperature')
plt.title('Temperature Distribution Over Time')
for n in range(int(time/dt/10)):
    plt.plot(sol2[10*n])
plt.show()
