
import numpy as np
import matplotlib.pyplot as plt
L=1.0
h=0.01
dt=1.*h**2
time=5.
alpha=0.5

#Funciones del profe

def stencil(n):
    A=np.diag(-2*np.ones(n))+np.diag(np.ones(n-1),1)+np.diag(np.ones(n-1),-1)
    return A

def exact(h,dt,time,L):
    t=0.0
    x=np.arange(0,L,h)
    T=np.sin(np.pi*x/L)
    sol=[]
    sol.append(T)
    while t<time:
        T=np.sin(np.pi*x/L)*np.exp(-np.pi**2*t/L**2)
        sol.append(T)
        T[0]=0
        T[-1]=0
        sol.append(T)
        t+=dt
    return sol

def explicit(h,dt,time,L):
    t=0.0
    n=int(L/h)
    x=np.arange(0,L,h)
    T=np.sin(np.pi*x/L)
    sol=[]
    sol.append(T)
    mat=dt*stencil(n)/h/h+np.eye(n)
    while t<time:
        T=np.dot(mat,T)
        T[0]=0
        T[-1]=0
        sol.append(T)
        t+=dt
    return sol

def implicit(h,dt,time,L):
    t=0.0
    n=int(L/h)
    x=np.arange(0,L,h)
    T=np.sin(np.pi*x/L)
    sol=[]
    sol.append(T)
    aux=np.eye(n)-dt*stencil(n)/h/h
    mat=np.linalg.inv(aux)
    while t<time:
        T1=np.dot(mat,T)
        T[0]=0
        T[-1]=0
        sol.append(T)
        t+=dt
    return sol

sol=exact(h,dt,time,L)
sol1=explicit(h,dt,time,L)
sol2=implicit(h,dt,time,L)

plt.ion()
plt.show()
for n in range(int(time/dt/10)):
    plt.plot(sol1[10*n])

error1=[]
error2=[]
for n in range (int(time/dt)):
    error1.append(np.linalg.norm(sol[n]-sol1[n]))
    error2.append(np.linalg.norm(sol[n]-sol2[n]))

tiempo=np.arange(0,time,dt)
plt.plot(tiempo,error1,label='Error Explicito')
plt.plot(tiempo,error2,label='Error Implicito')
plt.legend()

