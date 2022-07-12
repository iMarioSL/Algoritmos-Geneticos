import numpy as np
import matplotlib.pyplot as plt
from matplotlib import cm
from matplotlib.ticker import LinearLocator, FormatStrFormatter
from mpl_toolkits.mplot3d import Axes3D

# Parametros
pop = 100
dim = 2
elit = 2
cruce = 0.9
T_mut = 0.2
generation = 100
a = 1
b = 100
#X = np.empty([])
#Y = np.empty([])
#Z = np.empty([])

#for flag in range(10):

# Ecuacion de Rosenbrock
# f(x, y) = (a - x)^2 + b(y - x^2)^2
# a = 1
# b = 100




M_pop = np.zeros((pop, dim))
M_target = np.zeros((pop, 1))

for i in range(pop):
    for j in range(dim):
        M_pop[i, j] = np.random.uniform(-2, 3)
    M_target[i] = (a - M_pop[i, 0])**2 + b*(M_pop[i, 1] - M_pop[i, 0]**2)**2
    
M_ag =  np.hstack((M_pop, M_target))

gen = 1
AG = 2

while gen <= generation:
    # Ordenamos con respecto a la columna target
    M_ag = M_ag[M_ag[:, 2].argsort()]
    M_temp = np.zeros((pop, dim))

    # Asignacion de dos primeras lineas
    for i in range(elit):
        for j in range(dim):
            M_temp[i, j] = M_ag[i, j]
            
    for AG in range(2, pop-1): 
        # Seleccion por combate
        N1 = np.random.randint(0, 99)
        N2 = np.random.randint(0, 99)        
        if N1 == N2:
            while True:
                N1 = np.random.randint(0, 99)
                if N1 != N2:
                    break
        elif N1 < N2:
            N2 = np.random.randint(0, 99)
        elif N2 < N1:
            N1 = np.random.randint(0, 99)

        # Creacion de padres
        Padre1 = M_ag[N1, :dim]
        Padre2 = M_ag[N2, :dim]
        
        # Crossover
        # Probabilidad de cruzado
        Cross = np.random.rand()
        if Cross <= cruce:
            
            Pc = np.random.randint(0, dim)
            for j in range(dim):
                if j != Pc:
                    aux = Padre1[j]
                    Padre1[j] = Padre2[j]
                    Padre2[j] = aux
            
            # Si hay Crossover entonces puede haber Mutacion
            Mut = np.random.rand()
            if Mut < T_mut:
                Pm = np.random.randint(0, 1)
                Cm = np.random.randint(0, 1)
                if Cm == 0:
                    Padre1[Pm] = np.random.uniform(-2, 3)
                elif Cm == 1:
                    Padre2[Pm] = np.random.uniform(-2, 3)
        
        M_temp[AG] = Padre1
        M_temp[AG + 1] = Padre2
    
    for i in range(pop):
        M_target[i] = (a - M_temp[i, 0])**2 + b*(M_temp[i, 1] - M_temp[i, 0]**2)**2
    
    M_ag = np.hstack((M_temp, M_target))
    
    #print(f'Generacion: {gen}')
    gen += 1

# Ordenamos
M_ag = M_ag[M_ag[:, 2].argsort()]

print(f'Solucion real: X = 1, Y = 1')
print(f'Solucion AG: X = {M_ag[0,0]:.3f}, Y = {M_ag[0,1]:.3f}')



f = lambda x,y: (1-x)**2 + 100*(y-x**2)**2;

# Initialize figure 
figRos = plt.figure(figsize=(7, 7))
axRos = figRos.gca(projection='3d')

# Evaluate function
X = np.arange(-2, 2, 0.15)
Y = np.arange(-1, 3, 0.15)
X, Y = np.meshgrid(X, Y)
Z = f(X,Y)

Z_new = f(M_ag[0,0], M_ag[0,1])
# Plot the surface
surf = axRos.plot_surface(X, Y, Z, cmap="magma",
                       linewidth=0)
axRos.scatter(M_ag[0,0] ,M_ag[0,1] , Z_new ,color="r",s=20)


axRos.set_zlim(0, 2000)
figRos.colorbar(surf, shrink=0.5, aspect=10)
plt.show()



#X = np.append(X, [M_ag[0, 0]])
#Y = np.append(Y, [M_ag[0, 1]])
#Z = np.append(Z, [M_ag[0, 2]])
    
#print(f'Solucion real: X = -9, Y = -10, Z = -8')
#print(f'Solucion AG Avg: X = {np.average(X)}, Y = {np.average(Y)}, Z = {np.average(Z)}')                

