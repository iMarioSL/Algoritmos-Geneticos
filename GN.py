import numpy as np

# Parametros
pop = 100
dim = 3
elit = 2
cruce = 0.9
T_mut = 0.2
generation = 100

#X = np.empty([])
#Y = np.empty([])
#Z = np.empty([])

#for flag in range(10):

M_pop = np.zeros((pop, dim))
M_target = np.zeros((pop, 1))

for i in range(pop):
    for j in range(dim):
        M_pop[i, j] = np.random.uniform(-10, 10)
    M_target[i] = M_pop[i, 0]*2.3 + M_pop[i, 1]*5.1 + M_pop[i, 2]*6.6
    
M_ag =  np.hstack((M_pop, M_target))

gen = 1
AG = 2

while gen <= generation:
    # Ordenamos con respecto a la columna target
    M_ag = M_ag[M_ag[:, 3].argsort()]
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
            
            Pc = np.random.randint(0, 3)
            for j in range(dim):
                if j != Pc:
                    aux = Padre1[j]
                    Padre1[j] = Padre2[j]
                    Padre2[j] = aux
            
            # Si hay Crossover entonces puede haber Mutacion
            Mut = np.random.rand()
            if Mut < T_mut:
                Pm = np.random.randint(0, 2)
                Cm = np.random.randint(0, 1)
                if Cm == 1:
                    Padre1[Pm] = np.random.uniform(-10, 10)
                elif Cm == 2:
                    Padre2[Pm] = np.random.uniform(-10, 10)
        
        M_temp[AG] = Padre1
        M_temp[AG + 1] = Padre2
    
    for i in range(pop):
        M_target[i] = M_temp[i, 0]*2.3 + M_temp[i, 1]*5.1 + M_temp[i, 2]*6.6
    
    M_ag = np.hstack((M_temp, M_target))
    
    print(f'Generacion: {gen}')
    gen += 1

# Ordenamos
M_ag = M_ag[M_ag[:, 3].argsort()]

print(f'Solucion real: X = -9, Y = -10, Z = -8')
print(f'Solucion AG: X = {M_ag[0,0]}, Y = {M_ag[0,1]}, Z = {M_ag[0,2]}')

#X = np.append(X, [M_ag[0, 0]])
#Y = np.append(Y, [M_ag[0, 1]])
#Z = np.append(Z, [M_ag[0, 2]])
    
#print(f'Solucion real: X = -9, Y = -10, Z = -8')
#print(f'Solucion AG Avg: X = {np.average(X)}, Y = {np.average(Y)}, Z = {np.average(Z)}')                

