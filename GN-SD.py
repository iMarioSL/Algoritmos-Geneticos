import numpy as np


def functionSD(Vt, It, X):
    k = 1.380649E-23
    q = 1.602E-19
    T = 306.15
    expo = np.exp(q*(Vt + X[0]*It)/(X[4] * k * T))
    return It - X[2] + X[3]*(expo - 1) + (Vt + X[0]*It)/X[1]


Vtm = [-0.2057, -0.1291, -0.0588, 0.0057, 0.0646, 0.1185, 0.1678, 0.2132,
       0.2545, 0.2924, 0.3269, 0.3585, 0.3873, 0.4137, 0.4373, 0.459, 0.4784,
       0.496, 0.5119, 0.5265, 0.5398, 0.5521, 0.5633, 0.5736, 0.5833, 0.59]

Itm = [0.764, 0.762, 0.7605, 0.7605, 0.76, 0.759, 0.757, 0.757, 0.7555, 0.754,
       0.7505, 0.7465, 0.7385, 0.728, 0.7065, 0.6755, 0.632, 0.573, 0.499, 0.413,
       0.3165, 0.212, 0.1035, -0.01, -0.123, -0.21]

# Parametros
pop = 100
dim = 5
elit = 2
cruce = 0.9
T_mut = 0.2
generation = 100
suma = 0

RMSE = 0
experimento = 0

# Rs:{M_ag[0][0]}, Rsh:{M_ag[0][1]}, Iph:{M_ag[0][2]}, Isd:{M_ag[0][3]}, n:{M_ag[0][4]}

Rs_best = np.empty([])
Rsh_best = np.empty([])
Iph_best = np.empty([])
Isd_best = np.empty([])
n_best = np.empty([])
Itm_best = np.empty([])

for iteration in range(100):
    for valor in range(len(Vtm)):
        # Creamos las matrices de poblacion y el target.
        M_pop = np.zeros((pop, dim))
        M_target = np.zeros((pop, 1))

        # Rellenamos las matrices con valores aleatorios en una distribución
        for i in range(pop):
            for j in range(dim):
                if j == 0:
                    M_pop[i, j] = np.random.uniform(0, 0.5)
                if j == 1:
                    M_pop[i, j] = np.random.uniform(0, 100)
                if j == 2:
                    M_pop[i, j] = np.random.uniform(0, 1)
                if j == 3:
                    M_pop[i, j] = np.random.uniform(0, 1)
                if j == 4:
                    M_pop[i, j] = np.random.uniform(1, 2)

            M_target[i] = functionSD(Vtm[valor], Itm[valor], M_pop[i])

        # Concatenamos las matrices anteriores, de modo que tenemos una matriz de 100x3
        M_ag = np.hstack((M_pop, M_target))

        # Aqui comenzamos con el GA

        gen = 1
        AG = 2

        while gen <= generation:
            # Ordenamos con respecto a la columna target
            M_ag = M_ag[M_ag[:, 5].argsort()]
            # Creamos una matriz temporal, la que cada generacion
            # cambiara sus valores
            M_temp = np.zeros((pop, dim))

            # Asignacion de dos primeras lineas
            for i in range(elit):
                for j in range(dim):
                    M_temp[i, j] = M_ag[i, j]

            # Se realiza la seleccion por combate
            for AG in range(2, pop-1):
                N1 = np.random.randint(0, len(Vtm))
                N2 = np.random.randint(0, len(Vtm))
                if N1 == N2:
                    while True:
                        N1 = np.random.randint(0, len(Vtm))
                        if N1 != N2:
                            break
                elif N1 < N2:
                    N2 = np.random.randint(0, len(Vtm))
                elif N2 < N1:
                    N1 = np.random.randint(0, len(Vtm))

                # Creacion de padres
                Padre1 = M_ag[N1, :dim]
                Padre2 = M_ag[N2, :dim]

                # Crossover
                Cross = np.random.rand()
                if Cross <= cruce:
                    Pc = np.random.randint(0, dim-1)
                    for j in range(dim):
                        if j != Pc:
                            aux = Padre1[j]
                            Padre1[j] = Padre2[j]
                            Padre2[j] = aux

                    # Si hay Crossover entonces puede haber mutacion
                    Mut = np.random.rand()
                    if Mut < T_mut:
                        Pm = np.random.randint(0, 4)
                        Cm = np.random.randint(0, 1)
                        if Cm == 0:
                            if Pm == 0:
                                Padre1[Pm] = np.random.uniform(0, 0.5)
                            elif Pm == 1:
                                Padre1[Pm] = np.random.uniform(0, 100)
                            elif Pm == 2:
                                Padre1[Pm] = np.random.uniform(0, 1)
                            elif Pm == 3:
                                Padre1[Pm] = np.random.uniform(0, 1)
                            elif Pm == 4:
                                Padre1[Pm] = np.random.uniform(1, 2)
                        elif Cm == 1:
                            if Pm == 0:
                                Padre2[Pm] = np.random.uniform(0, 0.5)
                            elif Pm == 1:
                                Padre2[Pm] = np.random.uniform(0, 100)
                            elif Pm == 2:
                                Padre2[Pm] = np.random.uniform(0, 1)
                            elif Pm == 3:
                                Padre2[Pm] = np.random.uniform(0, 1)
                            elif Pm == 4:
                                Padre2[Pm] = np.random.uniform(1, 2)

                # Cargamos a la matriz temporal con el valor de los padres nuevos
                M_temp[AG] = Padre1
                M_temp[AG + 1] = Padre2

            # Ahora nuestra matriz target se rellana con los valores cargados en la matriz temporal
            for i in range(pop):
                M_target[i] = functionSD(Vtm[valor], Itm[valor], M_temp[i])

            M_ag = np.hstack((M_temp, M_target))
            gen += 1

        # Ordenamos el valor de la matriz resultado tras pasar 100 generaciones
        M_ag = M_ag[M_ag[:, 5].argsort()]

        
        #suma = 0
        #for index in range(1):
        #    suma += (Itm[index] - M_ag[index][-1])**2
        #RMSE = np.sqrt((suma/len(Vtm)))
        
        #print(f'RMSE: {RMSE:.3f}')
        #print("-"*30)
        experimento += 1
        Rs_best = np.append(Rs_best, M_ag[0][0])
        Rsh_best = np.append(Rsh_best, M_ag[0][1])
        Iph_best = np.append(Iph_best, M_ag[0][2])
        Isd_best = np.append(Isd_best, M_ag[0][3])
        n_best = np.append(n_best, M_ag[0][4])
        Itm_best = np.append(n_best, M_ag[0][5])

    Rs_best = np.average(Rs_best)
    Rsh_best= np.average(Rsh_best)
    Iph_best = np.average(Iph_best)
    Itm_best = np.average(Itm_best)
    Isd_best = np.average(Isd_best)
    n_best = np.average(n_best)
    
    RMSE = Itm[0] - Itm_best
    print(f"Iteracion: {iteration + 1}")
    print(f'RMSE: {RMSE:.3f}')
    print("-"*30)
    
    
    
    
print(f'Rs:{Rs_best:.2f}, Rsh:{Rsh_best:.2f}, Iph:{Iph_best:.2f}, Isd:{Isd_best:.2f}, n:{n_best:.2f}')
