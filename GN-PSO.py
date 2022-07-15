import numpy as np
import matplotlib.pyplot as plt
from matplotlib import cm
from matplotlib.ticker import LinearLocator, FormatStrFormatter
from mpl_toolkits.mplot3d import Axes3D


def graficar(X_best, Y_best):

    # Funcion de Rosenbrock
    def f(x, y): return (1-x)**2 + 100*(y-x**2)**2

    # Inicializamos la figura
    figRos = plt.figure(figsize=(7, 7))
    axRos = figRos.add_subplot(projection='3d')

    # Evaluamos la función.
    X = np.arange(-2, 2, 0.15)
    Y = np.arange(-1, 3, 0.15)
    X, Y = np.meshgrid(X, Y)
    Z = f(X, Y)

    # Evaluamos la función en los mejores puntos.
    Z_best = f(X_best, Y_best)

    # Graficamos la superficie
    surf = axRos.plot_surface(X, Y, Z, cmap="magma", linewidth=0)

    # Graficamos el punto (x,y)
    axRos.scatter(X_best, Y_best, Z_best, color="lime", s=50)

    # Damos un limite al eje Z
    axRos.set_zlim(0, 2000)

    figRos.colorbar(surf, shrink=0.5, aspect=10)

    plt.show()


# Parametros
pop = 100
dim = 2
elit = 2
cruce = 0.9
T_mut = 0.2
generation = 100
a = 1
b = 100

experimentos = 100
X_best = np.empty([])
Y_best = np.empty([])

for experimento in range(experimentos):
    # Creamos las matrices de poblacion y el target.
    M_pop = np.zeros((pop, dim))
    M_target = np.zeros((pop, 1))

    # Rellenamos las matrices con valores aleatorios en una distribución
    # uniforme entre -2 y 3.
    for i in range(pop):
        for j in range(dim):
            M_pop[i, j] = np.random.uniform(-2, 3)
        M_target[i] = (a - M_pop[i, 0])**2 + b * \
            (M_pop[i, 1] - M_pop[i, 0]**2)**2

    # Concatenamos las matrices anteriores, de modo que tenemos una matriz de 100x3
    M_ag = np.hstack((M_pop, M_target))

    # Aqui comenzamos con el GA

    gen = 1
    AG = 2

    while gen <= generation:
        # Ordenamos con respecto a la columna target
        M_ag = M_ag[M_ag[:, 2].argsort()]
        # Creamos una matriz temporal, la que cada generacion
        # cambiara sus valores
        M_temp = np.zeros((pop, dim))

        # Asignacion de dos primeras lineas
        for i in range(elit):
            for j in range(dim):
                M_temp[i, j] = M_ag[i, j]

        # Se realiza la seleccion por combate
        for AG in range(2, pop-1):
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
                    Pm = np.random.randint(0, 1)
                    Cm = np.random.randint(0, 1)
                    if Cm == 0:
                        Padre1[Pm] = np.random.uniform(-2, 3)
                    elif Cm == 1:
                        Padre2[Pm] = np.random.uniform(-2, 3)

            # Cargamos a la matriz temporal con el valor de los padres nuevos
            M_temp[AG] = Padre1
            M_temp[AG + 1] = Padre2

        # Ahora nuestra matriz target se rellana con los valores cargados en la matriz temporal
        for i in range(pop):
            M_target[i] = (a - M_temp[i, 0])**2 + b * \
                (M_temp[i, 1] - M_temp[i, 0]**2)**2

        M_ag = np.hstack((M_temp, M_target))
        gen += 1

    # Ordenamos el valor de la matriz resultado tras pasar 100 generaciones
    M_ag = M_ag[M_ag[:, 2].argsort()]

    # Guardamos los valores x, y encontrados en la corrida
    X_best = np.append(X_best, [M_ag[0, 0]])
    Y_best = np.append(Y_best, [M_ag[0, 1]])

    # Calculamos el promedio de los valores dados en la lista X_best y Y_best, de
    # modo que vemos el valor promedio total obtenido hasta los n numero
    # experimentos que hayan transcurrido
    X_best_avg = np.average(X_best)
    Y_best_avg = np.average(Y_best)

    print(f"Experimento: {experimento + 1}")
    print(f"X_best: {X_best_avg:.3f}, Y_best: {Y_best_avg:.3f}")
    print("-"*30)


print(f'Solucion real: X = 1, Y = 1')
print(f'Solucion AG Avg: X = {X_best_avg:.3f}, Y = {Y_best_avg:.3f}')
graficar(X_best_avg, Y_best_avg)
