'''
Robot que opera en una cuadrícula. Debe encontrar el camino más corto evitando obstáculos (ver imagen añadida).
    Entorno:
        - Cuadrícula de 5x5
        - Punto de inicio en esquina superior izquierda (0,0)
        - Punto objetivo en esquina inferior derecha (4,4)
        - Obstáculos distribuidos por la cuadrícula
    Acciones:
        - Arriba
        - Abajo
        - Izquierda
        - Derecha
    Recompensas:
        - Alcanzar objetivo +100
        - Colisionar con obstáculo -100
        - Cualquier otro movimiento -1 (para incentivar la eficiencia)
'''
import numpy as np
import random

dimensiones = (5, 5)
estado_inicial = (0, 0)
estado_objetivo = (4, 4)
obstaculos = [(1, 1), (1, 3), (2, 3), (3, 0)]
acciones = [(-1, 0), (1, 0), (0, -1), (0, 1)]

# Inicializamos tabla Q (matriz de dos dimensiones). Variables auxiliares:
#    - Numero total de estados o posiciones posibles en la cuadrícula
#    - Numero total de acciones posibles

num_estados = dimensiones[0] * dimensiones[1]
num_acciones = len(acciones)
Q = np.zeros((num_estados, num_acciones))


# Función para convertir la representación bidimensional del estado actual (posición del robot en la cuadrícula) a un índice lineal único
# Lo hacemos para trabajar de manera más efectiva con la tabla Q, que es un array que contiene 25 filas (cuyos índices van desde 0 hasta 24)
# Así cada estado posible se pueda representar por un número de índice de la tabla Q.

def estado_a_indice(estado):
    return estado[0] * dimensiones[1] + estado[1]

# Por ejemplo:
# ejemplo = estado_a_indice((1, 0)) # output = 5

# Definimos los parámetros clave que se van a utilizar en el algoritmo de Q-learning,
# dentro del contexto de nuestro ejemplo de navegación autónoma de un robot.
#   Estos parámetros se pueden ajustar después:
# alpha. Factor de tasa de aprendizaje. Controla cuánto se actualiza el valor Q en cada paso del aprendizaje.
#       Un valor de alpha más alto significa que la información más reciente tiene un peso mayor,
#       permitiendo un aprendizaje más rápido pero potencialmente menos estable.
#       Un valor más bajo hace que el aprendizaje sea más lento pero puede llevar a una estimación más estable de los valores Q.
# gamma. Factor de descuento. Determina la importancia de las recompensas futuras.
#       Un valor de gamma cercano a 1 hace que las recompensas futuras sean casi tan importantes como las recompensas inmediatas.
#       Así se incentiva al agente a considerar consecuencias a largo plazo de sus acciones.
#       Un valor más bajo haría que el agente valorase más las recompensas inmediatas.
# epsilon. Valor para que el agente no repita siempre las mismas decisiones.
#       Define la probabilidad de que el agente tome una acción aleatoria en lugar de la mejor acción conocida hasta el momento según la tabla Q.
#       Esto permite que el agente explore el entorno en lugar de explotar constantemente el conocimiento que ya dispone.
#       Así se consigue equilibrio entre exploración y explotación para asegurar que el agente siga aprendiendo eficazmente sobre el entorno.
# episodios. Número total de episodios para el proceso de entrenamiento.
#       Un episodio comienza con el agente en el estado inicial y termina cuando alcanza el objetivo o algún otro criterio de terminación.
#       A mayor número de episodios, más oportunidades tendrá al agente de tener más experiencias para aprender,
#       mejorando su política de acción.

alpha = 0.1
gamma = 0.99
epsilon = 0.2
episodios = 100

# Para que el agente elija acción
def elegir_accion(estado):
    if random.uniform(0, 1) < epsilon:
        return random.choice(range(num_acciones))
    else:
        return np.argmax(Q[estado_a_indice(estado)])

# Para aplicar acción y obtener nuevo estado
def aplicar_accion(estado, accion_idx):
    accion = acciones[accion_idx]
    nuevo_estado = tuple(np.add(estado, accion) % dimensiones)

    if nuevo_estado in obstaculos or nuevo_estado == estado:
        return estado, -100, False
    if nuevo_estado == estado_objetivo:
        return nuevo_estado, 100, True
    return nuevo_estado, -1, False

# Entrenamiento
for episodio in range(episodios):
    estado = estado_inicial
    terminado = False

    while not terminado:
        idx_estado = estado_a_indice(estado)
        accion_idx = elegir_accion(estado)
        nuevo_estado, recompensa, terminado = aplicar_accion(estado, accion_idx)
        idx_nuevo_estado = estado_a_indice(nuevo_estado)

        Q[idx_estado, accion_idx] = Q[idx_estado, accion_idx] + alpha + (
                    recompensa + gamma * np.max(Q[idx_nuevo_estado]) - Q[idx_estado, accion_idx])

        estado = nuevo_estado

# Visualizamos
politica = np.zeros(dimensiones, dtype=int)
# Llenamos la matriz de política con las mejores acciones para cada estado
for i in range(dimensiones[0]):
    for j in range(dimensiones[1]):
        estado = (i, j)
        idx_estado = estado_a_indice(estado)
        mejor_accion = np.argmax(Q[idx_estado])
        politica[i, j] = mejor_accion

print("Política aprendida (0: arriba, 1: abajo, 2: izquierda, 3: derecha)")
# Resultado = instantánea del conocimiento adquirido por el agente: cómo navegar el entorno de manera eficiente
#   basado en las recompensas y penalizaciones experimentadas durante el entrenamiento.
print(politica)