'''
Linear Regression
Relación precio vivienda - superficie
'''
import pandas as pd
import matplotlib.pyplot as plt
from sklearn import linear_model


# DataFrame inicial
df = pd.DataFrame()
df["Area"] = [2600, 3000, 3200, 3600, 4000]
df["Precio"] = [550000, 565000, 610000, 680000, 725000]

# Visualizamos
plt.scatter(df["Area"], df["Precio"])
# plt.show()
''' 
Es relación no proporcional: no podemos trazar línea recta o curva constante, 
no podemos predecir su precio en base a su superficie.
Crearemos una línea recta con la menor distancia posible a cada punto de información.
Algoritmo aplicado: ¿dónde colocar la línea recta este gráfico, de modo que cada punto
 de información esté a la menor distancia de la línea?.
Calculará y para cada punto x, minimizando los errores (distancia entre la línea y los puntos).
'''

# Entrenamos algoritmo
# Creamos Dataframe: aunque pasamos una sola columna, podríamos estar pasando muchas series o
# columnas al mismo tiempo; el eje debe ser bidimensional (muchas columnas y muchas filas)

# X contendrá los datos de entrada. Es la variable independiente del
# modelo. Es la variable de la que ya conocemos todos sus puntos posibles.
# DataFrame de una sola columna; al ser un DataFrame nombramos en mayúscula
X = df[["Area"]]
# Dependent variable
y = df["Precio"]

# Creamos modelo de regresión lineal
modelo = linear_model.LinearRegression()

# Entrenamos modelo con datos preexistentes
modelo.fit(X, y)
# Precio de una casa de 3300 pies cuadrados
modelo.predict(pd.DataFrame([[3300]],
                            columns=["Area"]))

plt.plot(df["Area"], modelo.predict(X))

plt.scatter(df["Area"], df["Precio"], color='black', label='Actual data')
plt.plot(df["Area"], modelo.predict(X), color='red', linewidth=2, label='Predicted line')
plt.xlabel('X')
plt.ylabel('y')
plt.title('Linear Regression')
plt.legend()
plt.show()
