import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.utils import resample


# contrataciones laborales, edad y género pueden introducir sesgos por no estar bien balanceadas
data = pd.DataFrame({
    'edad': [22, 45, 36, 29, 58],
    'género': ['F', 'M', 'M', 'F', 'M'],
    'contratado': [1, 0, 1, 0, 1]
})

# Agrupamos por género: hay diferentes características que podrían influir en el modelo final
agrupado = data.groupby('género')

# Vemos lo que contiene agrupado
for nombre, grupo in agrupado:
    print(nombre, grupo)

# Creamos dataframe vacío para almacernar los datos balanceados
data_balanceada = pd.DataFrame()

# Iteramos para crear un conjunto de datos balanceado
for nombre, grupo in agrupado:
    grupo_balanceado = resample(grupo,
                               replace=True,
                               n_samples=10,
                               random_state=123)
    data_balanceada = pd.concat([data_balanceada, grupo_balanceado])

# show
print(data_balanceada)

# ahora cada género está igualmente representado. Mejoramos equidad y precisión, las decisiones
# en estos modelos son más justas y confiables