import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.utils import resample

ruta = "customers.csv"
df = pd.read_csv(ruta)
df.head()

# Anonimizar el campo dirección a través de la eliminación
df.drop('direccion', axis=1, inplace=True)
df.head()

# Anonimizar el campo edad a través del redondeo
df['edad'] = (df['edad'] // 10) * 10

# Anonimizar el campo salario a través de la agregación de ruido
ruido = np.random.normal(0, 100, size=df['salario'].shape)
df['salario'] += ruido

# Pseudonimizar el campo nombre
for i in range(len(df)):
    df.loc[i, 'nombre'] = 'Cliente' + str(i)

    # Balancear los datos de categorías
    agrupado = df.groupby('categoria')

    df_balanceado = pd.DataFrame()

    for nombre, grupo in agrupado:
        grupo_balanceado = resample(grupo,
                                    replace=True,
                                    n_samples=100,
                                    random_state=123)
        df_balanceado = pd.concat([df_balanceado, grupo_balanceado])

print(df_balanceado)

# Visualización 1: Distribución de las edades con curva de densidad
plt.figure(figsize=(12, 7))
ax = sns.histplot(df['edad'],
                  kde=True,
                  color='skyblue',
                  bins=30)
ax.set(title='Distribución de Edades de los Clientes con Curva de Densidad',
       xlabel='Edad',
       ylabel='Frecuencia');
plt.show();

# Visualización 2: Relación entre Edad y Salario con tamaño variable
plt.figure(figsize=(12, 7))
sizes = df['categoria'].replace({0: 50, 1: 100})  # Asignar tamaño según categoría para ilustrar
scatter = sns.scatterplot(x='edad',
                          y='salario',
                          size=sizes,
                          legend=False,
                          sizes=(20, 200),
                          data=df,
                          color='red',
                          alpha=0.6)
scatter.set(title='Relación entre Edad y Salario de los Clientes con Tamaño Variable',
            xlabel='Edad',
            ylabel='Salario');

plt.show();

# Visualización 3: Mapa de Calor de Correlación entre Variables
plt.figure(figsize=(10, 8))
# Calculamos la matriz de correlación
correlation_matrix = df[['edad', 'salario', 'categoria']].corr()
heatmap = sns.heatmap(correlation_matrix,
                      annot=True,
                      cmap='coolwarm')
heatmap.set(title='Mapa de Calor de Correlación entre Variables');
plt.show();
