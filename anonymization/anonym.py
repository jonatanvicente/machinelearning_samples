import pandas as pd
import numpy as np

data = pd.DataFrame({
    'name': ['Ann', 'John', 'Luis', 'Peter', 'Silvy'],
    'email': ['ann@sample.com', 'john@sample.com', 'luis@sample.com', 'peter@sample.com', 'silvy@sample.com'],
    'age': [22, 37, 15, 49, 63],
    'location': ['New York', 'Chicago', 'L.A.', 'Portland', 'Washington'],
    'salary': [55000, 34000, 76000, 51000, 62000],
    'bank': ['Bank 1', 'Bank 3', 'Bank 1', 'Bank 2', 'Bank 3']
})

data.drop(['name', 'email'], axis=1, inplace=True) #delete all columns privacy sensitive
data['age'] = (data['age'] // 10) * 10 #generalize age

#add random noise to salary: distorting the data but keep statistical properties (average, deviation, etc)
'''
    Parameters:
        - mean: 0 (media). El promedio de los valores generados por esta distribución será 0
        - std: 100 (desviación estándar). La desviación estándar de los valores generados por esta distribución será 100.
            La mayoría de los valores generados estarán dentro de rango de 100 unidades de la media en ambas direcciones (positiva y negativa).
        - size=data['salary'].shape. Forma del array: forma de la columna 'salary' del DF. 
            Se generará un array aleatorio con la misma forma que la columna 'salary' (cada entrada tendrá entonces un nº aleatorio para añadirle noise).
'''
noise = np.random.normal(0, 100, size=data['salary'].shape)
data['salary'] += noise

# permutation or shuffing: cambiar el orden de ciertos datos, para perder la relación entre ellos (sólo cuando la relación no sea importante)
data['bank'] = np.random.permutation(data['bank'])

print(data)