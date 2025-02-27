import numpy as np
import matplotlib.pyplot as plt
from sklearn.datasets import load_iris
from sklearn.decomposition import PCA

iris = load_iris() # Array bidimensional, dataset de sklearn

# Definimos x e y
X = iris.data
y = iris.target

'''
Estandarizamos datos:
    - Medimos cuánto se desvía del promedio de todas las columnas
    - Ajustamos a todas las columnas para que este punto medio sea el mismo para todas (0 en este caso)
    - Ahora todas tienen un punto de partida común
'''
X_centrado = X - np.mean(X, axis=0)

# Aplicamos PCA
pca = PCA(n_components=2)
X_pca = pca.fit_transform(X_centrado)

# Show
especies = ["setosa", "versicolor", "virginica"]
plt.figure(figsize=(8, 6))
for i in range(0, 3):
    plt.scatter(X_pca[y == i, 0], X_pca[y == i, 1], label=especies[i])
plt.xlabel('Primer componente principal')
plt.ylabel('Segundo componente principal')
plt.legend()
plt.title('PCA del conjunto Iris');
plt.show()
