import numpy as np # svd sólo necesita numpy
import matplotlib.pyplot as plt
from sklearn.datasets import load_iris


iris= load_iris()
X = iris.data # sólo necesitamos sus características

# centramos los datos (similar a PCA)
# X_centrado será nuestra matriz original A. Aplicamos SVD sobre X_centrado
X_centrado = X - np.mean(X, axis=0)

U, sigma, VT = np.linalg.svd(X_centrado)
# tomamos los primeros 2 componentes principales para la visualización
# Reducimos dimensionalidad mediante svd
# sigma contiene los valores singulares, que representan la importancia de cada componente en la descripción
# de la variabilidad de los datos
# Componentes principales son nuevas direcciones que capturan la mayor variabilidad posible
# k = 2 establece que sólo queremos 2 componentes principales (espacio bidimensional (como PCA)
k = 2


'''
    - U[:, :k] selecciona las primeras 2 columnas de U, que corresponden a los vectores singulares 
        izquierdos asociados con los 2 valores singulares más grandes. 
    - sigma[:k] selecciona los primeros 2 valores singulares de sigma, que 
        indican la importancia de cada uno de los 2 componentes principales.
    Y la multiplicación se realiza elemento a elemento entre cada columna seleccionada de U y 
        el correspondiente valor singular, escalando así las columnas de U por la importancia de cada componente principal.
'''
X_transformado = U[:, :k] * sigma[:k]

especies = ["setosa", "versicolor", "virginica"]
plt.figure(figsize=(8, 6))
for i in range(3):
    plt.scatter(X_transformado[iris.target == i, 0],
               X_transformado[iris.target == i, 1],
               label=especies[i])
plt.xlabel('Componente Principal 1')
plt.ylabel('Componente Principal 2')
plt.legend()
plt.title('Dataset iris transformado por SVD');
plt.show()

'''
Esta visualización es como si hubiéramos tomado todas las flores de Iris, con todas sus cuatro diferentes medidas, 
y las hubiéramos dibujado en un mapa simplificado para ver cómo se agrupan.

Con SVD encontramos las dos direcciones principales que nos ayudan a ver las diferencias más grandes entre las flores, 
y luego, dibujamos cada flor en este nuevo mapa usando solamente esas dos direcciones.

El resultado es muy similar al de PCA, reducen dimensionalidad y encuentran las direcciones principales que agrupan elementos.
'''