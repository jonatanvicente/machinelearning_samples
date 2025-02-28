from sklearn.datasets import load_iris
import matplotlib.pyplot as plt
from scipy.cluster.hierarchy import dendrogram, linkage

'''
El dendograma ayuda a visualizar las decisiones tomadas por el algoritmo en cada paso del proceso de fusión y 
puede usarse para determinar un número adecuado de clusters cortando el dendrograma en una altura específica.
Clustering jerárquico es una técnica poderosa y muy versátil para el análisis de clusters que agrupa los datos 
y proporciona info sobre estructura jerárquica. 
Especialmente útil cuando la relación entre los puntos de datos es importante o cuando se desea explorar 
diferentes niveles de granularidad en agrupamiento de data.
'''

iris = load_iris()
X = iris.data
# Guardamos clustering en una variable
linked = linkage(X, 'ward')
'''
    - Ward minimiza la varianza de cada cluster en cada paso
    - Otros métodos de linkage:
        * 'single': Método del vecino más cercano. Fusiona los dos clusters que tienen la distancia mínima más pequeña entre sus puntos más cercanos. 
        Sensible a los valores atípicos y puede producir clusters "largos y finos".
        * 'complete': También conocido como el método del vecino más lejano. Fusiona los clusters con la menor distancia máxima entre sus puntos.
        Tiende a producir clusters más compactos que el método 'single', pero aún puede ser sensible a los valores atípicos.
        * 'average': Se basa en la distancia media entre todos los pares de puntos en los dos clusters. 
        Ofrece un equilibrio entre los métodos 'single' y 'complete', y a menudo produce resultados razonables sin ser demasiado sensible a los valores atípicos.
        * 'centroid': Fusiona los dos clusters que tienen la menor distancia entre sus centroides. 
        Menos sensible a los valores atípicos que 'single' o 'complete', pero puede sufrir del problema de inversión: los clusters más pequeños 
        pueden fusionarse más tarde de lo deseado.
'''

plt.figure(figsize=(10, 7))

'''
    - linked: Proporciona data de enlace generada por linkage(). 
    Contiene info sobre las distancias (o similitudes) entre los clusters en cada etapa del proceso y cómo se combinan en cada paso.
    - orientation='top': Raíces en la parte superior y las hojas se extienden hacia abajo. 
    - labels=iris.target: Labels asigna etiquetas a las hojas del dendrograma.
    - distance_sort='descending': Ordenación de las ramas en el dendrograma. 
    Con 'descending', las ramas que se fusionaron más tarde, se colocan más cerca de la parte superior del dendrograma (puede ayudar 
    a identificar los principales grupos o divisiones dentro de los datos).
    - show_leaf_counts=True: Añade números entre paréntesis al final de las etiquetas de las hojas para indicar el número de 
    puntos de datos originales contenidos en el cluster. Útil para entender el tamaño de los clusters sin contar manualmente las hojas del dendrograma.
'''
dendrogram(linked,
         orientation='top',
         labels=iris.target,
         distance_sort='descending',
         show_leaf_counts=True)
plt.title("Dendograma de Clustering Jerárquico");
plt.show()
