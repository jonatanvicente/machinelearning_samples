import numpy as np


# Creamos array numérico bidiemensional
A = np.array([[1, 2], [3, 4], [5, 6]])

# Aplicamos SVD
U, sigma, VT = np.linalg.svd(A)

# Descomposición en las 3 matrices
print(U)
print(sigma)
print(VT)

