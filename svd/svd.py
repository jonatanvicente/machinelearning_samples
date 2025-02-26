import numpy as np

# Generate a random matrix
np.random.seed(0)
A = np.random.rand(4, 5)

# Perform SVD
U, S, VT = np.linalg.svd(A, full_matrices=False)

# Reconstruct the original matrix
Sigma = np.diag(S)
A_reconstructed = np.dot(U, np.dot(Sigma, VT))

print("Original Matrix:\n", A)
print("Reconstructed Matrix:\n", A_reconstructed)