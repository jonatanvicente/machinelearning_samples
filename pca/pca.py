import numpy as np
import matplotlib.pyplot as plt
from sklearn.decomposition import PCA
from sklearn.preprocessing import StandardScaler

# Generate some sample data
np.random.seed(0)
X = np.random.rand(100, 5)

# Standardize the data
scaler = StandardScaler()
X_scaled = scaler.fit_transform(X)

# Create a PCA instance
pca = PCA(n_components=2)

# Fit the model to the data
X_pca = pca.fit_transform(X_scaled)

# Plot the results
plt.scatter(X_pca[:, 0], X_pca[:, 1], s=50, cmap='viridis')
plt.title('PCA of Sample Data')
plt.xlabel('Principal Component 1')
plt.ylabel('Principal Component 2')
plt.show()