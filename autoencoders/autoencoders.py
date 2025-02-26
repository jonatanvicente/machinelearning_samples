import numpy as np
from keras.layers import Input, Dense
from keras.models import Model
import matplotlib.pyplot as plt

# Generate some sample data
np.random.seed(0)
X = np.random.rand(1000, 20)

# Define the size of the encoding
encoding_dim = 10

# Input placeholder
input_data = Input(shape=(X.shape[1],))

# Encoder layer
encoded = Dense(encoding_dim, activation='relu')(input_data)

# Decoder layer
decoded = Dense(X.shape[1], activation='sigmoid')(encoded)

# Autoencoder model
autoencoder = Model(input_data, decoded)

# Encoder model
encoder = Model(input_data, encoded)

# Compile the autoencoder
autoencoder.compile(optimizer='adam', loss='binary_crossentropy')

# Train the autoencoder
autoencoder.fit(X, X, epochs=50, batch_size=256, shuffle=True, validation_split=0.2)

# Encode and decode some data
encoded_data = encoder.predict(X)
decoded_data = autoencoder.predict(X)

# Plot the original and reconstructed data
plt.figure(figsize=(10, 4))
plt.subplot(1, 2, 1)
plt.scatter(X[:, 0], X[:, 1], s=50, cmap='viridis')
plt.title('Original Data')
plt.subplot(1, 2, 2)
plt.scatter(decoded_data[:, 0], decoded_data[:, 1], s=50, cmap='viridis')
plt.title('Reconstructed Data')
plt.show()