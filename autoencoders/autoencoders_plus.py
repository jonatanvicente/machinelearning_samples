'''
Autoencoders intentan aprender una representación comprimida de los datos de entrada.
Lo reduce a una representación de menor dimensión
Después reconstruye los datos de vuelta a su forma original o a una representación cercana a ella
'''
import tensorflow as tf
import matplotlib.pyplot as plt
from keras.api.layers import Input, Dense
from keras.api.models import Model
from sklearn.model_selection import train_test_split
from sklearn.datasets import load_digits

print('TensorFlow version:', tf.__version__)
'''
Iris es un diccionario que contiene, entre otras cosas, una clave data, cuyo valor es una colección de arrays, 
que contiene 1797 imágenes de 8 pixels de ancho por 8 pixels de alto (representan dígitos escritos a mano). 
Contiene arrays de dos dimensiones cuyo contenido son números que representan el valor de gris de cada pixel
(info para crear mapas de bits que forman imágenes)
'''
digitos = load_digits() # Contiene info para cargar imágenes

### Ejemplo
print(digitos['data'][0].reshape(8, 8)) # 8x8 matriz de pixeles
# Cada elemento del array representa un bit de imagen (su valor indica el nivel de gris)
plt.imshow(digitos['data'][0].reshape(8, 8))
# Mostramos el número 4
plt.imshow(digitos.images[4]);
###

X = digitos.data # Necesitamos una variable con todas las variables de dígitos
# No necesitaremos variable Y

# Normalizamos (todos deben empezar y terminar en el mismo rango)
X = X / 16.0 # (Valores de 0 - 16 para cada píxel) Al dividir por 16, los valores estarán entre 0 y 1 (es como reducir la escala de un mapa)
'''
1.- Normalizar es importante para que el modelo funcione correctamente:
    - Previene problemas númericos durante el entrenamiento, como el desbordamiento de números
    - Permite que el algoritmo de optimización converja más rápido
'''
X_train, X_test = train_test_split(X, test_size=0.2, random_state=42) # Dividimos los datos en entrenamiento y prueba

# Nueva capa de entrada, capa tensorial (forma específica de datosque el modelo esperará)
# En este caso, array de una sola dimensión (8x8=64)
imagen_entrada = Input(shape=(64, ))

'''
2.- Creamos variable encoder que contendrá la red neuronal:
    - Contendra capa Dense (muchos puntos de info conectados entre sí)
    - 32 neuronas (número de nodos en la capa) -> la mitad de 64 pixeles en este caso. 
    - No excesivamente compleja
    - Activation = relu ('Rectified Linear Unit') -> Función de activación
'''
encoder = Dense(32, activation='relu') (imagen_entrada)
'''
3.- Creamos decoder:
    - Deseamos 64 nodos (generalmente igual a los puntos de entrada originales -64 pixels-)
    - Activation = sigmoid. Convierte los valores de entrada a valores entre 0 y 1
'''
decoder = Dense(64, activation='sigmoid') (encoder)

'''
4.- Creamos autoencoder y lo entrenamos:
    - optimizer = 'adam' (algoritmo de optimización). Ajusta los pesos del modelo. Adam ajusta la tasa
    de aprendizaje automáticamente 
    - loss = 'binary_crossentropy' (función de pérdida). Mide la diferencia entre la salida predicha y la salida real
    binary_crossentropy es elección común cuando datos de entrada son binarios entre 0-1. ESta función de pérdida
    compara cada píxel de la imagen de entrada con el de la imagen reconstruida
'''

autoencoder = Model(imagen_entrada, decoder)
autoencoder.compile(optimizer='adam',
                   loss='binary_crossentropy')


'''
    - X_train: Mismo conjunto de datos input/output esperadas. Objetivo: reconstruir las entradas originales a partir de las representaciones comprimidas.
    - epochs=100: Iteraciones que repasará todo el conjunto de datos de entrenamiento.
    - batch_size=256: Tamaño de lotes de entrenamiento. El modelo debería tomar 256 ejemplos de X_train, procesarlos, actualizar sus pesos 
    y luego pasar al siguiente. El uso de lotes ayuda a hacer el entrenamiento más eficiente. 
    La elección de 256 como tamaño del lote es una decisión basada en la experiencia práctica, funciona como un punto de partida razonable. 
    - shuffle=True: Los datos se deben mezclar antes de cada epoch. 
    Esto ayuda a prevenir que el modelo aprenda el orden de los datos en lugar de las características subyacentes, mejorando aún más la generalización.
    - validation_data: Además de entrenar el modelo, monitorizamos su desempeño en un conjunto de datos que no ha visto durante el entrenamiento.
'''
autoencoder.fit(X_train,
               X_train,
               epochs=100,
               batch_size=256,
               shuffle=True,
               validation_data=(X_test, X_test))

# Visualizamos input / output
for i in range(10):
    plt.subplot(2,
                10,
                i + 1)
    plt.imshow(X_test[i].reshape(8, 8))
    #plt.show()

    plt.subplot(2,
                10,
                i + 1 + 10)
    plt.imshow(autoencoder.predict(X_test)[i].reshape(8, 8));
    plt.show()