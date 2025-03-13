
import matplotlib.pyplot as plt
import numpy as np
from keras.api.datasets import mnist #dataset con imagenes de digitos escritos a mano. IMagenes B/N de 28x28 pixeles
from keras.api.models import Sequential #modelo secuencial de deep-learning
from keras.api.layers import Dense, Flatten
from keras.api.utils import to_categorical

(images_training, tags_training), (images_test, tags_test) = mnist.load_data()

# normalizar valores: escalamos todos los valores en rango de 0-1
# ya que los valores de arrays de arriba son de 0-255, dividimos por 255
images_training = images_training / 255.0
images_test = images_test / 255.0

# tags son valores 0-9 (indican el nº representado en cada image)
# to_categorical() transforma los valores en un array de 10 elementos (sólo uno de ellos es 1 y los demás 0)
tags_training = to_categorical(tags_training)
tags_test = to_categorical(tags_test)

# definimos el modelo
'''
Los modelos secuenciales: pila lineal de capas donde cada capa tiene exactamente un tensor de entrada y un tensor de salida. 
    - 1ª capa (Flatten): sin parámetros, reformatea input data. Transforma las imágenes de entrada de 28x28 píxeles en un vector de 784 píxeles (multiplicar 28x28). 
        Es necesario porque la siguiente capa (Dense) necesitará un solo vector de entrada y no una matriz de dos dimensiones.
    - 2ª capa (Dense, densa o completamente conectada): tiene 128 neuronas. Cada neurona en esta capa está conectada a todas las entradas de la capa anterior
        (las 784 unidades de info del vector aplanado). La función de activación relu (Rectified Linear Unit), se aplicará a la salida de cada neurona. 
        relu es muy común, ayuda a evitar problemas en el aprendizaje y permite que el modelo aprenda más rápido.
    - 3ª capa (Dense). Capa de salida. Tiene 10 neuronas, corresponden con las 10 clases de dígitos (que van del 0 al 9). 
        La función de activación softmax se utiliza convertir los valores de las neuronas en probabilidades que suman 1. 
        Cada salida puede interpretarse como la probabilidad de que la entrada pertenezca a una de las 10 clases.
'''
model = Sequential([
    Flatten(input_shape=(28, 28)),
    Dense(128, activation='relu'),
    Dense(10, activation='softmax')
])
# Compilamos el modelo
'''
    - optimizer: algoritmo que ajusta los pesos de la red para minimizar la función de pérdida. adam es popular (adaptable y eficiente).
    - loss: objetivo, cómo de bien está prediciendo el modelo. Objetivo: reducir la pérdida (si sucede, indicaría que el modelo está mejorando
        su aprendizaje). En este caso, usamos categorical_crossentropy, que mide la diferencia entre las etiquetas reales y las predicciones.
    - metrics. Medición del avance, evaluación del rendimiento. accuracy proporciona evaluación muy clara y directa del rendimiento en clasificación.
'''
model.compile(optimizer='adam', loss='categorical_crossentropy', metrics=['accuracy'])
# Training
model.fit(images_training, tags_training, epochs=5, validation_data=(images_test, tags_test))

predictions = model.predict(images_test)


def show_image(predictions, real_tag, img):
    real_tag, img = real_tag.argmax(), img.squeeze()
    plt.grid(False)
    plt.xticks([])
    plt.yticks([])

    plt.imshow(img, cmap=plt.cm.binary)

    predicted_tag = np.argmax(predictions)
    if predicted_tag == real_tag:
        color = 'blue'
    else:
        color = 'red'

    plt.xlabel(f'Pred: {predicted_tag} Real: {real_tag}', color=color)

# Evaluación de cómo está prediciendo el modelo
rows = 5
columns = 3
total_imgs = rows * columns
plt.figure(figsize=(2 * 2 * columns, 2 * rows))

for i in range(total_imgs):
    plt.plot(rows, 2 * columns, 2 * i + 1)
    plt.show()
    show_image(predictions[i], tags_test[i], images_test[i]);