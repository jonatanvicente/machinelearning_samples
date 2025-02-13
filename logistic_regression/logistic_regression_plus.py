import pandas as pd
import matplotlib.pyplot as plt
from sklearn.linear_model import LogisticRegression
from sklearn.model_selection import train_test_split

df = pd.read_csv("load_data.csv")
df.head()

plt.scatter(df.edad,
           df.compra);

'''
Menores de 40 años raramente compran.
Separamos info en conjuntos de datos para training y para testing
'''

# train_test_split() mezcla los registros y los separa en dos grupos.
# Train_size: 90% de los datos para entrenar, 10% para probar (por defecto usa 80-20)
# Así podemos comprobar el funcionamiento del modelo en base a los elementos que tenemos.
# Comprobamos el funcionamiento de nuestro modelo en base a los datos que tenemos. Lo entrenamos
# con una gran parte de nuestros datos, y ponemos a prueba sus predicciones con la otra parte más pequeña
# para medir la cantidad de aciertos o errores que ha conseguido (evaluamos para info añadida posterior).

X_entrena, X_prueba, y_entrena, y_prueba = train_test_split(
    df[["edad"]], df["compra"], train_size=0.9)

modelo = LogisticRegression()
modelo.fit(X_entrena, y_entrena)
# evaluamos desempeño del modelo
print(modelo.score(X_entrena, y_entrena)) # Output: 0.875, 87.5%

# añadimos nueva data
datos_nuevos = pd.DataFrame({"edad":[25, 35, 45, 55]})
probabilidades = modelo.predict_proba(datos_nuevos) # devuelve 4 tuplas, una por registro con probabilidades de compra
prob_compra = probabilidades[:, 1] # nos quedamos con la segunda columna de cada tupla

# Plot the results
#plt.scatter(df.edad, df.compra, color="blue", label='Age');
plt.scatter(datos_nuevos["edad"], prob_compra, color="red", label="Purchase probabilities");
plt.xlabel('Age')
plt.ylabel('Purchase probabilities')
plt.title('Logistic Regression')
plt.legend()
plt.show()