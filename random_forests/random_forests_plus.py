# Librerías conocidas
import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split

# Librerías nuevas
from sklearn.preprocessing import MinMaxScaler
from sklearn.ensemble import RandomForestClassifier

'''
Big dataset: anonym data from credit cards:
    - Clase: indicates whether the transaction was legitimate(0) or fraudulent(1)
    - We want to identify fraudulent transactions, strong relationships between all the columns and Clase column. We'll to know
    if the transaction will be fraudulent or not.
'''
df = pd.read_csv('credit_cards.csv')
df.head()

# Normalize the data: all data will be between 0 and 1
escala = MinMaxScaler(feature_range=(0, 1))
normado = escala.fit_transform(df)
df_normado = pd.DataFrame(data=normado, columns=df.columns)
df_normado.head()

# Independent variable = new DF without the Clase column
X = df_normado.drop("Clase", axis=1)
# Dependent variable
y = df_normado["Clase"]
# train and test
X_entrena, X_prueba, y_entrena, y_prueba = train_test_split(X, y, train_size=0.7, random_state=42)

forest = RandomForestClassifier()
forest.fit(X_entrena, y_entrena)
RandomForestClassifier() #this will take several minutes

# Is it reliable?
print(forest.score(X_prueba, y_prueba)) # Output: 0.9995786664794073

# High score: a good model to identify fraudulent transactions. Strong relationships between all the columns and Clase column.

# Now we can predict if a transaction will be fraudulent or not (probability)
nuevo_registro = pd.DataFrame({
    'Duracion': [0.000006], 'V1': [0.452345], 'V2': [0.564789], 'V3': [0.123456], 'V4': [0.654321],
    'V5': [0.987654], 'V6': [0.345678], 'V7': [0.234567], 'V8': [0.876543], 'V9': [0.456789],
    'V10': [0.567890], 'V11': [0.678901], 'V12': [0.789012], 'V13': [0.890123], 'V14': [0.901234],
    'V15': [0.012345], 'V16': [0.543210], 'V17': [0.432109], 'V18': [0.321098], 'V19': [0.210987],
    'V20': [0.109876], 'V21': [0.098765], 'V22': [0.887654], 'V23': [0.776543], 'V24': [0.665432],
    'V25': [0.554321],     'V26': [0.443210], 'V27': [0.332109], 'V28': [0.221098], 'Monto': [0.110987]
}, index=[0])
predict_class = forest.predict(nuevo_registro)
print(predict_class)
probabilities = forest.predict_proba(nuevo_registro)
print(probabilities) # Output: array([[0.62, 0.38]]). Only 62% of probability that the transaction will be legal


print("Predicted class: ", predict_class[0])
print("Legitimacy Probabilities: ", probabilities[0][0])
print("Fraud Probabilities: ", probabilities[0][1])


