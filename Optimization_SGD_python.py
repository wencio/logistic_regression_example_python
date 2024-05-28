import numpy as np
import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.linear_model import SGDClassifier
from sklearn.metrics import log_loss, accuracy_score, classification_report

# Crear un DataFrame de ejemplo
data = {
    'feature1': [2, 3, 5, 7, 11, 13, 17, 19, 23, 29],
    'feature2': [4, 9, 25, 49, 121, 169, 289, 361, 529, 841],
    'label': [0, 0, 0, 0, 1, 1, 1, 1, 1, 1]
}
df = pd.DataFrame(data)

# Separar características y etiquetas
X = df[['feature1', 'feature2']]
y = df['label']

# Dividir los datos en conjuntos de entrenamiento y prueba
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.3, random_state=42)

# Crear el modelo de regresión logística usando gradiente descendente estocástico
sgd_model = SGDClassifier(loss='log_loss', max_iter=1000, tol=1e-3, random_state=42)

# Entrenar el modelo
sgd_model.fit(X_train, y_train)

# Hacer predicciones de probabilidades
y_pred_prob = sgd_model.predict_proba(X_test)

# Calcular log-loss
loss = log_loss(y_test, y_pred_prob)

# Hacer predicciones de clases
y_pred = sgd_model.predict(X_test)

# Evaluar el modelo
accuracy = accuracy_score(y_test, y_pred)
classification_report_str = classification_report(y_test, y_pred)

print("Log-Loss:", loss)
print("Accuracy:", accuracy)
print("Classification Report:\n", classification_report_str)
