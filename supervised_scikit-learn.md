# Supervised Learning with Scikit-Learn

---

## Classification

### K-Nearest Neighbors (KNN)

Usa **majority voting** — mira los k vecinos más cercanos y el label que tenga más votos gana.

#### Predicting on unlabeled data

```python
X_new = np.array([[56.8, 17.5],
                  [24.4, 24.1],
                  [50.1, 10.9]])
print(X_new.shape)  # (3, 2) → 3 observations, 2 features
```

```python
predictions = knn.predict(X_new)
print('Predictions: {}'.format(predictions))
# Predictions: [1 0 0]
# 1 = churn, 0 = no churn
```

#### Example — Creating the classifier

```python
from sklearn.neighbors import KNeighborsClassifier

y = churn_df["churn"].values
X = churn_df[["account_length", "customer_service_calls"]].values

knn = KNeighborsClassifier(6)  # 6 vecinos

# Fit the classifier to the data
knn.fit(X, y)

# Predict new data
y_pred = knn.predict(X_new)
print("Predictions: {}".format(y_pred))
```

---

### Measuring Accuracy

El flujo general es:

```
Split data → Training set → Fit/train classifier → Calculate accuracy using test set
                 ↓                                         ↑
             Test set ─────────────────────────────────────┘
```

#### Train-test split

```python
from sklearn.model_selection import train_test_split

# X = todas las columnas excepto "churn" → lo que usamos para predecir
# y = la columna "churn" → lo que queremos predecir
X = churn_df.drop("churn", axis=1).values
y = churn_df["churn"].values

X_train, X_test, y_train, y_test = train_test_split(
    X, y, test_size=0.2, random_state=42, stratify=y
)
# 80% → para entrenar (X_train, y_train)
# 20% → para probar qué tan bien aprendió (X_test, y_test)
# random_state=42 → hace que la división sea siempre igual, como barajar
#                   las cartas siempre igual. Why 42? It's a meme lol xd
# stratify=y → asegura que la proporción de churn/no churn sea similar en ambos grupos
```

```python
knn = KNeighborsClassifier(n_neighbors=5)
knn.fit(X_train, y_train)
print(knn.score(X_test, y_test))  # % de predicciones correctas
```

Con `test_size=0.3` y 6 neighbors:
```python
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.3, random_state=21, stratify=y)
knn = KNeighborsClassifier(n_neighbors=6)
knn.fit(X_train, y_train)
print(knn.score(X_test, y_test))
# 0.8800599700149925
```

---

### Model Complexity

| k | Complexity | Risk |
|---|-----------|------|
| **Large k** | Less complex model | Underfitting |
| **Small k** | More complex model | Overfitting |

#### Checking under and over fitting

```python
neighbors = np.arange(1, 13)
train_accuracies = {}
test_accuracies = {}

for neighbor in neighbors:
    knn = KNeighborsClassifier(n_neighbors=neighbor)
    knn.fit(X_train, y_train)
    train_accuracies[neighbor] = knn.score(X_train, y_train)
    test_accuracies[neighbor] = knn.score(X_test, y_test)

print(neighbors, '\n', train_accuracies, '\n', test_accuracies)
```

#### Visualizing model complexity

```python
import matplotlib.pyplot as plt

plt.title("KNN: Varying Number of Neighbors")
plt.plot(neighbors, train_accuracies.values(), label="Training Accuracy")
plt.plot(neighbors, test_accuracies.values(), label="Testing Accuracy")
plt.legend()
plt.xlabel("Number of Neighbors")
plt.ylabel("Accuracy")
plt.show()
```

> Cuando las dos curvas se alejan mucho entre sí → señal de overfitting o underfitting.
> El sweet spot está donde el test accuracy es máximo sin que el train accuracy baje mucho.

---

## Regression

### Reshape

Cuando tenemos una sola feature (columna), scikit-learn necesita que X sea 2D (filas = observaciones, columna = la feature). Por eso hacemos reshape.

```python
import numpy as np

X = sales_df["radio"].values
y = sales_df["sales"].values

# Reshape: de (n,) a (n, 1) — una columna
X = np.array(X).reshape(-1, 1)

print(X.shape, y.shape)  # (n, 1) (n,)
```

> `-1` le dice a numpy "calcula tú cuántas filas hay", y `1` es el número de columnas.
> Es como doblar una lista en una tabla de una sola columna.

---

### Linear Regression

#### Building the model

```python
from sklearn.linear_model import LinearRegression

reg = LinearRegression()
reg.fit(X, y)

predictions = reg.predict(X)
print(predictions[:5])
```

#### Visualizar la regresión lineal

```python
import matplotlib.pyplot as plt

plt.scatter(X, y, color="blue")   # puntos reales
plt.plot(X, predictions, color="red")  # línea predicha
plt.xlabel("Radio Expenditure ($)")
plt.ylabel("Sales ($)")
plt.show()
```

#### Con train-test split y múltiples features

```python
X = sales_df.drop("sales", axis=1).values
y = sales_df["sales"].values

X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.3, random_state=42)

reg = LinearRegression()
reg.fit(X_train, y_train)

y_pred = reg.predict(X_test)
print("Predictions: {}, Actual Values: {}".format(y_pred[:2], y_test[:2]))
```

#### Calcular el error

```python
from sklearn.metrics import root_mean_squared_error

r_squared = reg.score(X_test, y_test)  # R² — qué tan bien el modelo explica la varianza
rmse = root_mean_squared_error(y_test, y_pred)  # error promedio en las mismas unidades que y

print("R^2: {}".format(r_squared))
print("RMSE: {}".format(rmse))
```

> **R²** cercano a 1 = el modelo explica bien los datos.
> **RMSE** más bajo = las predicciones están más cerca de los valores reales.
> Son complementarios — úsalos juntos para evaluar el modelo.

---

## Cross-Validation (CV)

En vez de depender de un solo split, CV divide los datos en N partes (folds) y entrena/evalúa N veces.
Así el score es más confiable y menos dependiente del azar del split.

```python
from sklearn.model_selection import cross_val_score, KFold

kf = KFold(n_splits=6, shuffle=True, random_state=42)
reg = LinearRegression()
cv_results = cross_val_score(reg, X, y, cv=kf)
```

#### Analizar los resultados del CV

```python
print(np.mean(cv_results))    # promedio de los scores
print(np.std(cv_results))     # qué tan variables fueron los scores entre folds
print(np.quantile(cv_results, [0.025, 0.975]))  # intervalo de confianza del 95%
```

#### Cómo calcular el intervalo de confianza

Si quieres el intervalo de confianza del N%:

```
(100% - N%) / 2 → resultado / 100
rango = [resultado, 1 - resultado]
```

Ejemplo para 95%:
```
(100 - 95) / 2 = 2.5 → 0.025
rango = [0.025, 0.975]
```

```
|─────2.5%─────|──────────95%──────────|─────2.5%─────|
0            0.025                   0.975             1
```

> Más folds = estimación más estable pero más lenta de computar.
> 5 o 10 folds es el estándar de la industria.
