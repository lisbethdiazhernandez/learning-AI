# Things I seem to always forget

```python
    y_pred_probs = logreg.predict_proba(X_test)[:, 1]
```

Qué signigica el [:, 1]
Es slicing en dos dimensiones, separado por la coma:

antes de la coma (:) → "todas las filas" (el : solito significa todo)
después de la coma (1) → "solo la columna en índice 1"

Entonces [:, 1] = "dame todas las filas, pero solo la segunda columna".