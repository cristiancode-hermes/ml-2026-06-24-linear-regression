# Learnings — 2026-06-24

## Qué se hizo

Proyecto de Regresión Lineal con scikit-learn usando el dataset de TechStore.
- Se cargaron datos del dataset_completo.csv (proyecto merge-join)
- Se agregaron 155 filas de productos a 120 pedidos únicos
- Se crearon features: num_productos, precio_promedio, precio_max, precio_min, etc.
- Se aplicó one-hot encoding a variables categóricas (segmento, ciudad, estado)
- Se imputaron valores nulos en puntuacion_promedio con la media
- Se dividió en train/test (80/20)
- Se entrenó un modelo LinearRegression
- Se evaluó con R², MAE, RMSE
- Se interpretaron los coeficientes
- Se comparó con un modelo simple de 1 feature
- Se hizo una predicción nueva (inferencia)
- Se generaron 5 visualizaciones

## Conceptos nuevos

- [x] Regresión Lineal con scikit-learn (LinearRegression)
- [x] Train/Test Split (train_test_split)
- [x] R² (coeficiente de determinación)
- [x] MAE (Mean Absolute Error)
- [x] RMSE (Root Mean Squared Error)
- [x] Coeficientes e intercepto del modelo
- [x] One-hot encoding con pd.get_dummies()
- [x] Imputación de nulos con la media
- [x] Predicción (inferencia) con .predict()
- [x] Comparación de modelos (completo vs simple)
- [x] Residuos y diagnóstico del modelo

## Dificultades encontradas

- Los datos originales están a nivel de producto, no de pedido → hubo que agregar
- 75/120 pedidos (62.5%) no tienen reseña → imputación necesaria, no se podía eliminar
- Dataset pequeño (120 pedidos) → el R² puede no ser muy alto, pero es realista
- Algunas variables dummy podrían no aparecer si la categoría no está en train

## Notas para el próximo proyecto

- Próximo skill Nivel 3: Clasificación con KNN o Árboles de Decisión
- Introducir validación cruzada (cross_val_score)
- Mostrar matriz de confusión y métricas de clasificación
- Usar el mismo dataset de TechStore pero con target categórico
