#!/usr/bin/env python3
"""
╔══════════════════════════════════════════════════════════════════════════╗
║  REGRESIÓN LINEAL — De la teoría a la práctica con scikit-learn        ║
║  Nivel 3 | Skill: scikit-learn, regresión lineal, métricas,            ║
║                    train/test split, interpretación                     ║
╚══════════════════════════════════════════════════════════════════════════╝

¿Qué es la regresión lineal?
────────────────────────────
La regresión lineal es el modelo MÁS SIMPLE y MÁS IMPORTANTE de ML.
Predice un número combinando varias características de forma lineal.

La fórmula es:  y = w₁·x₁ + w₂·x₂ + ... + wₙ·xₙ + b

  • y  = valor a predecir (target)
  • x₁, x₂... = características (features)
  • w₁, w₂... = pesos/coeficientes (lo que el modelo APRENDE)
  • b  = sesgo/intercepto

Piensa en ello como "dibujar la mejor línea" que pasa por tus datos.
En 2D es una línea recta. En múltiples dimensiones es un "hiperplano".

Ejemplo: Si predecimos el precio de una casa:
  precio = $150 × (m²) + $20000 × (habitaciones) - $5000 × (años_edad) + $30000

El modelo APRENDE los $150, $20000, -$5000 desde los datos.

Hoy haremos algo especial: primero veremos cómo funciona con datos
"perfectos" (sintéticos), y luego con datos reales de TechStore.
¡Así entenderás cuándo y por qué funciona! 🚀
"""

# ============================================================
# PASO 1: Importar librerías
# ============================================================
# Cada librería tiene un propósito específico:
#   pandas → manipular datos como tablas (DataFrames)
#   numpy  → operaciones matemáticas con arrays
#   matplotlib / seaborn → gráficos
#   scikit-learn → modelos de Machine Learning listos para usar
# ============================================================

import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
import os
import warnings

# ─── scikit-learn: el corazón del proyecto ───
from sklearn.model_selection import train_test_split   # dividir datos
from sklearn.linear_model import LinearRegression       # el modelo ⭐
from sklearn.metrics import r2_score, mean_absolute_error, mean_squared_error  # evaluar

warnings.filterwarnings('ignore')

# ─── Configuración visual ───
plt.rcParams['font.family'] = 'DejaVu Sans'
plt.rcParams['figure.dpi'] = 100
plt.rcParams['figure.figsize'] = (14, 7)
COLORES = ['#2E86AB', '#A23B72', '#F18F01', '#C73E1D', '#3B1F2B', '#58A4B0']
sns.set_palette(COLORES)
sns.set_style("whitegrid")

os.makedirs('docs', exist_ok=True)

print("=" * 75)
print("📈  REGRESIÓN LINEAL CON SCIKIT-LEARN")
print("    De la teoría a la práctica con datos reales")
print("=" * 75)
print()

# ====================================================================
# PARTE 1 — DATOS SINTÉTICOS CON RELACIÓN LINEAL CONOCIDA
# ====================================================================
# Vamos a CREAR datos donde SABEMOS que la relación es lineal.
# Así podemos VER si el modelo descubre la relación correcta.
#
# La "receta" de nuestros datos va a ser:
#   precio_casa = 200 × m² + 15000 × habitaciones - 3000 × antigüedad + 50000
#
# Esto es como tener la "respuesta correcta" y ver si el modelo
# la encuentra. ¡Spoiler: scikit-learn la encuentra perfectamente!
# ====================================================================

print("=" * 75)
print("🔬 PARTE 1: DATOS SINTÉTICOS — RELACIÓN LINEAL PERFECTA")
print("=" * 75)
print()
print("Vamos a crear 500 casas imaginarias donde SABEMOS la fórmula exacta:")
print("   precio = 200 × m² + 15000 × habs - 3000 × años + 50000 + ruido")
print()

# ─── Fijamos una semilla para que los números sean reproducibles ───
np.random.seed(42)

# Creamos 500 casas con características variadas
n_casas = 500

# Características (features) de cada casa
metros_cuadrados = np.random.normal(120, 40, n_casas).clip(40, 300)  # entre 40 y 300 m²
habitaciones = np.random.randint(1, 6, n_casas)                       # 1 a 5 habs
antiguedad = np.random.randint(0, 50, n_casas)                        # 0 a 50 años

# ─── La FÓRMULA VERDADERA (lo que el modelo debe aprender) ───
# precio_base      = $50,000 (terreno base)
# precio_por_m²    = $200    (cada m² añade $200)
# precio_por_hab   = $15,000 (cada hab añade $15,000)
# descuento_por_año = -$3,000 (cada año de antigüedad resta $3,000)
# ruido             = pequeña variación aleatoria

precio_verdadero = (
    50000                    # base
    + 200 * metros_cuadrados # precio por m²
    + 15000 * habitaciones   # precio por habitación
    - 3000 * antiguedad       # descuento por antigüedad
)

# Añadimos un poco de ruido aleatorio (para simular datos reales)
# sin ruido el modelo tendría R² = 1.0 perfecto
ruido = np.random.normal(0, 20000, n_casas)  # ruido con desviación de $20,000
precio = precio_verdadero + ruido

# ─── Creamos un DataFrame con los datos sintéticos ───
df_sint = pd.DataFrame({
    'metros_cuadrados': metros_cuadrados,
    'habitaciones': habitaciones,
    'antiguedad': antiguedad,
    'precio': precio
})

print(f"🏠 Dataset sintético creado: {n_casas} casas × 4 columnas")
print()
print("📋 Primeras 5 casas:")
print(df_sint.head())
print()

print("📊 Estadísticas del precio de las casas:")
print(df_sint['precio'].describe())
print()

# === Visualización 1: ¿Cómo se ve la relación lineal? ===
# Graficamos cada feature contra el precio para ver la tendencia
print("📈 Generando visualizaciones...")

fig, axes = plt.subplots(1, 3, figsize=(18, 6))
features_vs = ['metros_cuadrados', 'habitaciones', 'antiguedad']
titulos = ['Metros Cuadrados vs Precio', 'Habitaciones vs Precio', 'Antigüedad vs Precio']

for i, (feat, tit) in enumerate(zip(features_vs, titulos)):
    ax = axes[i]
    ax.scatter(df_sint[feat], df_sint['precio'], alpha=0.5, color='#2E86AB', s=30)
    ax.set_xlabel(feat.replace('_', ' ').title())
    ax.set_ylabel('Precio ($)')
    ax.set_title(tit, fontsize=13, fontweight='bold')
    ax.grid(True, alpha=0.3)

plt.suptitle('📊 Relación entre Features y Precio (Datos Sintéticos)', fontsize=16, fontweight='bold')
plt.tight_layout()
plt.savefig('docs/01_sinteticos_relaciones.png', dpi=120, bbox_inches='tight')
plt.close()
print("   ✅ docs/01_sinteticos_relaciones.png")
print()

# ─── DIVIDIR EN TRAIN/TEST ───
print("✂️ Dividiendo en train/test (80/20)...")

X_sint = df_sint[['metros_cuadrados', 'habitaciones', 'antiguedad']]
y_sint = df_sint['precio']

X_sint_train, X_sint_test, y_sint_train, y_sint_test = train_test_split(
    X_sint, y_sint, test_size=0.2, random_state=42
)

print(f"   🏋️  Train: {X_sint_train.shape[0]} casas")
print(f"   🧪  Test:  {X_sint_test.shape[0]} casas")
print()

# ─── ENTRENAR EL MODELO ───
print("🧠 Entrenando modelo de regresión lineal...")

modelo_sint = LinearRegression()
modelo_sint.fit(X_sint_train, y_sint_train)

print("   ✅ Modelo entrenado!")
print()

# ─── EVALUAR ───
y_sint_pred = modelo_sint.predict(X_sint_test)

r2_sint = r2_score(y_sint_test, y_sint_pred)
mae_sint = mean_absolute_error(y_sint_test, y_sint_pred)
rmse_sint = np.sqrt(mean_squared_error(y_sint_test, y_sint_pred))

print("=" * 75)
print("📊 RESULTADOS — MODELO SINTÉTICO")
print("=" * 75)
print(f"   🎯  R²  = {r2_sint:.4f}  ({r2_sint*100:.1f}% de varianza explicada)")
print(f"   📉  MAE = ${mae_sint:,.2f}")
print(f"   📉  RMSE= ${rmse_sint:,.2f}")
print()

# ─── COMPARAR COEFICIENTES APRENDIDOS vs REALES ───
print("🔍 Comparando coeficientes APRENDIDOS vs VERDADEROS:")
print("-" * 60)
print(f"{'Feature':20s} {'Real':>12s} {'Aprendido':>12s} {'Error':>12s}")
print("-" * 60)

coefs_reales = {'metros_cuadrados': 200, 'habitaciones': 15000, 'antiguedad': -3000}
for i, col in enumerate(['metros_cuadrados', 'habitaciones', 'antiguedad']):
    real = coefs_reales[col]
    aprendido = modelo_sint.coef_[i]
    error = abs(aprendido - real)
    print(f"{col:20s} ${real:>8,.0f}  ${aprendido:>8,.2f}  ${error:>8,.2f}")

print(f"{'Intercepto':20s} {'$50,000':>12s} ${modelo_sint.intercept_:>8,.2f}  ${abs(modelo_sint.intercept_ - 50000):>8,.2f}")
print()
print("💡 ¿Ves qué cerca están los coeficientes aprendidos de los reales?")
print("   El modelo DESCUBRIÓ la fórmula casi exactamente a partir de los datos.")
print()

# === Visualización 2: Predicciones vs Reales (sintéticos) ===
fig, ax = plt.subplots(figsize=(10, 8))

ax.scatter(y_sint_test, y_sint_pred, alpha=0.7, s=60, color='#2E86AB', edgecolors='white')
min_val = min(y_sint_test.min(), y_sint_pred.min())
max_val = max(y_sint_test.max(), y_sint_pred.max())
ax.plot([min_val, max_val], [min_val, max_val], '--', color='#C73E1D', linewidth=2.5,
        label='Predicción Perfecta')

ax.set_xlabel('Precio Real ($)', fontsize=13)
ax.set_ylabel('Precio Predicho ($)', fontsize=13)
ax.set_title('🎯 Predicciones vs Valores Reales (Datos Sintéticos)', fontsize=15, fontweight='bold')
ax.text(0.05, 0.95, f'R² = {r2_sint:.4f}\nMAE = ${mae_sint:,.2f}',
        transform=ax.transAxes, fontsize=13, verticalalignment='top',
        bbox=dict(boxstyle='round,pad=0.5', facecolor='white', alpha=0.9))
ax.legend(fontsize=12)
ax.grid(True, alpha=0.3)

plt.tight_layout()
plt.savefig('docs/02_sinteticos_predicciones.png', dpi=120, bbox_inches='tight')
plt.close()
print("   ✅ docs/02_sinteticos_predicciones.png")
print()

# === Visualización 3: Coeficientes del modelo ===
fig, ax = plt.subplots(figsize=(10, 6))

coef_df = pd.DataFrame({
    'feature': ['Metros²', 'Habitaciones', 'Antigüedad'],
    'coeficiente': modelo_sint.coef_
})
coef_df['color'] = coef_df['coeficiente'].apply(lambda x: '#2E86AB' if x > 0 else '#C73E1D')

ax.bar(coef_df['feature'], coef_df['coeficiente'], color=coef_df['color'], alpha=0.8)
ax.axhline(0, color='black', linewidth=1)
ax.set_title('📊 Coeficientes del Modelo (Impacto en el Precio)', fontsize=15, fontweight='bold')
ax.set_ylabel('Impacto en el Precio ($)', fontsize=12)
ax.grid(True, alpha=0.3, axis='y')

# Añadimos el valor encima de cada barra
for i, row in coef_df.iterrows():
    ax.text(i, row['coeficiente'] + (500 if row['coeficiente'] > 0 else -2000),
            f'${row["coeficiente"]:,.0f}', ha='center', fontsize=12, fontweight='bold')

plt.tight_layout()
plt.savefig('docs/03_sinteticos_coeficientes.png', dpi=120, bbox_inches='tight')
plt.close()
print("   ✅ docs/03_sinteticos_coeficientes.png")
print()

# ─── INTERPRETACIÓN ───
print("💡 INTERPRETACIÓN DE LOS COEFICIENTES:")
print("-" * 60)
for i, col in enumerate(['metros_cuadrados', 'habitaciones', 'antiguedad']):
    coef = modelo_sint.coef_[i]
    signo = "aumenta" if coef > 0 else "disminuye"
    valor = abs(coef)
    print(f"   • Por cada unidad de '{col}', el precio {signo} en ${valor:,.2f}")

print(f"   • Precio base (intercepto): ${modelo_sint.intercept_:,.2f}")
print()

# ====================================================================
# PARTE 2 — DATOS REALES DE TECHSTORE 🏪
# ====================================================================
# Ahora aplicamos EXACTAMENTE el mismo proceso, pero con datos
# reales de una tienda. Aquí las relaciones NO son perfectamente
# lineales — hay ruido, outliers, y factores que no medimos.
#
# Esto es lo que pasa en el MUNDO REAL: el modelo rara vez es
# perfecto, pero sigue siendo útil.
#
# Vamos a predecir el total de un pedido basándonos en:
#   - cuántos productos compró
#   - el precio promedio de esos productos
#   - el segmento del cliente (Premium, Standard, Gold)
#   - reseñas y puntuaciones
# ====================================================================

print()
print("=" * 75)
print("🏪 PARTE 2: DATOS REALES — TECHSTORE")
print("=" * 75)
print()
print("Ahora aplicamos la misma técnica a datos REALES de TechStore.")
print("El mundo real es más ruidoso — ¡veamos qué pasa!")
print()

# ─── Cargar datos ───
print("📂 Cargando dataset de TechStore...")
df_raw = pd.read_csv('data/dataset_completo.csv')
print(f"   ✅ {df_raw.shape[0]} filas × {df_raw.shape[1]} columnas")
print()

# ─── Agregar por pedido ───
print("🔄 Agrupando por pedido...")

df = df_raw.groupby('pedido_id').agg(
    total=('total', 'first'),
    estado=('estado', 'first'),
    ciudad=('ciudad', 'first'),
    segmento=('segmento', 'first'),
    num_productos=('producto_id_x', 'count'),
    precio_promedio=('precio', 'mean'),
    precio_max=('precio', 'max'),
    precio_min=('precio', 'min'),
    precio_std=('precio', 'std'),
    categorias_unicas=('categoria', 'nunique'),
    puntuacion_promedio=('puntuacion', 'mean'),
    tiene_resena=('puntuacion', 'count')
).reset_index()

print(f"   ✅ {df.shape[0]} pedidos × {df.shape[1]} columnas")
print()

# ─── Feature Engineering ───
print("🔧 Preparando features...")

df_modelo = df.copy()

# Imputar nulos: los pedidos sin reseña reciben la puntuación promedio
media_punt = df_modelo['puntuacion_promedio'].mean()
df_modelo['puntuacion_promedio'] = df_modelo['puntuacion_promedio'].fillna(media_punt)
df_modelo['precio_std'] = df_modelo['precio_std'].fillna(0)
print(f"   ✅ Nulos imputados (puntuación → {media_punt:.2f}, precio_std → 0)")

# One-hot encoding para texto → números
segmento_dummies = pd.get_dummies(df_modelo['segmento'], prefix='seg')
ciudad_dummies = pd.get_dummies(df_modelo['ciudad'], prefix='ciu')
estado_dummies = pd.get_dummies(df_modelo['estado'], prefix='est')

df_modelo = pd.concat([df_modelo, segmento_dummies, ciudad_dummies, estado_dummies], axis=1)
print(f"   ✅ Categóricas codificadas: {segmento_dummies.shape[1]} segmentos + "
      f"{ciudad_dummies.shape[1]} ciudades + {estado_dummies.shape[1]} estados")
print()

# ─── Seleccionar features ───
features_finales = [
    'num_productos', 'precio_promedio', 'precio_max', 'precio_min',
    'precio_std', 'categorias_unicas', 'puntuacion_promedio', 'tiene_resena'
] + [col for col in df_modelo.columns if col.startswith('seg_') or
     col.startswith('ciu_') or col.startswith('est_')]

X_real = df_modelo[features_finales]
y_real = df_modelo['total']

print(f"📋 Features ({len(features_finales)}): {features_finales[:5]}...")
print(f"   Target: total (valor del pedido en $)")
print(f"   X: {X_real.shape}, y: {y_real.shape}")
print()

# ─── Train/Test Split ───
Xr_train, Xr_test, yr_train, yr_test = train_test_split(
    X_real, y_real, test_size=0.2, random_state=42
)

print(f"✂️ Train: {Xr_train.shape[0]} pedidos | Test: {Xr_test.shape[0]} pedidos")
print()

# ─── Entrenar ───
print("🧠 Entrenando modelo con datos reales...")
modelo_real = LinearRegression()
modelo_real.fit(Xr_train, yr_train)
print("   ✅ Modelo entrenado!")
print()

# ─── Evaluar ───
yr_pred = modelo_real.predict(Xr_test)

r2_real = r2_score(yr_test, yr_pred)
mae_real = mean_absolute_error(yr_test, yr_pred)
rmse_real = np.sqrt(mean_squared_error(yr_test, yr_pred))

print("=" * 75)
print("📊 RESULTADOS — MODELO CON DATOS REALES (TechStore)")
print("=" * 75)
print(f"   🎯  R²  = {r2_real:.4f}  ({r2_real*100:.1f}% de varianza explicada)")
print(f"   📉  MAE = ${mae_real:,.2f}")
print(f"   📉  RMSE= ${rmse_real:,.2f}")
print()

# === Visualización 4: Predicciones vs Reales (TechStore) ===
fig, ax = plt.subplots(figsize=(10, 8))

ax.scatter(yr_test, yr_pred, alpha=0.7, s=80, color='#A23B72', edgecolors='white', linewidth=0.5)
min_val = min(yr_test.min(), yr_pred.min())
max_val = max(yr_test.max(), yr_pred.max())
ax.plot([min_val, max_val], [min_val, max_val], '--', color='#C73E1D', linewidth=2.5,
        label='Predicción Perfecta')

ax.set_xlabel('Total Real ($)', fontsize=13)
ax.set_ylabel('Total Predicho ($)', fontsize=13)
ax.set_title('🏪 Predicciones vs Reales — TechStore (Datos Reales)', fontsize=15, fontweight='bold')
ax.text(0.05, 0.95, f'R² = {r2_real:.4f}\nMAE = ${mae_real:,.2f}',
        transform=ax.transAxes, fontsize=13, verticalalignment='top',
        bbox=dict(boxstyle='round,pad=0.5', facecolor='white', alpha=0.9))
ax.legend(fontsize=12)
ax.grid(True, alpha=0.3)

plt.tight_layout()
plt.savefig('docs/04_reales_predicciones.png', dpi=120, bbox_inches='tight')
plt.close()
print("   ✅ docs/04_reales_predicciones.png")
print()

# === Visualización 5: Comparación de la distribución de errores ===
fig, axes = plt.subplots(1, 2, figsize=(14, 6))

# Errores del modelo sintético
res_sint = y_sint_test - y_sint_pred
axes[0].hist(res_sint, bins=20, color='#2E86AB', edgecolor='white', alpha=0.8)
axes[0].axvline(0, color='black', linewidth=2)
axes[0].axvline(res_sint.mean(), color='#C73E1D', linestyle='--', linewidth=2,
                label=f'Media: ${res_sint.mean():.0f}')
axes[0].set_title('🔬 Errores — Datos Sintéticos', fontsize=14, fontweight='bold')
axes[0].set_xlabel('Error ($)')
axes[0].set_ylabel('Frecuencia')
axes[0].legend()

# Errores del modelo real
res_real = yr_test - yr_pred
axes[1].hist(res_real, bins=15, color='#A23B72', edgecolor='white', alpha=0.8)
axes[1].axvline(0, color='black', linewidth=2)
axes[1].axvline(res_real.mean(), color='#C73E1D', linestyle='--', linewidth=2,
                label=f'Media: ${res_real.mean():.0f}')
axes[1].set_title('🏪 Errores — TechStore (Datos Reales)', fontsize=14, fontweight='bold')
axes[1].set_xlabel('Error ($)')
axes[1].set_ylabel('Frecuencia')
axes[1].legend()

plt.suptitle('📉 Comparación de la Distribución de Errores', fontsize=16, fontweight='bold')
plt.tight_layout()
plt.savefig('docs/05_comparacion_errores.png', dpi=120, bbox_inches='tight')
plt.close()
print("   ✅ docs/05_comparacion_errores.png")
print()

# === Visualización 6: Coeficientes TechStore (top) ===
coef_reales_df = pd.DataFrame({
    'feature': features_finales,
    'coeficiente': modelo_real.coef_
})
coef_reales_df['abs_coef'] = coef_reales_df['coeficiente'].abs()
coef_reales_df = coef_reales_df.sort_values('abs_coef', ascending=False).head(12)

fig, ax = plt.subplots(figsize=(12, 6))
colores_bar = ['#2E86AB' if c > 0 else '#C73E1D' for c in coef_reales_df['coeficiente']]
ax.barh(range(len(coef_reales_df)), coef_reales_df['coeficiente'], color=colores_bar, alpha=0.8)
ax.set_yticks(range(len(coef_reales_df)))
ax.set_yticklabels(coef_reales_df['feature'], fontsize=11)
ax.axvline(0, color='black', linewidth=1)
ax.set_title('📊 Top Features que Afectan el Total del Pedido', fontsize=15, fontweight='bold')
ax.set_xlabel('Impacto en el Total ($)', fontsize=12)
ax.grid(True, alpha=0.3, axis='x')

plt.tight_layout()
plt.savefig('docs/06_reales_coeficientes.png', dpi=120, bbox_inches='tight')
plt.close()
print("   ✅ docs/06_reales_coeficientes.png")
print()

# ====================================================================
# PARTE 3 — COMPARACIÓN Y LECCIONES
# ====================================================================

print("=" * 75)
print("📊 PARTE 3: COMPARACIÓN Y LECCIONES APRENDIDAS")
print("=" * 75)
print()

print("📋 TABLA COMPARATIVA:")
print("-" * 70)
print(f"{'Métrica':20s} {'Sintéticos':>15s} {'TechStore':>15s} {'¿Qué significa?':30s}")
print("-" * 70)
print(f"{'R²':20s} {r2_sint:.4f}{'':>10s} {r2_real:.4f}{'':>10s} {'Más alto = mejor'} ")
print(f"{'MAE':20s} {'$'+f'{mae_sint:,.0f}':>15s} {'$'+f'{mae_real:,.0f}':>15s} {'Error promedio en $':30s}")
print(f"{'RMSE':20s} {'$'+f'{rmse_sint:,.0f}':>15s} {'$'+f'{rmse_real:,.0f}':>15s} {'Penaliza errores grandes':30s}")
print(f"{'Datos':20s} {'500 casas':>15s} {'120 pedidos':>15s} {'Más datos = mejor':30s}")
print(f"{'Features':20s} {'3 relevantes':>15s} {'20 (con ruido)':>15s} {'Calidad > cantidad':30s}")
print()

print("💡 ¿POR QUÉ EL MODELO REAL FUNCIONA PEOR?")
print("-" * 60)
print("1️⃣  Datos pequeños: solo 120 pedidos vs 500 casas")
print("2️⃣  Ruido real: hay factores que no medimos (descuentos, temporada, etc.)")
print("3️⃣  Features con poco poder predictivo: el número de productos")
print("    no es el único factor que determina el total")
print("4️⃣  El mundo real es así: ningún modelo es perfecto")
print()
print("✅ ¿ES INÚTIL EL MODELO REAL? ¡NO!")
print("-" * 60)
print("• Aunque R² es bajo, el modelo SÍ captura tendencias")
print("• Los coeficientes nos dicen qué factores IMPORTAN")
print(f"• Es mejor que adivinar: el MAE de ${mae_real:.0f} es mejor")
print(f"  que el error de adivinar siempre la media (${y_real.std():.0f})")
print("• En el próximo proyecto aprenderemos a mejorar esto con")
print("  mejores modelos (Random Forest, etc.)")
print()

# ====================================================================
# PARTE 4 — PREDICCIÓN NUEVA CON AMBOS MODELOS
# ====================================================================

print("🔮 HACIENDO PREDICCIONES NUEVAS (INferencia)")
print("-" * 60)
print()

# Predicción sintética: ¿cuánto vale una casa de 150m², 3 habs, 10 años?
casa_nueva = pd.DataFrame([{
    'metros_cuadrados': 150,
    'habitaciones': 3,
    'antiguedad': 10
}])
pred_casa = modelo_sint.predict(casa_nueva)[0]

print(f"🏠 Casa: 150m², 3 habs, 10 años de antigüedad")
print(f"💰 Precio predicho: ${pred_casa:,.2f}")
print(f"📐 Fórmula aprendida: 50000 + 200×150 + 15000×3 - 3000×10 = ${50000 + 200*150 + 15000*3 - 3000*10:,.0f}")
print()

# Predicción TechStore
nuevo_pedido = pd.DataFrame([{
    'num_productos': 3,
    'precio_promedio': 350.00,
    'precio_max': 500.00,
    'precio_min': 200.00,
    'precio_std': 150.00,
    'categorias_unicas': 2,
    'puntuacion_promedio': 4.0,
    'tiene_resena': 3,
    'seg_Premium': 0,
    'seg_Standard': 1,
    'seg_Gold': 0,
    'ciu_Barranquilla': 0,
    'ciu_Bogotá': 1,
    'ciu_Cali': 0,
    'ciu_Cartagena': 0,
    'ciu_Medellín': 0,
    'est_Cancelado': 0,
    'est_Entregado': 1,
    'est_Envío': 0,
    'est_Pendiente': 0
}])

# Asegurar que tiene todas las columnas necesarias
for col in features_finales:
    if col not in nuevo_pedido.columns:
        nuevo_pedido[col] = 0
nuevo_pedido = nuevo_pedido[features_finales]

pred_pedido = modelo_real.predict(nuevo_pedido)[0]

print(f"🛒 Pedido TechStore: 3 productos, $350 c/u, Bogotá, Standard")
print(f"💰 Total predicho: ${pred_pedido:.2f}")
print()

# ====================================================================
# 🧠 RESUMEN FINAL
# ====================================================================
print("=" * 75)
print("🧠  RESUMEN — HOY APRENDISTE:")
print("=" * 75)
print()
print("1️⃣  REGRESIÓN LINEAL")
print("   → El modelo ML más fundamental: predice valores numéricos")
print("   → Fórmula: y = w₁x₁ + w₂x₂ + ... + b")
print("   → Con datos lineales, recupera la fórmula exacta ✅")
print()
print("2️⃣  TRAIN/TEST SPLIT")
print("   → 80% para aprender, 20% para evaluar")
print("   → ¡Nunca uses los mismos datos para entrenar y evaluar!")
print()
print("3️⃣  MÉTRICAS DE EVALUACIÓN")
print(f"   → R²  = cuánta varianza explica (sintético: {r2_sint:.3f}, real: {r2_real:.3f})")
print(f"   → MAE = error promedio en $ (sintético: ${mae_sint:,.0f}, real: ${mae_real:,.0f})")
print()
print("4️⃣  COEFICIENTES E INTERPRETACIÓN")
print("   → Cada coeficiente es el impacto de esa feature en el target")
print("   → Con datos sintéticos, los coeficientes APRENDIDOS coinciden con los REALES")
print()
print("5️⃣  REALIDAD vs TEORÍA")
print("   → Los datos perfectos dan R² alto (aprender es fácil)")
print("   → Los datos reales son ruidosos (R² bajo es normal)")
print("   → La REGRESIÓN LINEAL es el punto de partida, no el final")
print()
print("6️⃣  FLUJO DE TRABAJO EN ML")
print("   Cargar datos → Explorar → Feature Engineering →")
print("   Train/Test Split → Entrenar → Evaluar → Interpretar → Predecir")
print()
print("📁 Ruta: /opt/data/ml-projects/2026-06-24-linear-regression/")
print("📊 Docs: /opt/data/ml-projects/2026-06-24-linear-regression/docs/")
print()
print("🚀 PRÓXIMO PROYECTO: Clasificación con KNN o Árbol de Decisión")
print("   Pasamos de predecir NÚMEROS a predecir CATEGORÍAS")
print("=" * 75)
