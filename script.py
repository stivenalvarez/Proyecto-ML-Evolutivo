"""
script.py
Optimización de hiperparámetros de KNN usando un Algoritmo Genético (PyGAD).

Este script cumple con la actividad práctica de Machine Learning Evolutivo:
- Clasificador elegido: KNN.
- Hiperparámetros a optimizar: n_neighbors (k) y weights.
- Fitness con validación cruzada (3-fold).
- Algoritmo genético implementado con PyGAD.
"""

# =========================
# 1. Importar librerías
# =========================

import numpy as np

from sklearn.datasets import load_iris
from sklearn.model_selection import train_test_split, cross_val_score
from sklearn.preprocessing import StandardScaler
from sklearn.neighbors import KNeighborsClassifier
from sklearn.metrics import accuracy_score, ConfusionMatrixDisplay

import matplotlib.pyplot as plt
import pandas as pd

import pygad  # pip install pygad


# =========================
# 2. Cargar y preparar datos
# =========================

# Cargamos el dataset Iris (3 clases de flores, 4 características numéricas).
iris = load_iris()
X = iris.data       # Características
y = iris.target     # Etiquetas (0, 1, 2)

# Dividimos en entrenamiento y prueba (75% train, 25% test)
X_train, X_test, y_train, y_test = train_test_split(
    X,
    y,
    test_size=0.25,
    random_state=42,
    stratify=y  # mantiene proporciones de clase
)

# Escalamos las características (muy importante para KNN)
scaler = StandardScaler()
X_train = scaler.fit_transform(X_train)
X_test = scaler.transform(X_test)

print("Dimensiones X_train:", X_train.shape)
print("Dimensiones X_test :", X_test.shape)


# =========================
# 3. Modelo base (baseline) KNN
# =========================

# Definimos un KNN "por defecto" para tener con qué comparar
baseline_k = 5
baseline_weights = "uniform"

knn_baseline = KNeighborsClassifier(
    n_neighbors=baseline_k,
    weights=baseline_weights
)

# Entrenamos y evaluamos en test
knn_baseline.fit(X_train, y_train)
y_pred_baseline = knn_baseline.predict(X_test)
baseline_accuracy = accuracy_score(y_test, y_pred_baseline)

print(f"\n[BASELINE] KNN con k={baseline_k}, weights='{baseline_weights}'")
print(f"Accuracy en test (baseline): {baseline_accuracy:.4f}")

# (Opcional) Matriz de confusión del modelo base
fig, ax = plt.subplots(figsize=(5, 4))
ConfusionMatrixDisplay.from_predictions(
    y_test,
    y_pred_baseline,
    display_labels=iris.target_names,
    cmap=plt.cm.Blues,
    ax=ax
)
ax.set_title("Matriz de confusión - KNN baseline")
plt.tight_layout()
plt.show()


# =========================
# 4. Definir fitness para el AG (PyGAD)
# =========================

# Espacio de hiperparámetros:
# - k en [1, 30]
# - weights en {'uniform', 'distance'}
WEIGHTS_OPTIONS = ["uniform", "distance"]

def fitness_func(ga_instance, solution, solution_idx):
    """
    Calcula la aptitud (fitness) de una solución del GA.

    Params
    ------
    ga_instance : instancia de pygad.GA (no la usamos, pero PyGAD la pasa).
    solution    : arreglo con los genes [k, weights_idx].
    solution_idx: índice de la solución en la población (tampoco lo usamos).

    Devuelve
    --------
    float: precisión media (accuracy) en validación cruzada 3-fold.
    """
    # Decodificar los genes
    k = int(solution[0])              # n_neighbors
    weights_idx = int(solution[1])    # índice 0 o 1
    weights = WEIGHTS_OPTIONS[weights_idx]

    # Asegurarnos de que k sea al menos 1 por seguridad
    if k < 1:
        k = 1

    # Definir el modelo con estos hiperparámetros
    model = KNeighborsClassifier(n_neighbors=k, weights=weights)

    # Validación cruzada en el conjunto de entrenamiento
    scores = cross_val_score(
        model,
        X_train,
        y_train,
        cv=3,               # 3-fold mínimo como pide la actividad
        scoring="accuracy"
    )

    # Fitness = accuracy promedio
    fitness = scores.mean()

    return fitness


# =========================
# 5. Configurar el AG con PyGAD
# =========================

# gene_space define qué valores puede tomar cada gen:
# - gen 0 (k): enteros de 1 a 30
# - gen 1 (weights_idx): 0 o 1
gene_space = [
    range(1, 31),   # posibles valores para k
    [0, 1]          # índice de weights: 0 -> 'uniform', 1 -> 'distance'
]

# Parámetros del GA (dentro de los rangos sugeridos en clase)
num_generations = 20       # entre 10 y 30 generaciones
num_parents_mating = 5
sol_per_pop = 20           # tamaño de la población
num_genes = 2              # [k, weights_idx]

ga_instance = pygad.GA(
    num_generations=num_generations,
    num_parents_mating=num_parents_mating,
    fitness_func=fitness_func,
    sol_per_pop=sol_per_pop,
    num_genes=num_genes,
    gene_space=gene_space,
    # Parámetros por defecto de selección, cruce y mutación son suficientes
    # para este skeleton.
)


# =========================
# 6. Ejecutar el algoritmo genético
# =========================

print("\nIniciando optimización con Algoritmo Genético (PyGAD)...")
ga_instance.run()
print("Optimización finalizada.\n")

# (Opcional) Ver curva de fitness por generación
try:
    ga_instance.plot_fitness()
    plt.show()
except Exception as e:
    print("No se pudo graficar la curva de fitness:", e)


# =========================
# 7. Extraer la mejor solución encontrada
# =========================

best_solution, best_fitness, best_solution_idx = ga_instance.best_solution()

best_k = int(best_solution[0])
best_weights_idx = int(best_solution[1])
best_weights = WEIGHTS_OPTIONS[best_weights_idx]

print("=== Mejores hiperparámetros encontrados por el AG ===")
print(f"  n_neighbors (k): {best_k}")
print(f"  weights       : {best_weights}")
print(f"  Accuracy CV (fitness): {best_fitness:.4f}")


# =========================
# 8. Evaluar en test el mejor modelo encontrado
# =========================

knn_optimized = KNeighborsClassifier(
    n_neighbors=best_k,
    weights=best_weights
)

knn_optimized.fit(X_train, y_train)
y_pred_optimized = knn_optimized.predict(X_test)
optimized_accuracy = accuracy_score(y_test, y_pred_optimized)

print("\n=== Comparación en conjunto de prueba ===")
print(f"  KNN baseline  (k={baseline_k}, weights='{baseline_weights}'): "
      f"{baseline_accuracy:.4f}")
print(f"  KNN optimizado (k={best_k}, weights='{best_weights}'): "
      f"{optimized_accuracy:.4f}")

# Matriz de confusión del modelo optimizado
fig, ax = plt.subplots(figsize=(5, 4))
ConfusionMatrixDisplay.from_predictions(
    y_test,
    y_pred_optimized,
    display_labels=iris.target_names,
    cmap=plt.cm.Greens,
    ax=ax
)
ax.set_title("Matriz de confusión - KNN optimizado")
plt.tight_layout()
plt.show()


# =========================
# 9. Guardar resultados a CSV
# =========================

results_df = pd.DataFrame({
    "modelo": ["KNN_baseline", "KNN_optimizado"],
    "n_neighbors": [baseline_k, best_k],
    "weights": [baseline_weights, best_weights],
    "accuracy_test": [baseline_accuracy, optimized_accuracy],
    "accuracy_cv": [np.nan, best_fitness]  # CV solo para el optimizado
})

results_filename = "resultados_knn_ga.csv"
results_df.to_csv(results_filename, index=False)
print(f"\nArchivo de resultados guardado en: {results_filename}")
