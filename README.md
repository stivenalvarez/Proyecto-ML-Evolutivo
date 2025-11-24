
# 🔬 Optimización Evolutiva de Hiperparámetros en KNN con PyGAD

Repositorio correspondiente a la actividad práctica del curso  
**Machine Learning Evolutivo: Computación Evolutiva en Clasificadores ML**  
Universidad de Cundinamarca.

Este proyecto implementa un **Algoritmo Genético (AG)** para optimizar los
hiperparámetros del clasificador **KNN (K-Nearest Neighbors)** usando la
librería **PyGAD**, siguiendo las instrucciones de la presentación del docente.

---

## 📘 1. Clasificador elegido y justificación

Se eligió **KNN** como modelo base debido a que:

- Su rendimiento depende fuertemente de *k* y del esquema de pesos.
- Es un modelo simple e interpretativo, ideal para un ejercicio práctico.
- Permite demostrar claramente la utilidad de un Algoritmo Genético en la
  búsqueda de hiperparametrización óptima.

---

## ⚙️ 2. Hiperparámetros optimizados

| Hiperparámetro | Tipo | Rango |
|----------------|------|--------|
| `n_neighbors`  | entero | 1–30 |
| `weights` | categórico | `uniform`, `distance` |

Codificación utilizada por el AG:

```
[individuo] = [k, weights_idx]
k -> entero de 1 a 30
weights_idx -> 0 = 'uniform', 1 = 'distance'
```

---

## 🧬 3. Configuración del Algoritmo Genético (PyGAD)

- `sol_per_pop = 20`
- `num_generations = 20`
- `num_parents_mating = 5`
- `gene_space = [range(1, 31), [0, 1]]`
- Métrica objetivo (fitness): **Accuracy promedio CV con 3-fold**

---

## 📊 4. Resultados

Tras ejecutar el experimento en Google Colab, se obtuvieron los siguientes resultados:

### 🔹 Modelo baseline (sin optimizar)
- `k = 5`
- `weights = 'uniform'`
- **Accuracy en test:** `0.9211`

### 🔹 Modelo optimizado con Algoritmo Genético (PyGAD)
- `k óptimo = 7`
- `weights óptimo = 'distance'`
- **Accuracy promedio CV (fitness):** `0.9642`
- **Accuracy en test optimizado:** `0.9737`

📈 **Conclusión:**  
El modelo optimizado supera al modelo baseline, pasando de un accuracy de **0.9211**
a **0.9737**, lo cual demuestra que el Algoritmo Genético encontró una configuración
más efectiva para KNN.

---

## ▶️ 5. Ejecución del script

### Instalar dependencias:
```bash
pip install scikit-learn pygad matplotlib pandas
```

### Ejecutar:
```bash
python script.py
```

---

## 📂 6. Archivos incluidos

| Archivo | Descripción |
|---------|-------------|
| `script.py` | Implementación del AG + KNN + evaluación |
| `README.md` | Documentación completa del proyecto |
| `resultados_knn_ga.csv` | Resultados del baseline y del modelo optimizado |

---

## 👨‍🎓 7. Autor
Stiven David Alvarez Olmos
Proyecto desarrollado como parte de la actividad práctica del curso  
**Machine Learning Evolutivo**, Universidad de Cundinamarca.

