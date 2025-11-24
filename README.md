[readme.md](https://github.com/user-attachments/files/23701162/readme.md)

# Stiven David Alvarez Olmos
# Optimización Evolutiva de Hiperparámetros en KNN con PyGAD

## 1. Descripción general

Este trabajo implementa un **Algoritmo Genético (AG)** para optimizar los
hiperparámetros de un clasificador **K-Nearest Neighbors (KNN)** en el
dataset **Iris** (incluido en `sklearn.datasets`).

Se sigue la actividad propuesta en la presentación de
*Machine Learning Evolutivo: Computación Evolutiva en Clasificadores ML*:
elegir un clasificador, definir hiperparámetros y rangos, implementar
`fitness_func` con validación cruzada y configurar PyGAD para ejecutar
entre 10 y 30 generaciones.  

## 2. Clasificador elegido y justificación

- **Modelo**: `KNeighborsClassifier` de `sklearn`.
- **Motivación**:
  - Es un modelo sencillo e intuitivo (clasifica por mayoría entre los
    vecinos más cercanos).
  - Es muy sensible a la elección de *k* y al esquema de pesos, lo cual
    lo hace un buen candidato para optimización de hiperparámetros.
  - Permite ilustrar claramente el efecto de los hiperparámetros en el
    rendimiento del modelo.

## 3. Hiperparámetros y rangos de búsqueda

Se optimizan **2 hiperparámetros**:

1. `n_neighbors (k)`  
   - Tipo: entero.  
   - Rango: `1` a `30`.  
   - Intuición: valores pequeños tienden a sobreajustar; valores muy
     grandes pueden subajustar.

2. `weights`  
   - Tipo: categórico.  
   - Valores posibles: `['uniform', 'distance']`.  
   - Intuición: con `uniform` todos los vecinos pesan igual; con
     `distance` los vecinos cercanos tienen más influencia.

Estos hiperparámetros se codifican en el AG como un **individuo**:

```text
[individuo] = [k, weights_idx]
    k           -> entero en [1, 30]
    weights_idx -> 0 = 'uniform', 1 = 'distance'
```

## 4. Algoritmo Genético (PyGAD)

- Librería: `pygad.GA`.
- **Fitness**: accuracy promedio en validación cruzada 3-fold sobre
  `X_train, y_train` (función `cross_val_score`).
- Parámetros principales del GA:
  - `num_generations = 20`
  - `num_parents_mating = 5`
  - `sol_per_pop = 20`
  - `num_genes = 2`
  - `gene_space = [range(1, 31), [0, 1]]`

Con esto, cada generación contiene 20 soluciones distintas de
hiperparámetros, y el AG selecciona, cruza y muta soluciones para
mejorar la precisión media.

## 5. Resultados (resumen)

> **Nota:** los valores concretos dependen de cada ejecución del GA.
> A continuación se dejan campos para que el grupo los complete tras
> ejecutar el script.

- **Baseline KNN**  
  - Configuración: `k = 5`, `weights = 'uniform'`  
  - Accuracy en test: `____`

- **KNN optimizado con AG (PyGAD)**  
  - Hiperparámetros encontrados:
    - `k = ____`
    - `weights = 'uniform' / 'distance'`
  - Accuracy CV (fitness): `____`
  - Accuracy en test: `____`

En general, se espera que el KNN optimizado logre una precisión igual o
ligeramente superior al baseline, mostrando que la búsqueda evolutiva
en el espacio de hiperparámetros sí tiene impacto en el desempeño.

## 6. Cómo ejecutar

1. Instalar dependencias (por ejemplo, en Google Colab):

   ```bash
   pip install scikit-learn pygad matplotlib pandas
   ```

2. Ejecutar el script:

   ```bash
   python script.py
   ```

   Esto:
   - Entrena el KNN baseline.
   - Ejecuta el AG con PyGAD.
   - Entrena el KNN optimizado.
   - Muestra las métricas y matrices de confusión.
   - Guarda un archivo `resultados_knn_ga.csv` con un resumen.
