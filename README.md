# Algoritmos Fundamentales de Machine Learning en C

Este repositorio contiene la implementación modular en C de tres algoritmos fundamentales de Machine Learning:

- **K-Vecinos Más Cercanos (KNN)**
- **K-Means**
- **Regresión Lineal** (con opción Ridge)

Incluye utilidades para la lectura de archivos CSV, manejo de matrices y ejemplos de uso con el conjunto de datos Iris.

---

## Estructura del Proyecto

```
Tarea4-Algoritmos/
├── Makefile
├── data/
│   └── iris.csv
├── docs/
│   ├── informe.pdf
├── src/
│   ├── algorithms/
│   │   ├── knn.c / knn.h
│   │   ├── kmeans.c / kmeans.h
│   │   ├── linear_regression.c / linear_regression.h
│   ├── core/
│   │   ├── matrix.c / matrix.h
│   ├── utils/
│   │   ├── csv_reader.c / csv_reader.h
│   ├── main.c
│   └── main-iris.c
└── README.md
```

---

## Compilación y Ejecución

### Requisitos

- GCC (C99 o superior)
- Make
- Sistema operativo Linux o Windows

### Compilar todo el proyecto

```bash
make
```

### Ejecutar los ejemplos

- **Ejemplo general con datos sintéticos:**
  ```bash
  ./ml_demo
  ```
- **Ejemplo con el conjunto de datos Iris:**
  ```bash
  ./ml_iris
  ```

---

## Descripción de los Algoritmos

### K-Vecinos Más Cercanos (KNN)
- Clasificación supervisada.
- Soporta métricas Euclidiana, Manhattan y Coseno.
- Incluye votación ponderada por distancia.

### K-Means
- Clustering no supervisado.
- Inicialización K-Means++.
- Cálculo de índice de silueta para evaluar la calidad del agrupamiento.

### Regresión Lineal
- Ajuste por ecuaciones normales.
- Opción de regularización Ridge para evitar sobreajuste.
- Cálculo de MSE y $R^2$.

---

## Ejemplo de Uso

```bash
# Compilar
make

# Ejecutar ejemplo con Iris
./ml_iris

# Ejecutar ejemplo general
./ml_demo
```

---

## Documentación

- El informe técnico `docs/`.

---

## Créditos

- Autor: Manuel González
- Universidad de Magallanes, 2025

---

## Referencias

- Bishop, C. M. (2006). *Pattern Recognition and Machine Learning*. Springer.
