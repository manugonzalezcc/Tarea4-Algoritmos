#ifndef KNN_H
#define KNN_H

#include "../core/matrix.h"

/**
 * @brief Tipos de votación para el clasificador KNN
 */
typedef enum {
    KNN_VOTE_MAJORITY,  ///< Votación por mayoría
    KNN_VOTE_WEIGHTED   ///< Votación ponderada
} KNNVoteType;

/**
 * @brief Métricas de distancia para el clasificador KNN
 */
typedef enum {
    KNN_EUCLIDEAN,      ///< Distancia euclidiana
    KNN_MANHATTAN,      ///< Distancia Manhattan
    KNN_COSINE          ///< Distancia coseno
} KNNDistanceMetric;

/**
 * @brief Estructura para el clasificador K-Vecinos Más Cercanos
 */
typedef struct {
    Matrix* X_train;    // Datos de entrenamiento
    Matrix* y_train;    // Etiquetas de entrenamiento
    int k;              // Número de vecinos
    KNNDistanceMetric metric; ///< Métrica de distancia a utilizar
    KNNVoteType vote_type;    ///< Tipo de votación a utilizar
} KNNClassifier;

/**
 * @brief Crea un nuevo clasificador K-Vecinos Más Cercanos
 * @param k Número de vecinos a considerar
 * @return KNNClassifier* Puntero al clasificador o NULL si hay error
 */
KNNClassifier* knn_create(int k, KNNDistanceMetric metric, KNNVoteType vote_type);

/**
 * @brief Entrena el clasificador con los datos proporcionados
 * @param knn Clasificador a entrenar
 * @param X Matriz de características de entrenamiento
 * @param y Vector de etiquetas de entrenamiento
 * @return void
 */
void knn_fit(KNNClassifier* knn, Matrix* X, Matrix* y);

/**
 * @brief Realiza predicciones con el clasificador entrenado
 * @param knn Clasificador entrenado
 * @param X Matriz de características a predecir
 * @return Matrix* Vector de predicciones o NULL si hay error
 */
Matrix* knn_predict(KNNClassifier* knn, Matrix* X);

/**
 * @brief Libera la memoria utilizada por el clasificador
 * @param knn Clasificador a liberar
 */
void knn_free(KNNClassifier* knn);

#endif // KNN_H
//*