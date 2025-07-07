#include "knn.h"
#include <stdlib.h>
#include <float.h>
#include <math.h>

static double manhattan_distance(const double* x1, const double* x2, int n) {
    double sum = 0.0;
    for (int i = 0; i < n; i++) {
        sum += fabs(x1[i] - x2[i]);
    }
    return sum;
}

static double cosine_distance(const double* x1, const double* x2, int n) {
    double dot = 0.0, norm1 = 0.0, norm2 = 0.0;
    for (int i = 0; i < n; i++) {
        dot += x1[i] * x2[i];
        norm1 += x1[i] * x1[i];
        norm2 += x2[i] * x2[i];
    }
    if (norm1 == 0.0 || norm2 == 0.0) return 1.0;
    return 1.0 - (dot / (sqrt(norm1) * sqrt(norm2)));
}

// Estructura auxiliar para almacenar distancia e índice
typedef struct {
    double dist;
    int idx;
} Neighbor;

static int compare_neighbors(const void* a, const void* b) {
    double diff = ((Neighbor*)a)->dist - ((Neighbor*)b)->dist;
    return (diff > 0) - (diff < 0);
}

KNNClassifier* knn_create(int k, KNNDistanceMetric metric, KNNVoteType vote_type) {
    KNNClassifier* knn = (KNNClassifier*)malloc(sizeof(KNNClassifier));
    if (!knn) return NULL;
    knn->k = k;
    knn->X_train = NULL;
    knn->y_train = NULL;
    knn->metric = metric;
    knn->vote_type = vote_type;
    return knn;
}

void knn_fit(KNNClassifier* knn, Matrix* X, Matrix* y) {
    if (!knn) return;
    knn->X_train = X;
    knn->y_train = y;
}

Matrix* knn_predict(KNNClassifier* knn, Matrix* X) {
    if (!knn || !knn->X_train || !knn->y_train || !X) return NULL;
    int n_samples = X->rows;
    int n_train = knn->X_train->rows;
    int n_features = X->cols;
    int k = knn->k;

    Matrix* y_pred = matrix_create(n_samples, 1);
    if (!y_pred) return NULL;

    Neighbor* neighbors = (Neighbor*)malloc(n_train * sizeof(Neighbor));
    if (!neighbors) {
        matrix_free(y_pred);
        return NULL;
    }

    for (int i = 0; i < n_samples; i++) {
        // Calcular distancias a todos los puntos de entrenamiento
        for (int j = 0; j < n_train; j++) {
            if (knn->metric == KNN_EUCLIDEAN) {
                neighbors[j].dist = euclidean_distance(X->data[i], knn->X_train->data[j], n_features);
            } else if (knn->metric == KNN_MANHATTAN) {
                neighbors[j].dist = manhattan_distance(X->data[i], knn->X_train->data[j], n_features);
            } else if (knn->metric == KNN_COSINE) {
                neighbors[j].dist = cosine_distance(X->data[i], knn->X_train->data[j], n_features);
            }
            neighbors[j].idx = j;
        }
        // Ordenar vecinos por distancia
        qsort(neighbors, n_train, sizeof(Neighbor), compare_neighbors);

        int n_classes = 32; // O calcula dinámicamente el número de clases si lo prefieres

        if (knn->vote_type == KNN_VOTE_WEIGHTED) {
            // Votación ponderada por distancia
            double* class_weights = (double*)calloc(n_classes, sizeof(double));
            for (int n = 0; n < k; n++) {
                int label = (int)knn->y_train->data[neighbors[n].idx][0];
                double weight = 1.0 / (neighbors[n].dist + 1e-5); // Evita división por cero
                class_weights[label] += weight;
            }
            int pred_class = 0;
            double max_weight = class_weights[0];
            for (int c = 1; c < n_classes; c++) {
                if (class_weights[c] > max_weight) {
                    max_weight = class_weights[c];
                    pred_class = c;
                }
            }
            y_pred->data[i][0] = pred_class;
            free(class_weights);
        } else {
            // Votación mayoritaria (como antes)
            int* class_count = (int*)calloc(n_classes, sizeof(int));
            for (int n = 0; n < k; n++) {
                int label = (int)knn->y_train->data[neighbors[n].idx][0];
                class_count[label]++;
            }
            int max_votes = 0, pred_class = 0;
            for (int c = 0; c < n_classes; c++) {
                if (class_count[c] > max_votes) {
                    max_votes = class_count[c];
                    pred_class = c;
                }
            }
            y_pred->data[i][0] = pred_class;
            free(class_count);
        }
    }
    free(neighbors);
    return y_pred;
}

void knn_free(KNNClassifier* knn) {
    if (!knn) return;
    // No liberamos X_train ni y_train porque no son propiedad del clasificador
    free(knn);
}