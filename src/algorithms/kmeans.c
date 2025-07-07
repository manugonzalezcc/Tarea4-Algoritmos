#include "kmeans.h"
#include <float.h>
#include <math.h>
#include <stdlib.h>
#include <time.h>
#include "../core/matrix.h"

KMeans* kmeans_create(int k) {
    KMeans* model = (KMeans*)malloc(sizeof(KMeans));
    if (!model) return NULL;
    model->k = k;
    model->centroids = NULL;
    return model;
}

// Inicialización K-Means++
void kmeans_init_plus_plus(Matrix* X, Matrix* centroids) {
    int n_samples = X->rows;
    int n_features = X->cols;
    int k = centroids->rows;
    int* chosen = (int*)calloc(n_samples, sizeof(int));
    srand((unsigned int)time(NULL));

    // Elige el primer centroide al azar
    int idx = rand() % n_samples;
    for (int f = 0; f < n_features; f++)
        centroids->data[0][f] = X->data[idx][f];
    chosen[idx] = 1;

    for (int c = 1; c < k; c++) {
        // Calcula la distancia mínima al centroide más cercano para cada punto
        double* min_dist_sq = (double*)malloc(n_samples * sizeof(double));
        double total = 0.0;
        for (int i = 0; i < n_samples; i++) {
            if (chosen[i]) {
                min_dist_sq[i] = 0.0;
                continue;
            }
            double min_dist = DBL_MAX;
            for (int j = 0; j < c; j++) {
                double dist = euclidean_distance(X->data[i], centroids->data[j], n_features);
                if (dist < min_dist) min_dist = dist;
            }
            min_dist_sq[i] = min_dist * min_dist;
            total += min_dist_sq[i];
        }
        // Elige el siguiente centroide con probabilidad proporcional a min_dist_sq
        double r = ((double)rand() / RAND_MAX) * total;
        double sum = 0.0;
        int next_idx = 0;
        for (int i = 0; i < n_samples; i++) {
            if (chosen[i]) continue;
            sum += min_dist_sq[i];
            if (sum >= r) {
                next_idx = i;
                break;
            }
        }
        for (int f = 0; f < n_features; f++)
            centroids->data[c][f] = X->data[next_idx][f];
        chosen[next_idx] = 1;
        free(min_dist_sq);
    }
    free(chosen);
}

int kmeans_fit(KMeans* model, Matrix* X, int max_iter, double tol) {
    (void)tol;
    if (!model || !X) return 0;
    int n_samples = X->rows;
    int n_features = X->cols;
    int k = model->k;

    // Inicializar centroides usando K-Means++
    model->centroids = matrix_create(k, n_features);
    if (!model->centroids) return 0;
    kmeans_init_plus_plus(X, model->centroids);

    int* labels = (int*)malloc(n_samples * sizeof(int));
    if (!labels) return 0;

    for (int iter = 0; iter < max_iter; iter++) {
        int changed = 0;

        // Asignar cada punto al centroide más cercano
        for (int i = 0; i < n_samples; i++) {
            double min_dist = INFINITY;
            int best = 0;
            for (int c = 0; c < k; c++) {
                double dist = euclidean_distance(X->data[i], model->centroids->data[c], n_features);
                if (dist < min_dist) {
                    min_dist = dist;
                    best = c;
                }
            }
            if (labels[i] != best) changed = 1;
            labels[i] = best;
        }

        // Recalcular centroides
        for (int c = 0; c < k; c++) {
            int count = 0;
            for (int j = 0; j < n_features; j++)
                model->centroids->data[c][j] = 0.0;
            for (int i = 0; i < n_samples; i++) {
                if (labels[i] == c) {
                    for (int j = 0; j < n_features; j++)
                        model->centroids->data[c][j] += X->data[i][j];
                    count++;
                }
            }
            if (count > 0) {
                for (int j = 0; j < n_features; j++)
                    model->centroids->data[c][j] /= count;
            }
        }

        // Verificar convergencia
        if (!changed) break;
    }

    free(labels);
    return 1;
}

Matrix* kmeans_predict(KMeans* model, Matrix* X) {
    if (!model || !model->centroids || !X) return NULL;
    int n_samples = X->rows;
    int n_features = X->cols;
    int k = model->k;

    Matrix* labels = matrix_create(n_samples, 1);
    if (!labels) return NULL;

    for (int i = 0; i < n_samples; i++) {
        double min_dist = INFINITY;
        int best = 0;
        for (int c = 0; c < k; c++) {
            double dist = euclidean_distance(X->data[i], model->centroids->data[c], n_features);
            if (dist < min_dist) {
                min_dist = dist;
                best = c;
            }
        }
        labels->data[i][0] = best;
    }
    return labels;
}

double kmeans_inertia(KMeans* model, Matrix* X) {
    if (!model || !model->centroids || !X) return 0.0;
    int n_samples = X->rows;
    int n_features = X->cols;
    int k = model->k;
    double inertia = 0.0;

    for (int i = 0; i < n_samples; i++) {
        double min_dist = INFINITY;
        for (int c = 0; c < k; c++) {
            double dist = euclidean_distance(X->data[i], model->centroids->data[c], n_features);
            if (dist < min_dist) min_dist = dist;
        }
        inertia += min_dist * min_dist;
    }
    return inertia;
}

void kmeans_free(KMeans* model) {
    if (!model) return;
    if (model->centroids) matrix_free(model->centroids);
    free(model);
}

// Calcula la distancia euclidiana entre dos puntos
/*static double euclidean_distance(const double* x1, const double* x2, int n) {
    double sum = 0.0;
    for (int i = 0; i < n; i++) {
        double diff = x1[i] - x2[i];
        sum += diff * diff;
    }
    return sqrt(sum);
}*/

// Calcula el índice de silueta para clustering
double kmeans_silhouette(Matrix* X, Matrix* labels, Matrix* centroids) {
    int n_samples = X->rows;
    int n_features = X->cols;
    int k = centroids->rows;
    double total_silhouette = 0.0;

    for (int i = 0; i < n_samples; i++) {
        int cluster_i = (int)labels->data[i][0];
        double a = 0.0;
        int a_count = 0;
        double b = DBL_MAX;

        // Calcular a (distancia promedio a su propio cluster)
        for (int j = 0; j < n_samples; j++) {
            if (i == j) continue;
            if ((int)labels->data[j][0] == cluster_i) {
                a += euclidean_distance(X->data[i], X->data[j], n_features);
                a_count++;
            }
        }
        if (a_count > 0) a /= a_count;
        else a = 0.0; // Si está solo en su cluster

        // Calcular b (menor distancia promedio a otro cluster)
        for (int c = 0; c < k; c++) {
            if (c == cluster_i) continue;
            double dist_sum = 0.0;
            int count = 0;
            for (int j = 0; j < n_samples; j++) {
                if ((int)labels->data[j][0] == c) {
                    dist_sum += euclidean_distance(X->data[i], X->data[j], n_features);
                    count++;
                }
            }
            if (count > 0) {
                double avg = dist_sum / count;
                if (avg < b) b = avg;
            }
        }
        if (b == DBL_MAX) b = 0.0; // Si no hay otros clusters

        double s = 0.0;
        if (a == 0.0 && b == 0.0) {
            s = 0.0; // Silueta indefinida, se asigna 0
        } else {
            s = (b - a) / fmax(a, b);
        }
        total_silhouette += s;
    }
    return total_silhouette / n_samples;
}