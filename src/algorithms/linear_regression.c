#include "linear_regression.h"
#include <stdlib.h>
#include <math.h>

// Crea el modelo
LinearRegression* linear_regression_create(double learning_rate, int max_iterations, double tolerance) {
    (void)learning_rate;
    (void)max_iterations;
    (void)tolerance;
    LinearRegression* model = (LinearRegression*)malloc(sizeof(LinearRegression));
    if (!model) return NULL;
    model->weights = NULL;
    model->bias = 0.0;
    return model;
}

// Entrena usando ecuaciones normales (X^T X)w = X^T y
int linear_regression_fit(LinearRegression* model, Matrix* X, Matrix* y) {
    if (!model || !X || !y) return 0;
    int n_samples = X->rows;
    int n_features = X->cols;

    // Calcular medias para centrar los datos
    double* mean_X = (double*)calloc(n_features, sizeof(double));
    double mean_y = 0.0;
    for (int j = 0; j < n_features; j++)
        for (int i = 0; i < n_samples; i++)
            mean_X[j] += X->data[i][j];
    for (int j = 0; j < n_features; j++)
        mean_X[j] /= n_samples;
    for (int i = 0; i < n_samples; i++)
        mean_y += y->data[i][0];
    mean_y /= n_samples;

    // Calcular coeficientes (solo para 1 variable)
    double num = 0.0, den = 0.0;
    for (int i = 0; i < n_samples; i++) {
        num += (X->data[i][0] - mean_X[0]) * (y->data[i][0] - mean_y);
        den += (X->data[i][0] - mean_X[0]) * (X->data[i][0] - mean_X[0]);
    }
    double w = num / den;
    double b = mean_y - w * mean_X[0];

    if (!model->weights)
        model->weights = matrix_create(1, 1);
    model->weights->data[0][0] = w;
    model->bias = b;

    free(mean_X);
    return 1;
}

// Predice valores para nuevos datos
Matrix* linear_regression_predict(LinearRegression* model, Matrix* X) {
    if (!model || !model->weights || !X) return NULL;
    int n_samples = X->rows;
    Matrix* y_pred = matrix_create(n_samples, 1);
    for (int i = 0; i < n_samples; i++)
        y_pred->data[i][0] = model->weights->data[0][0] * X->data[i][0] + model->bias;
    return y_pred;
}

// Calcula el error cuadrático medio
double linear_regression_mse(LinearRegression* model, Matrix* X, Matrix* y) {
    Matrix* y_pred = linear_regression_predict(model, X);
    if (!y_pred) return 0.0;
    int n_samples = y->rows;
    double mse = 0.0;
    for (int i = 0; i < n_samples; i++) {
        double diff = y_pred->data[i][0] - y->data[i][0];
        mse += diff * diff;
    }
    mse /= n_samples;
    matrix_free(y_pred);
    return mse;
}

// Calcula el coeficiente de determinación R²
double linear_regression_r2_score(LinearRegression* model, Matrix* X, Matrix* y) {
    Matrix* y_pred = linear_regression_predict(model, X);
    if (!y_pred) return 0.0;
    int n_samples = y->rows;
    double ss_res = 0.0, ss_tot = 0.0, mean_y = 0.0;
    for (int i = 0; i < n_samples; i++)
        mean_y += y->data[i][0];
    mean_y /= n_samples;
    for (int i = 0; i < n_samples; i++) {
        double diff = y->data[i][0] - y_pred->data[i][0];
        ss_res += diff * diff;
        double diff_tot = y->data[i][0] - mean_y;
        ss_tot += diff_tot * diff_tot;
    }
    matrix_free(y_pred);
    return 1.0 - (ss_res / ss_tot);
}

// Supón que tienes una función para entrenar con Ridge:
void linear_regression_fit_ridge(LinearRegression* model, Matrix* X, Matrix* y, double lambda) {
    if (!model || !X || !y) return;
    int n_samples = X->rows;
    int n_features = X->cols;

    // Calcular medias para centrar los datos
    double* mean_X = (double*)calloc(n_features, sizeof(double));
    double mean_y = 0.0;
    for (int j = 0; j < n_features; j++)
        for (int i = 0; i < n_samples; i++)
            mean_X[j] += X->data[i][j];
    for (int j = 0; j < n_features; j++)
        mean_X[j] /= n_samples;
    for (int i = 0; i < n_samples; i++)
        mean_y += y->data[i][0];
    mean_y /= n_samples;

    // Calcular coeficientes (solo para 1 variable)
    double num = 0.0, den = 0.0;
    for (int i = 0; i < n_samples; i++) {
        num += (X->data[i][0] - mean_X[0]) * (y->data[i][0] - mean_y);
        den += (X->data[i][0] - mean_X[0]) * (X->data[i][0] - mean_X[0]);
    }
    double w = num / den;
    double b = mean_y - w * mean_X[0];

    if (!model->weights)
        model->weights = matrix_create(1, 1);
    model->weights->data[0][0] = w;
    model->bias = b;

    // Calcula X^T X
    Matrix* XtX = matrix_create(n_features, n_features);
    for (int i = 0; i < n_features; i++) {
        for (int j = 0; j < n_features; j++) {
            XtX->data[i][j] = 0.0;
            for (int k = 0; k < n_samples; k++) {
                XtX->data[i][j] += X->data[k][i] * X->data[k][j];
            }
        }
    }
    // Suma lambda*I a la diagonal
    for (int i = 0; i < n_features; i++) {
        XtX->data[i][i] += lambda;
    }
    // ...continúa con la inversión y el resto del ajuste...

    matrix_free(XtX);
    free(mean_X);
}

// Libera memoria
void linear_regression_free(LinearRegression* model) {
    if (!model) return;
    if (model->weights) matrix_free(model->weights);
    free(model);
}