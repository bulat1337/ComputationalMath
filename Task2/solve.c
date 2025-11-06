#include <math.h>
#include <stddef.h>
#include <stdio.h>

// #define LOG

// CholeskyBanachiewicz decomposition
void Decompose(const size_t dim, const double A[dim][dim], double L[dim][dim]) {
    for (size_t i = 0; i < dim; ++i) {
        for (size_t j = 0; j <= i; ++j) {
            double sum = 0;

            for (size_t k = 0; k < j; k++) {
                sum += L[i][k] * L[j][k];
            }
    
            if (i == j) {
                L[i][j] = sqrt(A[i][i] - sum);
            } else {
                L[i][j] = (1.0 / L[j][j] * (A[i][j] - sum));
            }
        }
    }
}

void Transpose(const size_t dim, const double L[dim][dim], double L_transposed[dim][dim]) {
    for (size_t i = 0 ; i < dim ; ++i) {
        for (size_t j = 0 ; j < dim ; ++j) {
            L_transposed[i][j] = L[j][i];
        }
    }
}

void ForwardSubstitution(const size_t dim, const double L[dim][dim], const double b[dim], double y[dim]) {
    y[0] = b[0] / L[0][0];

    for (size_t i = 1 ; i < dim ; ++i) {
        double sum = 0.0;
        for (size_t k = 0 ; k < i ; ++k) {
            sum += L[i][k] * y[k];
        }

        y[i] = (b[i] - sum) / L[i][i];
    }
}   

void BackSubstitution(const size_t dim, const double L_transposed[dim][dim], const double y[dim], double x[dim]) {
    for (size_t i = dim; i-- > 0;) {
        double sum = 0.0;

        for (size_t k = i + 1; k < dim; ++k) {
            sum += L_transposed[i][k] * x[k];   
        }

        x[i] = (y[i] - sum) / L_transposed[i][i];
    }
}

void Solve(const size_t dim, const double A[dim][dim], double x[dim], const double b[dim]) {
    double L[dim][dim] = {};
    Decompose(dim, A, L);

    double L_transposed[dim][dim] = {};
    Transpose(dim, L, L_transposed);

    double y[dim] = {};
    ForwardSubstitution(dim, L, b, y);

    BackSubstitution(dim, L_transposed, y, x);

#ifdef LOG
    printf("L:\n");
    for (size_t i = 0 ; i < dim ; ++i) {
        for (size_t j = 0 ; j < dim ; ++j) {
            printf("%lf ", L[i][j]);
        }
        printf("\n");
    }

    printf("L_transposed:\n");
    for (size_t i = 0 ; i < dim ; ++i) {
        for (size_t j = 0 ; j < dim ; ++j) {
            printf("%lf ", L_transposed[i][j]);
        }
        printf("\n");
    }

    printf("y:\n");
    for (size_t i = 0 ; i < dim ; ++i) {
        printf("%lf ", y[i]);
    }
    printf("\n");

    printf("x:\n");
    for (size_t i = 0 ; i < dim ; ++i) {
        printf("%lf ", x[i]);
    }
    printf("\n");
#endif
}