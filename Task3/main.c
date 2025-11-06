#include <stdio.h>
#include <stdlib.h>
#include <math.h>


size_t IdX(size_t i, size_t j, size_t n) {
    return i * n + j;
}

static void PrintMatrix(const char *name, double *M, int n) {
    printf("%s =\n", name);

    for (size_t i = 0; i < n; ++i) {
        for (size_t j = 0; j < n; ++j) {
            printf("%12.6f ", M[IdX(i,j,n)]);
        }

        printf("\n");
    }
}

static double OffDiagMatrixNorm(double *A, size_t n) {
    double sum = 0.0;

    for (size_t i = 0; i < n; ++i) {
        for (size_t j = 0; j < n; ++j) {
            if (i != j) { 
                double v = A[IdX(i,j,n)]; 
                sum += v*v; 
            }
        }
    }

    return sqrt(sum);
}

static double MaxOffDiagElem(double *A, size_t n) {
    double max_elem = 0.0;

    for (size_t i = 0; i < n; ++i) {
        for (size_t j = i+1; j < n; ++j) {
            double elem = fabs(A[IdX(i,j,n)]);

            if (elem > max_elem) {
                max_elem = elem;
            }
        }
    }

    return max_elem;
}

void JacobiWithBarrier(double *A, double *V, size_t n, double eps) {
    for (size_t i = 0; i < n * n; ++i) {
        V[i] = 0.0;  
    } 

    for (size_t i = 0; i < n; ++i) {
        V[IdX(i,i,n)] = 1.0;
    }

    double off_norm = OffDiagMatrixNorm(A, n);
    
    if (off_norm < eps) {
        return;
    }

    double threshold = MaxOffDiagElem(A, n);
    if (threshold == 0.0) {
        return;
    } 

    const size_t max_sweeps = 100 * n;
    size_t sweep = 0;

    while (threshold > eps && sweep < max_sweeps) {
        size_t rotations = 0;
        for (size_t i = 0; i < n-1; ++i) {
            for (size_t j = i+1; j < n; ++j) {
                double aij = A[IdX(i,j,n)];
                if (fabs(aij) <= threshold) {
                    continue;
                }

                double aii = A[IdX(i,i,n)];
                double ajj = A[IdX(j,j,n)];

                // calc cos and sin
                double tau = (ajj - aii) / (2.0 * aij);
                double tan = 0.0;

                if (tau >= 0.0) {
                    tan = 1.0 / (tau + sqrt(1.0 + tau * tau));
                } else {
                    tan = -1.0 / (-tau + sqrt(1.0 + tau * tau));
                }
                
                double cos = 1.0 / sqrt(1.0 + tan * tan);
                double sin = tan * cos;
                //

                // assign new diag elems
                double new_aii = cos*cos*aii - 2.0*cos*sin*aij + sin*sin*ajj;
                double new_ajj = sin*sin*aii + 2.0*cos*sin*aij + cos*cos*ajj;
                A[IdX(i,i,n)] = new_aii;
                A[IdX(j,j,n)] = new_ajj;
                A[IdX(i,j,n)] = 0.0;
                A[IdX(j,i,n)] = 0.0;
                //

                // assign new rows/cols that are k != i,j
                for (size_t k = 0; k < n; ++k) {
                    if (k == i || k == j) {
                        continue;
                    }

                    double aki = A[IdX(k,i,n)];
                    double akj = A[IdX(k,j,n)];

                    double new_aki = cos*aki - sin*akj;
                    double new_akj = sin*aki + cos*akj;

                    A[IdX(k,i,n)] = new_aki;
                    A[IdX(i,k,n)] = new_aki;
                    A[IdX(k,j,n)] = new_akj;
                    A[IdX(j,k,n)] = new_akj;
                }

                for (size_t k = 0; k < n; ++k) {
                    double vki = V[IdX(k,i,n)];
                    double vkj = V[IdX(k,j,n)];

                    V[IdX(k,i,n)] = cos*vki - sin*vkj;
                    V[IdX(k,j,n)] = sin*vki + cos*vkj;
                }

                rotations++;
            }
        }

        if (rotations == 0) {
            threshold *= 0.1;
        }

        double norm = OffDiagMatrixNorm(A, n);
        if (norm < eps) {
            break;
        }

        sweep++;
    }
}
static void CopyMatrix(double *dst, double *src, size_t n) {
    for (size_t i = 0; i < n*n; ++i) {
        dst[i] = src[i];
    } 
}

static void ReconstructMatrix(double *A_rec, double *V, double *eig, size_t n) {
    /* A_rec[i,j] = sum_k V[i,k] * eig[k] * V[j,k] */
    for (size_t i = 0; i < n; ++i) {
        for (size_t j = 0; j < n; ++j) {
            double s = 0.0;
            for (size_t k = 0; k < n; ++k) {
                s += V[IdX(i,k,n)] * eig[k] * V[IdX(j,k,n)];
            }
            A_rec[IdX(i,j,n)] = s;
        }
    }
}

static int CheckReconstruction(double *A_orig, double *A_rec, double *V, int n, double tol) {
    double max_abs = 0.0;
    double sum_sq = 0.0;
    for (size_t i = 0; i < n; ++i) {
        for (size_t j = 0; j < n; ++j) {
            double d = A_orig[IdX(i,j,n)] - A_rec[IdX(i,j,n)];
            double ad = fabs(d);
            if (ad > max_abs) max_abs = ad;
            sum_sq += d*d;
        }
    }
    double rms = sqrt(sum_sq / (n*n));

    printf("\nReconstruction check:\n");
    printf("\tmax absolute difference = %.12e\n", max_abs);
    printf("\tRMS difference          = %.12e\n", rms);

    if (max_abs <= tol) {
        printf("  Verdict: OK (tol = %.1e)\n", tol);
        return 0;
    } else {
        printf("  Результат: FAIL (tol = %.1e)\n", tol);
        return 1;
    }
}

int main(void) {
    size_t n = 3;

    double A[] = {
        4.0, 1.0, -2.0,
        1.0, 2.0, 0.0,
        -2.0, 0.0, 3.0
    };

    double *A_orig = malloc(n * n * sizeof(double));

    if (!A_orig) { 
        perror("malloc"); return 1; 
    }

    CopyMatrix(A_orig, A, n);

    double *V = malloc(n * n * sizeof(double));
    if (!V) { 
        perror("malloc V"); 
        free(A_orig); 
        return 1; 
    }

    printf("Input matrix:\n");
    PrintMatrix("A", A, n);

    JacobiWithBarrier(A, V, n, 1e-12);

    printf("\nAfter diagonalization:\n");
    PrintMatrix("A_diag", A, n);

    printf("\nEigenvectors matrix V:\n");
    PrintMatrix("V", V, n);

    double *eig = malloc(n * sizeof(double));
    if (!eig) { 
        perror("malloc eig"); 
        free(A_orig); 
        free(V); 
        return 1; 
    }

    for (size_t i = 0; i < n; ++i) {
        eig[i] = A[IdX(i,i,n)];
    }

    /* восстановление и проверка */
    double *A_rec = malloc(n * n * sizeof(double));
    if (!A_rec) { 
        perror("malloc A_rec"); 
        free(A_orig); free(V); 
        free(eig); return 1; 
    }

    ReconstructMatrix(A_rec, V, eig, n);

    CheckReconstruction(A_orig, A_rec, V, n, 1e-9);

    printf("\nReconstructed matrix A_rec:\n");
    PrintMatrix("A_rec", A_rec, n);

    free(A_orig);
    free(V);
    free(eig);
    free(A_rec);

    return 0;
}
