#include <stdio.h>
#include <math.h>
#include <stddef.h>
#include <stdlib.h>
#include <assert.h>

#include "solve.h"

static const double EPS = 1e-9;

static void check_array(const size_t dim, const double x[dim], const double expected[dim]) {
    for (size_t i = 0; i < dim; ++i) {
        if (fabs(x[i] - expected[i]) > EPS) {
            fprintf(stderr, "FAIL: index %zu: got %.12g expected %.12g\n", i, x[i], expected[i]);
            exit(EXIT_FAILURE);
        }
    }
}

static void test_1x1() {
    const size_t dim = 1;
    const double A[dim][dim] = {{5.0}};
    const double b[dim] = {10.0};
    double x[dim] = {0.0};
    const double expected[dim] = {2.0};

    Solve(dim, A, x, b);
    check_array(dim, x, expected);
    printf("test_1x1 passed\n");
}

static void test_diagonal_2x2() {
    const size_t dim = 2;
    const double A[dim][dim] = {    
        {2.0, 0.0},
        {0.0, 3.0}
    };
    const double b[dim] = {4.0, 9.0};
    double x[dim] = {0.0, 0.0};
    const double expected[dim] = {2.0, 3.0};

    Solve(dim, A, x, b);
    check_array(dim, x, expected);
    printf("test_diagonal_2x2 passed\n");
}

static void test_given_3x3() {
    const size_t dim = 3;
    const double A[3][3] = {
        {4, 12, -16},
        {12, 37, -43},
        {-16, -43, 98}
    };
    const double b[3] = {1, 2, 3};
    double x[3] = {0.0, 0.0, 0.0};

    const double expected[3] = {28.583333333333332, -7.666666666666667, 1.3333333333333333};

    Solve(dim, A, x, b);
    check_array(dim, x, expected);
    printf("test_given_3x3 passed\n");
}

static void test_given_4x4() {
    const size_t dim = 4;
    const double A[dim][dim] = {
        {12, 54,  26, -42},
        {54, 52,  -4,  54},
        {26, -4, -64,  24},
        {-42, 54,  24,  98}
    };
    const double b[dim] = {5, 32, -65, 345};
    double x[dim] = {};

    const double expected[dim] = {
        -8784469/3690086,
        5724759/3690086,
        1921553/3690086,
        5600793/3690086
    };

    Solve(dim, A, x, b);
    check_array(dim, x, expected);
    printf("test_given_4x4 passed\n");
}

int main() {
    test_1x1();
    test_diagonal_2x2();
    test_given_3x3();
    test_given_4x4();

    printf("All tests passed\n");

    return 0;
}
