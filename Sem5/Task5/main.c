#include <stdio.h>
#include <math.h>

#define EPS        1e-4
#define MAX_ITERS  100
double F1(double x, double y) {
    return cos(x - 1.0) + y - 0.5;
}

double F2(double x, double y) {
    return x - cos(y) - 3.0;
}

double dF1dx(double x, double y) {
    (void)y;  
    return -sin(x - 1.0);
}

double dF1dy(double x, double y) {
    (void)x; (void)y;
    return 1.0;
}

double dF2dx(double x, double y) {
    (void)y;
    return 1.0;
}

double dF2dy(double x, double y) {
    return sin(y);
}

int main(void) {
    double x = 3.0;
    double y = 0.5;

    for (int iter = 0; iter < MAX_ITERS; ++iter) {
        double f1 = F1(x, y);
        double f2 = F2(x, y);

        double a = dF1dx(x, y);
        double b = dF1dy(x, y);
        double c = dF2dx(x, y);
        double d = dF2dy(x, y);

        double det = a * d - b * c;

        if (fabs(det) < 1e-15) {
            printf("вырождение якобиана\n");
            return 1;
        }

        // крамер
        double dx = (-f1 * d + f2 * b) / det;
        double dy = ( f1 * c - f2 * a) / det;

        x += dx;
        y += dy;

        double step = sqrt(dx * dx + dy * dy);
        if (step < EPS) {
            printf("Решение найдено за %d итераций.\n", iter + 1);
            printf("x = %.3f\n", x);
            printf("y = %.3f\n", y);
            return 0;
        }
    }

    printf("Метод Ньютона не сошелся за %d итераций.\n", MAX_ITERS);
    printf("Последнее приближение: x = %.6f, y = %.6f\n", x, y);

    return 0;
}
