#include <stdio.h>
#include <math.h>

double func(double arg) {
    return 1 / (1 + arg * arg);
}

double actual_derivative(double arg) {
    double smth = (1 + arg * arg);
    return -1 * 1 / (smth * smth) * 2 * arg;
}

double derivative_1(double (*func_ptr)(double), double arg, double h) {
    return ((*func_ptr)(arg + h) - (*func_ptr)(arg)) / h;
}

int main() {
    const double arg = 5.0;
    const double step = 0.00000000001;
    const double h_0 = step;
    const size_t step_amount = 1000000;

    printf("h;error\n");

    double h = h_0;
    for (size_t id = 0; id < step_amount ; ++id) {
        const double actual = actual_derivative(arg);
        const double der_1 = derivative_1(func, arg, h);

        const double error = fabs(actual - der_1);

        printf("%.15lf;%.15lf\n", h, error);


        h += step;
    }

    
    return 0;
}