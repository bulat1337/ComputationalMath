#include <stdio.h>
#include <math.h>

double func(double arg) {
    return 1 / (1 + arg * arg);
}

double actual_derivative(double arg) {
    double smth = (1 + arg * arg);
    return -1 * 1 / (smth * smth) * 2 * arg;
}

double derivative_1(double arg, double h) {
    return (func(arg + h) - func(arg)) / h;
}

double derivative_2(double arg, double h) {
    return (func(arg + h) - func(arg - h)) / (2*h);
}

double derivative_3(double arg, double h) {
    return (func(arg - 2*h) - 8*func(arg - h) + 8*func(arg + h) - func(arg + 2*h)) / (12*h);
}

void ProcessDer(double (*der_func) (double, double), double arg, double step, double h_0, size_t step_amount, FILE* file_der)
{
    double h = h_0;
    for (size_t id = 0; id < step_amount ; ++id) {
        const double actual = actual_derivative(arg);
        const double der_1 = (*der_func)(arg, h);

        const double error = fabs(actual - der_1);

        fprintf(file_der, "%.15lf;%.15lf\n", h, error);

        h += step;
    }
}

int main() {
    FILE* file_der_1 = fopen("data/der_1.csv", "w");
    FILE* file_der_2 = fopen("data/der_2.csv", "w");
    FILE* file_der_3 = fopen("data/der_3.csv", "w");

    fprintf(file_der_1, "h;error\n");
    fprintf(file_der_2, "h;error\n");
    fprintf(file_der_3, "h;error\n");

    ProcessDer(derivative_1, /*arg=*/5.0, /*strp=*/0.000000001, /*h_0=*/0.000000001, /*step_amount=*/200, /*file_der=*/file_der_1);

    ProcessDer(derivative_2, /*arg=*/5.0, /*strp=*/0.000000001, /*h_0=*/0.000000001 * 1000, /*step_amount=*/50000, /*file_der=*/file_der_2);

    ProcessDer(derivative_3, /*arg=*/5.0, /*strp=*/0.000000001, /*h_0=*/0.000000001 * 100000, /*step_amount=*/5000000, /*file_der=*/file_der_3);

    fclose(file_der_1);
    fclose(file_der_2);
    fclose(file_der_3);

    
    return 0;
}