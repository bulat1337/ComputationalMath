#include <stdio.h>
#include <math.h>

// S_i(t) = a[i] + b[i]*(t-x[i]) + c[i]*(t-x[i])^2 + d[i]*(t-x[i])^3
void build_natural_cubic_spline(int n,
                                const double *x,
                                const double *y,
                                double *a,
                                double *b,
                                double *c,
                                double *d)
{
    int i, j;

    // h[i] = x[i+1] - x[i]
    double h[32];
    double alpha[32];

    double l[32];
    double mu[32];
    double z[32];

    for (i = 0; i < n - 1; ++i) {
        h[i] = x[i + 1] - x[i];
    }

    alpha[0] = alpha[n - 1] = 0.0;

    for (i = 1; i < n - 1; ++i) {
        double term1 = (3.0 / h[i])     * (y[i + 1] - y[i]);
        double term2 = (3.0 / h[i - 1]) * (y[i]     - y[i - 1]);
        alpha[i] = term1 - term2;
    }

    l[0]  = 1.0;
    mu[0] = 0.0;
    z[0]  = 0.0;

    for (i = 1; i < n - 1; ++i) {
        l[i] = 2.0 * (x[i + 1] - x[i - 1]) - h[i - 1] * mu[i - 1];
        mu[i] = h[i] / l[i];
        z[i]  = (alpha[i] - h[i - 1] * z[i - 1]) / l[i];
    }

    l[n - 1] = 1.0;
    z[n - 1] = 0.0;
    c[n - 1] = 0.0;

    for (j = n - 2; j >= 0; --j) {
        c[j] = z[j] - mu[j] * c[j + 1];
        b[j] = (y[j + 1] - y[j]) / h[j]
             - h[j] * (c[j + 1] + 2.0 * c[j]) / 3.0;
        d[j] = (c[j + 1] - c[j]) / (3.0 * h[j]);
        a[j] = y[j];
    }
}

// S'_i(t) = b[i] + 2*c[i]*(t-x[i]) + 3*d[i]*(t-x[i])^2
double spline_derivative(int n,
                         const double *x,
                         const double *a,
                         const double *b,
                         const double *c,
                         const double *d,
                         double t)
{
    int i;

    if (t <= x[0]) {
        i = 0;
    } else if (t >= x[n - 1]) {
        i = n - 2;
    } else {
        for (i = 0; i < n - 1; ++i) {
            if (t < x[i + 1])
                break;
        }
    }

    double dx = t - x[i];
    (void)a;

    return b[i] + 2.0 * c[i] * dx + 3.0 * d[i] * dx * dx;
}

void find_max_derivative(int n,
                         const double *x,
                         const double *a,
                         const double *b,
                         const double *c,
                         const double *d,
                         double *t_star,
                         double *dTdt_max)
{
    const double EPS = 1e-12;
    int i;

    *dTdt_max = -1e300;
    *t_star   = x[0];

    for (i = 0; i < n - 1; ++i) {
        double t, value;
        if (i == 0) {
            t = x[i];
            value = spline_derivative(n, x, a, b, c, d, t);
            if (value > *dTdt_max) {
                *dTdt_max = value;
                *t_star   = t;
            }
        }

        if (fabs(d[i]) > EPS) {
            double t_ext = x[i] - c[i] / (3.0 * d[i]);

            if (t_ext > x[i] && t_ext < x[i + 1]) {
                value = spline_derivative(n, x, a, b, c, d, t_ext);
                if (value > *dTdt_max) {
                    *dTdt_max = value;
                    *t_star   = t_ext;
                }
            }
        }

        t = x[i + 1];
        value = spline_derivative(n, x, a, b, c, d, t);
        if (value > *dTdt_max) {
            *dTdt_max = value;
            *t_star   = t;
        }
    }
}

int main(void)
{
    double t[] = { 1.0, 1.5, 2.0, 2.3, 3.0, 3.5, 4.0, 4.5, 5.0 };
    double T[] = { 37.3, 37.58, 37.86, 38.21, 38.70, 39.26, 40.17, 40.17, 40.17 };

    int n = (int)(sizeof(t) / sizeof(t[0]));

    /* Массивы коэффициентов сплайна для интервалов [t[i], t[i+1]] */
    double a[32], b[32], c[32], d[32];

    build_natural_cubic_spline(n, t, T, a, b, c, d);

    double t_star, max_deriv;
    find_max_derivative(n, t, a, b, c, d, &t_star, &max_deriv);

    printf("t* = %.4f часов\n", t_star);
    printf("Максимальная производная dT/dt ≈ %.4f °C/час\n", max_deriv);

    return 0;
}
