#include <stdio.h>
#include <stdlib.h>
#include <math.h>
#include <string.h>
#include <complex.h>
#include <time.h>

// настройки

#define DIM 2              /* размерность системы ODE */
#define MAX_STAGES 3       /* максимум стадий у наших методов */
#define MAX_NEWTON_IT 20   /* максимум итераций Ньютона на одном шаге */
#define NEWTON_TOL 1e-12   /* критерий остановки Ньютона */
#define PIVOT_TOL 1e-16    /* минимально допустимый ведущий элемент при Гауссе */

// параметры интегрирования
typedef struct {
    double T;          /* конечное время */
    double atol;       /* абсолютная точность для адаптивного шага */
    double rtol;       /* относительная точность для адаптивного шага */
    double h_init;     /* начальный шаг */
    double h_min;      /* минимальный допустимый шаг */
    double h_max;      /* максимальный допустимый шаг */
    double output_dt;  /* через какой шаг по времени сохраняем решение */
} SolverOptions;

// таблица бутчера
typedef struct {
    const char *name;                 /* имя метода */
    int s;                            /* число стадий */
    int order;                        /* порядок метода */
    double A[MAX_STAGES][MAX_STAGES]; /* матрица A */
    double b[MAX_STAGES];             /* веса b */
    double c[MAX_STAGES];             /* узлы c */
} RKMethod;

// статистика  метода
typedef struct {
    long long accepted_steps;         /* сколько шагов принято */
    long long rejected_steps;         /* сколько шагов отклонено */
    long long newton_iterations;      /* суммарно итераций Ньютона */
    long long newton_failures;        /* сколько раз Ньютон не сошелся */
    double min_used_h;                /* минимальный реально использованный шаг */
    double max_used_h;                /* максимальный реально использованный шаг */
    double cpu_seconds;               /* процессорное время */
} MethodStats;

// траектория на равномерной сетке времени
typedef struct {
    int n;        /* число сохраненных точек */
    double *t;    /* массив времен */
    double *y1;   /* первая компонента */
    double *y2;   /* вторая компонента */
} Trajectory;

/// настройки


static double max2(double a, double b) {
    return (a > b) ? a : b;
}

static double min2(double a, double b) {
    return (a < b) ? a : b;
}

static double clamp(double x, double lo, double hi) {
    if (x < lo) return lo;
    if (x > hi) return hi;
    return x;
}

static double vec_inf_norm(const double *v, int n) {
    double m = 0.0;
    for (int i = 0; i < n; ++i) {
        double a = fabs(v[i]);
        if (a > m) m = a;
    }
    return m;
}

static void vec_copy(double *dst, const double *src, int n) {
    for (int i = 0; i < n; ++i) dst[i] = src[i];
}

static void wipe_stats(MethodStats *st) {
    memset(st, 0, sizeof(*st));
    st->min_used_h = 1e300;
    st->max_used_h = 0.0;
}

static void free_trajectory(Trajectory *tr) {
    if (!tr) return;
    free(tr->t);
    free(tr->y1);
    free(tr->y2);
    tr->t = tr->y1 = tr->y2 = NULL;
    tr->n = 0;
}

/*
    Правая часть системы:
        f1(y1, y2) = y2
        f2(y1, y2) = -a * ( y2 * (y1^2 - 1) + y1 )
*/
static void vdp_rhs(double a, const double y[DIM], double f[DIM]) {
    const double y1 = y[0];
    const double y2 = y[1];

    f[0] = y2;
    f[1] = -a * ( y2 * (y1 * y1 - 1.0) + y1 );
}

/*
    Якобиан J = df/dy.

    f1 = y2
    f2 = -a * ( y2 * (y1^2 - 1) + y1 )

    Тогда:
      df1/dy1 = 0
      df1/dy2 = 1

      df2/dy1 = -a * ( 2*y1*y2 + 1 )
      df2/dy2 = -a * ( y1^2 - 1 )
*/
static void vdp_jacobian(double a, const double y[DIM], double J[DIM][DIM]) {
    const double y1 = y[0];
    const double y2 = y[1];

    J[0][0] = 0.0;
    J[0][1] = 1.0;
    J[1][0] = -a * (2.0 * y1 * y2 + 1.0);
    J[1][1] = -a * (y1 * y1 - 1.0);
}

// Решаем систему A x = b методом Гаусса с частичным выбором главного элемента.
static int solve_linear_system_real(int n, double A[6][6], double b[6], double x[6]) {
    double M[6][6];
    double rhs[6];

    for (int i = 0; i < n; ++i) {
        rhs[i] = b[i];
        for (int j = 0; j < n; ++j) {
            M[i][j] = A[i][j];
        }
    }

    for (int k = 0; k < n; ++k) {
        int pivot = k;
        double best = fabs(M[k][k]);
        for (int i = k + 1; i < n; ++i) {
            double cand = fabs(M[i][k]);
            if (cand > best) {
                best = cand;
                pivot = i;
            }
        }

        if (best < PIVOT_TOL) {
            return 0;
        }

        if (pivot != k) {
            for (int j = k; j < n; ++j) {
                double tmp = M[k][j];
                M[k][j] = M[pivot][j];
                M[pivot][j] = tmp;
            }
            double tmp_rhs = rhs[k];
            rhs[k] = rhs[pivot];
            rhs[pivot] = tmp_rhs;
        }

        for (int i = k + 1; i < n; ++i) {
            double factor = M[i][k] / M[k][k];
            M[i][k] = 0.0;
            for (int j = k + 1; j < n; ++j) {
                M[i][j] -= factor * M[k][j];
            }
            rhs[i] -= factor * rhs[k];
        }
    }

    for (int i = n - 1; i >= 0; --i) {
        double s = rhs[i];
        for (int j = i + 1; j < n; ++j) {
            s -= M[i][j] * x[j];
        }
        x[i] = s / M[i][i];
    }

    return 1;
}

static int solve_linear_system_complex(int n,
                                       double complex A[3][3],
                                       double complex b[3],
                                       double complex x[3]) {
    double complex M[3][3];
    double complex rhs[3];

    for (int i = 0; i < n; ++i) {
        rhs[i] = b[i];
        for (int j = 0; j < n; ++j) {
            M[i][j] = A[i][j];
        }
    }

    for (int k = 0; k < n; ++k) {
        int pivot = k;
        double best = cabs(M[k][k]);
        for (int i = k + 1; i < n; ++i) {
            double cand = cabs(M[i][k]);
            if (cand > best) {
                best = cand;
                pivot = i;
            }
        }

        if (best < PIVOT_TOL) {
            return 0;
        }

        if (pivot != k) {
            for (int j = k; j < n; ++j) {
                double complex tmp = M[k][j];
                M[k][j] = M[pivot][j];
                M[pivot][j] = tmp;
            }
            double complex tmp_rhs = rhs[k];
            rhs[k] = rhs[pivot];
            rhs[pivot] = tmp_rhs;
        }

        for (int i = k + 1; i < n; ++i) {
            double complex factor = M[i][k] / M[k][k];
            M[i][k] = 0.0;
            for (int j = k + 1; j < n; ++j) {
                M[i][j] -= factor * M[k][j];
            }
            rhs[i] -= factor * rhs[k];
        }
    }

    for (int i = n - 1; i >= 0; --i) {
        double complex s = rhs[i];
        for (int j = i + 1; j < n; ++j) {
            s -= M[i][j] * x[j];
        }
        x[i] = s / M[i][i];
    }

    return 1;
}

/*
    На одном шаге неявного RK надо решить систему относительно стадий Y_i:

        Y_i = y_n + h * sum_j A_ij * f(Y_j),   i = 1..s.

    Переносим всё влево и получаем нелинейную систему G(Y)=0:

        G_i(Y) = Y_i - y_n - h * sum_j A_ij * f(Y_j).

    Так как DIM = 2, а s <= 3, размер системы максимум 6.

    Вектор неизвестных храним как
       U = [Y_1^(1), Y_1^(2), Y_2^(1), Y_2^(2), ..., Y_s^(2)]

    Якобиан этой системы имеет блочную структуру:

       dG_i/dY_j = I - h * A_ij * J(Y_j),   если рассматривать блоки 2x2.
*/
static int implicit_rk_step(const RKMethod *method,
                            double a,
                            const double y_n[DIM],
                            double h,
                            double y_np1[DIM],
                            int *newton_iters_used) {
    const int s = method->s;
    const int N = s * DIM;

    // stages[i][k] = k-я компонента i-й стадии
    double stages[MAX_STAGES][DIM];

    // Y_i^(0) = y_n + c_i * h * f(y_n)
    double f0[DIM];
    vdp_rhs(a, y_n, f0);
    for (int i = 0; i < s; ++i) {
        for (int k = 0; k < DIM; ++k) {
            stages[i][k] = y_n[k] + method->c[i] * h * f0[k];
        }
    }

    int total_newton_iters = 0;

    for (int iter = 0; iter < MAX_NEWTON_IT; ++iter) {
        double f_stage[MAX_STAGES][DIM];
        double J_stage[MAX_STAGES][DIM][DIM];

        for (int i = 0; i < s; ++i) {
            vdp_rhs(a, stages[i], f_stage[i]);
            vdp_jacobian(a, stages[i], J_stage[i]);
        }

        // формируем невязку G(U)
        double G[6] = {0.0};
        for (int i = 0; i < s; ++i) {
            for (int k = 0; k < DIM; ++k) {
                double sum = 0.0;
                for (int j = 0; j < s; ++j) {
                    sum += method->A[i][j] * f_stage[j][k];
                }
                G[i * DIM + k] = stages[i][k] - y_n[k] - h * sum;
            }
        }

        if (vec_inf_norm(G, N) < NEWTON_TOL) {
            total_newton_iters = iter;
            break;
        }

        double M[6][6] = {{0.0}};
        for (int i = 0; i < s; ++i) {
            for (int j = 0; j < s; ++j) {
                for (int r = 0; r < DIM; ++r) {
                    for (int c = 0; c < DIM; ++c) {
                        double val = -h * method->A[i][j] * J_stage[j][r][c];
                        if (i == j && r == c) {
                            val += 1.0;
                        }
                        M[i * DIM + r][j * DIM + c] = val;
                    }
                }
            }
        }

        double rhs[6] = {0.0};
        double delta[6] = {0.0};
        for (int i = 0; i < N; ++i) rhs[i] = -G[i];

        if (!solve_linear_system_real(N, M, rhs, delta)) {
            // не получилось, уменьгаем шаг h и пробуем снова
            return 0;
        }

        for (int i = 0; i < s; ++i) {
            for (int k = 0; k < DIM; ++k) {
                stages[i][k] += delta[i * DIM + k];
            }
        }

        total_newton_iters = iter + 1;

        // критерий сходимости Ньютона: если поправка delta уже маленькая, итерации можно завершать
        if (vec_inf_norm(delta, N) < NEWTON_TOL * (1.0 + vec_inf_norm((double*)stages, N))) {
            break;
        }

        // неудача если все еще не сошлись на последней
        if (iter == MAX_NEWTON_IT - 1) {
            return 0;
        }
    }

    // y_{n+1} = y_n + h * sum_i b_i * f(Y_i)
    double f_stage[MAX_STAGES][DIM];
    for (int i = 0; i < s; ++i) {
        vdp_rhs(a, stages[i], f_stage[i]);
    }

    for (int k = 0; k < DIM; ++k) {
        double sum = 0.0;
        for (int i = 0; i < s; ++i) {
            sum += method->b[i] * f_stage[i][k];
        }
        y_np1[k] = y_n[k] + h * sum;
    }

    if (newton_iters_used) {
        *newton_iters_used = total_newton_iters;
    }

    return 1;
}


static int adaptive_step(const RKMethod *method,
                         double a,
                         const double y[DIM],
                         double h,
                         const SolverOptions *opt,
                         double y_out[DIM],
                         double *suggested_h,
                         MethodStats *stats) {
    double y_full[DIM], y_mid[DIM], y_half2[DIM];
    int it1 = 0, it2 = 0, it3 = 0;

    // сначала пробуем один полный шаг h
    if (!implicit_rk_step(method, a, y, h, y_full, &it1)) {
        stats->newton_failures++;
        *suggested_h = max2(0.1 * h, opt->h_min);
        return 0;
    }

    // затем два полушага h/2
    if (!implicit_rk_step(method, a, y, 0.5 * h, y_mid, &it2)) {
        stats->newton_failures++;
        *suggested_h = max2(0.1 * h, opt->h_min);
        return 0;
    }
    if (!implicit_rk_step(method, a, y_mid, 0.5 * h, y_half2, &it3)) {
        stats->newton_failures++;
        *suggested_h = max2(0.1 * h, opt->h_min);
        return 0;
    }

    stats->newton_iterations += (it1 + it2 + it3);

    double err = 0.0;
    for (int k = 0; k < DIM; ++k) {
        double sc = opt->atol + opt->rtol * max2(fabs(y_half2[k]), fabs(y[k]));
        double ek = fabs(y_half2[k] - y_full[k]) / sc;
        if (ek > err) err = ek;
    }

    /* Для безопасности, если ошибка точно нулевая по машинной арифметике,
       используем максимально разрешенное увеличение шага. */
    double factor;
    if (err == 0.0) {
        factor = 5.0;
    } else {
        factor = 0.9 * pow(1.0 / err, 1.0 / (method->order + 1.0));
        factor = clamp(factor, 0.2, 5.0);
    }

    *suggested_h = clamp(h * factor, opt->h_min, opt->h_max);

    if (err <= 1.0) {
        y_out[0] = y_half2[0];
        y_out[1] = y_half2[1];
        return 1;
    }

    return 0;
}


static int solve_problem(const RKMethod *method,
                             double a,
                             const SolverOptions *opt,
                             const double y0[DIM],
                             Trajectory *traj,
                             MethodStats *stats) {
    wipe_stats(stats);

    int n_out = (int)llround(opt->T / opt->output_dt) + 1;
    traj->n = n_out;
    traj->t  = (double*)malloc((size_t)n_out * sizeof(double));
    traj->y1 = (double*)malloc((size_t)n_out * sizeof(double));
    traj->y2 = (double*)malloc((size_t)n_out * sizeof(double));

    if (!traj->t || !traj->y1 || !traj->y2) {
        return 0;
    }

    clock_t start_clock = clock();

    double y[DIM] = { y0[0], y0[1] };
    double t = 0.0;
    double h = opt->h_init;

    traj->t[0] = 0.0;
    traj->y1[0] = y[0];
    traj->y2[0] = y[1];

    for (int out_idx = 1; out_idx < n_out; ++out_idx) {
        double target_t = out_idx * opt->output_dt;

        // последнйи
        if (out_idx == n_out - 1) {
            target_t = opt->T;
        }

        while (t < target_t - 1e-15) {
            h = min2(h, target_t - t);
            h = clamp(h, opt->h_min, opt->h_max);

            double y_new[DIM];
            double h_new;

            int ok = adaptive_step(method, a, y, h, opt, y_new, &h_new, stats);

            if (ok) {
                vec_copy(y, y_new, DIM);
                t += h;
                stats->accepted_steps++;
                if (h < stats->min_used_h) stats->min_used_h = h;
                if (h > stats->max_used_h) stats->max_used_h = h;
                h = h_new;
            } else {
                stats->rejected_steps++;
                h = h_new;
                if (h <= opt->h_min * 1.0001) {
                    fprintf(stderr,
                            "Ошибка: метод %s не может продолжить интегрирование для a=%g: слишком маленький шаг.\n",
                            method->name, a);
                    return 0;
                }
            }
        }

        traj->t[out_idx] = target_t;
        traj->y1[out_idx] = y[0];
        traj->y2[out_idx] = y[1];
    }

    clock_t finish_clock = clock();
    stats->cpu_seconds = (double)(finish_clock - start_clock) / CLOCKS_PER_SEC;

    return 1;
}

// R(z) = 1 + z * b^T * (I - z A)^(-1) * E,
static double complex stability_function(const RKMethod *method, double complex z) {
    const int s = method->s;
    double complex M[3][3] = {{0.0}};
    double complex e[3] = {0.0, 0.0, 0.0};
    double complex x[3] = {0.0, 0.0, 0.0};

    for (int i = 0; i < s; ++i) {
        e[i] = 1.0 + 0.0 * I;
        for (int j = 0; j < s; ++j) {
            M[i][j] = -z * method->A[i][j];
            if (i == j) M[i][j] += 1.0;
        }
    }

    if (!solve_linear_system_complex(s, M, e, x)) {
        return 1e300 + 0.0 * I;
    }

    double complex dot = 0.0 + 0.0 * I;
    for (int i = 0; i < s; ++i) {
        dot += method->b[i] * x[i];
    }

    return 1.0 + z * dot;
}

static int write_stability_real_axis_csv(const RKMethod *method,
                                         const char *filename,
                                         double xmin,
                                         double xmax,
                                         int npts) {
    FILE *fp = fopen(filename, "w");
    if (!fp) return 0;

    fprintf(fp, "x,ReR,ImR,AbsR\n");
    for (int i = 0; i < npts; ++i) {
        double x = xmin + (xmax - xmin) * i / (double)(npts - 1);
        double complex R = stability_function(method, x + 0.0 * I);
        fprintf(fp, "%.17g,%.17g,%.17g,%.17g\n", x, creal(R), cimag(R), cabs(R));
    }

    fclose(fp);
    return 1;
}

static int write_trajectory_csv(const char *filename, const Trajectory *tr) {
    FILE *fp = fopen(filename, "w");
    if (!fp) return 0;

    fprintf(fp, "t,y1,y2\n");
    for (int i = 0; i < tr->n; ++i) {
        fprintf(fp, "%.17g,%.17g,%.17g\n", tr->t[i], tr->y1[i], tr->y2[i]);
    }

    fclose(fp);
    return 1;
}

static double compare_trajectories_inf(const Trajectory *A, const Trajectory *B,
                                       double *max_y1_diff,
                                       double *max_y2_diff) {
    if (A->n != B->n) return -1.0;

    double maxnorm = 0.0;
    double d1max = 0.0;
    double d2max = 0.0;

    for (int i = 0; i < A->n; ++i) {
        double d1 = fabs(A->y1[i] - B->y1[i]);
        double d2 = fabs(A->y2[i] - B->y2[i]);
        double dn = max2(d1, d2);
        if (d1 > d1max) d1max = d1;
        if (d2 > d2max) d2max = d2;
        if (dn > maxnorm) maxnorm = dn;
    }

    if (max_y1_diff) *max_y1_diff = d1max;
    if (max_y2_diff) *max_y2_diff = d2max;
    return maxnorm;
}

// описание методов
static RKMethod make_radau_iia_1(void) {
    RKMethod m;
    memset(&m, 0, sizeof(m));
    m.name = "RadauIIA1_ImplicitEuler";
    m.s = 1;
    m.order = 1;

    m.c[0] = 1.0;
    m.A[0][0] = 1.0;
    m.b[0] = 1.0;

    return m;
}

static RKMethod make_radau_iia_3(void) {
    RKMethod m;
    memset(&m, 0, sizeof(m));
    m.name = "RadauIIA3";
    m.s = 2;
    m.order = 3;

    m.c[0] = 1.0 / 3.0;
    m.c[1] = 1.0;

    m.A[0][0] = 5.0 / 12.0;
    m.A[0][1] = -1.0 / 12.0;
    m.A[1][0] = 3.0 / 4.0;
    m.A[1][1] = 1.0 / 4.0;

    m.b[0] = 3.0 / 4.0;
    m.b[1] = 1.0 / 4.0;

    return m;
}

static RKMethod make_radau_iia_5(void) {
    RKMethod m;
    memset(&m, 0, sizeof(m));
    m.name = "RadauIIA5";
    m.s = 3;
    m.order = 5;

    const double r6 = sqrt(6.0);

    m.c[0] = (4.0 - r6) / 10.0;
    m.c[1] = (4.0 + r6) / 10.0;
    m.c[2] = 1.0;

    m.A[0][0] = (88.0 - 7.0 * r6) / 360.0;
    m.A[0][1] = (296.0 - 169.0 * r6) / 1800.0;
    m.A[0][2] = (-2.0 + 3.0 * r6) / 225.0;

    m.A[1][0] = (296.0 + 169.0 * r6) / 1800.0;
    m.A[1][1] = (88.0 + 7.0 * r6) / 360.0;
    m.A[1][2] = (-2.0 - 3.0 * r6) / 225.0;

    m.A[2][0] = (16.0 - r6) / 36.0;
    m.A[2][1] = (16.0 + r6) / 36.0;
    m.A[2][2] = 1.0 / 9.0;

    m.b[0] = (16.0 - r6) / 36.0;
    m.b[1] = (16.0 + r6) / 36.0;
    m.b[2] = 1.0 / 9.0;

    return m;
}

static RKMethod make_sdirk2_table4(void) {
    RKMethod m;
    memset(&m, 0, sizeof(m));
    m.name = "SDIRK2_Table4";
    m.s = 2;
    m.order = 2;

    const double r2 = sqrt(2.0);
    const double gamma = (2.0 - r2) / 2.0;

    m.c[0] = gamma;
    m.c[1] = (2.0 + r2) / 2.0;

    m.A[0][0] = gamma;
    m.A[0][1] = 0.0;
    m.A[1][0] = r2;
    m.A[1][1] = gamma;

    m.b[0] = 0.5;
    m.b[1] = 0.5;

    return m;
}
///

static void print_method_summary(const RKMethod *method,
                                 double a,
                                 const Trajectory *tr,
                                 const MethodStats *st) {
    int last = tr->n - 1;
    printf("Метод: %-28s | a = %.0e\n", method->name, a);
    printf("  y1(T) = %+ .12e\n", tr->y1[last]);
    printf("  y2(T) = %+ .12e\n", tr->y2[last]);
    printf("  accepted steps = %lld\n", st->accepted_steps);
    printf("  rejected steps = %lld\n", st->rejected_steps);
    printf("  Newton iterations total = %lld\n", st->newton_iterations);
    printf("  Newton failures         = %lld\n", st->newton_failures);
    printf("  h_min used = %.6e\n", st->min_used_h);
    printf("  h_max used = %.6e\n", st->max_used_h);
    printf("  CPU time   = %.3f sec\n", st->cpu_seconds);
}

int main(void) {
    // методы
    RKMethod methods[4];
    methods[0] = make_radau_iia_1();
    methods[1] = make_radau_iia_3();
    methods[2] = make_radau_iia_5();
    methods[3] = make_sdirk2_table4();

    // НУ
    const double y0[DIM] = {2.0, 0.0};
    const double a_values[2] = {1.0e3, 1.0e6};

    // параметры точности
    SolverOptions opt;
    opt.T = 20.0;
    opt.atol = 1e-7;
    opt.rtol = 1e-7;
    opt.h_min = 1e-12;
    opt.h_max = 0.5;
    opt.output_dt = 0.01;


    for (int case_id = 0; case_id < 2; ++case_id) {
        double a = a_values[case_id];

        opt.h_init = (a < 1.0e5) ? 1e-4 : 1e-7;

        printf("============================================================\n");
        printf("Решение задачи Ван-дер-Поля для a = %.0e\n", a);
        printf("============================================================\n");

        Trajectory trajectories[4] = {{0}};
        MethodStats stats[4];
        memset(stats, 0, sizeof(stats));

        for (int m = 0; m < 4; ++m) {
            if (!solve_problem(&methods[m], a, &opt, y0, &trajectories[m], &stats[m])) {
                fprintf(stderr, "Не удалось посчитать методом %s для a=%g\n",
                        methods[m].name, a);

                for (int k = 0; k < 4; ++k) {
                    free_trajectory(&trajectories[k]);
                }
                return 1;
            }

            print_method_summary(&methods[m], a, &trajectories[m], &stats[m]);
            printf("\n");

            char fname[256];
            snprintf(fname, sizeof(fname), "trajectory_a%.0e_%s.csv", a, methods[m].name);
            if (!write_trajectory_csv(fname, &trajectories[m])) {
                fprintf(stderr, "Не удалось записать файл %s\n", fname);
            }
        }

        printf("Сравнение всех методов с SDIRK2_Table4:\n");
        for (int m = 0; m < 4; ++m) {
            double max_y1, max_y2;
            double max_norm = compare_trajectories_inf(&trajectories[m], &trajectories[3],
                                                       &max_y1, &max_y2);
            printf("  %-28s : max|dy|_inf = %.6e,   max|dy1| = %.6e,   max|dy2| = %.6e\n",
                   methods[m].name, max_norm, max_y1, max_y2);
        }
        printf("\n");

        for (int m = 0; m < 4; ++m) {
            free_trajectory(&trajectories[m]);
        }
    }

    printf("Запись файлов с функциями устойчивости...\n");
    for (int m = 0; m < 4; ++m) {
        char fname[256];
        snprintf(fname, sizeof(fname), "stability_real_%s.csv", methods[m].name);
        if (!write_stability_real_axis_csv(&methods[m], fname, -50.0, 5.0, 2001)) {
            fprintf(stderr, "Не удалось записать %s\n", fname);
            return 1;
        }
    }

    printf("Готово.\n");
    return 0;
}
