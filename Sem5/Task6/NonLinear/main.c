/*
 * Численное решение задачи двух тел:
 *
 *   x' = z,
 *   y' = u,
 *   z' = - x / (x^2 + y^2)^{3/2},
 *   u' = - y / (x^2 + y^2)^{3/2},
 *
 *   x(0) = 0.5, y(0) = 0, z(0) = 0, u(0) = sqrt(3),
 *
 * на интервале 0 < t <= 20.
 *
 * В программе:
 *   1) Реализованы функции правой части, энергии и момента импульса.
 *   2) Реализованы шаговые методы:
 *        - явный Эйлер (1-й порядок),
 *        - Рунге–Кутта 2-го порядка,
 *        - классический Рунге–Кутта 4-го порядка.
 *   3) Есть интегратор с постоянным шагом для сравнения методов
 *      и шагов — он может одновременно писать траекторию в файл.
 *   4) Реализован адаптивный метод Эйлера (1-го порядка) на основе
 *      удвоения шага (step–doubling) и оценки локальной ошибки.
 *
 * Все важные шаги комментированы; код можно использовать как шаблон.
 */

#include <stdio.h>
#include <math.h>

/* Состояние системы: (x, y, z = x', u = y'). */
typedef struct {
    double x;
    double y;
    double z;
    double u;
} State;

/* -------------------- ПРАВАЯ ЧАСТЬ ОДУ -------------------- */
/* rhs(t, s, f) вычисляет f = F(t, s), т.е. правую часть системы.
 * t фактически не используется (система автономная), но аргумент
 * оставлен для совместимости с общими интерфейсами.
 */
void rhs(double t, const State *s, State *f)
{
    (void)t;  /* подавляем предупреждение компилятора */

    double r2 = s->x * s->x + s->y * s->y;
    double r  = sqrt(r2);
    double r3 = r2 * r;         /* r^3 */

    /* Скорости и ускорения */
    f->x = s->z;
    f->y = s->u;

    /* Гравитационное ускорение (µ = 1) */
    f->z = -s->x / r3;
    f->u = -s->y / r3;
}

/* -------------------- ИНТЕГРАЛЫ ДВИЖЕНИЯ -------------------- */

/* Полная энергия:
 *   E = 0.5 * (z^2 + u^2) - 1 / r
 * В идеале должна сохраняться.
 */
double energy(const State *s)
{
    double v2 = s->z * s->z + s->u * s->u;
    double r  = sqrt(s->x * s->x + s->y * s->y);
    return 0.5 * v2 - 1.0 / r;
}

/* Круговой момент импульса:
 *   L = x * u - y * z
 * Тоже интеграл движения.
 */
double angular_momentum(const State *s)
{
    return s->x * s->u - s->y * s->z;
}

/* -------------------- ШАГОВЫЕ МЕТОДЫ -------------------- */

/* Явный Эйлер (1-й порядок):
 *   Y_{n+1} = Y_n + h * F(t_n, Y_n)
 */
State step_euler(double t, double h, const State *y)
{
    State f, yn;
    rhs(t, y, &f);

    yn.x = y->x + h * f.x;
    yn.y = y->y + h * f.y;
    yn.z = y->z + h * f.z;
    yn.u = y->u + h * f.u;

    return yn;
}

/* Рунге–Кутта 2-го порядка (схема Хойна):
 *   k1 = F(t_n, Y_n)
 *   k2 = F(t_n + h/2, Y_n + h/2 * k1)
 *   Y_{n+1} = Y_n + h/2 * (k1 + k2)
 */
State step_rk2(double t, double h, const State *y)
{
    State k1, k2, tmp, yn;

    rhs(t, y, &k1);

    tmp.x = y->x + 0.5 * h * k1.x;
    tmp.y = y->y + 0.5 * h * k1.y;
    tmp.z = y->z + 0.5 * h * k1.z;
    tmp.u = y->u + 0.5 * h * k1.u;

    rhs(t + 0.5 * h, &tmp, &k2);

    yn.x = y->x + 0.5 * h * (k1.x + k2.x);
    yn.y = y->y + 0.5 * h * (k1.y + k2.y);
    yn.z = y->z + 0.5 * h * (k1.z + k2.z);
    yn.u = y->u + 0.5 * h * (k1.u + k2.u);

    return yn;
}

/* Классический Рунге–Кутта 4-го порядка:
 *   k1 = F( t_n,           Y_n            )
 *   k2 = F( t_n + h/2,     Y_n + h/2 k1   )
 *   k3 = F( t_n + h/2,     Y_n + h/2 k2   )
 *   k4 = F( t_n + h,       Y_n + h   k3   )
 *   Y_{n+1} = Y_n + h/6 * (k1 + 2k2 + 2k3 + k4)
 */
State step_rk4(double t, double h, const State *y)
{
    State k1, k2, k3, k4, tmp, yn;

    rhs(t, y, &k1);

    tmp.x = y->x + 0.5 * h * k1.x;
    tmp.y = y->y + 0.5 * h * k1.y;
    tmp.z = y->z + 0.5 * h * k1.z;
    tmp.u = y->u + 0.5 * h * k1.u;
    rhs(t + 0.5 * h, &tmp, &k2);

    tmp.x = y->x + 0.5 * h * k2.x;
    tmp.y = y->y + 0.5 * h * k2.y;
    tmp.z = y->z + 0.5 * h * k2.z;
    tmp.u = y->u + 0.5 * h * k2.u;
    rhs(t + 0.5 * h, &tmp, &k3);

    tmp.x = y->x + h * k3.x;
    tmp.y = y->y + h * k3.y;
    tmp.z = y->z + h * k3.z;
    tmp.u = y->u + h * k3.u;
    rhs(t + h, &tmp, &k4);

    yn.x = y->x + (h / 6.0) * (k1.x + 2.0 * k2.x + 2.0 * k3.x + k4.x);
    yn.y = y->y + (h / 6.0) * (k1.y + 2.0 * k2.y + 2.0 * k3.y + k4.y);
    yn.z = y->z + (h / 6.0) * (k1.z + 2.0 * k2.z + 2.0 * k3.z + k4.z);
    yn.u = y->u + (h / 6.0) * (k1.u + 2.0 * k2.u + 2.0 * k3.u + k4.u);

    return yn;
}

/* Указатель на шаговую функцию. */
typedef State (*Stepper)(double t, double h, const State *y);

/* -------------------- ИНТЕГРИРОВАНИЕ С ПОСТОЯННЫМ ШАГОМ -------------------- */

/* Интегрирование на [0, T] с постоянным шагом h.
 * step  — выбранный метод (Эйлер, RK2, RK4),
 * y0    — начальное состояние,
 * fname — если не NULL, траектория (t, x, y, r, E, L) пишется в этот файл.
 *
 * Функция возвращает состояние в момент времени T.
 */
State integrate_fixed(Stepper step, double h, double T,
                      const State *y0, const char *fname)
{
    int    n_steps = (int)floor(T / h + 0.5);  /* предполагаем, что T ≈ n h */
    double t       = 0.0;
    State y        = *y0;

    FILE *fp = NULL;
    if (fname != NULL) {
        fp = fopen(fname, "w");
        if (!fp) {
            perror("fopen");
        }
    }

    for (int n = 0; n < n_steps; ++n) {
        if (fp) {
            double r = sqrt(y.x * y.x + y.y * y.y);
            double E = energy(&y);
            double L = angular_momentum(&y);
            fprintf(fp, "%.8f %.8f %.8f %.8f %.8f %.8f\n",
                    t, y.x, y.y, r, E, L);
        }

        y = step(t, h, &y);
        t += h;
    }

    if (fp) {
        /* записываем последнюю точку t = T */
        double r = sqrt(y.x * y.x + y.y * y.y);
        double E = energy(&y);
        double L = angular_momentum(&y);
        fprintf(fp, "%.8f %.8f %.8f %.8f %.8f %.8f\n",
                t, y.x, y.y, r, E, L);
        fclose(fp);
    }

    return y;
}

/* -------------------- АДАПТИВНЫЙ ЭЙЛЕР (1-й ПОРЯДОК) -------------------- */

/* Адаптивный метод Эйлера на основе удвоения шага (step–doubling).
 *
 * На каждом шаге строятся два приближения:
 *   1) один шаг Эйлера длиной h (Y_big),
 *   2) два шага длиной h/2 (Y_small).
 * Оценка локальной ошибки: max|Y_small - Y_big|.
 *
 * Если ошибка <= tol, шаг принимается (берём Y_small, т.к. он точнее),
 * иначе шаг уменьшается и пересчитывается.
 *
 * Вход:
 *   T      — конечное время интегрирования,
 *   h0     — начальный шаг,
 *   tol    — допуск на локальную ошибку,
 *   y0     — начальное состояние.
 *
 * Выход:
 *   возвращается состояние в момент T;
 *   через указатели (если не NULL) можно получить статистику:
 *     *p_steps    — число принятых шагов,
 *     *p_rejects  — число отклонённых шагов,
 *     *p_h_min    — минимальный использованный шаг,
 *     *p_h_max    — максимальный использованный шаг.
 */
State euler_adaptive(double T, double h0, double tol,
                     const State *y0,
                     int *p_steps, int *p_rejects,
                     double *p_h_min, double *p_h_max)
{
    const double h_min = 1e-6;
    const double h_max = 0.1;
    const double safety = 0.9;      /* запас по безопасности при пересчёте шага */

    State y = *y0;
    double t = 0.0;
    double h = h0;

    int    steps   = 0;
    int    rejects = 0;
    double h_used_min = h_max;
    double h_used_max = h_min;

    while (t < T) {
        if (h < h_min) {
            /* шаг слишком мал — выходим, чтобы не зациклиться */
            fprintf(stderr, "h < h_min, прерывание адаптивного Эйлера\n");
            break;
        }

        if (t + h > T) {
            /* Последний шаг подгоняем, чтобы точно попасть в T. */
            h = T - t;
        }

        /* --- вычисляем Y_big: один шаг длиной h --- */
        State y_big = step_euler(t, h, &y);

        /* --- вычисляем Y_small: два шага длиной h/2 --- */
        double h2 = 0.5 * h;
        State y_half = step_euler(t,    h2, &y);
        State y_small = step_euler(t+h2, h2, &y_half);

        /* --- оценка локальной ошибки --- */
        double err_x = fabs(y_small.x - y_big.x);
        double err_y = fabs(y_small.y - y_big.y);
        double err_z = fabs(y_small.z - y_big.z);
        double err_u = fabs(y_small.u - y_big.u);

        double err = err_x;
        if (err_y > err) err = err_y;
        if (err_z > err) err = err_z;
        if (err_u > err) err = err_u;

        if (err <= tol) {
            /* шаг успешен, принимаем результат y_small */
            y = y_small;
            t += h;
            steps++;

            if (h < h_used_min) h_used_min = h;
            if (h > h_used_max) h_used_max = h;

            /* оцениваем новый шаг:
             * для метода порядка p=1 формула:
             *   h_new = h * (tol/err)^{1/(p+1)} = h * sqrt(tol/err)
             * ограничиваем рост шага во избежание слишком резких изменений.
             */
            if (err < 1e-16) {
                /* ошибка почти нулевая — просто немного увеличим шаг */
                h *= 2.0;
            } else {
                double factor = safety * sqrt(tol / err);
                if (factor > 2.0) factor = 2.0;   /* не увеличиваем более чем в 2 раза */
                if (factor < 0.5) factor = 0.5;   /* и не уменьшаем более чем в 2 раза */
                h *= factor;
            }

            if (h > h_max) h = h_max;
            if (h < h_min) h = h_min;
        } else {
            /* ошибка велика — уменьшаем шаг и повторяем */
            rejects++;
            double factor = safety * sqrt(tol / err);
            if (factor > 0.5) factor = 0.5;   /* уменьшаем шаг не менее чем в 2 раза */
            h *= factor;
            if (h < h_min) h = h_min;
            /* состояние y и t не меняем — шаг будет пересчитан */
        }
    }

    if (p_steps)    *p_steps   = steps;
    if (p_rejects)  *p_rejects = rejects;
    if (p_h_min)    *p_h_min   = h_used_min;
    if (p_h_max)    *p_h_max   = h_used_max;

    return y;
}

/* -------------------- ВСПОМОГАТЕЛЬНЫЕ ФУНКЦИИ -------------------- */

/* Печать состояния и интегралов в удобном формате. */
void print_state(const char *label, double T, const State *s)
{
    double r = sqrt(s->x * s->x + s->y * s->y);
    double E = energy(s);
    double L = angular_momentum(s);
    double phi = atan2(s->y, s->x);   /* фазовый угол */

    printf("%s at t = %.2f:\n", label, T);
    printf("  x = % .10f  y = % .10f  r = %.10f\n", s->x, s->y, r);
    printf("  z = % .10f  u = % .10f\n", s->z, s->u);
    printf("  E = % .10f  L = % .10f  phi = % .10f\n\n", E, L, phi);
}

/* Нормировка разности углов в диапазон (-pi, pi] для удобного сравнения. */
double angle_diff(double a, double b)
{
    double d = a - b;
    while (d <= -M_PI) d += 2.0 * M_PI;
    while (d >   M_PI) d -= 2.0 * M_PI;
    return d;
}

/* -------------------- MAIN: ЧИСЛЕННЫЕ ЭКСПЕРИМЕНТЫ -------------------- */

int main(void)
{
    const double T = 20.0;

    /* Начальные условия задачи. */
    State y0;
    y0.x = 0.5;
    y0.y = 0.0;
    y0.z = 0.0;
    y0.u = sqrt(3.0);

    double E0 = energy(&y0);
    double L0 = angular_momentum(&y0);

    printf("Initial state:\n");
    print_state("Y(0)", 0.0, &y0);
    printf("Initial invariants: E0 = %.10f, L0 = %.10f\n\n", E0, L0);

    /* --- Эталонное решение: RK4 с очень малым шагом --- */
    double h_ref = 1e-4;
    State y_ref = integrate_fixed(step_rk4, h_ref, T, &y0, NULL);

    printf("Reference solution (RK4, h = %.1e):\n", h_ref);
    print_state("Y_ref", T, &y_ref);

    double E_ref  = energy(&y_ref);
    double L_ref  = angular_momentum(&y_ref);
    double r_ref  = sqrt(y_ref.x * y_ref.x + y_ref.y * y_ref.y);
    double phi_ref = atan2(y_ref.y, y_ref.x);

    /* --- Сравнение методов и шагов с эталоном --- */
    double h_list[] = {0.10, 0.05, 0.02};
    const int HN = (int)(sizeof(h_list) / sizeof(h_list[0]));

    Stepper methods[] = { step_euler, step_rk2, step_rk4 };
    const char *mnames[] = { "Euler explicit", "RK2", "RK4" };
    const int MCOUNT = (int)(sizeof(methods) / sizeof(methods[0]));

    printf("=== Fixed step methods vs reference (T = %.2f) ===\n\n", T);

    for (int m = 0; m < MCOUNT; ++m) {
        for (int i = 0; i < HN; ++i) {
            double h = h_list[i];

            /* имя файла для траектории, можно потом рисовать орбиту */
            char fname[64];
            snprintf(fname, sizeof(fname), "orbit_%s_h%.2f.dat",
                     (m == 0 ? "euler" : (m == 1 ? "rk2" : "rk4")), h);

            State y_num = integrate_fixed(methods[m], h, T, &y0, fname);

            double E_num  = energy(&y_num);
            double L_num  = angular_momentum(&y_num);
            double r_num  = sqrt(y_num.x * y_num.x + y_num.y * y_num.y);
            double phi_num = atan2(y_num.y, y_num.x);

            double dE   = E_num  - E_ref;
            double dL   = L_num  - L_ref;
            double dr   = r_num  - r_ref;
            double dphi = angle_diff(phi_num, phi_ref);

            printf("%-15s  h = %.3f:\n", mnames[m], h);
            printf("  x(T) = % .8f  y(T) = % .8f  r(T) = %.8f\n",
                   y_num.x, y_num.y, r_num);
            printf("  dE   = % .3e  dL = % .3e  dr = % .3e  dphi = % .3e\n\n",
                   dE, dL, dr, dphi);
        }
    }

    /* --- Адаптивный Эйлер --- */
    double tol = 1e-5;     /* допуск на локальную ошибку */
    double h0  = 0.05;     /* начальный шаг */

    int steps, rejects;
    double h_min, h_max;

    State y_adapt = euler_adaptive(T, h0, tol, &y0,
                                   &steps, &rejects, &h_min, &h_max);

    double E_adapt  = energy(&y_adapt);
    double L_adapt  = angular_momentum(&y_adapt);
    double r_adapt  = sqrt(y_adapt.x * y_adapt.x + y_adapt.y * y_adapt.y);
    double phi_adapt = atan2(y_adapt.y, y_adapt.x);

    double dE_a   = E_adapt  - E_ref;
    double dL_a   = L_adapt  - L_ref;
    double dr_a   = r_adapt  - r_ref;
    double dphi_a = angle_diff(phi_adapt, phi_ref);

    printf("=== Adaptive Euler (tol = %.1e, h0 = %.2f) ===\n", tol, h0);
    print_state("Y_adapt", T, &y_adapt);
    printf("  steps   = %d (accepted)\n", steps);
    printf("  rejects = %d (rejected)\n", rejects);
    printf("  h_min   = %.3e  h_max = %.3e\n", h_min, h_max);
    printf("  dE      = % .3e  dL   = % .3e  dr = % .3e  dphi = % .3e\n",
           dE_a, dL_a, dr_a, dphi_a);

    printf("\nФайлы orbit_*.dat содержат траектории (t, x, y, r, E, L)\n"
           "и могут быть использованы для построения орбит в gnuplot/Excel и т.п.\n");

    return 0;
}
