#include <stdio.h>
#include <math.h>

/* Решаем систему
 *   u' =  98 u + 198 v
 *   v' = -99 u - 199 v
 * с начальными условиями u(0) = v(0) = 1.
 *
 * Система линейна: y' = A y,  y = (u, v)^T,
 *     A = |  98  198 |
 *         | -99 -199 |
 *
 * Собственные значения матрицы:
 *   λ1 = -1, λ2 = -100  (разные по модулю -> жесткая система).
 *
 * Точное решение получается как y(t) = exp(A t) y(0):
 *   u(t) =  4 e^{-t} - 3 e^{-100 t}
 *   v(t) = -2 e^{-t} + 3 e^{-100 t}
 */

typedef struct {
    double u;
    double v;
} State;

/* Правая часть задачи: f(t, y) = A y.
 * t не используется, так как система автономная,
 * но аргумент оставлен для универсальности.
 */
void rhs(double t, const State *y, State *f)
{
    (void)t;  /* подавляем предупреждение об неиспользуемом параметре */

    f->u =  98.0 * y->u + 198.0 * y->v;
    f->v = -99.0 * y->u -199.0 * y->v;
}

/* Точное решение системы в момент времени t. */
State exact_solution(double t)
{
    State y;
    double e1   = exp(-t);       /* медленно убывающая мода λ = -1  */
    double e100 = exp(-100.0*t); /* быстро затухающая мода λ = -100 */

    y.u =  4.0 * e1 - 3.0 * e100;
    y.v = -2.0 * e1 + 3.0 * e100;
    return y;
}

/* ---------- ЯВНЫЙ МЕТОД ЭЙЛЕРА -----------------------------------------
 * y_{n+1} = y_n + h f(t_n, y_n)
 *
 * Для скалярного уравнения w' = λ w множитель роста:
 *   R(z) = 1 + z,  z = h λ.
 * Для λ < 0 устойчивость: |1 + h λ| <= 1  ->  -2 <= h λ <= 0.
 * Для жесткого собственного λ2 = -100 получаем h <= 0.02.
 */
State step_euler_explicit(double t, double h, const State *y)
{
    State f, yn;
    rhs(t, y, &f);

    yn.u = y->u + h * f.u;
    yn.v = y->v + h * f.v;
    return yn;
}

/* ---------- НЕЯВНЫЙ МЕТОД ЭЙЛЕРА --------------------------------------
 * y_{n+1} = y_n + h f(t_{n+1}, y_{n+1}) = y_n + h A y_{n+1}
 * Для линейной системы получаем
 *   (I - h A) y_{n+1} = y_n
 * Здесь A постоянна, поэтому на каждом шаге решаем
 * 2×2 линейную систему с той же матрицей (I - h A).
 *
 * Множитель роста R(z) = 1 / (1 - z), для λ < 0 и h > 0 |R(z)| < 1,
 * т.е. метод A-устойчив: ограничений на шаг по устойчивости нет.
 */
State step_euler_implicit(double t, double h, const State *y)
{
    (void)t;

    /* Матрица I - h A */
    double a11 = 1.0 - 98.0 * h;
    double a12 =      -198.0 * h;
    double a21 =       99.0 * h;   /* -h * (-99) */
    double a22 = 1.0 + 199.0 * h;  /* 1 - h * (-199) */

    /* Решаем систему (2x2) линейных уравнений:
     *   a11 * u_{n+1} + a12 * v_{n+1} = y->u
     *   a21 * u_{n+1} + a22 * v_{n+1} = y->v
     */
    double det = a11 * a22 - a12 * a21;

    State yn;
    yn.u = ( y->u * a22 - y->v * a12) / det;
    yn.v = (-y->u * a21 + y->v * a11) / det;

    return yn;
}

/* ---------- МЕТОД ТРАПЕЦИЙ (КРАНКА–НИКОЛСОНА) ------------------------
 * y_{n+1} = y_n + (h/2) [ f(t_n, y_n) + f(t_{n+1}, y_{n+1}) ]
 * Для линейной системы:
 *   (I - h/2 A) y_{n+1} = (I + h/2 A) y_n
 *
 * Для w' = λ w множитель роста
 *   R(z) = (1 + z/2) / (1 - z/2),
 * что также дает A-устойчивость: для λ < 0 метод устойчив при любом h.
 */
State step_trapezoid(double t, double h, const State *y)
{
    (void)t;

    double s = 0.5 * h;

    /* Матрица (I + h/2 A) */
    double b11 = 1.0 + 98.0 * s;
    double b12 =       198.0 * s;
    double b21 =      -99.0 * s;
    double b22 = 1.0 -199.0 * s;

    /* Вычисляем правую часть: (I + h/2 A) y_n */
    double rhs_u = b11 * y->u + b12 * y->v;
    double rhs_v = b21 * y->u + b22 * y->v;

    /* Матрица (I - h/2 A) */
    double a11 = 1.0 - 98.0 * s;
    double a12 =      -198.0 * s;
    double a21 =       99.0 * s;
    double a22 = 1.0 +199.0 * s;

    /* Решаем (I - h/2 A) y_{n+1} = rhs */
    double det = a11 * a22 - a12 * a21;

    State yn;
    yn.u = ( rhs_u * a22 - rhs_v * a12) / det;
    yn.v = (-rhs_u * a21 + rhs_v * a11) / det;

    return yn;
}

/* ---------- КЛАССИЧЕСКИЙ ЯВНЫЙ РУНГЕ–КУТТА 2-ГО ПОРЯДКА (Хойна) ------
 * Базовый метод порядка p = 2:
 *   k1 = f(t_n,           y_n)
 *   k2 = f(t_n + h, y_n + h k1)
 *   y_{n+1} = y_n + h/2 (k1 + k2)
 *
 * Для w' = λ w множитель роста:
 *   R(z) = 1 + z + z^2/2.
 * Для z по отрицательной действительной оси устойчивость даёт
 * тот же интервал, что и явный Эйлер: -2 <= z <= 0.
 */
State step_rk2(double t, double h, const State *y)
{
    State k1, k2, temp, yn;

    rhs(t, y, &k1);

    temp.u = y->u + h * k1.u;
    temp.v = y->v + h * k1.v;
    rhs(t + h, &temp, &k2);

    yn.u = y->u + 0.5 * h * (k1.u + k2.u);
    yn.v = y->v + 0.5 * h * (k1.v + k2.v);
    return yn;
}

/* Метод Рунге–Кутты 2-го порядка с экстраполяцией Ричардсона.
 * Идея: выполним один шаг длиной h и два шага длиной h/2,
 * затем из двух приближений получим более точное:
 *   y_R = y_{h/2}^{(2 шага)} + (y_{h/2}^{(2 шага)} - y_h) / (2^p - 1)
 * где p = 2 — порядок базового метода.
 *
 * Для линейной задачи множитель роста комбинированного метода
 * очень близок к R(z) базовой схемы, поэтому для оценки устойчивости
 * можно использовать те же границы по шагу.
 */
State step_rk2_richardson(double t, double h, const State *y)
{
    /* Одно "грубое" приближение с шагом h */
    State y_h = step_rk2(t, h, y);

    /* Два "тонких" шага h/2 */
    double h2 = 0.5 * h;
    State y_half = step_rk2(t,      h2, y);
    y_half       = step_rk2(t+h2,   h2, &y_half);

    /* Экстраполяция Ричардсона (p = 2) */
    State yn;
    yn.u = (4.0 * y_half.u - y_h.u) / 3.0;
    yn.v = (4.0 * y_half.v - y_h.v) / 3.0;
    return yn;
}

/* ---------- КЛАССИЧЕСКИЙ РУНГЕ–КУТТА 3-ГО ПОРЯДКА (схема Кутты) -----
 * Базовый метод порядка p = 3:
 *   k1 = f(t,          y)
 *   k2 = f(t + h/2,    y + h/2 k1)
 *   k3 = f(t + h,      y + h(-k1 + 2 k2))
 *   y_{n+1} = y + h/6 (k1 + 4 k2 + k3)
 *
 * Для w' = λ w:
 *   R(z) = 1 + z + z^2/2 + z^3/6.
 * На отрицательной действительной оси R(z) достигает -1 при
 * z_min ≈ -2.512745, что даёт h_max ≈ 2.512745 / 100 ≈ 0.0251.
 */
State step_rk3(double t, double h, const State *y)
{
    State k1, k2, k3, temp, yn;

    rhs(t, y, &k1);

    temp.u = y->u + 0.5 * h * k1.u;
    temp.v = y->v + 0.5 * h * k1.v;
    rhs(t + 0.5 * h, &temp, &k2);

    temp.u = y->u + h * (-k1.u + 2.0 * k2.u);
    temp.v = y->v + h * (-k1.v + 2.0 * k2.v);
    rhs(t + h, &temp, &k3);

    yn.u = y->u + (h / 6.0) * (k1.u + 4.0 * k2.u + k3.u);
    yn.v = y->v + (h / 6.0) * (k1.v + 4.0 * k2.v + k3.v);
    return yn;
}

/* Рунге–Кутта 3-го порядка + экстраполяция Ричардсона (p = 3).
 *   y_R = (2^p y_{h/2}^{(2 шага)} - y_h) / (2^p - 1) = (8 y_{h/2} - y_h)/7
 */
State step_rk3_richardson(double t, double h, const State *y)
{
    State y_h = step_rk3(t, h, y);

    double h2 = 0.5 * h;
    State y_half = step_rk3(t,      h2, y);
    y_half       = step_rk3(t+h2,   h2, &y_half);

    State yn;
    yn.u = (8.0 * y_half.u - y_h.u) / 7.0;
    yn.v = (8.0 * y_half.v - y_h.v) / 7.0;
    return yn;
}

/* ---------- КЛАССИЧЕСКИЙ РУНГЕ–КУТТА 4-ГО ПОРЯДКА -------------------
 * Базовый метод порядка p = 4:
 *   k1 = f(t,        y)
 *   k2 = f(t + h/2,  y + h/2 k1)
 *   k3 = f(t + h/2,  y + h/2 k2)
 *   k4 = f(t + h,    y + h   k3)
 *   y_{n+1} = y + h/6 (k1 + 2 k2 + 2 k3 + k4)
 *
 * Для w' = λ w:
 *   R(z) = 1 + z + z^2/2 + z^3/6 + z^4/24.
 * На отрицательной действительной оси R(z) вновь становится равен 1
 * при z_min ≈ -2.785294, что даёт h_max ≈ 2.785294 / 100 ≈ 0.0279.
 */
State step_rk4(double t, double h, const State *y)
{
    State k1, k2, k3, k4, temp, yn;

    rhs(t, y, &k1);

    temp.u = y->u + 0.5 * h * k1.u;
    temp.v = y->v + 0.5 * h * k1.v;
    rhs(t + 0.5 * h, &temp, &k2);

    temp.u = y->u + 0.5 * h * k2.u;
    temp.v = y->v + 0.5 * h * k2.v;
    rhs(t + 0.5 * h, &temp, &k3);

    temp.u = y->u + h * k3.u;
    temp.v = y->v + h * k3.v;
    rhs(t + h, &temp, &k4);

    yn.u = y->u + (h / 6.0) * (k1.u + 2.0 * k2.u + 2.0 * k3.u + k4.u);
    yn.v = y->v + (h / 6.0) * (k1.v + 2.0 * k2.v + 2.0 * k3.v + k4.v);
    return yn;
}

/* Рунге–Кутта 4-го порядка + экстраполяция Ричардсона (p = 4).
 *   y_R = (2^4 y_{h/2} - y_h) / (2^4 - 1) = (16 y_{h/2} - y_h)/15
 */
State step_rk4_richardson(double t, double h, const State *y)
{
    State y_h = step_rk4(t, h, y);

    double h2 = 0.5 * h;
    State y_half = step_rk4(t,      h2, y);
    y_half       = step_rk4(t+h2,   h2, &y_half);

    State yn;
    yn.u = (16.0 * y_half.u - y_h.u) / 15.0;
    yn.v = (16.0 * y_half.v - y_h.v) / 15.0;
    return yn;
}

/* Тип "один шаг метода". */
typedef State (*Stepper)(double t, double h, const State *y);

/* В примере ниже выбираем h = 0.01 (< 0.02),
 * что заведомо находится в области устойчивости всех явных схем
 * и демонстрирует типичное поведение жесткой системы:
 * быстрая компонента (λ = -100) быстро затухает, остаётся медленная (λ = -1).
 */

int main(void)
{
    /* Конечное время интегрирования и шаг */
    const double T = 1.0;
    const int    N = 100;       /* число шагов */
    const double h = T / (double)N;

    /* Массив методов и их названия */
    Stepper methods[] = {
        step_euler_explicit,
        step_euler_implicit,
        step_trapezoid,
        step_rk2_richardson,
        step_rk3_richardson,
        step_rk4_richardson
    };
    const char *method_names[] = {
        "Euler explicit",
        "Euler implicit",
        "Trapezoid",
        "RK2 + Richardson",
        "RK3 + Richardson",
        "RK4 + Richardson"
    };
    const int METHOD_COUNT = (int)(sizeof(methods) / sizeof(methods[0]));

    printf("Solving u' = 98u + 198v, v' = -99u - 199v on [0, %.2f]\n", T);
    printf("Initial condition: u(0) = v(0) = 1\n");
    printf("Time step h = %.5f (%d steps)\n\n", h, N);

    /* Точное решение в конечный момент времени */
    State y_exact = exact_solution(T);

    printf("Exact solution at T = %.2f:\n", T);
    printf("  u(T) = %.10f, v(T) = %.10f\n\n", y_exact.u, y_exact.v);

    printf("Numerical solutions and max-norm errors:\n\n");

    for (int m = 0; m < METHOD_COUNT; ++m) {
        Stepper step = methods[m];

        /* Начальное состояние */
        State y;
        y.u = 1.0;
        y.v = 1.0;

        double t = 0.0;

        /* Основной цикл интегрирования */
        for (int n = 0; n < N; ++n) {
            y = step(t, h, &y);
            t += h;
        }

        /* Оценка ошибки: ||y_num - y_exact||_inf */
        double err_u = fabs(y.u - y_exact.u);
        double err_v = fabs(y.v - y_exact.v);
        double err_inf = (err_u > err_v) ? err_u : err_v;

        printf("%-20s: u(T) = % .10f, v(T) = % .10f,  |error|_inf = %.3e\n",
               method_names[m], y.u, y.v, err_inf);
    }

    return 0;
}
