#include <cmath>
#include <fstream>
#include <iomanip>
#include <iostream>
#include <limits>
#include <string>
#include <vector>

struct Params
{
    double Du;      // коэффициент диффузии для u
    double Dv;      // коэффициент диффузии для v
    double alpha;   // параметр alpha в h(s)
    double beta;    // параметр beta  в h(s)
    double gamma;   // параметр gamma
};

struct IntegrationResult
{
    // конечные значения основной системы в x=1
    double u = 0.0;
    double up = 0.0;
    double v = 0.0;
    double vp = 0.0;

    // матрица Якоби невязки по пристрелочным параметрам
    // J = [ dPhi1/dp1  dPhi1/dp2
    //       dPhi2/dp1  dPhi2/dp2 ]
    double J11 = 0.0;
    double J12 = 0.0;
    double J21 = 0.0;
    double J22 = 0.0;

    // невязки
    double Phi1 = 0.0;
    double Phi2 = 0.0;

    // Диагностика опасной близости к полюсам h(s)
    double min_abs_denom = std::numeric_limits<double>::infinity();
    double denom_at_min = 0.0;
    double x_at_min_denom = 0.0;
    double s_at_min_denom = 0.0;
    double u_at_min_denom = 0.0;
    double v_at_min_denom = 0.0;
};

// z(x) = gamma / (1 + gamma) - x
double z_func(double x, const Params& p)
{
    return p.gamma / (1.0 + p.gamma) - x;
}

// h(s) = s / (alpha + s + beta * s^2)
double denom_func(double s, const Params& p)
{
    return p.alpha + s + p.beta * s * s;
}

double h_func(double s, const Params& p)
{
    const double denom = denom_func(s, p);
    return s / denom;
}

// h'(s) = (alpha - beta * s^2) / (alpha + s + beta*s^2)^2
double dh_func(double s, const Params& p)
{
    const double denom = denom_func(s, p);
    return (p.alpha - p.beta * s * s) / (denom * denom);
}

// f(u,v) = u * h(z(x) - u - v)
double f_func(double x, double u, double v, const Params& p)
{
    const double s = z_func(x, p) - u - v;
    return u * h_func(s, p);
}

// g(u,v) = v * h(z(x) - u - v)
double g_func(double x, double u, double v, const Params& p)
{
    const double s = z_func(x, p) - u - v;
    return v * h_func(s, p);
}

// Частные производные f_u, f_v, g_u, g_v
void calc_partials(double x, double u, double v, const Params& p,
                   double& fu, double& fv, double& gu, double& gv)
{
    const double s  = z_func(x, p) - u - v;
    const double h  = h_func(s, p);
    const double dh = dh_func(s, p);

    // f(u,v) = u * h(s), s = z - u - v
    // df/du = h(s) + u * h'(s) * ds/du = h(s) - u*h'(s)
    // df/dv = u * h'(s) * ds/dv = -u*h'(s)
    fu = h - u * dh;
    fv = -u * dh;

    // g(u,v) = v * h(s)
    // dg/du = -v*h'(s)
    // dg/dv = h(s) - v*h'(s)
    gu = -v * dh;
    gv = h - v * dh;
}

// Правая часть объединённой системы:
// основная система + 2 вариационные системы
//
// Вектор Y имеет 12 компонент:
//
// 0: u
// 1: u'
// 2: v
// 3: v'
//
// Вариации по p1 = u(0):
// 4:  du/dp1
// 5:  d(u')/dp1
// 6:  dv/dp1
// 7:  d(v')/dp1
//
// Вариации по p2 = v(0):
// 8:  du/dp2
// 9:  d(u')/dp2
// 10: dv/dp2
// 11: d(v')/dp2
std::vector<double> rhs(double x, const std::vector<double>& Y, const Params& p)
{
    std::vector<double> dY(12, 0.0);

    const double u  = Y[0];
    const double up = Y[1];
    const double v  = Y[2];
    const double vp = Y[3];

    dY[0] = up;
    dY[1] = -(1.0 / p.Du) * f_func(x, u, v, p);
    dY[2] = vp;
    dY[3] = -(1.0 / p.Dv) * g_func(x, u, v, p);

    double fu, fv, gu, gv;
    calc_partials(x, u, v, p, fu, fv, gu, gv);

    // -------- Вариации по p1 --------
    {
        const double du  = Y[4];
        const double dup = Y[5];
        const double dv  = Y[6];
        const double dvp = Y[7];

        dY[4] = dup;
        dY[5] = -(1.0 / p.Du) * (fu * du + fv * dv);
        dY[6] = dvp;
        dY[7] = -(1.0 / p.Dv) * (gu * du + gv * dv);
    }

    // -------- Вариации по p2 --------
    {
        const double du  = Y[8];
        const double dup = Y[9];
        const double dv  = Y[10];
        const double dvp = Y[11];

        dY[8]  = dup;
        dY[9]  = -(1.0 / p.Du) * (fu * du + fv * dv);
        dY[10] = dvp;
        dY[11] = -(1.0 / p.Dv) * (gu * du + gv * dv);
    }

    return dY;
}

void rk4_step(double& x, std::vector<double>& Y, double h, const Params& p)
{
    const std::vector<double> k1 = rhs(x, Y, p);

    std::vector<double> Ytmp(12);
    for (int i = 0; i < 12; ++i)
        Ytmp[i] = Y[i] + 0.5 * h * k1[i];
    const std::vector<double> k2 = rhs(x + 0.5 * h, Ytmp, p);

    for (int i = 0; i < 12; ++i)
        Ytmp[i] = Y[i] + 0.5 * h * k2[i];
    const std::vector<double> k3 = rhs(x + 0.5 * h, Ytmp, p);

    for (int i = 0; i < 12; ++i)
        Ytmp[i] = Y[i] + h * k3[i];
    const std::vector<double> k4 = rhs(x + h, Ytmp, p);

    for (int i = 0; i < 12; ++i)
    {
        Y[i] += (h / 6.0) * (k1[i] + 2.0 * k2[i] + 2.0 * k3[i] + k4[i]);
    }

    x += h;
}

IntegrationResult integrate_system(double p1, double p2,
                                   const Params& par,
                                   int N,
                                   bool save_profile = false,
                                   const std::string& filename = "")
{
    // Начальные данные в x=0:
    // u(0)=p1, u'(0)=0
    // v(0)=p2, v'(0)=0
    //
    // Для вариаций:
    //
    // По p1:
    // d(u(0))/dp1 = 1, d(u'(0))/dp1 = 0
    // d(v(0))/dp1 = 0, d(v'(0))/dp1 = 0
    //
    // По p2:
    // d(u(0))/dp2 = 0, d(u'(0))/dp2 = 0
    // d(v(0))/dp2 = 1, d(v'(0))/dp2 = 0

    std::vector<double> Y(12, 0.0);
    Y[0] = p1;
    Y[1] = 0.0;
    Y[2] = p2;
    Y[3] = 0.0;

    Y[4] = 1.0;
    Y[5] = 0.0;
    Y[6] = 0.0;
    Y[7] = 0.0;

    Y[8]  = 0.0;
    Y[9]  = 0.0;
    Y[10] = 1.0;
    Y[11] = 0.0;

    double x = 0.0;
    const double h = 1.0 / static_cast<double>(N);
    IntegrationResult res;

    std::ofstream fout;
    if (save_profile)
    {
        fout.open(filename);
        fout << "x,u,up,v,vp\n";
        fout << std::setprecision(16)
             << x << "," << Y[0] << "," << Y[1] << "," << Y[2] << "," << Y[3] << "\n";
    }

    for (int i = 0; i < N; ++i)
    {
        const double s = z_func(x, par) - Y[0] - Y[2];
        const double denom = denom_func(s, par);
        if (std::abs(denom) < res.min_abs_denom)
        {
            res.min_abs_denom = std::abs(denom);
            res.denom_at_min = denom;
            res.x_at_min_denom = x;
            res.s_at_min_denom = s;
            res.u_at_min_denom = Y[0];
            res.v_at_min_denom = Y[2];
        }

        rk4_step(x, Y, h, par);

        if (save_profile)
        {
            fout << std::setprecision(16)
                 << x << "," << Y[0] << "," << Y[1] << "," << Y[2] << "," << Y[3] << "\n";
        }
    }

    res.u  = Y[0];
    res.up = Y[1];
    res.v  = Y[2];
    res.vp = Y[3];

    const double s_right = z_func(x, par) - res.u - res.v;
    const double denom_right = denom_func(s_right, par);
    if (std::abs(denom_right) < res.min_abs_denom)
    {
        res.min_abs_denom = std::abs(denom_right);
        res.denom_at_min = denom_right;
        res.x_at_min_denom = x;
        res.s_at_min_denom = s_right;
        res.u_at_min_denom = res.u;
        res.v_at_min_denom = res.v;
    }

    res.Phi1 = res.up + par.gamma * res.u;
    res.Phi2 = res.vp + par.gamma * res.v;

    // Якобиан:
    // Phi1 = u'(1) + gamma*u(1)
    // dPhi1/dp1 = d(u'(1))/dp1 + gamma*d(u(1))/dp1
    // dPhi1/dp2 = d(u'(1))/dp2 + gamma*d(u(1))/dp2
    //
    // Phi2 = v'(1) + gamma*v(1)
    // dPhi2/dp1 = d(v'(1))/dp1 + gamma*d(v(1))/dp1
    // dPhi2/dp2 = d(v'(1))/dp2 + gamma*d(v(1))/dp2

    res.J11 = Y[5]  + par.gamma * Y[4];
    res.J12 = Y[9]  + par.gamma * Y[8];
    res.J21 = Y[7]  + par.gamma * Y[6];
    res.J22 = Y[11] + par.gamma * Y[10];

    return res;
}

// Решение 2x2 системы:
// [a b] [x] = [r1]
// [c d] [y]   [r2]
bool solve_2x2(double a, double b, double c, double d,
               double r1, double r2,
               double& x, double& y)
{
    const double det = a * d - b * c;

    if (std::abs(det) < 1e-14)
        return false;

    x = ( r1 * d - b * r2) / det;
    y = ( a * r2 - r1 * c) / det;
    return true;
}

bool newton_shooting(double& p1, double& p2,
                     const Params& par,
                     int N,
                     int max_iter,
                     double tol,
                     bool verbose = true)
{
    for (int iter = 0; iter < max_iter; ++iter)
    {
        IntegrationResult res = integrate_system(p1, p2, par, N, false);

        const double normPhi = std::sqrt(res.Phi1 * res.Phi1 + res.Phi2 * res.Phi2);
        const double detJ = res.J11 * res.J22 - res.J12 * res.J21;

        if (verbose)
        {
            std::cout << "  Newton iter = " << iter
                      << "  p1 = " << p1
                      << "  p2 = " << p2
                      << "  |Phi| = " << normPhi
                      << "\n";
            std::cout << "    Phi = (" << res.Phi1 << ", " << res.Phi2 << ")"
                      << "  det(J) = " << detJ
                      << "\n";
            std::cout << "    min |alpha + s + beta*s^2| = " << res.min_abs_denom
                      << "  at x = " << res.x_at_min_denom
                      << "  s = " << res.s_at_min_denom
                      << "  denom = " << res.denom_at_min
                      << "\n";
            std::cout << "    state at minimum: u = " << res.u_at_min_denom
                      << "  v = " << res.v_at_min_denom
                      << "\n";
        }

        if (normPhi < tol)
            return true;

        // Решаем J * delta = -Phi
        double dp1 = 0.0, dp2 = 0.0;
        const bool ok = solve_2x2(
            res.J11, res.J12,
            res.J21, res.J22,
            -res.Phi1, -res.Phi2,
            dp1, dp2
        );

        if (!ok)
        {
            std::cerr << "  Ошибка: матрица Якоби вырождена или почти вырождена.\n";
            return false;
        }

        const double step_norm = std::sqrt(dp1 * dp1 + dp2 * dp2);

        if (verbose)
        {
            std::cout << "    delta = (" << dp1 << ", " << dp2 << ")"
                      << "  |delta| = " << step_norm
                      << "\n";
        }

        if (step_norm == 0.0 && normPhi >= tol)
        {
            std::cerr << "  Диагностика: шаг Ньютона занулился, хотя невязка ещё большая."
                      << " Вероятна потеря точности из-за огромных элементов Якобиана"
                      << " или близости к полюсу h(s).\n";
        }

        p1 += dp1;
        p2 += dp2;

        // Дополнительная защита от разлёта
        if (!std::isfinite(p1) || !std::isfinite(p2))
        {
            std::cerr << "  Ошибка: пристрелочные параметры стали нечисловыми.\n";
            return false;
        }
    }

    return false;
}

void save_solution_profile(double p1, double p2,
                           const Params& par,
                           int N,
                           const std::string& filename)
{
    (void)integrate_system(p1, p2, par, N, true, filename);
}

int main()
{
    std::cout << std::fixed << std::setprecision(10);

    Params par;
    par.Du    = 0.005;   // было: 0.01
    par.Dv    = 0.005;   // было: 0.05
    par.alpha = 0.5;     // было: 0.2
    par.beta  = 1.0;     // было: 0.3
    par.gamma = -0.95;   // было: 0.2

    const int N = 4000;

    const int max_newton_iter = 1000;
    const double tol = 1e-6;

    double p1 = 10.0;    // было: 0.1
    double p2 = 10.0;    // было: 0.3

    const double discr = 1.0 - 4.0 * par.alpha * par.beta;
    if (discr >= 0.0)
    {
        const double sqrt_discr = std::sqrt(discr);
        const double pole1 = (-1.0 + sqrt_discr) / (2.0 * par.beta);
        const double pole2 = (-1.0 - sqrt_discr) / (2.0 * par.beta);
        std::cout << "Полюса знаменателя h(s): s = " << pole1
                  << " и s = " << pole2 << "\n";
    }

    std::cout << "=== Решение для одного значения gamma ===\n";
    std::cout << "gamma = " << par.gamma << "\n";

    bool ok = newton_shooting(p1, p2, par, N, max_newton_iter, tol, true);

    if (!ok)
    {
        std::cerr << "Не удалось найти решение методом Ньютона.\n";
        return 1;
    }

    IntegrationResult res = integrate_system(p1, p2, par, N, false);

    std::cout << "\nНайденные пристрелочные параметры:\n";
    std::cout << "u(0) = " << p1 << "\n";
    std::cout << "v(0) = " << p2 << "\n";

    std::cout << "\nЗначения на правом конце:\n";
    std::cout << "u(1)  = " << res.u  << "\n";
    std::cout << "u'(1) = " << res.up << "\n";
    std::cout << "v(1)  = " << res.v  << "\n";
    std::cout << "v'(1) = " << res.vp << "\n";

    std::cout << "\nПроверка граничных условий справа:\n";
    std::cout << "u'(1) + gamma*u(1) = " << res.Phi1 << "\n";
    std::cout << "v'(1) + gamma*v(1) = " << res.Phi2 << "\n";

    // Сохраняем профиль решения
    save_solution_profile(p1, p2, par, N, "solution_single_gamma.csv");
    std::cout << "\nПрофиль решения сохранён в solution_single_gamma.csv\n";

    // меняем gamma по сетке и используем предыдущее найденное
    // решение как начальное приближение для следующего gamma.
    std::cout << "\n=== Исследование по gamma ===\n";

    std::ofstream branch("branch.csv");
    branch << "gamma,u0,v0,u1,v1,phi1,phi2\n";

    double gamma_min = -0.95;   // было: 0.05
    double gamma_max = -0.50;   // было: 1.00
    int gamma_steps = 10;       // было: 20

    double cont_p1 = p1;
    double cont_p2 = p2;

    for (int k = 0; k <= gamma_steps; ++k)
    {
        const double gamma =
            gamma_min + (gamma_max - gamma_min) * static_cast<double>(k) / gamma_steps;

        par.gamma = gamma;

        std::cout << "\n--- gamma = " << gamma << " ---\n";

        bool local_ok = newton_shooting(cont_p1, cont_p2, par, N, max_newton_iter, tol, false);

        if (!local_ok)
        {
            std::cout << "  Решение не найдено для gamma = " << gamma << "\n";
            branch << std::setprecision(16)
                   << gamma << ",nan,nan,nan,nan,nan,nan\n";
            continue;
        }

        IntegrationResult rr = integrate_system(cont_p1, cont_p2, par, N, false);

        std::cout << "  u(0) = " << cont_p1
                  << "  v(0) = " << cont_p2
                  << "  u(1) = " << rr.u
                  << "  v(1) = " << rr.v
                  << "  |Phi| = " << std::sqrt(rr.Phi1 * rr.Phi1 + rr.Phi2 * rr.Phi2)
                  << "\n";

        branch << std::setprecision(16)
               << gamma << ","
               << cont_p1 << ","
               << cont_p2 << ","
               << rr.u << ","
               << rr.v << ","
               << rr.Phi1 << ","
               << rr.Phi2 << "\n";
    }

    branch.close();
    std::cout << "\nДанные ветви решения сохранены в branch.csv\n";

    return 0;
}
