#include <algorithm>
#include <array>
#include <cmath>
#include <filesystem>
#include <fstream>
#include <iomanip>
#include <iostream>
#include <limits>
#include <stdexcept>
#include <string>
#include <vector>

namespace
{

constexpr double kPi = 3.141592653589793238462643383279502884;
constexpr double kT  = 1.0;
constexpr int kTimeFactor = 20;
constexpr std::array<int, 4> kGridSizes = {50, 100, 200, 400};
constexpr std::array<double, 3> kLambdas = {-3.0, 0.0, 17.0};

enum class Scheme
{
    Lax,
    Box
};

struct SystemSolution
{
    int M = 0;
    int N = 0;
    double h = 0.0;
    double tau = 0.0;
    std::vector<double> x;
    std::vector<std::array<double, 3>> u;
    std::vector<std::array<double, 3>> exact;
    double max_error = 0.0;
};

double characteristic_initial(int index, double x)
{
    const double s = std::sin(kPi * x);
    const double c = std::cos(kPi * x);

    switch (index)
    {
        case 0:
            return 0.5 + (8.0 / 9.0) * s + (5.0 / 18.0) * c;
        case 1:
            return -(1.0 / 9.0) * s - (2.0 / 9.0) * c;
        case 2:
            return 0.5 + (1.0 / 3.0) * s + (1.0 / 6.0) * c;
        default:
            throw std::runtime_error("invalid characteristic index");
    }
}

double characteristic_exact(int index, double x, double t)
{
    if (index == 1)
    {
        return characteristic_initial(index, x);
    }

    return characteristic_initial(index, x - kLambdas[index] * t);
}

double inflow_value(int index, double t)
{
    if (kLambdas[index] > 0.0)
    {
        return characteristic_exact(index, 0.0, t);
    }
    if (kLambdas[index] < 0.0)
    {
        return characteristic_exact(index, 1.0, t);
    }

    return 0.0;
}

std::array<double, 3> reconstruct_u(double w_minus, double w_zero, double w_plus)
{
    return {
        2.0 * w_minus + w_zero - 2.0 * w_plus,
        -w_minus - 5.0 * w_zero + w_plus,
        w_minus + 2.0 * w_zero + w_plus
    };
}

std::array<double, 3> exact_u(double x, double t)
{
    return reconstruct_u(
        characteristic_exact(0, x, t),
        characteristic_exact(1, x, t),
        characteristic_exact(2, x, t)
    );
}

std::vector<double> solve_scalar_lax(int index, int M, int N)
{
    std::vector<double> current(M + 1, 0.0);
    std::vector<double> next(M + 1, 0.0);

    const double h = 1.0 / static_cast<double>(M);
    const double tau = kT / static_cast<double>(N);
    const double lambda = kLambdas[index];
    const double r = lambda * tau / h;

    for (int m = 0; m <= M; ++m)
    {
        current[m] = characteristic_initial(index, m * h);
    }

    if (std::abs(lambda) < 1e-14)
    {
        return current;
    }

    for (int n = 0; n < N; ++n)
    {
        const double t_next = (n + 1) * tau;

        for (int m = 1; m < M; ++m)
        {
            next[m] = 0.5 * (current[m + 1] + current[m - 1])
                    - 0.5 * r * (current[m + 1] - current[m - 1]);
        }

        if (lambda > 0.0)
        {
            next[0] = inflow_value(index, t_next);
            next[M] = next[M - 1];
        }
        else
        {
            next[M] = inflow_value(index, t_next);
            next[0] = next[1];
        }

        current.swap(next);
    }

    return current;
}

std::vector<double> solve_scalar_box(int index, int M, int N)
{
    std::vector<double> current(M + 1, 0.0);
    std::vector<double> next(M + 1, 0.0);

    const double h = 1.0 / static_cast<double>(M);
    const double tau = kT / static_cast<double>(N);
    const double lambda = kLambdas[index];
    const double r = lambda * tau / h;

    for (int m = 0; m <= M; ++m)
    {
        current[m] = characteristic_initial(index, m * h);
    }

    if (std::abs(lambda) < 1e-14)
    {
        return current;
    }

    if (std::abs(1.0 + r) < 1e-14 || std::abs(1.0 - r) < 1e-14)
    {
        throw std::runtime_error("degenerate Courant number for box scheme");
    }

    for (int n = 0; n < N; ++n)
    {
        const double t_next = (n + 1) * tau;

        if (lambda > 0.0)
        {
            next[0] = inflow_value(index, t_next);

            for (int m = 0; m < M; ++m)
            {
                next[m + 1] = ((1.0 - r) * current[m + 1]
                             + (1.0 + r) * current[m]
                             - (1.0 - r) * next[m]) / (1.0 + r);
            }
        }
        else
        {
            next[M] = inflow_value(index, t_next);

            for (int m = M - 1; m >= 0; --m)
            {
                next[m] = ((1.0 - r) * current[m + 1]
                         + (1.0 + r) * current[m]
                         - (1.0 + r) * next[m + 1]) / (1.0 - r);
            }
        }

        current.swap(next);
    }

    return current;
}

SystemSolution solve_system(Scheme scheme, int M)
{
    SystemSolution solution;
    solution.M = M;
    solution.N = kTimeFactor * M;
    solution.h = 1.0 / static_cast<double>(M);
    solution.tau = kT / static_cast<double>(solution.N);
    solution.x.resize(M + 1);
    solution.u.resize(M + 1);
    solution.exact.resize(M + 1);

    std::array<std::vector<double>, 3> w;
    for (int index = 0; index < 3; ++index)
    {
        if (scheme == Scheme::Lax)
        {
            w[index] = solve_scalar_lax(index, M, solution.N);
        }
        else
        {
            w[index] = solve_scalar_box(index, M, solution.N);
        }
    }

    double max_error = 0.0;
    for (int m = 0; m <= M; ++m)
    {
        const double x = m * solution.h;
        solution.x[m] = x;
        solution.u[m] = reconstruct_u(w[0][m], w[1][m], w[2][m]);
        solution.exact[m] = exact_u(x, kT);

        for (int comp = 0; comp < 3; ++comp)
        {
            max_error = std::max(
                max_error,
                std::abs(solution.u[m][comp] - solution.exact[m][comp])
            );
        }
    }

    solution.max_error = max_error;
    return solution;
}

double max_difference_between_grids(const SystemSolution& coarse,
                                    const SystemSolution& fine)
{
    if (coarse.M <= 0 || fine.M % coarse.M != 0)
    {
        throw std::runtime_error("incompatible grids for convergence estimate");
    }

    const int ratio = fine.M / coarse.M;
    double diff = 0.0;

    for (int m = 0; m <= coarse.M; ++m)
    {
        const auto& uc = coarse.u[m];
        const auto& uf = fine.u[m * ratio];
        for (int comp = 0; comp < 3; ++comp)
        {
            diff = std::max(diff, std::abs(uc[comp] - uf[comp]));
        }
    }

    return diff;
}

double safe_order(double coarse_error, double fine_error)
{
    if (coarse_error <= 0.0 || fine_error <= 0.0)
    {
        return std::numeric_limits<double>::quiet_NaN();
    }

    return std::log(coarse_error / fine_error) / std::log(2.0);
}

std::string scheme_name(Scheme scheme)
{
    switch (scheme)
    {
        case Scheme::Lax:
            return "lax";
        case Scheme::Box:
            return "box";
    }

    return "unknown";
}

void save_final_profile(const std::filesystem::path& path,
                        const SystemSolution& lax,
                        const SystemSolution& box)
{
    std::ofstream out(path);
    out << "x,"
        << "lax_u1,lax_u2,lax_u3,"
        << "box_u1,box_u2,box_u3,"
        << "exact_u1,exact_u2,exact_u3\n";
    out << std::setprecision(16);

    for (int m = 0; m <= lax.M; ++m)
    {
        out << lax.x[m] << ","
            << lax.u[m][0] << "," << lax.u[m][1] << "," << lax.u[m][2] << ","
            << box.u[m][0] << "," << box.u[m][1] << "," << box.u[m][2] << ","
            << lax.exact[m][0] << "," << lax.exact[m][1] << "," << lax.exact[m][2]
            << "\n";
    }
}

void save_convergence_csv(const std::filesystem::path& path,
                          const std::vector<SystemSolution>& lax_solutions,
                          const std::vector<SystemSolution>& box_solutions)
{
    std::ofstream out(path);
    out << "scheme,M,N,h,tau,error_max,error_ratio_to_prev,order_from_exact\n";
    out << std::setprecision(16);

    const auto write_rows = [&out](Scheme scheme,
                                   const std::vector<SystemSolution>& solutions)
    {
        double prev_error = std::numeric_limits<double>::quiet_NaN();
        for (const auto& sol : solutions)
        {
            const double ratio = std::isnan(prev_error)
                               ? std::numeric_limits<double>::quiet_NaN()
                               : prev_error / sol.max_error;
            const double order = std::isnan(prev_error)
                               ? std::numeric_limits<double>::quiet_NaN()
                               : safe_order(prev_error, sol.max_error);

            out << scheme_name(scheme) << ","
                << sol.M << ","
                << sol.N << ","
                << sol.h << ","
                << sol.tau << ","
                << sol.max_error << ","
                << ratio << ","
                << order << "\n";

            prev_error = sol.max_error;
        }
    };

    write_rows(Scheme::Lax, lax_solutions);
    write_rows(Scheme::Box, box_solutions);
}

void save_summary(const std::filesystem::path& path,
                  const std::vector<SystemSolution>& lax_solutions,
                  const std::vector<SystemSolution>& box_solutions)
{
    std::ofstream out(path);
    out << std::fixed << std::setprecision(8);
    out << "Task 3: linear hyperbolic system, variant 17\n";
    out << "Courant numbers: lambda = {-3, 0, 17}, tau = h / 20.\n";
    out << "Therefore r_- = -0.15, r_0 = 0, r_+ = 0.85.\n\n";

    const auto write_table = [&out](const char* title,
                                    const std::vector<SystemSolution>& solutions)
    {
        out << title << "\n";
        out << "M\tN\tmax_error\torder_exact\n";

        double prev_error = std::numeric_limits<double>::quiet_NaN();
        for (const auto& sol : solutions)
        {
            const double order = std::isnan(prev_error)
                               ? std::numeric_limits<double>::quiet_NaN()
                               : safe_order(prev_error, sol.max_error);

            out << sol.M << "\t"
                << sol.N << "\t"
                << sol.max_error << "\t";

            if (std::isnan(order))
            {
                out << "-";
            }
            else
            {
                out << order;
            }
            out << "\n";

            prev_error = sol.max_error;
        }

        out << "\n";
    };

    write_table("Lax scheme", lax_solutions);
    write_table("Box scheme", box_solutions);

    out << "A posteriori orders from three nested grids\n";
    out << "scheme\t(M,2M,4M)\tdiff_h_h2\tdiff_h2_h4\torder\n";

    const auto write_apost = [&out](const char* title,
                                    const std::vector<SystemSolution>& solutions)
    {
        for (std::size_t i = 0; i + 2 < solutions.size(); ++i)
        {
            const double d1 = max_difference_between_grids(solutions[i], solutions[i + 1]);
            const double d2 = max_difference_between_grids(solutions[i + 1], solutions[i + 2]);
            const double p = safe_order(d1, d2);

            out << title << "\t("
                << solutions[i].M << ","
                << solutions[i + 1].M << ","
                << solutions[i + 2].M << ")\t"
                << d1 << "\t"
                << d2 << "\t"
                << p << "\n";
        }
    };

    write_apost("lax", lax_solutions);
    write_apost("box", box_solutions);
}

}

int main()
{
    try
    {
        const std::filesystem::path output_dir =
            "Sem6/Task3/results";
        std::filesystem::create_directories(output_dir);

        std::vector<SystemSolution> lax_solutions;
        std::vector<SystemSolution> box_solutions;
        lax_solutions.reserve(kGridSizes.size());
        box_solutions.reserve(kGridSizes.size());

        for (const int M : kGridSizes)
        {
            lax_solutions.push_back(solve_system(Scheme::Lax, M));
            box_solutions.push_back(solve_system(Scheme::Box, M));
        }

        save_convergence_csv(output_dir / "convergence.csv", lax_solutions, box_solutions);
        save_summary(output_dir / "summary.txt", lax_solutions, box_solutions);
        save_final_profile(output_dir / "final_profile_M400.csv",
                           lax_solutions.back(),
                           box_solutions.back());

        std::cout << std::fixed << std::setprecision(8);
        std::cout << "Results written to " << output_dir << "\n";
        std::cout << "Final max error, Lax  (M=400): " << lax_solutions.back().max_error << "\n";
        std::cout << "Final max error, Box  (M=400): " << box_solutions.back().max_error << "\n";
    }
    catch (const std::exception& ex)
    {
        std::cerr << "Error: " << ex.what() << "\n";
        return 1;
    }

    return 0;
}
