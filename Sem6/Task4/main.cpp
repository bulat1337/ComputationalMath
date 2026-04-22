#include <algorithm>
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
constexpr double kEps = 1e-6;
constexpr int kMaxIterations = 300000;
const std::vector<int> kGridSizes = {8, 16, 32, 64, 128, 256};

enum class Method
{
    ChebyshevRichardson,
    AlternatingDirections
};

struct Problem
{
    int N = 0;
    int m = 0;
    double h = 0.0;
    std::vector<double> rhs;
};

struct SolveResult
{
    Method method = Method::ChebyshevRichardson;
    int N = 0;
    int m = 0;
    int unknowns = 0;
    int iterations = 0;
    int parameter_count = 0;
    bool converged = false;
    double h = 0.0;
    double residual_inf = 0.0;
    double scaled_residual_inf = 0.0;
    double error_bound_l2 = 0.0;
    double lambda_min = 0.0;
    double lambda_max = 0.0;
    std::vector<double> u;
};
int index_of(int i, int j, int m)
{
    return (j - 1) * m + (i - 1);
}
std::string method_name(Method method)
{
    switch (method)
    {
        case Method::ChebyshevRichardson:
            return "chebyshev";
        case Method::AlternatingDirections:
            return "adi";
    }

    return "unknown";
}
double bottom_boundary(double x)
{
    return 1.0 - x * x;
}

double top_boundary(double x)
{
    return 1.0 + x * x;
}

double left_boundary(double y)
{
    return y * y;
}

double right_boundary(double y)
{
    return 2.0 * y * y;
}
double boundary_value(int i, int j, int N)
{
    const double x = static_cast<double>(i) / static_cast<double>(N);
    const double y = static_cast<double>(j) / static_cast<double>(N);

    double sum = 0.0;
    int count = 0;

    if (j == 0)
    {
        sum += bottom_boundary(x);
        ++count;
    }
    if (j == N)
    {
        sum += top_boundary(x);
        ++count;
    }
    if (i == 0)
    {
        sum += left_boundary(y);
        ++count;
    }
    if (i == N)
    {
        sum += right_boundary(y);
        ++count;
    }

    if (count == 0)
    {
        throw std::runtime_error("boundary_value called for an interior node");
    }

    return sum / static_cast<double>(count);
}
Problem build_problem(int N)
{
    if (N < 2)
    {
        throw std::runtime_error("N must be at least 2");
    }

    Problem problem;
    problem.N = N;
    problem.m = N - 1;
    problem.h = 1.0 / static_cast<double>(N);
    problem.rhs.assign(problem.m * problem.m, 0.0);

    const int axis_dx[4] = {-1, 1, 0, 0};
    const int axis_dy[4] = {0, 0, -1, 1};
    const int diag_dx[4] = {-1, -1, 1, 1};
    const int diag_dy[4] = {-1, 1, -1, 1};

    for (int j = 1; j <= problem.m; ++j)
    {
        for (int i = 1; i <= problem.m; ++i)
        {
            double rhs = 12.0 * problem.h * problem.h;
            for (int k = 0; k < 4; ++k)
            {
                const int ni = i + axis_dx[k];
                const int nj = j + axis_dy[k];
                if (ni == 0 || ni == N || nj == 0 || nj == N)
                {
                    rhs += 4.0 * boundary_value(ni, nj, N);
                }
            }
            for (int k = 0; k < 4; ++k)
            {
                const int ni = i + diag_dx[k];
                const int nj = j + diag_dy[k];
                if (ni == 0 || ni == N || nj == 0 || nj == N)
                {
                    rhs += boundary_value(ni, nj, N);
                }
            }

            problem.rhs[index_of(i, j, problem.m)] = rhs;
        }
    }

    return problem;
}
double lambda_9_point_scaled(int p, int q, int N)
{
    const double cx = std::cos(kPi * static_cast<double>(p) / static_cast<double>(N));
    const double cy = std::cos(kPi * static_cast<double>(q) / static_cast<double>(N));
    return 20.0 - 8.0 * (cx + cy) - 4.0 * cx * cy;
}
double lambda_min_scaled(int N)
{
    return lambda_9_point_scaled(1, 1, N);
}
double lambda_max_scaled(int N)
{
    return lambda_9_point_scaled(N - 1, N - 1, N);
}
double directional_mu_min(int N)
{
    return 2.0 - 2.0 * std::cos(kPi / static_cast<double>(N));
}
double directional_mu_max(int N)
{
    return 2.0 + 2.0 * std::cos(kPi / static_cast<double>(N));
}
std::vector<double> make_chebyshev_parameters(double lambda_min,
                                              double lambda_max,
                                              int count)
{
    std::vector<double> roots;
    roots.reserve(count);

    const double center = 0.5 * (lambda_max + lambda_min);
    const double radius = 0.5 * (lambda_max - lambda_min);
    for (int s = 0; s < count; ++s)
    {
        const double theta =
            kPi * static_cast<double>(2 * s + 1) / static_cast<double>(2 * count);
        const double lambda = center + radius * std::cos(theta);
        roots.push_back(lambda);
    }

    std::vector<double> parameters;
    parameters.reserve(count);

    std::vector<char> used(count, 0);
    std::vector<double> log_weight(count, 0.0);
    int current = 0;
    for (int step = 0; step < count; ++step)
    {
        used[current] = 1;
        parameters.push_back(1.0 / roots[current]);

        for (int i = 0; i < count; ++i)
        {
            if (!used[i])
            {
                log_weight[i] += std::log(std::abs(roots[i] - roots[current]));
            }
        }

        double best_weight = -std::numeric_limits<double>::infinity();
        int best_index = -1;
        for (int i = 0; i < count; ++i)
        {
            if (!used[i] && log_weight[i] > best_weight)
            {
                best_weight = log_weight[i];
                best_index = i;
            }
        }

        current = best_index;
    }

    return parameters;
}
std::vector<double> make_adi_parameters(int N, int count)
{
    const double mu_min = directional_mu_min(N);
    const double mu_max = directional_mu_max(N);
    const double alpha = mu_min * (6.0 - mu_max);
    const double beta = mu_max * (6.0 - mu_min);
    const double scale = std::sqrt(alpha * beta);

    std::vector<double> parameters;
    parameters.reserve(count);

    for (int s = 0; s < count; ++s)
    {
        const double theta =
            kPi * static_cast<double>(2 * s + 1) / static_cast<double>(4 * count);
        const double tangent = std::tan(theta);
        parameters.push_back(scale / (tangent * tangent));
    }

    return parameters;
}
void apply_B(const Problem& problem,
             const std::vector<double>& u,
             std::vector<double>& out)
{
    const int m = problem.m;
    out.assign(u.size(), 0.0);

    for (int j = 1; j <= m; ++j)
    {
        for (int i = 1; i <= m; ++i)
        {
            const int id = index_of(i, j, m);
            double value = 20.0 * u[id];
            if (i > 1)
            {
                value -= 4.0 * u[index_of(i - 1, j, m)];
            }
            if (i < m)
            {
                value -= 4.0 * u[index_of(i + 1, j, m)];
            }
            if (j > 1)
            {
                value -= 4.0 * u[index_of(i, j - 1, m)];
            }
            if (j < m)
            {
                value -= 4.0 * u[index_of(i, j + 1, m)];
            }
            if (i > 1 && j > 1)
            {
                value -= u[index_of(i - 1, j - 1, m)];
            }
            if (i > 1 && j < m)
            {
                value -= u[index_of(i - 1, j + 1, m)];
            }
            if (i < m && j > 1)
            {
                value -= u[index_of(i + 1, j - 1, m)];
            }
            if (i < m && j < m)
            {
                value -= u[index_of(i + 1, j + 1, m)];
            }

            out[id] = value;
        }
    }
}
void apply_H(const Problem& problem,
             const std::vector<double>& u,
             std::vector<double>& out)
{
    const int m = problem.m;
    out.assign(u.size(), 0.0);

    for (int j = 1; j <= m; ++j)
    {
        for (int i = 1; i <= m; ++i)
        {
            const int id = index_of(i, j, m);
            double value = 12.0 * u[id];
            if (i > 1)
            {
                value -= 6.0 * u[index_of(i - 1, j, m)];
            }
            if (i < m)
            {
                value -= 6.0 * u[index_of(i + 1, j, m)];
            }
            out[id] = value;
        }
    }
}
void apply_V(const Problem& problem,
             const std::vector<double>& u,
             std::vector<double>& out)
{
    const int m = problem.m;
    out.assign(u.size(), 0.0);

    for (int j = 1; j <= m; ++j)
    {
        for (int i = 1; i <= m; ++i)
        {
            const int id = index_of(i, j, m);
            double value = 12.0 * u[id];
            if (j > 1)
            {
                value -= 6.0 * u[index_of(i, j - 1, m)];
            }
            if (j < m)
            {
                value -= 6.0 * u[index_of(i, j + 1, m)];
            }
            out[id] = value;
        }
    }
}
void apply_C(const Problem& problem,
             const std::vector<double>& u,
             std::vector<double>& out)
{
    const int m = problem.m;
    out.assign(u.size(), 0.0);

    for (int j = 1; j <= m; ++j)
    {
        for (int i = 1; i <= m; ++i)
        {
            const int id = index_of(i, j, m);
            double value = -4.0 * u[id];

            if (i > 1)
            {
                value += 2.0 * u[index_of(i - 1, j, m)];
            }
            if (i < m)
            {
                value += 2.0 * u[index_of(i + 1, j, m)];
            }
            if (j > 1)
            {
                value += 2.0 * u[index_of(i, j - 1, m)];
            }
            if (j < m)
            {
                value += 2.0 * u[index_of(i, j + 1, m)];
            }

            if (i > 1 && j > 1)
            {
                value -= u[index_of(i - 1, j - 1, m)];
            }
            if (i > 1 && j < m)
            {
                value -= u[index_of(i - 1, j + 1, m)];
            }
            if (i < m && j > 1)
            {
                value -= u[index_of(i + 1, j - 1, m)];
            }
            if (i < m && j < m)
            {
                value -= u[index_of(i + 1, j + 1, m)];
            }

            out[id] = value;
        }
    }
}
double max_abs(const std::vector<double>& values)
{
    double result = 0.0;
    for (double value : values)
    {
        result = std::max(result, std::abs(value));
    }
    return result;
}
double l2_norm(const std::vector<double>& values)
{
    long double sum = 0.0;
    for (double value : values)
    {
        sum += static_cast<long double>(value) * static_cast<long double>(value);
    }
    return std::sqrt(static_cast<double>(sum));
}
double compute_scaled_residual(const Problem& problem,
                               const std::vector<double>& u,
                               std::vector<double>& residual)
{
    apply_B(problem, u, residual);
    for (std::size_t i = 0; i < residual.size(); ++i)
    {
        residual[i] = problem.rhs[i] - residual[i];
    }
    return max_abs(residual);
}
bool all_finite(const std::vector<double>& values)
{
    for (double value : values)
    {
        if (!std::isfinite(value))
        {
            return false;
        }
    }
    return true;
}
void solve_tridiagonal_constant(int n,
                                double lower,
                                double diagonal,
                                double upper,
                                const std::vector<double>& rhs,
                                std::vector<double>& solution)
{
    if (n <= 0)
    {
        return;
    }
    std::vector<double> c_prime(n, 0.0);
    std::vector<double> d_prime(n, 0.0);

    double denom = diagonal;
    if (std::abs(denom) < 1e-14)
    {
        throw std::runtime_error("singular tridiagonal system");
    }

    c_prime[0] = (n > 1) ? upper / denom : 0.0;
    d_prime[0] = rhs[0] / denom;
    for (int i = 1; i < n; ++i)
    {
        denom = diagonal - lower * c_prime[i - 1];
        if (std::abs(denom) < 1e-14)
        {
            throw std::runtime_error("singular tridiagonal system");
        }

        c_prime[i] = (i < n - 1) ? upper / denom : 0.0;
        d_prime[i] = (rhs[i] - lower * d_prime[i - 1]) / denom;
    }
    solution.assign(n, 0.0);
    solution[n - 1] = d_prime[n - 1];
    for (int i = n - 2; i >= 0; --i)
    {
        solution[i] = d_prime[i] - c_prime[i] * solution[i + 1];
    }
}
void solve_x_lines(const Problem& problem,
                   double rho,
                   const std::vector<double>& rhs,
                   std::vector<double>& out)
{
    const int m = problem.m;
    out.assign(rhs.size(), 0.0);

    std::vector<double> line_rhs(m, 0.0);
    std::vector<double> line_solution;

    for (int j = 1; j <= m; ++j)
    {
        for (int i = 1; i <= m; ++i)
        {
            line_rhs[i - 1] = rhs[index_of(i, j, m)];
        }
        solve_tridiagonal_constant(m, -6.0, rho + 12.0, -6.0,
                                   line_rhs, line_solution);
        for (int i = 1; i <= m; ++i)
        {
            out[index_of(i, j, m)] = line_solution[i - 1];
        }
    }
}
void solve_y_lines(const Problem& problem,
                   double rho,
                   const std::vector<double>& rhs,
                   std::vector<double>& out)
{
    const int m = problem.m;
    out.assign(rhs.size(), 0.0);

    std::vector<double> line_rhs(m, 0.0);
    std::vector<double> line_solution;

    for (int i = 1; i <= m; ++i)
    {
        for (int j = 1; j <= m; ++j)
        {
            line_rhs[j - 1] = rhs[index_of(i, j, m)];
        }
        solve_tridiagonal_constant(m, -6.0, rho + 12.0, -6.0,
                                   line_rhs, line_solution);
        for (int j = 1; j <= m; ++j)
        {
            out[index_of(i, j, m)] = line_solution[j - 1];
        }
    }
}
void fill_common_result_data(const Problem& problem,
                             SolveResult& result,
                             const std::vector<double>& residual)
{
    result.N = problem.N;
    result.m = problem.m;
    result.unknowns = problem.m * problem.m;
    result.h = problem.h;
    result.lambda_min = lambda_min_scaled(problem.N);
    result.lambda_max = lambda_max_scaled(problem.N);
    result.scaled_residual_inf = max_abs(residual);
    result.residual_inf = result.scaled_residual_inf /
                          (6.0 * problem.h * problem.h);
    result.error_bound_l2 = l2_norm(residual) / result.lambda_min;
    result.converged = result.residual_inf <= kEps;
}
SolveResult solve_chebyshev(const Problem& problem)
{
    SolveResult result;
    result.method = Method::ChebyshevRichardson;

    const int parameter_count = std::max(8, problem.N);
    const std::vector<double> parameters = make_chebyshev_parameters(
        lambda_min_scaled(problem.N),
        lambda_max_scaled(problem.N),
        parameter_count
    );
    std::vector<double> u(problem.rhs.size(), 0.0);
    std::vector<double> residual;
    double residual_scaled = compute_scaled_residual(problem, u, residual);
    int iterations = 0;

    while (residual_scaled / (6.0 * problem.h * problem.h) > kEps &&
           iterations < kMaxIterations)
    {
        const double tau = parameters[iterations % parameters.size()];
        for (std::size_t i = 0; i < u.size(); ++i)
        {
            u[i] += tau * residual[i];
        }

        ++iterations;
        if (!all_finite(u))
        {
            throw std::runtime_error("Chebyshev iteration produced non-finite values");
        }
        residual_scaled = compute_scaled_residual(problem, u, residual);
    }

    result.iterations = iterations;
    result.parameter_count = parameter_count;
    result.u = std::move(u);
    fill_common_result_data(problem, result, residual);
    return result;
}
SolveResult solve_adi(const Problem& problem)
{
    SolveResult result;
    result.method = Method::AlternatingDirections;

    const int parameter_count =
        std::max(6, static_cast<int>(std::ceil(std::log2(problem.N))) + 4);
    const std::vector<double> parameters = make_adi_parameters(problem.N,
                                                               parameter_count);

    std::vector<double> u(problem.rhs.size(), 0.0);
    std::vector<double> half(u.size(), 0.0);
    std::vector<double> rhs_step(u.size(), 0.0);
    std::vector<double> Hu;
    std::vector<double> Vu;
    std::vector<double> Cu;
    std::vector<double> residual;

    double residual_scaled = compute_scaled_residual(problem, u, residual);
    int iterations = 0;

    while (residual_scaled / (6.0 * problem.h * problem.h) > kEps &&
           iterations < kMaxIterations)
    {
        const double rho = parameters[iterations % parameters.size()];
        apply_V(problem, u, Vu);
        apply_C(problem, u, Cu);
        for (std::size_t i = 0; i < u.size(); ++i)
        {
            rhs_step[i] = rho * u[i] - Vu[i] - Cu[i] + problem.rhs[i];
        }
        solve_x_lines(problem, rho, rhs_step, half);
        apply_H(problem, half, Hu);
        apply_C(problem, half, Cu);
        for (std::size_t i = 0; i < u.size(); ++i)
        {
            rhs_step[i] = rho * half[i] - Hu[i] - Cu[i] + problem.rhs[i];
        }
        solve_y_lines(problem, rho, rhs_step, u);

        ++iterations;
        if (!all_finite(u))
        {
            throw std::runtime_error("ADI iteration produced non-finite values");
        }
        residual_scaled = compute_scaled_residual(problem, u, residual);
    }

    result.iterations = iterations;
    result.parameter_count = parameter_count;
    result.u = std::move(u);
    fill_common_result_data(problem, result, residual);
    return result;
}
double grid_value(const SolveResult& result, int i, int j)
{
    if (i == 0 || i == result.N || j == 0 || j == result.N)
    {
        return boundary_value(i, j, result.N);
    }

    return result.u[index_of(i, j, result.m)];
}
void save_iterations_csv(const std::filesystem::path& path,
                         const std::vector<SolveResult>& results)
{
    std::ofstream out(path);
    out << "method,N,h,unknowns,iterations,parameter_count,"
        << "residual_inf,scaled_residual_inf,error_bound_l2,"
        << "lambda_min,lambda_max,converged\n";
    out << std::setprecision(16);

    for (const SolveResult& result : results)
    {
        out << method_name(result.method) << ","
            << result.N << ","
            << result.h << ","
            << result.unknowns << ","
            << result.iterations << ","
            << result.parameter_count << ","
            << result.residual_inf << ","
            << result.scaled_residual_inf << ","
            << result.error_bound_l2 << ","
            << result.lambda_min << ","
            << result.lambda_max << ","
            << (result.converged ? 1 : 0) << "\n";
    }
}
void save_profile_csv(const std::filesystem::path& path,
                      const SolveResult& chebyshev,
                      const SolveResult& adi)
{
    if (chebyshev.N != adi.N)
    {
        throw std::runtime_error("cannot compare profiles on different grids");
    }

    const int N = chebyshev.N;
    const int j = N / 2;
    const double y = static_cast<double>(j) / static_cast<double>(N);

    std::ofstream out(path);
    out << "x,y,chebyshev,adi,difference\n";
    out << std::setprecision(16);

    for (int i = 0; i <= N; ++i)
    {
        const double x = static_cast<double>(i) / static_cast<double>(N);
        const double uc = grid_value(chebyshev, i, j);
        const double ua = grid_value(adi, i, j);
        out << x << ","
            << y << ","
            << uc << ","
            << ua << ","
            << std::abs(uc - ua) << "\n";
    }
}
void save_surface_csv(const std::filesystem::path& path,
                      const SolveResult& result)
{
    std::ofstream out(path);
    out << "x,y,u\n";
    out << std::setprecision(16);

    for (int j = 0; j <= result.N; ++j)
    {
        const double y = static_cast<double>(j) / static_cast<double>(result.N);
        for (int i = 0; i <= result.N; ++i)
        {
            const double x = static_cast<double>(i) / static_cast<double>(result.N);
            out << x << "," << y << "," << grid_value(result, i, j) << "\n";
        }
    }
}
void save_summary(const std::filesystem::path& path,
                  const std::vector<SolveResult>& results)
{
    std::ofstream out(path);
    out << std::fixed << std::setprecision(8);
    out << "Task 4: Poisson Dirichlet problem in the unit square.\n";
    out << "Equation: u_xx + u_yy = -2.\n";
    out << "Nine-point Laplace stencil:\n";
    out << "(4 axis neighbours + diagonal neighbours - 20 u_ij) / (6 h^2).\n";
    out << "The algebraic system is written as B u = b, where\n";
    out << "B = 20 u_ij - 4 axis neighbours - diagonal neighbours.\n";
    out << "Stopping criterion: max |Delta_h u - f| <= 1e-6.\n";
    out << "At the inconsistent corner (0,0), the average of the two boundary"
        << " values is used in the discrete stencil.\n\n";

    out << "method\tN\tunknowns\titerations\tparameters\tresidual_inf\terror_bound_l2\n";
    for (const SolveResult& result : results)
    {
        out << method_name(result.method) << "\t"
            << result.N << "\t"
            << result.unknowns << "\t"
            << result.iterations << "\t"
            << result.parameter_count << "\t"
            << result.residual_inf << "\t"
            << result.error_bound_l2 << "\n";
    }
}

}

int main()
{
    try
    {
        const std::filesystem::path output_dir = "Sem6/Task4/results";
        std::filesystem::create_directories(output_dir);

        std::vector<SolveResult> results;
        results.reserve(2 * kGridSizes.size());

        std::cout << std::fixed << std::setprecision(8);
        std::cout << "N\tmethod\titerations\tresidual\n";

        for (int N : kGridSizes)
        {
            const Problem problem = build_problem(N);
            SolveResult chebyshev = solve_chebyshev(problem);
            std::cout << N << "\t"
                      << method_name(chebyshev.method) << "\t"
                      << chebyshev.iterations << "\t"
                      << chebyshev.residual_inf << "\n";
            SolveResult adi = solve_adi(problem);
            std::cout << N << "\t"
                      << method_name(adi.method) << "\t"
                      << adi.iterations << "\t"
                      << adi.residual_inf << "\n";
            if (N == kGridSizes.back())
            {
                save_profile_csv(output_dir / "profile_N256.csv", chebyshev, adi);
                save_surface_csv(output_dir / "surface_chebyshev_N256.csv",
                                 chebyshev);
                save_surface_csv(output_dir / "surface_adi_N256.csv", adi);
            }

            results.push_back(std::move(chebyshev));
            results.push_back(std::move(adi));
        }
        save_iterations_csv(output_dir / "iterations.csv", results);
        save_summary(output_dir / "summary.txt", results);

        std::cout << "Results written to " << output_dir << "\n";
    }
    catch (const std::exception& ex)
    {
        std::cerr << "Error: " << ex.what() << "\n";
        return 1;
    }

    return 0;
}
