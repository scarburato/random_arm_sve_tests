#include <memory>
#include <map>
#include <chrono>
#include <numeric>
#include <valarray>

#include "Benchmark.h"


void Benchmark::run(std::ostream &teletype) const
{
    std::unique_ptr<float> base_data(new float[len]);
    std::unique_ptr<float> base_results(new float[len]);
    std::unique_ptr<float> run_results(new float[len]);
    std::map<std::string, Result> results;

    gen_data(base_data.get(), len);
    teletype << "built test data...\n";

    results["C/C++ library"] = run_single(lib_f, base_data.get(), base_results.get(), base_results.get());

    for (const auto &[name, fun] : fs_to_test)
    {
        teletype << "Benching " << name << "..." << std::endl;
        results[name] = run_single(fun, base_data.get(), run_results.get(), base_results.get());
    }

    teletype << std::format("{:<30} {:<12} {:<12} {:<12} {:<12}\n",
                            "Name", "Mean [ms]", "Std Dev [ms]", "Error Ratio", "Mean Error");

    for (const auto &[name, result] : results)
    {
        teletype << std::format("{:<30} {:12.5f} {:12.5f} {:12.5f} {:12.3e}\n",
                                name,
                                result.mean,
                                result.std_dev,
                                result.error_ratio,
                                result.mean_error);
    }
}

Benchmark::Result Benchmark::run_single(bench_fun fun, float *data, float *output, const float *cmp) const
{
    static constexpr size_t N_TESTS = 120;
    using namespace std::chrono_literals;

    Result res = {};
    std::vector<double> times;
    times.reserve(N_TESTS);

    for (size_t i = 0; i < N_TESTS; ++i)
    {
        std::copy(data, data + len, output);
        __builtin___clear_cache((char *) output, (char *) output + len);

        auto start_time = std::chrono::steady_clock::now();
        fun(output, len);
        auto stop_time = std::chrono::steady_clock::now();

        times.emplace_back((stop_time - start_time) / 1.0ms);
    }
    double err_sum = 0.0;
    size_t err_card = 0;
    for (size_t i = 0; i < len; ++i)
    {
        if (std::isnormal(cmp[i]) and std::isnormal(output[i]))
        {
            err_sum += std::abs(cmp[i] - output[i]);
            ++err_card;
        }
        else if (cmp[i] != output[i])
            res.error_ratio += 1;
    }

    res.mean_error = err_sum / err_card;
    res.error_ratio /= double(len);

    double sum = std::accumulate(times.begin(), times.end(), 0.0);
    res.mean = sum / times.size();

    double sq_sum = std::inner_product(times.begin(), times.end(), times.begin(), 0.0);
    res.std_dev = std::sqrt(sq_sum / times.size() - res.mean * res.mean);

    return res;
}
void Benchmark::test_single(const float x, std::ostream &teletype) const
{
    float r = x;
    lib_f(&r, 1);
    teletype << std::format("{:<30} {:12.8f}", "lib", r) << '\n';

    for (const auto &[name, fun] : fs_to_test)
    {
        float r = x;
        fun(&r, 1);
        teletype << std::format("{:<30} {:12.8f}", name, r) << std::endl;
    }
}
