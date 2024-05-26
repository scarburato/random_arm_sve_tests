#pragma once

#include <vector>
#include <iostream>

typedef void (*bench_fun)(float *data, size_t len);

struct Benchmark
{
    size_t len;
    bench_fun gen_data;
    bench_fun lib_f;
    std::vector<std::pair<std::string, bench_fun>> fs_to_test;

    void run(std::ostream &teletype = std::cout) const;
    void test_single(const float x, std::ostream &teletype = std::cout) const;
private:
    struct Result
    {
        double mean;
        double std_dev;
        double error_ratio;

        double mean_error;
    };

    Result run_single(bench_fun fun, float *data, float *output, const float *cmp) const;
};