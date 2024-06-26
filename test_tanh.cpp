#include <iostream>
#include <math.h>
#include <stddef.h>
#include <chrono>
#include <random>
#include <cstring>
#include <arm_sve.h>
#include "Benchmark.h"


#define ENABLE_ONE_APROX

constexpr size_t ENTRIES = 16 + 4096;

float exp_table[ENTRIES] = {0};

float fasttanh(float x)
{
    if (x <= -16)
        return -1.0f;
    if (x >= +16)
        return 1.0f;

    float nf;
    float eps = modff(2 * x, &nf);
    int index = (int) nf + 16;

    float exp2x = exp_table[index] * (1 + eps + 0.5 * eps * eps + (1 / 6.0f) * eps * eps * eps);

    return (exp2x - 1) / (exp2x + 1);
}

float fasttanh2(float x)
{
    if (x <= -16)
        return -1.0f;
    if (x >= +16)
        return 1.0f;

    bool neg = x < 0;
    if(neg)
	    x *= -1;

    float nf;
    float eps = modff(2 * x, &nf);
    int index = (int) nf + 16;

    float exp2x = exp_table[index] * (1 + eps + 0.5 * eps * eps + (1 / 6.0f) * eps * eps * eps);

    return ((exp2x - 1) / (exp2x + 1))*(neg ? -1 : 1);
}

void fasttanh_vec(float *values, size_t len)
{
    svfloat32_t c1 = svdup_f32(1.0f / 2);
    //svint32_t c2 = svdup_s32(sizeof(float));

    for (size_t i = 0; i < len; i += svcntw())
    {
        svbool_t pg = svwhilelt_b32(i, len);
        svfloat32_t v = svld1(pg, values + i);
        v = svscale_x(pg, v, 1); // 2x

#ifdef ENABLE_ONE_APROX
        svbool_t pg_tot = pg;
        svbool_t pg_above_min = svcmple(pg, v, 10.0f);
        svbool_t pg_below_max = svcmpge(pg, v, -10.0f);

        pg = svand_z(pg, pg_above_min, pg_below_max);
#endif
        // Get int and frac part
        //svfloat32_t int_v = svrintm_x(pg, v);
        svfloat32_t int_v = svrinta_x(pg, v);
        svfloat32_t mod_v = svsub_x(pg, v, int_v);

        // get indices
        svint32_t indices = svcvt_s32_x(pg, int_v);
        indices = svadd_x(pg, indices, 16);

        // Load integer approximations
        svfloat32_t exp_int_part_v = svld1_gather_index(pg, exp_table, indices);

        // Compute fractional approxiamtion
        // exp(eps) ~= 1 + eps*(1 + eps*(0.5 + eps*1/6))
        svfloat32_t part_taylor_v = svmla_x(pg, c1, mod_v, 1.0f / 6.0f);
        part_taylor_v = svmad_x(pg, mod_v, part_taylor_v, 1.0f);
        part_taylor_v = svmad_x(pg, mod_v, part_taylor_v, 1.0f);

        // compute exp(eps)*exp(n)
        v = svmul_x(pg, part_taylor_v, exp_int_part_v);

        svfloat32_t numerator_v = svadd_x(pg, v, -1.0f);
        svfloat32_t denominator_v = svadd_x(pg, v, +1.0f);
        v = svdiv_x(pg, numerator_v, denominator_v);

#ifdef ENABLE_ONE_APROX
        pg = pg_tot;

        v = svdup_f32_m(v, svnot_z(pg, pg_above_min), +1.0f);
        v = svdup_f32_m(v, svnot_z(pg, pg_below_max), -1.0f);
#endif
        // Store back
        svst1(pg, values + i, v);
    }
}

void fasttanh2_vec(float *values, size_t len)
{
    svfloat32_t c1 = svdup_f32(1.0f / 2);
    //svint32_t c2 = svdup_s32(sizeof(float));

    for (size_t i = 0; i < len; i += svcntw())
    {
        svbool_t pg = svwhilelt_b32(i, len);
        svfloat32_t v = svld1(pg, values + i);
        v = svscale_x(pg, v, 1); // 2x
        
        // store which one is negative and then convert everyone to positive
        svbool_t pg_neg = svcmplt(pg, v, 0.0f);
        v = svabs_x(pg, v);

#ifdef ENABLE_ONE_APROX
        svbool_t pg_tot = pg;
        pg = svcmple(pg, v, 10.0f);
#endif
        // Get int and frac part
        //svfloat32_t int_v = svrintm_x(pg, v);
        svfloat32_t int_v = svrinta_x(pg, v);
        svfloat32_t mod_v = svsub_x(pg, v, int_v);

        // get indices
        svint32_t indices = svcvt_s32_x(pg, int_v);
        indices = svadd_x(pg, indices, 16);

        // Load integer approximations
        svfloat32_t exp_int_part_v = svld1_gather_index(pg, exp_table, indices);

        // Compute fractional approxiamtion
        // exp(eps) ~= 1 + eps*(1 + eps*(0.5 + eps*1/6))
        svfloat32_t part_taylor_v = svmla_x(pg, c1, mod_v, 1.0f / 6.0f);
        part_taylor_v = svmad_x(pg, mod_v, part_taylor_v, 1.0f);
        part_taylor_v = svmad_x(pg, mod_v, part_taylor_v, 1.0f);

        // compute exp(eps)*exp(n)
        v = svmul_x(pg, part_taylor_v, exp_int_part_v);

        svfloat32_t numerator_v = svadd_x(pg, v, -1.0f);
        svfloat32_t denominator_v = svadd_x(pg, v, +1.0f);
        v = svdiv_x(pg, numerator_v, denominator_v);

#ifdef ENABLE_ONE_APROX
        v = svdup_f32_m(v, svnot_z(pg, pg_tot), +1.0f);
        pg = pg_tot;
#endif
	// Flip negative values
	v = svneg_m(v, pg_neg, v);

        // Store back
        svst1(pg, values + i, v);
    }
}

const size_t LEN = 2'500'000;

void create_data(float *data, size_t len)
{
    std::mt19937 gen{0xcafebabe};
    std::uniform_real_distribution<double> distrib_prob{-3, +3};

    for (size_t i = 0; i < len; ++i)
        data[i] = distrib_prob(gen);

    data[0] = -20;
}

void create_lib_data(float *data, size_t len)
{
    for (int i = 0; i < len; ++i)
        data[i] = tanhf(data[i]);
}

void fast_tanh_autovec(float *__restrict__ data, size_t len)
{
    for (size_t i = 0; i < len; ++i)
        data[i] = fasttanh(data[i]);
}

void fast_tanh2_autovec(float *__restrict__ data, size_t len)
{
    for (size_t i = 0; i < len; ++i)
        data[i] = fasttanh2(data[i]);
}


void fast_tanh_manvec(float *data, size_t len)
{
    fasttanh_vec(data, len);
}

void fast_tanh2_manvec(float *data, size_t len)
{
    fasttanh2_vec(data, len);
}

int main(int argc, char **argv)
{
    // INIT LUT
    for (int i = 0; i < ENTRIES; ++i)
    {
        auto n = i - 16;
        exp_table[i] = expf(n);
    }

    Benchmark benchmark = {
        .len = LEN,
        .gen_data = create_data,
        .lib_f = create_lib_data,
        .fs_to_test = {
            {"tanh with expf LUT (auto)", fast_tanh_autovec},
            {"tanh with expf LUT (m. vec)", fast_tanh_manvec},
	    {"tanh with expf LUT pos (auto)", fast_tanh2_autovec},
	    {"tanh with expf LUT pos (m. vec)", fast_tanh2_manvec}
	}
    };

    if(argc > 1 and std::string("interactive") == argv[1])
    {
        float input;
        while(std::cin >> input)
            benchmark.test_single(input);
    }
    else
        benchmark.run();

    return 0;
}
