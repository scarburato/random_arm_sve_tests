#include <iostream>
#include <math.h>
#include <stddef.h>
#include <chrono>
#include <random>
#include <cstring>
#include <arm_sve.h>
#include "Benchmark.h"


#define ENABLE_ZERO_APROX

constexpr size_t ENTRIES = 16 + 4096;

float exp_table[ENTRIES] = {0};

float myexp(float p)
{
    if (p <= -16)
        return 0.0f;

    float nf;
    float eps = modff(p, &nf);
    int index = (int) nf + 16;

    return exp_table[index] * (1 + eps + 0.5 * eps * eps + (1 / 6.0f) * eps * eps * eps);
}

void myexp_vec(float *values, uint32_t len)
{
    svfloat32_t c1 = svdup_f32(1.0f / 2);
    //svint32_t c2 = svdup_s32(sizeof(float));

    for (uint32_t i = 0; i < len; i += svcntw())
    {
        svbool_t pg = svwhilelt_b32(i, len);
        svfloat32_t v = svld1(pg, values + i);

#ifdef ENABLE_ZERO_APROX // @FIXME
        // exp(p) = 0 if p <= 1e-16
        // disabling this will cause a SEGFAULT for out-of-bound ns
        //svbool_t zero_mask = svcmpgt(pg, v, 16.0f);
        //pg = svnot_z(pg, zero_mask); // i.e. p st p > 1e-16
        svbool_t pg_tot = pg;
        pg = svcmple(pg, v, 16.0f);
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

#ifdef ENABLE_ZERO_APROX
        //v = svdup_f32_m(v, zero_mask, /*0.00000005f*/ 0.0f);
        v = svmax_x(pg_tot, v, 0.0f); // clamp negative values, faster that setting 0 manually

        // Store back
        svst1(pg_tot, values + i, v);
#else
        // Store back
        svst1(pg, values + i, v);
#endif
    }
}

void myexp_fexpa_vec(float *values, uint32_t len)
{
    svfloat32_t c1 = svdup_f32(1.0f / 2);
    //svint32_t c2 = svdup_s32(sizeof(float));

    for (uint32_t i = 0; i < len; i += svcntw())
    {
        svbool_t pg = svwhilelt_b32(i, len);
        svfloat32_t v = svld1(pg, values + i);

#ifdef ENABLE_ZERO_APROX // @FIXME
        // exp(p) = 0 if p <= 1e-16
        // disabling this will cause a SEGFAULT for out-of-bound ns
        //svbool_t zero_mask = svcmpgt(pg, v, 16.0f);
        //pg = svnot_z(pg, zero_mask); // i.e. p st p > 1e-16
        svbool_t pg_tot = pg;
        pg = svcmple(pg, v, 16.0f);
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

#ifdef ENABLE_ZERO_APROX
        //v = svdup_f32_m(v, zero_mask, /*0.00000005f*/ 0.0f);
        v = svmax_x(pg_tot, v, 0.0f); // clamp negative values, faster that setting 0 manually

        // Store back
        svst1(pg_tot, values + i, v);
#else
        // Store back
        svst1(pg, values + i, v);
#endif
    }
}

const size_t LEN = 2'500'000;

void create_data(float *data, size_t len)
{
    std::mt19937 gen{0xcafebabe};
    std::uniform_real_distribution<double> distrib_prob{-18, 5};

    for (size_t i = 0; i < len; ++i)
        data[i] = distrib_prob(gen);
    data[0] = -20;
}

void create_lib_data(float *data, size_t len)
{
    for (size_t i = 0; i < len; ++i)
        data[i] = expf(data[i]);
}

void myexp_autovec(float *data, size_t len)
{
    for (size_t i = 0; i < len; ++i)
        data[i] = myexp(data[i]);
}

void fast_log2_manvec(float *data, size_t len)
{
    myexp_vec(data, len);
}

int main()
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
            {"exp LUT (auto)", myexp_autovec},
            {"exp LUT (m. vec)", fast_log2_manvec}
        }
    };

    benchmark.run();

    return 0;
}
