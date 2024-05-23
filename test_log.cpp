#include <iostream>
#include <math.h>
#include <stddef.h>
#include <chrono>
#include <random>
#include <cstring>
#include <arm_sve.h>

#include "Benchmark.h"

inline float fast_log2(float val)
{
    int *const exp_ptr = (int *) (&val);
    int x = *exp_ptr;
    const int log_2 = ((x >> 23) & 255) - 128;
    x &= ~(255 << 23);
    x += 127 << 23;
    *exp_ptr = x;

    val = ((-1.0f / 3) * val + 2) * val - 2.0f / 3;   // (1)

    return (val + log_2);
}

void fast_log_vec(float *values, size_t len, float factor)
{
    // Cast
    int32_t *values_c = (int32_t *) values;

    svfloat32_t c1 = svdup_f32(-1.0f / 3);

    for (size_t i = 0; i < len; i += svcntw())
    {
        svbool_t pg = svwhilelt_b32(i, len);
        svint32_t vsrc = svld1(pg, values_c + i);

        // bitshift destro di 23 e maschera e sottrai 128
        svint32_t vlog2 = svasr_x(pg, vsrc, 23);      // log_2 = x >> 23
        vlog2 = svand_x(pg, vlog2, 255);     // vlog2 &= 255
        vlog2 = svsub_x(pg, vlog2, 128);     // vlog2 -= 128

        // convert to floating point
        svfloat32_t vlog2f = svcvt_f32_x(pg, vlog2);

        // seconda maschera e somma
        vsrc = svand_x(pg, vsrc, ~(255 << 23));    // x &= ~(255 << 23)
        vsrc = svadd_x(pg, vsrc, 127 << 23);       // x += 127 << 23

        // Now work with floats
        svfloat32_t vval = svreinterpret_f32(vsrc); //svld1(pg, values + i);

        svfloat32_t t1 = svmad_x(pg, vval, c1, 2.0f);      // t1 = val * (-1.0f/3) + 2
        vval = svmad_x(pg, vval, t1, -2.0f / 3);    // val = val*t1 - 2/3

        vval = svadd_x(pg, vval, vlog2f);   // val += log
        vval = svmul_x(pg, vval, factor);  // val *= factor

        // Store back
        svst1(pg, values + i, vval);
    }
}

// 11 bit precision
#define ENTRIES 0x800

float log2_mantissa_table[ENTRIES] = {0};

const uint32_t MASK_EXPONENT = 0b01111111100000000000000000000000;

const uint32_t MASK_MANTISSA = 0b00000000011111111111111111111111;

inline float log2_bis(float x)
{
    int exponent = ((*(uint32_t *) (&x) & MASK_EXPONENT) >> 23) - 127;
    //float mantissa = (float)(*(uint32_t*)(&x) & MASK_MANTISSA)/powf(2,23);
    uint32_t mantissa_r = (*(uint32_t *) (&x) & MASK_MANTISSA) >> (23 - 11);
    //std::cout << mantissa_r << '\t' << log2_mantissa_table[mantissa_r] << '\n';

    return exponent + log2_mantissa_table[mantissa_r];
}

void log2_bis(float *values, size_t len, float factor)
{
    // Cast
    int32_t *values_c = (int32_t *) values;


    for (size_t i = 0; i < len; i += svcntw())
    {
        svbool_t pg = svwhilelt_b32(i, len);
        svint32_t vsrc = svld1(pg, values_c + i);

        // @TODO
    }
}

const size_t LEN = 2'500'000;

void create_data(float *data, size_t len)
{
    std::mt19937 gen{0xcafebabe};
    std::uniform_real_distribution<float> distrib_prob{1e-10, 1e4};
    for (size_t i = 0; i < len; ++i)
        data[i] = distrib_prob(gen);
}

void create_lib_data(float *data, size_t len)
{
    for (size_t i = 0; i < len; ++i)
        data[i] = logf(data[i]);
}

void fast_log2_autovec(float *data, size_t len)
{
    for (size_t i = 0; i < len; ++i)
        data[i] = fast_log2(data[i]) * 0.69314718f;
}

void fast_log2_manvec(float *data, size_t len)
{
    fast_log_vec(data, len, 0.69314718f);
}

void log2_lut(float *data, size_t len)
{
    for (size_t i = 0; i < len; ++i)
        data[i] = log2_bis(data[i]) * 0.69314718f;
}

using namespace std::chrono_literals;

int main()
{
    // INIT TABLE
    for (long i = 0; i < ENTRIES; ++i)
    {
        float x = 1.0f + float(i) / ENTRIES;
        log2_mantissa_table[i] = log2f(x);
    }

    Benchmark benchmark = {
        .len = LEN,
        .gen_data = create_data,
        .lib_f = create_lib_data,
        .fs_to_test = {
            {"fast_log2 (auto)", fast_log2_autovec},
            {"fast_log2 (m. vec)", fast_log2_manvec},
            {"log2 LUT (auto)", log2_lut}
        }
    };

    benchmark.run();

    return 0;
}
