#include <iostream>
#include <math.h>
#include <stddef.h>
#include <chrono>
#include <random>
#include <cstring>
#include <arm_sve.h>

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

const size_t LEN = 500000;

using namespace std::chrono_literals;

int main()
{
    float numeri[LEN];
    float numeri_bis[LEN];
    float log_lib[LEN];
    float log_apr[LEN];

    std::mt19937 gen{0xcafebabe};
    std::uniform_real_distribution<double> distrib_prob{1e-10, 1e4};

    for (long i = 0; i < ENTRIES; ++i)
    {
        float x = 1.0f + float(i) / ENTRIES;
        log2_mantissa_table[i] = log2f(x);
    }

    auto start_time = std::chrono::steady_clock::now();
    for (int i = 0; i < LEN; ++i)
        numeri[i] = distrib_prob(gen);

    memcpy(numeri_bis, numeri, LEN * sizeof(float));
    memcpy(log_lib, numeri, LEN * sizeof(float));
    memcpy(log_apr, numeri, LEN * sizeof(float));

    auto stop_time = std::chrono::steady_clock::now();
    std::cout << "Built array in " << (stop_time - start_time) / 1.0ms << "ms" << std::endl;

    start_time = std::chrono::steady_clock::now();
    for (int i = 0; i < LEN; ++i)
        log_lib[i] = logf(log_lib[i]);
    stop_time = std::chrono::steady_clock::now();
    std::cout << "logf from math.h took\t" << (stop_time - start_time) / 1.0ms << "ms" << std::endl;

    start_time = std::chrono::steady_clock::now();
    for (int i = 0; i < LEN; ++i)
        log_apr[i] = fast_log2(log_apr[i]) * 0.69314718f;
    stop_time = std::chrono::steady_clock::now();
    std::cout << "fast_log2 took\t" << (stop_time - start_time) / 1.0ms << "ms" << std::endl;

    start_time = std::chrono::steady_clock::now();
    fast_log_vec(numeri, LEN, 0.69314718f);
    stop_time = std::chrono::steady_clock::now();
    std::cout << "fast_log2_vec took\t" << (stop_time - start_time) / 1.0ms << "ms" << std::endl;

    start_time = std::chrono::steady_clock::now();
    for (int i = 0; i < LEN; ++i)
        numeri_bis[i] = log2_bis(numeri_bis[i]) * 0.69314718f;
    stop_time = std::chrono::steady_clock::now();
    std::cout << "log2_bis took\t" << (stop_time - start_time) / 1.0ms << "ms" << std::endl;

    int count = 0;
    int count_bis = 0;
    float sum = 0;
    float sum_bis = 0;
    for (int i = 0; i < LEN; ++i)
    {
        sum_bis += std::fabs(numeri_bis[i] - log_lib[i]);
        if (std::fabs(numeri_bis[i] - log_lib[i]) > 1e-2)
        {
            ++count_bis;
            std::cout << "log2_bis\t" << log_lib[i] << '\t' << numeri_bis[i] << '\t' << log_apr[i] << std::endl;
        }
        sum += std::fabs(numeri[i] - log_lib[i]);
        if (std::fabs(numeri[i] - log_lib[i]) > 1e-2)
        {
            ++count;
            std::cout << "log2\t" << log_lib[i] << '\t' << numeri[i] << '\t' << log_apr[i] << std::endl;
        }
    }

    std::cout << "log2\tlog2_bis\n";
    std::cout << count << '\t' << count_bis << '\n';
    std::cout << (sum / LEN) << '\t' << (sum_bis / LEN) << '\n';

    return 0;
}
