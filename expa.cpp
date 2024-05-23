#include <iostream>
#include <arm_sve.h>
#include <format>

void stampa_v(svfloat32_t v, const uint32_t dim)
{
    float32_t u[dim];

    svbool_t pg = svptrue_b32();
    svst1(pg, u, v);

    std::cout << u[0] << ", ";
    std::cout << std::endl;
}

void stampa_v(svuint32_t v, const uint32_t dim)
{
    uint32_t u[dim];

    svbool_t pg = svptrue_b32();
    svst1(pg, u, v);

    std::cout << std::format("{:32b}", u[0]) << ", ";
    std::cout << std::endl;
}

svfloat32_t exp2(svbool_t pg, svfloat32_t p)
{
    svuint32_t index = svreinterpret_u32(p);
    stampa_v(index, svcntw());
    // low 6 bits I specify a coefficient
    index = svlsr_z(pg, index, 16);
    stampa_v(index, svcntw());

    return svexpa(index);
}

int main()
{
    float32_t x;
    while (std::cin >> x)
    {
        stampa_v(svdup_f32(x), svcntw());
        auto e = exp2(svptrue_b32(), svdup_f32(x));
        stampa_v(e, svcntw());
    }
    return 0;
}