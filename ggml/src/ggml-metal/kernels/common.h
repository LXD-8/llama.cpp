#pragma once

#include "ggml-metal-impl.h"

#include <metal_stdlib>

#ifdef GGML_METAL_HAS_TENSOR
#include <metal_tensor>

#include <MetalPerformancePrimitives/MetalPerformancePrimitives.h>
#endif

using namespace metal;

#define MAX(x, y) ((x) > (y) ? (x) : (y))
#define MIN(x, y) ((x) < (y) ? (x) : (y))
#define SWAP(x, y) { auto tmp = (x); (x) = (y); (y) = tmp; }

#define PAD2(x, n) (((x) + (n) - 1) & ~((n) - 1))

#define FOR_UNROLL(x) _Pragma("clang loop unroll(full)") for (x)

#ifdef GGML_METAL_HAS_TENSOR
// register fragments of the NAX kernels: a 16x32x16 simdgroup matmul2d on 16x16 fragments, each lane holding rows fm
// and fm + 8, columns fn .. fn + 3. the layout follows MLX steel/gemm/nax.h (MIT, Apple Inc.)
static inline short2 nax_coord(ushort lane) {
    const short qid = lane >> 2;
    const short fm  = (qid & 4) | ((lane >> 1) & 3);
    const short fn  = ((qid & 2) | (lane & 1))*4;
    return short2(fn, fm);
}

// C[16x32] += A[16x16] * B, with B given as two 16x16 fragments along N (transposed when TB)
template<typename CT, typename AT, typename BT, bool TB>
static inline void nax_mma(
        thread vec<CT, 8> & c0, thread vec<CT, 8> & c1,
        thread const vec<AT, 8> & a,
        thread const vec<BT, 8> & b0, thread const vec<BT, 8> & b1) {
    constexpr auto desc = mpp::tensor_ops::matmul2d_descriptor(16, 32, 16, false, TB, true,
            mpp::tensor_ops::matmul2d_descriptor::mode::multiply_accumulate);

    mpp::tensor_ops::matmul2d<desc, execution_simdgroup> op;

    auto ca = op.template get_left_input_cooperative_tensor <AT, BT, CT>();
    auto cb = op.template get_right_input_cooperative_tensor<AT, BT, CT>();
    auto cc = op.template get_destination_cooperative_tensor<remove_addrspace_t<decltype(ca)>, remove_addrspace_t<decltype(cb)>, CT>();

    FOR_UNROLL (short i = 0; i < 8; ++i) {
        ca[i]     = a[i];
        cb[i]     = b0[i];
        cb[8 + i] = b1[i];
        cc[i]     = c0[i];
        cc[8 + i] = c1[i];
    }

    op.run(ca, cb, cc);

    FOR_UNROLL (short i = 0; i < 8; ++i) {
        c0[i] = cc[i];
        c1[i] = cc[8 + i];
    }
}
#endif

#define N_SIMDWIDTH 32 // assuming SIMD group size is 32

// ref: https://developer.apple.com/metal/Metal-Shading-Language-Specification.pdf
//
// cmd:
//   .../usr/bin/metal -dM -E -c                             ggml/src/ggml-metal/kernels/<src>.metal
//   .../usr/bin/metal -dM -E -c -target air64-apple-ios14.0 ggml/src/ggml-metal/kernels/<src>.metal
//
#if __METAL_VERSION__ < 310 && defined(GGML_METAL_HAS_BF16)
#undef GGML_METAL_HAS_BF16
#endif

#if defined(GGML_METAL_HAS_BF16)
typedef matrix<bfloat, 4, 4> bfloat4x4;
typedef matrix<bfloat, 2, 4> bfloat2x4;
#endif

constexpr constant static float kvalues_iq4nl_f[16] = {
    -127.f, -104.f, -83.f, -65.f, -49.f, -35.f, -22.f, -10.f, 1.f, 13.f, 25.f, 38.f, 53.f, 69.f, 89.f, 113.f
};

constexpr constant static float kvalues_mxfp4_f[16] = {
    0, .5f, 1.f, 1.5f, 2.f, 3.f, 4.f, 6.f, -0, -.5f, -1.f, -1.5f, -2.f, -3.f, -4.f, -6.f
};

static inline int best_index_int8(int n, constant float * val, float x) {
    if (x <= val[0]) return 0;
    if (x >= val[n-1]) return n-1;
    int ml = 0, mu = n-1;
    while (mu-ml > 1) {
        int mav = (ml+mu)/2;
        if (x < val[mav]) mu = mav; else ml = mav;
    }
    return x - val[mu-1] < val[mu] - x ? mu-1 : mu;
}

// NVFP4 sub-block scale: unsigned E4M3 (bias 7, no infinities, 0x7F is NaN and decodes to 0 as on the CPU)
static inline float ue4m3_to_fp32(uint8_t x) {
    const uint e = (x >> 3) & 0xF;
    const uint m = x & 0x7;

    const float v = e == 0 ? (float) m * 0x1p-9f : as_type<float>(((e + 120) << 23) | (m << 20));

    return x == 0x7F ? 0.0f : v;
}

static inline float e8m0_to_fp32(uint8_t x) {
    uint32_t bits;

    if (x == 0) {
        bits = 0x00400000;
    } else {
        bits = (uint32_t) x << 23;
    }

    return as_type<float>(bits);
}

static inline float dot(float x, float y) {
    return x*y;
}

static inline float sum(float x) {
    return x;
}

static inline float sum(float4 x) {
    return x[0] + x[1] + x[2] + x[3];
}

enum ggml_sort_order {
    GGML_SORT_ORDER_ASC,
    GGML_SORT_ORDER_DESC,
};

constant float GELU_COEF_A     = 0.044715f;
constant float GELU_QUICK_COEF = -1.702f;
constant float SQRT_2_OVER_PI  = 0.79788456080286535587989211986876f;
constant float SQRT_2_INV      = 0.70710678118654752440084436210484f;

// based on Abramowitz and Stegun formula 7.1.26 or similar Hastings' approximation
// ref: https://www.johndcook.com/blog/python_erf/
constant float p_erf  = 0.3275911f;
constant float a1_erf = 0.254829592f;
constant float a2_erf = -0.284496736f;
constant float a3_erf = 1.421413741f;
constant float a4_erf = -1.453152027f;
constant float a5_erf = 1.061405429f;

template<typename T>
inline T erf_approx(T x) {
    T sign_x = sign(x);
    x = fabs(x);
    T t = 1.0f / (1.0f + p_erf * x);
    T y = 1.0f - (((((a5_erf * t + a4_erf) * t) + a3_erf) * t + a2_erf) * t + a1_erf) * t * exp(-x * x);
    return sign_x * y;
}

template<typename T> T elu_approx(T x);

template<> inline float elu_approx<float>(float x) {
    return (x > 0.f) ? x : (exp(x) - 1);
}

template<> inline float4 elu_approx<float4>(float4 x) {
    float4 res;

    res[0] = (x[0] > 0.0f) ? x[0] : (exp(x[0]) - 1.0f);
    res[1] = (x[1] > 0.0f) ? x[1] : (exp(x[1]) - 1.0f);
    res[2] = (x[2] > 0.0f) ? x[2] : (exp(x[2]) - 1.0f);
    res[3] = (x[3] > 0.0f) ? x[3] : (exp(x[3]) - 1.0f);

    return res;
}

// int8 NAX mat-mat: unnormalized Walsh-Hadamard transform of a group of 128 values held as one float4 per lane of a
// simdgroup (lane l holds values 4l..4l+3). applied to both operands it keeps their dot product (times 128) and spreads
// outliers over the group, so that the per-group int8 scale fits the bulk of the values
static inline float4 i8_wht128(float4 v, ushort lane) {
    v = float4(v[0] + v[1], v[0] - v[1], v[2] + v[3], v[2] - v[3]);
    v = float4(v[0] + v[2], v[1] + v[3], v[0] - v[2], v[1] - v[3]);
    for (ushort o = 1; o < 32; o <<= 1) {
        const float4 p = simd_shuffle_xor(v, o);
        v = (lane & o) ? p - v : v + p;
    }
    return v;
}
