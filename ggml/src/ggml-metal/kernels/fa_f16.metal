#include "common.h"
#include "dequantize.h"
#include "fa_common.metal"

// TODO: this is quite ugly. in the future these types will be hardcoded in the kernel, but for now keep them as
//       template to be able to explore different combinations

#define FA_TYPES \
    half,   half4,     simdgroup_half8x8,  \
    half,   half4x4,   simdgroup_half8x8,  \
    half,   half4x4,   simdgroup_half8x8,  \
    float,             simdgroup_float8x8, \
    float,  float2,    simdgroup_float8x8, \
    float,  float4,    simdgroup_float8x8
    //half,   half4,     simdgroup_half8x8

#define FA_TYPES_BF \
    bfloat, bfloat4,   simdgroup_bfloat8x8, \
    bfloat, bfloat4x4, simdgroup_bfloat8x8, \
    bfloat, bfloat4x4, simdgroup_bfloat8x8, \
    float,             simdgroup_float8x8,  \
    float,  float2,    simdgroup_float8x8,  \
    half,   half4,     simdgroup_half8x8
    //float,  float4,    simdgroup_float8x8

#define FA_TYPES_F32 \
    half,   half4,     simdgroup_half8x8,  \
    float,  float4x4,  simdgroup_float8x8, \
    float,  float4x4,  simdgroup_float8x8, \
    float,             simdgroup_float8x8, \
    float,  float2,    simdgroup_float8x8, \
    float,  float4,    simdgroup_float8x8
    //half,   half4,     simdgroup_half8x8

typedef decltype(kernel_flash_attn_ext<FA_TYPES, half4x4, 1, dequantize_f16, half4x4, 1, dequantize_f16, 64, 64>) flash_attn_ext_t;

template [[host_name("kernel_flash_attn_ext_f16_dk32_dv32"  )]]  kernel flash_attn_ext_t kernel_flash_attn_ext<FA_TYPES,    half4x4,    1, dequantize_f16,  half4x4,    1, dequantize_f16,  32,  32>;
template [[host_name("kernel_flash_attn_ext_f16_dk40_dv40"  )]]  kernel flash_attn_ext_t kernel_flash_attn_ext<FA_TYPES,    half4x4,    1, dequantize_f16,  half4x4,    1, dequantize_f16,  40,  40>;
template [[host_name("kernel_flash_attn_ext_f16_dk48_dv48"  )]]  kernel flash_attn_ext_t kernel_flash_attn_ext<FA_TYPES,    half4x4,    1, dequantize_f16,  half4x4,    1, dequantize_f16,  48,  48>;
template [[host_name("kernel_flash_attn_ext_f16_dk64_dv64"  )]]  kernel flash_attn_ext_t kernel_flash_attn_ext<FA_TYPES,    half4x4,    1, dequantize_f16,  half4x4,    1, dequantize_f16,  64,  64>;
template [[host_name("kernel_flash_attn_ext_f16_dk72_dv72"  )]]  kernel flash_attn_ext_t kernel_flash_attn_ext<FA_TYPES,    half4x4,    1, dequantize_f16,  half4x4,    1, dequantize_f16,  72,  72>;
template [[host_name("kernel_flash_attn_ext_f16_dk80_dv80"  )]]  kernel flash_attn_ext_t kernel_flash_attn_ext<FA_TYPES,    half4x4,    1, dequantize_f16,  half4x4,    1, dequantize_f16,  80,  80>;
template [[host_name("kernel_flash_attn_ext_f16_dk96_dv96"  )]]  kernel flash_attn_ext_t kernel_flash_attn_ext<FA_TYPES,    half4x4,    1, dequantize_f16,  half4x4,    1, dequantize_f16,  96,  96>;
template [[host_name("kernel_flash_attn_ext_f16_dk96_dv64"  )]]  kernel flash_attn_ext_t kernel_flash_attn_ext<FA_TYPES,    half4x4,    1, dequantize_f16,  half4x4,    1, dequantize_f16,  96,  64>;
template [[host_name("kernel_flash_attn_ext_f16_dk112_dv112")]]  kernel flash_attn_ext_t kernel_flash_attn_ext<FA_TYPES,    half4x4,    1, dequantize_f16,  half4x4,    1, dequantize_f16,  112, 112>;
template [[host_name("kernel_flash_attn_ext_f16_dk128_dv128")]]  kernel flash_attn_ext_t kernel_flash_attn_ext<FA_TYPES,    half4x4,    1, dequantize_f16,  half4x4,    1, dequantize_f16,  128, 128>;
template [[host_name("kernel_flash_attn_ext_f16_dk128_dv96" )]]  kernel flash_attn_ext_t kernel_flash_attn_ext<FA_TYPES,    half4x4,    1, dequantize_f16,  half4x4,    1, dequantize_f16,  128, 96>;
template [[host_name("kernel_flash_attn_ext_f16_dk192_dv192")]]  kernel flash_attn_ext_t kernel_flash_attn_ext<FA_TYPES,    half4x4,    1, dequantize_f16,  half4x4,    1, dequantize_f16,  192, 192>;
template [[host_name("kernel_flash_attn_ext_f16_dk192_dv128")]]  kernel flash_attn_ext_t kernel_flash_attn_ext<FA_TYPES,    half4x4,    1, dequantize_f16,  half4x4,    1, dequantize_f16,  192, 128>;
template [[host_name("kernel_flash_attn_ext_f16_dk256_dv256")]]  kernel flash_attn_ext_t kernel_flash_attn_ext<FA_TYPES,    half4x4,    1, dequantize_f16,  half4x4,    1, dequantize_f16,  256, 256>;
template [[host_name("kernel_flash_attn_ext_f16_dk320_dv256")]]  kernel flash_attn_ext_t kernel_flash_attn_ext<FA_TYPES,    half4x4,    1, dequantize_f16,  half4x4,    1, dequantize_f16,  320, 256>;
template [[host_name("kernel_flash_attn_ext_f16_dk512_dv512")]]  kernel flash_attn_ext_t kernel_flash_attn_ext<FA_TYPES,    half4x4,    1, dequantize_f16,  half4x4,    1, dequantize_f16,  512, 512>;
template [[host_name("kernel_flash_attn_ext_f16_dk576_dv512")]]  kernel flash_attn_ext_t kernel_flash_attn_ext<FA_TYPES,    half4x4,    1, dequantize_f16,  half4x4,    1, dequantize_f16,  576, 512>;

#if defined(GGML_METAL_HAS_BF16)
template [[host_name("kernel_flash_attn_ext_bf16_dk32_dv32"  )]] kernel flash_attn_ext_t kernel_flash_attn_ext<FA_TYPES_BF, bfloat4x4,  1, dequantize_bf16, bfloat4x4,  1, dequantize_bf16, 32,  32>;
template [[host_name("kernel_flash_attn_ext_bf16_dk40_dv40"  )]] kernel flash_attn_ext_t kernel_flash_attn_ext<FA_TYPES_BF, bfloat4x4,  1, dequantize_bf16, bfloat4x4,  1, dequantize_bf16, 40,  40>;
template [[host_name("kernel_flash_attn_ext_bf16_dk48_dv48"  )]] kernel flash_attn_ext_t kernel_flash_attn_ext<FA_TYPES_BF, bfloat4x4,  1, dequantize_bf16, bfloat4x4,  1, dequantize_bf16, 48,  48>;
template [[host_name("kernel_flash_attn_ext_bf16_dk64_dv64"  )]] kernel flash_attn_ext_t kernel_flash_attn_ext<FA_TYPES_BF, bfloat4x4,  1, dequantize_bf16, bfloat4x4,  1, dequantize_bf16, 64,  64>;
template [[host_name("kernel_flash_attn_ext_bf16_dk72_dv72"  )]] kernel flash_attn_ext_t kernel_flash_attn_ext<FA_TYPES_BF, bfloat4x4,  1, dequantize_bf16, bfloat4x4,  1, dequantize_bf16, 72,  72>;
template [[host_name("kernel_flash_attn_ext_bf16_dk80_dv80"  )]] kernel flash_attn_ext_t kernel_flash_attn_ext<FA_TYPES_BF, bfloat4x4,  1, dequantize_bf16, bfloat4x4,  1, dequantize_bf16, 80,  80>;
template [[host_name("kernel_flash_attn_ext_bf16_dk96_dv96"  )]] kernel flash_attn_ext_t kernel_flash_attn_ext<FA_TYPES_BF, bfloat4x4,  1, dequantize_bf16, bfloat4x4,  1, dequantize_bf16, 96,  96>;
template [[host_name("kernel_flash_attn_ext_bf16_dk96_dv64"  )]] kernel flash_attn_ext_t kernel_flash_attn_ext<FA_TYPES_BF, bfloat4x4,  1, dequantize_bf16, bfloat4x4,  1, dequantize_bf16, 96,  64>;
template [[host_name("kernel_flash_attn_ext_bf16_dk112_dv112")]] kernel flash_attn_ext_t kernel_flash_attn_ext<FA_TYPES_BF, bfloat4x4,  1, dequantize_bf16, bfloat4x4,  1, dequantize_bf16, 112, 112>;
template [[host_name("kernel_flash_attn_ext_bf16_dk128_dv128")]] kernel flash_attn_ext_t kernel_flash_attn_ext<FA_TYPES_BF, bfloat4x4,  1, dequantize_bf16, bfloat4x4,  1, dequantize_bf16, 128, 128>;
template [[host_name("kernel_flash_attn_ext_bf16_dk128_dv96" )]] kernel flash_attn_ext_t kernel_flash_attn_ext<FA_TYPES_BF, bfloat4x4,  1, dequantize_bf16, bfloat4x4,  1, dequantize_bf16, 128, 96>;
template [[host_name("kernel_flash_attn_ext_bf16_dk192_dv192")]] kernel flash_attn_ext_t kernel_flash_attn_ext<FA_TYPES_BF, bfloat4x4,  1, dequantize_bf16, bfloat4x4,  1, dequantize_bf16, 192, 192>;
template [[host_name("kernel_flash_attn_ext_bf16_dk192_dv128")]] kernel flash_attn_ext_t kernel_flash_attn_ext<FA_TYPES_BF, bfloat4x4,  1, dequantize_bf16, bfloat4x4,  1, dequantize_bf16, 192, 128>;
template [[host_name("kernel_flash_attn_ext_bf16_dk256_dv256")]] kernel flash_attn_ext_t kernel_flash_attn_ext<FA_TYPES_BF, bfloat4x4,  1, dequantize_bf16, bfloat4x4,  1, dequantize_bf16, 256, 256>;
template [[host_name("kernel_flash_attn_ext_bf16_dk320_dv256")]] kernel flash_attn_ext_t kernel_flash_attn_ext<FA_TYPES_BF, bfloat4x4,  1, dequantize_bf16, bfloat4x4,  1, dequantize_bf16, 320, 256>;
template [[host_name("kernel_flash_attn_ext_bf16_dk512_dv512")]] kernel flash_attn_ext_t kernel_flash_attn_ext<FA_TYPES_BF, bfloat4x4,  1, dequantize_bf16, bfloat4x4,  1, dequantize_bf16, 512, 512>;
template [[host_name("kernel_flash_attn_ext_bf16_dk576_dv512")]] kernel flash_attn_ext_t kernel_flash_attn_ext<FA_TYPES_BF, bfloat4x4,  1, dequantize_bf16, bfloat4x4,  1, dequantize_bf16, 576, 512>;
#endif

#undef FA_TYPES
#undef FA_TYPES_BF
#undef FA_TYPES_F32

#ifdef GGML_METAL_HAS_TENSOR

constant bool FC_flash_attn_ext_tensor_has_mask  [[function_constant(FC_FLASH_ATTN_EXT_TENSOR + 0)]];
constant bool FC_flash_attn_ext_tensor_has_sinks [[function_constant(FC_FLASH_ATTN_EXT_TENSOR + 1)]];
constant bool FC_flash_attn_ext_tensor_has_bias  [[function_constant(FC_FLASH_ATTN_EXT_TENSOR + 2)]];
constant bool FC_flash_attn_ext_tensor_has_scap  [[function_constant(FC_FLASH_ATTN_EXT_TENSOR + 3)]];

// ref: https://arxiv.org/pdf/2307.08691.pdf
template<
    short DK,                                   // K head size
    short DV,                                   // V head size
    short Q   = OP_FLASH_ATTN_EXT_TENSOR_NQPSG, // queries per threadgroup
    short C   = OP_FLASH_ATTN_EXT_TENSOR_NCPSG, // cache items per threadgroup
    short NSG = OP_FLASH_ATTN_EXT_TENSOR_NSG>   // number of simd groups
kernel void kernel_flash_attn_ext_tensor(
        constant ggml_metal_kargs_flash_attn_ext & args,
        device const char * q,
        device const char * k,
        device const char * v,
        device const char * mask,
        device const char * sinks,
        device const char * blk,
        device       char * dst,
        threadgroup  char * shmem [[threadgroup(0)]],
        uint3  tgpig [[threadgroup_position_in_grid]],
        ushort tiisg [[thread_index_in_simdgroup]],
        ushort sgitg [[simdgroup_index_in_threadgroup]]) {
    constexpr short NW = N_SIMDWIDTH;
    constexpr short NT = NW*NSG;
    constexpr short NQ = Q/NSG;
    constexpr short NC = C/NW; // columns per thread

    static_assert(DK % 4 == 0,  "DK must be divisible by 4");
    static_assert(Q % NSG == 0, "Q must be divisible by NSG");
    static_assert(C % NW == 0,  "C must be divisible by NW");

    const int iq3 = tgpig[2];
    const int iq2 = tgpig[1];
    const int iq1 = tgpig[0]*Q;

    const short tiitg = sgitg*NW + tiisg;

    threadgroup half  * sq = (threadgroup half  *) shmem;       // [Q, DK] queries
    threadgroup float * ss = (threadgroup float *) (sq + Q*DK); // [Q, C]  scores
    threadgroup half  * sp = (threadgroup half  *) (ss + Q*C);  // [Q, C]  probabilities
    threadgroup float * sr = (threadgroup float *) (sp + Q*C);  // [Q]     per-row scale of O
    threadgroup int   * sf = (threadgroup int   *) (sr + Q);    // [1]     last iteration (ic0 + 1) that rescaled O

    q += iq1*args.nb01 + iq2*args.nb02 + iq3*args.nb03;

    {
        const int ikv2 = iq2/(args.ne02/args.ne_12_2);
        const int ikv3 = iq3/(args.ne03/args.ne_12_3);

        k += ikv2*args.nb12 + ikv3*args.nb13;
        v += ikv2*args.nb22 + ikv3*args.nb23;
    }

    // with softcap the scale is small (scale/softcap), so it is applied to the scores to keep the precision of Q
    const float qscale = FC_flash_attn_ext_tensor_has_scap ? 1.0f : args.scale;

    // load the queries, with the scale folded in
    for (int i = tiitg; i < Q*DK/4; i += NT) {
        const int j = i/(DK/4);

        float4 q4 = 0.0f;
        if (iq1 + j < args.ne01) {
            q4 = ((device const float4 *) (q + j*args.nb01))[i%(DK/4)];
        }

        ((threadgroup half4 *) sq)[i] = (half4) (q4*qscale);
    }

    device const half * pm[NQ];

    FOR_UNROLL (short jj = 0; jj < NQ; ++jj) {
        const short j = jj*NSG + sgitg;

        pm[jj] = (device const half *) (mask + (iq1 + j)*args.nb31 + (iq2%args.ne32)*args.nb32 + (iq3%args.ne33)*args.nb33);
    }

    {
        const int nblk1 = (args.ne01 + Q - 1)/Q;
        const int nblk0 = (args.ne11 + C - 1)/C;

        blk += (((iq3%args.ne33)*args.ne32 + (iq2%args.ne32))*nblk1 + iq1/Q)*nblk0;
    }

    float M[NQ];
    float S[NQ];

    FOR_UNROLL (short jj = 0; jj < NQ; ++jj) {
        M[jj] = -FLT_MAX/2;
        S[jj] = 0.0f;
    }

    float slope = 1.0f;

    // ALiBi
    if (FC_flash_attn_ext_tensor_has_bias) {
        const short h = iq2;

        const float base = h < args.n_head_log2 ? args.m0 : args.m1;
        const short exph = h < args.n_head_log2 ? h + 1 : 2*(h - args.n_head_log2) + 1;

        slope = pow(base, exph);
    }

    const int sk = args.ns10;
    const int sv = args.ns20;

    auto tq = tensor(sq, dextents<int32_t, 2>(DK, Q));
    auto ts = tensor(ss, dextents<int32_t, 2>(C,  Q));
    auto tp = tensor(sp, dextents<int32_t, 2>(C,  Q));

    mpp::tensor_ops::matmul2d<
        mpp::tensor_ops::matmul2d_descriptor(Q, C, DK, false, true, false, mpp::tensor_ops::matmul2d_descriptor::mode::multiply),
        execution_simdgroups<NSG>> mm_qk;

    mpp::tensor_ops::matmul2d<
        mpp::tensor_ops::matmul2d_descriptor(Q, DV, C, false, false, false, mpp::tensor_ops::matmul2d_descriptor::mode::multiply_accumulate),
        execution_simdgroups<NSG>> mm_pv;

    auto tv0 = tensor((device half *) v, dextents<int32_t, 2>(DV, C), array<int, 2>({1, sv}));

    // the O matrix from the paper
    auto co = mm_pv.template get_destination_cooperative_tensor<decltype(tp), decltype(tv0), float>();

    FOR_UNROLL (short i = 0; i < co.get_capacity(); ++i) {
        if (co.is_valid_element(i)) {
            co[i] = 0.0f;
        }
    }

    if (tiitg == 0) {
        sf[0] = 0;
    }

    threadgroup_barrier(mem_flags::mem_threadgroup);

    // the host guarantees ne11 % C == 0
    for (int ic0 = 0, ic = 0; ic < args.ne11; ++ic0, ic += C) {
        char blk_cur = 1;

        if (FC_flash_attn_ext_tensor_has_mask) {
            blk_cur = blk[ic0];

            if (blk_cur == 0) {
                continue;
            }
        }

        // Q*K^T
        {
            auto tk = tensor((device half *) (k + (uint64_t) ic*args.nb11), dextents<int32_t, 2>(DK, C), array<int, 2>({1, sk}));

            mm_qk.run(tq, tk, ts);
        }

        threadgroup_barrier(mem_flags::mem_threadgroup);

        // online softmax
        FOR_UNROLL (short jj = 0; jj < NQ; ++jj) {
            const short j = jj*NSG + sgitg;

            float s[NC];

            FOR_UNROLL (short ii = 0; ii < NC; ++ii) {
                s[ii] = ss[j*C + ii*NW + tiisg];
            }

            if (FC_flash_attn_ext_tensor_has_scap) {
                FOR_UNROLL (short ii = 0; ii < NC; ++ii) {
                    s[ii] = args.logit_softcap*precise::tanh(s[ii]*args.scale);
                }
            }

            if (FC_flash_attn_ext_tensor_has_mask && blk_cur != 2 && iq1 + j < args.ne31) {
                FOR_UNROLL (short ii = 0; ii < NC; ++ii) {
                    s[ii] += slope*(float) pm[jj][ic + ii*NW + tiisg];
                }
            }

            float m = M[jj];

            FOR_UNROLL (short ii = 0; ii < NC; ++ii) {
                m = max(m, s[ii]);
            }

            m = simd_max(m);

            // lazy rescaling: move the running max only when it grows by more than 8 (e^8 fits in half)
            float ms = 1.0f;

            if (m > M[jj] + 8.0f) {
                ms    = exp(M[jj] - m);
                M[jj] = m;

                if (tiisg == 0) {
                    sf[0] = ic0 + 1;
                }
            }

            float sum = 0.0f;

            FOR_UNROLL (short ii = 0; ii < NC; ++ii) {
                // the sum uses the same rounded values as P*V
                const half p = (half) exp(s[ii] - M[jj]);

                sp[j*C + ii*NW + tiisg] = p;

                sum += (float) p;
            }

            S[jj] = S[jj]*ms + simd_sum(sum);

            if (tiisg == 0) {
                sr[j] = ms;
            }
        }

        threadgroup_barrier(mem_flags::mem_threadgroup);

        // O = diag(ms)*O + P*V
        if (sf[0] == ic0 + 1) {
            FOR_UNROLL (short i = 0; i < co.get_capacity(); ++i) {
                if (co.is_valid_element(i)) {
                    co[i] *= sr[co.get_multidimensional_index(i)[1]];
                }
            }
        }

        {
            auto tv = tensor((device half *) (v + (uint64_t) ic*args.nb21), dextents<int32_t, 2>(DV, C), array<int, 2>({1, sv}));

            mm_pv.run(tp, tv, co);
        }

        threadgroup_barrier(mem_flags::mem_threadgroup);
    }

    FOR_UNROLL (short jj = 0; jj < NQ; ++jj) {
        const short j = jj*NSG + sgitg;

        // the sink only adds to the denominator - its rescale of O is folded into the final scale
        float ms = 1.0f;

        if (FC_flash_attn_ext_tensor_has_sinks) {
            const float s = ((device const float *) sinks)[iq2];
            const float m = max(M[jj], s);

            ms = exp(M[jj] - m);

            S[jj] = S[jj]*ms + exp(s - m);
        }

        if (tiisg == 0) {
            sr[j] = S[jj] == 0.0f ? 0.0f : ms/S[jj];
        }
    }

    threadgroup_barrier(mem_flags::mem_threadgroup);

    FOR_UNROLL (short i = 0; i < co.get_capacity(); ++i) {
        if (co.is_valid_element(i)) {
            co[i] *= sr[co.get_multidimensional_index(i)[1]];
        }
    }

    // store to global memory - rows past ne01 are clipped by the tensor extents
    device float * pdst = (device float *) dst + ((uint64_t) iq3*args.ne2*args.ne1 + iq2 + (uint64_t) iq1*args.ne1)*DV;

    auto td = tensor(pdst, dextents<int32_t, 2>(DV, args.ne01 - iq1), array<int, 2>({1, args.ne1*DV}));

    co.store(td);
}

typedef decltype(kernel_flash_attn_ext_tensor<64, 64>) flash_attn_ext_tensor_t;

template [[host_name("kernel_flash_attn_ext_tensor_f16_dk64_dv64"  )]] kernel flash_attn_ext_tensor_t kernel_flash_attn_ext_tensor<64,  64>;
template [[host_name("kernel_flash_attn_ext_tensor_f16_dk128_dv128")]] kernel flash_attn_ext_tensor_t kernel_flash_attn_ext_tensor<128, 128>;
template [[host_name("kernel_flash_attn_ext_tensor_f16_dk192_dv128")]] kernel flash_attn_ext_tensor_t kernel_flash_attn_ext_tensor<192, 128>;
template [[host_name("kernel_flash_attn_ext_tensor_f16_dk256_dv256")]] kernel flash_attn_ext_tensor_t kernel_flash_attn_ext_tensor<256, 256>;
template [[host_name("kernel_flash_attn_ext_tensor_f16_dk512_dv512")]] kernel flash_attn_ext_tensor_t kernel_flash_attn_ext_tensor<512, 512, OP_FLASH_ATTN_EXT_TENSOR_NQPSG_LARGE>;
template [[host_name("kernel_flash_attn_ext_tensor_f16_dk576_dv512")]] kernel flash_attn_ext_tensor_t kernel_flash_attn_ext_tensor<576, 512, OP_FLASH_ATTN_EXT_TENSOR_NQPSG_LARGE>;

constant bool FC_flash_attn_ext_nax_has_mask  [[function_constant(FC_FLASH_ATTN_EXT_NAX + 0)]];
constant bool FC_flash_attn_ext_nax_has_sinks [[function_constant(FC_FLASH_ATTN_EXT_NAX + 1)]];

// 16x16 register fragment of a 16x32x16 simdgroup matmul: each lane holds rows fm and fm + 8, columns fn .. fn + 3
// the layout follows MLX steel/gemm/nax.h (MIT, Apple Inc.)
static inline short2 fa_nax_coord(ushort lane) {
    const short qid = lane >> 2;
    const short fm  = (qid & 4) | ((lane >> 1) & 3);
    const short fn  = ((qid & 2) | (lane & 1))*4;
    return short2(fn, fm);
}

// C[16x32] += A[16x16] * B, with B given as two 16x16 fragments along N (transposed when TB)
template<typename CT, typename AT, typename BT, bool TB>
static inline void fa_nax_mma(
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

// flash attention with the scores, the probabilities and O kept in registers
// WM simdgroups split the queries (16 each), WN simdgroups split the head dim; the WN partial Q*K^T scores are summed
// through threadgroup memory in a fixed order, then every simdgroup runs the same softmax on its own copy
// the structure follows MLX steel attention_nax_dsplit (MIT, Apple Inc.)
// no ALiBi or softcap - the host uses kernel_flash_attn_ext_tensor for those
template<
    short DK,
    short DV,
    short WN,
    short BQ = OP_FLASH_ATTN_EXT_NAX_NQPSG,
    short BK = OP_FLASH_ATTN_EXT_NAX_NCPSG>
kernel void kernel_flash_attn_ext_nax(
        constant ggml_metal_kargs_flash_attn_ext & args,
        device const char * q,
        device const char * k,
        device const char * v,
        device const char * mask,
        device const char * sinks,
        device const char * blk,
        device       char * dst,
        threadgroup  char * shmem [[threadgroup(0)]],
        uint3  tgpig [[threadgroup_position_in_grid]],
        ushort tiisg [[thread_index_in_simdgroup]],
        ushort sgitg [[simdgroup_index_in_threadgroup]]) {
    constexpr short kU  = 16;
    constexpr short TK  = BK/kU;      // KV fragments per block
    constexpr short TDh = DK/kU/WN;   // head-dim fragments per simdgroup
    constexpr short BDh = DK/WN;      // head-dim columns per simdgroup

    static_assert(DK == DV,     "DK must equal DV");
    static_assert(TK % 2 == 0,  "scores are computed in pairs of fragments");
    static_assert(TDh % 2 == 0, "O is accumulated in pairs of fragments");

    const int iq3 = tgpig[2];
    const int iq2 = tgpig[1];
    const int iq1 = tgpig[0]*BQ;

    const short row_group = sgitg/WN;
    const short d_group   = sgitg%WN;

    const short2 sc = fa_nax_coord(tiisg);
    const short fn = sc.x;
    const short fm = sc.y;

    // first query row held by this lane - the second one is r0 + 8
    const int r0 = iq1 + row_group*kU + fm;

    // queries, with the softmax scale and log2(e) folded in
    vec<half, 8> qf[TDh];
    {
        const float qs = args.scale*M_LOG2E_F;

        device const char * q0 = q + iq2*args.nb02 + iq3*args.nb03;

        FOR_UNROLL (short i = 0; i < 2; ++i) {
            const int r = r0 + 8*i;

            FOR_UNROLL (short id = 0; id < TDh; ++id) {
                float4 x = 0.0f;
                if (r < args.ne01) {
                    x = *((device const float4 *) (q0 + r*args.nb01) + (d_group*BDh + id*kU + fn)/4);
                }

                const half4 h = (half4) (x*qs);

                FOR_UNROLL (short j = 0; j < 4; ++j) {
                    qf[id][4*i + j] = h[j];
                }
            }
        }
    }

    {
        const int ikv2 = iq2/(args.ne02/args.ne_12_2);
        const int ikv3 = iq3/(args.ne03/args.ne_12_3);

        k += ikv2*args.nb12 + ikv3*args.nb13;
        v += ikv2*args.nb22 + ikv3*args.nb23;
    }

    const int sk = args.ns10;
    const int sv = args.ns20;

    device const half * pk = (device const half *) k + fm*sk + d_group*BDh + fn;
    device const half * pv = (device const half *) v + fm*sv + d_group*BDh + fn;

    device const half * pm[2];
    bool row_ok[2];

    // the mask has ne31 query rows - rows past that fall back to row 0 and skip the mask below
    FOR_UNROLL (short i = 0; i < 2; ++i) {
        const int r = r0 + 8*i;

        row_ok[i] = r < args.ne31;
        pm[i] = (device const half *) (mask + (row_ok[i] ? r : 0)*args.nb31 + (iq2%args.ne32)*args.nb32 + (iq3%args.ne33)*args.nb33) + fn;
    }

    if (FC_flash_attn_ext_nax_has_mask) {
        const int nblk1 = (args.ne01 + BQ - 1)/BQ;
        const int nblk0 = (args.ne11 + BK - 1)/BK;

        blk += (((iq3%args.ne33)*args.ne32 + (iq2%args.ne32))*nblk1 + iq1/BQ)*nblk0;
    }

    vec<float, 8> of[TDh];
    FOR_UNROLL (short id = 0; id < TDh; ++id) {
        of[id] = 0.0f;
    }

    float M[2] = { -FLT_MAX/2, -FLT_MAX/2 };
    float S[2] = { 0.0f, 0.0f };

    // [WM][WN][TK*8*32] partial scores, in the per-lane fragment layout
    threadgroup float * sx     = (threadgroup float *) shmem;
    threadgroup float * sx_own = sx + (row_group*WN + d_group)*(TK*8*32) + tiisg*(TK*8);
    threadgroup float * sx_row = sx + (row_group*WN          )*(TK*8*32) + tiisg*(TK*8);

    // the host guarantees ne11 % BK == 0
    const int nblk_k = args.ne11/BK;

    // prefetch the block flags so that the device load latency stays off the critical path
    char blk_next = 2;
    if (FC_flash_attn_ext_nax_has_mask) {
        blk_next = blk[0];
    }

    for (int ic0 = 0, ic = 0; ic < args.ne11; ++ic0, ic += BK) {
        char blk_cur = blk_next;

        if (FC_flash_attn_ext_nax_has_mask) {
            blk_next = ic0 + 1 < nblk_k ? blk[ic0 + 1] : 2;

            if (blk_cur == 0) {
                continue;
            }
        }

        vec<float, 8> sf[TK];
        FOR_UNROLL (short ik = 0; ik < TK; ++ik) {
            sf[ik] = 0.0f;
        }

        // partial S = Q*K^T over this simdgroup's slice of the head dim
        FOR_UNROLL (short ik = 0; ik < TK; ik += 2) {
            FOR_UNROLL (short id = 0; id < TDh; ++id) {
                vec<half, 8> k0;
                vec<half, 8> k1;

                device const half * pk0 = pk + (ic + ik*kU)*sk + id*kU;
                device const half * pk1 = pk0 + kU*sk;

                FOR_UNROLL (short i = 0; i < 2; ++i) {
                    const half4 a = *((device const half4 *) (pk0 + 8*i*sk));
                    const half4 b = *((device const half4 *) (pk1 + 8*i*sk));

                    FOR_UNROLL (short j = 0; j < 4; ++j) {
                        k0[4*i + j] = a[j];
                        k1[4*i + j] = b[j];
                    }
                }

                fa_nax_mma<float, half, half, true>(sf[ik], sf[ik + 1], qf[id], k0, k1);
            }
        }

        if (WN > 1) {
            // sum the WN partial scores in a fixed order
            FOR_UNROLL (short ik = 0; ik < TK; ++ik) {
                FOR_UNROLL (short i = 0; i < 8; ++i) {
                    sx_own[ik*8 + i] = sf[ik][i];
                }
            }

            threadgroup_barrier(mem_flags::mem_threadgroup);

            FOR_UNROLL (short ik = 0; ik < TK; ++ik) {
                FOR_UNROLL (short i = 0; i < 8; ++i) {
                    float s = sx_row[ik*8 + i];
                    FOR_UNROLL (short g = 1; g < WN; ++g) {
                        s += sx_row[g*(TK*8*32) + ik*8 + i];
                    }
                    sf[ik][i] = s;
                }
            }

            threadgroup_barrier(mem_flags::mem_threadgroup);
        }

        if (FC_flash_attn_ext_nax_has_mask && blk_cur != 2) {
            FOR_UNROLL (short i = 0; i < 2; ++i) {
                if (row_ok[i]) {
                    FOR_UNROLL (short ik = 0; ik < TK; ++ik) {
                        device const half * pmi = pm[i] + ic + ik*kU;

                        FOR_UNROLL (short j = 0; j < 4; ++j) {
                            sf[ik][4*i + j] += M_LOG2E_F*(float) pmi[j];
                        }
                    }
                }
            }
        }

        // online softmax, base 2 - the lanes that share a row differ in bits 0 and 3
        FOR_UNROLL (short i = 0; i < 2; ++i) {
            float m = M[i];

            FOR_UNROLL (short ik = 0; ik < TK; ++ik) {
                m = max(m, max(max(sf[ik][4*i + 0], sf[ik][4*i + 1]), max(sf[ik][4*i + 2], sf[ik][4*i + 3])));
            }

            m = max(m, simd_shuffle_xor(m, 1));
            m = max(m, simd_shuffle_xor(m, 8));

            const float ms = fast::exp2(M[i] - m);
            M[i] = m;

            float sum = 0.0f;

            FOR_UNROLL (short ik = 0; ik < TK; ++ik) {
                FOR_UNROLL (short j = 0; j < 4; ++j) {
                    const float p = fast::exp2(sf[ik][4*i + j] - m);
                    sf[ik][4*i + j] = p;
                    sum += p;
                }
            }

            sum += simd_shuffle_xor(sum, 1);
            sum += simd_shuffle_xor(sum, 8);

            S[i] = S[i]*ms + sum;

            FOR_UNROLL (short id = 0; id < TDh; ++id) {
                FOR_UNROLL (short j = 0; j < 4; ++j) {
                    of[id][4*i + j] *= ms;
                }
            }
        }

        // O += P*V over this simdgroup's slice of the head dim
        FOR_UNROLL (short id = 0; id < TDh; id += 2) {
            FOR_UNROLL (short ik = 0; ik < TK; ++ik) {
                vec<half, 8> v0;
                vec<half, 8> v1;

                device const half * pv0 = pv + (ic + ik*kU)*sv + id*kU;

                FOR_UNROLL (short i = 0; i < 2; ++i) {
                    const half4 a = *((device const half4 *) (pv0 + 8*i*sv));
                    const half4 b = *((device const half4 *) (pv0 + 8*i*sv + kU));

                    FOR_UNROLL (short j = 0; j < 4; ++j) {
                        v0[4*i + j] = a[j];
                        v1[4*i + j] = b[j];
                    }
                }

                fa_nax_mma<float, float, half, false>(of[id], of[id + 1], sf[ik], v0, v1);
            }
        }
    }

    // normalize and store - rows past ne01 are not written
    device float * pdst = (device float *) dst + ((uint64_t) iq3*args.ne2*args.ne1 + iq2)*DV + d_group*BDh + fn;

    FOR_UNROLL (short i = 0; i < 2; ++i) {
        const int r = r0 + 8*i;

        if (r < args.ne01) {
            // the sink only adds to the denominator - its rescale of O is folded into the final scale
            float ms = 1.0f;
            float sr = S[i];

            if (FC_flash_attn_ext_nax_has_sinks) {
                const float s = M_LOG2E_F*((device const float *) sinks)[iq2];
                const float m = max(M[i], s);

                ms = fast::exp2(M[i] - m);
                sr = sr*ms + fast::exp2(s - m);
            }

            const float rs = sr == 0.0f ? 0.0f : ms/sr;

            device float * pd = pdst + (uint64_t) r*args.ne1*DV;

            FOR_UNROLL (short id = 0; id < TDh; ++id) {
                *((device float4 *) (pd + id*kU)) = float4(of[id][4*i + 0], of[id][4*i + 1], of[id][4*i + 2], of[id][4*i + 3])*rs;
            }
        }
    }
}

typedef decltype(kernel_flash_attn_ext_nax<256, 256, 2>) flash_attn_ext_nax_t;

template [[host_name("kernel_flash_attn_ext_nax_f16_dk64_dv64")]]   kernel flash_attn_ext_nax_t kernel_flash_attn_ext_nax<64,  64,  1>;
template [[host_name("kernel_flash_attn_ext_nax_f16_dk128_dv128")]] kernel flash_attn_ext_nax_t kernel_flash_attn_ext_nax<128, 128, 1>;
template [[host_name("kernel_flash_attn_ext_nax_f16_dk256_dv256")]] kernel flash_attn_ext_nax_t kernel_flash_attn_ext_nax<256, 256, 2>;
template [[host_name("kernel_flash_attn_ext_nax_f16_dk512_dv512")]] kernel flash_attn_ext_nax_t kernel_flash_attn_ext_nax<512, 512, 4>;


#endif // GGML_METAL_HAS_TENSOR
