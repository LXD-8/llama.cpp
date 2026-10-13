#include "common.h"
#include "dequantize.h"

#if defined(GGML_METAL_HAS_TENSOR) && __ENVIRONMENT_OS_VERSION_MIN_REQUIRED__ >= 260200
// the NAX kernels (nax_mma) need the cooperative tensor layout of OS 26.2+: the host uses them only if this kernel exists
kernel void kernel_nax_ok() {}
#endif

constant bool FC_mul_mm_bc_inp [[function_constant(FC_MUL_MM + 0)]];
constant bool FC_mul_mm_bc_out [[function_constant(FC_MUL_MM + 1)]];
constant short FC_mul_mm_ne12  [[function_constant(FC_MUL_MM + 2)]];
constant short FC_mul_mm_ne13  [[function_constant(FC_MUL_MM + 3)]];
constant short FC_mul_mm_r2    [[function_constant(FC_MUL_MM + 4)]];
constant short FC_mul_mm_r3    [[function_constant(FC_MUL_MM + 5)]];
constant int32_t FC_mul_mm_id_map0_ne02 [[function_constant(FC_MUL_MM + 7)]];
constant bool FC_mul_mm_id_wide [[function_constant(FC_MUL_MM + 8)]]; // tiles of 64 entries instead of 32 (tensor kernel)

// converts the f32 activations of a mul_mm to contiguous f16 rows, saturated to the half range
kernel void kernel_mul_mm_cvt_f32_f16(
        constant ggml_metal_kargs_mul_mm_cvt & args,
        device const char * src1,
        device       half * dst,
        uint3  tgpig [[threadgroup_position_in_grid]],
        ushort tpitg [[thread_index_in_threadgroup]],
        ushort3 ntg  [[threads_per_threadgroup]]) {
    const int i1 = tgpig.x;
    const int i2 = tgpig.y;
    const int i3 = tgpig.z;

    device const float4 * s = (device const float4 *) (src1 + i1*args.nb11 + i2*args.nb12 + i3*args.nb13);
    device       half4  * d = (device       half4  *) (dst + (((uint64_t) i3*args.ne12 + i2)*args.ne11 + i1)*args.ne10);

    for (int i = tpitg; i < args.ne10/4; i += ntg.x) {
        d[i] = (half4) clamp(s[i], -65504.0f, 65504.0f);
    }
}

#ifdef GGML_METAL_HAS_TENSOR
// dequantizes a whole src0 matrix to contiguous f16 rows, one threadgroup per row
template<typename block_q, short nl, void (*dequantize_func)(device const block_q *, short, thread half4x4 &)>
kernel void kernel_mul_mm_dq_f16(
        constant ggml_metal_kargs_mul_mm_dq & args,
        device const char * src0,
        device       half * dst,
        uint3  tgpig [[threadgroup_position_in_grid]],
        ushort tpitg [[thread_index_in_threadgroup]],
        ushort3 ntg  [[threads_per_threadgroup]]) {
    const int i01 = tgpig.x;

    device const block_q * x = (device const block_q *) (src0 + i01*args.nb01);
    device       half4x4 * d = (device       half4x4 *) (dst + (uint64_t) i01*args.ne00);

    for (int i = tpitg; i < args.ne00/16; i += ntg.x) {
        half4x4 t;
        dequantize_func(x + i/nl, i%nl, t);
        d[i] = t;
    }
}

// the quantized weight types of the NAX mat-mat paths; q5_K is listed per path, since the few-row kernel uses dequantize_q5_K_nax
#define NAX_QTYPES(X) \
    X(q4_0,    block_q4_0,    2,     dequantize_q4_0) \
    X(q4_1,    block_q4_1,    2,     dequantize_q4_1) \
    X(q5_0,    block_q5_0,    2,     dequantize_q5_0) \
    X(q5_1,    block_q5_1,    2,     dequantize_q5_1) \
    X(q8_0,    block_q8_0,    2,     dequantize_q8_0) \
    X(q4_K,    block_q4_K,    QK_NL, dequantize_q4_K) \
    X(q6_K,    block_q6_K,    QK_NL, dequantize_q6_K) \
    X(q1_0,    block_q1_0,    8,     dequantize_q1_0) \
    X(q2_0,    block_q2_0,    4,     dequantize_q2_0) \
    X(mxfp4,   block_mxfp4,   2,     dequantize_mxfp4) \
    X(q2_K,    block_q2_K,    QK_NL, dequantize_q2_K) \
    X(q3_K,    block_q3_K,    QK_NL, dequantize_q3_K) \
    X(iq2_xxs, block_iq2_xxs, QK_NL, dequantize_iq2_xxs) \
    X(iq2_xs,  block_iq2_xs,  QK_NL, dequantize_iq2_xs) \
    X(iq2_s,   block_iq2_s,   QK_NL, dequantize_iq2_s) \
    X(iq3_xxs, block_iq3_xxs, QK_NL, dequantize_iq3_xxs) \
    X(iq3_s,   block_iq3_s,   QK_NL, dequantize_iq3_s) \
    X(iq1_s,   block_iq1_s,   QK_NL, dequantize_iq1_s) \
    X(iq1_m,   block_iq1_m,   QK_NL, dequantize_iq1_m) \
    X(iq4_nl,  block_iq4_nl,  2,     dequantize_iq4_nl) \
    X(iq4_xs,  block_iq4_xs,  QK_NL, dequantize_iq4_xs) \
    X(tq2_0,   block_tq2_0,   QK_NL, dequantize_tq2_0)

typedef decltype(kernel_mul_mm_dq_f16<block_q4_K, QK_NL, dequantize_q4_K>) mul_mm_dq_t;

#define DQ_F16_INSTANCE(name, block_q, nl, deq) \
template [[host_name("kernel_mul_mm_dq_" #name "_f16")]] kernel mul_mm_dq_t kernel_mul_mm_dq_f16<block_q, nl, deq>;

NAX_QTYPES(DQ_F16_INSTANCE)
DQ_F16_INSTANCE(q5_K, block_q5_K, QK_NL, dequantize_q5_K)

// dst[m][n] = sum_k src1[m][k] * src0[n][k] with f16 or int8 operands loaded straight into registers, no threadgroup memory
// int8: each OP_MUL_MM_I8_G group of K is accumulated in int32, then folded into the f32 accumulators with the scales s1 and s0
template<typename T, short BM, short BN, short WM, short WN>
static inline void kernel_mul_mm_nax_impl(
        constant ggml_metal_kargs_mul_mm_nax & args,
        device const T     * src0,
        device const T     * src1,
        device       float * dst,
        device const float * s0,
        device const float * s1,
        uint3  tgpig,
        ushort tiisg,
        ushort sgitg) {
    constexpr bool I8 = is_same<T, int8_t>::value;

    constexpr short SM = BM/WM;
    constexpr short SN = BN/WN;
    constexpr short TM = SM/16;
    constexpr short TN = SN/16;
    constexpr short TK = 2;      // 16-wide K fragments per step
    constexpr short SK = TK*16;

    static_assert(TN % 2 == 0, "src0 fragments are consumed in pairs");

    const int tile_m = (tgpig.y << args.swz) + (tgpig.x & ((1 << args.swz) - 1));
    const int tile_n = tgpig.x >> args.swz;

    if (tile_m >= args.tiles_m || tile_n >= args.tiles_n) {
        return;
    }

    const int m0 = tile_m*BM + SM*(sgitg/WN);
    const int n0 = tile_n*BN + SN*(sgitg%WN);

    const short2 sc = nax_coord(tiisg);

    // per-lane row pointers, clamped to the last row - rows past the end are computed but never stored
    device const T * pa[TM][2];
    device const T * pb[TN][2];
    int ra[TM][2]; // the clamped rows and the first of the 4 columns of the lane, for the int8 scales
    int rb[TN];

    FOR_UNROLL (short mm = 0; mm < TM; ++mm) {
        FOR_UNROLL (short i = 0; i < 2; ++i) {
            const int r = min(m0 + mm*16 + sc.y + i*8, args.M - 1);
            ra[mm][i] = r;
            pa[mm][i] = src1 + (uint64_t) r*args.lda + sc.x;
        }
    }
    FOR_UNROLL (short nn = 0; nn < TN; ++nn) {
        FOR_UNROLL (short i = 0; i < 2; ++i) {
            const int r = min(n0 + nn*16 + sc.y + i*8, args.N - 1);
            pb[nn][i] = src0 + (uint64_t) r*args.ldb + sc.x;
        }
        rb[nn] = n0 + nn*16 + sc.x;
    }

    vec<float, 8> c[TM][TN];
    FOR_UNROLL (short mm = 0; mm < TM; ++mm) {
        FOR_UNROLL (short nn = 0; nn < TN; ++nn) {
            c[mm][nn] = 0.0f;
        }
    }

    const int BK = args.bk;

    #pragma clang loop unroll(disable)
    for (int k0 = args.k_beg; k0 < args.k_end; k0 += BK) {
        // keep the simdgroups of the threadgroup close together so that they hit the same cache lines
        threadgroup_barrier(mem_flags::mem_none);

        const int k1 = min(args.k_end, k0 + BK);

        vec<int, 8> ci[TM][TN];
        if constexpr (I8) {
            FOR_UNROLL (short mm = 0; mm < TM; ++mm) {
                FOR_UNROLL (short nn = 0; nn < TN; ++nn) {
                    ci[mm][nn] = 0;
                }
            }
        }

        #pragma clang loop unroll(disable)
        for (int k = k0; k < k1; k += SK) {
            volatile int compiler_barrier;

            vec<T, 8> a[TM][TK];
            vec<T, 8> b[TN][TK];

            FOR_UNROLL (short mm = 0; mm < TM; ++mm) {
                FOR_UNROLL (short kk = 0; kk < TK; ++kk) {
                    FOR_UNROLL (short i = 0; i < 2; ++i) {
                        const vec<T, 4> v = *((device const vec<T, 4> *) (pa[mm][i] + k + kk*16));
                        a[mm][kk][i*4 + 0] = v[0];
                        a[mm][kk][i*4 + 1] = v[1];
                        a[mm][kk][i*4 + 2] = v[2];
                        a[mm][kk][i*4 + 3] = v[3];
                    }
                }
            }
            FOR_UNROLL (short nn = 0; nn < TN; ++nn) {
                FOR_UNROLL (short kk = 0; kk < TK; ++kk) {
                    FOR_UNROLL (short i = 0; i < 2; ++i) {
                        const vec<T, 4> v = *((device const vec<T, 4> *) (pb[nn][i] + k + kk*16));
                        b[nn][kk][i*4 + 0] = v[0];
                        b[nn][kk][i*4 + 1] = v[1];
                        b[nn][kk][i*4 + 2] = v[2];
                        b[nn][kk][i*4 + 3] = v[3];
                    }
                }
            }

            FOR_UNROLL (short mm = 0; mm < TM; ++mm) {
                FOR_UNROLL (short nn = 0; nn < TN; nn += 2) {
                    FOR_UNROLL (short kk = 0; kk < TK; ++kk) {
                        if constexpr (I8) {
                            nax_mma<int, T, T, true>(ci[mm][nn], ci[mm][nn + 1], a[mm][kk], b[nn][kk], b[nn + 1][kk]);
                        } else {
                            nax_mma<float, T, T, true>(c[mm][nn], c[mm][nn + 1], a[mm][kk], b[nn][kk], b[nn + 1][kk]);
                        }
                    }
                }
            }

            (void) compiler_barrier;
        }

        if constexpr (I8) {
            // fold the group into the f32 accumulators
            const int g = k0/OP_MUL_MM_I8_G;

            float sa[TM][2];
            FOR_UNROLL (short mm = 0; mm < TM; ++mm) {
                FOR_UNROLL (short i = 0; i < 2; ++i) {
                    sa[mm][i] = s1[g*args.M + ra[mm][i]];
                }
            }

            FOR_UNROLL (short nn = 0; nn < TN; ++nn) {
                float4 sw;
                if (rb[nn] + 3 < args.N) {
                    sw = *((device const float4 *) (s0 + g*args.N + rb[nn]));
                } else {
                    FOR_UNROLL (short j = 0; j < 4; ++j) {
                        sw[j] = s0[g*args.N + min(rb[nn] + j, args.N - 1)];
                    }
                }

                FOR_UNROLL (short mm = 0; mm < TM; ++mm) {
                    FOR_UNROLL (short i = 0; i < 2; ++i) {
                        FOR_UNROLL (short j = 0; j < 4; ++j) {
                            c[mm][nn][i*4 + j] = fma(sa[mm][i]*sw[j], float(ci[mm][nn][i*4 + j]), c[mm][nn][i*4 + j]);
                        }
                    }
                }
            }
        }
    }

    FOR_UNROLL (short mm = 0; mm < TM; ++mm) {
        FOR_UNROLL (short i = 0; i < 2; ++i) {
            const int r = m0 + mm*16 + sc.y + i*8;
            if (r >= args.M) {
                continue;
            }
            device float * d = dst + (uint64_t) r*args.ldd;
            FOR_UNROLL (short nn = 0; nn < TN; ++nn) {
                const int col = n0 + nn*16 + sc.x;
                const float4 v = float4(c[mm][nn][i*4 + 0], c[mm][nn][i*4 + 1], c[mm][nn][i*4 + 2], c[mm][nn][i*4 + 3]);
                if (col + 3 < args.N) {
                    if (args.accum) {
                        *((device float4 *) (d + col)) += v;
                    } else {
                        *((device float4 *) (d + col)) = v;
                    }
                } else {
                    for (short j = 0; j < 4; ++j) {
                        if (col + j < args.N) {
                            d[col + j] = args.accum ? d[col + j] + v[j] : v[j];
                        }
                    }
                }
            }
        }
    }
}

[[kernel, max_total_threads_per_threadgroup(OP_MUL_MM_NAX_WM*OP_MUL_MM_NAX_WN*32)]]
void kernel_mul_mm_nax_f16(
        constant ggml_metal_kargs_mul_mm_nax & args,
        device const half  * src0,
        device const half  * src1,
        device       float * dst,
        uint3  tgpig [[threadgroup_position_in_grid]],
        ushort tiisg [[thread_index_in_simdgroup]],
        ushort sgitg [[simdgroup_index_in_threadgroup]]) {
    kernel_mul_mm_nax_impl<half, OP_MUL_MM_NAX_BM, OP_MUL_MM_NAX_BN, OP_MUL_MM_NAX_WM, OP_MUL_MM_NAX_WN>(
            args, src0, src1, dst, nullptr, nullptr, tgpig, tiisg, sgitg);
}
#endif // GGML_METAL_HAS_TENSOR

#ifdef GGML_METAL_HAS_TENSOR
// int8 NAX mat-mat: symmetric int8 with one scale per (row, OP_MUL_MM_I8_G values of K), after a Walsh-Hadamard transform of each group
// the scales are stored group-major: s[g*rows + row]

// f32 activations -> int8 rows [M][K] + scales [K/G][M], one simdgroup per group
kernel void kernel_mul_mm_q8a_f32(
        constant ggml_metal_kargs_mul_mm_q8a & args,
        device const char   * src1,
        device       int8_t * dst,
        device       float  * dsc,
        uint3  tgpig [[threadgroup_position_in_grid]],
        ushort tiisg [[thread_index_in_simdgroup]],
        ushort sgitg [[simdgroup_index_in_threadgroup]],
        ushort3 ntg  [[threads_per_threadgroup]]) {
    const int i1 = tgpig.x;
    const int ng = args.ne10/OP_MUL_MM_I8_G;

    device const float4 * x = (device const float4 *) (src1 + i1*args.nb11);
    device       char4  * q = (device       char4  *) (dst + (uint64_t) i1*args.ne10);

    for (int g = sgitg; g < ng; g += ntg.y) {
        // the 1/128 of the transform is folded into the activation scale
        float d;
        q[g*(OP_MUL_MM_I8_G/4) + tiisg] = i8_quantize128(x[g*(OP_MUL_MM_I8_G/4) + tiisg], tiisg, OP_MUL_MM_I8_G, d);

        if (tiisg == 0) {
            dsc[g*args.ne11 + i1] = d;
        }
    }
}

// quantized weights -> int8 rows [N][K] + scales [K/G][N], one simdgroup per group
template<typename block_q, short nl, void (*dequantize_func)(device const block_q *, short, thread float4x4 &)>
kernel void kernel_mul_mm_dq_i8(
        constant ggml_metal_kargs_mul_mm_dq_i8 & args,
        device const char   * src0,
        device       int8_t * dst,
        device       float  * dsc,
        uint3  tgpig [[threadgroup_position_in_grid]],
        ushort tiisg [[thread_index_in_simdgroup]],
        ushort sgitg [[simdgroup_index_in_threadgroup]],
        ushort3 ntg  [[threads_per_threadgroup]]) {
    const int i01 = tgpig.x;
    const int ng  = args.ne00/OP_MUL_MM_I8_G;

    device const block_q * x = (device const block_q *) (src0 + i01*args.nb01);
    device       char4   * q = (device       char4   *) (dst + (uint64_t) i01*args.ne00);

    for (int g = sgitg; g < ng; g += ntg.y) {
        const int ic = g*(OP_MUL_MM_I8_G/16) + tiisg/4; // 16-value chunk

        float4x4 t;
        dequantize_func(x + ic/nl, ic%nl, t);

        float d;
        q[ic*4 + tiisg%4] = i8_quantize128(t[tiisg%4], tiisg, 1.0f, d);

        if (tiisg == 0) {
            dsc[g*args.ne01 + i01] = d;
        }
    }
}

typedef decltype(kernel_mul_mm_dq_i8<block_q4_K, QK_NL, dequantize_q4_K>) mul_mm_dq_i8_t;

#define DQ_I8_INSTANCE(name, block_q, nl, deq) \
template [[host_name("kernel_mul_mm_dq_" #name "_i8")]] kernel mul_mm_dq_i8_t kernel_mul_mm_dq_i8<block_q, nl, deq>;

NAX_QTYPES(DQ_I8_INSTANCE)
DQ_I8_INSTANCE(q5_K, block_q5_K, QK_NL, dequantize_q5_K)

[[kernel, max_total_threads_per_threadgroup(OP_MUL_MM_NAX_WM*OP_MUL_MM_NAX_WN*32)]]
void kernel_mul_mm_nax_i8(
        constant ggml_metal_kargs_mul_mm_nax & args,
        device const int8_t * src0,
        device const int8_t * src1,
        device       float  * dst,
        device const float  * s0,
        device const float  * s1,
        uint3  tgpig [[threadgroup_position_in_grid]],
        ushort tiisg [[thread_index_in_simdgroup]],
        ushort sgitg [[simdgroup_index_in_threadgroup]]) {
    kernel_mul_mm_nax_impl<int8_t, OP_MUL_MM_NAX_BM, OP_MUL_MM_NAX_BN, OP_MUL_MM_NAX_WM, OP_MUL_MM_NAX_WN>(
            args, src0, src1, dst, s0, s1, tgpig, tiisg, sgitg);
}

#endif // GGML_METAL_HAS_TENSOR

// each block_q contains 16*nl weights
#ifdef GGML_METAL_HAS_TENSOR
template<
    typename SA, typename SA_4x4, typename SA_8x8,
    typename SB, typename SB_2x4, typename SB_8x8,
    typename block_q, short nl, void (*dequantize_func)(device const block_q *, short, thread SA_4x4 &),
    typename T0, typename T0_4x4, typename T1, typename T1_2x4>
kernel void kernel_mul_mm(
        constant ggml_metal_kargs_mul_mm & args,
        device const char * srcA,
        device const char * srcB,
        device       char * dst,
        threadgroup  char * shmem [[threadgroup(0)]],
        uint3  tgpig [[threadgroup_position_in_grid]],
        ushort tiitg [[thread_index_in_threadgroup]],
        ushort sgitg [[simdgroup_index_in_threadgroup]]) {
    (void) sgitg;

    // Matrix dimensions: A(M,K) x B(K,N) -> C(M,N)
    const int K = args.ne00;
    const int M = args.ne0;
    const int N = args.ne1;

    // Batch dimension handling
    const int im = tgpig.z;
    const int i12 = im % FC_mul_mm_ne12;
    const int i13 = im / FC_mul_mm_ne12;

    // Batch offsets for srcA and srcB
    const uint64_t offset0 = (i12/FC_mul_mm_r2)*args.nb02 + (i13/FC_mul_mm_r3)*args.nb03;

    // Tile dimensions
    constexpr int NRB = SZ_SIMDGROUP * N_MM_BLOCK_X * N_MM_SIMD_GROUP_X;
    constexpr int NRA = SZ_SIMDGROUP * N_MM_BLOCK_Y * N_MM_SIMD_GROUP_Y;

    // Tile offsets in output matrix
    const int ra = tgpig.y * NRA;
    const int rb = tgpig.x * NRB;

    // Threadgroup memory for dequantized A tile only
    threadgroup SA * sa = (threadgroup SA *)(shmem);

    // Work-item count for A loading
    constexpr int A_WORK_ITEMS = NRA * N_MM_NK;
    constexpr int NUM_THREADS = N_SIMDWIDTH * N_MM_SIMD_GROUP_X * N_MM_SIMD_GROUP_Y;

    // tA wraps threadgroup memory
    auto tA = tensor(sa, dextents<int32_t, 2>(N_MM_NK_TOTAL, NRA));

    // tB wraps device memory directly
    device T1 * ptrB = (device T1 *)(srcB + args.nb12*i12 + args.nb13*i13);
    const int strideB = args.nb11 / sizeof(T1);
    auto tB = tensor(ptrB, dextents<int32_t, 2>(K, N), array<int, 2>({1, strideB}));

    // Configure matmul operation
    // note: K is dynamic_extent (clamped to the valid range in PHASE 2), since a static
    //       N_MM_NK_TOTAL K tile would read src1 out of bounds when K % N_MM_NK_TOTAL != 0
    // ref: https://github.com/ggml-org/llama.cpp/pull/27064
    mpp::tensor_ops::matmul2d<
        mpp::tensor_ops::matmul2d_descriptor(
            NRB, NRA, static_cast<int>(dynamic_extent), false, true, true,
            mpp::tensor_ops::matmul2d_descriptor::mode::multiply_accumulate),
        execution_simdgroups<N_MM_SIMD_GROUP_X * N_MM_SIMD_GROUP_Y>> mm;

    auto cT = mm.get_destination_cooperative_tensor<decltype(tB), decltype(tA), float>();

    // Accumulate partial results over K dimension
    for (int loop_k = 0; loop_k < K; loop_k += N_MM_NK_TOTAL) {
        // === PHASE 1: Dequantization of A into threadgroup memory ===
        for (int work = tiitg; work < A_WORK_ITEMS; work += NUM_THREADS) {
            const int row = work / N_MM_NK;
            const int k_chunk = work % N_MM_NK;
            const int k_pos = loop_k + k_chunk * 16;
            const short k_base = k_chunk * 16;

            // Bounds check: skip device read if row is out of matrix bounds
            if (ra + row < M) {
                if (is_same<T0_4x4, block_q>::value && FC_mul_mm_bc_inp) {
                    // Element-wise reads when K is not aligned (nb01 not aligned for half4x4/float4x4).
                    // MSL spec Table 2.5: half4x4 requires 8-byte alignment. When K is odd,
                    // nb01 = K*2 is not 8-byte aligned, so odd-row pointers are misaligned.
                    // Mirrors the legacy kernel's existing guard.
                    device const T0 * row_ptr = (device const T0 *)(srcA + args.nb01 * (ra + row) + offset0);

                    FOR_UNROLL (short i = 0; i < 16; i++) {
                        sa[row * N_MM_NK_TOTAL + (k_base + i)] = (k_pos + i < K) ? (SA) row_ptr[k_pos + i] : (SA)0;
                    }
                } else {
                    const int block_idx = k_pos / (16 * nl);
                    const short il = (k_pos / 16) % nl;

                    device const block_q * row_ptr = (device const block_q *)(srcA + args.nb01 * (ra + row) + offset0);

                    SA_4x4 temp_a;
                    dequantize_func(row_ptr + block_idx, il, temp_a);

                    FOR_UNROLL (short i = 0; i < 16; i++) {
                        // Zero-pad A for K positions beyond valid range (handles partial K iterations)
                        sa[row * N_MM_NK_TOTAL + (k_base + i)] = (k_pos + i < K) ? temp_a[i/4][i%4] : (SA)0;
                    }
                }
            } else {
                // Zero-pad rows beyond matrix bounds
                FOR_UNROLL (short i = 0; i < 16; i++) {
                    sa[row * N_MM_NK_TOTAL + (k_base + i)] = (SA)0;
                }
            }
        }

        threadgroup_barrier(mem_flags::mem_threadgroup);

        // === PHASE 2: Tensor matmul ===
        // Clamp the K extent of both operand tensors to the remaining valid K range so
        // the dynamic-K op never reads past the K extent of src1 (or the staged A tile).
        const int kExt = min(N_MM_NK_TOTAL, K - loop_k);

        auto tAv = tensor(sa, dextents<int32_t, 2>(kExt, NRA), array<int, 2>({1, N_MM_NK_TOTAL}));
        auto tBv = tensor(ptrB + loop_k + rb * strideB, dextents<int32_t, 2>(kExt, N - rb), array<int, 2>({1, strideB}));

        mm.run(tBv, tAv, cT);

        threadgroup_barrier(mem_flags::mem_threadgroup);
    }

    // Store result tile to output matrix (with batch offset)
    // a slice is not used because its byte offset is int32 and wraps at 2 GiB
    // ref: https://github.com/ggml-org/llama.cpp/pull/28748
    const int tileM = min(NRA, M - ra);
    const int tileN = min(NRB, N - rb);

    device float * dstTile = (device float *)dst + (uint64_t) im * N * M + (uint64_t) rb * M + ra;

    auto tD = tensor(dstTile, dextents<int32_t, 2>(tileM, tileN), array<int, 2>({1, M}));
    cT.store(tD);
}

#else

template<
    typename S0, typename S0_4x4, typename S0_8x8,
    typename S1, typename S1_2x4, typename S1_8x8,
    typename block_q, short nl, void (*dequantize_func)(device const block_q *, short, thread S0_4x4 &),
    typename T0, typename T0_4x4, typename T1, typename T1_2x4>
kernel void kernel_mul_mm(
        constant ggml_metal_kargs_mul_mm & args,
        device const char * src0,
        device const char * src1,
        device       char * dst,
        threadgroup  char * shmem [[threadgroup(0)]],
        uint3  tgpig[[threadgroup_position_in_grid]],
        ushort tiitg[[thread_index_in_threadgroup]],
        ushort sgitg[[simdgroup_index_in_threadgroup]]) {

    threadgroup S0 * sa = (threadgroup S0 *)(shmem);
    threadgroup S1 * sb = (threadgroup S1 *)(shmem + 4096);

    constexpr int NR0 = 64;
    constexpr int NR1 = 32;

    constexpr int NK  = 32;
    constexpr int NL0 = NK/16;
    constexpr int NL1 = NK/8;

    const int im = tgpig.z;
    const int r0 = tgpig.y*NR0;
    const int r1 = tgpig.x*NR1;

    // if this block is of 64x32 shape or smaller
    const short nr0 = (args.ne0 - r0 < NR0) ? (args.ne0 - r0) : NR0;
    const short nr1 = (args.ne1 - r1 < NR1) ? (args.ne1 - r1) : NR1;

    // a thread shouldn't load data outside of the matrix
    const short lr0 = ((short)tiitg/NL0) < nr0 ? ((short)tiitg/NL0) : nr0 - 1; // 0 .. 63
    const short lr1 = ((short)tiitg/NL1) < nr1 ? ((short)tiitg/NL1) : nr1 - 1; // 0 .. 31

    const short il0 = (tiitg % NL0);

    short il = il0;

    const int i12 = im % FC_mul_mm_ne12;
    const int i13 = im / FC_mul_mm_ne12;

    const uint64_t offset0 = (i12/FC_mul_mm_r2)*args.nb02 + (i13/FC_mul_mm_r3)*args.nb03;
    const short    offset1 = il0/nl;

    device const block_q * x = (device const block_q *)(src0 + args.nb01*(r0 + lr0) + offset0) + offset1;

    const short iy = 8*(tiitg % NL1);

    device const T1 * y = (device const T1 *)(src1
        + args.nb13*i13
        + args.nb12*i12
        + args.nb11*(r1 + lr1)
        + args.nb10*iy);

    S0_8x8 ma[4];
    S1_8x8 mb[2];

    simdgroup_float8x8 mc[8];

    for (short i = 0; i < 8; i++){
        mc[i] = make_filled_simdgroup_matrix<float, 8>(0.f);
    }

    for (int loop_k = 0; loop_k < args.ne00; loop_k += NK) {
        // load data and store to threadgroup memory
        if (is_same<T0_4x4, block_q>::value && FC_mul_mm_bc_inp) {
            threadgroup_barrier(mem_flags::mem_threadgroup);

            // no need for dequantization
            for (short i = 0; i < 16; i++) {
                const short sx = 2*il0 + i/8;
                const short sy = (tiitg/NL0)/8;

              //const short lx = i%8;
              //const short ly = (tiitg/NL0)%8;
                const short lx = (tiitg/NL0)%8;
                const short ly = i%8;

                const short ib = 8*sx + sy;

                *(sa + 64*ib + 8*ly + lx) = loop_k + 16*il + i < args.ne00 ? (S0) *((device T0 *) x + i) : 0;
            }
        } else {
            S0_4x4 temp_a;
            dequantize_func(x, il, temp_a);

            threadgroup_barrier(mem_flags::mem_threadgroup);

            FOR_UNROLL (short i = 0; i < 16; i++) {
                const short sx = 2*il0 + i/8;
                const short sy = (tiitg/NL0)/8;

              //const short lx = i%8;
              //const short ly = (tiitg/NL0)%8;
                const short lx = (tiitg/NL0)%8;
                const short ly = i%8;

                const short ib = 8*sx + sy;

                // NOTE: this is massively slower.. WTF?
                //sa[64*ib + 8*ly + lx] = temp_a[i/4][i%4];

                *(sa + 64*ib + 8*ly + lx) = temp_a[i/4][i%4];
            }
        }

        if (FC_mul_mm_bc_inp) {
            for (short i = 0; i < 8; ++i) {
                const short sx = (tiitg%NL1);
                const short sy = (tiitg/NL1)/8;

                const short lx = i;
                const short ly = (tiitg/NL1)%8;
              //const short lx = (tiitg/NL1)%8;
              //const short ly = i;

                const short ib = 4*sx + sy;

                *(sb + 64*ib + 8*ly + lx) = loop_k + iy + i < args.ne00 ? (S1) *((device T1 *) y + i) : 0;
            }
        } else {
            const short sx = (tiitg%NL1);
            const short sy = (tiitg/NL1)/8;

          //const short dx = sx;
          //const short dy = sy;

            const short ly = (tiitg/NL1)%8;

            const short ib = 4*sx + sy;

            *(threadgroup S1_2x4 *)(sb + 64*ib + 8*ly) = (S1_2x4)(*((device T1_2x4 *) y));
        }

        il = (il + 2 < nl) ? il + 2 : il % 2;
        x  = (il < 2) ? x + (2 + nl - 1)/nl : x;

        y += NK;

        threadgroup_barrier(mem_flags::mem_threadgroup);

        // load matrices from threadgroup memory and conduct outer products
        threadgroup const S0 * lsma = (sa + 4*64*(sgitg%2));
        threadgroup const S1 * lsmb = (sb + 2*64*(sgitg/2));

        FOR_UNROLL (short ik = 0; ik < NK/8; ik++) {
            simdgroup_barrier(mem_flags::mem_none);

            FOR_UNROLL (short i = 0; i < 4; i++) {
                simdgroup_load(ma[i], lsma + 64*i, 8, 0, false);
            }

            simdgroup_barrier(mem_flags::mem_none);

            FOR_UNROLL (short i = 0; i < 2; i++) {
                simdgroup_load(mb[i], lsmb + 64*i, 8, 0, false);
            }

            simdgroup_barrier(mem_flags::mem_none);

            FOR_UNROLL (short i = 0; i < 8; i++){
                simdgroup_multiply_accumulate(mc[i], mb[i/4], ma[i%4], mc[i]);
            }

            lsma += 8*64;
            lsmb += 4*64;
        }
    }

    if (!FC_mul_mm_bc_out || (r0 + NR0 <= args.ne0 && r1 + NR1 <= args.ne1)) {
        // if no bounds checks on the output are needed, we can directly write to device memory
        device float * C = (device float *) dst +
            (r0 + 32*(sgitg &  1)) + \
            (uint64_t) (r1 + 16*(sgitg >> 1))*args.ne0 + (uint64_t) im*args.ne1*args.ne0;

        for (short i = 0; i < 8; i++) {
            simdgroup_store(mc[i], C + 8*(i%4) + (uint64_t) 8*args.ne0*(i/4), args.ne0, 0, false);
        }
    } else {
        // block is smaller than 64x32, we should avoid writing data outside of the matrix
        threadgroup_barrier(mem_flags::mem_threadgroup);

        threadgroup float * temp_str = ((threadgroup float *) shmem) + 32*(sgitg&1) + (16*(sgitg >> 1))*NR0;

        for (short i = 0; i < 8; i++) {
            simdgroup_store(mc[i], temp_str + 8*(i%4) + 8*NR0*(i/4), NR0, 0, false);
        }

        threadgroup_barrier(mem_flags::mem_threadgroup);

        if (sgitg == 0) {
            for (int j = tiitg; j < nr1; j += NR1) {
                device float  * D  = (device float  *) dst + r0 + (uint64_t) (r1 + j)*args.ne0 + (uint64_t) im*args.ne1*args.ne0;
                device float4 * D4 = (device float4 *) D;

                threadgroup float  * C  = temp_str + (j*NR0);
                threadgroup float4 * C4 = (threadgroup float4 *) C;

                int i = 0;
                for (; i < nr0/4; i++) {
                    *(D4 + i) = *(C4 + i);
                }

                i *= 4;
                for (; i < nr0; i++) {
                    *(D + i) = *(C + i);
                }
            }
        }
    }
}

#endif // GGML_METAL_HAS_TENSOR

template<short ne20> // n_expert_used
kernel void kernel_mul_mm_id_map0(
        constant ggml_metal_kargs_mul_mm_id_map0 & args,
        device  const char * src2,
        device        char * htpe,
        device        char * hids,
        threadgroup   char * shmem [[threadgroup(0)]],
        ushort tpitg[[thread_position_in_threadgroup]],
        ushort   ntg[[threads_per_threadgroup]]) {
    const short ide = tpitg; // expert id

    // per expert: entry count (pass 1), replaced by the base offset of the slice (pass 2)
    threadgroup uint32_t * cnt = (threadgroup uint32_t *) (shmem);
    threadgroup uint32_t * bas = cnt + FC_mul_mm_id_map0_ne02;

    // staging of the ids of ntg tokens
    threadgroup uint16_t * sids = (threadgroup uint16_t *) (bas + FC_mul_mm_id_map0_ne02);

    // pass 1: count the entries of this expert - every occurrence of this expert within the
    // ids is an entry (duplicate ids produce multiple entries for the same token)
    uint32_t n_all = 0;

    for (int i21 = 0; i21 < args.ne21; i21 += ntg) { // n_tokens
        if (i21 + tpitg < args.ne21) {
            device const int32_t * src2_i32 = (device const int32_t *) (src2 + (i21 + tpitg)*args.nb21);

            // 0xFFFF marks a skipped slot, it never matches an expert since ne02 fits in a threadgroup
            FOR_UNROLL (short i20 = 0; i20 < ne20; i20++) {
                const int32_t id = src2_i32[i20];
                sids[tpitg*ne20 + i20] = id == -1 ? 0xFFFF : (uint16_t) id;
            }
        }

        threadgroup_barrier(mem_flags::mem_threadgroup);

        for (short t = 0; t < ntg; t++) {
            if (i21 + t >= args.ne21) {
                break;
            }

            threadgroup const uint16_t * sids_t = sids + t*ne20;

            FOR_UNROLL (short i20 = 0; i20 < ne20; i20++) {
                n_all += sids_t[i20] == ide;
            }
        }

        threadgroup_barrier(mem_flags::mem_threadgroup);
    }

    cnt[ide] = n_all;

    threadgroup_barrier(mem_flags::mem_threadgroup);

    // exclusive prefix sum over the counts: the entries of this expert start at bas[ide] -
    // every slice is padded by one entry for the uniform dummy store of pass 2 - and emit the
    // entry tile descriptors of kernel_mul_mm_id: (expert, first entry, valid rows) per tile
    if (tpitg == 0) {
#ifdef GGML_METAL_HAS_TENSOR
        const uint32_t NTILE = FC_mul_mm_id_wide ? 64 : 32; // must match the tiles of kernel_mul_mm_id
#else
        constexpr uint32_t NTILE = 32; // must match NR1 of kernel_mul_mm_id
#endif

        uint32_t base = 0;
        uint32_t tile = 0;

        device uint32_t * tpe_u32 = (device uint32_t *) (htpe);

        for (int e = 0; e < FC_mul_mm_id_map0_ne02; ++e) {
            const uint32_t n = cnt[e];

            bas[e] = base;

            for (uint32_t k = 0; k < n; k += NTILE) {
                tpe_u32[1 + 3*tile    ] = e;
                tpe_u32[1 + 3*tile + 1] = base + k;
                tpe_u32[1 + 3*tile + 2] = n - k < NTILE ? n - k : NTILE;

                tile++;
            }

            base += n + 1;
        }

        tpe_u32[0] = tile;
    }

    threadgroup_barrier(mem_flags::mem_threadgroup);

    // pass 2: write the entries of this expert into its slice of the compact list
    device int32_t * ids_i32 = (device int32_t *) (hids);

    uint32_t ient = bas[ide];

    for (int i21 = 0; i21 < args.ne21; i21 += ntg) { // n_tokens
        if (i21 + tpitg < args.ne21) {
            device const int32_t * src2_i32 = (device const int32_t *) (src2 + (i21 + tpitg)*args.nb21);

            FOR_UNROLL (short i20 = 0; i20 < ne20; i20++) {
                sids[tpitg*ne20 + i20] = src2_i32[i20];
            }
        }

        threadgroup_barrier(mem_flags::mem_threadgroup);

        for (short t = 0; t < ntg; t++) {
            if (i21 + t >= args.ne21) {
                break;
            }

            threadgroup const uint16_t * sids_t = sids + t*ne20;

            short first = -1;
            short n_ents = 0;

            FOR_UNROLL (short i20 = 0; i20 < ne20; i20++) {
                const short hit = sids_t[i20] == ide;

                first = hit && first < 0 ? i20 : first;
                n_ents += hit;
            }

            // one store per token for every thread, regardless of the match - conditional
            // stores of the divergent lanes are very slow - the dummy value of the tokens
            // without a match lands on the padding slot of this slice and is never read
            ids_i32[ient] = first >= 0 ? (i21 + t)*ne20 + first : 0;

            if (n_ents > 1) {
                // duplicate ids: the remaining matches of this token
                short j = 1;

                FOR_UNROLL (short i20 = 0; i20 < ne20; i20++) {
                    if (sids_t[i20] == ide && i20 != first) {
                        ids_i32[ient + j] = (i21 + t)*ne20 + i20;
                        j += 1;
                    }
                }
            }

            ient += n_ents;
        }

        threadgroup_barrier(mem_flags::mem_threadgroup);
    }
}

// kernel_mul_mm_id only writes the rows that an expert owns, so the skipped rows are zeroed here
kernel void kernel_mul_mm_id_zero(
        constant ggml_metal_kargs_mul_mm_id_zero & args,
        device const char * src2,
        device       char * dst,
        uint3   tgpig[[threadgroup_position_in_grid]],
        ushort3 tpitg[[thread_position_in_threadgroup]],
        ushort3   ntg[[threads_per_threadgroup]]) {
    const int i20 = tgpig.x; // slot
    const int i21 = tgpig.y; // token

    const int32_t id = ((device const int32_t *) (src2 + i21*args.nb21))[i20];

    if (id != -1) {
        return;
    }

    device float * dst_row = (device float *) (dst + i20*args.nb1 + i21*args.nb2);

    for (int i0 = tpitg.x; i0 < args.ne0; i0 += ntg.x) {
        dst_row[i0] = 0.0f;
    }
}

typedef decltype(kernel_mul_mm_id_map0<1>) kernel_mul_mm_id_map0_t;

template [[host_name("kernel_mul_mm_id_map0_ne20_1" )]] kernel kernel_mul_mm_id_map0_t kernel_mul_mm_id_map0<1>;
template [[host_name("kernel_mul_mm_id_map0_ne20_2" )]] kernel kernel_mul_mm_id_map0_t kernel_mul_mm_id_map0<2>;
template [[host_name("kernel_mul_mm_id_map0_ne20_4" )]] kernel kernel_mul_mm_id_map0_t kernel_mul_mm_id_map0<4>;
template [[host_name("kernel_mul_mm_id_map0_ne20_5" )]] kernel kernel_mul_mm_id_map0_t kernel_mul_mm_id_map0<5>;
template [[host_name("kernel_mul_mm_id_map0_ne20_6" )]] kernel kernel_mul_mm_id_map0_t kernel_mul_mm_id_map0<6>;
template [[host_name("kernel_mul_mm_id_map0_ne20_8" )]] kernel kernel_mul_mm_id_map0_t kernel_mul_mm_id_map0<8>;
template [[host_name("kernel_mul_mm_id_map0_ne20_9" )]] kernel kernel_mul_mm_id_map0_t kernel_mul_mm_id_map0<9>;
template [[host_name("kernel_mul_mm_id_map0_ne20_10")]] kernel kernel_mul_mm_id_map0_t kernel_mul_mm_id_map0<10>;
template [[host_name("kernel_mul_mm_id_map0_ne20_16")]] kernel kernel_mul_mm_id_map0_t kernel_mul_mm_id_map0<16>;
template [[host_name("kernel_mul_mm_id_map0_ne20_22")]] kernel kernel_mul_mm_id_map0_t kernel_mul_mm_id_map0<22>;

// TODO: refactor into two seperate functions to avoid ifdef branching: with and without tensor support
template<typename S0, typename S0_4x4, typename S0_8x8, typename S1, typename S1_2x4, typename S1_8x8, typename block_q, short nl, void (*dequantize_func)(device const block_q *, short, thread S0_4x4 &), typename T0, typename T0_4x4, typename T1, typename T1_2x4>
kernel void kernel_mul_mm_id(
        constant ggml_metal_kargs_mul_mm_id & args,
        device const char * src0,
        device const char * src1,
        device const char * htpe,
        device const char * hids,
        device       char * dst,
        threadgroup  char * shmem [[threadgroup(0)]],
        uint3  tgpig[[threadgroup_position_in_grid]],
        ushort tiitg[[thread_index_in_threadgroup]],
        ushort tiisg[[thread_index_in_simdgroup]],
        ushort sgitg[[simdgroup_index_in_threadgroup]]) {
    threadgroup S0 * sa = (threadgroup S0 *)(shmem);
    threadgroup S1 * sb = (threadgroup S1 *)(shmem + 4096);

#ifdef GGML_METAL_HAS_TENSOR
    threadgroup float * sc = (threadgroup float *)(shmem);
#endif

    constexpr int NR0 = 64;
#ifdef GGML_METAL_HAS_TENSOR
    // up to 64 entries per tile (FC_mul_mm_id_wide): an expert with many tokens dequantizes its weights half as often
    constexpr int NR1 = 64;
#else
    constexpr int NR1 = 32;
#endif

    constexpr int NK  = 32;
    constexpr int NL0 = NK/16;
    constexpr int NL1 = NK/8;

    const int tile = tgpig.x; // entry tile, described by the prepass
    const int r0   = tgpig.y*NR0;

    device const uint32_t * tpe_u32 = (device const uint32_t *) (htpe);
    device const int32_t  * ids_i32 = (device const int32_t  *) (hids);

    // the prepass emits one (expert, first entry, valid rows) descriptor per tile that has
    // work - the grid is sized by a static upper bound, so the tail early exits
    if (tile >= (int) tpe_u32[0]) {
        return;
    }

    const int im  = tpe_u32[1 + 3*tile];     // expert
    const int idb = tpe_u32[1 + 3*tile + 1]; // first entry of this tile in hids

    // number of entries in this tile, always <= NR1
    const short nr1 = (short) ((tpe_u32[1 + 3*tile + 2] < NR1) ? tpe_u32[1 + 3*tile + 2] : NR1);

    // if this block is of 64x32 shape or smaller
    const short nr0 = (args.ne0 - r0 < NR0) ? (args.ne0 - r0) : NR0;

    // a thread shouldn't load data outside of the matrix
    const short lr0 = ((short)tiitg/NL0) < nr0 ? ((short)tiitg/NL0) : nr0 - 1; // 0 .. 63

    const short il0 = (tiitg % NL0);

    const short i13 = 0;

    const uint64_t offset0 = im*args.nb02 + i13*args.nb03;
    const short    offset1 = il0/nl;

    const short iy = 8*(tiitg % NL1);


#ifndef GGML_METAL_HAS_TENSOR
    S0_8x8 ma[4];
    S1_8x8 mb[2];

    simdgroup_float8x8 mc[8];
#else
    auto tA  = tensor<threadgroup S0, dextents<int32_t, 2>, tensor_inline>(sa, dextents<int32_t, 2>(NK, NR0));

    // sb is [NR1][NK] row-major, multiplied in quarters of 16 rows - the quarters past the entries are skipped
    constexpr short NR1Q = NR1/4;

    auto tB0 = tensor<threadgroup S1, dextents<int32_t, 2>, tensor_inline>(sb,            dextents<int32_t, 2>(NK, NR1Q));
    auto tB1 = tensor<threadgroup S1, dextents<int32_t, 2>, tensor_inline>(sb + 1*NR1Q*NK, dextents<int32_t, 2>(NK, NR1Q));
    auto tB2 = tensor<threadgroup S1, dextents<int32_t, 2>, tensor_inline>(sb + 2*NR1Q*NK, dextents<int32_t, 2>(NK, NR1Q));
    auto tB3 = tensor<threadgroup S1, dextents<int32_t, 2>, tensor_inline>(sb + 3*NR1Q*NK, dextents<int32_t, 2>(NK, NR1Q));

    mpp::tensor_ops::matmul2d<
        mpp::tensor_ops::matmul2d_descriptor(NR1Q, NR0, NK, false, true, false, mpp::tensor_ops::matmul2d_descriptor::mode::multiply_accumulate),
        execution_simdgroups<4>> mm;

    // with wide tiles each thread also loads the src1 row lr1 + 32
    const short lr1b = ((short)tiitg/NL1 + 32) < nr1 ? ((short)tiitg/NL1 + 32) : nr1 - 1;

#endif

    const short lr1 = ((short)tiitg/NL1) < nr1 ? ((short)tiitg/NL1) : nr1 - 1; // 0 .. 31

    short il = il0;

    const int id = ids_i32[idb + lr1];

    const short i11 = (id % args.ne20) % args.ne11;
    const int   i12 = (id / args.ne20);

    device const block_q * x = (device const block_q *)(src0 + args.nb01*(r0 + lr0) + offset0) + offset1;

    device const T1 * y = (device const T1 *)(src1
        + args.nb13*i13
        + args.nb12*i12
        + args.nb11*i11
        + args.nb10*iy);

#ifndef GGML_METAL_HAS_TENSOR
    // skip the upper half of the token tile when the expert did not fill it
    constexpr short NR1H = NR1/2;

    const bool has_hi = nr1 > NR1H;

    for (short i = 0; i < 8; i++){
        mc[i] = make_filled_simdgroup_matrix<float, 8>(0.f);
    }

    // simdgroups 2,3 own rows NR1H..NR1-1
    const bool sg_active = has_hi || sgitg < 2;
#else
    const int id2 = FC_mul_mm_id_wide ? ids_i32[idb + lr1b] : 0;

    device const T1 * y2 = (device const T1 *)(src1
        + args.nb13*i13
        + args.nb12*(id2 / args.ne20)
        + args.nb11*((id2 % args.ne20) % args.ne11)
        + args.nb10*iy);

    auto cT0 = mm.get_destination_cooperative_tensor<decltype(tA), decltype(tB0), float>();
    auto cT1 = mm.get_destination_cooperative_tensor<decltype(tA), decltype(tB1), float>();
    auto cT2 = mm.get_destination_cooperative_tensor<decltype(tA), decltype(tB2), float>();
    auto cT3 = mm.get_destination_cooperative_tensor<decltype(tA), decltype(tB3), float>();
#endif

    for (int loop_k = 0; loop_k < args.ne00; loop_k += NK) {
#ifndef GGML_METAL_HAS_TENSOR
        // load data and store to threadgroup memory
        if (is_same<T0_4x4, block_q>::value && FC_mul_mm_bc_inp) {
            threadgroup_barrier(mem_flags::mem_threadgroup);

            // no need for dequantization
            for (short i = 0; i < 16; i++) {
                const short sx = 2*il0 + i/8;
                const short sy = (tiitg/NL0)/8;

              //const short lx = i%8;
              //const short ly = (tiitg/NL0)%8;
                const short lx = (tiitg/NL0)%8;
                const short ly = i%8;

                const short ib = 8*sx + sy;

                *(sa + 64*ib + 8*ly + lx) = loop_k + 16*il + i < args.ne00 ? (S0) *((device T0 *) x + i) : (S0) 0;
            }
        } else {
            S0_4x4 temp_a;
            dequantize_func(x, il, temp_a);

            threadgroup_barrier(mem_flags::mem_threadgroup);

            FOR_UNROLL (short i = 0; i < 16; i++) {
                const short sx = 2*il0 + i/8;
                const short sy = (tiitg/NL0)/8;

              //const short lx = i%8;
              //const short ly = (tiitg/NL0)%8;
                const short lx = (tiitg/NL0)%8;
                const short ly = i%8;

                const short ib = 8*sx + sy;

                // NOTE: this is massively slower.. WTF?
                //sa[64*ib + 8*ly + lx] = temp_a[i/4][i%4];

                *(sa + 64*ib + 8*ly + lx) = temp_a[i/4][i%4];
            }
        }

        if (FC_mul_mm_bc_inp) {
            for (short i = 0; i < 8; ++i) {
                const short sx = (tiitg%NL1);
                const short sy = (tiitg/NL1)/8;

                const short lx = i;
                const short ly = (tiitg/NL1)%8;
              //const short lx = (tiitg/NL1)%8;
              //const short ly = i;

                const short ib = 4*sx + sy;

                *(sb + 64*ib + 8*ly + lx) = loop_k + iy + i < args.ne00 ? (S1) *((device T1 *) y + i) : 0;
            }
        } else {
            const short sx = (tiitg%NL1);
            const short sy = (tiitg/NL1)/8;

          //const short dx = sx;
          //const short dy = sy;

            const short ly = (tiitg/NL1)%8;

            const short ib = 4*sx + sy;

            *(threadgroup S1_2x4 *)(sb + 64*ib + 8*ly) = (S1_2x4)(*((device T1_2x4 *) y));
        }
#else
        // load data and store to threadgroup memory
        if (is_same<T0_4x4, block_q>::value && FC_mul_mm_bc_inp) {
            threadgroup_barrier(mem_flags::mem_threadgroup);

            // no need for dequantization
            for (short i = 0; i < 16; i++) {
                const short sx = 2*il0 + i/8;
                const short sy = (tiitg/NL0)/8;

                const short lx = i%8;
                const short ly = (tiitg/NL0)%8;
                //const short lx = (tiitg/NL0)%8;
                //const short ly = i%8;

                *(sa + NK*(8*sy + ly) + 8*sx + lx) = loop_k + 16*il + i < args.ne00 ? (S0) *((device T0 *) x + i) : 0;
            }
        } else {
            S0_4x4 temp_a;
            dequantize_func(x, il, temp_a);

            threadgroup_barrier(mem_flags::mem_threadgroup);

            FOR_UNROLL (short i = 0; i < 16; i++) {
                const short sx = 2*il0 + i/8;
                const short sy = (tiitg/NL0)/8;

                const short lx = i%8;
                const short ly = (tiitg/NL0)%8;
                //const short lx = (tiitg/NL0)%8;
                //const short ly = i%8;

                *(sa + NK*(8*sy + ly) + 8*sx + lx) = temp_a[i/4][i%4];
            }
        }

        if (FC_mul_mm_bc_inp) {
            for (short i = 0; i < 8; ++i) {
                const short sx = (tiitg%NL1);
                const short sy = (tiitg/NL1)/8;

                const short lx = i;
                const short ly = (tiitg/NL1)%8;
                //const short lx = (tiitg/NL1)%8;
                //const short ly = i;

                *(sb + NK*(8*sy + ly) + 8*sx + lx)         = loop_k + iy + i < args.ne00 ? (S1) *((device T1 *) y  + i) : 0;
                if (FC_mul_mm_id_wide) {
                    *(sb + NK*(8*sy + ly + 32) + 8*sx + lx) = loop_k + iy + i < args.ne00 ? (S1) *((device T1 *) y2 + i) : 0;
                }
            }
        } else {
            const short sx = (tiitg%NL1);
            const short sy = (tiitg/NL1)/8;

            //const short lx = i;
            const short ly = (tiitg/NL1)%8;
            //const short lx = (tiitg/NL1)%8;
            //const short ly = i;

            *(threadgroup S1_2x4 *)(sb + NK*(8*sy + ly) + 8*sx)      = (S1_2x4)(*((device T1_2x4 *) y));
            if (FC_mul_mm_id_wide) {
                *(threadgroup S1_2x4 *)(sb + NK*(8*sy + ly + 32) + 8*sx) = (S1_2x4)(*((device T1_2x4 *) y2));
            }
        }

        y2 += NK;
#endif

        il = (il + 2 < nl) ? il + 2 : il % 2;
        x  = (il < 2) ? x + (2 + nl - 1)/nl : x;

        y += NK;

        threadgroup_barrier(mem_flags::mem_threadgroup);

#ifndef GGML_METAL_HAS_TENSOR
        if (sg_active) {
            // load matrices from threadgroup memory and conduct outer products
            threadgroup const S0 * lsma = (sa + 4*64*(sgitg%2));
            threadgroup const S1 * lsmb = (sb + 2*64*(sgitg/2));

            FOR_UNROLL (short ik = 0; ik < NK/8; ik++) {
                simdgroup_barrier(mem_flags::mem_none);

                FOR_UNROLL (short i = 0; i < 4; i++) {
                    simdgroup_load(ma[i], lsma + 64*i, 8, 0, false);
                }

                simdgroup_barrier(mem_flags::mem_none);

                FOR_UNROLL (short i = 0; i < 2; i++) {
                    simdgroup_load(mb[i], lsmb + 64*i, 8, 0, false);
                }

                simdgroup_barrier(mem_flags::mem_none);

                FOR_UNROLL (short i = 0; i < 8; i++){
                    simdgroup_multiply_accumulate(mc[i], mb[i/4], ma[i%4], mc[i]);
                }

                lsma += 8*64;
                lsmb += 4*64;
            }
        }
#else
        auto sA  = tA.slice(0, 0);
        auto sB0 = tB0.slice(0, 0);

        mm.run(sB0, sA, cT0);

        if (nr1 > 1*NR1Q) {
            auto sB1 = tB1.slice(0, 0);
            mm.run(sB1, sA, cT1);
        }
        if (nr1 > 2*NR1Q) {
            auto sB2 = tB2.slice(0, 0);
            mm.run(sB2, sA, cT2);
        }
        if (nr1 > 3*NR1Q) {
            auto sB3 = tB3.slice(0, 0);
            mm.run(sB3, sA, cT3);
        }
#endif
    }

    // block is smaller than 64x32, we should avoid writing data outside of the matrix
    threadgroup_barrier(mem_flags::mem_threadgroup);

#ifdef GGML_METAL_HAS_TENSOR
    auto tC0 = tensor<threadgroup float, dextents<int32_t, 2>, tensor_inline>(sc, dextents<int32_t, 2>(NR0, NR1Q));
    cT0.store(tC0);

    if (nr1 > 1*NR1Q) {
        auto tC1 = tensor<threadgroup float, dextents<int32_t, 2>, tensor_inline>(sc + 1*NR1Q*NR0, dextents<int32_t, 2>(NR0, NR1Q));
        cT1.store(tC1);
    }
    if (nr1 > 2*NR1Q) {
        auto tC2 = tensor<threadgroup float, dextents<int32_t, 2>, tensor_inline>(sc + 2*NR1Q*NR0, dextents<int32_t, 2>(NR0, NR1Q));
        cT2.store(tC2);
    }
    if (nr1 > 3*NR1Q) {
        auto tC3 = tensor<threadgroup float, dextents<int32_t, 2>, tensor_inline>(sc + 3*NR1Q*NR0, dextents<int32_t, 2>(NR0, NR1Q));
        cT3.store(tC3);
    }
#else
    if (sg_active) {
        threadgroup float * temp_str = ((threadgroup float *) shmem) + 32*(sgitg&1) + (16*(sgitg >> 1))*NR0;

        for (short i = 0; i < 8; i++) {
            simdgroup_store(mc[i], temp_str + 8*(i%4) + 8*NR0*(i/4), NR0, 0, false);
        }
    }
#endif

    threadgroup_barrier(mem_flags::mem_threadgroup);

    for (short j = sgitg; j < nr1; j += 4) {
        const int id = ids_i32[idb + j];

        const short ide = id % args.ne20;
        const int   idt = id / args.ne20;

        device float  * D  = (device float  *) dst + r0 + (uint64_t) ide*args.ne0 + (uint64_t) idt*args.ne1*args.ne0;
        device float4 * D4 = (device float4 *) D;

        threadgroup float  * C  = (threadgroup float  *) shmem + j*NR0;
        threadgroup float4 * C4 = (threadgroup float4 *) C;

        int i = tiisg;
        for (; i < nr0/4; i += 32) {
            *(D4 + i) = *(C4 + i);
        }

        i = (4*(nr0/4)) + tiisg;
        for (; i < nr0; i += 32) {
            *(D + i) = *(C + i);
        }
    }
}

//
// matrix-matrix multiplication
//

typedef decltype(kernel_mul_mm<half, half4x4, simdgroup_half8x8, half, half2x4, simdgroup_half8x8, float4x4, 1, dequantize_f32, float, float4x4, float, float2x4>) mul_mm_t;

template [[host_name("kernel_mul_mm_f32_f32")]]     kernel mul_mm_t kernel_mul_mm<half,   half4x4,   simdgroup_half8x8,   half,   half2x4,   simdgroup_half8x8,   float4x4,      1,     dequantize_f32,     float,  float4x4,  float, float2x4>;
template [[host_name("kernel_mul_mm_f16_f32")]]     kernel mul_mm_t kernel_mul_mm<half,   half4x4,   simdgroup_half8x8,   half,   half2x4,   simdgroup_half8x8,   half4x4,       1,     dequantize_f16,     half,   half4x4,   float, float2x4>;
#if defined(GGML_METAL_HAS_BF16)
template [[host_name("kernel_mul_mm_bf16_f32")]]    kernel mul_mm_t kernel_mul_mm<bfloat, bfloat4x4, simdgroup_bfloat8x8, bfloat, bfloat2x4, simdgroup_bfloat8x8, bfloat4x4,     1,     dequantize_bf16,    bfloat, bfloat4x4, float, float2x4>;
#endif
template [[host_name("kernel_mul_mm_q1_0_f32")]]    kernel mul_mm_t kernel_mul_mm<half,   half4x4,   simdgroup_half8x8,   half,   half2x4,   simdgroup_half8x8,   block_q1_0,    8,     dequantize_q1_0,    float,  float4x4,  float, float2x4>;
template [[host_name("kernel_mul_mm_q2_0_f32")]]    kernel mul_mm_t kernel_mul_mm<half,   half4x4,   simdgroup_half8x8,   half,   half2x4,   simdgroup_half8x8,   block_q2_0,    4,     dequantize_q2_0,    float,  float4x4,  float, float2x4>;
template [[host_name("kernel_mul_mm_q4_0_f32")]]    kernel mul_mm_t kernel_mul_mm<half,   half4x4,   simdgroup_half8x8,   half,   half2x4,   simdgroup_half8x8,   block_q4_0,    2,     dequantize_q4_0,    float,  float4x4,  float, float2x4>;
template [[host_name("kernel_mul_mm_q4_1_f32")]]    kernel mul_mm_t kernel_mul_mm<half,   half4x4,   simdgroup_half8x8,   half,   half2x4,   simdgroup_half8x8,   block_q4_1,    2,     dequantize_q4_1,    float,  float4x4,  float, float2x4>;
template [[host_name("kernel_mul_mm_q5_0_f32")]]    kernel mul_mm_t kernel_mul_mm<half,   half4x4,   simdgroup_half8x8,   half,   half2x4,   simdgroup_half8x8,   block_q5_0,    2,     dequantize_q5_0,    float,  float4x4,  float, float2x4>;
template [[host_name("kernel_mul_mm_q5_1_f32")]]    kernel mul_mm_t kernel_mul_mm<half,   half4x4,   simdgroup_half8x8,   half,   half2x4,   simdgroup_half8x8,   block_q5_1,    2,     dequantize_q5_1,    float,  float4x4,  float, float2x4>;
template [[host_name("kernel_mul_mm_q8_0_f32")]]    kernel mul_mm_t kernel_mul_mm<half,   half4x4,   simdgroup_half8x8,   half,   half2x4,   simdgroup_half8x8,   block_q8_0,    2,     dequantize_q8_0,    float,  float4x4,  float, float2x4>;
template [[host_name("kernel_mul_mm_mxfp4_f32")]]   kernel mul_mm_t kernel_mul_mm<half,   half4x4,   simdgroup_half8x8,   half,   half2x4,   simdgroup_half8x8,   block_mxfp4,   2,     dequantize_mxfp4,   float,  float4x4,  float, float2x4>;
template [[host_name("kernel_mul_mm_q2_K_f32")]]    kernel mul_mm_t kernel_mul_mm<half,   half4x4,   simdgroup_half8x8,   half,   half2x4,   simdgroup_half8x8,   block_q2_K,    QK_NL, dequantize_q2_K,    float,  float4x4,  float, float2x4>;
template [[host_name("kernel_mul_mm_q3_K_f32")]]    kernel mul_mm_t kernel_mul_mm<half,   half4x4,   simdgroup_half8x8,   half,   half2x4,   simdgroup_half8x8,   block_q3_K,    QK_NL, dequantize_q3_K,    float,  float4x4,  float, float2x4>;
template [[host_name("kernel_mul_mm_q4_K_f32")]]    kernel mul_mm_t kernel_mul_mm<half,   half4x4,   simdgroup_half8x8,   half,   half2x4,   simdgroup_half8x8,   block_q4_K,    QK_NL, dequantize_q4_K,    float,  float4x4,  float, float2x4>;
template [[host_name("kernel_mul_mm_q5_K_f32")]]    kernel mul_mm_t kernel_mul_mm<half,   half4x4,   simdgroup_half8x8,   half,   half2x4,   simdgroup_half8x8,   block_q5_K,    QK_NL, dequantize_q5_K,    float,  float4x4,  float, float2x4>;
template [[host_name("kernel_mul_mm_q6_K_f32")]]    kernel mul_mm_t kernel_mul_mm<half,   half4x4,   simdgroup_half8x8,   half,   half2x4,   simdgroup_half8x8,   block_q6_K,    QK_NL, dequantize_q6_K,    float,  float4x4,  float, float2x4>;
template [[host_name("kernel_mul_mm_iq2_xxs_f32")]] kernel mul_mm_t kernel_mul_mm<half,   half4x4,   simdgroup_half8x8,   half,   half2x4,   simdgroup_half8x8,   block_iq2_xxs, QK_NL, dequantize_iq2_xxs, float,  float4x4,  float, float2x4>;
template [[host_name("kernel_mul_mm_iq2_xs_f32")]]  kernel mul_mm_t kernel_mul_mm<half,   half4x4,   simdgroup_half8x8,   half,   half2x4,   simdgroup_half8x8,   block_iq2_xs,  QK_NL, dequantize_iq2_xs,  float,  float4x4,  float, float2x4>;
template [[host_name("kernel_mul_mm_iq3_xxs_f32")]] kernel mul_mm_t kernel_mul_mm<half,   half4x4,   simdgroup_half8x8,   half,   half2x4,   simdgroup_half8x8,   block_iq3_xxs, QK_NL, dequantize_iq3_xxs, float,  float4x4,  float, float2x4>;
template [[host_name("kernel_mul_mm_iq3_s_f32")]]   kernel mul_mm_t kernel_mul_mm<half,   half4x4,   simdgroup_half8x8,   half,   half2x4,   simdgroup_half8x8,   block_iq3_s,   QK_NL, dequantize_iq3_s,   float,  float4x4,  float, float2x4>;
template [[host_name("kernel_mul_mm_iq2_s_f32")]]   kernel mul_mm_t kernel_mul_mm<half,   half4x4,   simdgroup_half8x8,   half,   half2x4,   simdgroup_half8x8,   block_iq2_s,   QK_NL, dequantize_iq2_s,   float,  float4x4,  float, float2x4>;
template [[host_name("kernel_mul_mm_iq1_s_f32")]]   kernel mul_mm_t kernel_mul_mm<half,   half4x4,   simdgroup_half8x8,   half,   half2x4,   simdgroup_half8x8,   block_iq1_s,   QK_NL, dequantize_iq1_s,   float,  float4x4,  float, float2x4>;
template [[host_name("kernel_mul_mm_iq1_m_f32")]]   kernel mul_mm_t kernel_mul_mm<half,   half4x4,   simdgroup_half8x8,   half,   half2x4,   simdgroup_half8x8,   block_iq1_m,   QK_NL, dequantize_iq1_m,   float,  float4x4,  float, float2x4>;
template [[host_name("kernel_mul_mm_iq4_nl_f32")]]  kernel mul_mm_t kernel_mul_mm<half,   half4x4,   simdgroup_half8x8,   half,   half2x4,   simdgroup_half8x8,   block_iq4_nl,  2,     dequantize_iq4_nl,  float,  float4x4,  float, float2x4>;
template [[host_name("kernel_mul_mm_iq4_xs_f32")]]  kernel mul_mm_t kernel_mul_mm<half,   half4x4,   simdgroup_half8x8,   half,   half2x4,   simdgroup_half8x8,   block_iq4_xs,  QK_NL, dequantize_iq4_xs,  float,  float4x4,  float, float2x4>;
template [[host_name("kernel_mul_mm_tq2_0_f32")]]   kernel mul_mm_t kernel_mul_mm<half,   half4x4,   simdgroup_half8x8,   half,   half2x4,   simdgroup_half8x8,   block_tq2_0,   QK_NL, dequantize_tq2_0,   float,  float4x4,  float, float2x4>;

template [[host_name("kernel_mul_mm_f32_f16")]]     kernel mul_mm_t kernel_mul_mm<half,   half4x4,   simdgroup_half8x8,   half,   half2x4,   simdgroup_half8x8,   float4x4,      1,     dequantize_f32,     float,  float4x4,  half, half2x4>;
template [[host_name("kernel_mul_mm_f16_f16")]]     kernel mul_mm_t kernel_mul_mm<half,   half4x4,   simdgroup_half8x8,   half,   half2x4,   simdgroup_half8x8,   half4x4,       1,     dequantize_f16,     half,   half4x4,   half, half2x4>;
template [[host_name("kernel_mul_mm_q1_0_f16")]]    kernel mul_mm_t kernel_mul_mm<half,   half4x4,   simdgroup_half8x8,   half,   half2x4,   simdgroup_half8x8,   block_q1_0,    8,     dequantize_q1_0,    float,  float4x4,  half, half2x4>;
template [[host_name("kernel_mul_mm_q2_0_f16")]]    kernel mul_mm_t kernel_mul_mm<half,   half4x4,   simdgroup_half8x8,   half,   half2x4,   simdgroup_half8x8,   block_q2_0,    4,     dequantize_q2_0,    float,  float4x4,  half, half2x4>;
template [[host_name("kernel_mul_mm_q4_0_f16")]]    kernel mul_mm_t kernel_mul_mm<half,   half4x4,   simdgroup_half8x8,   half,   half2x4,   simdgroup_half8x8,   block_q4_0,    2,     dequantize_q4_0,    float,  float4x4,  half, half2x4>;
template [[host_name("kernel_mul_mm_q4_1_f16")]]    kernel mul_mm_t kernel_mul_mm<half,   half4x4,   simdgroup_half8x8,   half,   half2x4,   simdgroup_half8x8,   block_q4_1,    2,     dequantize_q4_1,    float,  float4x4,  half, half2x4>;
template [[host_name("kernel_mul_mm_q5_0_f16")]]    kernel mul_mm_t kernel_mul_mm<half,   half4x4,   simdgroup_half8x8,   half,   half2x4,   simdgroup_half8x8,   block_q5_0,    2,     dequantize_q5_0,    float,  float4x4,  half, half2x4>;
template [[host_name("kernel_mul_mm_q5_1_f16")]]    kernel mul_mm_t kernel_mul_mm<half,   half4x4,   simdgroup_half8x8,   half,   half2x4,   simdgroup_half8x8,   block_q5_1,    2,     dequantize_q5_1,    float,  float4x4,  half, half2x4>;
template [[host_name("kernel_mul_mm_q8_0_f16")]]    kernel mul_mm_t kernel_mul_mm<half,   half4x4,   simdgroup_half8x8,   half,   half2x4,   simdgroup_half8x8,   block_q8_0,    2,     dequantize_q8_0,    float,  float4x4,  half, half2x4>;
template [[host_name("kernel_mul_mm_mxfp4_f16")]]   kernel mul_mm_t kernel_mul_mm<half,   half4x4,   simdgroup_half8x8,   half,   half2x4,   simdgroup_half8x8,   block_mxfp4,   2,     dequantize_mxfp4,   float,  float4x4,  half, half2x4>;
template [[host_name("kernel_mul_mm_q2_K_f16")]]    kernel mul_mm_t kernel_mul_mm<half,   half4x4,   simdgroup_half8x8,   half,   half2x4,   simdgroup_half8x8,   block_q2_K,    QK_NL, dequantize_q2_K,    float,  float4x4,  half, half2x4>;
template [[host_name("kernel_mul_mm_q3_K_f16")]]    kernel mul_mm_t kernel_mul_mm<half,   half4x4,   simdgroup_half8x8,   half,   half2x4,   simdgroup_half8x8,   block_q3_K,    QK_NL, dequantize_q3_K,    float,  float4x4,  half, half2x4>;
template [[host_name("kernel_mul_mm_q4_K_f16")]]    kernel mul_mm_t kernel_mul_mm<half,   half4x4,   simdgroup_half8x8,   half,   half2x4,   simdgroup_half8x8,   block_q4_K,    QK_NL, dequantize_q4_K,    float,  float4x4,  half, half2x4>;
template [[host_name("kernel_mul_mm_q5_K_f16")]]    kernel mul_mm_t kernel_mul_mm<half,   half4x4,   simdgroup_half8x8,   half,   half2x4,   simdgroup_half8x8,   block_q5_K,    QK_NL, dequantize_q5_K,    float,  float4x4,  half, half2x4>;
template [[host_name("kernel_mul_mm_q6_K_f16")]]    kernel mul_mm_t kernel_mul_mm<half,   half4x4,   simdgroup_half8x8,   half,   half2x4,   simdgroup_half8x8,   block_q6_K,    QK_NL, dequantize_q6_K,    float,  float4x4,  half, half2x4>;
template [[host_name("kernel_mul_mm_iq2_xxs_f16")]] kernel mul_mm_t kernel_mul_mm<half,   half4x4,   simdgroup_half8x8,   half,   half2x4,   simdgroup_half8x8,   block_iq2_xxs, QK_NL, dequantize_iq2_xxs, float,  float4x4,  half, half2x4>;
template [[host_name("kernel_mul_mm_iq2_xs_f16")]]  kernel mul_mm_t kernel_mul_mm<half,   half4x4,   simdgroup_half8x8,   half,   half2x4,   simdgroup_half8x8,   block_iq2_xs,  QK_NL, dequantize_iq2_xs,  float,  float4x4,  half, half2x4>;
template [[host_name("kernel_mul_mm_iq3_xxs_f16")]] kernel mul_mm_t kernel_mul_mm<half,   half4x4,   simdgroup_half8x8,   half,   half2x4,   simdgroup_half8x8,   block_iq3_xxs, QK_NL, dequantize_iq3_xxs, float,  float4x4,  half, half2x4>;
template [[host_name("kernel_mul_mm_iq3_s_f16")]]   kernel mul_mm_t kernel_mul_mm<half,   half4x4,   simdgroup_half8x8,   half,   half2x4,   simdgroup_half8x8,   block_iq3_s,   QK_NL, dequantize_iq3_s,   float,  float4x4,  half, half2x4>;
template [[host_name("kernel_mul_mm_iq2_s_f16")]]   kernel mul_mm_t kernel_mul_mm<half,   half4x4,   simdgroup_half8x8,   half,   half2x4,   simdgroup_half8x8,   block_iq2_s,   QK_NL, dequantize_iq2_s,   float,  float4x4,  half, half2x4>;
template [[host_name("kernel_mul_mm_iq1_s_f16")]]   kernel mul_mm_t kernel_mul_mm<half,   half4x4,   simdgroup_half8x8,   half,   half2x4,   simdgroup_half8x8,   block_iq1_s,   QK_NL, dequantize_iq1_s,   float,  float4x4,  half, half2x4>;
template [[host_name("kernel_mul_mm_iq1_m_f16")]]   kernel mul_mm_t kernel_mul_mm<half,   half4x4,   simdgroup_half8x8,   half,   half2x4,   simdgroup_half8x8,   block_iq1_m,   QK_NL, dequantize_iq1_m,   float,  float4x4,  half, half2x4>;
template [[host_name("kernel_mul_mm_iq4_nl_f16")]]  kernel mul_mm_t kernel_mul_mm<half,   half4x4,   simdgroup_half8x8,   half,   half2x4,   simdgroup_half8x8,   block_iq4_nl,  2,     dequantize_iq4_nl,  float,  float4x4,  half, half2x4>;
template [[host_name("kernel_mul_mm_iq4_xs_f16")]]  kernel mul_mm_t kernel_mul_mm<half,   half4x4,   simdgroup_half8x8,   half,   half2x4,   simdgroup_half8x8,   block_iq4_xs,  QK_NL, dequantize_iq4_xs,  float,  float4x4,  half, half2x4>;
template [[host_name("kernel_mul_mm_tq2_0_f16")]]   kernel mul_mm_t kernel_mul_mm<half,   half4x4,   simdgroup_half8x8,   half,   half2x4,   simdgroup_half8x8,   block_tq2_0,   QK_NL, dequantize_tq2_0,   float,  float4x4,  half, half2x4>;

//
// indirect matrix-matrix multiplication
//

typedef decltype(kernel_mul_mm_id<half, half4x4, simdgroup_half8x8, half, half2x4, simdgroup_half8x8, float4x4, 1, dequantize_f32, float, float4x4, float, float2x4>) mul_mm_id;

template [[host_name("kernel_mul_mm_id_f32_f32")]]     kernel mul_mm_id kernel_mul_mm_id<half,   half4x4,   simdgroup_half8x8,   half,   half2x4,   simdgroup_half8x8,   float4x4,      1,     dequantize_f32,     float,  float4x4,  float, float2x4>;
template [[host_name("kernel_mul_mm_id_f16_f32")]]     kernel mul_mm_id kernel_mul_mm_id<half,   half4x4,   simdgroup_half8x8,   half,   half2x4,   simdgroup_half8x8,   half4x4,       1,     dequantize_f16,     half,   half4x4,   float, float2x4>;
#if defined(GGML_METAL_HAS_BF16)
template [[host_name("kernel_mul_mm_id_bf16_f32")]]    kernel mul_mm_id kernel_mul_mm_id<bfloat, bfloat4x4, simdgroup_bfloat8x8, bfloat, bfloat2x4, simdgroup_bfloat8x8, bfloat4x4,     1,     dequantize_bf16,    bfloat, bfloat4x4, float, float2x4>;
#endif
template [[host_name("kernel_mul_mm_id_q1_0_f32")]]    kernel mul_mm_id kernel_mul_mm_id<half,   half4x4,   simdgroup_half8x8,   half,   half2x4,   simdgroup_half8x8,   block_q1_0,    8,     dequantize_q1_0,    float,  float4x4,  float, float2x4>;
template [[host_name("kernel_mul_mm_id_q2_0_f32")]]    kernel mul_mm_id kernel_mul_mm_id<half,   half4x4,   simdgroup_half8x8,   half,   half2x4,   simdgroup_half8x8,   block_q2_0,    4,     dequantize_q2_0,    float,  float4x4,  float, float2x4>;
template [[host_name("kernel_mul_mm_id_q4_0_f32")]]    kernel mul_mm_id kernel_mul_mm_id<half,   half4x4,   simdgroup_half8x8,   half,   half2x4,   simdgroup_half8x8,   block_q4_0,    2,     dequantize_q4_0,    float,  float4x4,  float, float2x4>;
template [[host_name("kernel_mul_mm_id_q4_1_f32")]]    kernel mul_mm_id kernel_mul_mm_id<half,   half4x4,   simdgroup_half8x8,   half,   half2x4,   simdgroup_half8x8,   block_q4_1,    2,     dequantize_q4_1,    float,  float4x4,  float, float2x4>;
template [[host_name("kernel_mul_mm_id_q5_0_f32")]]    kernel mul_mm_id kernel_mul_mm_id<half,   half4x4,   simdgroup_half8x8,   half,   half2x4,   simdgroup_half8x8,   block_q5_0,    2,     dequantize_q5_0,    float,  float4x4,  float, float2x4>;
template [[host_name("kernel_mul_mm_id_q5_1_f32")]]    kernel mul_mm_id kernel_mul_mm_id<half,   half4x4,   simdgroup_half8x8,   half,   half2x4,   simdgroup_half8x8,   block_q5_1,    2,     dequantize_q5_1,    float,  float4x4,  float, float2x4>;
template [[host_name("kernel_mul_mm_id_q8_0_f32")]]    kernel mul_mm_id kernel_mul_mm_id<half,   half4x4,   simdgroup_half8x8,   half,   half2x4,   simdgroup_half8x8,   block_q8_0,    2,     dequantize_q8_0,    float,  float4x4,  float, float2x4>;
#if defined(GGML_METAL_HAS_BF16)
template [[host_name("kernel_mul_mm_id_mxfp4_f32")]]   kernel mul_mm_id kernel_mul_mm_id<bfloat, bfloat4x4, simdgroup_bfloat8x8, bfloat, bfloat2x4, simdgroup_bfloat8x8, block_mxfp4,   2,     dequantize_mxfp4,   float,  float4x4,  float, float2x4>;
#else
template [[host_name("kernel_mul_mm_id_mxfp4_f32")]]   kernel mul_mm_id kernel_mul_mm_id<half,   half4x4,   simdgroup_half8x8,   half,   half2x4,   simdgroup_half8x8,   block_mxfp4,   2,     dequantize_mxfp4,   float,  float4x4,  float, float2x4>;
#endif
template [[host_name("kernel_mul_mm_id_q2_K_f32")]]    kernel mul_mm_id kernel_mul_mm_id<half,   half4x4,   simdgroup_half8x8,   half,   half2x4,   simdgroup_half8x8,   block_q2_K,    QK_NL, dequantize_q2_K,    float,  float4x4,  float, float2x4>;
template [[host_name("kernel_mul_mm_id_q3_K_f32")]]    kernel mul_mm_id kernel_mul_mm_id<half,   half4x4,   simdgroup_half8x8,   half,   half2x4,   simdgroup_half8x8,   block_q3_K,    QK_NL, dequantize_q3_K,    float,  float4x4,  float, float2x4>;
template [[host_name("kernel_mul_mm_id_q4_K_f32")]]    kernel mul_mm_id kernel_mul_mm_id<half,   half4x4,   simdgroup_half8x8,   half,   half2x4,   simdgroup_half8x8,   block_q4_K,    QK_NL, dequantize_q4_K,    float,  float4x4,  float, float2x4>;
template [[host_name("kernel_mul_mm_id_q5_K_f32")]]    kernel mul_mm_id kernel_mul_mm_id<half,   half4x4,   simdgroup_half8x8,   half,   half2x4,   simdgroup_half8x8,   block_q5_K,    QK_NL, dequantize_q5_K,    float,  float4x4,  float, float2x4>;
template [[host_name("kernel_mul_mm_id_q6_K_f32")]]    kernel mul_mm_id kernel_mul_mm_id<half,   half4x4,   simdgroup_half8x8,   half,   half2x4,   simdgroup_half8x8,   block_q6_K,    QK_NL, dequantize_q6_K,    float,  float4x4,  float, float2x4>;
template [[host_name("kernel_mul_mm_id_iq2_xxs_f32")]] kernel mul_mm_id kernel_mul_mm_id<half,   half4x4,   simdgroup_half8x8,   half,   half2x4,   simdgroup_half8x8,   block_iq2_xxs, QK_NL, dequantize_iq2_xxs, float,  float4x4,  float, float2x4>;
template [[host_name("kernel_mul_mm_id_iq2_xs_f32")]]  kernel mul_mm_id kernel_mul_mm_id<half,   half4x4,   simdgroup_half8x8,   half,   half2x4,   simdgroup_half8x8,   block_iq2_xs,  QK_NL, dequantize_iq2_xs,  float,  float4x4,  float, float2x4>;
template [[host_name("kernel_mul_mm_id_iq3_xxs_f32")]] kernel mul_mm_id kernel_mul_mm_id<half,   half4x4,   simdgroup_half8x8,   half,   half2x4,   simdgroup_half8x8,   block_iq3_xxs, QK_NL, dequantize_iq3_xxs, float,  float4x4,  float, float2x4>;
template [[host_name("kernel_mul_mm_id_iq3_s_f32")]]   kernel mul_mm_id kernel_mul_mm_id<half,   half4x4,   simdgroup_half8x8,   half,   half2x4,   simdgroup_half8x8,   block_iq3_s,   QK_NL, dequantize_iq3_s,   float,  float4x4,  float, float2x4>;
template [[host_name("kernel_mul_mm_id_iq2_s_f32")]]   kernel mul_mm_id kernel_mul_mm_id<half,   half4x4,   simdgroup_half8x8,   half,   half2x4,   simdgroup_half8x8,   block_iq2_s,   QK_NL, dequantize_iq2_s,   float,  float4x4,  float, float2x4>;
template [[host_name("kernel_mul_mm_id_iq1_s_f32")]]   kernel mul_mm_id kernel_mul_mm_id<half,   half4x4,   simdgroup_half8x8,   half,   half2x4,   simdgroup_half8x8,   block_iq1_s,   QK_NL, dequantize_iq1_s,   float,  float4x4,  float, float2x4>;
template [[host_name("kernel_mul_mm_id_iq1_m_f32")]]   kernel mul_mm_id kernel_mul_mm_id<half,   half4x4,   simdgroup_half8x8,   half,   half2x4,   simdgroup_half8x8,   block_iq1_m,   QK_NL, dequantize_iq1_m,   float,  float4x4,  float, float2x4>;
template [[host_name("kernel_mul_mm_id_iq4_nl_f32")]]  kernel mul_mm_id kernel_mul_mm_id<half,   half4x4,   simdgroup_half8x8,   half,   half2x4,   simdgroup_half8x8,   block_iq4_nl,  2,     dequantize_iq4_nl,  float,  float4x4,  float, float2x4>;
template [[host_name("kernel_mul_mm_id_iq4_xs_f32")]]  kernel mul_mm_id kernel_mul_mm_id<half,   half4x4,   simdgroup_half8x8,   half,   half2x4,   simdgroup_half8x8,   block_iq4_xs,  QK_NL, dequantize_iq4_xs,  float,  float4x4,  float, float2x4>;
template [[host_name("kernel_mul_mm_id_tq2_0_f32")]]   kernel mul_mm_id kernel_mul_mm_id<half,   half4x4,   simdgroup_half8x8,   half,   half2x4,   simdgroup_half8x8,   block_tq2_0,   QK_NL, dequantize_tq2_0,   float,  float4x4,  float, float2x4>;

template [[host_name("kernel_mul_mm_id_f32_f16")]]     kernel mul_mm_id kernel_mul_mm_id<half,   half4x4,   simdgroup_half8x8,   half,   half2x4,   simdgroup_half8x8,   float4x4,      1,     dequantize_f32,     float,  float4x4,  half, half2x4>;
template [[host_name("kernel_mul_mm_id_f16_f16")]]     kernel mul_mm_id kernel_mul_mm_id<half,   half4x4,   simdgroup_half8x8,   half,   half2x4,   simdgroup_half8x8,   half4x4,       1,     dequantize_f16,     half,   half4x4,   half, half2x4>;
template [[host_name("kernel_mul_mm_id_q1_0_f16")]]    kernel mul_mm_id kernel_mul_mm_id<half,   half4x4,   simdgroup_half8x8,   half,   half2x4,   simdgroup_half8x8,   block_q1_0,    8,     dequantize_q1_0,    float,  float4x4,  half, half2x4>;
template [[host_name("kernel_mul_mm_id_q2_0_f16")]]    kernel mul_mm_id kernel_mul_mm_id<half,   half4x4,   simdgroup_half8x8,   half,   half2x4,   simdgroup_half8x8,   block_q2_0,    4,     dequantize_q2_0,    float,  float4x4,  half, half2x4>;
template [[host_name("kernel_mul_mm_id_q4_0_f16")]]    kernel mul_mm_id kernel_mul_mm_id<half,   half4x4,   simdgroup_half8x8,   half,   half2x4,   simdgroup_half8x8,   block_q4_0,    2,     dequantize_q4_0,    float,  float4x4,  half, half2x4>;
template [[host_name("kernel_mul_mm_id_q4_1_f16")]]    kernel mul_mm_id kernel_mul_mm_id<half,   half4x4,   simdgroup_half8x8,   half,   half2x4,   simdgroup_half8x8,   block_q4_1,    2,     dequantize_q4_1,    float,  float4x4,  half, half2x4>;
template [[host_name("kernel_mul_mm_id_q5_0_f16")]]    kernel mul_mm_id kernel_mul_mm_id<half,   half4x4,   simdgroup_half8x8,   half,   half2x4,   simdgroup_half8x8,   block_q5_0,    2,     dequantize_q5_0,    float,  float4x4,  half, half2x4>;
template [[host_name("kernel_mul_mm_id_q5_1_f16")]]    kernel mul_mm_id kernel_mul_mm_id<half,   half4x4,   simdgroup_half8x8,   half,   half2x4,   simdgroup_half8x8,   block_q5_1,    2,     dequantize_q5_1,    float,  float4x4,  half, half2x4>;
template [[host_name("kernel_mul_mm_id_q8_0_f16")]]    kernel mul_mm_id kernel_mul_mm_id<half,   half4x4,   simdgroup_half8x8,   half,   half2x4,   simdgroup_half8x8,   block_q8_0,    2,     dequantize_q8_0,    float,  float4x4,  half, half2x4>;
#if defined(GGML_METAL_HAS_BF16)
template [[host_name("kernel_mul_mm_id_mxfp4_f16")]]   kernel mul_mm_id kernel_mul_mm_id<bfloat, bfloat4x4, simdgroup_bfloat8x8, bfloat, bfloat2x4, simdgroup_bfloat8x8, block_mxfp4,   2,     dequantize_mxfp4,   float,  float4x4,  half, half2x4>;
#else
template [[host_name("kernel_mul_mm_id_mxfp4_f16")]]   kernel mul_mm_id kernel_mul_mm_id<half,   half4x4,   simdgroup_half8x8,   half,   half2x4,   simdgroup_half8x8,   block_mxfp4,   2,     dequantize_mxfp4,   float,  float4x4,  half, half2x4>;
#endif
template [[host_name("kernel_mul_mm_id_q2_K_f16")]]    kernel mul_mm_id kernel_mul_mm_id<half,   half4x4,   simdgroup_half8x8,   half,   half2x4,   simdgroup_half8x8,   block_q2_K,    QK_NL, dequantize_q2_K,    float,  float4x4,  half, half2x4>;
template [[host_name("kernel_mul_mm_id_q3_K_f16")]]    kernel mul_mm_id kernel_mul_mm_id<half,   half4x4,   simdgroup_half8x8,   half,   half2x4,   simdgroup_half8x8,   block_q3_K,    QK_NL, dequantize_q3_K,    float,  float4x4,  half, half2x4>;
template [[host_name("kernel_mul_mm_id_q4_K_f16")]]    kernel mul_mm_id kernel_mul_mm_id<half,   half4x4,   simdgroup_half8x8,   half,   half2x4,   simdgroup_half8x8,   block_q4_K,    QK_NL, dequantize_q4_K,    float,  float4x4,  half, half2x4>;
template [[host_name("kernel_mul_mm_id_q5_K_f16")]]    kernel mul_mm_id kernel_mul_mm_id<half,   half4x4,   simdgroup_half8x8,   half,   half2x4,   simdgroup_half8x8,   block_q5_K,    QK_NL, dequantize_q5_K,    float,  float4x4,  half, half2x4>;
template [[host_name("kernel_mul_mm_id_q6_K_f16")]]    kernel mul_mm_id kernel_mul_mm_id<half,   half4x4,   simdgroup_half8x8,   half,   half2x4,   simdgroup_half8x8,   block_q6_K,    QK_NL, dequantize_q6_K,    float,  float4x4,  half, half2x4>;
template [[host_name("kernel_mul_mm_id_iq2_xxs_f16")]] kernel mul_mm_id kernel_mul_mm_id<half,   half4x4,   simdgroup_half8x8,   half,   half2x4,   simdgroup_half8x8,   block_iq2_xxs, QK_NL, dequantize_iq2_xxs, float,  float4x4,  half, half2x4>;
template [[host_name("kernel_mul_mm_id_iq2_xs_f16")]]  kernel mul_mm_id kernel_mul_mm_id<half,   half4x4,   simdgroup_half8x8,   half,   half2x4,   simdgroup_half8x8,   block_iq2_xs,  QK_NL, dequantize_iq2_xs,  float,  float4x4,  half, half2x4>;
template [[host_name("kernel_mul_mm_id_iq3_xxs_f16")]] kernel mul_mm_id kernel_mul_mm_id<half,   half4x4,   simdgroup_half8x8,   half,   half2x4,   simdgroup_half8x8,   block_iq3_xxs, QK_NL, dequantize_iq3_xxs, float,  float4x4,  half, half2x4>;
template [[host_name("kernel_mul_mm_id_iq3_s_f16")]]   kernel mul_mm_id kernel_mul_mm_id<half,   half4x4,   simdgroup_half8x8,   half,   half2x4,   simdgroup_half8x8,   block_iq3_s,   QK_NL, dequantize_iq3_s,   float,  float4x4,  half, half2x4>;
template [[host_name("kernel_mul_mm_id_iq2_s_f16")]]   kernel mul_mm_id kernel_mul_mm_id<half,   half4x4,   simdgroup_half8x8,   half,   half2x4,   simdgroup_half8x8,   block_iq2_s,   QK_NL, dequantize_iq2_s,   float,  float4x4,  half, half2x4>;
template [[host_name("kernel_mul_mm_id_iq1_s_f16")]]   kernel mul_mm_id kernel_mul_mm_id<half,   half4x4,   simdgroup_half8x8,   half,   half2x4,   simdgroup_half8x8,   block_iq1_s,   QK_NL, dequantize_iq1_s,   float,  float4x4,  half, half2x4>;
template [[host_name("kernel_mul_mm_id_iq1_m_f16")]]   kernel mul_mm_id kernel_mul_mm_id<half,   half4x4,   simdgroup_half8x8,   half,   half2x4,   simdgroup_half8x8,   block_iq1_m,   QK_NL, dequantize_iq1_m,   float,  float4x4,  half, half2x4>;
template [[host_name("kernel_mul_mm_id_iq4_nl_f16")]]  kernel mul_mm_id kernel_mul_mm_id<half,   half4x4,   simdgroup_half8x8,   half,   half2x4,   simdgroup_half8x8,   block_iq4_nl,  2,     dequantize_iq4_nl,  float,  float4x4,  half, half2x4>;
template [[host_name("kernel_mul_mm_id_iq4_xs_f16")]]  kernel mul_mm_id kernel_mul_mm_id<half,   half4x4,   simdgroup_half8x8,   half,   half2x4,   simdgroup_half8x8,   block_iq4_xs,  QK_NL, dequantize_iq4_xs,  float,  float4x4,  half, half2x4>;
template [[host_name("kernel_mul_mm_id_tq2_0_f16")]]   kernel mul_mm_id kernel_mul_mm_id<half,   half4x4,   simdgroup_half8x8,   half,   half2x4,   simdgroup_half8x8,   block_tq2_0,   QK_NL, dequantize_tq2_0,   float,  float4x4,  half, half2x4>;

#ifdef GGML_METAL_HAS_TENSOR
// few-row mat-mat on the NAX units: the src1 rows are the M rows of a 16x32x16 matmul2d and 32 weight rows are its N columns
// each lane dequantizes its weights straight into a half fragment; the NSG simdgroups split K and are summed at the end

// the 16 weights of dequantize_q5_K, with 16-byte loads and bit operations instead of a branch per weight
template <typename type4x4>
void dequantize_q5_K_nax(device const block_q5_K * xb, short il, thread type4x4 & reg) {
    const uint4 qv = *((device const uint4 *) (xb->qs + 32*(il/4) + 16*(il&1)));
    const uint4 hv = *((device const uint4 *) (xb->qh + 16*(il&1)));

    const short hb = il/2;
    const short is = (il/4)*2;
    il = il & 3;

    const uchar2 sc = get_scale_min_k4_just2(is, il/2, xb->scales);
    const float dl = (float) xb->d    * sc[0];
    const float ml = (float) xb->dmin * sc[1];

    const short sh = il < 2 ? 0 : 4;

    FOR_UNROLL (short k = 0; k < 4; ++k) {
        const uint v = ((qv[k] >> sh) & 0x0F0F0F0F) | (((hv[k] >> hb) & 0x01010101) << 4);
        reg[k][0] = dl*(float) ((v      ) & 0xFF) - ml;
        reg[k][1] = dl*(float) ((v >>  8) & 0xFF) - ml;
        reg[k][2] = dl*(float) ((v >> 16) & 0xFF) - ml;
        reg[k][3] = dl*(float) ((v >> 24)       ) - ml;
    }
}

// K is permuted the same way for src0 and src1 inside each 64-weight chunk, logical (s, fn + jj) -> physical 16*(fn/4) + 4*s + jj
// so that each lane dequantizes whole 16-weight groups, the unit of the dequantize functions
template<typename block_q, short nl, void (*deq)(device const block_q *, short, thread half4x4 &), short MR, bool ADD>
kernel void kernel_mul_mv_nax_f32(
        constant ggml_metal_kargs_mul_mv_ext & args,
        device const char * src0,
        device const char * src1,
        device       char * dst,
        device const char * src2,
        threadgroup  char * shmem [[threadgroup(0)]],
        uint3  tgpig [[threadgroup_position_in_grid]],
        ushort tiisg [[thread_index_in_simdgroup]],
        ushort sgitg [[simdgroup_index_in_threadgroup]],
        ushort3 ntg  [[threads_per_threadgroup]]) {
    const short NSG = ntg.y;

    const short2 lc = nax_coord(tiisg);
    const short fn = lc.x;
    const short fm = lc.y;

    const int i01 = tgpig.x*32;
    const int i11 = tgpig.y*16*MR;
    const int im  = tgpig.z;

    const int i12 = im%args.ne12;
    const int i13 = im/args.ne12;

    const uint64_t offset0 = (i12/args.r2)*args.nb02 + (i13/args.r3)*args.nb03;
    const uint64_t offset1 = i12*args.nb12 + i13*args.nb13;

    // weight rows of this lane: half h, row 16*h + fm + 8*i
    device const block_q * x[2][2];
    FOR_UNROLL (short h = 0; h < 2; ++h) {
        FOR_UNROLL (short i = 0; i < 2; ++i) {
            const int r = min(i01 + 16*h + fm + 8*i, args.ne01 - 1);
            x[h][i] = (device const block_q *) (src0 + offset0 + (uint64_t) r*args.nb01);
        }
    }

    // src1 rows of this lane: i11 + 16*m + fm and i11 + 16*m + fm + 8 for each of the MR 16-row tiles (zero past ne11)
    device const float * y[MR][2];
    bool yok[MR][2];
    FOR_UNROLL (short m = 0; m < MR; ++m) {
        FOR_UNROLL (short i = 0; i < 2; ++i) {
            const int r = i11 + 16*m + fm + 8*i;
            yok[m][i] = r < args.ne11;
            y[m][i] = (device const float *) (src1 + offset1 + (uint64_t) (yok[m][i] ? r : 0)*args.nb11);
        }
    }

    vec<float, 8> c[MR][2];
    FOR_UNROLL (short m = 0; m < MR; ++m) {
        c[m][0] = 0.0f;
        c[m][1] = 0.0f;
    }

    const int nch = args.ne00/64;

    for (int ch = sgitg; ch < nch; ch += NSG) {
        // the 16-weight group of this lane: weights p .. p + 15 of the row
        const int p = 64*ch + 4*fn;
        const int ib = p/(16*nl);
        const short il = (p/16)%nl;

        vec<half, 8> b[4][2];
        FOR_UNROLL (short h = 0; h < 2; ++h) {
            FOR_UNROLL (short i = 0; i < 2; ++i) {
                half4x4 w;
                deq(x[h][i] + ib, il, w);
                FOR_UNROLL (short s = 0; s < 4; ++s) {
                    FOR_UNROLL (short jj = 0; jj < 4; ++jj) {
                        b[s][h][4*i + jj] = w[s][jj];
                    }
                }
            }
        }

        FOR_UNROLL (short m = 0; m < MR; ++m) {
            vec<half, 8> a[4];
            FOR_UNROLL (short s = 0; s < 4; ++s) {
                FOR_UNROLL (short i = 0; i < 2; ++i) {
                    const float4 v = yok[m][i] ? *((device const float4 *) (y[m][i] + p + 4*s)) : float4(0.0f);
                    FOR_UNROLL (short jj = 0; jj < 4; ++jj) {
                        a[s][4*i + jj] = (half) v[jj];
                    }
                }
            }

            FOR_UNROLL (short s = 0; s < 4; ++s) {
                nax_mma<float, half, half, true>(c[m][0], c[m][1], a[s], b[s][0], b[s][1]);
            }
        }
    }

    // sum the NSG partial results: lane layout [sg][m][h][8][32]
    threadgroup float * red = (threadgroup float *) shmem;

    if (NSG > 1) {
        FOR_UNROLL (short m = 0; m < MR; ++m) {
            FOR_UNROLL (short h = 0; h < 2; ++h) {
                FOR_UNROLL (short e = 0; e < 8; ++e) {
                    red[(((sgitg*MR + m)*2 + h)*8 + e)*32 + tiisg] = c[m][h][e];
                }
            }
        }

        threadgroup_barrier(mem_flags::mem_threadgroup);

        if (sgitg != 0) {
            return;
        }

        for (short sg = 1; sg < NSG; ++sg) {
            FOR_UNROLL (short m = 0; m < MR; ++m) {
                FOR_UNROLL (short h = 0; h < 2; ++h) {
                    FOR_UNROLL (short e = 0; e < 8; ++e) {
                        c[m][h][e] += red[(((sg*MR + m)*2 + h)*8 + e)*32 + tiisg];
                    }
                }
            }
        }
    }

    device       float * dst_f32 = (device       float *) dst  + (uint64_t) im*args.ne0*args.ne1;
    device const float * res_f32 = (device const float *) src2 + (uint64_t) im*args.ne0*args.ne1;

    FOR_UNROLL (short m = 0; m < MR; ++m) {
        FOR_UNROLL (short i = 0; i < 2; ++i) {
            const int r1 = i11 + 16*m + fm + 8*i;
            if (r1 >= args.ne11) {
                continue;
            }

            FOR_UNROLL (short h = 0; h < 2; ++h) {
                FOR_UNROLL (short jj = 0; jj < 4; ++jj) {
                    const int r0 = i01 + 16*h + fn + jj;
                    if (r0 < args.ne01) {
                        const uint64_t idx = (uint64_t) r1*args.ne0 + r0;
                        const float v = c[m][h][4*i + jj];
                        dst_f32[idx] = ADD ? v + res_f32[idx] : v;
                    }
                }
            }
        }
    }
}

typedef decltype(kernel_mul_mv_nax_f32<block_q4_K, QK_NL, dequantize_q4_K, 1, false>) mul_mv_nax_t;

#define MV_NAX_INSTANCE(name, block_q, nl, deq) \
template [[host_name("kernel_mul_mv_nax_" #name "_f32_mr1_add0")]] kernel mul_mv_nax_t kernel_mul_mv_nax_f32<block_q, nl, deq, 1, false>; \
template [[host_name("kernel_mul_mv_nax_" #name "_f32_mr1_add1")]] kernel mul_mv_nax_t kernel_mul_mv_nax_f32<block_q, nl, deq, 1, true>; \
template [[host_name("kernel_mul_mv_nax_" #name "_f32_mr2_add0")]] kernel mul_mv_nax_t kernel_mul_mv_nax_f32<block_q, nl, deq, 2, false>; \
template [[host_name("kernel_mul_mv_nax_" #name "_f32_mr2_add1")]] kernel mul_mv_nax_t kernel_mul_mv_nax_f32<block_q, nl, deq, 2, true>;

MV_NAX_INSTANCE(f32,     float4x4,      1,     dequantize_f32)
MV_NAX_INSTANCE(f16,     half4x4,       1,     dequantize_f16)
#if defined(GGML_METAL_HAS_BF16)
MV_NAX_INSTANCE(bf16,    bfloat4x4,     1,     dequantize_bf16)
#endif
NAX_QTYPES(MV_NAX_INSTANCE)
MV_NAX_INSTANCE(q5_K,    block_q5_K,    QK_NL, dequantize_q5_K_nax)
#endif // GGML_METAL_HAS_TENSOR
