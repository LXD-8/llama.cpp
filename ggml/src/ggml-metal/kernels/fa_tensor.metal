#include "common.h"
#include "dequantize.h"

#ifdef GGML_METAL_HAS_TENSOR

constant bool FC_flash_attn_ext_tensor_has_mask  [[function_constant(FC_FLASH_ATTN_EXT_TENSOR + 0)]];
constant bool FC_flash_attn_ext_tensor_has_sinks [[function_constant(FC_FLASH_ATTN_EXT_TENSOR + 1)]];
constant bool FC_flash_attn_ext_tensor_has_bias  [[function_constant(FC_FLASH_ATTN_EXT_TENSOR + 2)]];
constant bool FC_flash_attn_ext_tensor_has_scap  [[function_constant(FC_FLASH_ATTN_EXT_TENSOR + 3)]];

// ref: https://arxiv.org/pdf/2307.08691.pdf
template<
    typename T,                                 // Q, K, V and P types
    typename T4,
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

    threadgroup T     * sq = (threadgroup T     *) shmem;       // [Q, DK] queries
    threadgroup float * ss = (threadgroup float *) (sq + Q*DK); // [Q, C]  scores
    threadgroup T     * sp = (threadgroup T     *) (ss + Q*C);  // [Q, C]  probabilities
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

        ((threadgroup T4 *) sq)[i] = (T4) (q4*qscale);
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

    auto tv0 = tensor((device T *) v, dextents<int32_t, 2>(DV, C), array<int, 2>({1, sv}));

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
            auto tk = tensor((device T *) (k + (uint64_t) ic*args.nb11), dextents<int32_t, 2>(DK, C), array<int, 2>({1, sk}));

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
                const T p = (T) exp(s[ii] - M[jj]);

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
            auto tv = tensor((device T *) (v + (uint64_t) ic*args.nb21), dextents<int32_t, 2>(DV, C), array<int, 2>({1, sv}));

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

typedef decltype(kernel_flash_attn_ext_tensor<half, half4, 64, 64>) flash_attn_ext_tensor_t;

template [[host_name("kernel_flash_attn_ext_tensor_f16_dk64_dv64"  )]]  kernel flash_attn_ext_tensor_t kernel_flash_attn_ext_tensor<half, half4, 64,  64>;
template [[host_name("kernel_flash_attn_ext_tensor_f16_dk128_dv128")]]  kernel flash_attn_ext_tensor_t kernel_flash_attn_ext_tensor<half, half4, 128, 128>;
template [[host_name("kernel_flash_attn_ext_tensor_f16_dk192_dv128")]]  kernel flash_attn_ext_tensor_t kernel_flash_attn_ext_tensor<half, half4, 192, 128>;
template [[host_name("kernel_flash_attn_ext_tensor_f16_dk256_dv256")]]  kernel flash_attn_ext_tensor_t kernel_flash_attn_ext_tensor<half, half4, 256, 256>;
template [[host_name("kernel_flash_attn_ext_tensor_f16_dk512_dv512")]]  kernel flash_attn_ext_tensor_t kernel_flash_attn_ext_tensor<half, half4, 512, 512, OP_FLASH_ATTN_EXT_TENSOR_NQPSG_LARGE>;
template [[host_name("kernel_flash_attn_ext_tensor_f16_dk576_dv512")]]  kernel flash_attn_ext_tensor_t kernel_flash_attn_ext_tensor<half, half4, 576, 512, OP_FLASH_ATTN_EXT_TENSOR_NQPSG_LARGE>;

template [[host_name("kernel_flash_attn_ext_tensor_bf16_dk64_dv64"  )]] kernel flash_attn_ext_tensor_t kernel_flash_attn_ext_tensor<bfloat, bfloat4, 64,  64>;
template [[host_name("kernel_flash_attn_ext_tensor_bf16_dk128_dv128")]] kernel flash_attn_ext_tensor_t kernel_flash_attn_ext_tensor<bfloat, bfloat4, 128, 128>;
template [[host_name("kernel_flash_attn_ext_tensor_bf16_dk192_dv128")]] kernel flash_attn_ext_tensor_t kernel_flash_attn_ext_tensor<bfloat, bfloat4, 192, 128>;
template [[host_name("kernel_flash_attn_ext_tensor_bf16_dk256_dv256")]] kernel flash_attn_ext_tensor_t kernel_flash_attn_ext_tensor<bfloat, bfloat4, 256, 256>;
template [[host_name("kernel_flash_attn_ext_tensor_bf16_dk512_dv512")]] kernel flash_attn_ext_tensor_t kernel_flash_attn_ext_tensor<bfloat, bfloat4, 512, 512, OP_FLASH_ATTN_EXT_TENSOR_NQPSG_LARGE>;
template [[host_name("kernel_flash_attn_ext_tensor_bf16_dk576_dv512")]] kernel flash_attn_ext_tensor_t kernel_flash_attn_ext_tensor<bfloat, bfloat4, 576, 512, OP_FLASH_ATTN_EXT_TENSOR_NQPSG_LARGE>;

constant bool FC_flash_attn_ext_nax_has_mask  [[function_constant(FC_FLASH_ATTN_EXT_NAX + 0)]];
constant bool FC_flash_attn_ext_nax_has_sinks [[function_constant(FC_FLASH_ATTN_EXT_NAX + 1)]];

// K/V elements of the NAX attention kernels: bf16 is converted to f16 in registers and clamped to the f16 range
static inline half4 fa_nax_kv4(device const half * p) {
    return *((device const half4 *) p);
}

static inline half4 fa_nax_kv4(device const bfloat * p) {
    return (half4) clamp(float4(*((device const bfloat4 *) p)), -65504.0f, 65504.0f);
}

// a quantized cache is read as stored by the GQA kernel: 4 values at column col of a row
template<typename T> constexpr bool fa_nax_kv_is_q() { return !(is_same<T, half>::value || is_same<T, bfloat>::value); }

static inline void fa_nax_dq4(device const block_q4_0   * b, short il, thread float4 & r) { dequantize_q4_0_t4  (b, il, r); }
static inline void fa_nax_dq4(device const block_q4_1   * b, short il, thread float4 & r) { dequantize_q4_1_t4  (b, il, r); }
static inline void fa_nax_dq4(device const block_q5_0   * b, short il, thread float4 & r) { dequantize_q5_0_t4  (b, il, r); }
static inline void fa_nax_dq4(device const block_q5_1   * b, short il, thread float4 & r) { dequantize_q5_1_t4  (b, il, r); }
static inline void fa_nax_dq4(device const block_q8_0   * b, short il, thread float4 & r) { dequantize_q8_0_t4  (b, il, r); }
static inline void fa_nax_dq4(device const block_iq4_nl * b, short il, thread float4 & r) { dequantize_iq4_nl_t4(b, il, r); }

template<typename B>
static inline half4 fa_nax_kvq4(device const char * row, int col) {
    float4 r;
    fa_nax_dq4((device const B *) row + col/32, (col%32)/4, r);
    return (half4) r;
}

// flash attention with the scores, the probabilities and O kept in registers; no ALiBi or softcap
// WM simdgroups split the queries and WN simdgroups split the head dim, as in MLX steel attention_nax_dsplit (MIT, Apple Inc.)
template<
    typename KV,                                // K/V element type: half or bfloat
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
    constexpr short TDk = DK/kU/WN;   // K head-dim fragments per simdgroup
    constexpr short BDk = DK/WN;      // K head-dim columns per simdgroup
    constexpr short TDv = DV/kU/WN;   // V head-dim fragments per simdgroup
    constexpr short BDv = DV/WN;      // V head-dim columns per simdgroup

    static_assert(DK % (kU*WN) == 0 && DV % (kU*WN) == 0, "the head dims must split into WN fragment columns");
    static_assert(TK % 2 == 0,  "scores are computed in pairs of fragments");
    static_assert(TDv % 2 == 0, "O is accumulated in pairs of fragments");

    const int iq3 = tgpig[2];
    const int iq2 = tgpig[1];
    const int iq1 = tgpig[0]*BQ;

    const short row_group = sgitg/WN;
    const short d_group   = sgitg%WN;

    const short2 sc = nax_coord(tiisg);
    const short fn = sc.x;
    const short fm = sc.y;

    // first query row held by this lane - the second one is r0 + 8
    const int r0 = iq1 + row_group*kU + fm;

    // queries, with the softmax scale and log2(e) folded in
    vec<half, 8> qf[TDk];
    {
        const float qs = args.scale*M_LOG2E_F;

        device const char * q0 = q + iq2*args.nb02 + iq3*args.nb03;

        FOR_UNROLL (short i = 0; i < 2; ++i) {
            const int r = r0 + 8*i;

            FOR_UNROLL (short id = 0; id < TDk; ++id) {
                float4 x = 0.0f;
                if (r < args.ne01) {
                    x = *((device const float4 *) (q0 + r*args.nb01) + (d_group*BDk + id*kU + fn)/4);
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

    device const KV * pk = (device const KV *) k + fm*sk + d_group*BDk + fn;
    device const KV * pv = (device const KV *) v + fm*sv + d_group*BDv + fn;

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

    vec<float, 8> of[TDv];
    FOR_UNROLL (short id = 0; id < TDv; ++id) {
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
            FOR_UNROLL (short id = 0; id < TDk; ++id) {
                vec<half, 8> k0;
                vec<half, 8> k1;

                device const KV * pk0 = pk + (ic + ik*kU)*sk + id*kU;
                device const KV * pk1 = pk0 + kU*sk;

                FOR_UNROLL (short i = 0; i < 2; ++i) {
                    const half4 a = fa_nax_kv4(pk0 + 8*i*sk);
                    const half4 b = fa_nax_kv4(pk1 + 8*i*sk);

                    FOR_UNROLL (short j = 0; j < 4; ++j) {
                        k0[4*i + j] = a[j];
                        k1[4*i + j] = b[j];
                    }
                }

                nax_mma<float, half, half, true>(sf[ik], sf[ik + 1], qf[id], k0, k1);
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

            FOR_UNROLL (short id = 0; id < TDv; ++id) {
                FOR_UNROLL (short j = 0; j < 4; ++j) {
                    of[id][4*i + j] *= ms;
                }
            }
        }

        // O += P*V over this simdgroup's slice of the head dim
        FOR_UNROLL (short id = 0; id < TDv; id += 2) {
            FOR_UNROLL (short ik = 0; ik < TK; ++ik) {
                vec<half, 8> v0;
                vec<half, 8> v1;

                device const KV * pv0 = pv + (ic + ik*kU)*sv + id*kU;

                FOR_UNROLL (short i = 0; i < 2; ++i) {
                    const half4 a = fa_nax_kv4(pv0 + 8*i*sv);
                    const half4 b = fa_nax_kv4(pv0 + 8*i*sv + kU);

                    FOR_UNROLL (short j = 0; j < 4; ++j) {
                        v0[4*i + j] = a[j];
                        v1[4*i + j] = b[j];
                    }
                }

                nax_mma<float, float, half, false>(of[id], of[id + 1], sf[ik], v0, v1);
            }
        }
    }

    // normalize and store - rows past ne01 are not written
    device float * pdst = (device float *) dst + ((uint64_t) iq3*args.ne2*args.ne1 + iq2)*DV + d_group*BDv + fn;

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

            FOR_UNROLL (short id = 0; id < TDv; ++id) {
                *((device float4 *) (pd + id*kU)) = float4(of[id][4*i + 0], of[id][4*i + 1], of[id][4*i + 2], of[id][4*i + 3])*rs;
            }
        }
    }
}

typedef decltype(kernel_flash_attn_ext_nax<half, 256, 256, 2>) flash_attn_ext_nax_t;

template [[host_name("kernel_flash_attn_ext_nax_f16_dk64_dv64")]]   kernel flash_attn_ext_nax_t kernel_flash_attn_ext_nax<half, 64,  64,  1>;
template [[host_name("kernel_flash_attn_ext_nax_bf16_dk64_dv64")]]   kernel flash_attn_ext_nax_t kernel_flash_attn_ext_nax<bfloat, 64,  64,  1>;
template [[host_name("kernel_flash_attn_ext_nax_f16_dk128_dv128")]] kernel flash_attn_ext_nax_t kernel_flash_attn_ext_nax<half, 128, 128, 1>;
template [[host_name("kernel_flash_attn_ext_nax_bf16_dk128_dv128")]] kernel flash_attn_ext_nax_t kernel_flash_attn_ext_nax<bfloat, 128, 128, 1>;
template [[host_name("kernel_flash_attn_ext_nax_f16_dk256_dv256")]] kernel flash_attn_ext_nax_t kernel_flash_attn_ext_nax<half, 256, 256, 2>;
template [[host_name("kernel_flash_attn_ext_nax_bf16_dk256_dv256")]] kernel flash_attn_ext_nax_t kernel_flash_attn_ext_nax<bfloat, 256, 256, 2>;
template [[host_name("kernel_flash_attn_ext_nax_f16_dk512_dv512")]] kernel flash_attn_ext_nax_t kernel_flash_attn_ext_nax<half, 512, 512, 4>;
template [[host_name("kernel_flash_attn_ext_nax_bf16_dk512_dv512")]] kernel flash_attn_ext_nax_t kernel_flash_attn_ext_nax<bfloat, 512, 512, 4>;
template [[host_name("kernel_flash_attn_ext_nax_f16_dk192_dv128")]]  kernel flash_attn_ext_nax_t kernel_flash_attn_ext_nax<half,   192, 128, 1>;
template [[host_name("kernel_flash_attn_ext_nax_bf16_dk192_dv128")]] kernel flash_attn_ext_nax_t kernel_flash_attn_ext_nax<bfloat, 192, 128, 1>;
template [[host_name("kernel_flash_attn_ext_nax_f16_dk576_dv512")]]  kernel flash_attn_ext_nax_t kernel_flash_attn_ext_nax<half,   576, 512, 4>;
template [[host_name("kernel_flash_attn_ext_nax_bf16_dk576_dv512")]] kernel flash_attn_ext_nax_t kernel_flash_attn_ext_nax<bfloat, 576, 512, 4>;

// small batches: the queries of the G heads that share a KV head are packed into rows p = g*ne01 + i, so each K/V block is loaded once
// the KV range is split over NWG threadgroups, combined by kernel_flash_attn_ext_vec_reduce; no ALiBi or softcap
template<
    typename KV,                                // K/V element type: half or bfloat
    short DK,
    short DV,
    short WN,
    short WM,
    short BK  = OP_FLASH_ATTN_EXT_NAX_NCPSG>
kernel void kernel_flash_attn_ext_nax_gqa(
        constant ggml_metal_kargs_flash_attn_ext & args,
        device const char * q,
        device const char * k,
        device const char * v,
        device const char * mask,
        device const char * sinks,
        device       char * tmp,
        threadgroup  char * shmem [[threadgroup(0)]],
        uint3  tgpig [[threadgroup_position_in_grid]],
        uint3  tgpg  [[threadgroups_per_grid]],
        ushort tiisg [[thread_index_in_simdgroup]],
        ushort sgitg [[simdgroup_index_in_threadgroup]]) {
    constexpr short kU  = 16;
    constexpr short TK  = BK/kU;
    constexpr short TDh = DK/kU/WN;
    constexpr short BDh = DK/WN;

    static_assert(DK == DV,     "DK must equal DV");
    static_assert(TK % 2 == 0,  "scores are computed in pairs of fragments");
    static_assert(TDh % 2 == 0, "O is accumulated in pairs of fragments");

    const int G = args.ne02/args.ne_12_2; // query heads per KV head
    const int R = G*args.ne01;            // packed rows

    // grid x: NWG splits of the KV range for each block of WM*16 packed rows
    const int NWG = tgpg[0]/((R + WM*kU - 1)/(WM*kU));

    const int iwg = tgpig[0]%NWG;
    const int irb = tgpig[0]/NWG;
    const int hk  = tgpig[1];
    const int iq3 = tgpig[2];

    const short row_group = sgitg/WN;
    const short d_group   = sgitg%WN;

    const short2 sc = nax_coord(tiisg);
    const short fn = sc.x;
    const short fm = sc.y;

    // the two packed rows of this lane: query i, head h
    int  ri[2];
    int  rh[2];
    bool rok[2];

    FOR_UNROLL (short i = 0; i < 2; ++i) {
        const int p = (irb*WM + row_group)*kU + fm + 8*i;
        rok[i] = p < R;
        const int pp = rok[i] ? p : 0;
        ri[i] = pp % args.ne01;
        rh[i] = hk*G + pp/args.ne01;
    }

    vec<half, 8> qf[TDh];
    {
        const float qs = args.scale*M_LOG2E_F;

        FOR_UNROLL (short i = 0; i < 2; ++i) {
            device const char * q0 = q + ri[i]*args.nb01 + rh[i]*args.nb02 + iq3*args.nb03;

            FOR_UNROLL (short id = 0; id < TDh; ++id) {
                float4 x = 0.0f;
                if (rok[i]) {
                    x = *((device const float4 *) q0 + (d_group*BDh + id*kU + fn)/4);
                }

                const half4 h = (half4) (x*qs);

                FOR_UNROLL (short j = 0; j < 4; ++j) {
                    qf[id][4*i + j] = h[j];
                }
            }
        }
    }

    {
        const int ikv3 = iq3/(args.ne03/args.ne_12_3);

        k += hk*args.nb12 + ikv3*args.nb13;
        v += hk*args.nb22 + ikv3*args.nb23;
    }

    const int sk = args.ns10;
    const int sv = args.ns20;

    device const KV * pk = (device const KV *) k + fm*sk + d_group*BDh + fn;
    device const KV * pv = (device const KV *) v + fm*sv + d_group*BDh + fn;

    device const half * pm[2];
    FOR_UNROLL (short i = 0; i < 2; ++i) {
        pm[i] = (device const half *) (mask + ri[i]*args.nb31 + (rh[i]%args.ne32)*args.nb32 + (iq3%args.ne33)*args.nb33) + fn;
    }

    vec<float, 8> of[TDh];
    FOR_UNROLL (short id = 0; id < TDh; ++id) {
        of[id] = 0.0f;
    }

    float M[2] = { -FLT_MAX/2, -FLT_MAX/2 };
    float S[2] = { 0.0f, 0.0f };

    threadgroup float * sx     = (threadgroup float *) shmem;
    threadgroup float * sx_own = sx + (row_group*WN + d_group)*(TK*8*32) + tiisg*(TK*8);
    threadgroup float * sx_row = sx + (row_group*WN          )*(TK*8*32) + tiisg*(TK*8);

    // the host guarantees ne11 % BK == 0
    const int nblk_k  = args.ne11/BK;
    const int nblk_wg = (nblk_k + NWG - 1)/NWG;
    const int ib0 = iwg*nblk_wg;
    const int ib1 = min(nblk_k, ib0 + nblk_wg);

    for (int ib = ib0; ib < ib1; ++ib) {
        const int ic = ib*BK;

        vec<float, 8> sf[TK];
        FOR_UNROLL (short ik = 0; ik < TK; ++ik) {
            sf[ik] = 0.0f;
        }

        FOR_UNROLL (short ik = 0; ik < TK; ik += 2) {
            FOR_UNROLL (short id = 0; id < TDh; ++id) {
                vec<half, 8> k0;
                vec<half, 8> k1;

                device const KV * pk0 = pk + (ic + ik*kU)*sk + id*kU;
                device const KV * pk1 = pk0 + kU*sk;

                FOR_UNROLL (short i = 0; i < 2; ++i) {
                    half4 a;
                    half4 b;
                    if constexpr (fa_nax_kv_is_q<KV>()) {
                        const int r = ic + ik*kU + fm + 8*i;
                        const int c = d_group*BDh + id*kU + fn;
                        a = fa_nax_kvq4<KV>(k + (uint64_t)  r      *args.nb11, c);
                        b = fa_nax_kvq4<KV>(k + (uint64_t) (r + kU)*args.nb11, c);
                    } else {
                        a = fa_nax_kv4(pk0 + 8*i*sk);
                        b = fa_nax_kv4(pk1 + 8*i*sk);
                    }

                    FOR_UNROLL (short j = 0; j < 4; ++j) {
                        k0[4*i + j] = a[j];
                        k1[4*i + j] = b[j];
                    }
                }

                nax_mma<float, half, half, true>(sf[ik], sf[ik + 1], qf[id], k0, k1);
            }
        }

        if (WN > 1) {
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

        if (FC_flash_attn_ext_nax_has_mask) {
            FOR_UNROLL (short i = 0; i < 2; ++i) {
                FOR_UNROLL (short ik = 0; ik < TK; ++ik) {
                    device const half * pmi = pm[i] + ic + ik*kU;

                    FOR_UNROLL (short j = 0; j < 4; ++j) {
                        sf[ik][4*i + j] += M_LOG2E_F*(float) pmi[j];
                    }
                }
            }
        }

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

        FOR_UNROLL (short id = 0; id < TDh; id += 2) {
            FOR_UNROLL (short ik = 0; ik < TK; ++ik) {
                vec<half, 8> v0;
                vec<half, 8> v1;

                device const KV * pv0 = pv + (ic + ik*kU)*sv + id*kU;

                FOR_UNROLL (short i = 0; i < 2; ++i) {
                    half4 a;
                    half4 b;
                    if constexpr (fa_nax_kv_is_q<KV>()) {
                        device const char * row = v + (uint64_t) (ic + ik*kU + fm + 8*i)*args.nb21;
                        const int c = d_group*BDh + id*kU + fn;
                        a = fa_nax_kvq4<KV>(row, c);
                        b = fa_nax_kvq4<KV>(row, c + kU);
                    } else {
                        a = fa_nax_kv4(pv0 + 8*i*sv);
                        b = fa_nax_kv4(pv0 + 8*i*sv + kU);
                    }

                    FOR_UNROLL (short j = 0; j < 4; ++j) {
                        v0[4*i + j] = a[j];
                        v1[4*i + j] = b[j];
                    }
                }

                nax_mma<float, float, half, false>(of[id], of[id + 1], sf[ik], v0, v1);
            }
        }
    }

    // partial results in the layout of kernel_flash_attn_ext_vec_reduce: O, then S and M (natural log)
    const int nrows = args.ne1*args.ne2*args.ne3;

    device float4 * o4 = (device float4 *) tmp;
    device float  * ss = (device float  *) tmp + (uint64_t) nrows*DV*NWG;

    FOR_UNROLL (short i = 0; i < 2; ++i) {
        if (!rok[i]) {
            continue;
        }

        const uint64_t rid = ((uint64_t) iq3*args.ne2 + ri[i])*args.ne1 + rh[i];

        if (FC_flash_attn_ext_nax_has_sinks && iwg == 0) {
            const float s = M_LOG2E_F*((device const float *) sinks)[rh[i]];
            const float m = max(M[i], s);
            const float ms = fast::exp2(M[i] - m);

            S[i] = S[i]*ms + fast::exp2(s - m);
            M[i] = m;

            FOR_UNROLL (short id = 0; id < TDh; ++id) {
                FOR_UNROLL (short j = 0; j < 4; ++j) {
                    of[id][4*i + j] *= ms;
                }
            }
        }

        FOR_UNROLL (short id = 0; id < TDh; ++id) {
            const int c4 = (d_group*BDh + id*kU + fn)/4;
            o4[(rid*(DV/4) + c4)*NWG + iwg] = float4(of[id][4*i + 0], of[id][4*i + 1], of[id][4*i + 2], of[id][4*i + 3]);
        }

        if (fn == 0 && d_group == 0) {
            ss[rid*(2*NWG) + 2*iwg + 0] = S[i];
            ss[rid*(2*NWG) + 2*iwg + 1] = M[i]*M_LN2_F;
        }
    }
}

typedef decltype(kernel_flash_attn_ext_nax_gqa<half, 256, 256, 2, 1>) flash_attn_ext_nax_gqa_t;

#define FA_NAX_GQA_WM(name, T, dk, wn, wm) \
template [[host_name("kernel_flash_attn_ext_nax_gqa_" #name "_dk" #dk "_dv" #dk "_wm" #wm)]] \
kernel flash_attn_ext_nax_gqa_t kernel_flash_attn_ext_nax_gqa<T, dk, dk, wn, wm>;

#define FA_NAX_GQA_DK(name, T, dk, wn) \
    FA_NAX_GQA_WM(name, T, dk, wn, 1) \
    FA_NAX_GQA_WM(name, T, dk, wn, 2) \
    FA_NAX_GQA_WM(name, T, dk, wn, 4)

#define FA_NAX_GQA(name, T) \
    FA_NAX_GQA_DK(name, T, 64,  1) \
    FA_NAX_GQA_DK(name, T, 128, 1) \
    FA_NAX_GQA_DK(name, T, 256, 2) \
    FA_NAX_GQA_DK(name, T, 512, 4) \
    FA_NAX_GQA_WM(name, T, 64,  1, 8) \
    FA_NAX_GQA_WM(name, T, 128, 1, 8) \
    FA_NAX_GQA_WM(name, T, 256, 2, 8)

FA_NAX_GQA(f16,  half)
FA_NAX_GQA(bf16, bfloat)

// quantized caches (decode and small batches; larger batches convert the cache to f16 first)
FA_NAX_GQA(q4_0,   block_q4_0)
FA_NAX_GQA(q4_1,   block_q4_1)
FA_NAX_GQA(q5_0,   block_q5_0)
FA_NAX_GQA(q5_1,   block_q5_1)
FA_NAX_GQA(q8_0,   block_q8_0)
FA_NAX_GQA(iq4_nl, block_iq4_nl)
#endif // GGML_METAL_HAS_TENSOR
