# llama.cpp Metal optimizations for Apple M5 (NAX): for every model and weight format

[English](#llamacpp-metal-optimizations-for-apple-m5-nax-for-every-model-and-weight-format) | [中文](#apple-m5nax上的-llamacpp-metal-优化面向各种模型与量化格式)

This project optimizes the Metal backend of llama.cpp on Apple M5, with the goal that every kind of model (dense, MoE, different attention structures) and every weight format runs close to the hardware limit of the machine. It covers prompt processing (prefill), decode, speculative decoding (MTP / DFlash), concurrency, short prompts, start-up and the multi-turn conversation cache. All changes are based on upstream `master` at `67107ac34f` (b12164): 22 code commits, one per change (H3, H5 and H3b are one commit), each can be switched off on its own.

The table compares all changes on with upstream and oMLX (M5 Max, full power, server end to end, streaming client, temperature 0; oMLX with its INT8 prefill on, KV quantization off on both sides):

| Model | 8k prefill t/s: ours / upstream / oMLX | vs upstream | 30k decode t/s: ours / up / oMLX | vs upstream | 30k speculative decode t/s: ours / up / oMLX |
|---|---|---|---|---|---|
| Qwen3.8-27B Q5_K_M | 1086 / 642 / 1128 | 1.69x | 22.5 / 21.7 / 25.0 | 1.04x | 41.2 / 34.5 / 47.1 |
| gemma-4-26B-A4B Q4_0 | 4054 / 3313 / 4456 | 1.22x | 106.2 / 89.7 / 67.8 | 1.18x | 117.2 / 83.9 / 97.0 |

The "speculative decode" column: Qwen uses the MTP head inside the model file (`--spec-type draft-mtp --spec-draft-n-max 3`), gemma the DFlash draft model from the same ggml-org repository (`--spec-type draft-dflash`); oMLX uses the models' own MTP heads for both.

Scope: 22 code commits, 26 files, +3396 / -122. 21 are in the engine (mostly `ggml/src/ggml-metal/`, one line in `common/`, test cases added to `tests/test-backend-ops.cpp`), 1 in the built-in Web UI (`tools/ui`, the interface of Llama.app). The NAX kernels are only used on devices with the Metal tensor API (by default M5 / A19 and newer); other devices keep the upstream paths. Except for formats that have no switch of their own (NVFP4 / TQ1_0), every engine change can be switched off with an environment variable for A/B testing.

Models measured: Qwen3.8-27B Q5_K_M (dense, GDN + full attention, q5_K / q6_K) and gemma-4-26B-A4B Q4_0 (MoE, 8 of 128 experts, with head size 512 global attention layers). The other weight formats were checked numerically with `test-backend-ops` and timed at kernel level on synthetic weights (section 7).

Test machine: MacBook Pro (Mac17,6), Apple M5 Max: 18-core CPU (6 super + 12 performance cores), 40-core GPU, 128 GB unified memory; macOS 27.2. Every number in this document was measured on this machine.

### About this work and the use of AI

This project started from an everyday observation: on my machine, llama.cpp (the engine of Llama.app) was clearly slower at prefill and decode than the MLX-based oMLX. The goals, the research direction, the test methodology and the decisions on which approaches to keep based on the measurements are my own. AI tools were used for the research, writing code, testing and writing this document, and their output was reviewed by me. The Chinese version of this document is the original; this English version was translated from it by AI.

Contents:

- [1. Why](#1-why)
- [2. Changes](#2-changes)
- [3. Outcome](#3-outcome)
- [4. Method](#4-method)
- [5. Candidates found by research](#5-candidates-found-by-research)
- [6. Per-change results and selection](#6-per-change-results-and-selection)
- [7. Weight formats](#7-weight-formats)
- [8. Implementation of the kept changes](#8-implementation-of-the-kept-changes)
- [9. Accuracy](#9-accuracy)
- [10. Remaining gaps](#10-remaining-gaps)
- [11. Reproducing](#11-reproducing)
- [About the code](#about-the-code)

## 1. Why

On an M5 Max, the Metal backend of stock llama.cpp processed prompts at only about 55-70% of the speed of oMLX (for example Qwen3.8-27B, 8k prompt, 653 vs 1187 t/s), and the gap was larger for MoE models and long contexts. In everyday use there were further clear losses: each speculative decoding step verifies several tokens and was much slower than a single-token step; the app's default of 4 concurrent streams gave far less than 4x the single-stream throughput; short prompts (follow-up questions, new chat turns) fell into a very slow range of batch sizes; the app needed more than 3 s to the first token after start-up and kept about 21 GB resident; and turning thinking off in the interface made the whole conversation be processed again.

The M5 generation adds matrix units to every GPU core (NAX, used through the `matmul2d` of the Metal tensor API / MetalPerformancePrimitives). MLX drives them with register-resident kernels; upstream llama.cpp used them only through a kernel that tiles through threadgroup memory and takes f32 activations, leaving most of the new compute unused.

Goals:
- get as close as possible to the hardware limits of the M5 Max, and match or beat oMLX wherever the comparison is fair;
- borrow from Apple Silicon engines such as MLX and oMLX where it fits, and also do optimizations none of them have;
- cost-effectiveness: no change whose final gain is below 1.5% (noise level) while needing complex validation or a lot of code; the kept changes are implemented with the minimum necessary code; large refactors are allowed for large gains;
- change only the Metal backend (plus the minimum necessary elsewhere); every change is switchable and numerically verified;
- cover every weight format the Metal mat-mul kernels support, and dense, MoE and the various attention structures, not just what the test models use.

Hardware limits measured on this machine (M5 Max, 40-core GPU, 128 GB):

| | Value |
|---|---|
| Memory bandwidth (Apple spec) | 614 GB/s |
| Measured sustainable bandwidth | 530-580 GB/s (varies with kernel, transfer size and state) |
| NAX f16 mat-mat, reachable | ~60 TFLOP/s (MLX dense f16 gemm 59-60; A3 below 59.7) |
| NAX int8 mat-mat | ~2.1x f16 (same register-resident kernel structure) |

Decode is bandwidth bound: for example Qwen3.8-27B Q5_K_M reads 18.34 GB of weights per token; in this branch the mat-muls of single-token decode read at 517-540 GB/s, within the sustainable bandwidth, and the rest is attention and small ops (section 10).

## 2. Changes

In branch order, one commit per change (H3, H5 and H3b are one commit). The "Formats" column says which weight formats a mat-mul change applies to; the other changes do not depend on the weight format.

| # | Commit | Change | Lines | Formats | Switch |
|---|---|---|---|---|---|
| A1 | `a7f094140` | Register-resident NAX flash attention (prefill, head size 128 / 256 / 512) | +461 / -9 | - (F16 K/V) | `GGML_METAL_FA_NAX_DISABLE` |
| A2 | `0a39acbcd` | f32 -> f16 activation conversion before the tensor mul_mm | +136 / -7 | F16 and quantized | `GGML_METAL_MM_F16_ACT_DISABLE` |
| A3 | `d42a64b1b` | Dequant to f16 + register-resident NAX mat-mat, split-K for long K (>= 1024 rows) | +397 / -4 | F16 + 23 quantized formats | `GGML_METAL_MM_NAX_DISABLE` |
| A4 | `d5346f1aa` | Gated delta net with 4 lanes per state row | +157 / -2 | - | `GGML_METAL_GDN_R8_DISABLE` |
| I8 | `87b8d9db5` | int8 NAX mat-mat (one scale per 128 columns, Walsh-Hadamard transform before quantizing) | +276 / -7 | 23 quantized formats | `GGML_METAL_MM_I8_DISABLE` |
| A5 | `eb3b224e3` | SWIGLU fused into the NAX mat-mat that consumes it | +213 / -7 | as A3 | `GGML_METAL_GLU_MM_DISABLE` |
| S1 | `866581197` | Keep the conversation cache when thinking is turned off (Web UI) | +12 / -1 | - | UI setting |
| S2-a | `9510455c9` | Do not zero-fill fresh KV / recurrent-state buffers at load | +44 / -1 | - | `GGML_METAL_CLEAR_SKIP_DISABLE` |
| K3 | `8d0c74072` | Cross-row mat-vec for 2-3 rows | +316 / -1 | q4_K (2-3 rows), q5_K (2 rows) | `GGML_METAL_MV_XR_DISABLE` |
| S2-c | `c1506b989` | First residency request of fresh buffers in the background | +44 / -5 | - | `GGML_METAL_RSET_ASYNC_DISABLE` |
| A2 gate | `2ff0114a8` | f16 activation conversion only from 128 rows and 4 GFLOP | +9 / -2 | as A2 | `GGML_METAL_MM_F16_ACT_ROWS_128_DISABLE` |
| H4 | `9c50c4081` | NAX few-row mat-mat for 2-64 rows | +321 / -10 | all formats of the few-row kernel, row ranges per format in section 7 | `GGML_METAL_MV_NAX_DISABLE` |
| M1 | `be6587734` | MoE mat-mat (MUL_MAT_ID) from 16 tokens (was 32) | +9 / -8 | all formats | `GGML_METAL_MM_ID_MIN16_DISABLE` |
| M2 | `a03983a37` | MoE mat-mat tiles of 64 entries (when the experts get >= 128 entries on average) | +90 / -25 | all formats | `GGML_METAL_MM_ID_WIDE_DISABLE` |
| A6 | `8768dad0f` | NAX flash attention for a BF16 K/V cache (prefill and GQA, converted to f16 in registers) | +36 / -21 | BF16 | `GGML_METAL_FA_NAX_BF16_DISABLE` |
| A8 | `9d2cb5214` | NAX mat-mat for BF16 weights (from 1024 rows) | +20 / -6 | BF16 | `GGML_METAL_MM_NAX_BF16_DISABLE` |
| A7 | `7479b2600` | NAX flash attention for the MLA head sizes 192/128 and 576/512 | +34 / -20 | MLA | `GGML_METAL_FA_NAX_MLA_DISABLE` |
| A9 | `ca8a15013` | NVFP4 / TQ1_0 weights | +308 / -7 | NVFP4, TQ1_0 | - (no switch of its own) |
| A3b | `e11f1820d` | int8 NAX mat-mat from 448 rows | +13 / -9 | the quantized formats of the int8 path | `GGML_METAL_MM_NAX_MIN_ROWS` |
| A5b | `971b41a44` | The fused GLU writes the activations into the mat-mat result (memory race fix) | +16 / -24 | as A3 | `GGML_METAL_GLU_MM_DISABLE` |
| A10 | `e46a6c395` | BF16 prefill attention at head size 512 / 576 (more than 32 queries) yields to the tensor kernel | +5 / -2 | BF16 | `GGML_METAL_FA_NAX_BF16_DISABLE` |
| H3 / H5 / H3b | `06d72db6c` | One commit. H3: query heads that share a KV head packed into one NAX attention tile (decode, speculative verify; head size 128-512). H5: the same for quantized K/V caches. H3b: up to 8 tiles of 16 rows, row blocks, runtime KV split count. Also reads K/V through the `kv_rows` lists (section 6.5) | +526 / -56 | - (F16 / BF16 K/V); q4_0 / q4_1 / q5_0 / q5_1 / q8_0 / iq4_nl caches | `GGML_METAL_FA_GQA_DISABLE`, `GGML_METAL_FA_GQA_Q_DISABLE` |
| | | **Total** | **+3396 / -122** | | |

## 3. Outcome

Fairness towards oMLX: wherever a change in this branch has a counterpart option in oMLX, it is on in oMLX too: its INT8 activation prefill (the counterpart of I8) is on. The other changes (NAX attention, NAX mat-mat, GDN kernel, activation fusion) correspond to paths MLX/oMLX always use and cannot switch off. KV cache quantization is off on both sides; comparisons without speculative decoding have it off on both sides, comparisons with it have it on on both sides (oMLX MTP can be switched off in the model settings; it was toggled before the measurement and restored and checked afterwards). The oMLX settings were read before and restored and checked after the measurement. oMLX was measured in a separate session, so ratios against oMLX are across sessions.

End to end: this round the 8k / 30k prompts measured 8,020 / 29,686 tokens (oMLX tokenizes differently, so its token counts differ slightly), with 128 / 256 / 256 generated tokens; the older 2k / 16k, concurrency and start-up numbers are kept and marked as not re-measured. This branch and upstream, one forward and one reverse round, averaged. llama-bench: `-fa 1`, the 8k prefill this round with `-p 8192 -d 8192 -ub 2048` (llama-bench defaults to 512, below the 1024-row threshold of the NAX mat-mat), forward and reverse rounds.

### 3.1 Qwen3.8-27B Q5_K_M

`Qwen3.8-27B-Q5_K_M.gguf` (19.5 GB; 64 layers: 48 gated-delta-net + 16 full attention, 24 query heads sharing 4 KV heads, head size 256; weights q5_K and q6_K; MTP head in the same file). oMLX counterpart `Qwen3.8-27B-oQ5e-mtp`.

End to end (t/s):

| t/s | 2k prefill | 8k prefill | 30k prefill | 2k decode | 8k decode | 30k decode |
|---|---|---|---|---|---|---|
| this branch | 1062* | 1086 | 951* | 26.0* | 24.6* | 22.5 |
| upstream | 692* | 642 | 554* | 25.7* | 24.3* | 21.7 |
| oMLX (MTP off) | 1018* | 1128 | 974* | 26.5* | 25.8* | 25.0 |
| this branch, speculative | - | 1069* | 916* | - | 42.7* | 41.2 |
| upstream, speculative | - | 650* | 533* | - | 38.3* | 34.5 |
| oMLX, MTP | - | 1098 | 915* | - | 57.5* | 47.1 |

\* Only the 8k prefill, the 30k decode and the 30k speculative decode were re-measured this round; cells marked * keep the previous round's numbers. llama-bench `pp8192 @8k`: this branch 1008, upstream 569 t/s (1.77x).

Prefill and decode (llama-bench, t/s):

| t/s | pp2048 @0 | pp2048 @8k | pp2048 @32k | pp512 @0 | pp512 @8k | pp512 @32k | tg64 @0 | tg64 @8k | tg64 @32k |
|---|---|---|---|---|---|---|---|---|---|
| this branch | 1134 | 1018 | 790 | 809 | 723 | 613 | 25.9 | 24.6 | 23.0 |
| upstream | 650 | 585 | 451 | 719 | 608 | 469 | 25.9 | 24.3 | 22.0 |
| ratio | 1.75x | 1.74x | 1.75x | 1.12x | 1.19x | 1.31x | 1.00x | 1.01x | 1.05x |

Time per step (llama-bench, depth 0, ms per step; steps of 2-64 tokens are speculative verify, a few concurrent streams and short prompts):

| tokens per step | 1 | 2 | 4 | 8 | 16 | 24 | 32 | 48 | 64 | 96 |
|---|---|---|---|---|---|---|---|---|---|---|
| this branch | 38.9 | 42.5 | 51.2 | 51.0 | 56.0 | 74.4 | 82.1 | 132.2 | 145.2 | 220.9 |
| upstream | 38.8 | 55.8 | 56.8 | 56.2 | 90.4 | 195.9 | 195.6 | 201.7 | 207.3 | 223.9 |
| upstream / ours | 1.00x | 1.31x | 1.11x | 1.10x | 1.61x | 2.63x | 2.38x | 1.53x | 1.43x | 1.01x |

Concurrency (aggregate decode t/s):

| streams | 1 | 2 | 3 | 4 | 1, speculative | 4, speculative |
|---|---|---|---|---|---|---|
| this branch | 26.1 | 42.6 | 54.1 | 67.7 | 37.6 | 101.6 |
| upstream | 26.0 | 34.9 | 48.8 | 61.1 | 35.0 | 74.6 |
| ratio | 1.00x | 1.22x | 1.11x | 1.11x | 1.08x | 1.36x |

App start-up (the app's arguments: 262k context, mmproj, MTP, `--fit-target 1024`): this branch ready in 1.9 s with 0.24 s first-request TTFT, upstream 2.7 s / 0.33 s; resident memory after two requests (not counting the mmapped weights) 3.3 GB vs about 21 GB. The greedy outputs of both are byte-identical.

### 3.2 gemma-4-26B-A4B Q4_0

`gemma-4-26B-A4B-it-Q4_0.gguf` (ggml-org, 14.6 GB; 30 MoE layers, 8 of 128 experts per token plus a shared dense FFN; sliding-window layers with head size 256, global layers with head size 512). Speculative decoding uses `dflash-gemma-4-26B-A4B-it-Q8_0.gguf` from the same repository. oMLX counterpart `gemma-4-26B-A4B-it-oQ4e-mtp`.

| t/s | 2k prefill | 8k prefill | 30k prefill | 2k decode | 8k decode | 30k decode |
|---|---|---|---|---|---|---|
| this branch | 3869* | 4054 | 3354* | 121.4* | 114.9* | 106.2 |
| upstream | 3521* | 3313 | 2703* | 120.0* | 108.1* | 89.7 |
| oMLX (MTP off) | 3588* | 4456 | 3522* | 113.4* | 109.0* | 67.8 |
| this branch, speculative | - | 3893* | 3151* | - | 117.4* | 117.2 |
| upstream, speculative | - | 3439* | 2564* | - | 109.3* | 83.9 |
| oMLX, MTP | - | 4412 | 3534* | - | 107.0* | 97.0 |

\* Only the 8k prefill, the 30k decode and the 30k speculative decode were re-measured this round; cells marked * keep the previous round's numbers. llama-bench `pp8192 @8k`: this branch 3820, upstream 3144 t/s (1.21x). oMLX's `gemma-4-26B-A4B-it-oQ4e-mtp` decodes at 67.8 with MTP off and 97.0 with MTP on (30k).

| t/s | pp2048 @0 | pp2048 @8k | pp2048 @32k | pp512 @0 | pp512 @8k | pp512 @32k | tg64 @0 | tg64 @8k | tg64 @32k |
|---|---|---|---|---|---|---|---|---|---|
| this branch | 5218 | 4204 | 2613 | 4274 | 3359 | 2161 | 127.6 | 113.5 | 102.0 |
| upstream | 4567 | 3548 | 1958 | 4164 | 3220 | 1913 | 126.9 | 108.5 | 86.3 |
| ratio | 1.14x | 1.18x | 1.33x | 1.03x | 1.04x | 1.13x | 1.01x | 1.05x | 1.18x |

| tokens per step | 1 | 2 | 4 | 8 | 16 | 24 | 32 | 48 | 64 | 96 |
|---|---|---|---|---|---|---|---|---|---|---|
| this branch | 8.1 | 10.1 | 12.2 | 15.7 | 22.4 | 24.1 | 26.2 | 33.7 | 35.5 | 54.9 |
| upstream | 8.2 | 10.1 | 12.2 | 15.5 | 24.9 | 45.6 | 41.9 | 46.0 | 48.2 | 53.9 |
| upstream / ours | 1.00x | 1.00x | 1.00x | 0.99x | 1.11x | 1.89x | 1.60x | 1.36x | 1.36x | 0.98x |

| streams | 1 | 2 | 3 | 4 | 1, speculative | 4, speculative |
|---|---|---|---|---|---|---|
| this branch | 122.9 | 190.7 | 261.9 | 321.2 | 98.9 | 202.6 |
| upstream | 122.4 | 189.9 | 262.3 | 311.3 | 102.3 | 171.0 |
| ratio | 1.00x | 1.00x | 1.00x | 1.03x | 0.97x | 1.18x |

App start-up: this branch ready in 2.2 s, upstream 2.3 s.

### 3.3 Summary

- Prefill: at 8k end to end, Qwen 1.69x, gemma 1.22x; llama-bench `pp8192 @8k` 1.77 / 1.21x (the previous `pp2048` round was 1.5-2.0x).
- Decode: on par with upstream at short context (0.99-1.01x), faster at long context: at 30k Qwen +4%, gemma +18% (H3, the GQA-packed attention). Measured before upstream #29510; the GQA gain through the `kv_rows` lists is in section 6.5.
- Speculative decoding and small batches: steps of 16-64 tokens are 1.0-2.7x upstream (H4 / M1); at 30k, Qwen MTP 41.2 vs 34.5 t/s, gemma DFlash 117.2 vs 83.9 t/s.
- Concurrency (the app's 4-slot layout): 2-4 streams 1.0-1.22x in total, 4 speculative streams Qwen 1.36x, gemma 1.18x. The 0.97x for a single speculative gemma stream is a difference in acceptance on this one prompt (0.206 vs 0.227 with the A2 gate off, slightly different prompt numerics lead to different generated text), not a slower kernel: the only work the gate touches there is the 342-token prompt prefill.
- Against oMLX: 8k prefill Qwen 4% slower, gemma 10% slower; 30k decode (oMLX with MTP off) Qwen 10% slower, gemma 57% faster; speculative MTP vs MTP: Qwen 13% slower (41.2 vs 47.1), gemma 21% faster (117.2 vs 97.0).
- Start-up and memory (previous round): Qwen ready in 1.9 vs 2.7 s, resident memory 3.3 vs about 21 GB; turning thinking off no longer reprocesses the whole conversation.
- Accuracy: the changes other than I8 match upstream to the order of the reference; I8 has a visible cost on gemma and can be switched off alone (section 9).

## 4. Method

- **Account for the gap first.** Before any change, op classes were skipped one at a time with a debug switch to measure their time and compare it with the hardware limit; only parts with a limit clearly above 1.5% were worked on. The same applies to gemma: most of its MoE prefill is in the expert mat-muls (MUL_MAT_ID), which ran at half the throughput of the dense path; M1 / M2 came from that.
- **Screen, then confirm.** Compute-bound candidates were first A/B tested at kernel level on synthetic weights (interleaved in one process); only a gain above 1.5% in a consistent direction over several rounds went on to real-model measurements. Every kept change was confirmed at full power (`powermode 0`), and every number in this document is a full-power number.
- **Interleaved measurements.** Absolute values drift 5-15% between sessions with the thermal state. Every comparison runs the configurations forward and then in reverse within one session, and only ratios within a session are compared. Tests are 2 minutes apart; test blocks with too much drift were discarded and repeated.
- **Leave-one-out selection.** A change is judged by its contribution to the final combination (all on minus this one), not only by its effect alone.
- **Per format and per shape.** Mat-mul changes were A/B tested at kernel level for every weight format separately on synthetic weights (dense: 17408x5120 and 5120x17408; MoE: the expert shapes of gemma-4), and enabled only for the formats and row ranges that are actually faster; the output checksums of both paths were compared.
- **Per model structure.** Attention changes were measured on the Qwen and gemma structures: 6 / 8 query heads per KV head, head sizes 128 / 256 / 512. Cases that get slower on some structure are excluded by a threshold (for example H3 for single-token decode with 4 query heads per KV head).
- **Accuracy gate.** For every model the KL divergence to the full upstream logits is measured (paired per token, far more sensitive than perplexity), with the difference between upstream's own `-fa on` and `-fa off` as the reference for an ordinary numerical difference; a lossy change is either brought close to that reference or its cost is stated and it can be switched off on its own (section 9).
- **Fixed prompts.** Speculative decoding speed depends on how many drafts the generated text accepts; end-to-end comparisons of single changes use fixed prompts and compare times only after checking that the output text is identical.
- **Upstream check.** Every round is rebuilt and re-measured on upstream `master`; this document is based on `67107ac34f` (b12164).

## 5. Candidates found by research

Sources: MLX (steel attention / gemm NAX kernels, the `gated_delta` kernels, `gather_qmm`), oMLX 0.7.1, uzu, BaseRT, MTPLX, dflash-mlx, mlx-dspark, vllm-mlx, mlx-vlm, FastVLM, llama.cpp's own CUDA backend (MMQ), and llama.cpp's server and Web UI code.

| Area | Candidate | Source | Outcome |
|---|---|---|---|
| Prefill attention | Register-resident NAX flash attention | MLX `steel_attention_nax` | kept (A1) |
| Prefill mat-mul | Half activations for the tensor mat-mat | own | kept (A2) |
| Prefill mat-mul | Register-resident NAX mat-mat, split-K for long K | MLX `steel_gemm_fused_nax` / split-K | kept (A3) |
| Prefill GDN | Fewer lanes per state row in the recurrence | oMLX `gated_delta_pipelined` | kept (A4) |
| Prefill mat-mul | INT8 activation mat-mat | CUDA MMQ, uzu, oMLX oq_a8 | kept (I8) |
| Prefill | Activations written directly in the input format of the next mat-mat | own | kept (A5) |
| Multi-turn cache | Turning thinking off does not change the rendered prompt | own | kept (S1) |
| Start-up | No zero-fill of fresh KV / recurrent state at load | own | kept (S2-a) |
| Start-up | First residency request in the background | own | kept (S2-c) |
| Concurrency / speculative | Cross-row mat-vec for 2-3 rows | own | kept (K3) |
| Short prompts | No f16 activation conversion for small batches | follow-up of A2 | kept (A2 gate) |
| Speculative / long context | Query heads sharing a KV head packed into one NAX attention tile | idea of MLX multi-row attention, own | kept (H3) |
| Concurrency / speculative / short prompts | Mat-mul on NAX for 2-64 rows, one kernel for every format | own, on top of upstream few-row MMA | kept (H4) |
| MoE small batches | Threshold of the MoE mat-mat | own | kept (M1) |
| MoE prefill | More entries per tile, fewer dequantizations of the expert weights | idea of MLX `gather_qmm`'s per-expert segments | kept (M2) |

Out of scope: KV quantization (turboquant), GPU sampling, MTL4 command encoding, media caching.

## 6. Per-change results and selection

The per-change and kernel figures in section 6 come from the previous round; this round re-measured the 8k prefill, the 30k decode and the 30k speculative decode (see section 3).

Selection rule: a single-change gain above 1.5% in a consistent direction over several rounds, except for changes that are a prerequisite of another significant optimization.

### 6.1 Prefill, dense model (Qwen3.8-27B Q5_K_M, llama-bench `-fa 1 -ub 2048`, pp2048 at 8k context)

"Alone" = only this change on, everything else off (A3 needs A2; I8 needs A2 and A3; A5 needs A3). "Leave-one-out" = all on except this one. Forward and reverse rounds, mean of the per-round ratios.

| # | Alone (vs all off) | Leave-one-out |
|---|---|---|
| A1 | +4.5% | +3.5% |
| A2 | +12.1% | (prerequisite of A3/I8) |
| A3 | +10.9% on top of A2 | (prerequisite of I8) |
| A4 | +1.8% | +3.4% |
| I8 | +26.2% on top of A2+A3 | +32.2% |
| A5 | +0.4% on top of A2+A3 | +2.7% |
| **All on** | **1.73x upstream** | |

A4 alone gives only +1.8% but +3.4% in the full combination: once everything else is faster, GDN takes a larger share of the time. Likewise A5.

Kernel level (interleaved A/B in one process, 2048 rows):
- A1: flash attention q2048 x kv8192 21.6 -> 11.8 ms, kv32768 100 -> 55 ms (vs the upstream tensor-API kernel).
- A3: f16 GEMM 17408x5120 7.1 -> 6.1 ms (59.7 TFLOP/s, on par with MLX's dense f16 NAX gemm); quantized formats +8-17% on gate/up shapes, +15-22% on the down shape (K = 17408, split-K).
- A4: GDN per 2048 tokens 192 -> 97 ms.
- I8: per mat-mat the int8 path is 1.3x the f16 NAX path (raw int8 NAX throughput is 2.1x f16).

### 6.2 Attention structures and MoE (the extensions of A1 / H3, M1, M2)

| Change | Model | Effect (on vs off) |
|---|---|---|
| A1 extended to head size 512 | gemma-4-26B-A4B | pp2048 @0 / @8k / @32k: +0.4% / +3.4% / +12% |
| H3 (decode, 2k / 8k / 16k / 32k) | Qwen3.8-27B (6 query heads per KV head) | +5% / +3% / +6% / +8% |
| | gemma-4-26B-A4B (8) | +1.5% / +6% / +9% / +18% |
| H3 (a step of 4 tokens, i.e. speculative verify) | Qwen / gemma | +6% to +55% (2k-32k), 1.2-2.0x at 32k |
| M1 (a step of 16 / 20 / 24 / 28 tokens) | gemma-4-26B-A4B | +6% / +14% / +23% / +30% |
| | gemma-4-26B-A4B | 0 / +4.5% / +5-9% / +3-4% |

- H3: measured before upstream #29510, on the dense cache path (section 6.5).
- M1: below 32 tokens upstream reads the weights of an expert once per (token, expert) entry; with the mat-mat, up to 32 entries of a tile share one read. At 12 tokens the two models disagree (-3% / +7%), so the threshold is 16.
- M2: the upstream MoE mat-mat has tiles of up to 32 entries, and every tile dequantizes the expert weights once. From 128 entries per expert on average, tiles of 64 entries are faster for all 25 formats and all three expert shapes (+5-35%, section 7.3); with fewer entries the half-empty tiles cost more than they save, so they are used only from there. gemma's pp1024 (64 entries per expert) is below the threshold and stays unchanged.

### 6.3 Decode and serving (Qwen3.8-27B Q5_K_M)

| # | Effect (on vs off) | How |
|---|---|---|
| S1 | Tokens processed again after turning thinking off in a 7k-token conversation: 7330 -> 4 (prefill 38.3 s -> 0.37 s) | Qwen3.8 chat template, functional change |
| S2-a | Ready 2.96 -> 2.03 s (-31%); resident memory 21 GB -> 3.3 GB | app arguments, 2 runs each |
| S2-c | Ready 2.24 -> 2.01 s (-10%) | same |
| K3 | Time per step: 2 rows 57.7 -> 42.5 ms (+36%), 3-32 rows +3-5% | llama-bench, forward + reverse |
| H4 | Time per step: 12 / 16 / 24 / 32 / 48 / 64 rows 90.9 / 93.5 / 200.3 / 199.3 / 201 / 207 -> 52.8 / 56.0 / 74.3 / 82.0 / 132 / 145 ms; 2-8 rows +5-16% | same |
| H3 | Decode with MTP: 30k 27.9 -> 31.2 t/s (+11.9%), 8k 26.3 -> 27.5 (+4.8%); output text byte-identical | fixed prompts, 2 rounds each way, before #29510 |
| A2 gate | f32 activations below 128 rows, avoiding the extra conversion dispatch; time per step 2-6% faster for 12-96 rows | llama-bench |
| Concurrency (K3: 2 streams; H4: 4 streams with MTP) | 2 streams +18% (K3), 4 streams with MTP +42% (H4 and H3) | ratios within a round |

S2-c depends on S2-a: zero-filling touches the pages, so the buffer is no longer "fresh" and S2-c has no effect when only S2-a is off. K3 and H4 split the work by format and row count (section 7): q5_K takes K3 at 2 rows and H4 from 3 rows.

### 6.4 Data for the remaining changes

The numbers for the following changes come from their commit messages, measured on the same machine.

- **A6 NAX attention for a BF16 K/V cache**: the prefill and GQA kernels read the BF16 cache as stored and convert it to f16 in registers (clamped to the f16 range); KL divergence to the F16-cache logits: Qwen 0.00054, gemma 0.056, the same order as the tensor kernel on the same BF16 cache.
- **A8 NAX mat-mat for BF16 weights**: from 1024 rows the weights are converted to f16 once and take the NAX path; the int8 variant is not used for them.
- **A7 NAX attention for the MLA shapes 192/128 and 576/512**: the K and V head dims are split separately; the GQA kernel still handles equal head sizes only.
- **A9 NVFP4 / TQ1_0 weights**: these two formats used to fall back to the CPU on Metal; they now take mat-vec, the few-row MMA kernel, mat-mat, MoE mat-mat and the NAX paths, with 58 / 78 new test-backend-ops cases.
- **H5 GQA attention for quantized K/V caches**: a q4_0 / q4_1 / q5_0 / q5_1 / q8_0 / iq4_nl cache is read as stored and dequantized in registers; with a q8_0 cache at tg64 @32k: gemma 93.3 vs 73.0 for the vec kernel (before #29510).
- **H3b GQA kernel rewrite**: up to 8 tiles of 16 rows per threadgroup (4 for head size 512), dispatched in blocks along the grid, with the number of KV splits taken at run time as min(32, 1024/ne01); a step of 16 tokens at 32k: gemma 188 -> 427 t/s (before #29510).
- **A3b int8 threshold lowered to 448 rows**: 448-896 rows, +12-47% per mat-mul.
- **A5b the fused GLU writes in place**: the activations go into the mat-mat's own f32 result, which removes the race with the released gate/up buffers.
- **A10 BF16 + head size 512 / 576 + more than 32 queries** yields the prefill to the tensor kernel: gemma pp2048 at 8k / 32k context is 6% / 16% faster upstream.

### 6.5 GQA attention after upstream kv_rows

Upstream #29510 (`1b6935b227`, "flash_attn_ext_rows", merged 2026-11-03) gives FLASH_ATTN_EXT a list of K/V rows (`kv_rows`) for every ubatch of a unified KV cache when flash attention is on. llama-server uses a unified KV cache by default, so this is the path of Llama.app and the server. The first version of the GQA kernel read the cache by position, so it could not run there. Upstream has no GQA kernel of its own (PRs #30645 and #30728 are still open).

So the kernel learned the lists: each lane looks up the cache row of its K/V columns per block. It takes the small batches the vec kernel would take; larger batches keep upstream's gather to F16 and the dense kernel (A1 on the gathered rows). The decode numbers of section 3 and the H3 / H5 / H3b numbers of sections 6.2-6.4 were measured before #29510, on the dense cache path.

Through `kv_rows` (Qwen3.8-27B Q5_K_M, `llama-batched-bench -fa on -kvu -ntg 64`, M5 Max, Automatic power mode, GQA on vs `GGML_METAL_FA_GQA_DISABLE=1`, 4 runs each, on and off alternated, medians, decode t/s):

| Depth | Sequences | GQA on | GQA off | Gain |
|---|---|---|---|---|
| 8192 | 1 | 21.9 | 20.4 | +7% |
| 8192 | 4 | 59.0 | 49.8 | +18% |
| 32768 | 1 | 21.2 | 16.8 | +27% |
| 32768 | 4 | 43.8 | 38.5 | +14% |

These runs are noisy: Spotlight indexing and another app ran in the background, and the absolute values spread widely. 14 of the 16 on/off pairs favoured on. Speculative verify (one sequence, steps of 4 tokens) through `kv_rows` was not measured. `test-backend-ops -o FLASH_ATTN_EXT` passes 5100/5100 on Metal, with two new `kv_rows` cases (head size 256, 2 KV heads, 6 query heads per KV head, the Qwen3.8-27B shape; F16 and Q8_0) that run the GQA kernel.

## 7. Weight formats

### 7.1 Formats covered by each change

- **Independent of the weight format**: A1, H3 (attention, F16 K/V; H5 adds q4_0 / q4_1 / q5_0 / q5_1 / q8_0 / iq4_nl caches, through `kv_rows` without iq4_nl and BF16), A4 (GDN), A2 and the A2 gate (activation conversion), S1, S2-a, S2-c.
- **A3 / I8 / A5 (prefill, >= 1024 rows)**: F16 and 23 quantized formats: q4_0, q4_1, q5_0, q5_1, q8_0, q1_0, q2_0, mxfp4, q2_K, q3_K, q4_K, q5_K, q6_K, iq1_s, iq1_m, iq2_xxs, iq2_xs, iq2_s, iq3_xxs, iq3_s, iq4_nl, iq4_xs, tq2_0 (I8 without F16). The weights go through the dequantize function of each format into f16 first, so each format needs only one dequantize instance.
- **M1 / M2 (MoE)**: every format of the upstream MoE mat-mat; M2 was measured on 25 formats one by one (section 7.3).
- **H4 (2-64 rows)**: all 26 formats of the few-row kernel (the 23 above plus f32, f16, bf16), enabled per format and row count:

| Rows | Formats enabled |
|---|---|
| 33-64 | up to a per-format limit: 64 (q4_0, q4_1, q5_0, q5_1, q8_0, q2_K, q4_K, q5_K, q6_K, f16, bf16, tq2_0, q1_0, q2_0), 56 (mxfp4, iq4_nl, iq4_xs, iq2_xxs, iq2_s), 48 (q3_K, iq3_xxs, iq3_s, iq1_s), 40 (iq2_xs, iq1_m); not f32 |
| 17-32 | all 26 |
| 9-16 | all quantized formats (f32 / f16 / bf16 are not faster and keep the old path) |
| 2-8 | the iq formats, q1_0, q2_0 (from 2 rows); q2_K, tq2_0, q5_K (from 3 rows); q6_K (2-4 rows) |

- **K3 (2-3 rows)**: q4_K (2-3 rows), q5_K (2 rows).

### 7.2 Kernel-level data, dense mat-mul (synthetic weights, M5 Max, full power, speed relative to the old path)

| Format | 2-8 rows: H4 / old | 9-16 rows: H4 / MMA | 17-32 rows: H4 / mat-mat | 33-64 rows: H4 / mat-mat | 2048 rows: A3+I8 / mat-mat |
|---|---|---|---|---|---|
| f32 | 0.99-1.01x | 1.01-1.02x | 1.10-1.14x | - | - |
| f16 | 0.97-0.99x | 0.98x | 1.60-1.62x | 1.11-1.38x | already covered |
| bf16 | 0.99-1.00x | 0.97-0.99x | 1.63-1.68x | 1.19-1.50x | - |
| q1_0 | 1.01-1.56x | 2.13-2.41x | 2.67-3.32x | 1.42-1.96x | 1.42x |
| q2_0 | 1.16-1.49x | 2.17-2.28x | 2.65-3.06x | 1.43-1.79x | 1.44x |
| q4_0 | 1.00-1.17x | 1.64-1.74x | 2.35-3.11x | 1.35-1.74x | already covered |
| q4_1 | 0.99-1.22x | 1.58-1.62x | 2.27-2.89x | 1.27-1.77x | already covered |
| q5_0 | 0.92-1.00x | 1.71-1.72x | 2.27-2.56x | 1.23-1.48x | already covered |
| q5_1 | 0.94-1.02x | 1.62-1.66x | 2.17-2.44x | 1.17-1.42x | already covered |
| q8_0 | 0.92-0.93x | 1.09-1.13x | 1.90-2.16x | 1.21-1.46x | already covered |
| mxfp4 | 0.94-1.02x | 1.56-1.58x | 1.88-2.23x | 1.07-1.26x | 1.41x |
| q2_K | 0.93-1.40x | 1.77-2.04x | 2.38-2.91x | 1.35-1.63x | 1.46x |
| q3_K | 0.91-1.12x | 1.69-1.73x | 1.63-1.93x | 1.01-1.09x | 1.49x |
| q4_K | 0.80-1.04x | 1.48-1.49x | 2.13-2.52x | 1.23-1.45x | already covered |
| q5_K | 0.88-1.22x | 1.62-1.74x | 1.92-2.37x | 1.14-1.37x | already covered |
| q6_K | 0.99-1.68x | 1.55-1.56x | 1.73-2.08x | 1.11-1.20x | already covered |
| iq1_s | 1.10-1.44x | 1.94x | 1.75-2.13x | 1.00-1.23x | 1.45x |
| iq1_m | 1.00-1.31x | 1.92-2.00x | 1.71-2.07x | 1.01-1.16x | 1.43x |
| iq2_xxs | 0.96-1.25x | 1.86x | 1.65-2.24x | 1.11-1.23x | 1.41x |
| iq2_xs | 1.02-1.42x | 1.93-2.06x | 1.65-1.98x | 0.99-1.14x | 1.41x |
| iq2_s | 1.08-1.32x | 1.71-1.94x | 1.97-2.29x | 1.04-1.27x | 1.42x |
| iq3_xxs | 1.05-1.25x | 1.83-1.87x | 1.75-2.21x | 1.00-1.19x | 1.45x |
| iq3_s | 1.06-1.23x | 1.87-1.91x | 1.63-2.10x | 1.01-1.18x | 1.43x |
| iq4_nl | 0.98-1.15x | 1.57-1.65x | 1.84-2.34x | 1.06-1.27x | 1.45x |
| iq4_xs | 0.99-1.17x | 1.68-1.71x | 1.83-2.23x | 1.07-1.16x | 1.44x |
| tq2_0 | 0.65-1.25x | 2.06-2.09x | 2.51-2.95x | 1.41-1.71x | 1.43x |

- Shapes 17408x5120 and 5120x17408, mean of two rounds; ranges are the minimum and maximum over the row counts. The 2-8 rows column is measured for every format; only the formats and row counts listed in 7.1 are actually enabled. The 33-64 rows column covers the enabled range.
- "Already covered" means the format was supported by A3/I8 from the start; see section 6.1.
- Cross-row mat-vec (K3) for 2-3 rows: q5_K 21% faster than the old path at 2 rows, 6% at 3 rows (H4 is faster at 3 rows, so 3 rows use H4).

### 7.3 Kernel-level data, MoE mat-mul (M2: tiles of 64 entries vs 32, synthetic weights, random routing)

Shapes: gemma-4 gate_up (8 of 128 experts, K = 2816, N = 1408) and down (K = 704, N = 2816). Shapes whose K is not a multiple of 256 cannot hold the K-quant and IQ formats, which are therefore measured on gemma gate_up only. T = 128 / 512 / 2048 tokens, i.e. 8 / 32 / 128 entries per expert. Median of two rounds.

| Format | gemma gate_up (T = 128 / 512 / 2048) | gemma down |
|---|---|---|
| mxfp4 | +2%/+22%/+32% | -8%/-11%/+24% |
| q4_0 | -7%/+14%/+14% | -25%/-3%/+10% |
| q4_1 | -8%/-12%/+12% | -3%/+28%/+9% |
| q5_0 | -5%/+11%/+31% | -7%/+35%/+5% |
| q5_1 | -4%/+13%/+29% | -6%/-18%/+24% |
| q8_0 | +3%/-0%/+20% | -1%/+9%/+13% |
| f16 | +1%/+3%/+16% | -0%/+4%/+6% |
| bf16 | +1%/+3%/+9% | +2%/+4%/+8% |
| q2_K | -3%/+28%/+20% | - |
| q3_K | +202%/+13%/+33% | - |
| q4_K | -5%/-8%/+24% | - |
| q5_K | -3%/+18%/+32% | - |
| q6_K | +12%/+11%/+20% | - |
| iq2_xxs | -9%/+18%/+21% | - |
| iq2_xs | -4%/+15%/+19% | - |
| iq2_s | +10%/+12%/+19% | - |
| iq3_xxs | -26%/+23%/+21% | - |
| iq3_s | +12%/-16%/+24% | - |
| iq1_s | -5%/+19%/+19% | - |
| iq1_m | +15%/+27%/+21% | - |
| iq4_nl | -15%/+9%/+29% | -9%/-6%/+21% |
| iq4_xs | -13%/+14%/+28% | - |
| tq2_0 | +48%/+20%/+21% | - |
| q1_0 | -3%/+20%/+8% | - |
| q2_0 | +4%/-27%/+12% | -5%/-22%/+10% |

### 7.4 Bandwidth of the decode mat-vec (single token)

The mat-muls of single-token decode are bandwidth bound, and the existing mat-vec kernels are already close to the sustainable bandwidth for each format: f16 about 545 GB/s, q4_K 528, q8_0 524, q5_K 501-531, q6_K 507-520 (same session, synthetic weights). Tuning the q5_K / q6_K kernel parameters gave at most +0.9%, so the single-token path is unchanged.

## 8. Implementation of the kept changes

**A1: NAX flash attention** (`kernels/fa_tensor.metal`, dispatch in `ggml-metal-ops.cpp`). Prefill attention with F16 K/V keeps the scores, the probabilities and O in registers (16x32x16 `matmul2d` fragments, BQ = 64, BK = 32), following MLX's `steel_attention_nax`; the head dim is split over WN simdgroups (1 for head size 128, 2 for 256, 4 for 512), whose partial scores are added through threadgroup memory in a fixed order. The mask flag of each block is prefetched one block ahead: otherwise the branch that skips fully-masked blocks waits for a device memory load every iteration (+4-8% end to end from this alone). ALiBi and softcap fall back to the existing tensor kernel.

**A2: f16 activations** (`kernels/mul_mm.metal`, `ggml-metal-ops.cpp`, `ggml-metal.cpp`). The tensor `mul_mm` kernels are about 1.25x faster with half activations than with f32, so the f32 activations are converted first (saturating at +/-65504) into an extra buffer. Dense mat-mat only (the MoE variant was evaluated separately).

**A3: NAX mat-mat** (`kernels/mul_mm.metal`, `ggml-metal-impl.h`, `ggml-metal-ops.cpp`). For batches of at least 1024 rows (`GGML_METAL_MM_NAX_MIN_ROWS`, a test parameter), each mat-mat first dequantizes the weights to f16 once through the dequantize function of the format, then multiplies on a kernel without threadgroup memory: operands are loaded straight into fragments, each 64x128 tile is computed by 2x4 simdgroups, threadgroups are swizzled, one barrier per 256 K. Not unrolling the K loop is what makes it fast (0% -> 15%). When K >= 3*max(M, N), K is split into 4096-wide partitions dispatched one after another and accumulated into dst, so that the operand slices stay in the cache (MLX does the same for long K).

**A4: GDN with 4 lanes per row** (`kernels/gated_delta_net.metal`, `ggml-metal-ops.cpp`). With 32 lanes per state row, the recurrence was instruction bound (two 5-step reductions per token). With 4 lanes per row (8 for state size 256, otherwise each lane would need 64 registers) each lane holds 32 state values and a reduction takes 2 steps (8 lanes: 1.87x, 4 lanes: 1.98x). Other shapes, non-scalar gates and unaligned inputs keep the existing kernel.

**I8: int8 mat-mat** (`kernels/mul_mm.metal`: `kernel_mul_mm_q8a_f32`, `kernel_mul_mm_dq_<type>_i8`, `kernel_mul_mm_nax_i8`). Activations and dequantized weights are quantized to symmetric int8 with one scale per (row, 128 columns); before quantizing, each group of 128 values goes through an unnormalized Walsh-Hadamard transform (`i8_wht128`, 4 values in each of 32 lanes, 2 in-lane add/subtract stages + 5 `simd_shuffle_xor` stages). Both sides use the same transform, so the dot product is kept (the 1/128 is folded into the activation scale), while outliers are spread over the group instead of inflating its scale. `matmul2d<int8, int8, int32>` accumulates a 128-wide group of K, which is then folded into the f32 accumulators (`c += s_a*s_w*ci`). Same tiling and split-K as A3; it reuses the buffer reserved by A3 and needs no extra memory. One scale per 32 columns is also more accurate, but folds 4x as often and makes Qwen pp2048 25% slower, so the groups stay at 128 columns and the transform recovers the accuracy: KL divergence Qwen 0.0023 -> 0.0007, Llama 0.0009 -> 0.0004, gemma 0.112 -> 0.067, at a speed cost of 0-2%. This is the same kind of lossy shortcut as the q8_1 activations of the CUDA MMQ path and oMLX's oq_a8 (accuracy in section 9).

**A5: SWIGLU -> down projection** (`ggml-metal-fusion.cpp`, `kernels/unary.metal`). When the output of SWIGLU feeds a NAX mat-mat, the GLU kernel writes the f16 or int8 activations (with scales) that mat-mat needs directly; the f32 GLU output and the separate conversion pass are gone. Results are bit-identical to the unfused path.

**S1: keep the conversation cache when thinking is turned off** (`tools/ui`). The interface's thinking levels low / medium / high were already sent as `thinking_budget_tokens` and switching between them keeps the cache; only "off" sent `chat_template_kwargs.enable_thinking=false`, which makes the Qwen3.8 template drop a paragraph at the start of the system part, so the whole conversation was processed again from about the 5th token. "Off" now sends `thinking_budget_tokens: 0`, which gives the same output as the template's empty thinking block with an unchanged prompt prefix. A new setting "Keep the conversation cache when thinking is off" (Settings -> Developer, on by default).

**S2-a: no zero-fill at load** (`ggml-metal-device.cpp`, `common/common.cpp`). Metal shared buffers come from `vm_allocate` and are already zero, but the KV cache was cleared once after allocation and again after warm-up, touching and committing every page (about 16 GiB of KV and 2.4 GiB of recurrent state at 262k context). The Metal side records whether a buffer has been touched since allocation and skips the clear if not; the `common` side only resets the KV cells after warm-up instead of wiping the data.

**K3: cross-row mat-vec for 2-3 rows** (`kernels/mul_mv.metal`). Same lane layout as the single-row mat-vec, but the quants and scales of a block are unpacked once and dotted with the 2-3 activation rows; the unpacked values are kept in half to limit register use. q4_K for 2-3 rows, q5_K for 2 rows.

**S2-c: first residency request in the background** (`ggml-metal-device.cpp`). The first `requestResidency` of a residency set maps every page for the GPU, about 0.4 s for the KV of a 262k context. For shared buffers untouched since allocation the request runs on a background queue, after which the set joins the keep-alive set; freeing the buffer waits for it.

**A2 gate** (`ggml-metal-ops.cpp`). A2 converts the f32 activations to f16 before the tensor mul_mm, one more dispatch with two barriers per mul_mm, a fixed cost; below 128 rows it costs more than half activations save, and so it does at 128-512 rows for small products (K/V projections, the small recurrent-state projections). The conversion now starts at 128 rows and 4 GFLOP per product (the NAX mat-mat takes f16 only and still converts), and the extra buffer is reserved under the same condition. Thresholds of 2 / 4 / 8 / 16 GFLOP were swept: 4 is the one that is not slower on Qwen and gemma at pp128 / 256 / 512.

**H3 / H3b: GQA-packed NAX attention** (`kernels/fa_tensor.metal`: `kernel_flash_attn_ext_nax_gqa`; `ggml-metal-ops.cpp`). In decode and speculative verify the vec kernel reads K/V once per query head. The new kernel packs the G query heads that share a KV head and the ne01 tokens into the rows of one NAX tile (p = g*ne01 + i), so each K/V block is read once; each threadgroup packs up to 8 tiles of 16 rows (4 for head size 512), dispatched in blocks along the grid, and the KV range is split over them as min(32, 1024/ne01); the results are written in the temp-buffer layout of the vec kernels and combined by the existing `kernel_flash_attn_ext_vec_reduce`. Conditions: head size 128-512, F16 K/V, a mask, no ALiBi / softcap, at least 2 query heads per KV head; with fewer than 5 packed rows (single-token decode with 4 query heads per KV head) and KV shorter than 16k, most of the tile would be empty rows and the vec kernel stays. With `kv_rows` (section 6.5) each lane looks up the cache row of its K/V columns per block; padding rows (< 0) are clamped and masked, and all q slices share one K/V stream. Larger batches keep upstream's gather to F16 and the dense kernel.

**H4: NAX few-row mat-mat for 2-64 rows** (`kernels/mul_mm.metal`: `kernel_mul_mv_nax_f32`; `ggml-metal-common.cpp`, `ggml-metal-ops.cpp`). The src1 rows are the M = 16 rows of a 16x32x16 matmul2d (one or two 16-row tiles per pass, more rows as further passes along the second grid dimension), the 32 weight rows its N; each lane dequantizes 16 weights straight into a half fragment through the dequantize function of the format, without threadgroup memory; 4 simdgroups split K and are summed in threadgroup memory at the end. So that each lane dequantizes a whole 16-weight group at once (the unit of the dequantize functions), K is permuted within each 64-weight chunk, identically for src0 and src1. q5_K has its own dequantize function with 16-byte loads and bit operations instead of per-weight branches. One function decides which row counts are enabled for each format (section 7.1). The residual ADD fuses into its store (MUL_MAT + ADD).

**M1: MoE mat-mat from 16 tokens** (`ggml-metal-common.cpp`). `ggml_metal_op_mul_mat_id_use_mm` lowers the threshold from 32 to 16 on devices with tensor units; others keep 32.

**M2: MoE mat-mat tiles of 64 entries** (`kernels/mul_mm.metal`: `kernel_mul_mm_id_map0`, `kernel_mul_mm_id`; `ggml-metal-device.cpp`, `ggml-metal-ops.cpp`). The routing prepass cuts the entries of each expert into tiles, and the main kernel dequantizes 64 weight rows once per tile and multiplies them with the entries of the tile. A function constant switches the tile size from 32 to 64 in the prepass and the kernel together: a tile of 64 entries is multiplied in quarters of 16 rows with `matmul2d`, and quarters without entries are skipped; each thread loads one more activation row; threadgroup memory grows from 8 KB to 16 KB. Used only on devices with tensor units and when the experts get at least 128 entries on average.

**A6 / A7 / A8 / A10: BF16 and MLA** (`kernels/fa_tensor.metal`, `kernels/mul_mm.metal`, `ggml-metal-ops.cpp`). A BF16 cache is read as stored, clamped to f16 in registers and fed to the MMA; the MLA shapes 192/128 and 576/512 split the K and V head dims separately (K over 1 simdgroup, V over 4 for 576/512); BF16 weights are converted to f16 once from 1024 rows and then take the NAX path. A BF16 prefill with head size 512 / 576 and more than 32 queries (the GQA batch limit) still yields to the tensor kernel.

**A9 / A3b: more weight formats** (`kernels/mul_mv.metal`, `kernels/dequantize.h`, `ggml-metal-ops.cpp`). The UE4M3 sub-block scales of NVFP4 and the base-3 digits of TQ1_0 are decoded in the kernel, and both formats take every Metal path; the int8 NAX threshold drops from 1024 to 448 rows.

**H3b / H5: the GQA kernel** (`kernels/fa_tensor.metal`, `ggml-metal-ops.cpp`). Up to 8 tiles of 16 rows per threadgroup (4 for head size 512), dispatched in blocks along the grid; a quantized cache is read as stored with 4 values dequantized per load; the number of KV splits is min(32, 1024/ne01). H3, H5 and H3b are now one commit. With `kv_rows` the q4_0 / q4_1 / q5_0 / q5_1 / q8_0 caches go through it too; upstream does not accept iq4_nl or BF16 with `kv_rows`.

**A5b: the fused GLU writes in place** (`ggml-metal-fusion.cpp`). The activations go into the mat-mat's own f32 result, which removes the race with the released gate/up buffers. The environment variable of every change is listed in the switch column of section 2.

## 9. Accuracy

- `test-backend-ops -b MTL0`: MUL_MAT 3892/3892 (the few-row cases cover 2-64 rows for every format; also passes with `GGML_METAL_MM_NAX_MIN_ROWS=1`, which sends every eligible case through the NAX / int8 kernels), MUL_MAT_ADD 384/384, MUL_MAT_ID 867/867 (with tiles of 32 and of 64 entries), FLASH_ATTN_EXT 5100/5100 (including the NAX paths for head sizes 128 / 256 / 512 and MLA, and two `kv_rows` cases that run the GQA kernel), GATED_DELTA_NET 30/30. Every commit builds on its own and passes the tests relevant to it.
- KL divergence (the main measure): against the full upstream logits (`llama-perplexity --kl-divergence-base`, 8 x 2048 tokens, `-c 2048 -b 2048 -ub 2048 -fa on`, the llama.cpp repository docs as text), comparing this branch's output distribution token by token. Each cell is mean KL divergence / top-1 agreement. The reference columns are upstream against itself with an equivalent setting (`-fa off`, or `-ub 512`), i.e. an ordinary numerical difference:

| Model | this branch, all on | this branch, I8 off | reference: upstream -fa off | reference: upstream -ub 512 |
|---|---|---|---|---|
| Qwen3.8-27B Q5_K_M | 0.0007 / 99.00% | <0.0001 / 99.96% | <0.0001 / 99.96% | <0.0001 / 100.00% |
| gemma-4-26B-A4B Q4_0 | 0.0670 / 90.20% | 0.0236 / 94.89% | 0.0244 / 94.92% | 0.0042 / 98.90% |

  - The changes other than I8 are practically bit-identical to upstream on Qwen (KL divergence < 0.0001) and of the same order as the reference on gemma. gemma is more sensitive to any numerical difference: upstream's own `-fa on` vs `-fa off` already differ by 0.024. Switching the changes on one at a time (all off, then one on) attributes this to A1 (NAX attention) and A2/A3 (the f16 path), about 0.02 each; the other changes (H3, H4, K3, M1, M2, A5) are bit-identical to upstream. The results are deterministic: repeated runs and runs with concurrency off give exactly the same numbers.
  - I8 is the only change with a real cost: int8 activations add about 0.0007 on Qwen and about +0.043 on gemma (top-1 agreement 94.9% -> 90.2%), about 3x the reference. The Walsh-Hadamard transform before quantizing cut this error to a third (Qwen) to three fifths (gemma) of what it was, see section 8. I8 stays on by default because it is the largest single prefill gain on dense models (Qwen leave-one-out +32%, gemma +7%); where accuracy matters, `GGML_METAL_MM_I8_DISABLE=1` turns it off alone and leaves the other changes in place.
- Perplexity (same text and settings), all changes on vs upstream, as a complement:

| Model | this branch | upstream | difference | error |
|---|---|---|---|---|
| Qwen3.8-27B Q5_K_M | 3.4648 | 3.4652 | -0.01% | +/-0.08 |
| gemma-4-26B-A4B Q4_0 | 65.5324 | 64.6739 | +1.33% | +/-3.42 |

  The statistical error of perplexity is far larger than the KL divergence differences above, so it only rules out gross errors. Note that for gemma the `--kl-divergence-base` run reports a perplexity (62.71) different from a normal run of the same upstream binary (64.67) while the logits of the two runs are identical (KL divergence 0), so the gemma perplexity ratio is not reliable; the KL divergence is what counts.
- Greedy outputs: byte-identical between this branch and upstream in the start-up test; byte-identical with H3 on and off; A5 with the same KL divergence as the unfused path (Qwen 0.00070 vs 0.00073, within the error; the compiler contracts the multiply-add differently in the fused kernel, so it is no longer bit-identical); M1 and M2 bit-identical to the existing MoE path.

## 10. Remaining gaps

1. **MoE prefill.** gemma's 8k prompt is 9% slower than oMLX (the 30k prefill was not re-measured this round). Before M2, most of gemma prefill was MUL_MAT_ID at about half the throughput of the dense path; M2 made it 24-37% faster, but the per-expert mat-mat is still clearly below the dense path. The next step is a register-resident NAX kernel like A3 for the per-expert mat-mat (the per-expert few-row approach is slower at large batches).
2. **Qwen MTP.** At 8k, oMLX decodes with MTP at 57.5 t/s against 42.7 here (previous round; the 8k speculative decode was not re-measured this round); at 30k it is 47.1 vs 41.2. Without speculative decoding the two are 10% apart (25.0 vs 22.5), so the gap is mostly in the speculative layer (draft length, acceptance, the cost of the draft head itself). This project changed only the Metal backend and did not break that down.
3. **Decode bandwidth.** Short-context decode is on par with upstream but still short of the bandwidth limit: Qwen 2k decode 26.0 t/s x 19.5 GB ≈ 507 GB/s, i.e. 83% of the nominal 614 GB/s. oMLX is 10% faster on Qwen.
4. **I8 accuracy.** On gemma, I8 brings the KL divergence to about 3x the reference of upstream's `-fa on`/`-fa off` (section 9). Finer groups are too slow (one scale per 32 columns: Qwen -25%), the Walsh-Hadamard transform recovered part of it; a full fix needs a way to tell per matrix how the activations are distributed, for now it can only be switched off as a whole.
5. **Speculative verify through `kv_rows`.** Steps of 4 tokens on one sequence through the `kv_rows` lists are not measured yet (section 6.5).

## 11. Reproducing

```sh
# build (macOS, Metal)
cmake -B build -DCMAKE_BUILD_TYPE=Release && cmake --build build -j --target llama-bench llama-server llama-perplexity test-backend-ops

# numerical checks
build/bin/test-backend-ops -o MUL_MAT -b MTL0
GGML_METAL_MM_NAX_MIN_ROWS=1 build/bin/test-backend-ops -o MUL_MAT -b MTL0
build/bin/test-backend-ops -o MUL_MAT_ADD -b MTL0
build/bin/test-backend-ops -o MUL_MAT_ID -b MTL0
build/bin/test-backend-ops -o FLASH_ATTN_EXT -b MTL0
build/bin/test-backend-ops -o GATED_DELTA_NET -b MTL0

# prefill A/B: llama-bench defaults to -ub 512; the NAX mat-mat needs at least 1024 rows
build/bin/llama-bench -m model.gguf -fa 1 -ub 2048 -p 2048 -n 0 -d 8192
GGML_METAL_MM_I8_DISABLE=1 build/bin/llama-bench -m model.gguf -fa 1 -ub 2048 -p 2048 -n 0 -d 8192
GGML_METAL_MM_ID_WIDE_DISABLE=1 build/bin/llama-bench -m moe-model.gguf -fa 1 -ub 2048 -p 2048 -n 0

# time per step (2-64 tokens: K3, H4, M1) and long-context decode (H3)
build/bin/llama-bench -m model.gguf -fa 1 -ub 2048 -p 1,2,4,16,24,32,48,64 -n 0
GGML_METAL_MV_NAX_DISABLE=1 GGML_METAL_MV_XR_DISABLE=1 GGML_METAL_MM_ID_MIN16_DISABLE=1 build/bin/llama-bench -m model.gguf -fa 1 -ub 2048 -p 1,2,4,16,24,32,48,64 -n 0
build/bin/llama-bench -m model.gguf -fa 1 -p 4 -n 64 -d 32768
GGML_METAL_FA_GQA_DISABLE=1 build/bin/llama-bench -m model.gguf -fa 1 -p 4 -n 64 -d 32768
build/bin/llama-batched-bench -m model.gguf -fa on -kvu -npp 8192,32768 -ntg 64 -npl 1,4
GGML_METAL_FA_GQA_DISABLE=1 build/bin/llama-batched-bench -m model.gguf -fa on -kvu -npp 8192,32768 -ntg 64 -npl 1,4

# server with speculative decoding (no separate draft model needed when the model file has a built-in MTP head)
build/bin/llama-server -m model.gguf -fa on -np 4 -kvu --spec-type draft-mtp --spec-draft-n-max 3
build/bin/llama-server -m gemma-4-26B-A4B-it-Q4_0.gguf -fa on -np 4 -kvu --spec-type draft-dflash -md dflash-gemma-4-26B-A4B-it-Q8_0.gguf
```

## About the code

This document describes every change but does not explain the source line by line; the code is on the `metal-m5` branch of `LXD-8/llama.cpp`, based on upstream `master` at `67107ac34f` (b12164), one commit per change (H3, H5 and H3b are one commit), each of which can be switched off on its own. If you are considering adopting any of these optimizations, I am happy to submit a PR or to provide the code in any other form that suits you. English is not my native language; for a long text like this one, the Chinese version is the original and AI translated it into English.

---

# Apple M5（NAX）上的 llama.cpp Metal 优化：面向各种模型与量化格式

[English](#llamacpp-metal-optimizations-for-apple-m5-nax-for-every-model-and-weight-format) | [中文](#apple-m5nax上的-llamacpp-metal-优化面向各种模型与量化格式)

中文为原文，英文版由 AI 基于中文版翻译。/ The Chinese version is the original; the English version is an AI translation of it.

本项目优化 llama.cpp 的 Metal 后端在 Apple M5 上的性能，目标是让各种模型（稠密、MoE、不同注意力结构）和各种量化格式都基本达到这台机器的硬件上限。范围包括提示处理（预填充）、解码、投机解码（MTP / DFlash）、并发、短提示、启动和多轮对话缓存。全部改动基于上游 `master` 的 `67107ac34f`（b12164）：22 个代码提交，每项改动一个提交（H3、H5 与 H3b 合为一个提交），每项都可以单独关闭。

下表是全部改动开启时、与上游和 oMLX 的对比（M5 Max，满功率，服务器端到端，流式客户端，temperature 0；oMLX 开启其 INT8 预填充，两边都关闭 KV 量化）：

| 模型 | 8k 预填充 t/s：本分支 / 上游 / oMLX | 对上游 | 30k 解码 t/s：本 / 上 / oMLX | 对上游 | 30k 投机解码 t/s：本 / 上 / oMLX |
|---|---|---|---|---|---|
| Qwen3.8-27B Q5_K_M | 1086 / 642 / 1128 | 1.69x | 22.5 / 21.7 / 25.0 | 1.04x | 41.2 / 34.5 / 47.1 |
| gemma-4-26B-A4B Q4_0 | 4054 / 3313 / 4456 | 1.22x | 106.2 / 89.7 / 67.8 | 1.18x | 117.2 / 83.9 / 97.0 |

"投机解码"一列：Qwen 用模型内置的 MTP 头（`--spec-type draft-mtp --spec-draft-n-max 3`），gemma 用 ggml-org 同一仓库提供的 DFlash 草稿模型（`--spec-type draft-dflash`）；oMLX 两者都用模型自带的 MTP 头。

范围：22 个代码提交，26 个文件，+3396 / -122。21 个在引擎（以 `ggml/src/ggml-metal/` 为主，`common/` 一行，`tests/test-backend-ops.cpp` 增加测试用例），1 个在内置 Web UI（`tools/ui`，即 Llama.app 的界面）。NAX 内核只在支持 Metal tensor API 的设备上生效（默认为 M5 / A19 及更新的芯片），其他设备保持上游路径。除 NVFP4 / TQ1_0 这类格式本身没有单独开关外，每项引擎改动都可以用环境变量关闭，便于 A/B 测试。

测过的模型：Qwen3.8-27B Q5_K_M（稠密，GDN + 全注意力，q5_K / q6_K）、gemma-4-26B-A4B Q4_0（MoE，128 专家选 8，含 head size 512 的全局注意力层）。其余量化格式用 `test-backend-ops` 做了数值校验，并在合成权重上做了内核级计时（第 7 节）。

测试设备：MacBook Pro（Mac17,6），Apple M5 Max：18 核 CPU（6 个超大核 + 12 个性能核）、40 核 GPU、128 GB 统一内存；系统为 macOS 27.2。本文所有数据均在这台机器上测得。

### 关于本项目与 AI 的使用

本项目源于日常使用中的发现：在我的设备上，llama.cpp（Llama.app 的引擎）的预填充和解码明显慢于基于 MLX 的 oMLX。项目目标、调研思路、测试方法以及依据测试结果对方案的取舍，均由我本人决定。调研、编写代码、测试和撰写本文档的过程中使用了 AI 工具，其产出都经过我人工检查。本文档的中文版为原文，英文版由 AI 基于中文版翻译。

目录：

- [1. 为什么做](#1-为什么做)
- [2. 改动一览](#2-改动一览)
- [3. 效果](#3-效果)
- [4. 方法](#4-方法)
- [5. 调研到的候选方案](#5-调研到的候选方案)
- [6. 各项结果与取舍](#6-各项结果与取舍)
- [7. 量化格式](#7-量化格式)
- [8. 保留项的实现](#8-保留项的实现)
- [9. 数值精度](#9-数值精度)
- [10. 剩余差距](#10-剩余差距)
- [11. 复现](#11-复现)
- [关于代码](#关于代码)

## 1. 为什么做

在 M5 Max 上，原版 llama.cpp 的 Metal 后端处理提示的速度只有 oMLX 的约 55-70%（例如 Qwen3.8-27B 8k 提示 653 对 1187 t/s），MoE 模型和长上下文下差距更大。实际使用中还有几处明显的损失：投机解码每一步要验证几个 token，这一步比单 token 慢很多；App 默认的 4 路并发，合计吞吐远低于单路的 4 倍；追问、新一轮对话这类短提示落在一段很慢的批次大小上；App 启动后要 3 秒多才出第一个 token，并占用约 21 GB 内存；在界面里关闭思考会让整段对话被重新处理。

M5 这一代在每个 GPU 核心里加入了矩阵单元（NAX，通过 Metal tensor API / MetalPerformancePrimitives 的 `matmul2d` 调用）。MLX 用寄存器常驻的内核驱动它；上游 llama.cpp 只通过一个用 threadgroup 内存分块、以 f32 激活为输入的内核来用它，大部分新增算力没有用上。

目标：
- 尽可能接近 M5 Max 的硬件上限，在可公平对比的项目上追平或超过 oMLX；
- 合适的地方借鉴 MLX、oMLX 等 Apple Silicon 引擎的做法，也做出它们都没有的优化；
- 讲究性价比：单项最终增幅低于 1.5%（噪声级）、却需要复杂验证或大量改动的方案不做；保留的方案用必要最小的改动实现；收益大时允许较大的重构；
- 只改 Metal 后端（以及外面必要的最少几处），每项改动都可开关、都经过数值验证；
- 覆盖 Metal 矩阵乘内核支持的全部权重格式，以及稠密、MoE 和各种注意力结构，而不只是测试模型用到的情形。

在本机实测的硬件上限（M5 Max，40 核 GPU，128 GB）：

| | 数值 |
|---|---|
| 内存带宽（Apple 规格） | 614 GB/s |
| 实测可持续带宽 | 530-580 GB/s（随测试内核、传输大小和运行状态浮动） |
| NAX f16 mat-mat，可达 | ~60 TFLOP/s（MLX 稠密 f16 gemm 59-60；下文 A3 为 59.7） |
| NAX int8 mat-mat | ~2.1x f16（同样的寄存器常驻内核结构） |

解码受带宽限制：例如 Qwen3.8-27B Q5_K_M 每个 token 读 18.34 GB 权重；本分支单 token 解码中矩阵乘部分的读取速度为 517-540 GB/s，已在可持续带宽范围内，剩下的是注意力和小算子（第 10 节）。

## 2. 改动一览

按分支中的顺序，每项一个提交（H3、H5 与 H3b 是一个提交）。"格式"一栏说明矩阵乘类改动作用于哪些权重格式；其余改动与权重格式无关。

| # | 提交 | 改动 | 行数 | 格式 | 开关 |
|---|---|---|---|---|---|
| A1 | `a7f094140` | 寄存器常驻的 NAX flash attention（预填充，head size 128 / 256 / 512） | +461 / -9 | -（F16 K/V） | `GGML_METAL_FA_NAX_DISABLE` |
| A2 | `0a39acbcd` | tensor mul_mm 前的 f32 -> f16 激活转换 | +136 / -7 | F16 与量化格式 | `GGML_METAL_MM_F16_ACT_DISABLE` |
| A3 | `d42a64b1b` | 反量化到 f16 + 寄存器常驻的 NAX mat-mat，长 K 用 split-K（>= 1024 行） | +397 / -4 | F16 + 23 种量化格式 | `GGML_METAL_MM_NAX_DISABLE` |
| A4 | `d5346f1aa` | gated delta net 每状态行 4 个 lane | +157 / -2 | - | `GGML_METAL_GDN_R8_DISABLE` |
| I8 | `87b8d9db5` | int8 NAX mat-mat（每 128 列一个 scale，量化前做 Walsh-Hadamard 变换） | +276 / -7 | 23 种量化格式 | `GGML_METAL_MM_I8_DISABLE` |
| A5 | `eb3b224e3` | SWIGLU 融合进使用它的 NAX mat-mat | +213 / -7 | 同 A3 | `GGML_METAL_GLU_MM_DISABLE` |
| S1 | `866581197` | 关闭思考时保留对话缓存（Web UI） | +12 / -1 | - | 界面设置 |
| S2-a | `9510455c9` | 加载时不对新分配的 KV / 递推状态缓冲区清零 | +44 / -1 | - | `GGML_METAL_CLEAR_SKIP_DISABLE` |
| K3 | `8d0c74072` | 2-3 行的跨行 mat-vec | +316 / -1 | q4_K（2-3 行）、q5_K（2 行） | `GGML_METAL_MV_XR_DISABLE` |
| S2-c | `c1506b989` | 新缓冲区的首次驻留请求放到后台 | +44 / -5 | - | `GGML_METAL_RSET_ASYNC_DISABLE` |
| A2 门槛 | `2ff0114a8` | 128 行且 4 GFLOP 起才做 f16 激活转换 | +9 / -2 | 同 A2 | `GGML_METAL_MM_F16_ACT_ROWS_128_DISABLE` |
| H4 | `9c50c4081` | 2-64 行的 NAX few-row mat-mat | +321 / -10 | few-row 内核支持的全部格式，行数范围见第 7 节 | `GGML_METAL_MV_NAX_DISABLE` |
| M1 | `be6587734` | MoE mat-mat（MUL_MAT_ID）从 16 个 token 起使用（原为 32） | +9 / -8 | 全部格式 | `GGML_METAL_MM_ID_MIN16_DISABLE` |
| M2 | `a03983a37` | MoE mat-mat 每块 64 个条目（专家平均 >= 128 个条目时） | +90 / -25 | 全部格式 | `GGML_METAL_MM_ID_WIDE_DISABLE` |
| A6 | `8768dad0f` | BF16 K/V 缓存的 NAX flash attention（预填充与 GQA，寄存器内转 f16） | +36 / -21 | BF16 | `GGML_METAL_FA_NAX_BF16_DISABLE` |
| A8 | `9d2cb5214` | BF16 权重的 NAX mat-mat（1024 行起） | +20 / -6 | BF16 | `GGML_METAL_MM_NAX_BF16_DISABLE` |
| A7 | `7479b2600` | MLA 头尺寸 192/128、576/512 的 NAX flash attention | +34 / -20 | MLA | `GGML_METAL_FA_NAX_MLA_DISABLE` |
| A9 | `ca8a15013` | NVFP4 / TQ1_0 权重 | +308 / -7 | NVFP4、TQ1_0 | -（格式本身无开关） |
| A3b | `e11f1820d` | int8 的 NAX mat-mat 门槛降到 448 行 | +13 / -9 | int8 路径的量化格式 | `GGML_METAL_MM_NAX_MIN_ROWS` |
| A5b | `971b41a44` | 融合 GLU 把激活写进 mat-mat 自己的结果（修内存竞争） | +16 / -24 | 同 A3 | `GGML_METAL_GLU_MM_DISABLE` |
| A10 | `e46a6c395` | BF16 + head size 512 / 576 的大 batch 预填充让回 tensor 内核 | +5 / -2 | BF16 | `GGML_METAL_FA_NAX_BF16_DISABLE` |
| H3 / H5 / H3b | `06d72db6c` | 一个提交。H3：共享 KV 头的 query 头打包进一个 NAX 注意力 tile（解码、投机解码验证；head size 128-512）。H5：同样用于量化 K/V 缓存。H3b：最多 8 个 16 行 tile、行块、运行期 KV 份数。也能通过 `kv_rows` 列表读 K/V（第 6.5 节） | +526 / -56 | -（F16 / BF16 K/V）；q4_0 / q4_1 / q5_0 / q5_1 / q8_0 / iq4_nl 缓存 | `GGML_METAL_FA_GQA_DISABLE`、`GGML_METAL_FA_GQA_Q_DISABLE` |
| | | **合计** | **+3396 / -122** | | |

## 3. 效果

对 oMLX 的公平原则：凡是本分支的改动在 oMLX 里有同类选项的，oMLX 也打开：其 INT8 激活预填充（与 I8 对应）开启。其余改动（NAX 注意力、NAX mat-mat、GDN 内核、激活融合）对应的是 MLX/oMLX 一直在用、没有开关的路径。KV 缓存量化两边都关闭；不开投机解码的对比两边都关，开的对比两边都开（oMLX 的 MTP 可在模型设置里关闭，测量前切换、测量后恢复并核对）。oMLX 的设置在测量前读出、测量后恢复并核对。oMLX 在单独的时间段测量，所以对 oMLX 的比值是跨时间段的。

端到端：本轮 8k / 30k 提示实测 8,020 / 29,686 token（oMLX 分词不同，token 数略有出入），生成 128 / 256 / 256 token；2k / 16k 与并发、启动的旧数字保留并标注未复测。本分支与上游正序、反序各一轮取均值。llama-bench：`-fa 1`，本轮 8k 预填充用 `-p 8192 -d 8192 -ub 2048`（llama-bench 默认的 512 低于 NAX mat-mat 1024 行的门槛），正反序各一轮。

### 3.1 Qwen3.8-27B Q5_K_M

`Qwen3.8-27B-Q5_K_M.gguf`（19.5 GB；64 层：48 层 gated-delta-net + 16 层全注意力，24 个 query 头共享 4 个 KV 头，head size 256；权重 q5_K 与 q6_K；MTP 头内置在同一文件中）。oMLX 对照 `Qwen3.8-27B-oQ5e-mtp`。

端到端（t/s）：

| t/s | 2k 预填充 | 8k 预填充 | 30k 预填充 | 2k 解码 | 8k 解码 | 30k 解码 |
|---|---|---|---|---|---|---|
| 本分支 | 1062* | 1086 | 951* | 26.0* | 24.6* | 22.5 |
| 上游 | 692* | 642 | 554* | 25.7* | 24.3* | 21.7 |
| oMLX（关 MTP） | 1018* | 1128 | 974* | 26.5* | 25.8* | 25.0 |
| 本分支，投机解码 | - | 1069* | 916* | - | 42.7* | 41.2 |
| 上游，投机解码 | - | 650* | 533* | - | 38.3* | 34.5 |
| oMLX，MTP | - | 1098 | 915* | - | 57.5* | 47.1 |

\* 本轮只复测 8k 预填充、30k 解码与 30k 投机解码，带 * 的单元格保留上一轮数字。llama-bench `pp8192 @8k`：本分支 1008、上游 569 t/s（1.77 倍）。

预填充与解码（llama-bench，t/s）：

| t/s | pp2048 @0 | pp2048 @8k | pp2048 @32k | pp512 @0 | pp512 @8k | pp512 @32k | tg64 @0 | tg64 @8k | tg64 @32k |
|---|---|---|---|---|---|---|---|---|---|
| 本分支 | 1134 | 1018 | 790 | 809 | 723 | 613 | 25.9 | 24.6 | 23.0 |
| 上游 | 650 | 585 | 451 | 719 | 608 | 469 | 25.9 | 24.3 | 22.0 |
| 本分支 / 上游 | 1.75x | 1.74x | 1.75x | 1.12x | 1.19x | 1.31x | 1.00x | 1.01x | 1.05x |

每步耗时（llama-bench，深度 0，ms/步；一步 2-64 个 token 对应投机解码验证、几路并发和短提示）：

| 每步 token | 1 | 2 | 4 | 8 | 16 | 24 | 32 | 48 | 64 | 96 |
|---|---|---|---|---|---|---|---|---|---|---|
| 本分支 | 38.9 | 42.5 | 51.2 | 51.0 | 56.0 | 74.4 | 82.1 | 132.2 | 145.2 | 220.9 |
| 上游 | 38.8 | 55.8 | 56.8 | 56.2 | 90.4 | 195.9 | 195.6 | 201.7 | 207.3 | 223.9 |
| 上游 / 本分支 | 1.00x | 1.31x | 1.11x | 1.10x | 1.61x | 2.63x | 2.38x | 1.53x | 1.43x | 1.01x |

并发（合计解码 t/s）：

| 并发路数 | 1 | 2 | 3 | 4 | 1，投机解码 | 4，投机解码 |
|---|---|---|---|---|---|---|
| 本分支 | 26.1 | 42.6 | 54.1 | 67.7 | 37.6 | 101.6 |
| 上游 | 26.0 | 34.9 | 48.8 | 61.1 | 35.0 | 74.6 |
| 本分支 / 上游 | 1.00x | 1.22x | 1.11x | 1.11x | 1.08x | 1.36x |

App 启动（App 的参数：262k 上下文、mmproj、MTP、`--fit-target 1024`）：本分支就绪 1.9 s、首个请求 TTFT 0.24 s，上游 2.7 s / 0.33 s；两个请求后的常驻内存（不含 mmap 的权重文件）本分支 3.3 GB，上游约 21 GB。两边的贪心输出逐字节相同。

### 3.2 gemma-4-26B-A4B Q4_0

`gemma-4-26B-A4B-it-Q4_0.gguf`（ggml-org，14.6 GB；30 层 MoE，128 个专家每 token 选 8 个，另有共享的稠密 FFN；滑动窗口层 head size 256，全局层 head size 512）。投机解码用同一仓库的 `dflash-gemma-4-26B-A4B-it-Q8_0.gguf`。oMLX 对照 `gemma-4-26B-A4B-it-oQ4e-mtp`。

| t/s | 2k 预填充 | 8k 预填充 | 30k 预填充 | 2k 解码 | 8k 解码 | 30k 解码 |
|---|---|---|---|---|---|---|
| 本分支 | 3869* | 4054 | 3354* | 121.4* | 114.9* | 106.2 |
| 上游 | 3521* | 3313 | 2703* | 120.0* | 108.1* | 89.7 |
| oMLX（关 MTP） | 3588* | 4456 | 3522* | 113.4* | 109.0* | 67.8 |
| 本分支，投机解码 | - | 3893* | 3151* | - | 117.4* | 117.2 |
| 上游，投机解码 | - | 3439* | 2564* | - | 109.3* | 83.9 |
| oMLX，MTP | - | 4412 | 3534* | - | 107.0* | 97.0 |

\* 本轮只复测 8k 预填充、30k 解码与 30k 投机解码，带 * 的单元格保留上一轮数字。llama-bench `pp8192 @8k`：本分支 3820、上游 3144 t/s（1.21 倍）。oMLX 的 `gemma-4-26B-A4B-it-oQ4e-mtp` 关闭 MTP 后 30k 解码为 67.8，开启 MTP 为 97.0。

| t/s | pp2048 @0 | pp2048 @8k | pp2048 @32k | pp512 @0 | pp512 @8k | pp512 @32k | tg64 @0 | tg64 @8k | tg64 @32k |
|---|---|---|---|---|---|---|---|---|---|
| 本分支 | 5218 | 4204 | 2613 | 4274 | 3359 | 2161 | 127.6 | 113.5 | 102.0 |
| 上游 | 4567 | 3548 | 1958 | 4164 | 3220 | 1913 | 126.9 | 108.5 | 86.3 |
| 本分支 / 上游 | 1.14x | 1.18x | 1.33x | 1.03x | 1.04x | 1.13x | 1.01x | 1.05x | 1.18x |

| 每步 token | 1 | 2 | 4 | 8 | 16 | 24 | 32 | 48 | 64 | 96 |
|---|---|---|---|---|---|---|---|---|---|---|
| 本分支 | 8.1 | 10.1 | 12.2 | 15.7 | 22.4 | 24.1 | 26.2 | 33.7 | 35.5 | 54.9 |
| 上游 | 8.2 | 10.1 | 12.2 | 15.5 | 24.9 | 45.6 | 41.9 | 46.0 | 48.2 | 53.9 |
| 上游 / 本分支 | 1.00x | 1.00x | 1.00x | 0.99x | 1.11x | 1.89x | 1.60x | 1.36x | 1.36x | 0.98x |

| 并发路数 | 1 | 2 | 3 | 4 | 1，投机解码 | 4，投机解码 |
|---|---|---|---|---|---|---|
| 本分支 | 122.9 | 190.7 | 261.9 | 321.2 | 98.9 | 202.6 |
| 上游 | 122.4 | 189.9 | 262.3 | 311.3 | 102.3 | 171.0 |
| 本分支 / 上游 | 1.00x | 1.00x | 1.00x | 1.03x | 0.97x | 1.18x |

App 启动：本分支就绪 2.2 s，上游 2.3 s。

### 3.3 小结

- 预填充：本轮 8k 预填充服务器端到端 Qwen 1.69 倍、gemma 1.22 倍；llama-bench `pp8192 @8k` 为 1.77 / 1.21 倍（上一轮 `pp2048` 为 1.5-2.0 倍）。
- 解码：短上下文与上游持平（0.99-1.01 倍），长上下文更快：30k 时 Qwen +4%、gemma +18%（H3 的 GQA 打包注意力）。测于上游 #29510 之前；通过 `kv_rows` 列表的 GQA 增幅见第 6.5 节。
- 投机解码与小批量：一步 16-64 个 token 是上游的 1.0-2.7 倍（H4 / M1）；30k 上 Qwen 的 MTP 41.2 对 34.5 t/s，gemma 的 DFlash 117.2 对 83.9 t/s。
- 并发（App 的 4 路布局）：2-4 路合计 1.0-1.22 倍，4 路投机解码 Qwen 1.36 倍、gemma 1.18 倍。gemma 单路投机解码 0.97 倍是这一个提示上的接受率差异（0.206 对关闭 A2 门槛时 0.227，提示的数值略有不同导致生成的文本不同），不是内核变慢：这一步里被门槛影响的只有 342 token 的提示预填充。
- 与 oMLX：8k 预填充 Qwen 慢 4%、gemma 慢 10%；30k 解码（oMLX 关 MTP）Qwen 慢 10%、gemma 快 57%；投机解码 MTP 对 MTP：Qwen 慢 13%（41.2 对 47.1）、gemma 快 21%（117.2 对 97.0）。
- 启动与内存（上一轮）：Qwen 就绪 1.9 对 2.7 s，常驻内存 3.3 对约 21 GB；关闭思考后不再重新处理整段对话。
- 精度：除 I8 外的改动与上游一致到参照量级；I8 在 gemma 上有可见代价，可单独关闭（第 9 节）。

## 4. 方法

- **先算差距账。** 动手之前，先用调试开关逐类跳过算子，测出每类算子的耗时并与硬件上限比较；只处理上限明显高于 1.5% 的部分。例如 gemma 的 MoE 预填充大部分时间在专家矩阵乘（MUL_MAT_ID）上，这一部分只跑到稠密路径一半的算力，M1 / M2 由此而来。
- **先筛选，再确认。** 受算力限制的候选先在合成权重上做内核级 A/B（同一进程内交错），增幅超过 1.5% 且多轮方向一致才进入真实模型测量。保留的方案全部在满功率（`powermode 0`）下确认，本文所有数字都是满功率数据。
- **交错测量。** 不同场次之间绝对数值会随发热状态漂移 5-15%。每次对比都在同一场次里按正序再反序运行各配置，只比较同一场次内的比值。测试之间间隔 2 分钟；漂移过大的测试段作废重测。
- **按留一法取舍。** 一项改动看的是它在最终组合中的贡献（全部开启减去这一项），而不只是单独开启时的效果。
- **按格式、按形状逐一验证。** 矩阵乘类改动对每一种权重格式分别在合成权重上做内核级 A/B（稠密：17408x5120 与 5120x17408；MoE：gemma-4 的专家形状），只对确实更快的格式和行数范围开启；同时比对两条路径的输出校验和。
- **按模型结构验证。** 注意力类改动在 Qwen 与 gemma 的结构上测：每个 KV 头对应 6 / 8 个 query 头，head size 128 / 256 / 512。某个结构上变慢的情形用门槛排除（例如 H3 对每 KV 头 4 个 query 头的单 token 解码）。
- **精度把关。** 每个模型都以上游的完整 logits 为基准测 KL 散度（逐 token 配对比较，比困惑度灵敏得多），并以上游自身 `-fa on` 与 `-fa off` 之间的差作为"正常数值差异"的参照；有损的改动要么改进到参照量级附近，要么写明代价并可单独关闭（见第 9 节）。
- **固定提示。** 投机解码的速度取决于生成文本接受了多少草稿；对单项改动的端到端对比使用固定提示，并确认输出文本相同后再比较耗时。
- **上游核查。** 每次都在上游 `master` 上重建并重新测量；本文基于 `67107ac34f`（b12164）。

## 5. 调研到的候选方案

调研来源：MLX（steel attention / gemm NAX 内核、`gated_delta` 内核、`gather_qmm`）、oMLX 0.7.1、uzu、BaseRT、MTPLX、dflash-mlx、mlx-dspark、vllm-mlx、mlx-vlm、FastVLM、llama.cpp 自己的 CUDA 后端（MMQ），以及 llama.cpp 的服务器和 Web UI 代码。

| 方面 | 候选方案 | 来源 | 结局 |
|---|---|---|---|
| 预填充注意力 | 寄存器常驻的 NAX flash attention | MLX `steel_attention_nax` | 保留（A1） |
| 预填充矩阵乘 | tensor mat-mat 使用 half 激活 | 自研 | 保留（A2） |
| 预填充矩阵乘 | 寄存器常驻的 NAX mat-mat，长 K 用 split-K | MLX `steel_gemm_fused_nax` / split-K | 保留（A3） |
| 预填充 GDN | 递推每状态行用更少的 lane | oMLX `gated_delta_pipelined` | 保留（A4） |
| 预填充矩阵乘 | INT8 激活 mat-mat | CUDA MMQ、uzu、oMLX oq_a8 | 保留（I8） |
| 预填充 | 激活直接写成下一个 mat-mat 的输入格式 | 自研 | 保留（A5） |
| 多轮缓存 | 关闭思考时不改变渲染出的提示 | 自研 | 保留（S1） |
| 启动 | 加载时不清零新分配的 KV / 递推状态 | 自研 | 保留（S2-a） |
| 启动 | 首次驻留请求放到后台 | 自研 | 保留（S2-c） |
| 并发 / 投机解码 | 2-3 行的跨行 mat-vec | 自研 | 保留（K3） |
| 短提示 | 小批次不做 f16 激活转换 | A2 的后续 | 保留（A2 门槛） |
| 投机解码 / 长上下文 | 共享 KV 头的 query 头打包进一个 NAX 注意力 tile | MLX 多行注意力的思路，自研 | 保留（H3） |
| 并发 / 投机解码 / 短提示 | 2-64 行在 NAX 上做矩阵乘，所有格式共用一个内核 | 自研，基于上游 few-row MMA | 保留（H4） |
| MoE 短批次 | MoE mat-mat 的使用门槛 | 自研 | 保留（M1） |
| MoE 预填充 | 每块更多条目，减少专家权重的反量化次数 | MLX `gather_qmm` 按专家分段的思路 | 保留（M2） |

不在本项目范围内：KV 量化（turboquant）、GPU 采样、MTL4 命令编码、媒体缓存。

## 6. 各项结果与取舍

取舍规则：单项增幅在多轮中方向一致且超过 1.5%；作为其他显著优化前提的改动除外。

### 6.1 预填充，稠密模型（Qwen3.8-27B Q5_K_M，llama-bench `-fa 1 -ub 2048`，8k 上下文下 pp2048）

"单独" = 只开这一项、其余全关（A3 依赖 A2；I8 依赖 A2 和 A3；A5 依赖 A3）。"留一法" = 全部开启、只关这一项。正反序两轮，按每轮内的比值取均值。

| # | 单独（相对全关） | 留一法 |
|---|---|---|
| A1 | +4.5% | +3.5% |
| A2 | +12.1% | （A3/I8 的前提） |
| A3 | 相对 A2 +10.9% | （I8 的前提） |
| A4 | +1.8% | +3.4% |
| I8 | 相对 A2+A3 +26.2% | +32.2% |
| A5 | 相对 A2+A3 +0.4% | +2.7% |
| **全部开启** | **上游的 1.73 倍** | |

A4 单独开启只有 +1.8%，在完整组合中为 +3.4%：其他部分变快后，GDN 在总时间中的占比变大。A5 同理。

内核级（同进程交错 A/B，2048 行）：
- A1：flash attention q2048 x kv8192 21.6 -> 11.8 ms，kv32768 100 -> 55 ms（对比上游 tensor-API 内核）。
- A3：f16 GEMM 17408x5120 7.1 -> 6.1 ms（59.7 TFLOP/s，与 MLX 稠密 f16 NAX gemm 持平）；量化格式 gate/up 形状 +8-17%，down 形状（K = 17408，split-K）+15-22%。
- A4：GDN 每 2048 token 192 -> 97 ms。
- I8：每个 mat-mat 上 int8 路径是 f16 NAX 路径的 1.3 倍（纯 int8 NAX 吞吐为 f16 的 2.1 倍）。

### 6.2 注意力结构与 MoE（A1 / H3 的扩展，M1，M2）

| 项 | 模型 | 效果（开 vs 关） |
|---|---|---|
| A1 扩展到 head size 512 | gemma-4-26B-A4B | pp2048 @0 / @8k / @32k：+0.4% / +3.4% / +12% |
| H3（解码，2k / 8k / 16k / 32k） | Qwen3.8-27B（每 KV 头 6 个 query 头） | +5% / +3% / +6% / +8% |
| | gemma-4-26B-A4B（8 个） | +1.5% / +6% / +9% / +18% |
| H3（4 token 的一步，即投机解码验证） | Qwen / gemma | +6% 至 +55%（2k-32k），32k 时 1.2-2.0 倍 |
| M1（一步 16 / 20 / 24 / 28 token） | gemma-4-26B-A4B | +6% / +14% / +23% / +30% |
| | gemma-4-26B-A4B | 0 / +4.5% / +5~9% / +3~4% |

- H3：测于上游 #29510 之前，走的是稠密缓存路径（第 6.5 节）。
- M1：上游在 32 个 token 以下对每个（token, 专家）条目各读一遍专家权重；改用 mat-mat 后每块最多 32 个条目共用一次读取。12 token 时两个模型方向相反（-3% / +7%），所以门槛定在 16。
- M2：上游的 MoE mat-mat 每块最多 32 个条目，每块都要把专家权重反量化一次。每专家平均 128 个条目及以上时，64 条目的块对全部 25 种格式、三种专家形状都更快（+5~35%，第 7.3 节）；条目更少时半空的块得不偿失，所以只在此时启用。gemma 的 pp1024（每专家 64 个条目）低于门槛，保持原样。

### 6.3 解码与服务路径（Qwen3.8-27B Q5_K_M）

| # | 效果（开 vs 关） | 测法 |
|---|---|---|
| S1 | 7k token 对话关闭思考后需要重新处理的 token：7330 -> 4（预填充 38.3 s -> 0.37 s） | Qwen3.8 的聊天模板，功能性改动 |
| S2-a | 就绪 2.96 -> 2.03 s（-31%）；常驻内存 21 GB -> 3.3 GB | App 参数，各 2 次 |
| S2-c | 就绪 2.24 -> 2.01 s（-10%） | 同上 |
| K3 | 每步耗时：2 行 57.7 -> 42.5 ms（+36%），3-32 行 +3-5% | llama-bench，正反序 |
| H4 | 每步耗时：12 / 16 / 24 / 32 / 48 / 64 行 90.9 / 93.5 / 200.3 / 199.3 / 201 / 207 -> 52.8 / 56.0 / 74.3 / 82.0 / 132 / 145 ms；2-8 行 +5-16% | 同上 |
| H3 | 开 MTP 解码：30k 27.9 -> 31.2 t/s（+11.9%），8k 26.3 -> 27.5（+4.8%）；输出文本逐字节相同 | 固定提示，正反序各 2 轮，#29510 之前 |
| A2 门槛 | 128 行以下保持 f32 激活，避免额外的转换派发；每步耗时 12-96 行快 2-6% | llama-bench |
| 并发（K3：2 路；H4：4 路开 MTP） | 2 路 +18%（K3），4 路开 MTP +42%（H4 与 H3） | 同一轮次内的比值 |

S2-c 依赖 S2-a：清零会触碰页面，缓冲区就不再是"新的"，所以只关 S2-a 时 S2-c 不起作用。K3 与 H4 按格式和行数分工（第 7 节）：q5_K 在 2 行走 K3，3 行起走 H4。

### 6.4 其余改动的数据

以下改动的效果数据来自各自的测量（提交说明），测于同一台机器。

- **A6 BF16 K/V 的 NAX 注意力**：预填充和 GQA 内核直接读 BF16 缓存、在寄存器里转成 f16（钳到 f16 范围）；与 F16 缓存 logits 的 KL 为 Qwen 0.00054、gemma 0.056，与同一 BF16 缓存上的 tensor 内核同量级。
- **A8 BF16 权重的 NAX mat-mat**：1024 行起把权重转一次 f16 后走 NAX，不使用 int8 变体。
- **A7 MLA 形状 192/128、576/512 的 NAX 注意力**：K、V 头维分别拆分；GQA 内核仍只做等头尺寸。
- **A9 NVFP4 / TQ1_0 权重**：这两种格式原先在 Metal 上回退到 CPU；现在覆盖 mat-vec、few-row MMA、mat-mat、MoE mat-mat 和 NAX 路径，test-backend-ops 新增 58 / 78 个用例。
- **H5 量化 K/V 缓存的 GQA 注意力**：q4_0 / q4_1 / q5_0 / q5_1 / q8_0 / iq4_nl 缓存按原格式读、在寄存器内反量化；q8_0 缓存 tg64 @32k：gemma 93.3 对 vec 内核 73.0（#29510 之前）。
- **H3b GQA 内核重写**：每个 threadgroup 最多 8 个 16 行 tile（head size 512 为 4），沿网格分块，KV 份数在运行期取 min(32, 1024/ne01)；16 token 一步 @32k：gemma 188 -> 427 t/s（#29510 之前）。
- **A3b int8 门槛降到 448 行**：448-896 行每个 mat-mul +12-47%。
- **A5b 融合 GLU 写回原位**：激活写进 mat-mat 自己的 f32 结果，消除与 gate/up 释放后的内存竞争。
- **A10 BF16 + head size 512 / 576 + 超过 32 个 query 的预填充**让回 tensor 内核：gemma pp2048 在 8k / 32k 上下文下上游快 6% / 16%。

### 6.5 上游 kv_rows 之后的 GQA 注意力

上游 #29510（`1b6935b227`，"flash_attn_ext_rows"，2026-11-03 合入）在开启 flash attention 时，为统一 KV 缓存的每个 ubatch 给 FLASH_ATTN_EXT 传入一份 K/V 行列表（`kv_rows`）。llama-server 默认使用统一 KV 缓存，所以这就是 Llama.app 与服务器的路径。GQA 内核的第一版按位置读缓存，所以在那里无法运行。上游自己没有 GQA 内核（PR #30645 与 #30728 仍未合入）。

所以内核学会了读列表：每个 lane 在每个块里查出自己那几列 K/V 对应的缓存行。它接手原本由 vec 内核处理的小批次；更大的批次仍走上游的"收集到 F16 + 稠密内核"（A1 在收集后的行上运行）。第 3 节的解码数字与第 6.2-6.4 节的 H3 / H5 / H3b 数字都测于 #29510 之前，走的是稠密缓存路径。

通过 `kv_rows` 的测量（Qwen3.8-27B Q5_K_M，`llama-batched-bench -fa on -kvu -ntg 64`，M5 Max，自动电源模式，GQA 开 vs `GGML_METAL_FA_GQA_DISABLE=1`，各 4 次，开关交替，取中位数，解码 t/s）：

| 深度 | 序列数 | GQA 开 | GQA 关 | 增幅 |
|---|---|---|---|---|
| 8192 | 1 | 21.9 | 20.4 | +7% |
| 8192 | 4 | 59.0 | 49.8 | +18% |
| 32768 | 1 | 21.2 | 16.8 | +27% |
| 32768 | 4 | 43.8 | 38.5 | +14% |

这组数据噪声大：后台有 Spotlight 索引和另一个应用在运行，绝对值波动很大。16 组开/关配对中有 14 组是开更快。通过 `kv_rows` 的投机解码验证（单序列、4 token 一步）没有测。`test-backend-ops -o FLASH_ATTN_EXT` 在 Metal 上 5100/5100 通过，其中新增两个 `kv_rows` 用例（head size 256、2 个 KV 头、每 KV 头 6 个 query 头，即 Qwen3.8-27B 的结构；F16 与 Q8_0），走的是 GQA 内核。

## 7. 量化格式

### 7.1 各改动覆盖的格式

- **与权重格式无关**：A1、H3（注意力，只看 F16 K/V；H5 增加 q4_0 / q4_1 / q5_0 / q5_1 / q8_0 / iq4_nl 缓存，通过 `kv_rows` 时不含 iq4_nl 与 BF16）、A4（GDN）、A2 与 A2 门槛（激活转换）、S1、S2-a、S2-c。
- **A3 / I8 / A5（>= 1024 行的预填充）**：F16 与 23 种量化格式：q4_0、q4_1、q5_0、q5_1、q8_0、q1_0、q2_0、mxfp4、q2_K、q3_K、q4_K、q5_K、q6_K、iq1_s、iq1_m、iq2_xxs、iq2_xs、iq2_s、iq3_xxs、iq3_s、iq4_nl、iq4_xs、tq2_0（I8 不含 F16）。权重先经各格式的反量化函数转成 f16，所以每种格式只需要一个反量化实例。
- **M1 / M2（MoE）**：上游 MoE mat-mat 支持的全部格式；M2 在 25 种格式上逐一测过（第 7.3 节）。
- **H4（2-64 行）**：few-row 内核支持的全部 26 种格式（上面 23 种加 f32、f16、bf16），按格式和行数开启：

| 行数 | 开启的格式 |
|---|---|
| 33-64 | 按格式的上限：64（q4_0、q4_1、q5_0、q5_1、q8_0、q2_K、q4_K、q5_K、q6_K、f16、bf16、tq2_0、q1_0、q2_0），56（mxfp4、iq4_nl、iq4_xs、iq2_xxs、iq2_s），48（q3_K、iq3_xxs、iq3_s、iq1_s），40（iq2_xs、iq1_m）；f32 不开 |
| 17-32 | 全部 26 种 |
| 9-16 | 全部量化格式（f32 / f16 / bf16 不更快，保持原路径） |
| 2-8 | iq 系列、q1_0、q2_0（2 行起）；q2_K、tq2_0、q5_K（3 行起）；q6_K（2-4 行） |

- **K3（2-3 行）**：q4_K（2-3 行）、q5_K（2 行）。

### 7.2 稠密矩阵乘的内核级数据（合成权重，M5 Max，满功率，相对原路径的速度比）

| 格式 | 2-8 行：H4 / 原路径 | 9-16 行：H4 / MMA | 17-32 行：H4 / mat-mat | 33-64 行：H4 / mat-mat | 2048 行：A3+I8 / mat-mat |
|---|---|---|---|---|---|
| f32 | 0.99-1.01x | 1.01-1.02x | 1.10-1.14x | - | - |
| f16 | 0.97-0.99x | 0.98x | 1.60-1.62x | 1.11-1.38x | 原有覆盖 |
| bf16 | 0.99-1.00x | 0.97-0.99x | 1.63-1.68x | 1.19-1.50x | - |
| q1_0 | 1.01-1.56x | 2.13-2.41x | 2.67-3.32x | 1.42-1.96x | 1.42x |
| q2_0 | 1.16-1.49x | 2.17-2.28x | 2.65-3.06x | 1.43-1.79x | 1.44x |
| q4_0 | 1.00-1.17x | 1.64-1.74x | 2.35-3.11x | 1.35-1.74x | 原有覆盖 |
| q4_1 | 0.99-1.22x | 1.58-1.62x | 2.27-2.89x | 1.27-1.77x | 原有覆盖 |
| q5_0 | 0.92-1.00x | 1.71-1.72x | 2.27-2.56x | 1.23-1.48x | 原有覆盖 |
| q5_1 | 0.94-1.02x | 1.62-1.66x | 2.17-2.44x | 1.17-1.42x | 原有覆盖 |
| q8_0 | 0.92-0.93x | 1.09-1.13x | 1.90-2.16x | 1.21-1.46x | 原有覆盖 |
| mxfp4 | 0.94-1.02x | 1.56-1.58x | 1.88-2.23x | 1.07-1.26x | 1.41x |
| q2_K | 0.93-1.40x | 1.77-2.04x | 2.38-2.91x | 1.35-1.63x | 1.46x |
| q3_K | 0.91-1.12x | 1.69-1.73x | 1.63-1.93x | 1.01-1.09x | 1.49x |
| q4_K | 0.80-1.04x | 1.48-1.49x | 2.13-2.52x | 1.23-1.45x | 原有覆盖 |
| q5_K | 0.88-1.22x | 1.62-1.74x | 1.92-2.37x | 1.14-1.37x | 原有覆盖 |
| q6_K | 0.99-1.68x | 1.55-1.56x | 1.73-2.08x | 1.11-1.20x | 原有覆盖 |
| iq1_s | 1.10-1.44x | 1.94x | 1.75-2.13x | 1.00-1.23x | 1.45x |
| iq1_m | 1.00-1.31x | 1.92-2.00x | 1.71-2.07x | 1.01-1.16x | 1.43x |
| iq2_xxs | 0.96-1.25x | 1.86x | 1.65-2.24x | 1.11-1.23x | 1.41x |
| iq2_xs | 1.02-1.42x | 1.93-2.06x | 1.65-1.98x | 0.99-1.14x | 1.41x |
| iq2_s | 1.08-1.32x | 1.71-1.94x | 1.97-2.29x | 1.04-1.27x | 1.42x |
| iq3_xxs | 1.05-1.25x | 1.83-1.87x | 1.75-2.21x | 1.00-1.19x | 1.45x |
| iq3_s | 1.06-1.23x | 1.87-1.91x | 1.63-2.10x | 1.01-1.18x | 1.43x |
| iq4_nl | 0.98-1.15x | 1.57-1.65x | 1.84-2.34x | 1.06-1.27x | 1.45x |
| iq4_xs | 0.99-1.17x | 1.68-1.71x | 1.83-2.23x | 1.07-1.16x | 1.44x |
| tq2_0 | 0.65-1.25x | 2.06-2.09x | 2.51-2.95x | 1.41-1.71x | 1.43x |

- 形状 17408x5120 与 5120x17408，两轮取均值；范围是各行数中的最小与最大。2-8 行一栏是对所有格式测得的结果，只有第 7.1 节列出的格式和行数实际开启；33-64 行一栏是开启范围内的结果。
- "原有覆盖"表示该格式在 A3/I8 最初就已支持，其效果见第 6.1 节。
- 2-3 行跨行 mat-vec（K3）：q5_K 在 2 行比原路径快 21%，3 行快 6%（3 行 H4 更快，所以 3 行用 H4）。

### 7.3 MoE 矩阵乘的内核级数据（M2：64 条目的块 vs 32 条目，合成权重，随机路由）

形状：gemma-4 的 gate_up（128 专家选 8，K = 2816，N = 1408）与 down（K = 704，N = 2816）。K 不是 256 倍数的形状无法存放 K-quant 与 IQ 格式，这些格式只测 gemma gate_up。T = 128 / 512 / 2048 个 token，相当于每专家 8 / 32 / 128 个条目。两轮中位数。

| 格式 | gemma gate_up（T = 128 / 512 / 2048） | gemma down |
|---|---|---|
| mxfp4 | +2%/+22%/+32% | -8%/-11%/+24% |
| q4_0 | -7%/+14%/+14% | -25%/-3%/+10% |
| q4_1 | -8%/-12%/+12% | -3%/+28%/+9% |
| q5_0 | -5%/+11%/+31% | -7%/+35%/+5% |
| q5_1 | -4%/+13%/+29% | -6%/-18%/+24% |
| q8_0 | +3%/-0%/+20% | -1%/+9%/+13% |
| f16 | +1%/+3%/+16% | -0%/+4%/+6% |
| bf16 | +1%/+3%/+9% | +2%/+4%/+8% |
| q2_K | -3%/+28%/+20% | - |
| q3_K | +202%/+13%/+33% | - |
| q4_K | -5%/-8%/+24% | - |
| q5_K | -3%/+18%/+32% | - |
| q6_K | +12%/+11%/+20% | - |
| iq2_xxs | -9%/+18%/+21% | - |
| iq2_xs | -4%/+15%/+19% | - |
| iq2_s | +10%/+12%/+19% | - |
| iq3_xxs | -26%/+23%/+21% | - |
| iq3_s | +12%/-16%/+24% | - |
| iq1_s | -5%/+19%/+19% | - |
| iq1_m | +15%/+27%/+21% | - |
| iq4_nl | -15%/+9%/+29% | -9%/-6%/+21% |
| iq4_xs | -13%/+14%/+28% | - |
| tq2_0 | +48%/+20%/+21% | - |
| q1_0 | -3%/+20%/+8% | - |
| q2_0 | +4%/-27%/+12% | -5%/-22%/+10% |

### 7.4 解码 mat-vec 的带宽（单 token）

单 token 解码的矩阵乘受带宽限制，原有的 mat-vec 内核在各格式上已接近可持续带宽：f16 约 545 GB/s，q4_K 528，q8_0 524，q5_K 501-531，q6_K 507-520（同一场次，合成权重）。q5_K / q6_K 的内核参数调优最多只有 +0.9%，所以单 token 路径没有改动。

## 8. 保留项的实现

**A1：NAX flash attention**（`kernels/fa_tensor.metal`，派发在 `ggml-metal-ops.cpp`）。F16 K/V 的预填充注意力把分数、概率和 O 都留在寄存器里（16x32x16 的 `matmul2d` 片段，BQ = 64，BK = 32），结构参照 MLX 的 `steel_attention_nax`；head dim 分给 WN 个 simdgroup（head size 128 为 1 个，256 为 2 个，512 为 4 个），各自的部分分数按固定顺序经 threadgroup 内存相加。每个块的掩码标志提前一块预取：否则跳过全掩码块的那个分支每次迭代都要等设备内存读取（单这一点端到端 +4-8%）。遇到 ALiBi 和 softcap 时回退到原有的 tensor 内核。

**A2：f16 激活**（`kernels/mul_mm.metal`、`ggml-metal-ops.cpp`、`ggml-metal.cpp`）。tensor `mul_mm` 内核用 half 激活比用 f32 快约 1.25 倍，所以先把 f32 激活转换（饱和到 +/-65504）到一个额外的缓冲区。只用于稠密 mat-mat（MoE 的版本单独评估）。

**A3：NAX mat-mat**（`kernels/mul_mm.metal`、`ggml-metal-impl.h`、`ggml-metal-ops.cpp`）。批次至少 1024 行时（`GGML_METAL_MM_NAX_MIN_ROWS`，测试用参数），每个 mat-mat 先把权重经该格式的反量化函数转成 f16 一次，然后在一个不用 threadgroup 内存的内核上做乘法：操作数直接载入片段，每个 64x128 tile 由 2x4 个 simdgroup 计算，threadgroup 做 swizzle，每 256 K 一次 barrier。禁止 K 循环展开是它变快的关键（0% -> 15%）。当 K >= 3*max(M, N) 时，K 被切成 4096 宽的分段依次派发、累加到 dst，使操作数切片留在缓存里（MLX 对长 K 也这样做）。

**A4：GDN 每行 4 个 lane**（`kernels/gated_delta_net.metal`、`ggml-metal-ops.cpp`）。原来每个状态行 32 个 lane，递推受指令数限制（每个 token 两次 5 步归约）。改为每行 4 个 lane（状态大小 256 时为 8 个，否则每个 lane 需要 64 个寄存器）后，每个 lane 保存 32 个状态值，归约只需 2 步（8 lane：1.87 倍，4 lane：1.98 倍）。其他形状、非标量 gate 和未对齐的输入仍走原有内核。

**I8：int8 mat-mat**（`kernels/mul_mm.metal`：`kernel_mul_mm_q8a_f32`、`kernel_mul_mm_dq_<type>_i8`、`kernel_mul_mm_nax_i8`）。激活和反量化后的权重都量化为对称 int8，每（行，128 列）一个 scale；量化前对每组 128 个值做一次不归一化的 Walsh-Hadamard 变换（`i8_wht128`，32 个 lane 各持 4 个值，组内 2 级加减 + 5 级 `simd_shuffle_xor`），两边用同一个变换，点积不变（1/128 折进激活的 scale），而离群值被摊到整组，scale 不再被少数大值撑大。`matmul2d<int8, int8, int32>` 累加 128 宽的一组 K，然后折回到 f32 累加器（`c += s_a*s_w*ci`）。tile 结构和 split-K 与 A3 相同；它复用 A3 预留的缓冲区，不需要额外内存。每 32 列一个 scale 精度也更好，但折回次数多 4 倍，Qwen pp2048 慢 25%，所以保持 128 列、用变换补精度：KL 散度 Qwen 0.0023 -> 0.0007、Llama 0.0009 -> 0.0004、gemma 0.112 -> 0.067，速度代价 0-2%。这与 CUDA MMQ 路径的 q8_1 激活以及 oMLX 的 oq_a8 属于同一类有损捷径（精度见第 9 节）。

**A5：SWIGLU -> down 投影**（`ggml-metal-fusion.cpp`、`kernels/unary.metal`）。当 SWIGLU 的输出送入一个 NAX mat-mat 时，GLU 内核直接写出该 mat-mat 所需的 f16 或 int8 激活（含 scale）；f32 的 GLU 输出和单独的转换那一遍都省掉了。结果与不融合的路径逐位相同。

**S1：关闭思考时保留对话缓存**（`tools/ui`）。界面的思考档位 低 / 中 / 高 原本就以 `thinking_budget_tokens` 发送，切换不会打断缓存；只有"关闭"发送 `chat_template_kwargs.enable_thinking=false`，Qwen3.8 的模板会因此去掉系统部分开头的一段说明，整段对话从第 5 个 token 左右起被重新处理。现在"关闭"改为发送 `thinking_budget_tokens: 0`，输出与模板的空思考块相同，提示前缀不变。新增设置"关闭思考时保留对话缓存"（设置 -> 开发者，默认开启）。

**S2-a：加载时不清零**（`ggml-metal-device.cpp`、`common/common.cpp`）。Metal 的共享缓冲区来自 `vm_allocate`，本来就是零，但 KV 缓存在分配后清零一次、预热后又清零一次，触碰并提交了每个页面（262k 上下文时 KV 约 16 GiB、递推状态 2.4 GiB）。Metal 端记录缓冲区自分配以来是否被触碰过，未触碰时跳过清零；`common` 端在预热后只重置 KV 单元，不擦除数据。

**K3：2-3 行的跨行 mat-vec**（`kernels/mul_mv.metal`）。与单行 mat-vec 相同的 lane 布局，但一个块的量化值和 scale 只解包一次，与 2-3 行激活分别做点积；解包后的值用 half 保存以限制寄存器用量。q4_K 用于 2-3 行，q5_K 用于 2 行。

**S2-c：首次驻留请求放到后台**（`ggml-metal-device.cpp`）。驻留集的第一次 `requestResidency` 会为 GPU 映射每个页面，262k 上下文的 KV 约需 0.4 s。对自分配以来未被触碰的共享缓冲区，这个请求放到后台队列执行，完成后驻留集再加入保活集合；释放缓冲区时会等待它。

**A2 门槛**（`ggml-metal-ops.cpp`）。A2 在 tensor mul_mm 之前把 f32 激活转为 f16，每次 mul_mm 多一次派发和前后两个 barrier，是固定成本；128 行以下这部分成本大于 half 激活省下的时间；128-512 行时，计算量小的矩阵乘（K/V 投影、递推状态的小投影）也是如此。现在 128 行且单次矩阵乘 4 GFLOP 起才转换（NAX mat-mat 只吃 f16，照旧转换），额外缓冲区也按同一条件预留。门槛扫过 2 / 4 / 8 / 16 GFLOP：4 是 Qwen 与 gemma 的 pp128 / 256 / 512 上都不变慢的取值。

**H3 / H3b：GQA 打包的 NAX 注意力**（`kernels/fa_tensor.metal`：`kernel_flash_attn_ext_nax_gqa`；`ggml-metal-ops.cpp`）。解码和投机解码验证时，vec 内核对每个 query 头读一遍 K/V。新内核把共享一个 KV 头的 G 个 query 头和 ne01 个 token 打包成一个 NAX tile 的行（p = g*ne01 + i），每个 K/V 块只读一次；每个 threadgroup 打包最多 8 个 16 行 tile（head size 512 为 4），沿网格按行块派发，KV 范围按 min(32, 1024/ne01) 切分交给这些 threadgroup；结果写成 vec 内核的临时缓冲区格式，由现有的 `kernel_flash_attn_ext_vec_reduce` 合并。条件：head size 128-512、F16 K/V、有掩码、没有 ALiBi / softcap、每个 KV 头至少 2 个 query 头；打包后只有不到 5 行（每 KV 头 4 个 query 头的单 token 解码）且 KV 短于 16k 时，tile 的大部分是空行，保持 vec 内核。有 `kv_rows` 时（第 6.5 节），每个 lane 在每个块里查出自己那几列 K/V 对应的缓存行；填充行（< 0）被钳位并屏蔽，所有 q 分片共用一条 K/V 读取流。更大的批次仍走上游的收集到 F16 + 稠密内核。

**H4：2-64 行的 NAX few-row mat-mat**（`kernels/mul_mm.metal`：`kernel_mul_mv_nax_f32`；`ggml-metal-common.cpp`、`ggml-metal-ops.cpp`）。src1 的行是 16x32x16 matmul2d 的 M = 16 行（一次一个或两个 16 行 tile，更多的行沿网格的第二维分成多遍），32 个权重行是它的 N；每个 lane 通过该格式的反量化函数把 16 个权重直接解到 half 片段里，不用 threadgroup 内存；4 个 simdgroup 切分 K，最后在 threadgroup 内存里求和。为了让每个 lane 一次解出一个完整的 16 权重组（反量化函数的单位），K 在每个 64 权重的块内做了排列，src0 与 src1 用同一个排列。q5_K 另有一个用 16 字节加载和位运算代替逐权重分支的反量化函数。各格式在哪些行数开启由一个函数集中决定（第 7.1 节）。支持与 MUL_MAT + ADD（残差加）融合。

**M1：MoE mat-mat 从 16 个 token 起**（`ggml-metal-common.cpp`）。`ggml_metal_op_mul_mat_id_use_mm` 在有 tensor 单元的设备上把门槛从 32 改为 16；其他设备保持 32。

**M2：MoE mat-mat 每块 64 个条目**（`kernels/mul_mm.metal`：`kernel_mul_mm_id_map0`、`kernel_mul_mm_id`；`ggml-metal-device.cpp`、`ggml-metal-ops.cpp`）。路由预处理把每个专家的条目切成块，主内核对每块把 64 行权重反量化一次并与块内的条目相乘。一个函数常量把块大小从 32 切换到 64（预处理与主内核同时切换）：64 条目的块按 16 行的四分之一块做 `matmul2d`，没有条目的四分之一块跳过；每个线程多装载一行激活；threadgroup 内存从 8 KB 增加到 16 KB。只在有 tensor 单元的设备上、专家平均至少 128 个条目时使用。

**A6 / A7 / A8 / A10：BF16 与 MLA**（`kernels/fa_tensor.metal`、`kernels/mul_mm.metal`、`ggml-metal-ops.cpp`）。BF16 缓存按原格式读入、在寄存器里钳到 f16 再做 MMA；MLA 的 192/128 与 576/512 把 K、V 头维分开拆（576/512 的 K 用 1 个 simdgroup、V 用 4 个）；BF16 权重 1024 行起转一次 f16 后走 NAX。BF16 + head size 512 / 576 + 超过 32 个 query（GQA 批次上限）的预填充仍让回 tensor 内核。

**A9 / A3b：更多权重格式**（`kernels/mul_mv.metal`、`kernels/dequantize.h`、`ggml-metal-ops.cpp`）。NVFP4 的 UE4M3 子块 scale 与 TQ1_0 的三进制位在核心里解码，两种格式接入全部 Metal 路径；int8 的 NAX 门槛从 1024 行降到 448 行。

**H3b / H5：GQA 内核**（`kernels/fa_tensor.metal`、`ggml-metal-ops.cpp`）。每个 threadgroup 打包最多 8 个 16 行 tile（head size 512 为 4），沿网格分块；量化缓存按原格式读、每次载入解 4 个值；KV 份数取 min(32, 1024/ne01)。H3、H5 与 H3b 现在是一个提交。有 `kv_rows` 时，q4_0 / q4_1 / q5_0 / q5_1 / q8_0 缓存也走它；上游在有 `kv_rows` 时不接受 iq4_nl 与 BF16。

**A5b：融合 GLU 写回原位**（`ggml-metal-fusion.cpp`）。激活写进 mat-mat 自己的 f32 结果，消除与 gate/up 释放后的内存竞争。每项改动的环境变量见第 2 节的开关一列。

## 9. 数值精度

- `test-backend-ops -b MTL0`：MUL_MAT 3892/3892（few-row 用例覆盖全部格式的 2-64 行；在 `GGML_METAL_MM_NAX_MIN_ROWS=1` 下也通过，该设置让所有符合条件的用例都走 NAX / int8 内核），MUL_MAT_ADD 384/384，MUL_MAT_ID 867/867（M2 的 32 / 64 条目两种块都通过），FLASH_ATTN_EXT 5100/5100（含 head size 128 / 256 / 512 与 MLA 的 NAX 路径，以及两个走 GQA 内核的 `kv_rows` 用例），GATED_DELTA_NET 30/30。每个提交单独编译并通过与它相关的测试。
- KL 散度（主要指标）：以上游的完整 logits 为基准（`llama-perplexity --kl-divergence-base`，8 x 2048 token，`-c 2048 -b 2048 -ub 2048 -fa on`，文本为 llama.cpp 仓库文档），逐 token 比较本分支的输出分布。每格为平均 KL 散度 / top-1 一致率。参照列是上游自己换一种等价设置（`-fa off`，或 `-ub 512`）时与它本身的差，代表"正常的数值差异"：

| 模型 | 本分支，全部开启 | 本分支，关 I8 | 参照：上游 -fa off | 参照：上游 -ub 512 |
|---|---|---|---|---|
| Qwen3.8-27B Q5_K_M | 0.0007 / 99.00% | <0.0001 / 99.96% | <0.0001 / 99.96% | <0.0001 / 100.00% |
| gemma-4-26B-A4B Q4_0 | 0.0670 / 90.20% | 0.0236 / 94.89% | 0.0244 / 94.92% | 0.0042 / 98.90% |

  - 除 I8 以外的改动在 Qwen 上与上游几乎逐位一致（KL 散度 < 0.0001）；在 gemma 上与参照同一量级。gemma 对任何数值差异都更敏感：上游自己 `-fa on` 与 `-fa off` 之间就差 0.024。逐项开关定位（全部关闭再单独打开一项）显示这部分来自 A1（NAX 注意力）与 A2/A3（f16 路径），各约 0.02；其余改动（H3、H4、K3、M1、M2、A5）与上游逐位相同。结果是确定的：重复运行、关闭并发都得到完全相同的数值。
  - I8 是唯一有实质代价的改动：int8 激活在 Qwen 上增加约 0.0007，在 gemma 上约 +0.043（top-1 一致率 94.9% -> 90.2%），是参照的约 3 倍。量化前的 Walsh-Hadamard 变换已把这部分误差降到原来的 1/3（Qwen）到 3/5（gemma），见第 8 节。I8 默认开启，因为它是稠密模型预填充最大的单项收益（Qwen 留一法 +32%，gemma +7%）；在意精度的场景可以用 `GGML_METAL_MM_I8_DISABLE=1` 单独关闭，其余改动不受影响。
- 困惑度（同样的文本与设置），全部改动开启 vs 上游，作为补充：

| 模型 | 本分支 | 上游 | 差 | 统计误差 |
|---|---|---|---|---|
| Qwen3.8-27B Q5_K_M | 3.4648 | 3.4652 | -0.01% | +/-0.08 |
| gemma-4-26B-A4B Q4_0 | 65.5324 | 64.6739 | +1.33% | +/-3.42 |

  困惑度的统计误差远大于上面的 KL 散度差，只能排除大的错误。注意 gemma 的 `--kl-divergence-base` 那一轮报告的困惑度（62.71）与同一个上游二进制普通运行的结果（64.67）不同，而两轮 logits 完全相同（KL 散度为 0），所以 gemma 的困惑度比值不可信，以 KL 散度为准。
- 贪心输出：启动测试中本分支与上游的输出逐字节相同；H3 开与关的输出逐字节相同；A5 与不融合的路径 KL 散度相同（Qwen 0.00070 vs 0.00073，差在误差内；融合后编译器对乘加的合并不同，不再逐位相同）；M1、M2 与上游的 MoE 路径逐位相同。

## 10. 剩余差距

1. **MoE 预填充。** gemma 的 8k 提示比 oMLX 慢 9%（30k 预填充本轮未复测）。M2 之前，gemma 的预填充大部分时间在 MUL_MAT_ID 上，只跑到稠密路径约一半的算力；M2 之后快了 24-37%，但按专家分块的 mat-mat 仍明显低于稠密路径。下一步是把 A3 那样寄存器常驻的 NAX 内核用到按专家分块的 mat-mat 上（按专家分块的 few-row 方案在大批量下更慢）。
2. **Qwen 的 MTP。** 8k 时 oMLX 的 MTP 解码 57.5 t/s，本分支 42.7（上一轮，本轮未复测 8k 投机）；30k 时 47.1 对 41.2。不用投机解码时两者差 10%（25.0 对 22.5），所以差距主要在投机解码这一层（草稿长度、接受率、草稿头本身的开销），本项目只改了 Metal 后端，没有拆分这部分。
3. **解码带宽。** 短上下文的解码已和上游持平，但离带宽上限还有距离：Qwen 2k 解码 26.0 t/s x 19.5 GB ≈ 507 GB/s，即 614 GB/s 标称带宽的 83%。oMLX 在 Qwen 上快 10%。
4. **I8 的精度。** 在 gemma 上，I8 使 KL 散度约为上游 `-fa on`/`-fa off` 参照的 3 倍（第 9 节）。更细的分组太慢（每 32 列：Qwen -25%），Walsh-Hadamard 变换已收回一部分；彻底解决需要能逐矩阵判断激活分布的办法，目前只能整体关闭。
5. **通过 `kv_rows` 的投机解码验证。** 单序列、4 token 一步、经 `kv_rows` 列表的情形还没有测（第 6.5 节）。

## 11. 复现

```sh
# 构建（macOS，Metal）
cmake -B build -DCMAKE_BUILD_TYPE=Release && cmake --build build -j --target llama-bench llama-server llama-perplexity test-backend-ops

# 数值校验
build/bin/test-backend-ops -o MUL_MAT -b MTL0
GGML_METAL_MM_NAX_MIN_ROWS=1 build/bin/test-backend-ops -o MUL_MAT -b MTL0
build/bin/test-backend-ops -o MUL_MAT_ADD -b MTL0
build/bin/test-backend-ops -o MUL_MAT_ID -b MTL0
build/bin/test-backend-ops -o FLASH_ATTN_EXT -b MTL0
build/bin/test-backend-ops -o GATED_DELTA_NET -b MTL0

# 预填充 A/B：llama-bench 默认 -ub 512，NAX mat-mat 需要至少 1024 行
build/bin/llama-bench -m model.gguf -fa 1 -ub 2048 -p 2048 -n 0 -d 8192
GGML_METAL_MM_I8_DISABLE=1 build/bin/llama-bench -m model.gguf -fa 1 -ub 2048 -p 2048 -n 0 -d 8192
GGML_METAL_MM_ID_WIDE_DISABLE=1 build/bin/llama-bench -m moe-model.gguf -fa 1 -ub 2048 -p 2048 -n 0

# 每步耗时（2-64 token：K3、H4、M1）与长上下文解码（H3）
build/bin/llama-bench -m model.gguf -fa 1 -ub 2048 -p 1,2,4,16,24,32,48,64 -n 0
GGML_METAL_MV_NAX_DISABLE=1 GGML_METAL_MV_XR_DISABLE=1 GGML_METAL_MM_ID_MIN16_DISABLE=1 build/bin/llama-bench -m model.gguf -fa 1 -ub 2048 -p 1,2,4,16,24,32,48,64 -n 0
build/bin/llama-bench -m model.gguf -fa 1 -p 4 -n 64 -d 32768
GGML_METAL_FA_GQA_DISABLE=1 build/bin/llama-bench -m model.gguf -fa 1 -p 4 -n 64 -d 32768
build/bin/llama-batched-bench -m model.gguf -fa on -kvu -npp 8192,32768 -ntg 64 -npl 1,4
GGML_METAL_FA_GQA_DISABLE=1 build/bin/llama-batched-bench -m model.gguf -fa on -kvu -npp 8192,32768 -ntg 64 -npl 1,4

# 投机解码的服务器（模型文件内置 MTP 头时不需要单独的草稿模型）
build/bin/llama-server -m model.gguf -fa on -np 4 -kvu --spec-type draft-mtp --spec-draft-n-max 3
build/bin/llama-server -m gemma-4-26B-A4B-it-Q4_0.gguf -fa on -np 4 -kvu --spec-type draft-dflash -md dflash-gemma-4-26B-A4B-it-Q8_0.gguf
```

## 关于代码

本文档描述每项改动，但不逐行解释源码；代码在 `LXD-8/llama.cpp` 的 `metal-m5` 分支，基于上游 `master` 的 `67107ac34f`（b12164），每项改动一个提交（H3、H5 与 H3b 合为一个），各自可以单独关闭。如果你考虑采用其中的某项优化，我很乐意提交 PR 或以你方便的其他形式提供代码。英语不是我的母语，像本文这样的长文，我以中文为原文，再由 AI 翻译成英文。
