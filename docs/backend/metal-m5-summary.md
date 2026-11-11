# llama.cpp Metal optimizations for Apple M5 (NAX): short version

[English](#llamacpp-metal-optimizations-for-apple-m5-nax-short-version) | [中文](#apple-m5nax上的-llamacpp-metal-优化简略版)

This project optimizes the Metal backend of llama.cpp on Apple M5, with the goal that every kind of model (dense, MoE, different attention structures) and weight format runs close to the hardware limit, covering prefill, decode, speculative decoding, concurrency, short prompts, start-up and the multi-turn conversation cache. 22 code commits on upstream `master` at `b8abe41a50` (after b12283), one commit per change (H3, H5 and H3b are one commit), each can be switched off on its own. Test machine: MacBook Pro (Mac17,6), Apple M5 Max, 40-core GPU, 128 GB, macOS 27.2. See the [full report](metal-m5.md).

All changes on, compared with upstream and oMLX (full power, server end to end; oMLX with INT8 prefill on, KV quantization off on both sides; oMLX MTP is off for the decode column and on for the speculative column):

| Model | 8k prefill t/s: ours / upstream / oMLX | vs upstream | 30k decode t/s: ours / up / oMLX | vs upstream | 30k speculative decode t/s: ours / up / oMLX |
|---|---|---|---|---|---|
| Qwen3.8-27B Q5_K_M | 1086 / 642 / 1128 | 1.69x | 22.5 / 21.7 / 25.0 | 1.04x | 41.2 / 34.5 / 47.1 |
| gemma-4-26B-A4B Q4_0 | 4054 / 3313 / 4456 | 1.22x | 106.2 / 89.7 / 67.8 | 1.18x | 117.2 / 83.9 / 97.0 |

- Prefill: at 8k end to end, Qwen 1.69x, gemma 1.22x; llama-bench `pp8192 @8k` 1.77 / 1.21x.
- Decode: on par with upstream at short context (0.99-1.01x), faster at long context: at 30k Qwen +4%, gemma +18% (H3, the GQA-packed attention). Measured before upstream #29510; through the `kv_rows` lists of the server path, Qwen decode with GQA on is +7-27% (noisy, [section 6.5](metal-m5.md#65-gqa-attention-after-upstream-kv_rows)).
- Speculative decoding and small batches: steps of 16-64 tokens are 1.0-2.7x upstream (H4 / M1); at 30k, Qwen MTP 41.2 vs 34.5 t/s, gemma DFlash 117.2 vs 83.9 t/s.
- Concurrency (the app's 4-slot layout): 2-4 streams 1.0-1.22x in total, 4 speculative streams Qwen 1.36x, gemma 1.18x. The 0.97x for a single speculative gemma stream is a difference in acceptance on this one prompt (0.206 vs 0.227 with the A2 gate off, slightly different prompt numerics lead to different generated text), not a slower kernel: the only work the gate touches there is the 342-token prompt prefill.
- Against oMLX: 8k prefill Qwen 4% slower, gemma 10% slower; 30k decode (oMLX with MTP off) Qwen 10% slower, gemma 57% faster; speculative MTP vs MTP: Qwen 13% slower, gemma 21% faster.
- Start-up and memory (previous round): Qwen ready in 1.9 vs 2.7 s, resident memory 3.3 vs about 21 GB; turning thinking off no longer reprocesses the whole conversation.
- Three more models (llama-bench against upstream, [section 3.3](metal-m5.md#33-qwen36-35b-a3b-qwen38-flash-next-and-gemma-4-31b)): pp2048 Qwen3.6-35B-A3B Q4_K_M 1.20 / 1.41x (depth 0 / 8k), Qwen3.8-Flash-Next UD-IQ4_XS 1.24 / 1.25x, gemma-4-31B Q4_0 1.56 / 1.49x; steps of 16-24 tokens 1.0-2.7x; decode on par.
- Accuracy: the changes other than I8 match upstream to the order of the reference; I8 has a visible cost on gemma and can be switched off alone (section 9).

## Changes

| # | Change | Main effect |
|---|---|---|
| A1 | Register-resident NAX flash attention (prefill, head size 128-512) | Qwen prefill leave-one-out +3.5%; gemma pp2048 @32k +12% |
| A2 | f16 activations for the tensor mat-mat | alone +12.1% (prerequisite of A3/I8) |
| A3 | Dequant to f16 + register-resident NAX mat-mat, split-K for long K | +10.9% on top of A2 (prerequisite of I8) |
| A4 | GDN with 4 lanes per state row | leave-one-out +3.4% |
| I8 | int8 NAX mat-mat (one scale per 128 columns, after a Walsh-Hadamard transform) | leave-one-out +32.2% |
| A5 | SWIGLU fused into the next mat-mat | leave-one-out +2.7% |
| S1 | Keep the conversation cache when thinking is turned off (Web UI) | reprocessed tokens 7330 -> 4 |
| S2-a | No zero-fill of fresh KV / recurrent-state buffers at load | ready -31%, resident memory 21 -> 3.3 GB |
| K3 | Cross-row mat-vec for 2-3 rows (q4_K, q5_K) | step of 2 tokens +36% |
| S2-c | First residency request in the background | ready -10% |
| A2 gate | f16 conversion only from 128 rows and 4 GFLOP | time per step +2-6% at 12-96 rows, pp128 +2-7% |
| H4 | NAX few-row mat-mat for 2-64 rows, one kernel for every format | step of 16 / 32 / 64 tokens 1.6 / 2.4 / 1.4x (Qwen) |
| M1 | MoE mat-mat from 16 tokens | steps of 16-28 tokens: gemma +6-30%; 24 tokens: Qwen3.6-35B-A3B +10-14%, Qwen3.8-Flash-Next +26-28% |
| M2 | MoE mat-mat tiles of 64 entries, from 32 entries per expert | gemma prefill +4.5-9%; the threshold of 32 (instead of 128): Qwen3.6-35B-A3B +5%, Qwen3.8-Flash-Next +3% |
| A6 | NAX attention for a BF16 K/V cache | KL on the order of the tensor kernel; BF16 caches no longer fall back |
| A8 | NAX mat-mat for BF16 weights | from 1024 rows, BF16 takes NAX |
| A7 | NAX attention for the MLA shapes 192/128 and 576/512 | covers DeepSeek / GLM / Kimi style models |
| A9 | NVFP4 / TQ1_0 weights | both formats no longer fall back to the CPU |
| A3b | int8 NAX threshold lowered to 448 rows | 448-896 rows, +12-47% per mat-mul |
| A5b | fused GLU writes in place | fixes the race with the released gate/up buffers |
| A10 | BF16 large-batch prefill yields to the tensor kernel | gemma pp2048 8k/32k: upstream 6%/16% faster |
| H3 | GQA-packed NAX attention (decode, speculative verify) | decode @32k +8-18%, step of 4 tokens @32k 1.2-2.0x |
| H5 | GQA attention for quantized K/V caches | q8_0 cache at tg64 @32k: gemma 93.3 vs 73.0 |
| H3b | GQA kernel: 8 tiles / row blocks / runtime split count | 16-token step at 32k: gemma 188 -> 427 t/s |

H3, H5 and H3b are one commit, and the kernel also reads K/V through the `kv_rows` lists that upstream #29510 uses for the unified KV cache (the default of llama-server). Their numbers above were measured before #29510. Through `kv_rows` (Qwen, `llama-batched-bench -kvu`, noisy): depth 8k +7% (1 sequence) / +18% (4), depth 32k +27% / +14%. See [section 6.5](metal-m5.md#65-gqa-attention-after-upstream-kv_rows).

Of the prefill numbers, A2 and A3 are alone (they are prerequisites of later changes), the others leave-one-out, so they do not add up. See [section 6](metal-m5.md#6-per-change-results-and-selection).

## Weight formats

- Prefill (A3 / I8 / A5): F16, BF16 and 25 quantized formats; the newly covered formats are 41-49% faster per mat-mul on synthetic weights.
- 2-64 rows (H4): 28 formats enabled per format and row count; 1.1-3.3x at 17-32 rows, 1.0-2.0x at 33-64 rows.
- MoE (M1 / M2): every format; M2 measured on 25 formats one by one, 1-26% faster from 32 entries per expert.
- Real models: q5_K / q6_K (Qwen3.8-27B), q4_K (Qwen3.6-35B-A3B), iq3_xxs / iq4_xs / q5_0 / iq4_nl (Qwen3.8-Flash-Next), q4_0 (gemma). See [section 7](metal-m5.md#7-weight-formats).

## Method and accuracy

Account for the gap first, then screen candidates; keep only changes with a gain above 1.5% in a consistent direction over several rounds (except prerequisites of other optimizations), implemented with the minimum necessary code. Interleaved forward/reverse measurements within a session, 2 minutes between tests; mat-mul changes A/B tested at kernel level per format and per shape, attention changes verified on the Qwen and gemma structures. `test-backend-ops` passes (MUL_MAT 4099, MUL_MAT_ADD 384, MUL_MAT_ID 1109, FLASH_ATTN_EXT 5100, GATED_DELTA_NET 37). Accuracy is measured as the KL divergence to the full upstream logits: the changes other than I8 are practically bit-identical to upstream on Qwen and of the same order as upstream's own `-fa on`/`-fa off` difference on gemma; I8 (int8 prefill) is the only change with a real cost, about 0.0007 on Qwen and about 3x the reference on gemma, and `GGML_METAL_MM_I8_DISABLE=1` turns it off alone ([section 9](metal-m5.md#9-accuracy)).

## Remaining gaps

MoE prefill is still 3-9% behind oMLX (the per-expert mat-mat is still clearly slower than the dense path); Qwen MTP decode at 8k trails oMLX (42.7 vs 57.5 t/s), mostly in the speculative layer; short-context decode is at about 76-83% of the bandwidth limit; I8 has a visible accuracy cost on gemma; speculative verify (4-token steps) through `kv_rows` is not measured yet. See [section 10](metal-m5.md#10-remaining-gaps).

## Code and use of AI

The code is on the `metal-m5` branch of `LXD-8/llama.cpp`, based on upstream `master` at `b8abe41a50` (after b12283), one commit per change (H3, H5 and H3b are one commit). The b8abe41a50 rebase was checked commit by commit: no upstream change duplicates or replaces these optimizations, and no optimization was removed; the one Metal change (MUL_MAT src1 precision gating) is honored by keeping f32 activations when a graph pins their precision. The goals, test methodology and decisions are my own; AI tools were used for research, writing code, testing and writing the documents, and their output was reviewed by me. The Chinese version is the original; the English version was translated by AI.

---

# Apple M5（NAX）上的 llama.cpp Metal 优化：简略版

[English](#llamacpp-metal-optimizations-for-apple-m5-nax-short-version) | [中文](#apple-m5nax上的-llamacpp-metal-优化简略版)

本项目优化 llama.cpp 的 Metal 后端在 Apple M5 上的性能，目标是让各种模型（稠密、MoE、不同注意力结构）和量化格式都基本达到硬件上限，范围包括预填充、解码、投机解码、并发、短提示、启动和多轮对话缓存。22 个代码提交，每项改动一个提交（H3、H5 与 H3b 合为一个）、可单独关闭，基于上游 `master` 的 `b8abe41a50`（已过 b12283）。测试设备：MacBook Pro（Mac17,6），Apple M5 Max，40 核 GPU，128 GB，macOS 27.2。详见[完整报告](metal-m5.md)。

全部改动开启，与上游、oMLX 对比（满功率，服务器端到端；oMLX 开启 INT8 预填充，两边都关 KV 量化；oMLX 的 MTP 在解码列关闭、投机解码列开启）：

| 模型 | 8k 预填充 t/s：本分支 / 上游 / oMLX | 对上游 | 30k 解码 t/s：本 / 上 / oMLX | 对上游 | 30k 投机解码 t/s：本 / 上 / oMLX |
|---|---|---|---|---|---|
| Qwen3.8-27B Q5_K_M | 1086 / 642 / 1128 | 1.69x | 22.5 / 21.7 / 25.0 | 1.04x | 41.2 / 34.5 / 47.1 |
| gemma-4-26B-A4B Q4_0 | 4054 / 3313 / 4456 | 1.22x | 106.2 / 89.7 / 67.8 | 1.18x | 117.2 / 83.9 / 97.0 |

- 预填充：8k 服务器端到端 Qwen 1.69 倍、gemma 1.22 倍；llama-bench `pp8192 @8k` 1.77 / 1.21 倍。
- 解码：短上下文与上游持平（0.99-1.01 倍），长上下文更快：30k 时 Qwen +4%、gemma +18%（H3 的 GQA 打包注意力）。测于上游 #29510 之前；在服务器路径的 `kv_rows` 列表上，Qwen 解码开 GQA 快 7-27%（噪声大，[第 6.5 节](metal-m5.md#65-上游-kv_rows-之后的-gqa-注意力)）。
- 投机解码与小批量：一步 16-64 个 token 是上游的 1.0-2.7 倍（H4 / M1）；30k 上 Qwen 的 MTP 41.2 对 34.5 t/s，gemma 的 DFlash 117.2 对 83.9 t/s。
- 并发（App 的 4 路布局）：2-4 路合计 1.0-1.22 倍，4 路投机解码 Qwen 1.36 倍、gemma 1.18 倍。gemma 单路投机解码 0.97 倍是这一个提示上的接受率差异（0.206 对关闭 A2 门槛时 0.227，提示的数值略有不同导致生成的文本不同），不是内核变慢：这一步里被门槛影响的只有 342 token 的提示预填充。
- 与 oMLX：8k 预填充 Qwen 慢 4%、gemma 慢 10%；30k 解码（oMLX 关 MTP）Qwen 慢 10%、gemma 快 57%；投机解码 MTP 对 MTP：Qwen 慢 13%、gemma 快 21%。
- 启动与内存（上一轮）：Qwen 就绪 1.9 对 2.7 s，常驻内存 3.3 对约 21 GB；关闭思考后不再重新处理整段对话。
- 另外三个模型（llama-bench 对上游，[第 3.3 节](metal-m5.md#33-qwen36-35b-a3bqwen38-flash-next-与-gemma-4-31b)）：pp2048 Qwen3.6-35B-A3B Q4_K_M 1.20 / 1.41 倍（深度 0 / 8k）、Qwen3.8-Flash-Next UD-IQ4_XS 1.24 / 1.25 倍、gemma-4-31B Q4_0 1.56 / 1.49 倍；一步 16-24 token 1.0-2.7 倍；解码持平。
- 精度：除 I8 外的改动与上游一致到参照量级；I8 在 gemma 上有可见代价，可单独关闭（第 9 节）。

## 改动

| # | 改动 | 主要效果 |
|---|---|---|
| A1 | 寄存器常驻的 NAX flash attention（预填充，head size 128-512） | Qwen 预填充留一法 +3.5%；gemma pp2048 @32k +12% |
| A2 | tensor mat-mat 使用 f16 激活 | 单独 +12.1%（A3/I8 的前提） |
| A3 | 反量化到 f16 + 寄存器常驻的 NAX mat-mat，长 K 用 split-K | 相对 A2 +10.9%（I8 的前提） |
| A4 | GDN 每状态行 4 个 lane | 留一法 +3.4% |
| I8 | int8 NAX mat-mat（每 128 列一个 scale，先做 Walsh-Hadamard 变换） | 留一法 +32.2% |
| A5 | SWIGLU 融合进下一个 mat-mat | 留一法 +2.7% |
| S1 | 关闭思考时保留对话缓存（Web UI） | 重新处理 7330 -> 4 token |
| S2-a | 加载时不清零新的 KV / 递推状态缓冲区 | 就绪 -31%，常驻内存 21 -> 3.3 GB |
| K3 | 2-3 行的跨行 mat-vec（q4_K、q5_K） | 一步 2 token +36% |
| S2-c | 首次驻留请求放到后台 | 就绪 -10% |
| A2 门槛 | 128 行且 4 GFLOP 起才做 f16 转换 | 每步耗时 12-96 行 +2-6%，pp128 +2~7% |
| H4 | 2-64 行的 NAX few-row mat-mat，所有格式共用一个内核 | 一步 16 / 32 / 64 token 1.6 / 2.4 / 1.4 倍（Qwen） |
| M1 | MoE mat-mat 从 16 个 token 起使用 | 一步 16-28 token：gemma +6~30%；24 token：Qwen3.6-35B-A3B +10~14%、Qwen3.8-Flash-Next +26~28% |
| M2 | MoE mat-mat 每块 64 个条目，每专家 32 个条目起 | gemma 预填充 +4.5~9%；门槛 32（原 128）：Qwen3.6-35B-A3B +5%、Qwen3.8-Flash-Next +3% |
| A6 | BF16 K/V 缓存的 NAX 注意力 | KL 与 tensor 内核同量级，BF16 缓存不再回退 |
| A8 | BF16 权重的 NAX mat-mat | 1024 行起，BF16 走 NAX |
| A7 | MLA 形状 192/128、576/512 的 NAX 注意力 | 覆盖 DeepSeek / GLM / Kimi 类结构 |
| A9 | NVFP4 / TQ1_0 权重 | 两种格式不再回退到 CPU |
| A3b | int8 的 NAX 门槛降到 448 行 | 448-896 行每 mat-mul +12~47% |
| A5b | 融合 GLU 写回原位 | 修 gate/up 释放后的内存竞争 |
| A10 | BF16 大 batch 预填充让回 tensor 内核 | gemma pp2048 8k/32k 上游快 6%/16% |
| H3 | GQA 打包的 NAX 注意力（解码、投机解码验证） | 解码 @32k +8~18%，4 token 一步 @32k 1.2-2.0 倍 |
| H5 | 量化 K/V 缓存的 GQA 注意力 | q8_0 缓存 tg64 @32k：gemma 93.3 对 73.0 |
| H3b | GQA 内核 8 tile / 行块 / 运行期份数 | 16 token 一步 @32k：gemma 188 -> 427 t/s |

H3、H5 与 H3b 是一个提交，内核也能通过上游 #29510 为统一 KV 缓存（llama-server 的默认设置）使用的 `kv_rows` 列表读 K/V。上表中它们的数字测于 #29510 之前。通过 `kv_rows`（Qwen，`llama-batched-bench -kvu`，噪声大）：深度 8k 时 +7%（1 个序列）/ +18%（4 个），深度 32k 时 +27% / +14%。详见[第 6.5 节](metal-m5.md#65-上游-kv_rows-之后的-gqa-注意力)。

预填充的单项数字中，A2、A3 是单独开启（它们是后面改动的前提），其余是留一法，所以不能相加。详见[第 6 节](metal-m5.md#6-各项结果与取舍)。

## 量化格式

- 预填充（A3 / I8 / A5）：F16、BF16 与 25 种量化格式；新增的格式在合成权重上每个 mat-mul 快 41-49%。
- 2-64 行（H4）：28 种格式按格式和行数开启；17-32 行快 1.1-3.3 倍，33-64 行快 1.0-2.0 倍。
- MoE（M1 / M2）：全部格式；M2 在 25 种格式上逐一测过，每专家 32 个条目起快 1-26%。
- 真实模型：q5_K / q6_K（Qwen3.8-27B）、q4_K（Qwen3.6-35B-A3B）、iq3_xxs / iq4_xs / q5_0 / iq4_nl（Qwen3.8-Flash-Next）、q4_0（gemma）。详见[第 7 节](metal-m5.md#7-量化格式)。

## 方法与精度

先算差距账，再筛选候选；只保留多轮方向一致、增幅超过 1.5% 的方案（作为其他优化前提的除外），保留项用必要最小的改动实现。同一场次正反序交错测量，测试间隔 2 分钟；矩阵乘类改动按格式、按形状做内核级 A/B，注意力类改动在 Qwen 与 gemma 的结构上验证。`test-backend-ops` 全部通过（MUL_MAT 4099、MUL_MAT_ADD 384、MUL_MAT_ID 1109、FLASH_ATTN_EXT 5100、GATED_DELTA_NET 37）。精度以对上游完整 logits 的 KL 散度衡量：除 I8 外的改动在 Qwen 上与上游几乎逐位一致，在 gemma 上与上游自身 `-fa on`/`-fa off` 的差同一量级；I8（int8 预填充）是唯一有实质代价的一项，Qwen 约 0.0007，gemma 约为参照的 3 倍，可用 `GGML_METAL_MM_I8_DISABLE=1` 单独关闭（[第 9 节](metal-m5.md#9-数值精度)）。

## 剩余差距

MoE 预填充仍比 oMLX 慢 3-9%（按专家分块的 mat-mat 仍明显慢于稠密路径）；Qwen 的 MTP 解码在 8k 上落后 oMLX（42.7 对 57.5 t/s），主要在投机解码层；短上下文解码约为带宽上限的 76-83%；I8 在 gemma 上有可见的精度代价；通过 `kv_rows` 的投机解码验证（4 token 一步）还没有测。详见[第 10 节](metal-m5.md#10-剩余差距)。

## 代码与 AI 使用说明

代码在 `LXD-8/llama.cpp` 的 `metal-m5` 分支，基于上游 `master` 的 `b8abe41a50`（已过 b12283），每项改动一个提交（H3、H5 与 H3b 合为一个）。本次 b8abe41a50 rebase 逐个提交核对过：没有上游改动重复或替代这些优化，也没有删除任何优化；唯一的 Metal 改动（MUL_MAT 的 src1 精度门控）已通过在图明确要求精度时保留 f32 激活来遵从。项目目标、测试方法和取舍由我决定；调研、编写代码、测试和撰写文档时使用了 AI 工具，产出经过我人工检查。中文为原文，英文版由 AI 翻译。
