# Apple M5（NAX）上的 llama.cpp Metal 优化：简略版

[中文](#apple-m5nax上的-llamacpp-metal-优化简略版) | [English](#llamacpp-metal-optimizations-for-apple-m5-nax-short-version)

本项目优化 llama.cpp 的 Metal 后端在 Apple M5 上的性能，目标是让各种模型（稠密、MoE、不同注意力结构）和量化格式都基本达到硬件上限，范围包括预填充、解码、投机解码、并发、短提示、启动和多轮对话缓存。15 个代码提交，基于上游 `master` 的 `62df1815b`（b11632），每项改动一个提交、可单独关闭。测试设备：MacBook Pro（Mac17,6），Apple M5 Max，40 核 GPU，128 GB，macOS 27.2。详见[完整报告](metal-m5.md)。

全部改动开启，与上游、oMLX 对比（满功率，服务器端到端；oMLX 开启 INT8 预填充，两边都关 KV 量化）：

| 模型 | 8k 预填充 t/s：本分支 / 上游 / oMLX | 对上游 | 30k 解码 t/s：本 / 上 / oMLX | 对上游 | 30k 投机解码 t/s：本 / 上 / oMLX |
|---|---|---|---|---|---|
| Qwen3.8-27B Q5_K_M | 1101 / 652 / 1101 | 1.69x | 22.8 / 21.9 / 23.4 | 1.04x | 40.8 / 32.9 / 43.7 |
| gemma-4-26B-A4B Q4_0 | 4211 / 3714 / 4343 | 1.13x | 102.7 / 84.7 / 95.0 | 1.21x | 109.8 / 76.6 / 81.1 |
| gpt-oss-20b MXFP4 | 4605 / 3488 / 4898 | 1.32x | 126.6 / 108.1 / 100.4 | 1.17x | - |
| Llama-3.1-8B Q4_K_M | 3871 / 2274 / 3079 | 1.70x | 58.3 / 55.5 / 60.9 | 1.05x | - |
| Llama-3.1-8B Q8_0 | 3204 / 1700 / - | 1.88x | - | - | - |
| Llama-3.1-8B IQ4_XS | 3172 / 1676 / - | 1.89x | - | - | - |

- 预填充：稠密模型 llama-bench `pp2048` 为上游的 1.5-2.0 倍（Qwen 1.75 倍，Llama 三种格式在 8k / 32k 深度 1.9-2.0 倍），MoE 模型 1.1-1.9 倍（gemma 1.14-1.33 倍，gpt-oss 1.26-1.86 倍，深度越大越多）。服务器端到端 8k 提示：Qwen 1.69 倍，Llama 1.70 倍，gpt-oss 1.32 倍，gemma 1.13 倍。
- 解码：短上下文与上游持平（0.99-1.01 倍），长上下文更快：30k 时 Qwen +4%、Llama +5%、gpt-oss +17%、gemma +21%（H3 的 GQA 打包注意力）。
- 投机解码与小批量：一步 16-64 个 token 是上游的 1.0-2.7 倍（H4 / M1）；30k 上 Qwen 的 MTP 40.8 对 32.9 t/s，gemma 的 DFlash 109.8 对 76.6 t/s。
- 并发（App 的 4 路布局）：2-4 路合计 1.0-1.22 倍，4 路投机解码 Qwen 1.36 倍、gemma 1.18 倍。gemma 单路投机解码 0.97 倍是这一个提示上的接受率差异（0.206 对关闭 A2 门槛时 0.227，提示的数值略有不同导致生成的文本不同），不是内核变慢：这一步里被门槛影响的只有 342 token 的提示预填充。
- 与 oMLX：预填充 Qwen 持平、Llama 快 26%，gemma / gpt-oss 慢 3-9%；解码 gemma 快 8%、gpt-oss 快 26%，Qwen / Llama 慢 2-7%（Llama 部分来自 oMLX 的 4-bit 文件小 8%）；Qwen 的 MTP 在 8k 上 oMLX 明显更快（见第 11 节）。
- 启动与内存：Qwen 就绪 1.9 对 2.7 s，常驻内存 3.3 对约 21 GB；关闭思考后不再重新处理整段对话。
- 精度：除 I8 外的改动与上游一致到参照量级；I8 在 gemma 上有可见代价，可单独关闭（第 10 节）。

## 改动

| # | 改动 | 主要效果 |
|---|---|---|
| A1 | 寄存器常驻的 NAX flash attention（预填充，head size 64-512，含 sinks） | Qwen 预填充留一法 +3.5%；gpt-oss pp2048 @32k +63% |
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
| H3 | GQA 打包的 NAX 注意力（解码、投机解码验证） | 解码 @32k +8~19%，4 token 一步 @32k 1.2-2.0 倍 |
| H4 | 2-64 行的 NAX few-row mat-mat，所有格式共用一个内核 | 一步 16 / 32 / 64 token 1.6 / 2.4 / 1.4 倍（Qwen） |
| M1 | MoE mat-mat 从 16 个 token 起使用 | 一步 16-28 token：gemma +6~30%，gpt-oss +39~90% |
| M2 | MoE mat-mat 每块 64 个条目 | gpt-oss 预填充 +24~37%，gemma +4.5~9% |

预填充的单项数字中，A2、A3 是单独开启（它们是后面改动的前提），其余是留一法，所以不能相加。详见[第 6 节](metal-m5.md#6-各项结果与取舍)。

## 量化格式

- 预填充（A3 / I8 / A5）：F16 与 23 种量化格式；新增的格式在合成权重上每个 mat-mul 快 41-49%。
- 2-64 行（H4）：26 种格式按格式和行数开启；17-32 行快 1.1-3.3 倍，33-64 行快 1.0-2.0 倍。
- MoE（M1 / M2）：全部格式；M2 在 25 种格式上逐一测过，每专家 128 个条目以上时快 5-35%。
- 真实模型：q5_K / q6_K（Qwen）、q4_0（gemma）、mxfp4 + q8_0（gpt-oss）、q8_0 / q4_K / iq4_xs（Llama）。详见[第 7 节](metal-m5.md#7-量化格式)。

## 方法与精度

先算差距账，再筛选候选；只保留多轮方向一致、增幅超过 1.5% 的方案（作为其他优化前提的除外），保留项用必要最小的改动实现。同一场次正反序交错测量，测试间隔 2 分钟；矩阵乘类改动按格式、按形状做内核级 A/B，注意力类改动在四种模型结构上验证。`test-backend-ops` 全部通过（MUL_MAT 3892、MUL_MAT_ADD 384、MUL_MAT_ID 867、FLASH_ATTN_EXT 4973、GATED_DELTA_NET 30）。精度以对上游完整 logits 的 KL 散度衡量：除 I8 外的改动在 Qwen、Llama 上与上游几乎逐位一致，在 gemma、gpt-oss 上与上游自身 `-fa on`/`-fa off` 的差同一量级；I8（int8 预填充）是唯一有实质代价的一项，Qwen / Llama 约 0.0007 / 0.0004，gemma 约为参照的 3 倍，可用 `GGML_METAL_MM_I8_DISABLE=1` 单独关闭（[第 10 节](metal-m5.md#10-数值精度)）。试过但未保留的 40 多项方案及原因见[第 9 节](metal-m5.md#9-试过但未保留)。

## 剩余差距

MoE 预填充仍比 oMLX 慢 3-9%（按专家分块的 mat-mat 仍明显慢于稠密路径）；Qwen 的 MTP 解码在 8k 上落后 oMLX（42.7 对 57.5 t/s），主要在投机解码层；短上下文解码约为带宽上限的 76-83%；I8 在 gemma 上有可见的精度代价。详见[第 11 节](metal-m5.md#11-剩余差距)。

## 代码与 AI 使用说明

代码在 `LXD-8/llama.cpp` 的 `metal-m5` 分支，基于上游 `master` 的 `62df1815b`（b11632），每项改动一个提交。项目目标、测试方法和取舍由我决定；调研、编写代码、测试和撰写文档时使用了 AI 工具，产出经过我人工检查。中文为原文，英文版由 AI 翻译。

---

# llama.cpp Metal optimizations for Apple M5 (NAX): short version

This project optimizes the Metal backend of llama.cpp on Apple M5, with the goal that every kind of model (dense, MoE, different attention structures) and weight format runs close to the hardware limit, covering prefill, decode, speculative decoding, concurrency, short prompts, start-up and the multi-turn conversation cache. 15 code commits on upstream `master` at `62df1815b` (b11632), one commit per change, each can be switched off on its own. Test machine: MacBook Pro (Mac17,6), Apple M5 Max, 40-core GPU, 128 GB, macOS 27.2. See the [full report](metal-m5.md).

All changes on, compared with upstream and oMLX (full power, server end to end; oMLX with INT8 prefill on, KV quantization off on both sides):

| Model | 8k prefill t/s: ours / upstream / oMLX | vs upstream | 30k decode t/s: ours / up / oMLX | vs upstream | 30k speculative decode t/s: ours / up / oMLX |
|---|---|---|---|---|---|
| Qwen3.8-27B Q5_K_M | 1101 / 652 / 1101 | 1.69x | 22.8 / 21.9 / 23.4 | 1.04x | 40.8 / 32.9 / 43.7 |
| gemma-4-26B-A4B Q4_0 | 4211 / 3714 / 4343 | 1.13x | 102.7 / 84.7 / 95.0 | 1.21x | 109.8 / 76.6 / 81.1 |
| gpt-oss-20b MXFP4 | 4605 / 3488 / 4898 | 1.32x | 126.6 / 108.1 / 100.4 | 1.17x | - |
| Llama-3.1-8B Q4_K_M | 3871 / 2274 / 3079 | 1.70x | 58.3 / 55.5 / 60.9 | 1.05x | - |
| Llama-3.1-8B Q8_0 | 3204 / 1700 / - | 1.88x | - | - | - |
| Llama-3.1-8B IQ4_XS | 3172 / 1676 / - | 1.89x | - | - | - |

- Prefill: on dense models llama-bench `pp2048` is 1.5-2.0x upstream (Qwen 1.75x, the three Llama formats 1.9-2.0x at 8k / 32k depth), on MoE models 1.1-1.9x (gemma 1.14-1.33x, gpt-oss 1.26-1.86x, more at larger depth). Server end to end with an 8k prompt: Qwen 1.69x, Llama 1.70x, gpt-oss 1.32x, gemma 1.13x.
- Decode: on par with upstream at short context (0.99-1.01x), faster at long context: at 30k Qwen +4%, Llama +5%, gpt-oss +17%, gemma +21% (H3, the GQA-packed attention).
- Speculative decoding and small batches: steps of 16-64 tokens are 1.0-2.7x upstream (H4 / M1); at 30k, Qwen MTP 40.8 vs 32.9 t/s, gemma DFlash 109.8 vs 76.6 t/s.
- Concurrency (the app's 4-slot layout): 2-4 streams 1.0-1.22x in total, 4 speculative streams Qwen 1.36x, gemma 1.18x. The 0.97x for a single speculative gemma stream is a difference in acceptance on this one prompt (0.206 vs 0.227 with the A2 gate off, slightly different prompt numerics lead to different generated text), not a slower kernel: the only work the gate touches there is the 342-token prompt prefill.
- Against oMLX: prefill on par for Qwen, 26% faster for Llama, 3-9% slower for gemma / gpt-oss; decode 8% faster for gemma, 26% for gpt-oss, 2-7% slower for Qwen / Llama (for Llama partly because oMLX's 4-bit file is 8% smaller); Qwen MTP at 8k is clearly faster in oMLX (section 11).
- Start-up and memory: Qwen ready in 1.9 vs 2.7 s, resident memory 3.3 vs about 21 GB; turning thinking off no longer reprocesses the whole conversation.
- Accuracy: the changes other than I8 match upstream to the order of the reference; I8 has a visible cost on gemma and can be switched off alone (section 10).

## Changes

| # | Change | Main effect |
|---|---|---|
| A1 | Register-resident NAX flash attention (prefill, head size 64-512, with sinks) | Qwen prefill leave-one-out +3.5%; gpt-oss pp2048 @32k +63% |
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
| H3 | GQA-packed NAX attention (decode, speculative verify) | decode @32k +8-19%, step of 4 tokens @32k 1.2-2.0x |
| H4 | NAX few-row mat-mat for 2-64 rows, one kernel for every format | step of 16 / 32 / 64 tokens 1.6 / 2.4 / 1.4x (Qwen) |
| M1 | MoE mat-mat from 16 tokens | steps of 16-28 tokens: gemma +6-30%, gpt-oss +39-90% |
| M2 | MoE mat-mat tiles of 64 entries | gpt-oss prefill +24-37%, gemma +4.5-9% |

Of the prefill numbers, A2 and A3 are alone (they are prerequisites of later changes), the others leave-one-out, so they do not add up. See [section 6](metal-m5.md#6-per-change-results-and-selection).

## Weight formats

- Prefill (A3 / I8 / A5): F16 and 23 quantized formats; the newly covered formats are 41-49% faster per mat-mul on synthetic weights.
- 2-64 rows (H4): 26 formats enabled per format and row count; 1.1-3.3x at 17-32 rows, 1.0-2.0x at 33-64 rows.
- MoE (M1 / M2): every format; M2 measured on 25 formats one by one, 5-35% faster from 128 entries per expert.
- Real models: q5_K / q6_K (Qwen), q4_0 (gemma), mxfp4 + q8_0 (gpt-oss), q8_0 / q4_K / iq4_xs (Llama). See [section 7](metal-m5.md#7-weight-formats).

## Method and accuracy

Account for the gap first, then screen candidates; keep only changes with a gain above 1.5% in a consistent direction over several rounds (except prerequisites of other optimizations), implemented with the minimum necessary code. Interleaved forward/reverse measurements within a session, 2 minutes between tests; mat-mul changes A/B tested at kernel level per format and per shape, attention changes verified on four model structures. `test-backend-ops` passes (MUL_MAT 3892, MUL_MAT_ADD 384, MUL_MAT_ID 867, FLASH_ATTN_EXT 4973, GATED_DELTA_NET 30). Accuracy is measured as the KL divergence to the full upstream logits: the changes other than I8 are practically bit-identical to upstream on Qwen and Llama and of the same order as upstream's own `-fa on`/`-fa off` difference on gemma and gpt-oss; I8 (int8 prefill) is the only change with a real cost, about 0.0007 / 0.0004 on Qwen / Llama and about 3x the reference on gemma, and `GGML_METAL_MM_I8_DISABLE=1` turns it off alone ([section 10](metal-m5.md#10-accuracy)). More than 40 approaches tried and not kept, with reasons, in [section 9](metal-m5.md#9-tried-and-not-kept).

## Remaining gaps

MoE prefill is still 3-9% behind oMLX (the per-expert mat-mat is still clearly slower than the dense path); Qwen MTP decode at 8k trails oMLX (42.7 vs 57.5 t/s), mostly in the speculative layer; short-context decode is at about 76-83% of the bandwidth limit; I8 has a visible accuracy cost on gemma. See [section 11](metal-m5.md#11-remaining-gaps).

## Code and use of AI

The code is on the `metal-m5` branch of `LXD-8/llama.cpp`, based on upstream `master` at `62df1815b` (b11632), one commit per change. The goals, test methodology and decisions are my own; AI tools were used for research, writing code, testing and writing the documents, and their output was reviewed by me. The Chinese version is the original; the English version was translated by AI.
