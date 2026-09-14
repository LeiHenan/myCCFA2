# Prior art: LLM inference serving on consumer Ada — 8× RTX 4090, 24 GB, PCIe-only, no NVLink, 2 NUMA nodes

Research date: **2026-09-14**. Every arXiv ID below was verified by fetching `arxiv.org/abs/<id>` and reading the title back. Every URL below was actually fetched with `curl`; every quote marked VERBATIM is text that appeared in the fetched content.

**Headline answer:** the *box* (8× RTX 4090, PCIe-only, 24 GB) is used in published work, and the balance of that work is **training / fine-tuning** (RoundPipe 2604.27085, LLMQ 2512.15306). For **inference serving** on that box there is exactly one paper I verified — **2311.03687**, which benchmarks an **8× RTX 4090 / PCIe 4.0 x16** server against 8× A800 (NVLink) and 8× RTX 3090, had to set `NCCL_P2P_DISABLE=1`, and then openly states its cross-platform numbers may be invalidated by that setting. Beyond it: a 2026 peer-reviewed journal paper doing controlled **TP-vs-PP on 2× RTX 3090 PCIe** (BNTU), and community measurements on **4× RTX 4090** (Qiita, cnblogs, kentino). **I did not find a published TP-vs-PP scaling measurement specifically on 8× RTX 4090.**

**Three things to know before claiming novelty:**
1. **The exact 8×4090 hardware string is prior art** — RoundPipe (2604.27085) evaluates on it, verbatim: *"4090 server: 8× NVIDIA RTX 4090 GPUs (24 GB VRAM each) … PCIe 4.0 (32 GB/s) interconnect."* But for **fine-tuning**, and it avoids GPU P2P entirely.
2. **"TP is bad / PP is right on PCIe-only consumer hardware" is peer-reviewed** — BNTU 2026: *"The results unequivocally indicate the unsuitability of Tensor Parallelism for systems without NVLink due to critical synchronization delays."* (on 2× 3090).
3. **NUMA/PCIe-topology-aware placement is established prior art** — Mobius (ASPLOS '23, root-complex-aware stage mapping) and 2508.20274 (PCIe-aware placement for LLM serving, measured). Do not re-map this as novel.

---

## Section A — The "home-lab 4090 box" literature and community corpus

### A.1 `noonghunna/club-3090` — the most rigorous consumer-PCIe corpus I found

Repo: <https://github.com/noonghunna/club-3090> · multi-card doc: <https://github.com/noonghunna/club-3090/blob/master/docs/MULTI_CARD.md>

**Hardware ceiling, stated up front** (VERBATIM, MULTI_CARD.md):
> "⛔ This rig has 2 GPUs. Permanently. Nothing above TP=2 is validated here, and nothing above TP=2 ever will be — that is the maintainer's hardware ceiling, not a backlog item."
> "TP=8 has zero boots anywhere. Those sections are arithmetic, and labelled as such."

**TP vs PP vs EP traffic** (VERBATIM, `docs/PCIE_P2P.md`, §0a) — the single cleanest statement of *why* PP wins on PCIe I found anywhere:

| Loading mode | inter-GPU traffic | does P2P help? |
|---|---|---|
| PP (layer) | activations at the boundary — **~7 MB/s measured** | Barely. Nothing to accelerate |
| TP (row/tensor) | all-reduce every layer — **~1400 MB/s, many small serialized transfers** | Yes — and via LATENCY, not bandwidth. **15.23 → 1.01 µs** is the real win |

> "⚠️ PP is not parallelism at -np 1. With a single sequence the GPUs run in sequence — GPU0 does its layers while GPU1 idles, then swaps. Measured: **1 GPU 128.72 → 2 GPU layer 144.29 TPS, just +12%**. The second card is buying capacity, not speed. Expect real pipelining only with multiple in-flight sequences."
> "⚠️ Reason about latency, not bandwidth. Link utilisation is often under 20% while the latency saving is 15×. Concluding 'P2P can't matter, we're not bandwidth-bound' is the single most common analysis error here — we made it ourselves."

**Replicate-vs-TP, with numbers** (VERBATIM, MULTI_CARD.md; measured by @alesha-pro on 4× 3090 PCIe, all arms matched at 220 W, peak aggregate tok/s):

| Model | TP=4, 1 instance | replicated | gain |
|---|---|---|---|
| gemma-4-12B AWQ | 1,007 | 3,425 (TP=1 × 4) | 3.40× |
| Qwen3.6-27B INT8-W8A8 | 358 | 741 (TP=2 × 2) | 2.07× |
| Qwen3.6-27B INT4 | 307 | 737 (TP=2 × 2) | 2.40× |

> "If you read one section of this page, read this one. Getting it wrong costs up to **3.4× aggregate throughput**, and no amount of TP tuning further down recovers it."
> "**The rule: latency wants max TP, throughput wants max instances.**"

And the explicit caveat that per-stream scaling is *recipe-dependent, not topological* (VERBATIM):
> "⚠️ 'Per-stream TPS doesn't scale on PCIe' is a property of our recipe, not a law of topology. … On a matched-recipe A/B (same rig, same sitting, @alesha-pro): **prefill @90K went 1,202 → 1,948 tok/s (+62%) and TTFT @90K 73.1 s → 45.6 s (−37.6%)**. Cards 3 and 4 buy **prefill and TTFT**, not decode."

**vLLM's own TP gate on PCIe** (VERBATIM, PCIE_P2P.md):
> "vLLM's own log: the `Custom allreduce is disabled…` line means no. **At >2 GPUs it requires a full NVLink mesh and never consults P2P (#786)** — so on 3+-card PCIe rigs layer 3 is always off, by design, not misconfiguration"
> "We measured the bypass and deliberately do not ship it. Forcing `is_fully_connected → True` at TP=4 on a patched 4×3090 gives ≈ **+15% decode**, independently reproduced by @superalesha at **+14.8% (80.9 → 92.9 tok/s single-stream, 144 → 165 at 16 users)** — with prefill paying for it (TTFT +9% at c16)"

**NUMA / topology relevance for our box** (VERBATIM):
> "Two pairs can both report PHB and differ 2× in real bandwidth. Worked example from a community 3× 3090 rig (2026-08-05): all three pairs showed PHB, yet [P2P bandwidth] **13.17 GB/s** … **~6.5 GB/s**. Only `p2pBandwidthLatencyTest` exposed it."
> "Set NPS1 (one NUMA node per socket) so both GPUs share a domain and report PHB … **On a true multi-socket box, keep both GPUs on the same socket (otherwise you get SYS, the worst case).**"
> "On awkward counts, pick the best-connected pair or quad, not the first N. Inspect with `nvidia-smi topo -m` and prefer **NV# > PIX > PXB > PHB > SYS**, then pin with `CUDA_VISIBLE_DEVICES`."

**Bearing on 8× 4090 PCIe-only / 2 NUMA:** *high for the topology+NUMA discipline and for the "link class, not link type" finding; but the rig is 2× 3090, so none of it is Ada and none of it is 8 cards. It is the best-documented community PCIe methodology, not a 4090 measurement.*

Measured P2P bandwidth table on that 2× 3090 PCIe gen4 x16 rig (VERBATIM, §4a):

| metric | P2P off | P2P on | gain |
|---|---|---|---|
| Unidirectional | 11.28 GB/s | 27.12 GB/s | 2.4× |
| Bidirectional | 16.77 GB/s | 54.21 GB/s | 3.2× |
| Latency | 15.23 µs | 1.01 µs | 15× |

> "27 GB/s is gen4 x16 line rate — the full physical link, not a partial path."
> "⛔⛔ VERDICT ON THIS RIG: the clique grant BROKE both engines' collective paths … `llama.cpp --split-mode tensor` ❌ hangs in warmup · `vLLM TP=2` ❌ hangs at `ncclCommInitRank`"
> "⚠️ Corrected 2026-08-10 — the copies never worked either. … every peer path (CE copy, SM store, SM load, system-scope atomics) misroutes to a private per-direction backing."

### A.2 4× RTX 4090 PCIe-only, vLLM, **with a TP=4 vs TP=2 A/B on the same box** — the best 4090 numbers I found

Source: <https://www.cnblogs.com/softlin/p/22508619> (AiFly, 2026-08-17). Rig: 4× RTX 4090 24 GB, **PCIe interconnect, no NVLink**; vLLM 0.25.0; Qwen3.8-27B-AWQ-INT4; 262,144 ctx; `--kv-cache-dtype fp8_e4m3`; `--tensor-parallel-size 4`.

VERBATIM (translated from Chinese):
> "实测吞吐——单流 ~108 tok/s、聚合 ~2,274 tok/s、预填峰值 ~4,400 tok/s——**基本只到单张 4090 的量级**… 四张卡没有换来四倍性能，头号嫌疑是 4090 之间只有 PCIe（无 NVLink），张量并行每一步的通信开销吃掉了多卡收益。"
> ("measured throughput — single stream ~108 tok/s, aggregate ~2,274 tok/s, prefill peak ~4,400 tok/s — **basically only at the level of a single 4090** … four cards did not buy 4× performance; the prime suspect is that the 4090s have only PCIe (no NVLink), and tensor parallelism's per-step communication overhead eats the multi-GPU gain.")
> "参照系：H100 跑 27B FP8 预填 10~20K tok/s；本机四张卡实测峰值 4.4K，**仅相当于单张 4090 水平**——结合 PCIe 互联推断，**TP 通信开销才是首要瓶颈**"
> ("reference point: H100 does 10–20K tok/s prefill on 27B FP8; this machine's 4-card measured peak is 4.4K, **only equivalent to a single 4090** — combined with PCIe interconnect, **TP communication overhead is the primary bottleneck**")

Same box re-measured at TP=2 (`CUDA_VISIBLE_DEVICES=3,4`, all other params identical):

| metric | TP=4 (measured) | TP=2 (predicted) | TP=2 (measured) |
|---|---|---|---|
| KV cache total capacity | **1,629K tokens** | 600–700K | **469K** |
| 256K-context headroom | 6.2× | ok | 1.79× |
| simultaneous full-length requests | ~6 | ~2 | **1** |
| single-stream decode | ~108 tok/s | 90–110 | **79 tok/s** |
| aggregate decode saturation | ~2,274 tok/s | 1,200–1,500 | ~1,560 tok/s |
| prefill peak | ~4,400 tok/s | 2,200–2,500 | ~3,200 tok/s |
| 240K prefill wall time | 89.7 s | ~2× | 128.7 s (×1.43) |

> "预填充好于'减半'直觉：双卡拿到四卡 **73%** 的预填速率（32K 峰值 ~3,200 tok/s）。**TP=2 的通信开销显著低于 TP=4，单卡效率反而更高**——反向印证了四卡 PCIe 通信吃收益的判断。"
> ("prefill is better than the 'halving' intuition: two cards get **73%** of four cards' prefill rate. **TP=2's communication overhead is significantly lower than TP=4, so per-card efficiency is actually higher** — this inversely confirms the judgement that 4-card PCIe communication eats the gains.")

**Bearing on our box:** *This is the single most on-point community measurement — same GPU (4090), same link (PCIe, no NVLink), same framework, and it contains a real TP-degree A/B with KV-capacity numbers. But it is 4 cards on one NUMA-legal board, not 8 cards across 2 NUMA nodes, and the author is a blogger, not a peer-reviewed source.*

### A.3 4× RTX 4090, vLLM **and** SGLang, with a per-step overhead decomposition

Source: <https://qiita.com/engchina/items/f2621c103f9e695a4d02> (vLLM, 2026) and <https://qiita.com/engchina/items/d37f3169be4da2d31548> (SGLang, 2026-08-27). Same physical machine across four articles. Rig: **NVIDIA GeForce RTX 4090 24GB × 4, GPU connection: PCIe only (no NVLink, P2P unsupported)**, Windows + WSL2; Qwen3.8-27B BF16 (checkpoint 51.75 GiB).

VERBATIM (translated):
> "GPU 接続: PCIe のみ（NVLink なし、**P2P 非対応**）"
> "**Custom allreduce is disabled because it's not supported on more than two PCIe-only GPUs**" *(this is vLLM's own log line, quoted by the author)*
> "→ 4090 は symmetric memory 非対応。PYNCCL にフォールバックします。" ("4090 does not support symmetric memory. It falls back to PYNCCL.")

**The PCIe TP overhead decomposition — the most quantitative "what does PCIe-only cost you per step" figure I found:**
> "重み 51.75 GiB を 4 枚で割ると 12.94 GiB / 枚 / ステップ、RTX 4090 の帯域を約 939 GiB/s とすると 13.8 ms、つまり **72 tok/s が上限** です。実測 **31 tok/s（32 ms）** なので、**差分の約 18 ms が allreduce とカーネル起動のオーバーヘッド** ということになります。**P2P 非対応の PCIe 4枚構成では、ここは構造的に取り返せません。**"
> ("51.75 GiB of weights ÷ 4 cards = 12.94 GiB/card/step; at the RTX 4090's ~939 GiB/s that is 13.8 ms, i.e. a **ceiling of 72 tok/s**. Measured **31 tok/s (32 ms)**, so the **~18 ms difference is allreduce + kernel-launch overhead**. **On a 4-card P2P-unsupported PCIe configuration this is structurally unrecoverable.**")
> "期待できるのは 1.3x 程度（31.1 → 24 ms 前後）、というのがアムダール則からの上限です。… 実測で判明した「オーバーヘッド 17 ms」は **FP8 でも一切減らない**ので、ここが天井を決めます。"
> ("Amdahl's-law ceiling is ~1.3× (31.1 → ~24 ms). … the measured **17 ms overhead does not shrink at all even with FP8**, so this is what sets the ceiling.")

Benchmarks: 183 tok/s at concurrency 8 (1024 in / 256 out), TPOT 35 ms. TP=4 weights 13.11 GiB/card. Uses `-e NCCL_P2P_DISABLE=1`.

**The Ada-verification gap, stated by a practitioner** (VERBATIM, SGLang article) — this is directly about the hardware axis:
> "なお SGLang 公式 cookbook の Qwen3.8-27B ページには、**Ada（SM89）の verified cell がありません**。検証済みとして載っているのは **H200 / RTX PRO 6000 / RTX 5090 / DGX Spark / GB300** です。以下の設定は cookbook の「幾何学」（層構成とメモリ式）から自分で導いたもので、検証済みレシピの引き写しではありません。"
> ("Also, the SGLang official cookbook's Qwen3.8-27B page has **no verified cell for Ada (SM89)**. The verified entries are **H200 / RTX PRO 6000 / RTX 5090 / DGX Spark / GB300**. The settings below I derived myself from the cookbook's 'geometry' (layer structure and memory formulas); they are not copied from a verified recipe.")

Also VERBATIM on the PCIe-only failure mode:
> "この WSL2 + PCIe-only の箱では、SGLang が TP>1 で無条件に武装する multimem all-gather のせいで rank 0 が SIGFPE（exit code -8）で即死します。"
> ("On this WSL2 + PCIe-only box, SGLang's multimem all-gather — which it unconditionally arms at TP>1 — instantly kills rank 0 with SIGFPE (exit code -8).")
> "これは SGLang 自身が「PCIe-only の GPU が 3 枚以上なら明示的に付けろ」と言ってくるので、素直に従います。**4090 は P2P 非対応なので、ここは構造的にどうにもなりません。**"

Engine delta on the same box: vLLM 29.4 tok/s vs SGLang ~40 tok/s decode (~1.35×); KV pool SGLang 281,143 tokens vs vLLM 440,673 tokens.

**Bearing on our box:** *Extremely direct. Same GPU, same link class, same "P2P unsupported" fact, and it converts the PCIe penalty into a number (≈17–18 ms/step, FP8-invariant). Plus independent testimony that the official SGLang cookbook carries no verified Ada cell.*

### A.4 Vendor case study: 4× RTX 4090, vLLM TP=4, 70B AWQ INT4

Source: <https://kentino.se/blogs/ai-corner/case-study-4x-rtx-4090-ai-workstation> (Swedish; vendor case study — treat numbers as vendor-reported, unrefereed).

Measured interconnect table (VERBATIM, translated):
> "GPU ↔ GPU (PCIe peer-to-peer) | **19–22 GB/s**"
> "Värd → Enhet (PCIe) | 26.2–26.3 GB/s" / "Enhet → Värd (PCIe) | 1.4 GB/s"
> "**GPU-till-GPU-siffran (19–22 GB/s över PCIe peer-to-peer) är den arkitektoniska begränsningen som är relevant för tensorparallell servering. Med NVLink körs denna väg med 900 GB/s. Med endast PCIe får du ungefär 2 % av det.**"
> ("The GPU-to-GPU figure (19–22 GB/s over PCIe peer-to-peer) is the architectural limitation relevant to tensor-parallel serving. With NVLink this path runs at 900 GB/s. With PCIe only you get roughly 2% of that.")

> "Latensen på 2 043 ms på en begäran med 16 tokens är TTFT-golvet … Den huvudsakliga drivkraften är **tensorparallell spridnings-/insamlingsoverhead över fyra GPU:er över PCIe** — varje förfyllningssteg kräver en AllReduce över alla fyra kort via PCIe-strukturen. **Med NVLink skulle detta vara ungefär 50–100 ms TTFT; med PCIe P2P vid 20 GB/s sträcker det sig längre.**"
> ("The 2,043 ms latency … is the TTFT floor … The main driver is **tensor-parallel scatter/gather overhead across four GPUs over PCIe** — every prefill step requires an AllReduce across all four cards via the PCIe fabric. **With NVLink this would be ~50–100 ms TTFT; with PCIe P2P at 20 GB/s it stretches longer.**")

TP=4 results: single request 8.0 tok/s; batch of 32 concurrent → 179.3 tok/s aggregate; TTFT 2,043 ms; all PCIe links confirmed Gen4 x16 under load. (They attribute the low single-stream 8 tok/s to vLLM picking the `awq` kernel instead of `awq_marlin`, expected 2–3× with marlin.)

**Bearing:** *Gives the 4090's measured P2P ceiling (19–22 GB/s) and an NVLink-vs-PCIe TTFT contrast. Vendor source, so I would not build a claim on it alone.*

### A.5 Other community items checked

- **`ikawrakow/ik_llama.cpp` #629** — <https://github.com/ikawrakow/ik_llama.cpp/issues/629>. VERBATIM on the host-staged fallback: *"I don't know if peer-to-peer copy works on your system, but if it doesn't, this is probably quite bad for TG performance because data copies from one GPU to another goes via `GPU1 -> CPU -> GPU2`, which adds quite a bit of extra latency."* Reply: *"It doesn't. **Nvidia blocks it for consumer-level cards in their Windows drivers.**"* Also the graph-split cost: *"For good TG performance you want to have as few graph splits as possible as each graph split requires synchronization and copying data from one device to another, which adds up to a measurable TG performance drop when there are many splits."* And a 7-GPU owner's Windows-vs-Linux delta: *"DeepSeek Q2_K_XL, offloading ~140GB to RAM … **5 t/s PP, 1.5 t/s TG on Windows** vs **60 t/s PP, 7 t/s TG on Linux**."* **Bearing:** confirms host-staged P2P is the consumer reality and that it is an *engine/path* problem, but it is llama.cpp/GGUF and not a controlled 4090 measurement.
- **`ikawrakow/ik_llama.cpp` discussion #532** — "Guidance on GPU Layer Offloading Strategy in ik_llama.cpp for Multi GPU Rig (**2x5090 + 2x4090**)" — exists (title fetched); I did not extract a numeric finding.
- **`steveseguin/b70-optimization-lab`** — real repo, community result packets (e.g. PR #16: "**MoE beats dense** on speed and efficiency (30B-A3B 79 tg/s > dense 14B 47)"), but it is **Intel Arc Pro B70/B60**, not NVIDIA Ada. **Not relevant to the 4090 axis**; noting it because the task named it.
- **`turboderp`/ExLlamaV2 multi-GPU** — I could not retrieve a canonical ExLlamaV2 multi-GPU doc (`doc/multi_gpu.md` → 404; the `leeroopedia.com` "Exllamav2 Multi GPU Compatibility" page is a third-party wiki). The substantive ExLlamaV2-vs-llama.cpp TP argument I did find is Osman's blog: <https://www.ahmadosman.com/blog/do-not-use-llama-cpp-or-ollama-on-multi-gpus-setups-use-vllm-or-exllamav2/> — VERBATIM: *"You should only use llama.cpp when doing partial—or full—CPU offloading of an LLM. But with multi-GPU setups, optimized batch inference with Tensor Parallelism is required, and vLLM or ExLlamaV2—among others—are the correct choices."* Motivated by ThePrimeagen's *"6x RTX 4090 build"*; author runs *"14x RTX 3090 GPUs and 336GB of VRAM"*. **No 4090 TP-vs-PP numbers.**
- **"4090 48GB" modded cards** — <https://www.hardware-corner.net/48gb-rtx-4090-first-tests/>. VERBATIM: *"an NVIDIA GeForce RTX 4090 equipped with a staggering **48GB of GDDR6X** memory, double the stock configuration"*; modders *"employ a custom, longer PCB that features memory pads on both sides, allowing for the installation of 24x 2GB GDDR6X modules (12 front, 12 back)"*; the card is *"dual-slot, blower-style (turbine)"*, *"65 dB in tests"*, *"~1 TB/s memory bandwidth"*. **Bearing:** this is the *capacity escape hatch* from the 24 GB constraint — it removes the memory-capacity forcing function but not the PCIe one, and it doubles nothing about interconnect.
- **8× 4090 CSDN article** — <https://blog.csdn.net/weixin_33402252/article/details/162800118>. Titled "8卡RTX 4090部署DeepSeek：性能测试与成本效益分析". **The performance section is paywalled** ("最低0.47元/天 解锁文章") and the free portion is generic hardware-list text; the site's own "related articles" are near-duplicate SEO posts. **I am not treating this as evidence of an 8×4090 measurement.**
- **`cyberpulstech.com` "Multi-GPU Local LLM Inference in 2026: TP vs PP vs vLLM Distributed Serving"** — SEO-tier guidance ("TP requires massive inter-GPU bandwidth … but delivers maximum generation speed (45+ TPS)") with no original measurement. **Not evidence.**
- **cnblogs "消费集显卡集群生产部署策略"** — <https://www.cnblogs.com/aibi1/p/19439475>. VERBATIM (translated): *"No NVLink → only PCIe + network communication → Tensor Parallel cross-card communication cost is extremely high … in 13B/70B models, communication latency directly eats the benefit of batching."* Recommends *"2–4 4090s per machine, each running an independent vLLM instance … do not: cross-machine Tensor Parallel, model splitting."* **Bearing:** the practitioner consensus is *replicate, do not TP* — consistent with club-3090's measured 3.4×; but no numbers.
- **Green Tinybox / 6×4090 (ThePrimeagen)** — referenced via Osman's blog only; **I did not find a published rigorous benchmark from it.**
- **r/LocalLLaMA** — ⚠️ **reddit.com and old.reddit.com are network-blocked from this environment** (curl returns a "Blocked … network policy" page; `search.json` returns HTTP 403; the `r.jina.ai` proxy returns 401 "bad network reputation"). **I could not read any r/LocalLLaMA thread directly.** What `web_search` surfaced for reddit was either blocked or non-substantive. Treat the r/LocalLLaMA axis as **unsearched by direct reading**.

### A.6 (question A2) — Is there a published TP-vs-PP scaling measurement specifically on **8× RTX 4090 PCIe-only**?

**I did not find one.** The closest items, in descending order of proximity:

1. **2311.03687** — *8× RTX 4090, PCIe 4.0 x16, inference serving with vLLM/LightLLM/TGI, but the parallelism axis is not TP-vs-PP* and the paper runs `NCCL_P2P_DISABLE=1` (see B.4). It is a framework benchmark on the 8×4090 box, not a TP-vs-PP study.
2. **RoundPipe 2604.27085** — *8× RTX 4090, PCIe, 24 GB*, with an explicit **A800 vs 4090** comparison and a **1→8 GPU scaling curve** — but for **fine-tuning**, not serving, and it deliberately uses **no** GPU P2P.
3. **BNTU 2026 journal paper** — a real controlled **TP vs PP on a PCIe-only consumer rig**, but **2× RTX 3090**, not 8× 4090 (see B.5).
4. **cnblogs 4×4090** — a real **TP=4 vs TP=2** A/B on 4× 4090 PCIe-only, same box, same sitting. Not 8 cards, not PP.

**Named queries used (all returned no 8×4090 TP-vs-PP measurement):** `"8x RTX 4090 tensor parallel pipeline parallel comparison measurement published"`, `"\"8× RTX 4090\" tensor parallel scaling efficiency published study"`, `"eight consumer GPUs PCIe pipeline parallel vs tensor parallel measured throughput paper"`, `"8x RTX 4090 vLLM tensor parallel vs pipeline parallel benchmark"`, `"8卡RTX 4090 张量并行 流水线并行 实测"`, `"vLLM blog post consumer GPUs multi-GPU tensor parallel PCIe scaling efficiency 4090 3090"`. **Surfaces:** web_search (multiple phrasings), arxiv.org/abs + arxiv.org/html for every ID encountered, github.com repo-scoped issue search, huggingface.co/papers, aclanthology.org, cnblogs/CSDN/Qiita/Zenn, kentino.se, hardware-corner.net.

### A.7 (question A3) — RTX 4090 GPU-to-GPU P2P disabled, and the host-staged fallback in GB/s

**This is well documented, and it is Ada-specific in a way the 3090 findings are not.**

**(a) The canonical measurement — `NVIDIA/nccl-tests` issue #117**, <https://github.com/NVIDIA/nccl-tests/issues/117>. Title (fetched): *"NCCL all_reduce_perf test hangs with multiple RTX 4090 GPUs, works fine when I swap in 2080tis."* VERBATIM from the issue thread — a user's `p2pBandwidthLatencyTest` on 2× RTX 4090, dual EPYC 7B13, **NPS4**:

```
P2P Connectivity Matrix
     D\D     0     1
     0       1     1
     1       1     1

Unidirectional P2P=Disabled Bandwidth Matrix (GB/s)
   D\D     0      1
     0 912.14  11.97
     1  12.02 920.74
```

> "by the way, cuda `bandwidthTest` and `./p2pBandwidthLatencyTest` worked."
> "**You are not alone, 2080ti passed but 4090 failed.**"
> "I have IOMMU and ACS turned off in the BIOS. CPU virtualization is also off. **Numa NPS set to 4.** … When I take out the 4090s and replace with 2080tis, the test runs normally."
> "although the p2pBandwidthLatencyTest shows p2p is enabled on the new 4090s, perhaps it actually doesn't work or is disabled, and the NCCL test freezes because of it."

**Numbers:** the **host-staged (P2P-disabled) unidirectional GPU↔GPU path on 4090 measures ~12 GB/s** (11.97 / 12.02 GB/s) against ~912–920 GB/s device-local. `NCCL_P2P_DISABLE=1` makes `all_reduce_perf` hang disappear — i.e. the fallback works but the direct path does not.

This issue is **cited by an academic paper** (see B.4) as *"An acknowledged bug exists for the RTX40X0 GPU series."* That is important: the 4090's P2P defect is not folklore, it is in the literature.

**(b) The SysMem-staging penalty, quantified at ~15×** — `local-inference-lab/rtx6kpro`, `optimization/pcie-oneshot-allreduce.md`: <https://github.com/local-inference-lab/rtx6kpro>. VERBATIM:
> "On **direct-attach topologies (NODE — GPUs connected through CPU root ports, no PCIe switch)**, the nvidia driver defaults to **SysMem staging** for GPU-to-GPU memory accesses from CUDA kernels. This makes the PCIe oneshot allreduce **~15× slower than NCCL**, completely negating its benefit."
> "**This affects the majority of users** — PCIe switches are uncommon."
> "`nvidia-smi topo -m` shows **NODE** (not PIX/PXB) between GPUs"
> "NCCL uses its own SHM transport for small messages — doesn't rely on SM P2P loads, unaffected · `cudaMemcpy` D2D uses the DMA copy engine — separate path, unaffected · PCIe oneshot allreduce issues direct `load` instructions from CUDA kernels reading remote GPU memory via IPC handles — this goes through the driver's P2P BAR1 mapping, which requires `ForceP2P`"

Their driver-level workaround: `options nvidia NVreg_RegistryDwords="ForceP2P=0x11;RMForceP2PType=1;RMPcieP2PType=2;GrdmaPciTopoCheckOverride=1;EnableResizableBar=1"`. **Note this is a *different mechanism* from club-3090's bare-metal BAR1/Aikitoria patched-module path and from the QEMU `x-nv-gpudirect-clique` path — three independent schemes to get P2P on consumer cards, all fragile.** This repo is **RTX PRO 6000 Blackwell / SM120**, not Ada, but the `NODE`-vs-`PIX/PXB` driver default is a driver behaviour, not a silicon one.

**(c) 4090 measured P2P when it *is* granted** — kentino's 4× 4090 box: **19–22 GB/s** GPU↔GPU PCIe peer-to-peer, vs ~920 GB/s VRAM, vs 900 GB/s NVLink. Their framing (VERBATIM, translated): *"With PCIe only you get roughly 2% of that."*

**(d) The 3090 comparison point** — club-3090 measures **11.28 GB/s** with P2P off and **27.12 GB/s** with P2P on, latency **15.23 µs → 1.01 µs**, on PCIe gen4 x16.

**Bearing on our box:** *The ~12 GB/s host-staged figure and the ~19–22 GB/s granted-P2P figure bracket what an 8×4090 all-reduce can possibly run on. The critical, non-obvious consequence — stated by club-3090 and independently by the rtx6kpro repo — is that at these bandwidths TP all-reduce is **latency**-bound, not bandwidth-bound, so GB/s reasoning understates the damage. The rtx6kpro doc is the only source I found that quantifies the **cross-NUMA** penalty of a barrier-based collective: **-4.3% at ctx=0, -5.2% at 16K, -15.7% at 32K on 8 GPUs across two sockets**, with the crossover threshold collapsing from 120 KB (4 GPU, same NUMA) to 48 KB (8 GPU, cross-NUMA).*

---

## Section B — Academic papers that use consumer Ada / 4090 in their evaluation

### B.4 Papers whose evaluation testbed is explicitly RTX 4090

#### B.4.1 Quest — **verified**: arXiv **2406.10774**, *"Quest: Query-Aware Sparsity for Efficient Long-Context LLM Inference"*, submitted 16 Jun 2024, revised 26 Aug 2024.
<https://arxiv.org/abs/2406.10774> · full text <https://arxiv.org/html/2406.10774v2>

The task's claim is **correct, with an important qualification**. VERBATIM from the full text:
> "Tested with **FP16 FlashInfer implementation on an RTX4090**, limiting the overall throughput."
> "We first evaluate Quest's kernel-level efficiency under the configuration of **Llama2-7B on an RTX4090** with CUDA 12.2 in Sec 4.3.1. Besides, we show the end-to-end speedup of Quest in text generation as shown in Sec 4.3.2. … Note that we use an **Ada 6000 GPU** in [the end-to-end setting]"

**Bearing:** *The 4090 is used for the **kernel-level** (self-attention) evaluation; the **end-to-end** evaluation moves to an RTX 6000 Ada. It is a **single-GPU** paper — no PCIe, no multi-GPU, no NUMA. So it is an example of "4090 used as the consumer stand-in for a kernel result", not prior art on the interconnect axis.*

#### B.4.2 **"Dissecting the Runtime Performance of the Training, Fine-tuning, and Inference of Large Language Models"** — **verified**: arXiv **2311.03687**, submitted 7 Nov 2023, revised 1 Dec 2023.
<https://arxiv.org/abs/2311.03687> · full text <https://arxiv.org/html/2311.03687v2>

**This is the most important paper I found for the 8×4090 axis.** Table I (VERBATIM):

| Platform | A800 | **RTX4090** | RTX3090 |
|---|---|---|---|
| GPU | A800-80G | **RTX4090-24G** | RTX3090-24G |
| CPU | 2× AMD EPYC 7402 | **2× Intel Xeon Gold 6230** | 2× AMD EPYC 7302 |
| Memory | 512 GiB DDR4 | **512 GB DDR4** | 128 GB |
| Network | NVLink | **PCIe4.0x16** | NVLink |

> "we benchmark the runtime and memory performance of existing systems in the LLM pipeline … (1) On the framework-level, we choose DeepSpeed and Megatron-LM … on three types of hardware (**A800, RTX4090, and RTX3090 servers**). … (4) We study the end-to-end inference performance using highly optimized inference libraries including **vLLM, LightLLM, and TGI**."

**The 4090-specific methodological admission, VERBATIM — this is the sentence that matters most for our project:**
> "As an acknowledged bug [RTX 40x0 NCCL Issue Comment, linking `NVIDIA/nccl-tests/issues/117`] exists for the **RTX40X0 GPU series**, to rectify this issue and ensure that the inference framework functions appropriately on the RTX4090, the configuration **`NCCL_P2P_DISABLE=1`** was applied. **However, this configuration might impact the final performance, putting RTX4090 at a disadvantage against other platforms.**"

And the consequences they observe, VERBATIM:
> "The performance results on the **RTX4090 platform are different from the other two platforms. This discrepancy might be due to the `NCCL_P2P_DISABLE=1` setting.**"
> "the **RTX3090 GPU platform demonstrates a lower latency than the RTX4090** in the majority of experiments, a situation that might also result from the `NCCL_P2P_DISABLE=1` setting."
> "Specifically on the RTX4090 platform, the inference time difference between Llama2-7B and Llama2-70B can reach up to **13 times, from 120 seconds to 1600 seconds**. However, this phenomenon is not observed on the A800 GPU platform"
> "On the A800 platform, LightLLM exhibits superior throughput. In converse, **on the 24G GPU platform, TGI demonstrates enhanced throughput**, whereas the vLLM and LightLLM display comparable levels of throughput."
> "This finding shows that LightLLM is specifically optimized for high-performance GPUs such as the A800/A100 series."
> (training scaling) "A800 has almost linear scaling, whereas RTX4090 and RTX3090 have slightly low scaling efficiency (**90.8% and 85.9%** respectively). RTX4090 achieves **4.9% higher** scaling efficiency than RTX3090. In the RTX3090 platform, **NVLink connection helps improve the scaling efficiency by 10% over without NVLink.**"

**Bearing on our box — very high.** *This is a peer-reviewed paper with an explicit 8× RTX 4090 24 GB / PCIe 4.0 x16 testbed, running inference-serving frameworks, that had to disable P2P on the 4090 and then could not attribute its own cross-platform numbers cleanly because of it. It is simultaneously (i) prior art that the 8×4090 box is used for serving evaluation and (ii) a published statement that the community lacks a clean measurement because the 4090's P2P is broken. Note: the box is **Xeon Gold 6230 ×2** — so it is also a 2-socket NUMA box — but the paper does not do NUMA analysis. Also note A800 is Ampere, not Hopper, so this is not a cross-generation Ada→Hopper→Blackwell study.*

#### B.4.3 RoundPipe — **verified**: arXiv **2604.27085**, *"Efficient Training on Multiple Consumer GPUs with RoundPipe"*, submitted 29 Apr 2026.
<https://arxiv.org/abs/2604.27085> · full text <https://arxiv.org/html/2604.27085>

VERBATIM:
> "Evaluations on an **8× RTX 4090 server** demonstrate that RoundPipe achieves **1.48–2.16× speedups** over state-of-the-art baselines when fine-tuning 1.7B to 32B models. Remarkably, RoundPipe enables LoRA fine-tuning of the **Qwen3-235B** model with **31K sequence length** on a single server."
> "**4090 server: 8× NVIDIA RTX 4090 GPUs (24 GB VRAM each), Intel Xeon Gold 6330 CPU, 800 GB available DDR4 host memory, PCIe 4.0 (32 GB/s) interconnect.**"
> "**A800 server: 8× NVIDIA A800 SXM GPUs (80 GB HBM2e each), Intel Xeon Platinum 8352Y CPU, 800 GB available DDR4 host memory, NVLink 3.0 (200 GB/s) interconnect.**"
> "Consumer-grade GPUs use PCIe interconnects, offering **less than 20% of NVLink bandwidth**. This physical limitation is further compounded by **root complex contention in PCIe topologies**."
> "(2) Slow inter-GPU communication: … Prior studies indicate that these communications can consume **up to 70% of the training time**."
> "its **4090 throughput reaches at least 76% of existing A800 solutions across all models**, effectively bridging the performance gap between consumer and data-center hardware."
> "**RoundPipe achieves highly competitive performance without utilizing any GPU peer-to-peer communication (NVLink), relying entirely on PCIe host-to-device transfers** under the Computation Dispatch Paradigm."
> "Excluding **Megatron-TP, whose throughput becomes impractical under PCIe**, RoundPipe extends the maximum sequence length by 4.7~7.3× over the next-best baseline."
> (footnote 4) "N/A on Megatron-TP when LoRA finetuning Qwen3-235B because the model has **four key value heads, which does not support TP=8** on Megatron."
> (scalability) "RoundPipe achieves near-linear throughput scaling from 1 to 8 GPUs across all model sizes."

**Bearing on our box — highest of anything I found for the "8× 4090 PCIe-only" hardware string, with one large caveat: it is a TRAINING/fine-tuning paper.** It establishes (a) the exact box exists in the literature, (b) TP is "impractical under PCIe" on it, (c) the answer chosen by SOTA is to abandon GPU P2P entirely and use host-staged transfers, and (d) a controlled **4090-vs-A800** comparison on the same 8-GPU shape. It says nothing about inference-serving TP-vs-PP, nothing about NUMA placement, and it explicitly is not a generation study (A800 = Ampere).

#### B.4.4 LLMQ — **verified**: arXiv **2512.15306**, *"LLMQ: Efficient Lower-Precision Pretraining for Consumer GPUs"*.
<https://arxiv.org/abs/2512.15306>

VERBATIM: *"These devices are characterized by **low memory availability and slow communication** compared to datacentre-grade GPUs. … LLMQ is able to train or fine-tune a 7B model on a single 16GB mid-range gaming card, or a **32B model on a workstation equipped with 4 RTX 4090s**. … The efficiency of LLMQ rivals that of production-scale systems on much more expensive cloud-grade GPUs."*

**Bearing:** *Training again; 4×4090; no interconnect or NUMA analysis. Confirms the "consumer GPU = slow communication" framing is now standard paper boilerplate.*

#### B.4.5 PowerInfer — **verified via ar5iv**: arXiv **2312.12456**, *"PowerInfer: Fast Large Language Model Serving with a Consumer-grade GPU"*, SOSP '24 (DOI 10.1145/3694715.3695964).
<https://ar5iv.labs.arxiv.org/html/2312.12456v2>

VERBATIM:
> "The evaluation shows that PowerInfer significantly outperforms llama.cpp by up to **11.69×** while retaining model accuracy across various LLMs (including OPT-175B) on **a single NVIDIA RTX 4090 GPU**."
> "However, this method is hindered by the **slow PCIe interconnect** and the CPUs' limited computational capabilities, resulting in high inference latency." … "thereby **minimizing the need for costly PCIe data transfers**."

**Bearing:** *Single-GPU consumer-serving prior art (the canonical one), and it names the PCIe penalty — but its answer is to **avoid** PCIe traffic by keeping hot neurons on-GPU, not to reason about multi-GPU topology. No multi-GPU testbed.*

#### B.4.6 MoLink — peer-reviewed, ACL Anthology: *"Distributed LLM Serving on Consumer-Grade GPUs by Reconciling Computation and Communication"*, Findings of EMNLP 2025, pp. 17633–17642, DOI 10.18653/v1/2025.findings-emnlp.957.
<https://aclanthology.org/2025.findings-emnlp.957/> · PDF <https://aclanthology.org/2025.findings-emnlp.957.pdf>

VERBATIM (abstract): *"Large language models are reshaping internet services. Serving these models is often costly, as it requires multiple high-end GPUs. **Consumer-grade GPUs offer cheaper computational power, providing an opportunity for more cost-efficient LLM serving.** … communication efficiency has emerged as a challenge due to the imbalance in data transfer volumes between the two phases of inference: prefill and decode. **Prefill requests can involve transmitting up to 1000 times more data than decode requests**, leading to decode requests being delayed."*

VERBATIM (evaluation): *"the distributed cluster setup has clusters that contain **3 servers, each server is equipped with a RTX 4090 GPU**. Inter-node communication has an average bandwidth of **100 Mbps** and an average latency of **30 ms**. These servers are configured to use **pipeline parallelism** for LLM serving."*

Also: *"For instance, an **RTX 4090 delivers 330 TFLOPS** … of an A100, with **over 4× lower hourly pricing**"*.

**Bearing on our box:** *Peer-reviewed prior art for "distributed LLM serving on consumer-grade GPUs" using **RTX 4090**, and it chooses **PP**, not TP. But the topology is 3 single-GPU servers over a 100 Mbps WAN-like link — the *opposite* extreme from an 8-GPU intra-node PCIe box. Its contribution (prefill/decode transmission scheduling) is orthogonal to intra-node all-reduce.*

#### B.4.7 Papers I could NOT verify
- **"Exploring the Feasibility and Performance of Distributed LLM Serving on Consumer-Grade GPUs"** (IEEE Xplore doc 11618998; Semantic Scholar corpus ID `de73f530166648e4fc47d1e8a58c23b9a3ba06bd`). **Title confirmed only from search-result listings.** `ieeexplore.ieee.org` returned **HTTP 202 with a 0-byte body**; `semanticscholar.org` returned **HTTP 202 / 0 bytes**; `api.openalex.org` **timed out**. **UNVERIFIED — I did not read its abstract or testbed.** It may be the journal version of MoLink (same "Jin, Liu" author lead) or a different paper; I could not establish which.

### B.5 **Cross-generation controlled hardware comparison for LLM inference** — the key question

Sub-questions: does a paper exist that measures the *same* algorithm on Ada vs Hopper vs Blackwell, treating generation as a controlled variable?

> **⚠️ CORRECTED VERDICT.** My first-pass conclusion here was **"I did not find it."** A dedicated parallel sub-search then found verified counterexamples, and **I independently re-verified its three load-bearing IDs myself** (fetched `arxiv.org/abs/<id>`, read the title back, and matched the abstract text). The corrected verdict is: **controlled cross-generation comparisons for LLM inference DO exist, but none spans Ada → Hopper → Blackwell on the multi-GPU *serving-system* axis.** The original statement was too strong and is retracted.

**Verified counterexamples (I re-verified each of these three myself):**

| Paper | Generation span | Controlled? | Inference? |
|---|---|---|---|
| **SlideSparse 2603.05232** (5 Mar 2026) | **Ada sm89 + Hopper sm90 + Blackwell sm100/sm120 in one system**: *"A100, H100, B200, RTX 4090, RTX 5080, DGX-spark"* | ✅ integrated into vLLM | ⚠️ sparse/quantized **GEMM kernels**, not serving |
| **Watt Counts 2604.09048** (10 Apr 2026) | **5 architectures, 10 GPUs**: V100, T4, A100/A30/RTX3090, **RTX 4090**/L40S/L4, H100 NVL/H200 NVL. *"over 5,000 experiments for 50 LLMs across 10 NVIDIA GPUs in batch and server scenarios"* | ✅ one benchmark, one dataset | ✅ inference (energy + latency), **no Blackwell** |
| **2605.30571** (28 May 2026) | A100-80GB SXM4, **H100 SXM5**, **L40S, L4** — *"44 valid cells under a controlled bf16 SDPA setup"* | ✅ same models, same harness | ✅ batch-1 decode — **and it is a (b)-type finding** |

**VERBATIM from 2605.30571** — this is the clearest published statement that an inference measurement does *not* transfer across generations:
> "We measure batch-1 decode for three 7 to 8B-class GQA transformers across four NVIDIA GPUs: **H100 SXM5, A100-80GB SXM4, L40S and L4**."
> "**The achieved fraction of peak HBM bandwidth falls as peak bandwidth rises.** On the headline Qwen-2.5-7B ctx=2048 cell, **an L4 reaches roughly 81 percent of its analytic memory floor, while an H100 reaches only 27 percent.**"
> "On H100 at ctx=2048, CUDA Graphs improves decode latency by **1.259x** … On L4, the same intervention gives only **1.028x**. This isolates a launch-side overhead that **becomes visible on fast GPUs but remains mostly hidden on slower, bandwidth-bound GPUs**."

**VERBATIM from Watt Counts 2604.09048** (another (b)-type finding):
> "the **older-generation A100 achieves 10% energy savings over the newer H100**, suggesting that **newer GPU generations do not always translate to better energy efficiency** across all deployment scenarios."
> "**optimal hardware choices vary significantly across models and deployment scenarios**, demonstrating the critical importance of hardware-aware deployment"

**VERBATIM from SlideSparse 2603.05232** (its abstract names the generation set directly):
> "Integrated into vLLM, SlideSparse is evaluated across various GPUs (**A100, H100, B200, RTX 4090, RTX 5080, DGX-spark**), precisions (FP4, INT8, FP8, BF16, FP16), and model families (Llama, Qwen, BitNet)."

**Other verified items from the same sub-search (not re-verified by me — treat as single-source):**

| Paper | Span | Note |
|---|---|---|
| **2504.11750** (16 Apr 2025) | PCIe A100 / PCIe H100 / GH200 | *"loosely-coupled (PCIe A100/H100) and closely-coupled (GH200) systems"* — but **generation is confounded with CPU** (EPYC vs Xeon vs Grace); the real axis is coupling |
| **APEX4 2606.08761** (7 Jun 2026) | Ampere + Ada | *"the W4A4-g128 kernel yields 2.0–2.5× speedup on RTX 3090 (ρ=16) yet degrades to 0.43–0.47× on A100 (ρ=64) … establishing **W4A4 viability as platform-dependent rather than universally infeasible**"* — a (b)-type finding |
| **2402.13499** (21 Feb 2024) | Hopper + Ada + Ampere | *"conventional latency and throughput comparison benchmarks across the three most recent GPU architectures"* — microbenchmarks, not LLM inference |
| **FlashAttention-4 2603.05451** (5 Mar 2026) | Hopper + Blackwell | *"asymmetric hardware scaling: tensor core throughput doubles while other functional units … scale more slowly or remain unchanged"* |
| **2601.22076** (29 Jan 2026) | H100 + B200 | *"1,858 different configurations on NVIDIA H100 and B200 GPUs"* |
| **2607.02391** (2 Jul 2026) | 8 server-grade GPUs | *"leave-one-GPU-out and leave-one-LLM-out cross-validation"* — a **predictor** for unseen GPUs |
| **"Microarchitecture Is Destiny"** | Pascal, Turing, Ampere, Ada | ⚠️ **UNVERIFIED** — no arXiv ID found; OpenReview forum `SzQGRR65c0` returned **HTTP 403 challenge on 5 retries**. Reviewer text (quoted by the sub-search) claims *"once VRAM capacity suffices to load a model, inference throughput is primarily determined by Tensor Core generation and memory bandwidth"*. **Do not cite without opening it.** |

**CORRECTED VERDICT on (ii):** controlled cross-generation LLM-inference comparison **is published** — Ada↔Hopper is well covered (2605.30571, 2604.09048, 2504.11750, 2402.13499), Hopper↔Blackwell is covered (2603.05451, 2601.22076), and **SlideSparse (2603.05232) puts Ada + Hopper + Blackwell in one controlled vLLM system but only for sparse GEMM kernels**. What remains **not found** is a cross-generation study on the axis that matters here: **multi-GPU serving topology — TP/PP degree, PCIe-only vs NVLink, NUMA placement — with generation as the variable.** Also note the three (b)-type findings above (2605.30571, 2604.09048, APEX4) are *precedent that generation-dependence is a publishable finding*, which strengthens rather than weakens a "does this transfer?" framing.

*(Caveat from the sub-search: its second 15-query batch was cancelled mid-run, so this axis is **partially** searched — treat as incomplete rather than exhaustive.)*

### B.6 NUMA-aware / topology-aware placement for multi-GPU inference

#### B.6.1 Albireo — **verified**: arXiv **2606.01927**, *"Scaling LLM Inference Beyond Amdahl's Limits via Eliminating Non-Scalable Overheads"*, submitted 1 Jun 2026.
<https://arxiv.org/abs/2606.01927> · full text <https://arxiv.org/html/2606.01927v1>

**⚠️ The task brief described 2606.01927 as "Albireo … which compares NVLink vs PCIe A100." The ID is correct and the paper *is* Albireo; the comparison is real, but it is a *sub-part* of the paper, not its thesis.** VERBATIM:
> "We evaluate Albireo on three testbeds: **H100^N, A100^N and A100^P**. Each has 8 GPUs (80 GB memory): **H100^N uses NVLink-connected H100 GPUs, A100^N uses NVLink-connected A100 GPUs, and A100^P uses PCIe-connected A100 GPUs.** All testbeds feature 2 TB RAM, an Intel Xeon Platinum 8468 CPU with 192 logical cores."
> "Tensor parallelism (TP) is necessary to fit modern models but scales sub-linearly as the TP degree *t* grows, due to cross-GPU communication and non-scalable runtime work, as predicted by Amdahl's Law. **Conversely, increasing *t* improves memory efficiency and alleviates KV-cache contention and swapping. We identify and validate an empirical optimal TP degree t_e that balances these effects.**"
> "the **KV cache grows with sequence length and batch size and can exceed the weight footprint**. Given that even high-end GPUs (e.g., H100) provide at most 80 GB per device, fitting 32B or larger models on a single GPU is challenging."
> "increasing *t* expands an instance's effective memory budget, especially for the KV cache, which **mitigates request preemption and GPU–CPU KV swapping**. The tension between these effects yields an **empirical optimum t_e (per model/workload/hardware)**."

**Bearing on our box — high, and it is the closest thing to a formalization of the C7 coupling.** Albireo names the two opposing forces (TP communication cost ↑ vs KV-capacity benefit ↑) and defines an optimal TP degree as their balance — i.e. it *does* treat "how much TP" as an optimization driven partly by KV capacity. What it does **not** do: it does not name "KV-cache-forced sharding on small-VRAM cards", does not treat high context length as the *forcing* variable, and its testbeds are 80 GB cards where KV pressure is far weaker than on 24 GB. Its PCIe arm is A100-PCIe (80 GB), not consumer 24 GB.

#### B.6.2 Mobius — ASPLOS '23, *"Mobius: Fine Tuning Large-Scale Models on Commodity GPU Servers"*, DOI 10.1145/3575693.3575703. PDF: <http://storage.cs.tsinghua.edu.cn/papers/asplos23-Mobius.pdf>

**This is the most explicit PCIe-topology-aware placement prior art I found.** VERBATIM:
> "we find that the **naive sequential mapping scheme of existing pipeline systems is not PCIe topology-aware, thus can cause severe communication contention on the CPU root complexes of commodity GPU servers**. To alleviate this contention, our key idea is to **prevent GPUs under the same root complex from transferring data simultaneously**, where possible. With this key idea, Mobius tries the best to **map two adjacent stages to two GPUs under different CPU root complexes; we call this cross mapping scheme.** … **Mobius automatically searches for the best one by estimating the communication contention degree based on the PCIe topology of the GPU server.**"
> "Since commodity GPUs **lack GPUDirect peer to peer (GPUDirect P2P) support, inter-GPU communication is first routed through CPU to DRAM and then transferred to the target GPU.** Thus, when multiple GPUs transfer data simultaneously, there is **serious bandwidth contention at CPU's root complexes.**"
> "traditional **pipeline parallelism is more suited for commodity GPUs** than the existing system's ZeRO data parallelism, since it only transfers small activations and activation gradients between adjacent GPUs, bringing remarkably fewer communications."
> Table 1: 3090-Ti vs A100 — **GPUDirect P2P: not support / support**; commodity "can only communicate with other GPUs using **PCIe-3.0 with a bandwidth of 16 GB/s**" vs NVLink "up to 900 GB/s"; "about **70% of training time** using DeepSpeed is spent on communication in our evaluation."

**Bearing on our box — high for placement, moderate for task.** Mobius is *training*, on 3090-Ti (Ampere consumer), PCIe 3.0. But it is a **peer-reviewed, PCIe-root-complex-aware stage↔GPU placement algorithm** whose entire motivation is exactly our hardware axis (consumer cards, no P2P, root-complex contention). It is direct prior art for "topology-aware placement on a consumer PCIe box".

#### B.6.3 TCCL — *"TCCL: Discovering Better Communication Paths for PCIe GPU Clusters"*, ASPLOS '24, DOI 10.1145/3620666.3651362, pp. 999–1015.
<https://dl.acm.org/doi/10.1145/3620666.3651362>
**I did not fetch the abstract** (ACM DL is Cloudflare-blocked; `sciprofiles` mirror returned 420 bytes). **Title/venue/DOI verified from search listings and from RoundPipe's bibliography, where it is cited as:** *"Kim et al. (2024) Heehoon Kim, Junyeol Ryu, and Jaejin Lee. 2024. TCCL: Discovering Better Communication Paths for PCIe GPU Clusters."* **UNVERIFIED beyond title/venue/DOI.**

#### B.6.4 Topology-Aware Data Movement — **verified**: arXiv **2607.28633**, *"Topology-Aware Data Movement for Disaggregated GPU Inference"*, submitted 19 Apr 2026, revised 7 Aug 2026.
<https://arxiv.org/abs/2607.28633>

VERBATIM:
> "Yet DistServe, Splitwise, and Mooncake all use uniform RDMA, ignoring that **bandwidth between two GPUs varies by 72× depending on their physical relationship: 900 GB/s via NVLink 4.0 within a domain (1.8 TB/s on NVLink 5, widening the gap to 144×), 50 GB/s via InfiniBand across nodes, 12.5 GB/s via TCP across data centers.**"
> "**Full evaluation requires multi-node clusters with heterogeneous interconnects and CXL 3.0 hardware that is beyond academic resources**"

**Bearing on our box — low-to-moderate.** The title sounds exactly on-axis, but the content is **datacenter disaggregated prefill/decode** (NVLink/IB/TCP/CXL), the "72×" span is NVLink↔TCP, and by the authors' own admission the **full evaluation was not performed**. It does establish the "topology-aware transport selection" idea is published. It says nothing about consumer Ada, 24 GB, or NUMA placement.

#### B.6.5 **Predictable LLM Serving on GPU Clusters** — verified: arXiv **2508.20274**, submitted 27 Aug 2025.
<https://arxiv.org/abs/2508.20274>

**This is the most direct "PCIe-aware placement for LLM serving" prior art I verified.** VERBATIM:
> "**Latency-sensitive inference on shared A100 clusters often suffers noisy-neighbor interference on the PCIe fabric, inflating tail latency and SLO violations.** We present a fabric-agnostic, VM-deployable host-level controller that combines dynamic Multi-Instance GPU (MIG) reconfiguration, **PCIe-aware placement**, and lightweight guardrails (MPS quotas, cgroup I/O). It samples per-tenant tails and system signals, **uses topology hints to avoid PCIe hot spots**, and gates actions with dwell/cool-down to avoid thrash. On a single host and a **2-node (16-GPU) cluster**, SLO miss-rate is reduced by **≈32%** (≈1.5) and **p99 latency improves ≈15% with ≤5% throughput cost** versus static MIG and naive placement; ablations show MIG and placement contribute comparably. We also evaluate LLM serving with **vLLM on OLMo 2 7B Instruct: TTFT p99 improves ≈10–15% at ≤5% cost**."

**Bearing on our box — high for the placement axis, medium for hardware.** It is *LLM serving*, it is *placement on a PCIe fabric driven by topology hints*, and it is *measured* (vLLM, TTFT p99). It is A100 + MIG + VM-deployable, i.e. datacenter-professional, not 24 GB consumer — but it removes "nobody has done topology-aware placement for LLM serving" as a novel claim.

#### B.6.6 `local-inference-lab/rtx6kpro` — `hardware/topology.md` and `optimization/pcie-oneshot-allreduce.md` (community field wiki, **RTX PRO 6000 Blackwell / SM120, dual-socket EPYC, PCIe-only**)
<https://github.com/local-inference-lab/rtx6kpro> · topology: <https://raw.githubusercontent.com/local-inference-lab/rtx6kpro/b5e5c38105de40d5eb6c472c4cf629ae5b5c1921/hardware/topology.md> · allreduce: <https://raw.githubusercontent.com/local-inference-lab/rtx6kpro/61842c4f5c3489dc4edaba7639a2f0656b587f9e/optimization/pcie-oneshot-allreduce.md>

**This is the only source I found that measures the 8-GPU, cross-socket, PCIe-only, no-NVLink regime directly.** VERBATIM:

> "Same-NUMA P2P: **~0.36 µs latency, ~56 GB/s unidirectional** · Cross-NUMA P2P: **~0.44 µs latency (Turin) / ~14 µs (Genoa w/o P2P)**"
> "**Cross-NUMA P2P Latency: ~14 µs (P2P disabled) | ~0.44 µs (P2P enabled)**" (Genoa vs Turin table)
> "xGMI Bandwidth: ~96 GB/s (Genoa) | **192–256 GB/s actual** (Turin)"
> "**Tensor-parallel inference performs AllReduce operations after every attention and MoE layer.** For models like Qwen3.5-397B or Kimi K2.5, these are small messages (32–256 KB) where **latency dominates over bandwidth**."

Measured **8-GPU AllReduce bus bandwidth on PCIe-only boxes** (VERBATIM table):

| Setup | Bus BW (GB/s) | Notes |
|---|---|---|
| 3× switches, single CPU | **41.1** | Highest measured |
| 4× switches, single CPU | 39.4 | With `NCCL_MIN_NCHANNELS=8` |
| dual Turin, no switches | **37.6** | **after NCCL tuning** |
| dual Turin, no switches | **22.2** | **default NCCL** |

> "**Expert Parallelism (EP)** was tested and found to be consistently slower or equivalent on PCIe setups … 'EP is dead in the water for these setups' — luke … Without NVLink (~900 GB/s), PCIe (~56 GB/s unidirectional) cannot keep up."
> (on the oneshot kernel) "**~7% faster decode throughput on 4 GPU (same NUMA)** but **does not help on 8 GPU cross-socket** where Infinity Fabric latency makes the system-scope barrier too expensive. … For 8 GPU cross-socket, stick with `--disable-custom-all-reduce` (NCCL is faster)."
> "**Why 8 GPU is slower:** On dual-socket AMD EPYC, GPUs 0-3 and 4-7 are on separate NUMA nodes connected via Infinity Fabric. The PCIe oneshot kernel's system-scope barrier (`__threadfence_system`) must wait for visibility across the Infinity Fabric link, which adds significant latency. **NCCL's ring/tree algorithms pipeline data and avoid global barriers**, making them faster for cross-socket communication. With MTP, the speculative tokens increase the effective batch size (and thus allreduce payload), pushing more messages above the 48KB crossover where NCCL wins — compounding the penalty."

Their end-to-end 8-GPU cross-socket degradation with a barrier-based collective (GLM-5 TP=8, b12x, MTP): **+7.2% at ctx=0 (no MTP) but −4.3% at ctx=0 (MTP), −5.2% at 16K, −15.7% at 32K**. Auto-crossover threshold collapses **120 KB (4 GPU, same NUMA) → 48 KB (8 GPU, cross-NUMA)**.

**Bearing on our box — very high on the topology axis, and the single most useful thing I found for the "2 NUMA nodes" half of the problem.** It is the only measured evidence I have that **the same collective optimization that wins on 4 same-NUMA GPUs loses on 8 cross-socket GPUs, and that the loss grows with context length**. Two caveats: (i) hardware is **Blackwell RTX PRO 6000**, not Ada 4090; (ii) it is a community field wiki with named contributors, not peer-reviewed. But note the *mechanism* it identifies is CPU-fabric, not GPU-generation: **the inter-socket barrier cost is a property of the host, not the GPU** — which is exactly why it should transfer to an 8×4090 2-NUMA box, and why it is worth measuring there.
#### B.6.7 A NUMA-Aware Compiler Framework for … PCIe-Based Multi-Accelerator Systems — NeurIPS 2025 workshop (MATH-AI), JooHyoung Cha & Yongin Kwon.
<https://nips.cc/virtual/2025/loc/san-diego/131087> (OpenReview PDF also surfaced: <https://openreview.net/pdf?id=gPqxViA1fe>)

VERBATIM (abstract, fetched from nips.cc):
> "Mathematical reasoning workloads … induce frequent collectives under model/tensor parallelism. **In commodity dual socket NUMA servers with PCIe interconnects, non uniform link bandwidth/latency and host mediated cross socket routes make these collectives the bottleneck**, inflating end to end latency … We present a **NUMA aware compiler framework** for large scale math reasoning inference. The system **profiles compute and memory paths, learns a latency bandwidth cost model for hierarchical collectives, and jointly optimizes data, model, and tensor partitioning along with device memory placement** under static feasibility constraints. Using MLIR/TOSA templates, it emits host and accelerator code with explicit comm–compute overlap and schedule shaping via **ring, tree, and hybrid schemes, without relying on vendor specific fabrics.**"

**Bearing on our box — the closest title-level match to "2 NUMA nodes, PCIe-only, tensor-parallel inference."** It explicitly targets *commodity dual-socket NUMA + PCIe*, explicitly names *host-mediated cross-socket routes* as the bottleneck, and jointly optimizes *tensor partitioning + device memory placement*. Two caveats: (i) it is a **workshop poster**, and the abstract says *"We outline ablations to isolate how … affect throughput and p95 latency"* — i.e. the evaluation is **planned, not reported**; (ii) it does not name specific GPUs or consumer Ada. **It is the strongest NUMA-axis prior art I verified, and its evaluation appears to be prospective.**

---

## Section C — Memory-capacity-forced sharding on 24 GB

*(This section draws on both my own searches and the parallel sub-search; items are marked with the surface they were fetched from.)*

### C.7 Is the "KV cache forces TP, TP costs PCIe bandwidth" coupling named?

**Both halves of the coupling are well published, and — as of the later part of this sweep — so is the coupling itself, at 80 GB-class scale.** See the ⚠️ IMPORTANT UPDATE at the end of this subsection before claiming C7 as a gap.

**Half 1 — TP expands KV capacity (published, multiple venues):**
- **Albireo 2606.01927** (VERBATIM): *"Tensor parallelism (TP) is a widely used technique to **expand KV cache capacity** in LLM inference. It distributes model weights across multiple GPUs … This eliminates redundant weight loading and improves memory efficiency, enabling the model to support more concurrent requests without exceeding the memory capacity."* and *"increasing t expands an instance's effective memory budget, especially for the KV cache, which **mitigates request preemption and GPU–CPU KV swapping**."*
- **2311.03687**: *"Since A800 has 80GB memory, while RTX4090 and RTX3090 have only 24GB each, so some cases … cannot run on RTX4090 and RTX3090 GPUs"* and *"Even on hardware-rich servers … For smaller models (1.7B and 8B), Data Parallelism (DP) performs best by fully exploiting the high-bandwidth NVLink"* (RoundPipe, on why DP wins when memory is ample).
- **RoundPipe 2604.27085**, on the **KV-head constraint on TP degree** — a mechanistically important, non-obvious coupling (VERBATIM, footnote 4): *"N/A on Megatron-TP when LoRA finetuning Qwen3-235B because the model has **four key value heads, which does not support TP=8**."* The same constraint appears in club-3090 (VERBATIM): *"Because KV heads replicate above TP=4, going 4 → 8 cards does not shrink per-card KV — it only shards weights. Budget accordingly."*
- **club-3090** ships the artefact of exactly this (MULTI_CARD.md table): KV pool capacity TP=1 → smallest, TP=2 → 2×, TP=4 → ~4×, TP=8 → ~8× (derived).
- The **cnblogs 4×4090** measurement is the cleanest empirical instance I found: raising TP from 2 to 4 raised the KV pool from **469K → 1,629K tokens** (3.47×) on the same box, while single-stream decode went **79 → 108 tok/s** and aggregate saturation **~1,560 → ~2,274 tok/s** — i.e. **the KV-capacity win and the interconnect cost are visibly traded against each other in one experiment.**

**Half 2 — the interconnect cost of higher TP on PCIe (published):**
- **Albireo** formalizes it as Amdahl's law and defines the **empirical optimal TP degree `t_e`** (VERBATIM): *"Tensor parallelism (TP) is necessary to fit modern models but scales sub-linearly as the TP degree t grows, due to cross-GPU communication and non-scalable runtime work, as predicted by Amdahl's Law. **Conversely, increasing t improves memory efficiency and alleviates KV-cache contention and swapping. We identify and validate an empirical optimal TP degree t_e that balances these effects.**"* — this is the closest published statement of the coupling I verified, and it is a *balance*, not a "forced" relationship.
- **2311.03687** observes the 4090's scaling efficiency and latency ordering flipped by the disabled P2P.
- **RoundPipe**: *"Megatron-TP, whose throughput becomes impractical under PCIe."*
- **BNTU 2026**: *"Тензорный параллелизм в условиях ограниченной пропускной способности шины PCIe демонстрирует критическое падение эффективности из-за высоких задержек синхронизации"* ("Tensor parallelism under limited PCIe bus bandwidth shows a critical drop in efficiency due to high synchronization latencies").
- **Mobius**: *"traditional pipeline parallelism is more suited for commodity GPUs."*

**⚠️ IMPORTANT UPDATE — the coupling IS substantially formalized in two places found later in the sweep. The "nobody has framed this" version of the claim is NOT safe.**

**(A) "More GPUs or a Smaller Cache? Tensor Parallelism versus KV Compression for Memory-Bound LLM Serving" — verified: arXiv 2608.23962, submitted 25 Aug 2026.**
<https://arxiv.org/abs/2608.23962>

This paper exists *specifically* to put the two escape hatches from KV-cache exhaustion on one axis. VERBATIM:
> "**When an LLM serving deployment runs out of KV cache room, there are two well-established ways out. Tensor parallelism shards the weights and the KV cache across two, four, or eight devices, buying memory headroom at the price of an all-reduce on every layer and a hardware bill that grows with the device count.** The algorithms community shrinks the cache in place, with KV quantisation and eviction keeping a single GPU and spending a little quality instead. Compression papers report memory ratios, parallel-scaling papers report throughput curves, and **almost nobody puts the two on the same cost axis.**"
> "We place tensor-parallel configurations (**degree 1 to 8**) and KV-compressed configurations (16/8/4-bit, keep-ratios down to 0.25) on one cost-normalised axis, cost per million tokens against latency, using a **profiled simulator calibrated on A100, A40, and H100 hardware**, and we go looking for the cost-equivalence crossover."
> "**the boundary that decides between the strategies is model size relative to device memory, at roughly 36B parameters for an 80 GB card. Below that wall, compression dominates and extra GPUs are largely wasted spend; above it, tensor parallelism stops being a choice and becomes an entry ticket: Llama-2-70B is infeasible on one A100 at any KV setting, because the binding resource is weights, which KV compression does not touch.**"
> "Tensor parallelism is the only lever that improves latency … while compression is the only lever that multiplies capacity per dollar (**16.5×, against 1.21× for an eightfold spend on GPUs**)."

**Bearing:** *This is the closest prior art to the C7 claim, and it is close in framing ("run out of KV cache room" → TP vs compress; "TP … buying memory headroom at the price of an all-reduce on every layer"). Two gaps remain, and they are exactly our hardware axis: (i) it is calibrated on **A100/A40/H100 (40–80 GB)**, never a 24 GB consumer card; (ii) its crossover is "model size relative to device memory," and it places that wall at **~36B params on 80 GB** — on a 24 GB card that wall would move down roughly with the memory ratio (my arithmetic, not their claim: ≈11B), i.e. TP becomes an "entry ticket" at a far smaller model and the "compression dominates" regime shrinks drastically. It also uses a **profiled simulator**, not real multi-GPU runs. So: the **tradeoff is published; its behaviour in the 24 GB / PCIe-only regime is not.***

**(B) The KV-head-caused TP ceiling is named, in a paper and in vLLM's own tracker:**
- **Helix Parallelism — verified: arXiv 2507.07120**, *"Helix Parallelism: Rethinking Sharding Strategies for Interactive Multi-Million-Token LLM Decoding"*, NVIDIA, 7 Jul 2025. <https://arxiv.org/abs/2507.07120>. VERBATIM: *"While Tensor Parallelism (TP) helps mitigate the cost of FFN weight reads, **it does not scale well for attention. When TP width exceeds the number of KV heads, it leads to inefficient KV duplication, limits parallelism, and constrains batch size.** Simultaneously, DRAM reads for long KV histories scale linearly with batch size, further capping efficiency. We introduce Helix Parallelism, a hybrid execution strategy that **applies KV parallelism during attention to shard KV caches across GPUs, then reuses the same GPUs for TP** … **pushing forward the throughput-latency Pareto on Blackwell**."* — **Bearing:** Helix is the named, published statement that once KV heads run out you must change the sharding *shape*, not raise TP. But it is **Blackwell**-framed and motivated by TTL at multi-million-token contexts, not 24 GB capacity.
- **vLLM RFC #34018, "[RFC]: Helix (Context + Tensor) Parallelism for Efficient Long-Context Decoding"** — <https://github.com/vllm-project/vllm/issues/34018> (title and body fetched from GitHub HTML). VERBATIM: *"Helix enables efficient long-context decoding by combining Context Parallelism (sequence sharding) with Tensor Parallelism (head sharding), **eliminating KV cache duplication that occurs in traditional Tensor Parallelism when `TP > num_kv_heads`**."* and *"When `TP > num_kv_heads`, traditional TP must **duplicate** the KV cache across GPUs, wasting memory bandwidth during decode."* With a table: DeepSeek-V2/R1 MLA 1 effective KV head → **8× KV duplication at TP=8**; Llama-3 70B/405B 8 KV heads → 1× at TP=8; Qwen2.5 4–8 KV heads → 2–8× duplication. — **Bearing:** the coupling is in the *framework's own tracker*: past a certain TP degree, extra cards stop buying KV capacity and start costing communication. Directly applicable to a 4-KV-head model on 8 cards — exactly club-3090's note: *"Because KV heads replicate above TP=4, going 4 → 8 cards does not shrink per-card KV — it only shards weights."*

**Verdict on C.7 (revised):** the *quantity* "optimal TP degree = f(communication cost, KV capacity)" is published and named (`t_e`) by **Albireo**; the *tradeoff* "when you run out of KV room, choose TP vs KV compression, and the crossover is model-size-relative-to-device-memory" is **published by 2608.23962**; the *mechanism* "TP stops buying KV capacity once TP > num_kv_heads, so you must change the sharding shape" is **published by Helix 2507.07120 and live in vLLM RFC #34018**. What I **did not find** is any of this **instantiated at 24 GB / consumer Ada / PCIe-only**: all three items are 80 GB-class or Blackwell, and none treats the **PCIe all-reduce cost** as the rising term (2608.23962 treats GPU count as a *dollar* cost, not a *bandwidth* cost). **That specific framing — "on a 24 GB PCIe-only box, long context forces TP up, TP up costs PCIe all-reduce, so the capacity fix and the bandwidth penalty are the same knob" — is the part I could not find stated.**

### C.7b Additional items found by the parallel sub-search (I re-verified the load-bearing ones myself)

**(A) The chain is stated *with the PCIe term* in one paper — arXiv 2405.08944, "Challenges in Deploying Long-Context Transformers: A Theoretical Peak Performance Analysis" (14 May 2024).** ✅ **I verified this ID myself** (`arxiv.org/abs/2405.08944` → title matches) and grepped the full text at `arxiv.org/html/2405.08944v1`. VERBATIM:
> "all additional computational costs, compared to 4K context, trace back to *one single source: the large size of the KV cache*"
> "(4) **context switching is PCIE bound**: offloading user 1's KV cache to the CPU DDR and loading user 2's KV cache to the HBM **is bounded by the PCIE bandwidth**."
> "We discuss how these four metrics are bounded by the size of HBM, the GPU flops, the HBM bandwidth, and **the PCIE bandwidth** respectively, and how these challenges eventually trace back to **the size of the KV cache**"
> "**Tensor Parallelism utilizes multiple devices for accelerating inference with negligible communication overhead.** Linearly increasing the number of devices to 2, 4, and 8 introduces more HBM space, thus linearly increasing concurrency."

**Bearing — this is the most important C7 item, and it cuts both ways.** It *is* the KV-cache-drives-everything argument with PCIe named as one of the four binding resources. But note the last quote: its analysis **assumes TP's communication is negligible** (its running example is *"a 34B GPT-3.5 level model of 50K context on **A100 NVLink**"*). So the paper gives you the KV→PCIe link *and* demonstrates exactly the NVLink-shaped assumption that a PCIe-only box violates. The gap is therefore sharper than "nobody discussed this": **the canonical long-context cost model assumes away the term that dominates on our hardware.**

**(B) The KV-forced TP coupling is stated in framework documentation — vLLM `docs/serving/context_parallel_deployment.md`** (single-source, from the sub-search; **I did not fetch this doc myself**). VERBATIM per the sub-search: *"If one GPU cannot hold them all, or we want to hold more requests in the KV cache, we can first shard the KV cache along the `H` dimension, that's the plain tensor parallel sharding. It's as simple as adding `-tp <num_gpus>`"*; and *"With larger dcp size, the KV cache duplication is reduced, but the communication overhead increases."* Reported as the **only** framework doc that raises TP *because of KV* — vLLM's `parallelism_scaling.md` is interconnect-driven only, TensorRT-LLM's sharding guide has no context/KV criterion, SGLang's server-arguments docs have no such rule.

**(C) vLLM blog, "Efficient Decode Context Parallelism with vLLM for Long Context Workloads"** (7 Aug 2026), <https://vllm.ai/blog/2026-08-07-decode-context-parallelism>. ✅ **I fetched this URL myself** (HTTP 200). VERBATIM:
> "Agent-trace benchmarks now run from 64K all the way to 1M tokens and their KV caches are correspondingly large. **Under a baseline tensor-parallel (TP) setup, this KV cache is partitioned by attention head, which puts a hard floor on how much it can shrink.**"
> "Under tensor parallelism, the KV cache is partitioned by the attention head. … **Once TP [exceeds the KV-head count]** …"
> "**On systems with high-bandwidth GPU-to-GPU interconnects**, this helps preserve interactive responsiveness while serving many long-context agents at once."

**Bearing:** the second quote is the KV-head ceiling again; the third shows the fix is explicitly scoped to *high-bandwidth* interconnects — i.e. **the vendor's own long-context KV fix presumes NVLink-class fabric.**

**(D) ⚠️ A community measurement on 4× RTX 3090 that CUTS AGAINST the C7 chain — `mtecnic/vllm-topology-bench`** (<https://github.com/mtecnic/vllm-topology-bench>; ✅ README fetched by me). Measured, 4× RTX 3090 24 GB, no NVLink, Qwen3.6-35B-A3B-AWQ: `1× TP=4` saturates ~600 tok/s; `2× TP=2` reaches **1,437 tok/s (+136% at c=128)**; TTFT 1.4 s vs 3.9 s. Crucially the repo **explicitly denies** that KV is the driver here. VERBATIM:
> "**🚫 'One copy per card' is impossible.** `4× TP=1` **OOMs** — the AWQ weights are **~23 GB**, nearly the whole 24 GB card, with no room for KV cache. **TP ≥ 2 is mandatory** for this model on 3090s. *(Lowering `max-model-len` doesn't help — **the weights are the wall, not the KV**.)*"
> "**🔬 Why (it's the interconnect, not the KV cache).** Throughput here is **not** KV-limited — both feasible modes have **44–117× more KV cache** than the ≤128 concurrency tested. The lever is **interconnect**: tensor parallelism does an **all-reduce every single layer**, and with **no NVLink that crosses PCIe**."

**Bearing:** this is the sharpest available counter-evidence to C7 as a *general* claim. In the one controlled 24 GB no-NVLink A/B I have with KV headroom reported, **the KV cache was not the binding resource — the weights were**, and the repo says so in bold. It also flags the untested case: *"Longer contexts would further favor whichever topology has more KV headroom."* **So the honest C7 statement is not "KV forces TP" but "KV forces TP *only once context length pushes KV past the weight footprint*, and I did not find that crossover measured on 24 GB cards."** That crossover condition is precisely what 2608.23962 predicts analytically (its "model size relative to device memory" wall) — and it is exactly what a 24 GB long-context experiment would settle.

**(E) `noonghunna/qwen36-dual-3090` `docs/INTERNALS.md`** (2× 3090, PCIe-only, 262K ctx) — quoted by the sub-search: *"per-stream TPS gain from TP=2 is small (~5%)"*; *"All-reduce on PCIe Gen 4 (~32 GB/s practical) is ~3-5x slower than NVLink"*; *"All-reduce overhead approximately cancels the memory-bandwidth halving."* ⚠️ **I could not verify this: `raw.githubusercontent.com/noonghunna/qwen36-dual-3090/main/docs/INTERNALS.md` returned 404** (wrong path/branch, or private). **Treat as single-source and unverified.**

**(F) No source uses the candidate terms.** The sub-search reports that **no** source uses *"KV cache pressure"* as a defined term, *"KV-budget-driven sharding"*, *"memory-capacity-forced parallelism"*, or *"context-driven TP"*. The nearest named concept remains Albireo's `t_e`.

### C.8 The "24 GB serving regime" and the TP↑ ⇒ PCIe-cost↑ tension

**Verified items:**

| Source | What it establishes about the 24 GB regime |
|---|---|
| **PowerInfer** 2312.12456 (SOSP '24) | The canonical consumer-serving paper; single RTX 4090; names *"the slow PCIe interconnect"* as the obstacle; solution = avoid PCIe transfers. |
| **2311.03687** | Explicitly calls RTX4090/RTX3090 *"the 24G GPU platform"* and finds framework rankings **flip** there: *"on the 24G GPU platform, TGI demonstrates enhanced throughput, whereas the vLLM and LightLLM display comparable levels of throughput"* — and attributes the RTX4090's poor latency to `NCCL_P2P_DISABLE=1`. |
| **RoundPipe** 2604.27085 | *"far larger than 24GB for NVIDIA RTX 4090 or 32GB for NVIDIA RTX 5090"*; the only system able to LoRA-finetune 235B on 24 GB cards. |
| **BNTU 2026** | *"На потребительских GPU с ограниченным объемом VRAM (24 ГБ) это критично"* ("On consumer GPUs with limited VRAM (24 GB) this is critical") — in the context of PagedAttention/KV fragmentation. |
| **Qiita 4×4090** | Per-step overhead ≈17–18 ms, FP8-invariant, *"P2P 非対応の PCIe 4枚構成では、ここは構造的に取り戻せません"* — the ceiling is structural on a 4-card P2P-less PCIe box. |
| **cnblogs 4×4090** | TP=2 vs TP=4 A/B with KV capacity (469K vs 1,629K) and the explicit conclusion *"TP=2 の通信开销显著低于 TP=4，单卡效率反而更高"*. |

**Verdict on C.8:** the 24 GB regime is **well-populated at the single-GPU level** (PowerInfer and successors) and the **consumer-PCIe multi-GPU level is documented by practitioners with numbers**. What I did **not** find is a paper that (i) treats 24 GB-per-GPU as the regime variable in a multi-GPU *serving* study, or (ii) quantifies "TP degree must rise ⇒ PCIe all-reduce cost rises" as a stated tension. The nearest formalization remains **Albireo's `t_e`**, on 80 GB cards.

---

## Section D — Direct prior art verdict

### (i) TP-vs-PP on PCIe-only consumer Ada — **PARTIAL prior art. The specific combination is open.**

| Claim component | Status | Evidence |
|---|---|---|
| "TP is bad and PP is good on PCIe-only" (concept) | **PUBLISHED, peer-reviewed** | **BNTU 2026**, DOI 10.21122/2309-4923-2026-1-54-59 — *"The results unequivocally indicate the **unsuitability of Tensor Parallelism for systems without NVLink** due to critical synchronization delays. It is proven that **Pipeline Parallelism is the only viable strategy for PCIe clusters**"* (2× RTX 3090 vs 1× RTX A6000, vLLM, DeepSeek-R1-Distill-Llama-14B; TPOT > 350 ms for TP). Also **Mobius** ASPLOS '23: *"traditional pipeline parallelism is more suited for commodity GPUs."* |
| The same claim *on Ada / RTX 4090* | **Community-measured, not peer-reviewed** | Qiita 4×4090 (per-step allreduce overhead), cnblogs 4×4090 (TP=4 vs TP=2 A/B), kentino 4×4090 (P2P 19–22 GB/s), club-3090 (3090, not 4090). |
| A **quantified scaling-efficiency curve** for TP vs PP on **8× RTX 4090 PCIe-only** | **⚠️ I did not find it.** | Named queries in **A.6**. The nearest is 2311.03687 (8×4090, serving frameworks, but no TP-vs-PP axis and confounded by `NCCL_P2P_DISABLE=1`). |
| TP-vs-PP **on 8 cards across 2 NUMA nodes** | **⚠️ I did not find it, for any GPU generation.** | Mobius does PCIe-root-complex-aware stage placement but for **training** and not with a NUMA-node axis; the rtx6kpro doc gives the **8-GPU cross-socket** collective penalty but on Blackwell RTX PRO 6000. |
| The **capacity** alternative to TP (KV compression) and its crossover vs TP | **PUBLISHED** | **2608.23962** — TP degree 1–8 vs KV compression on one cost axis; crossover at *"model size relative to device memory, at roughly 36B parameters for an 80 GB card."* A100/A40/H100 only; simulator, not hardware. |

**Bottom line:** the *qualitative* TP-vs-PP-on-PCIe conclusion is prior art (peer-reviewed, on 2× 3090), and the *capacity-vs-TP* tradeoff is prior art (simulated, 80 GB-class). The *quantitative, 8× RTX 4090, 2-NUMA-node, PCIe-only* TP/PP/replicate scaling study appears **not** to be published — and neither does the TP-vs-KV-compression crossover **re-derived for 24 GB cards**, where 2608.23962's own logic implies the "compression dominates" regime collapses.

### (ii) Cross-generation controlled hardware comparison for inference — **⚠️ CORRECTED: it EXISTS, but not on the serving-topology axis.**

> My first pass concluded "I did not find it." A parallel sub-search found counterexamples, **three of which I re-verified myself** (`arxiv.org/abs` + title read-back + abstract match). The corrected position:

- **Controlled cross-generation LLM-inference comparisons ARE published.** Ada↔Hopper is well covered: **2605.30571** (A100/H100/L40S/L4, controlled bf16 SDPA, batch-1 decode), **2604.09048** "Watt Counts" (5 architectures / 10 GPUs / 50 LLMs / >5,000 experiments, including RTX 4090 and H100/H200), **2504.11750** (PCIe A100 / PCIe H100 / GH200), **2402.13499** (Hopper/Ada/Ampere microbenchmarks). Hopper↔Blackwell is covered by **2603.05451** (FlashAttention-4) and **2601.22076** (H100 + B200).
- **All three of Ada + Hopper + Blackwell in one controlled system exists too** — **SlideSparse 2603.05232**, whose abstract itself states: *"Integrated into vLLM, SlideSparse is evaluated across various GPUs (A100, H100, B200, RTX 4090, RTX 5080, DGX-spark)."* But the measured object is **sparse GEMM kernels**, not a serving system, and there is no multi-GPU/PCIe/NUMA axis.
- **Generation-dependence is already a published *finding*, not just a methodology.** 2605.30571 VERBATIM: *"The achieved fraction of peak HBM bandwidth falls as peak bandwidth rises. … an L4 reaches roughly 81 percent of its analytic memory floor, while an H100 reaches only 27 percent."* Watt Counts VERBATIM: *"the older-generation A100 achieves 10% energy savings over the newer H100, suggesting that newer GPU generations do not always translate to better energy efficiency across all deployment scenarios."* APEX4 2606.08761: *"establishing W4A4 viability as platform-dependent rather than universally infeasible."* **So "does this result transfer across generations?" is an established, publishable question.**
- **What is still not found:** a cross-generation comparison on the **multi-GPU serving-topology** axis — TP/PP degree, PCIe-only vs NVLink, NUMA/socket placement — with generation as the controlled variable. The only item combining class + serving is still 2311.03687 (A800/4090/3090), and its own authors flag the 4090 arm as confounded by `NCCL_P2P_DISABLE=1`.
- ⚠️ The sub-search's second query batch was **cancelled mid-run**, so this axis is **partially** searched, not exhaustive.
- **Net effect on strategy:** "nobody has compared GPU generations for LLM inference" is **false and must not be claimed.** "Nobody has compared generations *on the multi-GPU PCIe-only consumer serving topology*, and the 4090's P2P defect is why" is defensible.

### (iii) NUMA/topology-aware inference placement — **PRIOR ART EXISTS. Do not re-map this as novel.**

| Work | Venue | Topology axis | Task | Evaluation status |
|---|---|---|---|---|
| **Mobius** DOI 10.1145/3575693.3575703 | ASPLOS '23 | PCIe **root complex**; "cross mapping" adjacent stages onto GPUs under *different* root complexes; searches for min contention | fine-tuning | **fully evaluated**, vs DeepSpeed, 3.8–5.1× |
| **Predictable LLM Serving on GPU Clusters** 2508.20274 | arXiv, Aug 2025 | **PCIe-aware placement**, "topology hints to avoid PCIe hot spots", on shared A100 clusters | **LLM serving (vLLM, OLMo 2 7B)** | **fully evaluated** (TTFT p99 −10–15% at ≤5% cost; SLO miss-rate −32%) |
| **A NUMA Aware Compiler Framework…** (Cha & Kwon) | NeurIPS 2025 workshop | **dual-socket NUMA, PCIe**, host-mediated cross-socket routes; joint tensor partitioning + device memory placement | math-reasoning inference | **abstract says "we outline ablations" — evaluation appears prospective** |
| **Topology-Aware Data Movement** 2607.28633 | arXiv, Apr 2026 | NVLink-domain-aware placement for MoE; topology-detecting transport orchestrator | disaggregated serving | **full evaluation explicitly not performed** |
| **Albireo** 2606.01927 | arXiv, Jun 2026 | three testbeds incl. PCIe-connected A100 | serving | fully evaluated |
| **TCCL** DOI 10.1145/3620666.3651362 | ASPLOS '24 | discovering better communication paths for PCIe GPU clusters | collectives | **UNVERIFIED beyond title/venue/DOI** |
| **rtx6kpro** topology + PCIe oneshot allreduce | community field wiki | **4-GPU same-NUMA vs 8-GPU cross-socket**; auto-crossover threshold 120 KB → 48 KB; **8-GPU AllReduce bus BW 41.1 / 39.4 / 37.6 (tuned) / 22.2 (default) GB/s**; cross-socket barrier degradation −4.3% → −15.7% as ctx grows 0 → 32K | serving (SGLang, RTX PRO 6000 **Blackwell**), dual-socket EPYC | measured; community, **not peer-reviewed** |

**Bottom line:** "exploit PCIe topology / root-complex / NUMA affinity for multi-GPU model parallelism" is **established prior art (Mobius, ASPLOS '23)**, and "PCIe-topology-aware placement *for LLM serving*" is **also already published and measured (2508.20274, on A100)**. A **NUMA-aware placement/partitioning framework specifically for LLM inference on 2-socket PCIe consumer boxes** is **claimed** by the NeurIPS 2025 workshop abstract but (per that abstract) **not yet evaluated**. The **8-GPU cross-socket collective penalty on a PCIe-only no-NVLink box is measured in a community repo (Blackwell RTX PRO 6000), not in a paper on Ada.**

---

## Section E — What I could NOT verify

### E.1 Surfaces that were blocked or unusable (so the absence claims above are bounded by these)
| Surface | Result |
|---|---|
| `reddit.com`, `old.reddit.com` (HTML **and** `search.json`) | **Network-blocked** — "Blocked … network policy" page; JSON returns HTTP 403 |
| `r.jina.ai` proxy | HTTP 401 — "blocked from performing anonymous queries due to bad network reputation (AS30058)" |
| `ieeexplore.ieee.org/document/11618998` | HTTP **202, 0-byte body** |
| `semanticscholar.org` paper page | HTTP **202, 0-byte body** |
| `dl.acm.org` (ACL/TCCL/agentic-inference pages) | **Cloudflare "Just a moment…" challenge**; no content retrieved |
| `sciprofiles.com` (TCCL mirror) | 420 bytes, no abstract |
| `api.openalex.org` | **timed out (60 s)** |
| `api.github.com` | **rate-limited to zero** (per project guidance; not attempted) |
| `export.arxiv.org/api` | not used (429-prone per project guidance) |
| `raw.githubusercontent.com/.../exllamav2/doc/multi_gpu.md` | **404** |
| `raw.githubusercontent.com/noonghunna/qwen36-dual-3090/main/docs/INTERNALS.md` | **404** (wrong path/branch, or private) |
| `arxiv.org/search`, `arxiv.org/list` | **HTTP 406 hard block** — no arXiv full-text search available |
| `openreview.net` (PDF and `/notes?forum=`) | **HTTP 403 challenge** (5 retries + cookie jar) |
| `semanticscholar.org` API | 429 / 202 |
| `searx.be` (captcha), `mojeek` (403), `duckduckgo` html (202) | unusable |
| `bing.com/search` HTML | worked twice, then returned **locale-poisoned spam for the same query** — used for **zero** claims |
| CSDN 8×4090 article performance section | **paywalled** ("最低0.47元/天 解锁文章") |

### E.2 Unverified IDs / items (do not cite these as established)
| Item | Status |
|---|---|
| **IEEE 11618998** — "Exploring the Feasibility and Performance of Distributed LLM Serving on Consumer-Grade GPUs" | **Title only**, from search listings. Abstract and testbed **not read**. Possibly the journal version of MoLink. |
| **TCCL** DOI 10.1145/3620666.3651362 | **Title/venue/DOI only** (from RoundPipe's bibliography + search listings). Abstract not read. |
| **"Invited Paper: Experimental Analysis of Distributed Agentic Large-Language Model Inference Architectures"** (DOI 10.1145/3820355.3820384) | Surfaced by search; **ACM DL blocked**; not read. |
| **"Characterizing LLM Kernel Access and Memory Interaction in Multi-Partition NUMA GPUs"** (Semantic Scholar) | Surfaced by search; **Semantic Scholar blocked**; not read. |
| **"Efficient MoE Inference on Single Consumer-grade GPU with Dynamic Expert Caching"** (IEEE 11575328) | Surfaced by search; **IEEE blocked**; not read. |
| **"Predictable LLM Serving on GPU Clusters"** (2508.20274) | ✅ **NOW VERIFIED** — see B.6.5. Fetching `arxiv.org/abs/2508.20274` returned HTTP 200 and the title read back exactly. |
| **specedge** (NeurIPS 2025 Spotlight, `kaist-ina/specedge`) | Surfaced by search; **not fetched**. |
| **OpenReview PDF for the NUMA-aware compiler** (`gPqxViA1fe`) | Surfaced; I read the **nips.cc** abstract instead. |
| **"A NUMA Aware Compiler Framework…" evaluation numbers** | Abstract says ablations are *outlined*, not reported. No numbers. |
| **RTX 4090 P2P bandwidth on an 8-card box specifically** | Not found. Numbers found are 2-card (~12 GB/s host-staged; `nccl-tests#117`) and 4-card (19–22 GB/s granted P2P; kentino, vendor). |
| **The `steveseguin/b70-optimization-lab` TP-vs-PP findings** | The repo exists and I fetched a commit/PR page, but its content is **Intel Arc Pro B70/B60**, not Ada. **No 4090 finding exists there to report.** |
| **"Microarchitecture Is Destiny: Performance and Accuracy of Quantized LLMs on Consumer Hardware"** | **No arXiv ID found.** OpenReview forum `SzQGRR65c0` and its PDF both returned **HTTP 403 challenge**. Only reviewer text is available. **Do not cite.** |
| **`noonghunna/qwen36-dual-3090` INTERNALS.md** | **404** on the path tried. Its quotes (All-reduce on PCIe Gen 4 ~3–5× slower than NVLink; per-stream TP=2 gain ~5%) are **single-source, unverified**. |
| **vLLM `docs/serving/context_parallel_deployment.md`** | Quoted by the sub-search; **I did not fetch it myself**. Treat the quote as single-source until opened. |
| **MLPerf** | **UNSEARCHED, not negative** — no MLPerf surface was successfully fetched. Do not claim MLPerf lacks this. |
| **Generation ablation for speculative decoding across 3+ generations** | Sub-search: **not found**. (Sequoia, arXiv 2402.12374, title verified — but its evaluation coverage was **not read**; do not cite its testbed.) |
| **Generation ablation for KV-cache compression/offloading across 3+ generations** | Sub-search: **not found**. |
| **SGLang / vLLM / TensorRT-LLM default-config cross-generation ablation** | Sub-search: **not found**. |

### E.3 Exact queries used for the absence claims

> ⚠️ **Scope note.** Only **one** of the original two absence claims survived correction: the 8×4090 TP-vs-PP measurement. The cross-generation claim was **overturned** by the parallel sub-search (see B.5 and D(ii)) — its queries are listed below anyway, as the record of what *my own* search covered before the counterexamples arrived.

**Surviving claim — "no published TP-vs-PP scaling measurement on 8× RTX 4090 PCIe-only":**
`8x RTX 4090 tensor parallel vs pipeline parallel vLLM PCIe no NVLink benchmark` · `multi 4090 rig tensor parallelism pipeline parallelism PCIe bandwidth LocalLLaMA` · `"8x 4090" OR "8 x 4090" vLLM tensor parallel pipeline parallel tok/s measurement blog` · `"8× RTX 4090" tensor parallel scaling efficiency published study` · `eight consumer GPUs PCIe pipeline parallel vs tensor parallel measured throughput paper` · `8 GPU RTX 4090 server LLM inference benchmark NCCL all-reduce` · `arXiv paper evaluation "8x RTX 4090" OR "eight RTX 4090" testbed LLM` · `8卡RTX 4090 张量并行 流水线并行 实测` · `vLLM blog tensor parallel vs pipeline parallel PCIe consumer GPU recommendation` · `SGLang blog multi-GPU PCIe topology tensor parallel scaling` · `8x RTX 4090 tensor parallel pipeline parallel comparison measurement published` · `eight consumer GPUs PCIe pipeline parallel vs tensor parallel measured throughput paper`

**Overturned claim — my first-pass "no cross-generation controlled comparison for LLM inference":**
`arXiv paper cross generation GPU comparison H100 4090 Blackwell LLM inference benchmark` · `"consumer GPU" LLM serving 24GB paper evaluation RTX 4090` · `paper "RTX 4090" experimental setup LLM inference PCIe no NVLink penalty` · `SGLang 4090 multi-GPU deployment tensor parallel pipeline parallel benchmark report` · `MLPerf inference hardware generation comparison H100 B200 Ada` · `arXiv benchmark Ada Hopper Blackwell generation comparison LLM serving kernel efficiency study 2026` · `"sm90" "sm89" comparison LLM inference paper Ada Hopper kernel generation` · `MLPerf inference results across GPU generations H100 B200 Ada analysis paper` · plus full-text grep of every verified paper for generation-spanning evaluation tables.
**Why it failed:** my queries were phrased around *serving systems* and *generations as a headline axis*. The counterexamples surface under *kernel/hardware-characterisation* vocabulary ("asymmetric hardware scaling", "physical AI inference gap", "energy-aware benchmark heterogeneous GPU architectures") and are indexed by `huggingface.co/api/papers/search`, not by web_search. **Lesson: `huggingface.co/api/papers/search?q=` was the single most productive discovery surface the sub-search used (15 queries → 984 unique papers); I did not use it at all.**

**Sub-search queries (huggingface.co/api/papers/search):** `LLM inference evaluated on H100 and RTX 4090 and B200` · `LLM serving benchmark Hopper Ada Blackwell comparison` · `speculative decoding evaluation multiple GPU architectures H100 4090` · `KV cache compression evaluated across GPU architectures generations` · `attention kernel benchmark across NVIDIA generations Hopper Ada Blackwell` · `hardware software co-design ablation GPU generation LLM` · `MLPerf inference benchmark hardware scaling analysis GPU generations` · `roofline model arithmetic intensity across GPU generations LLM` · `does speedup transfer across GPU architectures LLM inference` · `GPU architecture dependent performance conclusion generalization LLM` · `quantization evaluation Hopper Ada Blackwell RTX 4090 H100` · `8x RTX 4090 PCIe inference serving without NVLink` · `consumer GPU vs datacenter GPU LLM inference comparison 4090 H100` · `L40S L4 inference benchmark comparison H100` · `cross-generation GPU performance portability LLM inference`

**Surfaces actually searched:** web_search; `arxiv.org/abs/<id>` and `arxiv.org/html/<id>` for every ID encountered; **`huggingface.co/api/papers/search`**; `huggingface.co/buckets/huggingchat/papers-content` full-text mirrors; `aclanthology.org`; `proceedings`/`nips.cc`; GitHub repo-scoped issue search (`github.com/<org>/<repo>/issues?q=…`) and individual issue pages; `raw.githubusercontent.com`; `ar5iv.labs.arxiv.org`; cnblogs / CSDN / Qiita / Zenn; `kentino.se`; `hardware-corner.net`; `local-inference-lab/rtx6kpro`; `noonghunna/club-3090`; `ikawrakow/ik_llama.cpp`; `vllm.ai/blog`; `api2.openreview.net/notes/search`.

**Per project guidance: these are "I did not find it" statements, not existence claims — and one of them was wrong until a second search with different vocabulary overturned it. Treat the corrected D(ii) as the authority.**

---

## Appendix — Verified arXiv ID ledger (title read back from `arxiv.org/abs/`)

| ID | Title (as fetched) | Date |
|---|---|---|
| 2406.10774 | Quest: Query-Aware Sparsity for Efficient Long-Context LLM Inference | 16 Jun 2024 (v2 26 Aug 2024) |
| 2607.28633 | Topology-Aware Data Movement for Disaggregated GPU Inference | 19 Apr 2026 (v2 7 Aug 2026) |
| 2606.01927 | Scaling LLM Inference Beyond Amdahl's Limits via Eliminating Non-Scalable Overheads | 1 Jun 2026 |
| 2604.27085 | Efficient Training on Multiple Consumer GPUs with RoundPipe | 29 Apr 2026 |
| 2512.15306 | LLMQ: Efficient Lower-Precision Pretraining for Consumer GPUs | — |
| 2311.03687 | Dissecting the Runtime Performance of the Training, Fine-tuning, and Inference of Large Language Models | 7 Nov 2023 (v2 1 Dec 2023) |
| 2312.12456 | PowerInfer: Fast Large Language Model Serving with a Consumer-grade GPU | SOSP '24 |
| 2608.23962 | More GPUs or a Smaller Cache? Tensor Parallelism versus KV Compression for Memory-Bound LLM Serving | 25 Aug 2026 |
| 2507.07120 | Helix Parallelism: Rethinking Sharding Strategies for Interactive Multi-Million-Token LLM Decoding | 7 Jul 2025 |
| 2605.08467 | CUDAHercules: Benchmarking Hardware-Aware Expert-level CUDA Optimization for LLMs | 8 May 2026 |
| 2508.20274 | Predictable LLM Serving on GPU Clusters | 27 Aug 2025 |
| 2405.08944 | Challenges in Deploying Long-Context Transformers: A Theoretical Peak Performance Analysis | 14 May 2024 |
| 2603.05232 | SlideSparse: Fast and Flexible (2N-2):2N Structured Sparsity | 5 Mar 2026 |
| 2604.09048 | Watt Counts: Energy-Aware Benchmark for Sustainable LLM Inference on Heterogeneous GPU Architectures | 10 Apr 2026 |
| 2605.30571 | Memory-Bound but Not Bandwidth-Limited: The Physical AI Inference Gap in Batch-1 LLM Decode | 28 May 2026 |

**Re-verified by me directly (fetched `arxiv.org/abs/<id>`, title read back, abstract text matched against the quote):** 2603.05232, 2604.09048, 2605.30571, 2405.08944. **Verified only by the parallel sub-search (single-source, treat with care):** 2504.11750, 2606.08761, 2402.13499, 2603.05451, 2601.22076, 2607.02391.

**Two task-brief corrections:**
1. **2606.01927 is Albireo, but its thesis is Amdahl's-law non-scalable overhead removal and the `t_e` optimal TP degree — the NVLink-vs-PCIe A100 comparison is a *generality check* (three testbeds), not the paper's main axis.** The brief's description ("compares NVLink vs PCIe A100") is accurate as far as it goes but understates the paper.
2. **2607.28633 is exactly the title given, but its abstract explicitly states "Full evaluation requires multi-node clusters with heterogeneous interconnects and CXL 3.0 hardware that is beyond academic resources"** — i.e. it is a design/position paper without a full evaluation. Its testbed is datacenter (NVLink/IB/TCP/CXL), **not** a 4090 or PCIe-consumer box.

*Non-arXiv peer-reviewed items verified by fetching their landing pages or PDFs: BNTU DOI 10.21122/2309-4923-2026-1-54-59; Mobius ASPLOS '23 PDF (storage.cs.tsinghua.edu.cn); MoLink Findings of EMNLP 2025 (aclanthology.org, PDF read); NeurIPS 2025 workshop abstract (nips.cc).*
