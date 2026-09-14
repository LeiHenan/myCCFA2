# S10 — Benchmark-methodology defects in vLLM / SGLang / TensorRT-LLM

Sub-agent S10. Scope: cases where a maintainer or contributor states, in their own words, that a
benchmark was wrong, misleading, measured the wrong thing, included prefix caching, used too few
prompts/warmup, compared against an unfair baseline, or that a reported number was noise / not
reproducible. Every row below was retrieved with `python3 fetch.py gh <url>` (title, state, labels,
stateReason, all comment bodies) or `python3 fetch.py get <url>`; raw HTML is cached in
`/tmp/accelsrc/`. All rows are EVIDENCE: READ BODY. No row is included on a title alone.
Exact queries issued are listed in the SEARCH LOG at the end.

---

## A. CLOSED / resolved methodology fixes

### A.gh1 | CLOSED
QUESTION: Does repeating `vllm bench serve` against a live server inflate measured throughput via the prefix cache, and by how much?
WHO: brianosaurus (contributor, issue author)
ARTIFACT: vllm-project/vllm issue #52884 — "[Feature]: warn from vllm bench serve when a repeated random-dataset run hits a warm prefix cache, which inflates throughput by up to 86%"
URL: https://github.com/vllm-project/vllm/issues/52884
STATUS: closed (stateReason COMPLETED; closed by #53920)
EVIDENCE: READ BODY
QUOTE: "Anyone building a concurrency ladder by calling `vllm bench serve` in a loop against one server, which is a common thing to do by hand, is measuring a warm cache from the second point on and has no signal that it is happening."

### A.gh2 | CLOSED
QUESTION: What exactly makes a repeated random-dataset bench run inflate throughput, stated by the author of the fix?
WHO: zupengwang (PR author)
ARTIFACT: vllm-project/vllm PR #53920 — "[Benchmark] Warn on warm prefix cache for random serve runs"
URL: https://github.com/vllm-project/vllm/pull/53920
STATUS: merged
EVIDENCE: READ BODY
QUOTE: "Repeating the same vllm bench serve workload against a server with a warm prefix cache can therefore reuse most prompt tokens and materially inflate the reported throughput."

### A.gh3 | CLOSED
QUESTION: Is the fixed-seed / warm-prefix-cache / no-warmup hazard documented in the vLLM version on the target rig (0.29.0)?
WHO: BillJPG (issue author, verifying against main before closing)
ARTIFACT: vllm-project/vllm issue #54227 — "vllm bench serve with a fixed --seed against a default server can end up benchmarking the prefix cache (measured: TPOT -34%, TTFT 49x, throughput 2.32x on one cell)"
URL: https://github.com/vllm-project/vllm/issues/54227 ; https://raw.githubusercontent.com/vllm-project/vllm/v0.29.0/docs/benchmarking/cli.md
STATUS: closed as resolved; the caution is shipped in the v0.29.0 benchmark CLI docs
EVIDENCE: READ BODY
QUOTE: "Verified against current `main`: the actionable parts are covered — `docs/benchmarking/cli.md` now warns about prefix-cache reuse across repeated `vllm bench serve` runs (mitigations: vary `--seed`, restart the server, or use `vllm bench sweep serve`, which resets caches between runs), and `--num-warmups` handles first-request warm-up. Closing as resolved."
NOTE (retrieved text of the shipped doc, same URL list, `fetch.py get`): "Repeating `vllm bench serve` against the same server can reuse prompts left in the prefix cache and inflate throughput."

### A.gh4 | CLOSED
QUESTION: Are the TTFT/E2EL percentiles printed by `vllm bench serve` under `--max-concurrency` measuring the quantity they claim to measure?
WHO: QHarshil (issue author)
ARTIFACT: vllm-project/vllm issue #54101 — "[Bug]: bench serve latency metrics exclude client-side queueing under --max-concurrency"
URL: https://github.com/vllm-project/vllm/issues/54101
STATUS: closed (stateReason COMPLETED)
EVIDENCE: READ BODY
QUOTE: "When --max-concurrency is set, a request is generated on schedule and then waits on a benchmark-side asyncio.Semaphore before the HTTP request function is entered. The TTFT and E2EL timers start inside that function, so the wait is not counted."

### A.gh5 | CLOSED
QUESTION: Does `vllm bench serve` reproduce what `benchmark_serving.py` measured, or did a silent default change alter the workload being measured?
WHO: ycma8 (diagnosed the cause); yeqcharlotte (maintainer, stated the intended default); gshtras (reporter)
ARTIFACT: vllm-project/vllm issue #24684 — "[Bug]: `vllm bench serve` isn't an exact replacement of benchmark_serving.py"
URL: https://github.com/vllm-project/vllm/issues/24684
STATUS: closed (stateReason COMPLETED; CLI error-message fix in PR #24819)
EVIDENCE: READ BODY
QUOTE: "Looks like the issue comes from the default value of --dataset-name. Old version defaults to **sharegpt**, but the new version defaults to **random**. Not sure if that change was intentional — if not, I can send a quick PR to fix it 🙂"
NOTE (second retrieved comment, same page): "Providing a --dataset-path was forcing the old benchmark to use sharegpt instead of its default random. Seems like in the new one it's ignored, breaking existing workflows"

### A.gh6 | CLOSED
QUESTION: A user measured SGLang only ~30% faster than vLLM against an advertised 3.8x — was the advertised speedup inflated, or was the comparison non-equivalent?
WHO: zhyncs (SGLang maintainer)
ARTIFACT: sgl-project/sglang issue #998 — "[Feature] Inference speed difference between sglang and vllm is smaller than advertised"
URL: https://github.com/sgl-project/sglang/issues/998
STATUS: closed (stateReason COMPLETED)
EVIDENCE: READ BODY
QUOTE: "Your conclusion is based on your model and test scenario, while the blog's conclusions come from its models, test scenarios, and device types. I believe they are **NOT EQUIVALENT**."

### A.gh7 | CLOSED
QUESTION: Was the reproducing benchmark in #998 run with enough prompts and the documented request pattern?
WHO: zhyncs (SGLang maintainer)
ARTIFACT: sgl-project/sglang issue #998 — "[Feature] Inference speed difference between sglang and vllm is smaller than advertised"
URL: https://github.com/sgl-project/sglang/issues/998
STATUS: closed (stateReason COMPLETED)
EVIDENCE: READ BODY
QUOTE: "Additionally, I noticed that you have very few test prompts(only 38 items) and the method for sending requests differs from what was described in the blog."

### A.gh8 | CLOSED
QUESTION: A SGLang maintainer also flags the batch size used by that comparison — what was measured instead of throughput?
WHO: merrymercy (SGLang maintainer/co-author), same thread
ARTIFACT: sgl-project/sglang issue #998 — "[Feature] Inference speed difference between sglang and vllm is smaller than advertised"
URL: https://github.com/sgl-project/sglang/issues/998
STATUS: closed (stateReason COMPLETED); the same comment closes the thread: "Thanks for reporting your results. Close this for now."
EVIDENCE: READ BODY
QUOTE: "You can add `--enable-torch-compile` when you launch the sglang server, it will be much faster. It seems you only test batch size = 1, but we typically test the throughput under the maximum batch size. Please follow the above links to reproduce our benchmark."

### A.gh9 | CLOSED
QUESTION: Does `bench_one_batch_server`'s throughput/latency output describe the workload it is pointed at?
WHO: hanming-lu (SGLang maintainer)
ARTIFACT: sgl-project/sglang issue #18712 — "[Bug] bench_one_batch_server metrics calculations and nightlies"
URL: https://github.com/sgl-project/sglang/issues/18712
STATUS: closed (stateReason COMPLETED); the agreed fix is to guard the script with batch-size and token-capacity checks
EVIDENCE: READ BODY
QUOTE: "The reason it's called bench one batch is to bench just one batch. If the input/output pair requires more than one batch, the metrics is not accurate, which is expected. The right way is to guard the script with these two checks instead."

### A.gh10 | CLOSED
QUESTION: Does an SGLang benchmark client report valid metrics when the served model streams reasoning tokens instead of content tokens?
WHO: JustinTong0323 (SGLang contributor, MiMo-V2.5 cookbook validation)
ARTIFACT: sgl-project/sglang issue #23949 — "bench_serving reports zero chat metrics for reasoning image workloads"
URL: https://github.com/sgl-project/sglang/issues/23949
STATUS: closed (stateReason COMPLETED)
EVIDENCE: READ BODY
QUOTE: "sglang.bench_serving can print misleading zero-valued metrics for OpenAI-compatible chat image workloads when the model streams reasoning content instead of normal delta.content chunks."
NOTE (second retrieved passage, same page): "The current output looks like valid benchmark data but contains placeholder zeros, which is easy to copy into docs or benchmark reports by mistake."

### A.gh11 | CLOSED
QUESTION: Does `bench_gdn_replayssm_decode.py` time the quantity its speedup column claims?
WHO: davidli1515 (issue author)
ARTIFACT: sgl-project/sglang issue #38398 — "[Bug] bench_gdn_replayssm_decode.py never advances write_pos, so it times only the non-flush phase and overstates the speedup by up to 21%"
URL: https://github.com/sgl-project/sglang/issues/38398
STATUS: closed (stateReason COMPLETED)
EVIDENCE: READ BODY
QUOTE: "It measures one phase of the L-phase cycle, the cheapest one, and reports it as the per-step cost."

### A.gh12 | CLOSED
QUESTION: Why could a published trtllm-bench throughput figure (27,688 tok/s for Llama-3.1-8B-FP8, ISL/OSL 128/128) not be reproduced (7,099 tok/s measured)?
WHO: zbpatel (NVIDIA, TensorRT-LLM maintainer)
ARTIFACT: NVIDIA/TensorRT-LLM issue #6294 — "[Performance issue] Unable to Reproduce the Throughput of Llama 3.1 8B FP8 on H100"
URL: https://github.com/NVIDIA/TensorRT-LLM/issues/6294
STATUS: closed (stateReason COMPLETED)
EVIDENCE: READ BODY
QUOTE: "I think the remaining difference here is likely that the test we ran was using the H100 SXM 80GB variant which is a higher performance version than H100 PCIE variant your test used. The SXM variant has a significantly higher power budget, memory bandwidth and cuda core count which is going to lead to higher performance on this application vs PCIE."
NOTE (second retrieved comment by the same maintainer, same page, on the doc defect this exposed): "I'm also working on updating our performance document right now, I will add some more information about the GPU variants used as this is important information and should be made more clear on the page."

### A.gh13 | CLOSED
QUESTION: Was the reported non-reproduction also caused by the reproducing benchmark being misconfigured (CUDA-graph batch sizes, recorded sequence lengths)?
WHO: zbpatel (NVIDIA, TensorRT-LLM maintainer)
ARTIFACT: NVIDIA/TensorRT-LLM issue #6294 — "[Performance issue] Unable to Reproduce the Throughput of Llama 3.1 8B FP8 on H100"
URL: https://github.com/NVIDIA/TensorRT-LLM/issues/6294
STATUS: closed (stateReason COMPLETED)
EVIDENCE: READ BODY
QUOTE: "Your config only has 128 for the cuda graph batch sizes, which is going to limit your performance gains from this feature - if you check the `trtllm-bench` output, you can see the runtime max batch size is auto-tuned to 4096, so I would recommend trying at least this high on the cuda graph batch sizes."
NOTE (second retrieved comment by the same maintainer, same page, on the reporter's own run): "in both of the rerun tests it looks like you only ran 128 isl, 28 osl. guessing this was just a typo :)"

### A.gh14 | CLOSED
QUESTION: Are TensorRT-LLM's perf-regression measurements comparable run to run, and is the comparison confounded?
WHO: MrGeva (issue author)
ARTIFACT: NVIDIA/TensorRT-LLM issue #8391 — "[Feature]: Solve the Perf regression test large variance"
URL: https://github.com/NVIDIA/TensorRT-LLM/issues/8391
STATUS: closed (stateReason COMPLETED)
EVIDENCE: READ BODY
QUOTE: "The test_perf.py regression test shows large variance when run in CI. I see that it runs on two different device types: H100 NVL and H100 PCIE. according to that the perf varies and also the clock frequencies."

---

## B. OPEN asks

### B.gh1 | OPEN
WHO IS STILL ASKING: ZFXzzz (issue author)
URL: https://github.com/vllm-project/vllm/issues/56653
ASKING QUOTE: "Is inclusive one-second bucket occupancy intentional for `max_concurrent_requests`, or should this field represent exact instantaneous concurrency? If the latter, I can prepare a small CPU-only regression test and a minimal fix."
WHAT IS MISSING: `vllm bench serve` still reports `max_concurrent_requests` by incrementing integer-second buckets inclusively, so adjacent half-open request intervals report a peak of 2x the configured `--max-concurrency` (observed 1→2, 2→4, 4→8, 8→16, 16→32 on a vLLM 0.28.0 two-node run). The metric contract is still unconfirmed by maintainers.
EVIDENCE: READ BODY

### B.gh2 | OPEN
WHO IS STILL ASKING: akhilHaze (commenter on jacob-sunho-kim's RFC thread)
URL: https://github.com/vllm-project/vllm/issues/42484
ASKING QUOTE: "One thing I'd push on in your protocol: are you holding the KV cache occupancy constant across your c=4 and c=16 runs, or is cache pressure itself a confound in the plateau you're observing? That delta could explain a lot of what looks like a concurrency ceiling."
WHAT IS MISSING: The production-boundary RFC's own methodology section does not report a KV-cache-occupancy control across its concurrency points, and the issue is still open (last activity was the automated stale notice); the author explicitly asks for "methodology critique" and no maintainer has answered the confound question.
EVIDENCE: READ BODY

### B.gh3 | OPEN
WHO IS STILL ASKING: spped2000 (RFC author)
URL: https://github.com/vllm-project/vllm/issues/51963
ASKING QUOTE: "For English that distinction rarely matters. For scripts whose tokenizers are less efficient it inverts the ranking: **tokens/s flatters exactly the models that make the reader wait longest.**"
WHAT IS MISSING: `BenchmarkMetrics` in `vllm/benchmarks/serve.py` still reports only `output_throughput` and `total_token_throughput`, with no per-character rate, so cross-model rankings on non-English workloads remain inverted; the RFC is open with no implementation.
EVIDENCE: READ BODY

### B.gh4 | OPEN
WHO IS STILL ASKING: yuyz-cyber (PR author)
URL: https://github.com/vllm-project/vllm/pull/46938
ASKING QUOTE: "vllm bench serve reports latency and throughput, but not the prefix cache hit rate — so benchmarking prefix caching means manually scraping /metrics and computing the rate by hand. That is error-prone: the Prometheus counters are cumulative since server start, and the server's logged hit rate uses a rolling 1000-request window — neither corresponds to \"this run\"."
WHAT IS MISSING: The PR that would print the per-run prefix-cache hit rate in the console summary and saved JSON is still open (not merged), so a bench run cannot self-report whether its throughput was cache-assisted.
EVIDENCE: READ BODY

### B.gh5 | OPEN
WHO IS STILL ASKING: gpulost (PR author)
URL: https://github.com/sgl-project/sglang/pull/38868
ASKING QUOTE: "An HTTP 200 SSE response can report a generation error after its headers have been sent. The serving benchmark currently counts these responses as successful for native /generate and OpenAI chat. An error before the first token even credits the requested maximum output length as generated tokens."
WHAT IS MISSING: SGLang's `bench_serving` still counts server-side streaming errors as successful requests and can credit the requested `max_tokens` as generated tokens, so failed requests pollute the success-throughput and latency aggregates; the fixing PR is open.
EVIDENCE: READ BODY

### B.gh6 | OPEN
WHO IS STILL ASKING: Roxy341 (PR author)
URL: https://github.com/sgl-project/sglang/pull/38917
ASKING QUOTE: "Running the generated-shared-prefix benchmark with --gsp-fast-prepare caches placeholder prompt_len=1 values. A later run without that flag reuses the same cache and reports those placeholders as measured input-token counts."
WHAT IS MISSING: The generated-shared-prefix benchmark's dataset cache key still omits the preparation mode, so a later run can report placeholder token counts as real measured input lengths; the fixing PR is open.
EVIDENCE: READ BODY

---

## C. ABANDONED / defective measurement / claims retracted or corrected by their own author  (benchmark-methodology defects)

### C.gh1 | ABANDONED
WHO: Robinwm049 (issue author, correcting his own benchmark harness)
ARTIFACT: vllm-project/vllm issue #54677 — "[Perf]: DeepSeek-V4-Flash TP4 on 4xH800: v0.28.0 needs ~4 GiB/GPU more non-KV memory than v0.26.0; my earlier TTFT-regression claim was caused by VLLM_USE_BREAKABLE_CUDAGRAPH=0 on my side (retracted)"
URL: https://github.com/vllm-project/vllm/issues/54677
STATED REASON: "My "fixed" harness counted `delta.reasoning_content`. In 0.28 the field is **`reasoning`** (`vllm/entrypoints/openai/engine/protocol.py:404` — `DeltaMessage` has `role/content/reasoning/tool_calls`). So thinking-enabled runs were still mis-timed."
EVIDENCE: READ BODY

### C.gh2 | ABANDONED
WHO: Robinwm049 (issue author, retracting his own throughput claim)
ARTIFACT: vllm-project/vllm issue #54677 — "[Perf]: DeepSeek-V4-Flash TP4 on 4xH800 ... (retracted)"
URL: https://github.com/vllm-project/vllm/issues/54677
STATED REASON: "I was reporting per-stream `completion_tokens / (total − TTFT)`. When requests queue, fewer streams decode concurrently and each one's rate looks better. Measured as aggregate output tokens per second, 0.28 is **slower**, not faster."
EVIDENCE: READ BODY

### C.gh3 | ABANDONED
WHO: Robinwm049 (issue author; the two comparison arms were not running the same code path)
ARTIFACT: vllm-project/vllm issue #54677 — "[Perf]: DeepSeek-V4-Flash TP4 on 4xH800 ... (retracted)"
URL: https://github.com/vllm-project/vllm/issues/54677
STATED REASON: "My 0.26 baseline was running with the auto-enabled breakable graph, so the two versions were never executing the same way."
EVIDENCE: READ BODY

### C.gh4 | ABANDONED
WHO: jacob-sunho-kim (RFC author, correcting his own published fairness number)
ARTIFACT: vllm-project/vllm issue #42484 — "[RFC] Production-boundary measurement on H100 + vLLM 0.19.1: throughput plateau at c=4→16, methodology critique invited"
URL: https://github.com/vllm-project/vllm/issues/42484
STATED REASON: "The original 4.3× framing is therefore Spheron-specific. Generalizing it to a universal fairness rule would be wrong."
EVIDENCE: READ BODY

### C.gh5 | ABANDONED
WHO: jacob-sunho-kim (RFC author, reporting a benchmark that measured server state rather than the workload)
ARTIFACT: vllm-project/vllm issue #42484 — "[RFC] Production-boundary measurement on H100 + vLLM 0.19.1 ..."
URL: https://github.com/vllm-project/vllm/issues/42484
STATED REASON: "New Finding 5 — Server State Drift. The same Lambda instance produced different ctx 8K severity under warm (60 min uptime) vs fresh-restart server states. Mechanism not attributed. 3. The original 4.3× fairness number is Spheron-specific. Lambda 90/10 mix degraded short P99 only 1.08× under both n=20 and n=100."
EVIDENCE: READ BODY

### C.gh6 | ABANDONED
WHO: zeng-zc (issue author, withdrawing his own benchmark-defect claim after merrymercy pointed at the docs)
ARTIFACT: sgl-project/sglang issue #1630 — "[Bug] The time unit in bench_serving is wrong on A800-SXM4-40GB, using perf_counter_ns could fix"
URL: https://github.com/sgl-project/sglang/issues/1630
STATED REASON: "So the benchmark output is right. What an amazing thingthe the latency is so big (mean ~800s) when processing 3000 requests... Sorry, it's my fault, please close this issue."
EVIDENCE: READ BODY

---

## DROPPED CANDIDATES (no human closure statement — recorded so they are not counted as rows)

- vllm-project/vllm issue #19210 "[Bug]: Strange metrics when running `benchmark_serving` under high concurrency" —
  https://github.com/vllm-project/vllm/issues/19210 — closed by the github-actions stale bot
  ("This issue has been automatically closed due to inactivity."); the only human statements are the
  reporter's own diagnosis, e.g. "After I turned off the prefix cache by setting
  `--no-enable-prefix-caching`, I did find that the subsequent inferences no longer had abnormally low
  ttft, but the tpot was still abnormally high". No maintainer statement; DROPPED per the stale-bot rule.
- sgl-project/sglang issue #24254 "[Bug] bench_serving sglang-oai reports wrong output_len ... causes
  incorrect TPOT / throughput / retokenized metrics" — https://github.com/sgl-project/sglang/issues/24254
  — closed by inactivity bot, only comment retrieved is "This issue has been automatically closed due to
  inactivity. Please feel free to reopen it if needed."; DROPPED.
- sgl-project/sglang issue #20839 "Benchmark results on GB200 are better than on GB300 for the same model"
  — https://github.com/sgl-project/sglang/issues/20839 — human reply exists (Fridge003) but is about
  hardware workload regimes, not a benchmark defect; closure was the inactivity bot. DROPPED.
- sgl-project/sglang issue #21061 "Benchmark: SGLang vs. vLLM Scaling under High Concurrency" —
  https://github.com/sgl-project/sglang/issues/21061 — closed by inactivity bot; the only human comment
  points at an experimental C++ radix tree flag. DROPPED.
- vllm-project/vllm issue #37666 — https://github.com/vllm-project/vllm/issues/37666 — metrics discussion,
  closed by inactivity bot after a contributor proposed a fix; no maintainer statement of a defect. DROPPED.
- vllm-project/vllm issues #55739, #55441, #55982 (sweep-bounds mismatch, structured-output correctness
  check never validating the schema, PyTorch benchmark export mislabelling compilation mode) — real
  benchmark-tool defects with READ BODY detail, but they carry no human closure statement and are not
  statements that "a benchmark was wrong/misleading" in the sense this map needs. Not scored.
- TensorRT-LLM issue #7357 "[Bug]: trtllm-bench Throughput Test Fails with 4+ Requests for Qwen2.5-VL-7B-Instruct"
  — https://github.com/NVIDIA/TensorRT-LLM/issues/7357 — human maintainer reply exists ("For Qwen2.5-VL,
  we have an known issue that it fails on cuda_graph_padding. We are fixing them in this PR.") but it is a
  benchmark *crash*, not a benchmark that measured a wrong number. Not scored.

## SEARCH LOG (queries actually issued)

Repo list pages fetched with `python3 s10_srch.py` / `s10_batch*.py` (server-rendered;
`https://github.com/search?q=...&type=issues` was NOT used because it returns no result list):

- https://github.com/vllm-project/vllm/issues?q=is%3Aissue+benchmark+prefix+caching (and variants: "benchmark prefix caching throughput inflated", "is:issue prefix caching inflates benchmark", "is:issue benchmark random dataset prefix", "is:issue in:title \"prefix caching\" throughput", "is:issue prefix cache benchmark docs pitfall")
- https://github.com/vllm-project/vllm/issues?q=is%3Aissue+benchmark+wrong+methodology
- https://github.com/vllm-project/vllm/issues?q=is%3Aissue+misleading+benchmark
- https://github.com/vllm-project/vllm/issues?q=is%3Aissue+benchmark+warmup ; .../issues?q=is:issue+in:title+warmup ; .../issues?q=is:issue+warmup+request+first+request+latency
- https://github.com/vllm-project/vllm/issues?q=is%3Aissue+benchmark+variance+run+to+run ; .../issues?q=is:issue+in:title+variance ; .../issues?q=is:issue+benchmark+noise+speedup ; .../issues?q=is:issue+in:title+benchmark+noise
- https://github.com/vllm-project/vllm/issues?q=is%3Aissue+benchmark+not+reproducible ; .../issues?q=is:issue+in:title+%22not+reproducible%22
- https://github.com/vllm-project/vllm/issues?q=is%3Aissue+ignore_eos+benchmark ; .../issues?q=is:issue+benchmark_serving ; .../issues?q=is:issue+in:title+%22bench+serve%22 ; .../issues?q=is:issue+in:title+benchmark+sort:updated-desc
- https://github.com/vllm-project/vllm/issues?q=is%3Aissue+enforce-eager+benchmark
- https://github.com/vllm-project/vllm/issues?q=is%3Aissue+cuda+graph+capture+first+request+latency+benchmark
- https://github.com/vllm-project/vllm/issues?q=is%3Aissue+measured+incorrectly ; .../issues?q=is:issue+benchmark+throughput+number+wrong ; .../issues?q=is:issue+throughput+measurement+wrong
- https://github.com/vllm-project/vllm/issues?q=is%3Aissue+in:title+retract ; .../issues?q=is:issue+in:title+speedup
- https://github.com/vllm-project/vllm/pulls?q=is%3Apr+is%3Aclosed+benchmark ; .../pulls?q=is:pr+is:closed+num-warmups ; .../pulls?q=is:pr+is:closed+ignore-eos ; .../pulls?q=is:pr+bench+serve ; .../pulls?q=is:pr+is:closed+prefix+cache+benchmark+warn ; .../pulls?q=is:pr+is:closed+bench+serve+docs+warmup
- https://github.com/sgl-project/sglang/issues?q=is%3Aissue+bench_serving+wrong ; .../issues?q=is:issue+bench_serving+metrics+wrong ; .../issues?q=is:issue+benchmark+misleading ; .../issues?q=is:issue+benchmark+inaccurate
- https://github.com/sgl-project/sglang/issues?q=is%3Aissue+in:title+benchmark ; .../issues?q=is:issue+in:title+warmup ; .../issues?q=is:issue+in:title+variance ; .../issues?q=is:issue+in:title+%22prefix+cache%22 ; .../issues?q=is:issue+in:title+reproducible ; .../issues?q=is:issue+in:title+speedup ; .../issues?q=is:issue+in:title+ignore_eos
- https://github.com/sgl-project/sglang/issues?q=is%3Aissue+speedup+not+reproducible ; .../issues?q=is:issue+benchmark+prefix+cache+hit ; .../issues?q=is:issue+benchmark+number+inflated ; .../issues?q=is:issue+benchmark+run+to+run+variation
- https://github.com/sgl-project/sglang/pulls?q=is%3Apr+in:title+benchmark ; .../pulls?q=is:pr+is:closed+benchmark+metric+fix ; .../pulls?q=is:pr+is:closed+is:unmerged+benchmark ; .../pulls?q=is:pr+in:title+warmup
- https://github.com/NVIDIA/TensorRT-LLM/issues?q=is%3Aissue+in:title+benchmark ; .../issues?q=is:issue+benchmark+unfair ; .../issues?q=is:issue+benchmark+measurement ; .../issues?q=is:issue+comparison+with+vllm ; .../issues?q=is:issue+speedup+claim ; .../issues?q=is:issue+benchmark+script ; .../issues?q=is:issue+perf+regression+variance ; .../issues?q=is:issue+unable+to+reproduce+performance ; .../issues?q=is:issue+trtllm-bench ; .../issues?q=is:issue+latency+measurement+wrong ; .../issues?q=is:issue+benchmark+latency+wrong ; .../issues?q=is:issue+in:title+variance
- https://github.com/NVIDIA/TensorRT-LLM/pulls?q=is%3Apr+in:title+benchmark ; .../pulls?q=is:pr+in:title+warmup ; .../pulls?q=is:pr+is:closed+is:unmerged+benchmark
- https://raw.githubusercontent.com/vllm-project/vllm/v0.29.0/vllm/benchmarks/serve.py (grepped for `num_warmups` default and `ignore_eos` handling)
- https://raw.githubusercontent.com/vllm-project/vllm/v0.29.0/docs/benchmarking/cli.md (prefix-cache warning and `--num-warmups`)

Individual artifacts fetched in full with `python3 fetch.py gh <url>` (all listed as URLs above, plus):
https://github.com/vllm-project/vllm/issues/14795 , /15793 , /16375 , /17788 , /19210 , /21294 , /21596 ,
/24684 , /25677 , /37666 , /39903 , /39963 , /42484 , /42548 , /45178 , /51963 , /52884 , /54101 ,
/54227 , /54677 , /55441 , /55739 , /55982 , /56653 , /7619 ;
https://github.com/vllm-project/vllm/pull/46938 , /53920 ;
https://github.com/sgl-project/sglang/issues/998 , /1630 , /1729 , /5184 , /18712 , /20839 , /21061 ,
/22013 , /23949 , /24254 , /26632 , /27406 , /28969 , /34709 , /36131 , /37834 , /38398 ;
https://github.com/sgl-project/sglang/pull/38705 , /38868 , /38917 , /39284 ;
https://github.com/NVIDIA/TensorRT-LLM/issues/2487 , /3058 , /6294 , /7357 , /7364 , /7406 , /8391 ,
/8655 , /9470 , /10705 , /11279 , /12651 , /12933 , /16827 ;
https://github.com/NVIDIA/TensorRT-LLM/pull/6270

## NEGATIVE FINDINGS

- No vLLM/SGLang/TensorRT-LLM issue or PR was found in which a maintainer states that a reported speedup
  was *within run-to-run noise* (error bars / confidence intervals on a speedup). Queries tried:
  "is:issue in:title variance" (vLLM: 10 hits, all about tensor/logprob variance or cross-instance
  quality, not benchmark error bars), "is:issue benchmark error bars confidence interval" (vLLM: 0 relevant
  hits), "is:issue benchmark noise" (vLLM: 0 relevant hits), "is:issue in:title variance" (SGLang: 3 hits,
  none benchmark-noise), "is:issue in:title variance" (TRT-LLM: 4 hits, of which #8391 is the only one and
  it is about CI device heterogeneity, not speedup significance).
- No vLLM/SGLang issue was found where the maintainer states that the *published blog speedup* was wrong;
  the closest are sglang#998 (maintainer states the reader's comparison was NOT EQUIVALENT, A.gh6-A.gh8)
  and the retractions in vllm#54677 (C.gh1-C.gh3), which are user-side not vendor-side.
- No SGLang counterpart was found to vLLM's `--num-warmups` default-0 discussion: `--warmup-requests`
  default 1 was not the subject of any retrieved human statement. Queries tried:
  "is:issue in:title warmup" (SGLang), "is:issue warmup request benchmark latency" (SGLang),
  "is:pr in:title warmup" (SGLang), "is:pr is:closed warmup-requests" (SGLang).
- No `--disable-ignore-eos` / `ignore_eos` benchmark-inflation statement with a human diagnostic was found
  in SGLang: "is:issue in:title ignore_eos" returned only #10315 (MTP + `ignore_eos=true` not working,
  a correctness bug, not a benchmark-methodology statement). Queries tried on vLLM
  ("is:issue ignore_eos benchmark", "is:pr is:closed ignore-eos") returned no benchmark-inflation statement.
- No TensorRT-LLM issue was found in which a maintainer states "our benchmark was measuring X incorrectly"
  about the `trtllm-bench` harness itself; the closest retrieved items are #6294 (misconfigured reproducing
  benchmark + wrong GPU variant in the published table, A.gh12-A.gh13) and the dropped #7357 (bench crash).
  Queries tried: "is:issue trtllm-bench", "is:issue benchmark measurement", "is:issue benchmark wrong",
  "is:issue benchmark script", "is:issue latency measurement wrong", "is:pr in:title benchmark".
