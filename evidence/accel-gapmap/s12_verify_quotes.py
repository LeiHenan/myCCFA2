#!/usr/bin/env python3
"""S12 final quote verification: re-fetch each URL and confirm the quote appears verbatim."""
import sys, json, concurrent.futures as cf
sys.path.insert(0, "/Users/leihenan/Desktop/myProject/evidence/accel-gapmap")
import s12_gh

ITEMS = [
 ("https://github.com/NVIDIA/TensorRT-LLM/issues/7131",
  "@Shixiaowei02 please highlight on docs that trtllm-build flow is deprecated."),
 ("https://github.com/NVIDIA/TensorRT-LLM/issues/9981",
  "Closing this as the comments above should have addressed it. Please open a new one if it persists on the latest version.  Additionally, please be noted that the TensorRT backend has been deprecated. :)"),
 ("https://github.com/NVIDIA/TensorRT-LLM/issues/10306",
  "Sorry we never replied to this one. Closing as stale since the TensorRT engine-build path is deprecated in favor of the PyTorch backend. Please open a new issue if the problem persists in the latest version."),
 ("https://github.com/NVIDIA/TensorRT-LLM/issues/7138",
  "No, we\u2019re not planning to add new features to the TensorRT backend. Instead, our focus will be on the PyTorch backend moving forward."),
 ("https://github.com/NVIDIA/TensorRT-LLM/issues/11569",
  "This backend is now in \"do not harm\" mode, and hence not seeing active development. The Qwen3 VL model is explicitly not supported in that path."),
 ("https://github.com/NVIDIA/TensorRT-LLM/issues/2474",
  "Additionally, since the 1.0 release, TensorRT-LLM has transitioned to the PyTorch workflow, which no longer relies on engine builds."),
 ("https://github.com/NVIDIA/TensorRT-LLM/issues/12703",
  "The legacy TensorRT backend targeted by this issue, including its SmoothQuant layers and trtllm-build was removed by #15918, with its CI coverage removed by #16610. Therefore, please hold off on the proposed implementation; we should close this issue as no longer applicable."),
 ("https://github.com/NVIDIA/TensorRT-LLM/issues/15021",
  "One key thing to flag: the `quantize.py `exports the checkpoint for the legacy TensorRT backend, which is being deprecated (#11723) in favor of the PyTorch backend, so we  recommend using PyTorch backend workflow instead."),
 ("https://github.com/NVIDIA/TensorRT-LLM/pull/11723",
  "Surface deprecation warnings across all legacy TensorRT engine-build entry points,"),
 ("https://github.com/NVIDIA/TensorRT-LLM/pull/15918",
  "It is step 1.5 of the staged TensorRT-backend removal"),
 ("https://github.com/NVIDIA/TensorRT-LLM/pull/15918",
  "Behavior change: LLM(backend=\"tensorrt\"/\"trt\") is rejected at construction; the engine-execution path no longer exists."),
 ("https://github.com/NVIDIA/TensorRT-LLM/pull/14941",
  "XQA has moved to JIT path since #10335 ."),
 ("https://github.com/NVIDIA/TensorRT-LLM/pull/13395",
  "The --disable_xqa flag was removed from trtllm-build; the legacy gpt-attention doc still referenced it."),
 ("https://github.com/NVIDIA/TensorRT-LLM/issues/11799",
  "TRTLLM-Gen is designed mainly considering support of sm100/103. I believe we have no plan to SM120/121 so far. cc @PerkzZheng for trtllm-gen related queries, @peaceh-nv for SM120 related issues and @laikhtewari for feature request."),
 ("https://github.com/NVIDIA/TensorRT-LLM/issues/11799",
  "Closing per @pengbowang-nv's note above: trtllm-gen targets sm100/103 with no plan for SM120/121, and the XQA / FMHA_v2 fallback covers consumer Blackwell. Thanks for the clear answer."),
 ("https://github.com/NVIDIA/TensorRT-LLM/issues/5581",
  "A new kernel have to be implemented in order to enable FP8 block scaling GEMM on SM120 but unfortunately it's not on the priority because of VRAM capacity and bandwidth concern."),
 ("https://github.com/NVIDIA/TensorRT-LLM/issues/17052",
  "Hi @jiuzhuanzhuan Currently, TRTLLM does not support Gemma NVFP4 on SM120."),
 ("https://github.com/NVIDIA/TensorRT-LLM/issues/1140",
  "We\u2019ve shared the information with the team, and while there are no immediate plans to support it, we\u2019re always open to contributions."),
 ("https://github.com/NVIDIA/TensorRT-LLM/issues/13582",
  "Closing as not planned for now. Please open a new issue if this is still needed on the latest version"),
 ("https://github.com/NVIDIA/TensorRT-LLM/issues/2663",
  "Closing as not planned for now. Please open a new issue if this is still needed on the latest version"),
 ("https://github.com/NVIDIA/TensorRT-LLM/issues/2765",
  "At this moment, there is no immediate plan to implement DualPipe parallelism."),
 ("https://github.com/NVIDIA/TensorRT-LLM/issues/24",
  "We do not plan to publish performance numbers that compare TensorRT-LLM with vLLM."),
 ("https://github.com/NVIDIA/TensorRT-LLM/issues/74",
  "For the moment, we do not plan to open-source the batch manager."),
 ("https://github.com/NVIDIA/TensorRT-LLM/issues/112",
  "We do not plan to update it in a near future."),
 ("https://github.com/NVIDIA/TensorRT-LLM/issues/190",
  "we do not plan to make \u201cbuilding outside a Docker container\u201d one of our recommended solutions in a near future."),
 ("https://github.com/NVIDIA/TensorRT-LLM/issues/6805",
  "Closing this out. FP8 block-scale MoE still requires SM90 or SM120, so the `-FP8` checkpoint won't run on L40 (SM89) as-is."),
 ("https://github.com/NVIDIA/TensorRT-LLM/pull/17570",
  "Fallbacks will soon be deprecated. Users should explicitly set CUTLASS rather than relying on fallbacks provided by TRTLLM."),
 ("https://github.com/NVIDIA/TensorRT-LLM/pull/18866",
  "Note that those Triton+Executor examples were removed with the TensorRT backend, and point readers to live Disaggregated Serving and examples/disaggregated ."),
 ("https://github.com/NVIDIA/TensorRT-LLM/pull/18471",
  "trtllm-eval docs still tell users to pass --backend tensorrt , but the CLI only accepts pytorch ."),
 ("https://github.com/NVIDIA/TensorRT-LLM/issues/8223",
  "I\u2019m closing this issue as stale, assuming the comments above have addressed your question."),
]


def check(item):
    url, q = item
    try:
        t = s12_gh.main(url)
    except Exception as e:
        return url, q, "FETCH-ERR:" + str(e)[:60]
    ok = q in t
    if not ok:
        # try whitespace-normalized fallback
        import re
        n = lambda s: re.sub(r"\s+", " ", s)
        ok = n(q) in n(t)
        if ok:
            return url, q, "OK-NORMWS"
    return url, q, "OK" if ok else "**MISS**"


if __name__ == "__main__":
    with cf.ThreadPoolExecutor(8) as ex:
        for url, q, r in ex.map(check, ITEMS):
            print(f"{r}\t{url}\t{q[:90]}")
