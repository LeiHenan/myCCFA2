#!/usr/bin/env python3
"""p25（零 GPU）—— 判定「FlashInfer 禁用 cascade attention」在我们负载上**到底有没有代价**。

背景（**我已在安装树亲验**，`vllm/v1/attention/backends/flashinfer.py`）：
    def use_cascade_attention(self, *args, **kwargs) -> bool:
        if self.kv_cache_spec.dtype != self.vllm_config.model_config.dtype: return False
        # TODO: Cascade attention doesn't work, disable it for now
        # return use_cascade_attention(*args, **kwargs)
        return False
而**其它后端都实现了它**（`flash_attn.py:1665` 的 `use_cascade_attention(...)` 是真正的启发式）。
上游状态（S3b，独立检索）：引入禁用的 PR **#26130 MERGED**；其原因是 **#25679（CI 测试失败，正文写明
"there's a correctness issue with cascade attention on the FlashInfer backend"）被 CLOSED / NOT_PLANNED**；
早先给 FlashInfer 加 cascade 的 **#8132 CLOSED 未 merge** ⇒ **没人正在修**。

**本脚本要回答的**：把 FlashInfer 改成"没被禁用"的话，启发式**会不会真的启用 cascade**？
若启发式本身就会返回 False ⇒ 这条禁用**对我们零代价**，候选当场作废（§0.6：先量上界，再决定开卡）。
若返回 True ⇒ 禁用**确实吃掉了一个本该生效的共享前缀 prefill 优化**，才值得花 GPU 测量级。

用法： check_cascade_heuristic.py [--common-prefix 4096] [--n-reqs 24] [--query-len 4096] [--sms 188]
        check_cascade_heuristic.py --selftest
"""
from __future__ import annotations
import argparse, sys
import numpy as np

# Qwen3-4B 的真实形状（本工作区 ledger 记录：hidden 2560 / 32 Q heads / 8 KV heads / 36 层）
QWEN3_4B = dict(num_query_heads=32, num_kv_heads=8)


def call(common_prefix, query_lens, num_sms, **kw):
    from vllm.v1.attention.backends.flash_attn import use_cascade_attention as h
    return h(
        common_prefix_len=common_prefix,
        query_lens=np.asarray(query_lens, dtype=np.int32),
        num_query_heads=kw.get("num_query_heads", 32),
        num_kv_heads=kw.get("num_kv_heads", 8),
        use_alibi=kw.get("use_alibi", False),
        use_sliding_window=kw.get("use_sliding_window", False),
        use_local_attention=kw.get("use_local_attention", False),
        num_sms=num_sms,
        dcp_world_size=kw.get("dcp_world_size", 1),
    )


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--common-prefix", type=int, default=4096)
    ap.add_argument("--n-reqs", type=int, default=24)
    ap.add_argument("--query-len", type=int, default=4096)
    ap.add_argument("--sms", type=int, default=188)
    ap.add_argument("--selftest", action="store_true")
    a = ap.parse_args()
    if a.selftest:
        return selftest()

    ql = [a.query_len] * a.n_reqs
    print(f"负载参数：common_prefix_len={a.common_prefix}  请求数={a.n_reqs}  每请求 query_len={a.query_len}  "
          f"Q/KV heads={QWEN3_4B['num_query_heads']}/{QWEN3_4B['num_kv_heads']}  num_sms={a.sms}")
    print("（我们 p20/p21/p22 的负载正是：24 个请求共享 4096-token 前缀 ⇒ 远超 256 的阈值）\n")

    cases = [
        ("我们的负载（共享 4096 前缀 × 24 请求）", dict(common_prefix=a.common_prefix, query_lens=ql)),
        ("对照：前缀只有 128 token（<256 阈值）", dict(common_prefix=128, query_lens=ql)),
        ("对照：只有 4 个请求（<8 阈值）", dict(common_prefix=a.common_prefix, query_lens=[a.query_len] * 4)),
        ("对照：用了 sliding window", dict(common_prefix=a.common_prefix, query_lens=ql, use_sliding_window=True)),
        ("对照：用了 alibi", dict(common_prefix=a.common_prefix, query_lens=ql, use_alibi=True)),
        ("对照：DCP>1（多卡）", dict(common_prefix=a.common_prefix, query_lens=ql, dcp_world_size=2)),
    ]
    verdicts = {}
    for name, kw in cases:
        try:
            r = call(num_sms=a.sms, **kw)
        except Exception as exc:  # noqa: BLE001
            r = f"ERR {type(exc).__name__}: {exc}"
        verdicts[name] = r
        print(f"  {name:44s} ⇒ use_cascade = {r}")

    ours = verdicts[cases[0][0]]
    print()
    if ours is True:
        print("  ⇒ **我们负载上启发式会启用 cascade** ⇒ FlashInfer 的禁用**确实吃掉了一个本该生效的优化**")
        print("     ⇒ 值得花 GPU 测量级（FLASH_ATTN[cascade 活] vs FLASHINFER[cascade 死]，共享前缀负载）。")
    elif ours is False:
        print("  ⇒ **我们负载上启发式本来也会返回 False** ⇒ 该禁用对我们**零代价** ⇒ 候选当场作废（§0.6）。")
    else:
        print("  ⇒ 调用失败，结论不可用；请检查参数或环境。")
    return 0


def selftest() -> int:
    ql = [4096] * 24
    r_long = call(4096, ql, 188)
    r_short = call(128, ql, 188)
    r_few = call(4096, [4096] * 4, 188)
    assert r_short is False, f"前缀 <256 必须是 False，得到 {r_short}"
    assert r_few is False, f"请求数 <8 必须是 False，得到 {r_few}"
    assert isinstance(r_long, (bool, np.bool_)), f"长前缀应返回 bool，得到 {type(r_long)}"
    print(f"selftest OK —— 阈值行为正确（<256⇒{r_short}, <8请求⇒{r_few}, 我们的负载⇒{r_long}）")
    return 0


if __name__ == "__main__":
    sys.exit(main())
