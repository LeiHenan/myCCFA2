#!/usr/bin/env python3
"""E1 仪器化补丁：为 vLLM 打入未提交投机 KV 的记账探针。

用法：
    python e1_patch.py --apply            # 打补丁（幂等）
    python e1_patch.py --revert           # 撤销
    python e1_patch.py --status           # 查看当前状态

锚点基于本地快照 `prior_art_fetch/vllm.tar.gz`（vllm-main，version "dev"）。
若你的 pin 不同，本脚本会在锚点缺失时报错并打印期望的源码片段，便于手工对齐。

两个注入点：
  A. vllm/v1/core/sched/scheduler.py —— 每步每请求的 draft/accepted/rejected 与回滚后的 ctx
  B. vllm/v1/core/kv_cache_manager.py —— allocate_slots 的请求 token 数与实际块数（预留口径）

日志（JSONL，一行一条）写到 $E1_LOG；未设置则完全不产生开销。
"""

import argparse
import importlib.util
import re
import sys
from pathlib import Path

MARK_BEGIN = "# --- E1-INSTRUMENT-BEGIN ---"
MARK_END = "# --- E1-INSTRUMENT-END ---"

HELPER = f'''
{MARK_BEGIN}
def _e1_emit(rec):
    """E1 探针：把一条记账记录追加到 $E1_LOG（未设置则静默返回）。"""
    import os
    if not os.environ.get("E1_LOG"):
        return
    try:
        import json
        import threading
        import time
        rec["ts"] = time.time()
        rec["pid"] = os.getpid()
        lock = globals().get("_E1_LOCK")
        if lock is None:
            lock = threading.Lock()
            globals()["_E1_LOCK"] = lock
        with lock:
            with open(os.environ["E1_LOG"], "a", encoding="utf-8") as fh:
                fh.write(json.dumps(rec, separators=(",", ":")) + "\\n")
    except Exception:
        pass
{MARK_END}
'''

HOOK_A_ANCHOR = """                    if request.num_output_placeholders > 0:
                        request.num_output_placeholders -= num_rejected"""

HOOK_A_INSERT = HOOK_A_ANCHOR + f"""
{MARK_BEGIN}
                    _e1_emit(
                        {{
                            "ev": "spec",
                            "req": req_id,
                            "ctx": request.num_computed_tokens,
                            "draft": num_draft_tokens,
                            "accepted": num_accepted,
                            "rejected": num_rejected,
                            "invalid": (scheduler_output.num_invalid_spec_tokens or {{}}).get(req_id, 0),
                        }}
                    )
{MARK_END}"""

HOOK_B = f'''

{MARK_BEGIN}
def _e1_install_alloc_hook():
    """包装 KVCacheManager.allocate_slots，记录每次分配的请求 token 数与实际块数。"""
    try:
        from vllm.v1.core.kv_cache_manager import KVCacheManager
    except Exception:
        return
    if getattr(KVCacheManager, "_e1_wrapped", False):
        return
    _orig = KVCacheManager.allocate_slots

    def _wrapped(self, request, num_new_tokens, *args, **kwargs):
        out = _orig(self, request, num_new_tokens, *args, **kwargs)
        try:
            blocks = None
            if out is not None:
                ids = out.get_block_ids()
                blocks = len(ids[0]) if ids else 0
            _e1_emit(
                {{
                    "ev": "alloc",
                    "req": request.request_id,
                    "new_tokens": int(num_new_tokens),
                    "blocks": blocks,
                    "ctx": getattr(request, "num_computed_tokens", None),
                }}
            )
        except Exception:
            pass
        return out

    KVCacheManager.allocate_slots = _wrapped
    KVCacheManager._e1_wrapped = True


_e1_install_alloc_hook()
{MARK_END}
'''


def vllm_root() -> Path:
    spec = importlib.util.find_spec("vllm")
    if spec is None or not spec.origin:
        sys.exit("找不到已安装的 vllm —— 请在装了 vllm 的环境里运行（或先 pip install -e upstream/vllm）")
    return Path(spec.origin).parent


def _strip(text: str) -> str:
    return re.sub(
        re.escape(MARK_BEGIN) + r".*?" + re.escape(MARK_END) + r"\n?",
        "",
        text,
        flags=re.DOTALL,
    )


def _backup_path(p: Path) -> Path:
    """原文件备份路径：保证 --revert 能字节级还原。"""
    return p.with_name(p.name + ".e1orig")


def apply(root: Path) -> None:
    sched = root / "v1/core/sched/scheduler.py"
    kvm = root / "v1/core/kv_cache_manager.py"

    for p in (sched, kvm):
        if not p.exists():
            sys.exit(f"缺少目标文件：{p}")
        if MARK_BEGIN not in p.read_text(encoding="utf-8"):
            bak = _backup_path(p)
            if not bak.exists():
                bak.write_text(p.read_text(encoding="utf-8"), encoding="utf-8")

    t = sched.read_text(encoding="utf-8")
    if MARK_BEGIN in t:
        print(f"已是打过补丁的状态：{sched}")
    else:
        if t.count(HOOK_A_ANCHOR) != 1:
            sys.exit(
                "锚点缺失（Hook A）。请在你的 pin 上找到下面片段并手工插入 _e1_emit(...)：\n"
                + HOOK_A_ANCHOR
                + f"\n（当前出现 {t.count(HOOK_A_ANCHOR)} 次）"
            )
        t = t.replace(HOOK_A_ANCHOR, HOOK_A_INSERT, 1)
        # helper 追加到文件尾：只在运行时被调用，尾部定义同样安全，且不依赖类名/结构
        t = t.rstrip("\n") + "\n\n" + HELPER.strip() + "\n"
        sched.write_text(t, encoding="utf-8")
        print(f"已注入 Hook A：{sched}")

    t = kvm.read_text(encoding="utf-8")
    if MARK_BEGIN in t:
        print(f"已是打过补丁的状态：{kvm}")
    else:
        kvm.write_text(t + HOOK_B, encoding="utf-8")
        print(f"已注入 Hook B：{kvm}")


def revert(root: Path) -> None:
    sched = root / "v1/core/sched/scheduler.py"
    kvm = root / "v1/core/kv_cache_manager.py"
    for p in (sched, kvm):
        bak = _backup_path(p)
        if bak.exists():
            p.write_text(bak.read_text(encoding="utf-8"), encoding="utf-8")
            bak.unlink()
            print(f"已从备份字节级还原：{p}")
            continue
        t = p.read_text(encoding="utf-8")
        if MARK_BEGIN not in t:
            print(f"无需撤销：{p}")
            continue
        p.write_text(_strip(t), encoding="utf-8")
        print(f"已按标记撤销（无备份）：{p}")


def status(root: Path) -> None:
    for rel in ("v1/core/sched/scheduler.py", "v1/core/kv_cache_manager.py"):
        p = root / rel
        t = p.read_text(encoding="utf-8") if p.exists() else ""
        print(f"{'已打补丁' if MARK_BEGIN in t else '未打补丁'}  {p}")


if __name__ == "__main__":
    ap = argparse.ArgumentParser()
    g = ap.add_mutually_exclusive_group(required=True)
    g.add_argument("--apply", action="store_true")
    g.add_argument("--revert", action="store_true")
    g.add_argument("--status", action="store_true")
    a = ap.parse_args()
    root = vllm_root()
    print(f"vllm 安装路径：{root}")
    if a.apply:
        apply(root)
    elif a.revert:
        revert(root)
    else:
        status(root)
