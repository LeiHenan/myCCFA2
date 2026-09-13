#!/usr/bin/env python3
"""对已安装的 vLLM 打两个**可回滚**的小补丁，用于定位并干预"命中被取整"。

1) **打点**：在 `find_longest_cache_hit_per_group` 里打印每个组的 `alignment_tokens` 与返回的 `group_hit`；
2) **干预开关**：环境变量 `CCFA_ALIGN_OVERRIDE=<int>` 可覆盖 `alignment_tokens`
   （用于验证"命中是否被对齐取整归零"）。

用法（实例上）：
  python probes/p10-prefix-reuse-lag/patch_instrument.py apply     # 打补丁（先备份）
  python probes/p10-prefix-reuse-lag/patch_instrument.py revert    # 回滚
  python probes/p10-prefix-reuse-lag/patch_instrument.py status    # 查看状态
"""
import os
import shutil
import sys

TARGET = "/root/ccfa_venv/lib/python3.12/site-packages/vllm/v1/core/kv_cache_coordinator.py"
BACKUP = TARGET + ".ccfa-bak"

HELPER = '''
# ==== CCFA 插桩/干预（由 patch_instrument.py 注入，可回滚）====
import os as _ccfa_os
import sys as _ccfa_sys


def _ccfa_align(v):
    ov = _ccfa_os.environ.get("CCFA_ALIGN_OVERRIDE")
    if ov:
        v = int(ov)
    print(f"[T-ALIGN] alignment_tokens={v}", file=_ccfa_sys.stderr, flush=True)
    return v


def _ccfa_log_group(group_ids, group_hit):
    print(f"[T-GROUP] gids={group_ids} group_hit={group_hit}", file=_ccfa_sys.stderr, flush=True)
# ==== CCFA 结束 ====
'''

ANCHOR_ALIGN = "                alignment_tokens=self._cache_hit_alignment_tokens,"
ANCHOR_AFTER = "            for gid, blks in zip(group_ids, blocks):"


def report(status):
    print(f"[{status}] target={TARGET}")
    print(f"          backup={'存在' if os.path.exists(BACKUP) else '不存在'}")


def apply():
    src = open(TARGET, encoding="utf-8").read()
    if "_ccfa_align" in src:
        report("已打过补丁，跳过")
        return 0
    if ANCHOR_ALIGN not in src or ANCHOR_AFTER not in src:
        print("!! 锚点未命中，放弃（不改文件）")
        print("   align 锚点:", ANCHOR_ALIGN in src, " after 锚点:", ANCHOR_AFTER in src)
        return 1
    shutil.copy2(TARGET, BACKUP)
    out = src.replace(
        ANCHOR_ALIGN,
        "                alignment_tokens=_ccfa_align(self._cache_hit_alignment_tokens),",
    ).replace(
        ANCHOR_AFTER,
        "            _ccfa_log_group(group_ids, group_hit)\n" + ANCHOR_AFTER,
    )
    # 在第一处 class 定义前插入 helper（模块级）
    idx = out.index("class ")
    out = out[:idx] + HELPER + "\n\n" + out[idx:]
    open(TARGET, "w", encoding="utf-8").write(out)
    report("已打补丁")
    return 0


def revert():
    if not os.path.exists(BACKUP):
        print("!! 没有备份，无法回滚")
        return 1
    shutil.move(BACKUP, TARGET)
    report("已回滚")
    return 0


if __name__ == "__main__":
    cmd = sys.argv[1] if len(sys.argv) > 1 else "status"
    sys.exit({"apply": apply, "revert": revert, "status": lambda: report("状态") or 0}[cmd]())
