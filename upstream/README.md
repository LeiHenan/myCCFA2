# upstream/ — 引擎检出（不入库）

实验需要可改的 vLLM / SGLang 源码。本目录用于放检出，**内容不入库**（见 `.gitignore`）。

## 已有的本地快照（只读参考，勿在此目录改）

| 路径 | 内容 |
|---|---|
| `archive/prior_art/srcfull/sglang-main/` | SGLang 完整源码树（含 `python/sglang/srt/...`，本方案的 `unified_tree_core.py:402` 引用即出自此树） |
| `archive/prior_art/sglang2/` | 第二份 SGLang 树 |
| `archive/c2prior/vllm_src/` | vLLM 源码片段（含 tests/fixtures） |
| `prior_art_fetch/vllm.tar.gz`（~41 MB） | vLLM 源码快照 tar |
| `prior_art_fetch/sglang.tar.gz`（~10 MB） | SGLang 源码快照 tar |

> 这些是**冻结证据**，用于核对引用（行号、默认值、开关语义）。**不要修改**，否则审计链断裂。

## 建议用法

```bash
# 例：在 upstream/ 下检出可改的工作副本
git clone https://github.com/vllm-project/vllm.git upstream/vllm
cd upstream/vllm && git checkout <pin>
```

## 必须记录的 pin

每次跑探针时，把以下信息写进 `results/<probe>/<date>/summary.md`：

- 引擎版本 / commit hash
- 是否含关键 PR：vLLM `#53614`（MERGED 2026-09-06）、`#55760`（MERGED 2026-09-08）、`#50172` / `#56142` / RFC `#55697` + `#55873-6`（均 OPEN）
- 模型与 drafter 家族及层数（见 `probes/p06-frontier/README.md` §1）
- 关键开关：`mamba_cache_mode`、`prefix_cache_retention_interval`、`num_speculative_tokens`

## 数据

trace 数据库与 DuckDB 二进制在 `archive/data/`（`syfi_coding_trace.duckdb` 160 MB；`duckdb` 117 MB；查询脚本 `q1.sql`–`q6.sql`），同样不入库。
