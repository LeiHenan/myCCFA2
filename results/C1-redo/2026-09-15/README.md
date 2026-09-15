# data/ — C1 重做（A800 / vLLM 0.29.0 / 2026-09-15）

**本目录按 `results/README.md` 的规矩分层**：原始逐请求记录体积大（6.3 MB）⇒ **压缩入库**（392 KB，可还原）；
派生判定体积小、且报告直接引用 ⇒ **明文入库**。

## 目录

```
raw/*.json.gz          # 26 份原始 launch 记录（含每臂逐请求 TTFT/解码段/文本）
derived/               # 判定（明文，报告直接引用）
├── analysis_*.json        # 噪声底 / P1' / P2' / P3' 与逐格数字
├── acceptance_*.json      # 接受率对照（PC on/off、冷/暖）
├── manifest_*.json        # 每个网格的启动顺序与每格结局
└── workload_probe.json    # 工作负载有效性（drafter 接受率非零的证明）
instrument/            # 探针（可重跑）
run/                   # 四个网格的终端日志（gzip）
```

## 还原原始记录

```bash
gunzip -k raw/*.json.gz        # 得到同名 .json
```

自检（任取一份应能解析出 run_id / engine_status / arms）：

```bash
gunzip -dc raw/Qwen3.5-4B_dflash_k3_pc1.json.gz | python3 -c \
  "import json,sys; d=json.load(sys.stdin); print(d['run_id'], d['engine_status'], len(d['arms']))"
```

## 完整性

四份 manifest 合计声明 **28** 个 run，`raw/` 有 **26** 份文件 —— 差额是 `core-dense` 与 `k-sens`
**共用了 2 个 run_id**（K 敏感性网格的 K=3 档重跑了 core-dense 已有的两格并覆盖同名文件）。
**这不是巧合而是当时 `run_id` 不含网格前缀的缺陷，已修；影响范围、复算核对与结论判定见
[`CORRECTION.md`](CORRECTION.md)。** 四份 manifest 分别是 `core-dense`(8) / `core-hybrid`(8) /
`k-sens`(8) / `falcon`(4)。

## 每份原始记录的字段

| 字段 | 含义 |
|---|---|
| `arms[]` | 每个臂一条：`tag` 形如 `cold\|c8\|s3`（阶段\|并发\|seed） |
| `arms[].decode_tps` | **主量**：输出 token ÷ 解码段墙钟（排除预填充） |
| `arms[].ttft_p50` | 预填充侧，单独报 |
| `arms[].cache_hits_delta` | 引擎自带 `vllm:prefix_cache_hits_total` 的臂前后差分（**命中仪器**） |
| `arms[].metrics_delta` | 该臂的全部引擎计数差分（含 `spec_decode_*`） |
| `arms[].per_request[]` | 逐请求：TTFT、解码段、chunk 数、usage |
| `correctness[]` | 正确性臂：同实例冷/暖两遍的**生成文本**（逐请求） |
| `engine_status` | `OK` / `FAILED`——**装置失败与"效应为零"严格分开** |

## 能回答什么 / 不能回答什么

**能**：dense 与 hybrid 两架构上，`{none,ngram,eagle3,dflash} × PC{on,off} × 并发{1,8} × 5 seed`
的解码期吞吐、接受率、命中记账、以及同实例冷/暖的贪心输出一致率；K∈{1,3,5,7} 的敏感性。

**不能**：**MTP 路径**（本轮未测——该模型有内建 MTP head，是我的遗漏，不是不可测）；
`mamba_cache_mode` 的 align/none 对照（它是 `enable_prefix_caching` 的函数，无法 pin）；
27B/80B scale 点；跨引擎结论。
