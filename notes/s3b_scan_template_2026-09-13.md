# S3b 已解决性核查 — 检索记录（手工模式）

> ⚠️ 手工模式**不产生证据**：下面的每一行都必须由人/Agent 实际打开链接、读过正文后填写，
> 并把结论抄进 `35_solved_check.md` 的判定矩阵。**未读正文的行标『未取证』**。

**检索关键字**：`prompt lookup decoding`、`ngram draft corpus`、`speculative decoding draft cache`

**必查对象**：`vllm-project/vllm`、`sgl-project/sglang`、`NVIDIA/TensorRT-LLM`、`ggml-org/llama.cpp`

## 待查清单

| # | 对象 | 状态（已 ship / RFC / 论文 / 无人做） | URL | 查证深度（读了哪一节 / file:line） | 与我们的差异 |
|---|---|---|---|---|---|
| 1 | `vllm-project/vllm` | 待填 | 待填 | 待填 | 待填 |
| 2 | `sgl-project/sglang` | 待填 | 待填 | 待填 | 待填 |
| 3 | `NVIDIA/TensorRT-LLM` | 待填 | 待填 | 待填 | 待填 |
| 4 | `ggml-org/llama.cpp` | 待填 | 待填 | 待填 | 待填 |
| 5 | 会议论文（把关键字丢进检索式） | 待填 | 待填 | 待填 | 待填 |
| 6 | 生产实践（工程博客 / 论坛 / 用户抱怨） | 待填 | 待填 | 待填 | 待填 |

## 自检（提交前逐条核）

- [ ] 判定矩阵 ≥5 行，且覆盖 ≥3 个引擎/实现与 ≥2 个研究社区
- [ ] 每行都有**可点开的 URL** 与**查证深度**（未读正文的必须写『未取证』）
- [ ] 写了一条**反证**：主动找证据说明这个方向可能已被解决
- [ ] 写明了『若被占，降级成什么』且降级不需要新半径
