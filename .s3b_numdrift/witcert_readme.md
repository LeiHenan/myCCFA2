# witcert-kv-certificates

Artifacts, guards, and machine-checked proofs for the paper
**"WitCert: Sound Runtime Risk Observability and Gating for KV-Cache
Quantization"** — Fanzhe Wei, Li Liu, Ziyang Wang, Chenyu Wang,
[arXiv:2607.28699](https://arxiv.org/abs/2607.28699).

## The four-paper series

| | Paper | Paper link | Artifact |
|---|---|---|---|
| **P1** | WitCert: Sound Runtime Risk Observability and Gating for KV-Cache Quantization | [arXiv:2607.28699](https://arxiv.org/abs/2607.28699) | [witcert-kv-certificates](https://github.com/metask-ai/witcert-kv-certificates) ← **this repository** |
| **P2** | Runtime Observability for Heterogeneous Attention Memory | [arXiv:2608.05863](https://arxiv.org/abs/2608.05863) | [witprobe-attention-memory](https://github.com/metask-ai/witprobe-attention-memory) |
| **P3** | Pricing the Risk of Runtime Compression: Anytime-Valid Admission and a Served-Output Law for Compressed Serving State | [arXiv:2608.15810](https://arxiv.org/abs/2608.15810) | [witcert-w-certified-precision](https://github.com/metask-ai/witcert-w-certified-precision) |
| **P4** | What to Protect When You Quantize a Mixture of Experts | not yet public | not yet public |

These four papers are one line of work, not four topics. **P1** asks whether
compression is damaging *the request being served right now*, and answers it
for the KV cache with a provably sound runtime meter and meter-driven gating.
**P2** carries the same question to the fact that a modern model's memory is no
longer a plain KV cache — latent caches, learned sparse selectors and recurrent
states each fail differently — and gives one observability contract for all
four classes. Together they answer *whether it can be measured*. **P3** asks
what the measured risk is worth and how to spend it: the union budget those
systems rely on exhausts on every long production request, and what replaces it
is an anytime-valid ledger, a law carrying the certified witness to the served
output, and the quantifier that makes the bound hold on a request never seen.
**P4** asks the converse — *what that machinery should be pointed at* — and
prices the field's shared instinct that MoE routing invariance must be
protected, finding it wrong in three independent ways.

Each paper stands alone: P3 and P4 inherit the typed-contract vocabulary of P1
and P2 but restate no result of theirs, and neither claims the other's.

Built for lowest-cost verification: **every number in the paper
regenerates from the frozen artifacts shipped here, and a claim guard
fails the build on any mismatch.**

## One-command reproduction

```bash
pip install -r requirements.txt   # matplotlib + mpmath; L0 needs stdlib only
bash reproduce.sh                 # L0 numbers -> L1 figures -> L2 gates, single verdict at the end
```

Tested environment: Python 3.9.6 / matplotlib 3.9.4 on macOS and Linux;
no GPU, no network access, no model weights required. The Lean layer is
pinned by `formal/lean-toolchain` and run separately (see below).

## 30-second verification (no GPU, Python 3.9+ only)

```bash
python3 tools/make_canon.py          # regenerate the frozen-number canon from run artifacts
python3 tests/test_paper_claims.py   # every paper number must appear and match — or FAIL
```

`ALL PAPER CLAIM CHECKS PASSED` means every quantitative claim traces to
a JSON artifact under `experiments/out/` or `experiments/out_siteB/`.

## Layered reproduction

| Layer | What it verifies | Command | Needs | Time |
|---|---|---|---|---|
| L0 | every paper number ↔ artifact | `python3 tools/make_canon.py && python3 tests/test_paper_claims.py` | Python 3.9 | < 1 min |
| L1 | figures regenerate | `python3 tools/make_figs.py` | + matplotlib | < 1 min |
| L2 | certificate mathematics self-checks | `python3 tests/test_certificates.py && python3 tests/test_independent_crosscheck.py` | Python + mpmath | minutes |
| L3 | all Lean theorems, zero `sorry`, standard axioms only | `cd formal && bash check_all.sh` | elan + Mathlib cache | ~30 min first run |
| L4 | serving-stack integration reruns | — | private patches + GPUs | see boundary below |

## What is here / not here

Here: frozen artifacts for every cited number, the canon/claim/figure
generators, the certificate self-check suites, and the complete Lean
development (three standard axioms only).

Not here: the measurement-platform implementation and the serving-stack
patches (they modify a third-party numeric path); their measured outputs
are shipped, and the paper states this boundary explicitly.

## Requirements

- Python ≥ 3.9; L1 needs `matplotlib`; L2 needs `mpmath`
- Lean 4 via `elan` for L3

## License

Apache-2.0 (see `LICENSE`). Please cite
[arXiv:2607.28699](https://arxiv.org/abs/2607.28699); machine-readable
metadata is in `CITATION.cff`.
