# Prior-art search: is there a LAW / BOUND / PROBABILITY MODEL for LLM greedy-argmax token FLIPS under numerical perturbation?

Scope: focused prior-art search only. Date of search: 2026-09-14. All files created in
`/Users/leihenan/Desktop/myProject/.s3b_numdrift/subA/` with prefix `fliplaw_`.

**Verification discipline used:** every arXiv ID below was fetched at `https://arxiv.org/abs/<id>` and the
`<title>` was read back; PDF/HTML bodies were then fetched and grepped. Quotes marked *[extracted]* come from
PDF text extraction and may contain flattened math layout / spacing artifacts — flagged inline where relevant.

---

## 1. Evidence table

| # | Exact URL | Title I actually read back | State | VERBATIM quote | DECISION or SOFT? |
|---|---|---|---|---|---|
| 1 | https://arxiv.org/abs/2506.09501 | `[2506.09501] Understanding and Mitigating Numerical Sources of Nondeterminism in LLM Inference` | **VERIFIED** (abs `<title>` + `citation_title` meta + authors Yuan, Li, Ding, Xie…) | "However, such fluctuations do not always lead to a token flip, since a flip only occurs when the variation at a specific position exceeds the original top-1 and top-2 probability gap, which is inherently a stochastic event." | **DECISION** — LLM-specific. A necessary condition for a token flip (`variation > top1−top2 gap`). No closed-form probability; the paper's own headline metric (`Avg_Std@top1_prob`) is SOFT. |
| 2 | https://arxiv.org/html/2506.09501v2 | same paper, §4 body | VERIFIED (body text fetched, 541,987 bytes) | "Figure 3 shows an example of token logits when two runs diverge, illustrating how numerical precision errors can flip the order of the top-1 and top-2 token probabilities." | DECISION (mechanism statement), LLM-specific. |
| 3 | https://arxiv.org/abs/2605.30218 | `[2605.30218] MarginGate: Sparse Margin-Triggered Verification for Batch-Invariant LLM Inference` | **VERIFIED** (abs + full arXiv HTML read; authors Kexin Chu, Yang Zhou, Wei Zhang) | "The threshold $\tau$ is a deployment parameter for each model and serving configuration. The standard argmax perturbation bound  ( Thinking Machines Lab, 2026 ; Yuan et al., 2025 ) supplies a practical starting point: if two batch-shaped logit vectors differ by at most $\varepsilon_{\mathrm{pert}}$ in $\ell_{\infty}$ , their argmaxes can diverge only when the relevant margin is about $2\varepsilon_{\mathrm{pert}}$ ." | **DECISION** — LLM-specific deterministic bound, but explicitly presented as a *transferred/standard* bound (no derivation, no probability). This is the single closest statement to the target form. |
| 4 | https://arxiv.org/html/2605.30218v1 | same paper, §2.5 / §3.1 | VERIFIED | "The current logit margin provides a deployable risk signal because divergent steps are enriched for near-tied top candidates." | DECISION (empirical risk signal), LLM-specific. |
| 5 | https://proceedings.iclr.cc/paper_files/paper/2026/hash/0a48036026dc7946ef6033ae14719cc5-Abstract-Conference.html | `How Stable is the Next Token? A Geometric View of LLM Prediction Stability` (Deyuan Liu, Zecheng Wang, Zhanyue Qin, Zhiying Tu, Dianhui Chu, Dianbo Sui — ICLR 2026) | **VERIFIED** (official ICLR proceedings abstract page; PDF also fetched, 38 pp.) | "We introduce the Token Constraint Bound (δTCB), a novel metric that quantifies the maximum internal state perturbation an LLM can withstand before its dominant next-token prediction significantly changes." | **SOFT (decision-motivated)** — must be labelled honestly. Despite the abstract's "dominant next-token prediction" framing, the formal Definition 1 bounds the **probability vector**, not the argmax: "quantifies the L2-norm radius of the largest hyper-sphere of perturbations ∆h around the current hidden state h that, to a first-order approximation, guarantees the change in the output probability vector o remains within ϵ." |
| 6 | https://proceedings.iclr.cc/paper_files/paper/2026/file/0a48036026dc7946ef6033ae14719cc5-Paper-Conference.pdf | same paper (official PDF) | VERIFIED (PDF fetched, `fliplaw_iclr_stable.pdf`) | "Deﬁnition 1 (Token Constraint Bound δTCB) . Given the output weight matrix W, hidden state h, resulting output distribution o = softmax(Wh), and a tolerance ϵ > 0 for the maximum L2 change allowed in o, the Token Constraint Bound δTCB at state h is deﬁned as: δTCB(h) := ϵ ∥JW(h)∥F" | SOFT (L2 change of `o`), LLM-specific. Correlates with `ztop1 − ztop2` (reported r = 0.62 in the low-Veff regime) but is not a flip identity/probability. |
| 7 | https://arxiv.org/abs/2406.13187 | `[2406.13187] Decouple then Converge: Handling Unknown Unlabeled Distributions in Long-Tailed Semi-Supervised Learning` | **VERIFIED** (abs `<title>`; PDF metadata `/Title` and p.1 header both match; authors Kai Gan, Tong Wei, Min-Ling Zhang, IEEE TPAMI format) | "Lemma 6(Argmax robustness to log- prior perturbation)." … "If∆ dda t (x;πu)>2τ 2δ(e) t , then the argmax under˜πt equals that underπ u." — with "δ (e) t :=∥log ˜πt−logπ u∥∞" and "∆" defined as the margin. *[extracted, math layout flattened]* | **DECISION** — a genuine deterministic argmax-stability bound of the exact form `margin > 2·(perturbation scale) ⇒ argmax unchanged`. **NOT LLM-SPECIFIC: CLASSICAL/TRANSFERRED** (long-tailed semi-supervised learning, TPAMI). |
| 8 | https://arxiv.org/pdf/2406.13187v2 | same paper, Remark after Lemma 6 | VERIFIED | "Remark: Lemma 6 is a conservative robustness statement (prevents catastrophic flips on high-margin points). We donotclaim it proves performance gain by itself." | DECISION, classical/transferred. Authors themselves flag it as a conservative bound, not a rate/law. |
| 9 | https://arxiv.org/abs/2405.14064 | `[2405.14064] Building a stable classifier with the inflated argmax` (Soloff, Barber, Willett) | **VERIFIED** (abs `<title>` + `citation_author` metas + NeurIPS PDF) | "Theorem 9. For anyε> 0 and anyv,w ∈ RL, if∥v−w∥<ε then argmaxε(v)∩argmaxε(w)̸= ∅." *[extracted]* | **DECISION** — but for a *relaxed, set-valued* argmax (`ε`-compatible selection rule), not the plain argmax. **CLASSICAL/TRANSFERRED** (statistics, NeurIPS 2024). |
| 10 | https://papers.nips.cc/paper_files/paper/2024/file/822621aaa88635437ea51023afdeaec2-Paper-Conference.pdf | same paper (official NeurIPS PDF) | VERIFIED | "sε margin(w) ={j :wj > max ℓ wℓ−ε/ √ 2}, (3)" | DECISION helper rule (fixed-margin selection). Classical/transferred. Shows the `ε/√2` margin threshold form exists in the literature. |
| 11 | https://arxiv.org/abs/2608.13756 | `[2608.13756] The Integer Alibi: Localizing Cross-Kernel Divergence in INT8-Quantized LLM Inference` | **VERIFIED** (abs `<title>` + abstract read back) | "Teacher-forced replay ties layers to tokens: flips concentrate at small logit margins, which predict flip risk with ROC-AUC 0.94 on 16,384 positions." | **DECISION** — LLM-specific. Empirical **predictive** relationship (margin → flip risk, AUC 0.94). Not a law/bound; no functional form given in the abstract. |
| 12 | https://arxiv.org/abs/2608.11212 | `[2608.11212] Detecting a Route Flip Is Easier Than Knowing Whether to Fix It: Causal Route-Mediated Damage in Quantized Mixture-of-Experts` | **VERIFIED** (abs `<title>` + abstract read back) | "The deployable router margin detects that a flip occurred (AUC 0.772) but cannot tell a harmful flip from a helpful one (at chance)" | **DECISION** — LLM-specific (top-k MoE expert routing flips). Empirical AUC, plus an explicit negative result on predicting flip harm. |
| 13 | https://github.com/vllm-project/vllm/pull/55944 | `[Docs] Explain cross-mode numerical variance (eager vs compile vs CUDA graphs) by spa5k · Pull Request #55944 · vllm-project/vllm` | **VERIFIED** (HTML `<title>`; PR body extracted from embedded HTML) | "Measured basis (A100, latest main, data in #55238 ): argmax flips occurred only at positions with a top1-top2 margin below ~0.7. A Qwen2.5-7B control flipped only at its 0.125-margin position, while its sampled token (margin > 5) stayed stable across modes." | **DECISION** — LLM-specific, measured margin threshold for flips. Engineering artifact, not a law. |
| 14 | https://github.com/vllm-project/vllm/issues/55238 | `[Bug]: sm100 only (B200/B300): CUDA graph replay changes greedy output for gemma-4-26B-A4B-it with torch.compile off in both arms · Issue #55238 · vllm-project/vllm` | **VERIFIED** (HTML `<title>`; body extracted) | "Holding torch.compile off in both arms and changing only cudagraph_mode , the failure still reproduces on a B300: worst logprob delta 6.773 nats, argmax moves at 3/13 positions, next token 992 → 9079." | **DECISION** (argmax moves), LLM-specific measurement. Note: the issue **retracts** three of its original claims in an update — cite carefully. |
| 15 | https://arxiv.org/abs/1902.02918 | `[1902.02918] Certified Adversarial Robustness via Randomized Smoothing` | **VERIFIED** (abs `<title>` + PDF body read) | "Theng(x +δ) =cA for all‖δ‖2 <R , where R = σ 2 (Φ−1(pA)−Φ−1(pB)) (3)" *[extracted]* | **DECISION** (argmax/decision preserved inside certified radius R). **CLASSICAL/TRANSFERRED** — adversarial-robustness certification, not LLMs. This is the canonical classical "perturbation < radius ⇒ decision unchanged" bound, expressed via Gaussian CDFs. |
| 16 | https://web.stanford.edu/class/polisci350c/classonly/lecture1.pdf | No HTML `<title>`; read-back first page: `Mathematical Theory Likelihood References / Models of Discrete Choice / Jonathan Wand / Polisci 350C / Stanford University / April 10, 2006 (revised)` | **VERIFIED** (PDF fetched, 25 pp., text extracted) | "P(j, k ) = P(uj > uk ) = P(µj + ϵj > µ k + ϵk )" … "= Φ ( µj − µk √ σ2 j + σ2 k )" … and for extreme-value noise: "= 1 1 + exp{−(µ1 − µ2)}" *[extracted, math layout flattened]* | **DECISION** — this IS an exact closed-form **probability model for which alternative wins / flips** under additive noise (Thurstone discriminal process, 1927; probit and logit cases). **CLASSICAL/TRANSFERRED — psychometrics / discrete choice, NOT LLM.** I found no LLM paper that instantiates this for token logits. |
| 17 | https://arxiv.org/abs/2601.16880 | `[2601.16880] Theory of Minimal Weight Perturbations in Deep Networks and its Applications for Low-Rank Activated Backdoor Attacks` | **VERIFIED** (abs `<title>` + abstract read back; body not mined) | "These expressions reveal how back-propagated margins govern layer-wise sensitivity and provide certifiable guarantees on the smallest parameter updates consistent with a desired output shift." | DECISION-adjacent (output shift / class flip). **CLASSICAL/TRANSFERRED** robustness theory applied to precision/compression. |
| 18 | https://arxiv.org/abs/2605.23955 | `[2605.23955] From Accuracy to Auditability: A Survey of Determinism in Financial AI Systems` | **VERIFIED** (abs `<title>` + HTML body read) | "A flip rate mixes pipeline noise with distance to the decision threshold, so it is not an instrument for mechanical nondeterminism, and a tolerance written against it does not bound the serving stack." | **DECISION** — but about **classifier (GNN/credit-risk) flip rates**, not LLM token argmax. **CLASSICAL/TRANSFERRED**; useful mainly as a caution that "flip rate" is a noisy proxy. |
| 19 | https://arxiv.org/abs/2601.17768 | `[2601.17768] LLM-42: Enabling Determinism in LLM Inference with Verified Speculation` | **VERIFIED title only** (abs `<title>` read back; README fetched; paper body NOT read) | README only: "**LLM-42** enables deterministic LLM inference via a **decode–verify–rollback** protocol, without rewriting GPU kernels." | DECISION (systems-level per-token verification). **No flip law at README level.** Listed here because MarginGate/vLLM context depends on it; do not cite it for a bound. |

---

## 2. Verdict

**No genuine LLM-specific decision-level FLIP LAW / PROBABILITY MODEL was found.** Specifically:

1. **No closed-form `P(argmax flips) = f(margin, ‖δ‖)` for LLM tokens exists in what I could retrieve.**
   Nobody I found writes a flip probability as a function of the top1−top2 logit margin and the perturbation size.

2. **What does exist, and its correct label:**
   - **A deterministic margin-vs-perturbation bound, transferred/classical, restated for LLM logits.**
     MarginGate states it as *"the standard argmax perturbation bound"*: logits differing by ≤ ε in ℓ∞ can have
     divergent argmaxes **only if the relevant margin is about 2ε**. This is the classical result, **not an
     LLM-specific discovery**, and MarginGate itself gives **no derivation and no probability**. Its two cited
     sources are loose: I verified that the Thinking Machines Lab blog *"Defeating Nondeterminism in LLM
     Inference"* and the `batch_invariant_ops` README contain **no** argmax/margin bound (zero matches for
     "top-1", "logit gap", "margin"), so the bound's real traceable content is the Yuan et al. flip condition.
   - **The same bound in fully classical form** (independently verified, non-LLM): **Lemma 6** of arXiv
     2406.13187 — `margin > 2·(perturbation scale) ⇒ argmax unchanged` — and **Theorem 1** of Cohen et al.
     2019 — `‖δ‖₂ < R = (σ/2)(Φ⁻¹(p_A) − Φ⁻¹(p_B)) ⇒ decision unchanged`. Both are **CLASSICAL /
     TRANSFERRED** (semi-supervised learning; adversarial robustness), not LLM results.
   - **A genuine exact probability model for decision flips exists only classically**: the Thurstone
     discriminal process (probit: `Φ((µj−µk)/√(σj²+σk²))`; logit case: `1/(1+exp{−(µ1−µ2)})`). This is
     **CLASSICAL / TRANSFERRED (psychometrics / discrete choice, 1927)**. I found **no** LLM paper applying it
     to token-logit flips.
   - **LLM-specific evidence is empirical, not a law**: logit margin is a *measured* flip-risk predictor with
     reported AUC 0.94 (INT8 cross-kernel, arXiv 2608.13756) and AUC 0.772 (MoE route flips, arXiv 2608.11212),
     and a measured engineering threshold "flips occurred only at positions with a top1-top2 margin below ~0.7"
     (vLLM PR #55944). Yuan et al. give the necessary condition `flip ⟺ variation > top1−top2 gap`, calling it
     "inherently a stochastic event" — a *condition*, not a probability law.
   - **The one LLM paper with a formal "bound" framing is SOFT, not decision-level**: ICLR 2026 `δ_TCB` bounds
     the **L2 change of the output probability vector** (`‖Δo‖₂ ≤ ε`), not the argmax identity. Its abstract
     frames it as "before its dominant next-token prediction significantly changes", which is a **decision-level
     motivation over a soft formal object** — do not cite it as an argmax-flip bound.

3. **Bottom line for the parent agent:** the target statement is **partly anticipated as a transferred classical
   result** (deterministic margin bound; Thurstone gives the probability model classically) and is **empirically
   supported but not formalised for LLMs**. A paper that states and tests an explicit
   `P(token flip) = f(top1−top2 margin, perturbation magnitude)` for LLM logits would be **novel in its
   LLM-specific probability law**, but it must cite the classical margin bound (Lemma 6 form / Cohen et al.
   certified radius / Thurstone) as prior art and must not claim the deterministic `2ε` bound itself as new.

---

## 3. Could NOT verify (and why)

1. **`Greedy Decoding Is Not Precision-Invariant: Cross-Precision Output Divergence in LLM Inference`** —
   apparently an OpenReview submission (id `QDOKyg7a5e`). Both `https://openreview.net/forum?id=QDOKyg7a5e` and
   the `/pdf` route return **HTTP 200 but a 4,787-byte "Verifying your browser | OpenReview" challenge page**
   (confirmed: `<title>Verifying your browser | OpenReview</title>`). **I never read this paper and cannot vouch
   for its title** — the title above appears only in search-result snippets, and this project has a documented
   history of fabricated/wrong titles in search snippets. **NOT CITED as evidence.** It is the single most
   promising un-retrieved item: its title, if real, is directly on-target.
2. **`epsilon_to_flip_token`** — a search snippet attributed to arXiv 2605.10893 the definition
   *"Smallest perturbation magnitude along the gradient direction sufficient to change the argmax prediction"*.
   I verified the ID: `https://arxiv.org/abs/2605.10893` → `[2605.10893] Grounded or Guessing? LVLM Confidence
   Estimation via Blind-Image Contrastive Ranking`, and fetched its full HTML (2,230,277 bytes). **The strings
   `epsilon_to_flip_token` / `flip_token` do not appear**; the paper uses `flip_swap` (an image-swap behavioural
   diagnostic) instead. **Snippet unverified — likely a mismatched/fabricated search snippet.**
3. **`Give Me FP32 or Give Me Death? Challenges and Solutions for Reproducible Reasoning`** — `web_search`,
   `huggingface.co/papers/2506.09501`, and `ar5iv.labs.arxiv.org/html/2506.09501` all associate this title with
   **2506.09501**, but the read-back title at `https://arxiv.org/abs/2506.09501` is
   **`Understanding and Mitigating Numerical Sources of Nondeterminism in LLM Inference`** (same first author,
   Jiayi Yuan). **The title↔ID mapping is unverified/contradictory; I did not determine the real ID for
   "Give Me FP32 or Give Me Death?" and do not cite it.**
4. **The "Thinking Machines Lab, 2026" half of MarginGate's bound citation.** I verified the cited artifact
   (`batch_invariant_ops` README, 2,714 bytes, and the blog `Defeating Nondeterminism in LLM Inference`, title
   read back at `https://thinkingmachines.ai/blog/defeating-nondeterminism-in-llm-inference/`). **Neither contains
   any argmax/margin perturbation bound** (zero grep matches for "margin", "top-1", "logit gap", "argmax"). The
   attribution is therefore **unverified as a source of the bound**.
5. **`api.semanticscholar.org/graph/v1/paper/search`** — repeatedly returned
   `{"message": "Too Many Requests..."} , "code": "429"` and one 60 s timeout. **No Semantic Scholar-based
   discovery was possible**, so my recall may be incomplete for venues not indexed by the search tool used.
6. **`openreview.net`** (forum + pdf) — browser challenge as described. **`ieeexplore`** — not attempted after
   the documented empty-return behaviour. `export.arxiv.org` (429) and `arxiv.org/search` (406) were avoided as
   instructed; all arXiv verification went through `arxiv.org/abs/<id>` and `arxiv.org/pdf/<id>`.
7. **ICLR 2026 `δ_TCB` has no verified arXiv ID.** I read the official proceedings abstract page and the
   official conference PDF, but **did not locate or verify an arXiv preprint** of it. Cite the ICLR 2026
   proceedings URLs.
8. **`The Integer Alibi` (2608.13756) and `MoE route flip` (2608.11212)**: **abstracts only.** The ROC-AUC
   figures (0.94 / 0.772) are quoted from their abstracts; I did not read their methods sections, so I cannot
   report the functional form or the evaluation protocol behind those numbers.
9. **`LLM-42` (2601.17768)**: title verified; **paper body not read** (README only). Do not cite it for any
   bound.
10. **`EmergentMind` "Token-Flipping Phenomenon in LLMs"**
    (https://www.emergentmind.com/topics/token-flipping-phenomenon) — fetched (353,287 bytes) but it is about a
    **different sense of "token flipping"** (adversarial control-token polarity flips, LLM-as-a-judge decision
    flips), **not** numerical-perturbation argmax flips. Marked irrelevant; not used.

---

## 4. Exact search queries run

`web_search` queries (each 3-query batch run as shown):

1. `probability argmax flips under perturbation logit margin bound` / `logit margin top1 top2 gap argmax stability LLM quantization token flip` / `margin-based bound probability of argmax change perturbation epsilon`
2. `argmax stability margin perturbation bound theorem gamma epsilon` / `LLM greedy decoding token flip numerical precision BF16 top-1 agreement` / `quantization induced token flips LLM greedy argmax change rate`
3. `"Greedy Decoding Is Not Precision-Invariant" cross-precision output divergence` / `"margin" "flip" probability bound top-1 prediction perturbation classifier` / `batch-invariant LLM inference logit margin verification argmax`
4. `token flip rate logit margin LLM quantization top-1 disagreement` / `"decision margin" LLM token argmax robustness bound` / `probability argmax changes gaussian noise logit gap closed form`
5. `"Greedy Decoding Is Not Precision-Invariant" arXiv` / `"Give Me FP32 or Give Me Death" arXiv reproducible reasoning` / `Lemma argmax preserved logit gap exceeds perturbation norm theorem`
6. `margin bound probability top-1 changes under perturbation softmax logits noise` / `"flip" "margin" bound argmax LLM non-determinism batch invariance theory` / `Ouyang batch invariance LLM thinking machines logit margin token disagreement`
7. `"How Stable is the Next Token" geometric view LLM prediction stability arXiv` / `arxiv 2607.15467 defended teacher argmax` / `"token flip" probability model logit margin LLM`
8. `"How Stable is the Next Token" Sui geometric prediction stability ICLR 2026 abstract` / `logit margin bound probability argmax flip epsilon perturbation theorem LLM 2026` / `emergentmind token flipping phenomenon LLMs margin`
9. `LLM-42 per-token verification deterministic inference Gond 2026` / `arXiv 2601.00781 token stability` / `quantization top-1 agreement rate logit gap theoretical bound LLM weights perturbation`
10. `theorem argmax unchanged if margin exceeds twice perturbation epsilon multiclass classifier` / `probability argmax changes under Gaussian noise logit difference normal CDF` / `Thurstone law of comparative judgment probability one stimulus exceeds another margin`
11. `inflated argmax stability probability perturbation bound` / `"flip rate" model logit margin LLM quantization prediction churn` / `argmax stability epsilon perturbation theorem margin gamma learning theory`
12. `multiclass margin argmax unchanged if perturbation smaller than half margin theorem` / `Thurstone law of comparative judgment probability normal CDF discriminal dispersion` / `logit margin predicts token flip ROC-AUC quantization`
13. `"route flip" quantized mixture-of-experts router margin logit` / `choice modeling probability object i preferred j Thurstone normal CDF sigma` / `"flip" bound "margin" "perturbation" argmax theorem statement two classes`
14. `randomized smoothing certified radius argmax flip probability Gaussian noise classifier` / `"decision flip" probability model margin perturbation neural network theorem` / `logit margin threshold predicts greedy token divergence batch invariance measured`
15. `token flip rate versus logit margin curve LLM quantization flip rate` / `top-1 agreement under precision change greedy decoding rate LLM` / `prediction churn classifier perturbation probability model`
16. `"Greedy Decoding Is Not Precision-Invariant" cross-precision divergence arxiv 2026` / `closed form probability token flip logit margin perturbation LLM law` / `argmax flip probability bound logits epsilon infinity norm theorem statement`
Intermediate batch that **FAILED** with a transient search-endpoint error (`TypeError: fetch failed` from the
search endpoint) and returned no results — listed for completeness; its intent was partially re-covered by
batches 12–13: `"probability of a token flip" logit gap model` / `certified robustness flip probability Gaussian
noise margin binary classifier Phi` / `random utility model probability argmax changes perturbation Thurstone
discriminal`. No `web_search` batches were run after batch 16.

`curl` targets fetched and read (verification set): `arxiv.org/abs/{2506.09501, 2605.30218, 2406.13187, 2405.14064, 1902.02918, 2608.13756, 2608.11212, 2601.16880, 2605.23955, 2601.17768, 2605.10893}`; `arxiv.org/html/{2605.30218v1, 2506.09501v2, 2605.10893v1, 2605.23955v3}`; `arxiv.org/pdf/{2406.13187v2, 1902.02918v2}`; `proceedings.iclr.cc/.../0a48036026dc7946ef6033ae14719cc5-{Abstract-Conference.html, Paper-Conference.pdf}`; `papers.nips.cc/.../822621aaa88635437ea51023afdeaec2-Paper-Conference.pdf`; `github.com/vllm-project/vllm/{pull/55944, issues/55238}`; `raw.githubusercontent.com/thinking-machines-lab/batch_invariant_ops/main/README.md`; `raw.githubusercontent.com/microsoft/llm-42/main/README.md`; `raw.githubusercontent.com/vllm-project/vllm/main/docs/features/batch_invariance.md`; `thinkingmachines.ai/blog/defeating-nondeterminism-in-llm-inference/`; `web.stanford.edu/class/polisci350c/classonly/lecture1.pdf`; `emergentmind.com/topics/token-flipping-phenomenon`; `openreview.net/forum?id=QDOKyg7a5e` (BLOCKED); `api.semanticscholar.org/graph/v1/paper/search` (429).

### Local artifacts created (all under `.s3b_numdrift/subA/`, prefix `fliplaw_`)
`fliplaw_FINDINGS.md` (this file), `fliplaw_h2t.py`, `fliplaw_h2t2.py` (HTML→text helpers that preserve MathML `alttext`),
plus the fetched/derived sources: `fliplaw_2506.09501.html` / `_math.txt`, `fliplaw_margingate.html` / `_math.txt` /
`.txt`, `fliplaw_2406.13187.pdf` / `_pdf.txt`, `fliplaw_inflated.pdf` / `_pdf.txt`, `fliplaw_cohen.pdf` / `.txt`,
`fliplaw_iclr_stable.html` / `.pdf` / `_pdf.txt`, `fliplaw_thurstone.pdf` / `.txt`, `fliplaw_2608.13756.html`,
`fliplaw_2608.11212.html`, `fliplaw_2601.16880.html`, `fliplaw_2605.23955.html` / `h.html` / `.txt`,
`fliplaw_2601.17768.html`, `fliplaw_2605.10893.html` / `.txt`, `fliplaw_vllm55944.html`, `fliplaw_vllm55238.html`,
`fliplaw_vllm_batchinv.md`, `fliplaw_biops_readme.md`, `fliplaw_llm42.md`, `fliplaw_tml_blog.html` / `.txt`,
`fliplaw_emergent.html` / `.txt`, `fliplaw_1902.02918.html`, `fliplaw_s2_a.json`, `fliplaw_libs/` (local pypdf install).
No files outside `.s3b_numdrift/subA/` were modified.
