# Prior-art sweep: "approximate-inference safety / certified approximation"

**Scope:** work that BOUNDS or CERTIFIES how an approximate attention / approximate kernel changes
(i) the attention output, (ii) the final next-token DISTRIBUTION (e.g. TV distance), (iii) the final greedy TOKEN (argmax).

**Method.** Every arXiv ID below was fetched at `https://arxiv.org/abs/<id>` and the `<title>` read back **before** being
cited, per this project's rule that `web_search` has returned fabricated IDs and wrong titles. Every verbatim quote was
copied out of the full text I fetched myself (arXiv HTML or ar5iv HTML), saved under
`/Users/leihenan/Desktop/myProject/.s3b_numdrift/subB/`.

**Quoting convention (important).** The arXiv HTML-to-text conversion emits the *rendered* math and the *raw LaTeX
source* concatenated on one line, and it also inserts zero-width spaces (`U+200B`) inside rendered math. Rendered-math
strings are therefore not reliably greppable. **All quotes below are given as the raw LaTeX source substring**, copied
exactly from the saved file, with file and line number recorded. Where I also give a readable rendering it is explicitly
labelled "(rendered + source)".

---

## Summary table

| # | Artifact | URL | State I verified by fetching | Verbatim quote (exact LaTeX source) | Guarantee is ON | Structure or magnitude? |
|---|---|---|---|---|---|---|
| 1 | **SPRINTER** — Speeding up Speculative Decoding via Sequential Approximate Verification (Zhong, Teku, Tandon) | https://arxiv.org/abs/2502.04557 | VERIFIED — title readback `[2502.04557] Speeding up Speculative Decoding via Sequential Approximate Verification`; full text via ar5iv | `d_{\text{TV}}(p,p_{\text{SPRINTER}})=\eta_{\text{FP}}d_{\text{TV}}(p,q)` | **NEXT-TOKEN DISTRIBUTION** (exact TV equality) | **MAGNITUDE ONLY** (scalar false-positive rate η_FP) |
| 2 | **Top-k TV theory** — A Mathematical Theory of Top-k Sparse Attention via Total Variation Distance (Tzachristas, Deng, Tzachristas, Zhang, Chen) | https://arxiv.org/abs/2512.07647 | VERIFIED — title readback `[2512.07647] A Mathematical Theory of Top-$k$ Sparse Attention via Total Variation Distance` | `\operatorname{TV}(P,\widehat{P})\leq\frac{(n-k)e^{s_{k+1}}}{ke^{s_{k}}+(n-k)e^{s_{k+1}}}=\frac{1}{1+\frac{k}{\,n-k\,}e^{\Delta_{k}}}.` | **ATTENTION OUTPUT** + attention-weight TV | **STRUCTURE** — function of the boundary score gap Δ_k = s_k − s_{k+1} (i.e. *which* keys are kept) |
| 3 | **vAttention** — Verified Sparse Attention via Sampling (Desai, Agrawal, Yang, Cuadron, Schroeder, Zaharia, Gonzalez, Stoica), ICLR 2026 | https://arxiv.org/abs/2510.05688 | VERIFIED — title readback `[2510.05688] vAttention: Verified Sparse Attention`; ICLR 2026 title confirmed at mlanthology | `\mathbf{Pr}(||\textrm{vAttention{}}(K,V,q)-\textrm{SDPA}(K,V,q)||_{2}>\epsilon||\textrm{SDPA}(K,q,v)||_{2})\leq\delta` | **ATTENTION OUTPUT** (relative ℓ2, w.p. 1−δ) | **STRUCTURE** — budget depends on per-instance Σ, σ, ‖N‖₂, D |
| 4 | **SANTA** — Stochastic Sparse Attention for Memory-Bound Inference (Lee, Delacour, Callahan-Coray, Jiang, Yaras, Oymak, Srimani, Camsari), ICML 2026 | https://arxiv.org/abs/2605.01910 | VERIFIED — title readback `[2605.01910] Stochastic Sparse Attention for Memory-Bound Inference`; abstract defines SANTA = "Stochastic Additive No-mulT Attention" | `\Pr\!\left(\bigl\|\widehat{AV}_{q}-\mu_{q}\bigr\|_{2}\geq t\right)\;\leq\;2\exp\!\left(-\frac{S\,t^{2}}{2\,(v_{q}+V_{\max}t/3)}\right).` | **ATTENTION OUTPUT** (value aggregation) | MIXED — magnitude (V_max) + variance (v_q = tr(Σ_q)) |
| 5 | **Vashista Sparse Attention** — Attention in Constant Time (Nobaub) | https://arxiv.org/abs/2602.13804 | VERIFIED — title readback `[2602.13804] Attention in Constant Time: Vashista Sparse Attention for Long-Context Decoding with Exponential Guarantees` | `\left\lVert y_{\varepsilon}-y^{\ast}(q)\right\rVert\leq C_{\mathrm{lin}}\,\varepsilon+D(M-|I|)\exp\!\left(-\frac{\Delta(q)}{2\varepsilon}\right).` | **ATTENTION OUTPUT** | **STRUCTURE** — KKT support gap Δ(q) = min_{j∉I} μ*_j |
| 6 | **Runtime-Certified Bounded-Error Quantized Attention** (Calver) | https://arxiv.org/abs/2605.20868 | VERIFIED — title readback `[2605.20868] Runtime-Certified Bounded-Error Quantized Attention` | `\mathrm{TV}(a,a^{\prime})\leq\tanh(\Delta)=\frac{e^{2\Delta}-1}{e^{2\Delta}+1}` | **ATTENTION DISTRIBUTION → ATTENTION OUTPUT** | **MAGNITUDE ONLY** — only max per-token logit error Δ; "independent of sequence length N" |
| 7 | **Exact Attention Sensitivity and the Geometry of Transformer Stability** (Emadi) | https://arxiv.org/abs/2602.18849 | VERIFIED — title readback `[2602.18849] Exact Attention Sensitivity and the Geometry of Transformer Stability` | `\|J\|_{\infty\to 1}=\frac{\theta(p)}{\tau}.` | **ATTENTION DISTRIBUTION** (mass redistributed) | **STRUCTURE** — exact; entire shape dependence in the balanced-mass factor θ(p) |
| 8 | **MTI / cache-side vulnerabilities** — Can Transformer Memory Be Corrupted? (Hossain, Saha, Roy, Prasad) | https://arxiv.org/abs/2510.17098 | VERIFIED — title readback `[2510.17098] Can Transformer Memory Be Corrupted? Investigating Cache-Side Vulnerabilities in Large Language Models` | `\|\Delta z_{t}\|_{2}\;\leq\;\epsilon\cdot\|q_{t}^{(\ell)}\|_{2}.` | **NEXT-TOKEN LOGITS**, then attention distribution | **MAGNITUDE ONLY** in the theorem; structure only empirically |
| 9 | **TPA** — Towards Poisoning Robustness Certification for Natural Language Generation (Ghitu, Wicker) | https://arxiv.org/abs/2602.09757 | VERIFIED — title readback `[2602.09757] Towards Poisoning Robustness Certification for Natural Language Generation` | "certify validity/targeted attacks by computing the minimum poisoning budget needed to induce a specific harmful class, token, or phrase" | **DECISION** (token/phrase) — but vs. **data poisoning**, not attention perturbation | N/A (poisoning budget, not perturbation geometry) |
| 10 | **VeriCache** — Turning Lossy KV Cache into Lossless LLM Inference (Yao, Shen, Du, et al.) | https://arxiv.org/abs/2605.17613 | VERIFIED — title readback `[2605.17613] VeriCache: Turning Lossy KV Cache into Lossless LLM Inference` | "ensures the same output as full-KV-cache decoding" (draft-then-verify exact recovery; no error bound) | **DECISION** (exact equality by construction) | N/A — verification removes the question |
| 11 | **Is Flash Attention Stable?** (Golden, Hsia, Sun, et al.) | https://arxiv.org/abs/2405.02803 | VERIFIED — title readback `[2405.02803] Is Flash Attention Stable?` | "Flash Attention sees roughly an order of magnitude more numeric deviation as compared to Baseline" | **EMPIRICAL NUMERIC DEVIATION** — no certified bound | N/A — empirical, not a bound |
| 12 | **Lipschitz Normalization for Self-Attention Layers** (Dasoulas, Scaman, Virmaux) | https://arxiv.org/abs/2103.04886 | VERIFIED — title readback `[2103.04886] Lipschitz Normalization for Self-Attention Layers with Application to Graph Neural Networks` | derives Lipschitz constants of attention modules | **ATTENTION MODULE (Lipschitz constant)** | MAGNITUDE (operator norms) |
| 13 | **Inference Time Context Sparsity: Illusion or Opportunity?** (Joshi, Dixit, Chowdhury, Shrivastava, Gonzalez, Stoica, Agrawal, Desai) | https://arxiv.org/abs/2605.24168 | VERIFIED — title readback `[2605.24168] Inference Time Context Sparsity: Illusion or Opportunity?` | see §13 (non-injectivity of `a ↦ Vᵀa`) | **ATTENTION OUTPUT** (expressivity collapse, not an error bound) | STRUCTURE (rank/dimension) |
| 14 | **IEEE version of #2** — DOI 10.1109/ISIT62367.2026.11653946 | https://ieeexplore.ieee.org/document/11653946 | **COULD NOT VERIFY** — my own fetch returned HTTP 202, 0 bytes | — | — | — |

---

## Detail on the strongest artifacts

### 1. SPRINTER — arXiv 2502.04557 — strongest FINAL-DISTRIBUTION result

Source: `https://ar5iv.labs.arxiv.org/html/2502.04557`, Appendix A.1 (also stated in the main body).
Saved as `sprinter_fulltext.txt`.

Verbatim (exact LaTeX source, `sprinter_fulltext.txt:271`; same string at `:61`):

> `d_{\text{TV}}(p,p_{\text{SPRINTER}})=\eta_{\text{FP}}d_{\text{TV}}(p,q)`

Full surrounding sentence, verbatim (rendered + source as extracted):

> `where η FP \eta_{\text{FP}} is the false positive rate of the verifier. Furthermore, the total-variation distance between the target and SPRINTER distributions is d TV  ( p , p SPRINTER ) = η FP  d TV  ( p , q ) d_{\text{TV}}(p,p_{\text{SPRINTER}})=\eta_{\text{FP}}d_{\text{TV}}(p,q) .`

The mixture identity, verbatim (exact LaTeX source, `sprinter_fulltext.txt:269`):

> `p_{\text{SPRINTER}}(x)=(1-\eta_{\text{FP}})p(x)+\eta_{\text{FP}}q(x),`

And the final derivation line, verbatim (`sprinter_fulltext.txt:292`):

> `= \eta_{\text{FP}}\frac{1}{2}\sum_{x}|p(x)-q(x)|=\eta_{\text{FP}}d_{\text{TV}}(p,q).`

**Classification.** A **next-token distribution** guarantee, and an *exact equality*, not an inequality: the approximate
verification pipeline's token distribution is exactly a convex mixture of the target and draft distributions, so its TV
distance from the target is exactly `η_FP · d_TV(p,q)`.

**Structure vs magnitude.** Purely **magnitude**. The right-hand side depends only on the scalar false-positive rate
η_FP and the (already fixed) draft/target mismatch `d_TV(p,q)`. Which token was mis-verified, at which position, or in
which head plays no role.

**Honest caveat.** The perturbation here is the *verifier's accept/reject decision*, not attention weights. So it is a
final-distribution bound for *approximate inference*, but not for *approximate attention*.

### 2. Top-k TV theory — arXiv 2512.07647 — strongest ATTENTION-OUTPUT certificate

Full text: `https://arxiv.org/html/2512.07647v1` (fetched, 805,557 bytes), saved as `arxiv_html_2512.07647v1.html`
and `topk_tv_fulltext.txt`. Also mirrored at `https://ar5iv.labs.arxiv.org/html/2512.07647` (fetched, 753,879 bytes).

Metadata verified from the abs page: authors **Tzachristas, Georgios; Deng, Lei; Tzachristas, Ioannis; Zhang, Gong;
Chen, Renhai**; date 2025/12/08; subjects cs.LG, cs.AI. The abstract self-describes the work as
"a unified mathematical framework for **certified Top-$k$ attention truncation**".

Main results, verbatim as exact LaTeX source:

**Tail–mass identity (Lemma 4.1)** — `topk_tv_fulltext.txt:365`:
> `\operatorname{TV}(P,\widehat{P})=\sum_{i>k}p_{i}.`

**Exact TV–KL identity (Theorem 4.3)** — `topk_tv_fulltext.txt:375`:
> `\operatorname{KL}(\widehat{P}\|P)=\log\frac{\sum_{j=1}^{n}e^{s_{j}}}{\sum_{j=1}^{k}e^{s_{j}}},\hskip 20.00003pt\operatorname{TV}(P,\widehat{P})=1-e^{-\operatorname{KL}(\widehat{P}\|P)}.`

**Deterministic gap bound (Theorem 4.5)** — `topk_tv_fulltext.txt:402`:
> `\operatorname{TV}(P,\widehat{P})\leq\frac{(n-k)e^{s_{k+1}}}{ke^{s_{k}}+(n-k)e^{s_{k+1}}}=\frac{1}{1+\frac{k}{\,n-k\,}e^{\Delta_{k}}}.`

**Exact head–tail identity (Theorem 5.2)** — `topk_tv_fulltext.txt:554`:
> `\operatorname{Attn}(q,K,V)-\operatorname{Attn}_{k}(q,K,V)=\tau\,(\mu_{\mathrm{tail}}-\mu_{\mathrm{head}})`

**Best available certificate (Theorem 5.6)** — `topk_tv_fulltext.txt:589`:
> `\boxed{\bigl\|\operatorname{Attn}(q,K,V)-\operatorname{Attn}_{k}(q,K,V)\bigr\|_{2}\leq\min\!\Bigl\{\tau\,\mathrm{diam}_{H,T},\;\sqrt{D_{\chi^{2}}(\widehat{P}\|P)}\,\sqrt{\operatorname{Var}_{P}(V)},\;2C\tau\Bigr\}.}`

Supporting ℓ2 output bound (Proposition 5.1) and diameter bound (Proposition 5.3), verbatim (rendered + source):
> `‖ Attn ⁡ ( q , K , V ) − Attn k ⁡ ( q , K , V ) ‖ 2 ≤ 2 ​ C ​ TV ⁡ ( P , P ^ ) . \bigl\|\operatorname{Attn}(q,K,V)-\operatorname{Attn}_{k}(q,K,V)\bigr\|_{2}\leq 2C\,\operatorname{TV}(P,\widehat{P}).`
> `‖ Attn ⁡ ( q , K , V ) − Attn k ⁡ ( q , K , V ) ‖ 2 ≤ τ ​ diam H , T . \bigl\|\operatorname{Attn}(q,K,V)-\operatorname{Attn}_{k}(q,K,V)\bigr\|_{2}\leq\tau\,\mathrm{diam}_{H,T}.`

**Classification.** Guarantee is on the **attention output** (ℓ2), with attention-weight **TV** as the intermediate. It is
*not* propagated to the next-token distribution or to the argmax.

**Structure vs magnitude.** Structurally explicit. TV error is controlled by the **boundary score gap**
`Δ_k = s_k − s_{k+1}` — *which* keys fall inside versus outside the top-k set and how well separated they are — not merely
by the size of some perturbation. The exact identity says the output error is the product of the tail *mass* and the
*geometry of the head/tail cut* in value space. Theorem 5.8 then optimizes *which* partition to use, via a maximum
spanning tree.

**The paper's own stated boundary** (Limitations, verbatim plain text, `topk_tv_fulltext.txt:1703`; the sentence begins on `:1702`):

> `Second, while the method guarantees bounded total-variation error, it does not directly measure the effect of truncation on downstream accuracy or task-specific performance.`

This is the explicit statement that the guarantee stops at the attention output.

**Independent corroboration.** The cert-quant paper (arXiv 2605.20868) cites exactly this work as its reference [14],
verbatim from its bibliography (`html_2605.20868.html`):

> `[14] Tzachristas, G., Deng, L., Tzachristas, I., Zhang, G., and Chen, R. A mathematical theory of top- k k sparse attention via total variation distance. arXiv preprint arXiv:2512.07647 , 2025.`

### 3. vAttention — arXiv 2510.05688 — ICLR 2026, what the (ε, δ) is actually ON

Full text: `https://arxiv.org/html/2510.05688v1`, saved as `text_2510.05688.txt`.

**Answer to the direct question: the (ε, δ) guarantee is ON THE ATTENTION OUTPUT**, as a *relative* ℓ2 error. It is **not**
on the final token distribution and **not** on the argmax.

Theorem 4.3, verbatim (exact LaTeX source, `text_2510.05688.txt:173`):
> `\mathbf{Pr}(||\textrm{vAttention{}}(K,V,q)-\textrm{SDPA}(K,V,q)||_{2}>\epsilon||\textrm{SDPA}(K,q,v)||_{2})\leq\delta`

Its stated population parameters, verbatim (rendered + source, `text_2510.05688.txt:171`):
> `Let Σ \Sigma be the covariance matrix for the population { exp ⁡ ⟨ K ⁡ [ i ] , q ​ V ​ [ i ] ⟩ } i ∈ I f ¯ \{\exp{\langle K[i],q}V[i]\rangle\}_{i\in\bar{\mathcal{I}_{f}}}`

Supporting Lemma 4.1 (CLT budget), verbatim (exact LaTeX source, `text_2510.05688.txt:158`):
> `b\geq\left(\Phi^{-1}\left(1-\frac{\delta}{2}\right)\frac{n_{s}\sqrt{\mathbf{Tr}(\Sigma)}}{\tau}\right)^{2}\;\;\;\;\textrm{then}\;\;\;\;\mathbf{Pr}(||\hat{\mathbf{s}}-\mathbf{s}||_{2}>\tau)\leq\delta`

Lemma 4.2, verbatim (exact LaTeX source, `text_2510.05688.txt:165`):
> `\mathbf{Pr}\left(\left\lVert\frac{N}{D}-\frac{\hat{N}}{\hat{D}}\right\rVert_{2}>2(\epsilon_{1}+\epsilon_{2})\right)\left\lVert\frac{N}{D}\right\rVert_{2}<(\delta_{1}+\delta_{2})`

**Structure vs magnitude.** **Structure-dependent.** The required budget `b` is a function of the *actual* population
covariance Σ, the standard deviation σ, `‖N‖₂` and `D`, computed from the real `K, V, q` for that head and step. It is a
per-instance randomized guarantee: randomness is over the sampling, and the guarantee is relative to the real attention
configuration. It is not a function of a single scalar perturbation magnitude.

**Caveat the paper states itself** (verbatim plain text `text_2510.05688.txt:175`; the Algorithmic-Relaxations paragraph reference is `:176`):

> `In practice, Σ \Sigma and σ \sigma as well as exact ‖ N ‖ 2 ||N||_{2} and D D are not known apriori. However, we can use a base sample from residual tokens to estimate them in order to compute these quantities.`

...and all experiments use the relaxation that certifies "the denominator alone", not the full numerator+denominator bound.
So the shipped system's guarantee is weaker than Theorem 4.3.

**Venue confirmation.** mlanthology (fetched, `vattn.html`) gives ICLR 2026, authors Desai, Agrawal, Yang, Cuadron,
Schroeder, Zaharia, Gonzalez, Stoica, linking `https://openreview.net/forum?id=zzTDulLys0`. The ICLR 2026 title is
"vAttention: Verified Sparse Attention via Sampling"; the arXiv title (2510.05688) is "vAttention: Verified Sparse
Attention" — same authors, same abstract.

### 4. SANTA — arXiv 2605.01910 — the unbiased-estimator lead, corrected

**Correction to the brief.** No paper titled "SANTA: Sampling-based approximate attention" was found. The verifiable SANTA
is **"Stochastic Additive No-mulT Attention"**, in *Stochastic Sparse Attention for Memory-Bound Inference*
(arXiv 2605.01910; Kyle Lee, Corentin Delacour, Kevin Callahan-Coray, Kyle Jiang, Can Yaras, Samet Oymak, Tathagata
Srimani, Kerem Y. Camsari; 2026/05/03; ICML 2026 poster). Abstract, verbatim:

> `We present Stochastic Additive No-mulT Attention (SANTA), a method that sparsifies value-cache access by sampling S ≪ n k indices from the post-softmax distribution and aggregates only those value rows. This yields an unbiased estimator of the post-softmax value aggregation [...]`

Theorem B.1 (Vector Bernstein tail for SANTA), verbatim (exact LaTeX source, `text_2605.01910.txt:722`):
> `\Pr\!\left(\bigl\|\widehat{AV}_{q}-\mu_{q}\bigr\|_{2}\geq t\right)\;\leq\;2\exp\!\left(-\frac{S\,t^{2}}{2\,(v_{q}+V_{\max}t/3)}\right).`

with `v_q` defined, verbatim (`text_2605.01910.txt:721`):
> `v_{q}\triangleq\mathbb{E}\!\left[\|V_{i}-\mu_{q}\|_{2}^{2}\right]=\mathrm{tr}(\Sigma_{q})`

**Guarantee ON:** the **attention output** (post-softmax value aggregation). **Structure vs magnitude:** mixed — depends
on the value-distribution variance `v_q = tr(Σ_q)` (a structure/statistics term) and the magnitude bound `V_max`.

### 5. Vashista Sparse Attention — arXiv 2602.13804 — exponential, structure-dependent

Theorem 1 (Face stability with exponential leakage), verbatim (exact LaTeX source, `text_2602.13804.txt:267`):
> `\left\lVert y_{\varepsilon}-y^{\ast}(q)\right\rVert\leq C_{\mathrm{lin}}\,\varepsilon+D(M-|I|)\exp\!\left(-\frac{\Delta(q)}{2\varepsilon}\right).`

The support gap, verbatim (exact LaTeX source, `text_2602.13804.txt:252`):
> `\Delta(q)\;:=\;\min_{j\notin I}\mu_{j}^{\ast}.`

Corollary 1, verbatim (rendered + source, `text_2602.13804.txt:261`):
> `α ε , j ≤ exp ⁡ ( − Δ ⁡ ( q ) c ​ ε ) , ∑ j ∉ I α ε , j ≤ ( M − | I | ) ​ exp ⁡ ( − Δ ⁡ ( q ) c ​ ε ) . \alpha_{\varepsilon,j}\ \leq\ \exp\!\Big(-\frac{\Delta(q)}{c\,\varepsilon}\Big),\qquad\sum_{j\notin I}\alpha_{\varepsilon,j}\ \leq\ (M-|I|)\exp\!\Big(-\frac{\Delta(q)}{c\,\varepsilon}\Big).`

**Structure vs magnitude:** strongly **structure**-dependent — the exponential term is governed by `Δ(q)`, the minimum
KKT multiplier over *inactive* tokens, i.e. by which tokens are on the active face and how strictly complementarity
separates them. Guarantee is on the **attention output** `y_ε`.

### 6. Runtime-Certified Bounded-Error Quantized Attention — arXiv 2605.20868

This is the closest thing found to the requested form *"if attention weights are perturbed by ε then the distribution
changes by ≤ f(ε)"*.

TV-from-logit-perturbation bound, verbatim (exact LaTeX source, `text_2605.20868.txt:281`, repeated at `:1509`):
> `\mathrm{TV}(a,a^{\prime})\leq\tanh(\Delta)=\frac{e^{2\Delta}-1}{e^{2\Delta}+1}`

Immediately following, verbatim (plain text, `text_2605.20868.txt:283`):
> `This bound is tight, independent of sequence length N N , and satisfies tanh ⁡ ( Δ ) ≈ Δ \tanh(\Delta)\approx\Delta for small Δ \Delta .`

Theorem 4.1 (Value Compression Error), verbatim (rendered + source, `text_2605.20868.txt:240`):
> `E val = ‖ ∑ t a t ​ V ^ t − ∑ t a t ​ V t ‖ 2 ≤ η E_{\mathrm{val}}=\lVert\sum_{t}a_{t}\hat{V}_{t}-\sum_{t}a_{t}V_{t}\rVert_{2}\leq\eta`

Theorem 4.3 (Key Compression Error), verbatim (rendered + source, `text_2605.20868.txt:262`):
> `E key = ‖ ∑ t a t ′ ​ V t − ∑ t a t ​ V t ‖ 2 ≤ 2 ​ V max ⋅ TV ⁡ ( a , a ′ ) E_{\mathrm{key}}=\lVert\sum_{t}a_{t}^{\prime}V_{t}-\sum_{t}a_{t}V_{t}\rVert_{2}\leq 2V_{\max}\cdot\mathrm{TV}(a,a^{\prime})`

Explicit "no bound on N" property, verbatim (`text_2605.20868.txt:254`):
> `The bound depends only on the worst-case per-token reconstruction error η \eta , not on the number of tokens N N .`

**Classification.** Guarantee is on the **attention distribution TV** and the **attention output**. **MAGNITUDE ONLY** —
depends solely on `Δ`, the max absolute per-token logit error.

**Explicit ranking / argmax caveat** (Section 6.1, verbatim plain text, `text_2605.20868.txt:375-376`):
> `The formal mass bound (Theorem 4.4 ) guarantees that the aggregate tail mass is small, but does not guarantee that the ranking within the top- K ∗ is correct.`
> `[...] if a high-mass block is demoted to INT8 while a lower-mass block is promoted to FP16, the resulting mixed-precision softmax distribution may differ from the all-FP16 distribution in ways that affect the argmax over output logits.`

That is the closest thing in the whole sweep to an explicit **argmax** statement — and it says the existing certificate
does **not** cover the argmax. Their remedy is an empirical "ranking-consistency check" that escalates to FP16 fallback,
which they report triggering 10,221 times on PG-19 at 64K and 905 times on NIAH at 64K.

### 7. Exact Attention Sensitivity — arXiv 2602.18849 — the key STRUCTURE-vs-MAGNITUDE artifact

Theorem 4.2, verbatim (exact LaTeX source, `text_2602.18849.txt:174`):
> `\|J\|_{\infty\to 1}=\frac{\theta(p)}{\tau}.`

Immediately following, verbatim (plain text, `text_2602.18849.txt:175`):
> `This is an equality, not merely an upper bound.`

Definition 4.1 (Balanced-mass factor), verbatim (exact LaTeX source, `text_2602.18849.txt:167`):
> `\theta(p)=4\max_{S\subseteq[L]}p(S)(1-p(S))\in[0,1],`

The paper's own contrast with magnitude-only bounds, verbatim (`text_2602.18849.txt:176`):
> `Unlike prior ℓ 2 \ell_{2} bounds that are distribution-independent ( Kim et al., 2021 ; Castin et al., 2024 ; Yudin et al., 2025 ) , our result provides an exact equality where θ ⁡ ( p ) \theta(p) captures how distribution shape determines sensitivity.`

Theorem 5.1 (MHA Lipschitz bound), verbatim (rendered + source, `text_2602.18849.txt:203`):
> `L MHA ≤ ‖ W O ‖ 2 ​ ∑ h = 1 H ( ‖ W h V ‖ 2 + θ ~ h τ ​ Φ h ) , L_{\MHA}\leq\|W^{O}\|_{2}\sum_{h=1}^{H}\left(\|W^{V}_{h}\|_{2}+\frac{\tilde{\theta}_{h}}{\tau}\Phi_{h}\right),`

**Why this matters for the occupancy check.** This is the cleanest existing formalism for separating *perturbation
magnitude* from *perturbation structure*. The `ℓ∞→ℓ1` operator norm is exact, and the entire distributional dependence is
isolated in `θ(p)`. It bounds the **attention distribution** (mass redistributed) — not the final token distribution. The
paper also reports empirically that trained transformers sit at `θ(p) ≈ 1`, i.e. at maximum sensitivity, throughout
training.

### 8. MTI — arXiv 2510.17098 — next-token LOGIT bound from cache perturbation

Theorem 1, verbatim (exact LaTeX source, Appendix A.1, `text_2510.17098.txt:765`):
> `\|\Delta z_{t}\|_{2}\;\leq\;\epsilon\cdot\|q_{t}^{(\ell)}\|_{2}.`

and (`text_2510.17098.txt:767`):
> `\|\alpha_{t}-\tilde{\alpha}_{t}\|_{1}\;\leq\;L\cdot\epsilon\cdot\|q_{t}^{(\ell)}\|_{2},`

Multi-head extension, verbatim (rendered + source, `text_2510.17098.txt:791`):
> `‖ Δ ​ z t multi ‖ 2 ≤ ‖ W O ‖ 2 ⋅ ∑ h = 1 H ‖ Δ ​ z t ( h ) ‖ 2 ≤ ‖ W O ‖ 2 ⋅ H ⋅ ϵ ⋅ max h ⁡ ‖ q t ( h ) ‖ 2 . \|\Delta z_{t}^{\text{multi}}\|_{2}\;\leq\;\|W_{O}\|_{2}\cdot\sum_{h=1}^{H}\|\Delta z_{t}^{(h)}\|_{2}\;\leq\;\|W_{O}\|_{2}\cdot H\cdot\epsilon\cdot\max_{h}\|q_{t}^{(h)}\|_{2}.`

**⚠ Internal inconsistency worth flagging.** The main body states the perturbation budget as a **Frobenius-norm equality**,
while the appendix theorem states a **2-norm inequality** — these are not the same hypothesis:

- main body, verbatim (rendered + source, `text_2510.17098.txt:261`): `with ‖ δ j ‖ F = ϵ \|\delta_{j}\|_{F}=\epsilon`
- appendix Theorem 1, verbatim (rendered + source, `text_2510.17098.txt:763`): `‖ δ j ( ℓ ) ‖ 2 ≤ ϵ \|\delta_{j}^{(\ell)}\|_{2}\leq\epsilon`

**Classification.** Reaches the **next-token logits** (the closest to the final distribution among attention-side bounds),
then the attention distribution. **MAGNITUDE ONLY** in the theorem — uses only `ε`, the per-key perturbation norm.
Structure is *not* in the bound; it appears only experimentally, where a layer-wise ablation (verbatim, `:621`) reports
"the KL divergence under MTI V.1 rose from 0.0367 at Layer 1 to 0.0447 at Layer 3". So: **structure matters empirically
but is absent from the theory.**

### 13. Survey — arXiv 2605.24168 — framing / gap statement

Position paper from the vAttention group. Theorem 1, verbatim (rendered + source, `text_2605.24168.txt:97-103`; the non-injectivity sentence is on `:99`):

> `Let V ∈ ℝ N × d V\in\mathbb{R}^{N\times d} be any value matrix, and let the dense attention output be o = V ⊤ ​ a o=V^{\top}a [...] If d < N − 1 d<N-1 , then this map is not injective on the attention simplex. In particular, there exist two distinct dense attention distributions a , a ′ a,a^{\prime} such that a ≠ a ′ a\neq a^{\prime} , and V ⊤ ​ a = V ⊤ ​ a ′ V^{\top}a=V^{\top}a^{\prime} .`

This is an expressivity/collapse argument, not an error bound, but it is a citable framing for why attention-output bounds
do not imply distributional or decision guarantees: the output map already discards attention-distribution information.

---

## (i) Everything I could NOT verify, and why

| Item | What I tried | Result |
|---|---|---|
| **IEEE Xplore document 11653946** | Direct `curl` (browser UA, retries) to `https://ieeexplore.ieee.org/document/11653946` | **HTTP 202, 0 bytes.** Matches this project's known rule. The IEEE landing page is **unverified by me**. |
| **ISIT 2026 venue metadata for #2** | `https://dblp.org/pid/239/7382.html` | **Blocked** — DBLP now serves an **Anubis bot challenge**; the page `<title>` read back as `Making sure you're not a bot!`. Could not read the DBLP record. |
| **DOI / venue for #2** | Semantic Scholar Graph API | **Partially verified.** One call succeeded before rate-limiting and returned `"DOI": "10.1109/ISIT62367.2026.11653946"`, `"DBLP": "conf/isit/TzachristasDTZC26"`, `"venue": "International Symposium on Information Theory"`, `"year": 2025`, open-access PDF pointing at arXiv 2512.07647. Reporting as **verified-via-S2-API-JSON, NOT verified on IEEE's own site**. Note the S2 record says year 2025 while the DBLP key suffix is `26`; I could not resolve that discrepancy from a primary source. |
| **vAttention OpenReview forum** (`openreview.net/forum?id=zzTDulLys0`) | Direct `curl` | **HTTP 200 but 4,787 bytes = browser-challenge page.** No content. |
| **vAttention OpenReview PDF** (`https://openreview.net/pdf/b353d1d314c97b316faf589c3fd73cec7d570b3a.pdf`, taken from the mlanthology page's own `PDF` link) | Direct `curl` | **HTTP 403.** |
| **OpenReview API** for vAttention | `https://api2.openreview.net/notes?forum=zzTDulLys0` and `https://api.openreview.net/notes?forum=zzTDulLys0` | Both **HTTP 403** `{"name":"ChallengeRequiredError","message":"Challenge verification required"}`. Fell back to arXiv 2510.05688, which carries the same theorems. I could **not** verify reviewer scores, decision letter, or camera-ready text. |
| **Unidentified OpenReview PDF** `openreview.net/pdf?id=Q3cbXoUgFF` (surfaced with fragment `D_{\mathrm{KL}}(p\| p^{\prime})\leq C\epsilon^{2}`) | Search only | **Could not identify the paper.** I did not fetch it (openreview is challenge-walled) and I do not cite it. Listed only so it is on record as *seen-but-unattributed*. |
| **Semantic Scholar API generally** | Repeated calls with retries | Mostly **HTTP 429** ("Too Many Requests"). Only one of four queries ever succeeded. |
| **arXiv full-text search** (`arxiv.org/search`) | Not attempted | Known 406 in this project; used `arxiv.org/abs/<id>` + title readback for every citation instead. |
| **`export.arxiv.org` API** | Not attempted | Known 429 in this project. |
| **"SANTA: Sampling-based approximate attention"** as literally titled | Multiple search phrasings | **No such paper found.** The verifiable SANTA is "Stochastic Additive No-mulT Attention" in arXiv 2605.01910. Flagging the brief's title as unverified/likely misremembered rather than inventing a match. |
| **A theorem proving argmax preservation under an attention-perturbation condition C** | Many phrasings | **Not found.** The only explicit argmax statements located are negative ones (arXiv 2605.20868 saying its certificate does *not* cover the argmax over output logits). |
| **FlashAttention output-error *theorem*** | Multiple phrasings | **Not found.** What exists and is verified is the empirical *Is Flash Attention Stable?* (2405.02803), which quantifies numeric deviation without a certified output bound. Search also surfaced `arXiv 2502.12063`; I fetched it and it is **"Low-Rank Thinning"** (Carrell, Gong, Shetty, Dwivedi, Mackey) — *not* a FlashAttention error paper. Recording this as a confirmed mis-association. |

### Naming-collision warnings (all verified by title readback)

1. **vAttention** — two distinct papers share the name:
   - `arXiv 2405.04437` = **"vAttention: Dynamic Memory Management for Serving LLMs without PagedAttention"**
     (Prabhu, Nayak, Mohan, Ramjee, Panwar; 2024/05/07). This is the one cited as [15] by the cert-quant paper.
   - `arXiv 2510.05688` = **"vAttention: Verified Sparse Attention"** (Desai et al.) — the relevant one here.
2. **SpecAttn** (the trap flagged in the brief) — both confirmed, and they are indeed different papers:
   - `arXiv 2510.27641` = **"SpecAttn: Speculating Sparse Attention"** (Shah, Harsh; 2025/10/31).
   - `arXiv 2602.07223` = **"Vegas: Self-Speculative Decoding with Verification-Guided Sparse Attention"** (Yue, Xue, Huang; 2026/02/06).
   I did not rely on either for any bound.

---

## (ii) Exact search queries I ran

**`web_search` queries (verbatim, in order):**

1. `SANTA Sampling-based approximate attention unbiased estimator sparse attention arxiv`
2. `"SANTA" sampling approximate attention softmax unbiased estimator paper`
3. `SPRINTER speculative decoding approximate verification drift bound`
4. `"SPRINTER" speculative decoding verification approximation error bound arxiv`
5. `vAttention "Verified Sparse Attention via Sampling" Desai pdf arxiv`
6. `Aditya Desai vAttention sparse attention epsilon delta guarantee ICML ICLR 2026`
7. `quantized KV cache bound on output distribution total variation LLM error analysis`
8. `KV cache quantization error bound next-token distribution attention output perturbation`
9. `FlashAttention numerical error bound output error analysis floating point`
10. `approximate attention kernel error bound output perturbation theory`
11. `certified robustness LLM argmax token certification randomized smoothing text`
12. `randomized smoothing transformer next token certification guarantee`
13. `bound on final token distribution from attention output perturbation transformer Lipschitz`
14. `attention perturbation bounds next-token logits distribution shift guarantee`
15. `certified robustness LLM argmax preservation provable guarantee token prediction`
16. `randomized smoothing argmax certification language model provable`
17. `FlashAttention error bound output approximation floating point analysis theorem`
18. `"attention" approximate kernel output error bound certified numerical`
19. `"argmax" certified robust language model provable prediction margin token`
20. `top-1 prediction certification transformer provable margin guarantee`
21. `"SANTA" sampling approximate attention linear attention unbiased estimator 2024`
22. `SANTA sparse attention sampling approximation acronym paper`
23. `openreview Q3cbXoUgFF KL divergence bound epsilon attention`
24. `certified sparse attention blockwise mass certificate pruning guarantee`
25. `sparse attention approximation bound KL divergence final output logits distribution guarantee theorem`
26. `provable guarantee sparse attention preserves next token prediction distribution`
27. `Vashista Sparse Attention exponential guarantees long-context decoding arxiv`
28. `"Attention in Constant Time" Vashista sparse attention arxiv`
29. `"Inference Time Context Sparsity: Illusion or Opportunity" arxiv survey`
30. `openreview Q3cbXoUgFF`

**Direct `curl` fetches (browser UA, `--retry 4 --retry-delay 2 --retry-all-errors`):**

- `https://mlanthology.org/iclr/2026/desai2026iclr-vattention/` — 200
- `https://api.semanticscholar.org/graph/v1/paper/search?query=Mathematical%20Theory%20of%20Top-k%20Sparse%20Attention%20Total%20Variation&fields=...` — **200** (the one S2 success)
- `.../paper/search?query=SANTA%20sampling%20based%20approximate%20attention` — 429
- `.../paper/search?query=SANTA%20sampling%20approximate%20attention` — 429
- `.../paper/search?query=vAttention%20Verified%20Sparse%20Attention%20via%20Sampling` — 429
- `https://arxiv.org/abs/{2512.07647, 2502.04557, 2510.05688, 2605.20868, 2605.17613, 2502.12063, 2602.18849, 2510.17098, 2103.04886, 2405.02803, 2405.04437, 2602.09757, 2605.01910, 2602.13804, 2605.24168, 2510.27641, 2602.07223}` — all 200, every title read back
- `https://arxiv.org/html/{2512.07647v1, 2510.05688v1, 2605.20868v1, 2602.18849v1, 2510.17098v1, 2605.01910v1, 2602.09757v1, 2602.13804v1, 2605.24168v1}` — all 200
- `https://ar5iv.labs.arxiv.org/html/2502.04557` — 200
- `https://ar5iv.labs.arxiv.org/html/2512.07647` — 200
- `https://openreview.net/pdf/b353d1d314c97b316faf589c3fd73cec7d570b3a.pdf` — **403**
- `https://openreview.net/forum?id=zzTDulLys0` — 200 (challenge page, no content)
- `https://api2.openreview.net/notes?forum=zzTDulLys0` — **403**
- `https://api.openreview.net/notes?forum=zzTDulLys0` — **403**
- `https://ieeexplore.ieee.org/document/11653946` — **202, 0 bytes**
- `https://dblp.org/pid/239/7382.html` — 200 (Anubis challenge page)

---

## Bottom line for the occupancy check

- **Strongest FINAL-DISTRIBUTION bound under approximate inference:** **SPRINTER** (arXiv 2502.04557), exact identity
  `d_TV(p, p_SPRINTER) = η_FP · d_TV(p, q)`. Exact, not an inequality — but it concerns approximate *verification*, and the
  perturbation enters only as a scalar rate.
- **Strongest NEXT-TOKEN-LOGIT bound from an attention-side perturbation:** **MTI** (arXiv 2510.17098),
  `‖Δz_t‖₂ ≤ ε·‖q_t‖₂` — magnitude-only.
- **Strongest ATTENTION-OUTPUT certificate for genuine top-k attention:** **arXiv 2512.07647**, which is also the one
  artifact here that is explicitly *structural* (boundary gap `Δ_k`, head/tail cut identity, MaxST-optimal partition).
- **Does ANY of them distinguish perturbation STRUCTURE from MAGNITUDE?** **Yes — three do, but none reaches the token
  decision:**
  - **arXiv 2602.18849** is the cleanest: an *exact* operator norm `‖J‖_{∞→1} = θ(p)/τ` isolating the entire
    distributional dependence in the balanced-mass factor `θ(p)`, explicitly contrasted against
    "distribution-independent" magnitude bounds. Bounds the **attention distribution**.
  - **arXiv 2512.07647** makes the guarantee a function of *which* keys are kept (via `Δ_k`) and of the value-space
    geometry of the head/tail cut.
  - **arXiv 2602.13804** makes the guarantee exponential in the KKT support gap `Δ(q)` — the strict complementarity
    margin separating active from inactive tokens.
  - By contrast **SPRINTER**, **cert-quant (2605.20868)**, **SANTA (2605.01910)** and **MTI (2510.17098)** are
    magnitude-only.
- **Argmax / decision-level certification under attention perturbation: NO such theorem found.** The only argmax-aware
  statement located is *negative* (arXiv 2605.20868: its certificate "does not guarantee that the ranking within the
  top-K* is correct", and mixed-precision softmax "may differ from the all-FP16 distribution in ways that affect the
  argmax over output logits"), handled by an empirical check plus FP16 fallback rather than a proof. Token-level
  certification does exist (TPA, arXiv 2602.09757), but against **data poisoning**, not against approximation.
