# Exactness / distribution-preservation of speculative decoding under approximate (sparse, evicted, quantised) KV access

Recon date 2026-09-14. Every arXiv ID below was confirmed by fetching `https://arxiv.org/abs/<id>` and reading the title back off the fetched page. Every quote was copied from the stripped full text of `https://arxiv.org/html/<id>v<N>`. Nothing below is reconstructed from memory or from a search snippet.

## Headline

The SD exactness guarantee is treated in the literature as a property of the **verification step only**. Every verified paper that combines KV approximation with SD confines the approximation to the **draft** and verifies against an unapproximated target KV, so the guarantee is inherited by construction and almost never tested. The one paper that moved the approximation **into** the verification step reports quality damage and simply drops the claim. I did not find any paper giving a distribution-preservation result for a sparse / evicted / quantised KV used as the SD **target**.

See `kv-eviction-specdecode-exactness-recon.md` for the delegated 21-paper KV-eviction audit with its own verbatim table.
