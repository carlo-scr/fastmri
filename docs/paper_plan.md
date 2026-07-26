# WACV 2027 Paper Plan

The active plan is
[`docs/sketchsense_proposal.md`](sketchsense_proposal.md).

The previous PVAS plan was retired on 2026-07-25 after a branch-wide code,
results, and literature audit found:

1. ADS already performs training-free active sensing inside a diffusion
   trajectory, so the former “first mid-trajectory” claim was false.
2. AdaSense already performs projected-posterior-variance acquisition across
   several operators, so the former selection-rule claim was not novel.
3. The legacy multi-coil implementation estimated a squared Jacobian row
   energy in a single Fourier basis, not posterior variance under the full
   SENSE candidate operator.
4. Several static/random, particle, seed, noise, and inference controls
   confounded the exploratory results.
5. The experiments were compute-accounted, not matched-NFE.

The original implementation and draft remain in Git history at commit
`dbc5714` for provenance. Existing output files are real but carry the status
`LEGACY-INVALIDATED` until a corrected run reproduces them.

## Current decision

Develop **SketchSense**, an operator-space PSD covariance sketch built from a
few denoiser JVPs along one live trajectory. Its load-bearing experiment is
whether off-diagonal covariance enables joint log-det batches to beat a
corrected marginal-variance policy across both multi-coil MRI and a
natural-image operator.

There is no automatic diagonal-PVAS fallback: if the conditional low-rank and
joint-selection gates fail, the project should be repositioned or withheld
from the WACV Algorithms track. The go/no-go criteria, mandatory baselines,
evidence policy, and calendar are specified in the proposal.
