# CS231N Milestone 2 Presentation (Claude revision)

Self-contained folder for Overleaf import. This is a re-pass over
`presentations/cs231n_milestone2` with the following changes:

- Replaced the broken SSIM panel: `fig3_scaling_curve.png` is now used
  with a `trim`/`clip` to show only the $\Delta$PSNR and absolute-PSNR
  panels in the scaling slide; SSIM is summarized via the cross-domain
  bar figure and the main table.
- Numbers reconciled to `paper/fakgd/fa_kgd.tex` throughout (PSNR/SSIM
  table, at-scale Wilcoxon sweep, oracle ablation, FA-KGD-3D rel. errors).
- Added three slides that surface the paper's main results that were
  missing from the codex draft: $\gamma$-family analysis,
  FA-KGD-3D volumetric ACS pooling, and the superadditive oracle ablation
  ($+1.9$ dB jointly vs.\ $0 + 0.7$ dB separately).
- Method-overview diagram now shows all three components (gain, FPDC,
  M-step) with the closed-loop M-step$\to$gain edge highlighted; codex
  draft was missing the M-step.
- Implementation-status slide now reflects what is actually done at
  milestone 2 (sampler implemented, table + scaling + oracle ablation
  measured) and what remains for the final report.

## Files

- `main.tex` --- Beamer source.
- `figures/` --- PNGs from the paper (`fig1`, `fig3`, `fig4`, `fig7`
  used in this deck; `fig2`, `fig5`, `fig6` included for completeness).

## Overleaf

1. Zip this folder and upload to a new Overleaf project, or upload the
   folder directly.
2. Set `main.tex` as the main document.
3. Compile with `pdfLaTeX`.

The deck follows the same conventions as Milestone 1:
16:9 Beamer, `metropolis` theme, Stanford red accent, local image
assets with relative paths only, and an inline bibliography (no `.bib`
import step required).
