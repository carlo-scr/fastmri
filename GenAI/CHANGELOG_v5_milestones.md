# v5 — Milestone presentations & paper polish

**Dates:** May 19 – May 26, 2026
**Commits:** `0e49cb2`, `e5c860d`, `4dfee9b`, `2ce6ca6`, `01396ef`,
`e98c701`, `c93519f`, `dfa923c`.

## What changed

- CS231N milestone 1 deck (`presentations/cs231n_milestone1/`).
- CS231N milestone 2 deck in two variants
  (`presentations/codex_cs231n_milestone2/`,
  `presentations/claude_cs231n_milestone2/`) — kept both so I could compare
  AI-suggested narrative flows side-by-side before consolidating my own.
- Method-overview pipeline figure redrawn to match the FigJam whiteboard
  layout I'd been using verbally; calibration row shifted left; EM-loop
  labels fixed (`2ce6ca6`).
- Paper refactor in `dfa923c`: tightened intro/method/results, moved
  ablations into a single subsection, harmonized notation between fakgd
  and pvas drafts.

## GenAI usage

**What was AI-assisted**
- LaTeX Beamer / Tikz scaffolding for the slide decks.
- Two parallel AI-drafted narrative outlines for milestone 2 (one from
  Claude, one from another model — hence the `claude_*` / `codex_*`
  directory names) which I used as *contrasting straw-men*; the final
  delivered narrative was my own consolidation.
- Pipeline-figure SVG/TikZ generation, with iterative back-and-forth to
  match the FigJam reference image.
- Paper prose tightening (sentence-level edits, removing repetition,
  consistency of math symbols).

**What was human-led**
- All scientific claims, ablation choices, and what to put on each slide.
- The decision to expose the FA-KGD-vs-ΠGDM gap honestly (small with
  realistic init; larger only with oracle init) rather than overclaim.
- Every number on every slide was copied from my own JSON sweep outputs,
  not from any AI suggestion.

## Representative prompts

> "Here is my draft milestone-2 outline (8 slides). Suggest two alternative
> orderings: one that leads with the oracle gap, one that leads with the
> realistic-init result. For each, flag the strongest objection a reviewer
> might raise."

> "Convert this FigJam-style pipeline (attached PNG) into a clean TikZ
> figure with three rows (calibration / sampling / DC), keeping the same
> spatial arrangement. Use only black/navy strokes."

> "Read `paper/fakgd/main.tex` section 4 (Method). Find sentences that
> over-claim relative to the numerics in Table 1. Suggest minimal edits
> that preserve voice."
