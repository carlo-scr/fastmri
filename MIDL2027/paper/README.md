# MIDL 2027 paper scaffold (FA-KGD, reframed)

- `main.tex` + `sections/*.tex`, official MIDL 2027 class (`midl.cls`, `jmlr.cls`/`jmlrutils.sty` bundled).
- Bibliography: `../references.bib` (shared with `../revision_plan.tex`).
- Every pending number is `\xx` (renders `XX.XX`); `\pxx` = `+XX.XX`; `\ci` = `[XX.XX, XX.XX]`.
- Orange **author notes** state what each paragraph must argue. Set `\shownotesfalse` in
  `main.tex` to hide them. MIDL 2027 allows LLMs only for minor language edits, so
  the authors write the prose that replaces each note.

Build: `make`. Local BasicTeX lacks `placeins algorithm2e relsize ifoddpage`;
install them (`sudo tlmgr install placeins algorithm2e relsize ifoddpage`) or
build on Overleaf, which has them.
