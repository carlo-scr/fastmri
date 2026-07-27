# RaoSense WACV working paper

The imaging result cells are intentionally marked `PENDING`.  Build with:

```bash
make
```

The main text target is eight pages before references.  `main.log` prints
`RAOSENSE_REFERENCES_START_PAGE=<n>` after floats are flushed.

Regenerate vector figures with:

```bash
python ../../scripts/make_raosense_paper_figures.py
```

Do not replace pending cells with legacy PVAS/SketchSense numbers.  The
production protocol is in `../../docs/raosense_experiments.md`, and the exact
mathematical contract is in `../../docs/raosense_theory.md`.
