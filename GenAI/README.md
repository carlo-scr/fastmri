# GenAI use, in plain English

**Course:** CS231N — Deep Learning for Computer Vision (Stanford, Spring 2026)
**Project:** FA-KGD — Frequency-Adaptive Kalman-Guided Diffusion for Accelerated MRI

This folder is where we (the authors) keep an honest paper-trail of how we
used generative AI on the project. It exists for two reasons: (1) Stanford's
[GenAI policy](https://communitystandards.stanford.edu/generative-ai-policy-guidance)
and the Honor Code expect disclosure, and (2) we genuinely want anyone reading
the repo to know which parts of the code and writing were AI-assisted and
which were ours.

## The short version

The science is ours. FA-KGD, the FPDC schedule, the volumetric ACS pooling
trick, the PV-gated active sampling extension, the choice of baselines and
ablations, and the final paper narrative — those came from us.

What AI helped with was the grind: boilerplate, debugging stack traces,
LaTeX wrangling, Colab cells that kept breaking, refactoring once a file got
too long, and the occasional "hey, does this derivation look right to you?"
sanity check. Every AI suggestion got read, edited, and tested before it
landed in a commit. No numbers were made up by an AI; every PSNR/SSIM in the
paper was produced by our code on our machines and copied over by hand.

## Tools we used

| Tool | Mostly used for |
|---|---|
| GitHub Copilot (chat + agent in VS Code) | Day-to-day coding, refactors, multi-file edits, debugging |
| Claude (Sonnet / Opus) | Longer code review, paper prose edits, math sanity checks |
| ChatGPT | Occasional second opinion, BibTeX cleanup |

We didn't share any course-restricted material (assignment starter code,
etc.) with these tools. Everything they saw was either our own work, the
public fastMRI dataset, or the public ADPS checkpoints.

## Per-version changelogs

Each file below covers one rough chunk of the project and says what changed
and how AI was (or wasn't) involved. Dates line up with the git history on
`main`.

- [v1 — Repository scaffold & oracle baseline](CHANGELOG_v1_scaffold_oracle.md) (Apr 17–18)
- [v2 — EDM integration & first realistic results](CHANGELOG_v2_edm_realistic.md) (Apr 18–28)
- [v3 — Multicoil ACS, M-step clamp, hyperparameter sweeps](CHANGELOG_v3_multicoil_sweeps.md) (Apr 28 – May 3)
- [v4 — SENSE forward, PV-gated active sampling (PVAS)](CHANGELOG_v4_pvas.md) (May 4)
- [v5 — Milestone presentations & paper polish](CHANGELOG_v5_milestones.md) (May 19–26)
- [v6 — 192-slice production runs & final numbers](CHANGELOG_v6_production.md) (Jun 3–4)

## Where we stand

To be explicit:

- The ideas in the paper are ours.
- The code in the repo was written, reviewed, and tested by us, with AI
  help on the parts noted in the changelogs.
- No AI tool was used to invent results or generate final paper prose
  verbatim — anything that started as an AI draft got rewritten in our
  voice and checked against the actual numbers.
- The authors take full responsibility for what's in this repo and the
  submitted paper.
