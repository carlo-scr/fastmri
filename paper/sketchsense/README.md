# SketchSense paper scaffold

This directory is the only active WACV 2027 manuscript. The earlier
`paper/pvas*` drafts are historical and their numerical claims are invalidated
by `docs/protocol_audit.md`.

The source intentionally contains `PENDING` result cells. Do not replace them
until a corrected artifact has passed the evidence policy in
`docs/sketchsense_proposal.md`.

Build from this directory with the repository's existing WACV placeholder
style:

```bash
make
```

Before submission:

1. replace the placeholder style with the official WACV 2027 author kit;
2. insert the assigned paper ID and keep anonymous review mode enabled;
3. populate tables only through a reviewed result-generation script;
4. remove repository URLs, local paths, and other deanonymizing metadata;
5. verify the eight-page main-paper limit and the no-rebuttal Round 2 policy.
