# Changelog

## Unreleased (v1.1.0)

- Use the posterior mean for deterministic validation, reconstruction, and mode extraction.
- Add group-balanced reconstruction loss through per-muscle motor-unit counts.
- Add five-fold blocked temporal cross-validation with 70/10/20 partitions,
  boundary gaps, pooled out-of-fold explained variance, and selectable global
  or training-only normalization.
- Add individualized mode selection using three elbow heuristics and majority voting.
- Add Integrated Gradients attribution for motor-unit and muscle contributions.
- Add force–mode mutual-information analysis with lag selection and circular-shift surrogates.
- Repair repository metadata, citation links, requirements encoding, and demo imports.
