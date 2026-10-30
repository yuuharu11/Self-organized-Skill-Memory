# Intermediate-action experiments

For setup, required research assets, and the limits of reproducing the recorded
results on another machine, see the [reproduction guide](../../../../../../docs/reproduction.md).
Some later runners still refer to machine-specific defaults beyond the CLI
arguments; consult that guide before launching a full experiment.

Choose the module by the question being tested. Run from the repository root
after activating the project environment, for example:

```bash
python -m skill_moe.skillnet.examples.emergent.light.intermediate.fair_compare --help
```

## Experiment entry points

| Module | Question / role |
| --- | --- |
| `experiment.py` | Initial intermediate-action comparison; also provides row loading and small model training |
| `abstracted_m1.py` | Compare abstracted M1 base/residual action representations |
| `compare_splits.py` | Run intermediate-action baselines on the two held-out-suite splits |
| `clip_retrieval.py` | Compare learned dual-encoder retrieval and subtask-text query variants |
| `task_ambiguity.py` | Diagnose task ambiguity using held-out goal episodes |
| `task_ambiguity_language_free.py` | Repeat ambiguity checks with language-free camera features |
| `geo_query.py` | Diagnose next-command geometry with hindsight targets; an upper-bound diagnostic |
| `denoise_steps.py` | Compare denoising step counts and inference cost |
| `anchor_flow.py` | S1 anchor-flow training and prequential memory insertion |
| `fair_compare.py` | Compare confidence-adaptive anchor starts with the ordinary flow under matched conditions |
| `roundtrip.py` | Trace reconstruction and retrieval errors from a demo through memory to the final action |
| `hier_skills.py` | S3: group short M6 primitives into longer, open-ended trajectory skills |
| `review_tests.py` | R1–R8 intervention checks on transfer, confidence, training paths, and memory size |
| `skill_v2.py` | V2/V2.1 skill-centred action generation and trajectory-skill selection |
| `v2_compare.py` | Compare V2 combinations using frozen review-test checkpoints |
| `selection_review.py` | E0–E3/E6–E7 audit of functional selection, frequency controls, and merging |
| `skill_v3.py` | V3 scene-conditioned latent skills |
| `skill_v3_competition.py` | Competitive experts and V4 geometric canonicalization |
| `skill_v3c2.py` | Open-ended expert count, soft competition, and chance-aware scoring |
| `skill_v3e.py` | E4/E5 representation follow-ups, including balanced assignment |

These entry points test different conditions. Their module paths, CLI options,
and result formats stay available for reproducing existing runs.

## Shared implementation

| Module | Responsibility |
| --- | --- |
| `splits.py` | Split A/B task definitions and selection of aligned row arrays |
| `physical_scales.py` | Fit per-axis action-to-motion scales using training episodes |
| `../runtime.py` | Initialize Python, NumPy, and Torch RNGs for the initial light runners |
| `../methods/_common.py` | Shared method input conversion and numerical/memory helpers |
| `../../research_paths.py` | Checkout root and `SKILLNET_STATE_DIR` for portable research defaults |

Shared split users import `splits.py` directly. `compare_splits.SPLITS` and
`compare_splits._subset` remain importable for existing consumers. `geo_query.py`
keeps its two-field split definitions and historical Split B task order.

The V2/V3 scripts are distinct recorded experiments. Keep their original module
names when reproducing a run, and use the table to choose the intended variant.
Their historical run-directory defaults identify required prior artifacts;
they do not download those artifacts. The input dataset, feature cache,
checkpoints, and prior-run results must be supplied separately.

The `fit_physical_scales` functions in `fair_compare.py` and `roundtrip.py`
delegate to the same implementation, retaining their respective default caps of
300 and 400 training episodes. Fitting, seeded sampling, units, and output keys
are unchanged. SciPy is imported only when scales are fitted.

The production LIBERO closed-loop pipeline is documented separately in
[`memory_pilot/anchor_flow/DESIGN.md`](../../../memory_pilot/anchor_flow/DESIGN.md).
