# Intermediate-action experiments

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

These entry points test different conditions. Their module paths, CLI options,
and result formats stay available for reproducing existing runs.

## Shared implementation

| Module | Responsibility |
| --- | --- |
| `splits.py` | Split A/B task definitions and selection of aligned row arrays |
| `physical_scales.py` | Fit per-axis action-to-motion scales using training episodes |
| `../runtime.py` | Initialize Python, NumPy, and Torch RNGs for the initial light runners |
| `../methods/_common.py` | Shared method input conversion and numerical/memory helpers |

Shared split users import `splits.py` directly. `compare_splits.SPLITS` and
`compare_splits._subset` remain importable for existing consumers. `geo_query.py`
keeps its two-field split definitions and historical Split B task order.

The `fit_physical_scales` functions in `fair_compare.py` and `roundtrip.py`
delegate to the same implementation, retaining their respective default caps of
300 and 400 training episodes. Fitting, seeded sampling, units, and output keys
are unchanged. SciPy is imported only when scales are fitted.

The production LIBERO closed-loop pipeline is documented separately in
[`memory_pilot/anchor_flow/DESIGN.md`](../../../memory_pilot/anchor_flow/DESIGN.md).
