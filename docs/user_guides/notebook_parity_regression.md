# Notebook Parity Regression

This benchmark is a regression harness for the Split-MNIST continual-learning
port. It does not claim exact notebook reproduction. It provides two stable,
representative profiles that future refactors can re-run:

- `v18_like`
  - teacher-first style
  - later V20 mechanisms muted
- `v20_2b_like`
  - replay-assisted
  - conservative causal / TransWeave style enabled

## Files

- Runner: [notebook_parity_regression.py](/home/ni/repos/freeFabricPC/FabricPC/examples/notebook_parity_regression.py)
- Harness: [parity.py](/home/ni/repos/freeFabricPC/FabricPC/fabricpc/continual/parity.py)
- Baselines: [notebook_parity_baselines.json](/home/ni/repos/freeFabricPC/FabricPC/benchmarks/notebook_parity_baselines.json)

## Run

```bash
python examples/notebook_parity_regression.py
```

The same benchmark is enforced in CI by
[notebook-parity.yml](/home/ni/repos/freeFabricPC/FabricPC/.github/workflows/notebook-parity.yml)
on `push`, `pull_request`, and `workflow_dispatch`.

Run a single profile:

```bash
python examples/notebook_parity_regression.py --profile v20_2b_like
```

Update the checked-in baseline after an intentional benchmark refresh:

```bash
python examples/notebook_parity_regression.py --update-baseline
```

## What Is Checked

The harness compares observed metrics to the baseline with explicit tolerances:

- `final_mean_accuracy`
- `average_forgetting`
- `support_diversity`
- `mean_test_accuracy`
- `mean_causal_examples`
- `mean_causal_mix_gate`
- `mean_transweave_sources`

Training time is recorded but not used as a pass/fail gate.

## Notes

- The benchmark uses the current FabricPC port, not the original notebooks.
- The profiles are meant to preserve the *control regime distinction* described
  in the notebooks and the V18 vs V20.2b PDF.
- Small run-to-run variation is expected, so tolerances are intentionally not
  zero.
