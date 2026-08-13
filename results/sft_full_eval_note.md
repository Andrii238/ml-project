# SFT Full Evaluation Result

Date: 2026-08-13

Command:

```bash
PYTHONPATH=/content/ml_project python -m training.evaluate \
  --checkpoints policy_0=BASE policy_1=./ckpts/sft \
  --samples-per-layout 2 \
  --n-val 20 \
  --out results/eval_sft_vs_base.json
```

Result:

| Policy | Mean reward | Green science/sec | Parse OK | Valid output | Machines | Conveyors | Cells | Materials |
|---|---:|---:|---:|---:|---:|---:|---:|---:|
| policy_0 base | -1.720 | 0.0000 | 100% | 100% | 1.00 | 11.00 | 23.00 | 12.00 |
| policy_1 SFT | 46.089 | 0.6146 | 100% | 100% | 2.95 | 46.60 | 76.15 | 49.55 |

Presentation sentence:

SFT moved the baseline from zero green-science production to 0.6146 science/sec on validation, with 100% parse and validity rate.
