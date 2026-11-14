# pepguard

predicts which peptides kill cancer cells without harming red blood cells (hack 2 the future, biology track).

two classifiers, trained separately on their own datasets:
- anticancer: probability a peptide kills cancer cells
- hemolytic: probability a peptide damages red blood cells

the two predictions are crossed into a verdict: `Selective`, `Works but toxic`, `Toxic and doesn't work` or `Does nothing`.

## setup

```bash
python3 -m venv .venv && source .venv/bin/activate
pip install -r requirements.txt
python -m pytest
```

## layout

```
src/pepguard/   package code
tests/          pytest tests
data/initial/   real training sets
data/demo/      fake data, only for testing the pipeline (never train on it)
models/         saved models (not committed)
outputs/        submission csvs (not committed)
```
