# pepguard 🧬

predicts which peptides kill cancer cells without harming red blood cells (hack 2 the future, biology track).

two classifiers, trained separately on their own datasets:
- anticancer: probability a peptide kills cancer cells
- hemolytic: probability a peptide damages red blood cells

the two predictions are crossed into a verdict: `Selective`, `Works but toxic`, `Toxic and doesn't work` or `Does nothing`.

## docs

- `CODE_GUIDE.md`: the code explained like you're 5, with live demos
- `EXPLAIN.md`: how it works, for the judges' questions
- `RUNBOOK.md`: the drop-day steps

## setup

```bash
python3 -m venv .venv && source .venv/bin/activate
pip install -r requirements.txt
python -m pytest
pip install -e .      # makes `pepguard` importable
```

## train and predict

```bash
# trains both classifiers and saves them to models/
python -m pepguard.train

# scores a csv with a `sequence` column and writes the submission csv
python -m pepguard.predict final_test_sequences.csv --out outputs/submission.csv
```

the output is checked against the official template columns before it is written.

## esm-2 (feature extractor only)

some models use esm-2 embeddings as features. esm-2 is a general protein language model and is only used to turn a peptide into numbers. our own classifier is trained on the labeled data we were given. the embedding runs in a separate process (torch and sklearn in one process can crash on macos) and is cached in `data/cache/`. the esm-2 weights are expected in `~/.cache/torch/hub/checkpoints`.

## layout

```
src/pepguard/   package code
tests/          pytest tests
data/initial/   real training sets
data/demo/      fake data, only for testing the pipeline (never train on it)
models/         saved models (not committed)
outputs/        submission csvs (not committed)
```
