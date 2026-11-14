# drop-day runbook

the final file `final_test_sequences.csv` (columns `id`, `sequence`, about 650 rows) arrives on discord 30 minutes before the 4:00 pm deadline. the upload form is https://forms.gle/fR6RCTNvPrGh4hMg8 and only the last csv counts.

## before the file arrives (do this now)

```bash
cd ~/Projects/intellij/pepguard && source .venv/bin/activate
python -m pytest -q                      # all tests should pass
ls models/ models_fallback/              # anticancer.joblib and hemolytic.joblib in both
```

if `models/` is empty, rebuild it (about 1 minute the first time because of the esm embeddings):

```bash
python -m pepguard.train
python -m pepguard.train --no-esm --models models_fallback    # the safety net, about 20 seconds
```

## when the file arrives

1. download it and put it at `data/final/final_test_sequences.csv`.
2. run the main models (about 15 seconds for 650 peptides):

```bash
python -m pepguard.predict data/final/final_test_sequences.csv --out outputs/submission.csv
```

3. check the printed line says `scored 650 peptides`. the script refuses to write a file that breaks the format rules.
4. open `outputs/submission.csv` and look at the first rows. the header must be exactly:
   `sequence,anticancer_probability,anticancer_prediction,hemolytic_probability,hemolytic_prediction,verdict`
5. upload `outputs/submission.csv` through the google form. submit early, you can resubmit.

## if something fails

| symptom | do this |
|---|---|
| torch or esm error during predict | use the fallback models: `python -m pepguard.predict data/final/final_test_sequences.csv --out outputs/submission.csv --models models_fallback` (hemolytic accuracy drops from about 0.95 to about 0.91, anticancer is unchanged) |
| `models/... not found` | run the training commands above |
| `invalid submission: ...` | the message says which rule broke, fix the input file, do not edit the csv by hand |
| row count is not 650 | check the input has a `sequence` column and no blank lines at the end |

## verdict wording (settled)

the organizers said either spelling of the verdicts is fine and the verdict column's wording is not scored. the judging script needs: a `sequence` column matching the test file, the two predictions as 0 or 1, and the two probabilities between 0 and 1. `predict.py` guarantees all of that before it writes the file.
