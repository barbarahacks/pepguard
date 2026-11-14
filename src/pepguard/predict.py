import argparse
import time
from pathlib import Path

import joblib
import pandas as pd

from pepguard.feature_sets import build_features
from pepguard.metrics import apply_threshold

# the exact columns the judging script expects, in this order
SUBMISSION_COLUMNS = [
	'sequence',
	'anticancer_probability', 'anticancer_prediction',
	'hemolytic_probability', 'hemolytic_prediction',
	'verdict'
]

# the four verdicts, keyed by (anticancer call, hemolytic call)
VERDICTS = {
	(1, 0): 'Selective',
	(1, 1): 'Works but toxic',
	(0, 1): "Toxic and doesn't work",
	(0, 0): 'Does nothing'
}

MODELS_DIR = Path('models')


# this function loads the saved model bundle for one classifier
def load_bundle(task: str, models_dir: Path = MODELS_DIR) -> dict:
	path = Path(models_dir) / f'{task}.joblib'
	if not path.exists():
		raise FileNotFoundError(f'{path} not found. run `python -m pepguard.train` first.')
	return joblib.load(path)


# this function scores sequences with one saved classifier and returns probability and 0/1 call
def score(sequences, bundle: dict):
	# build the same kind of features the model was trained on, in the same column order
	X = build_features(sequences, bundle['feature_set'])[bundle['feature_columns']]
	proba = bundle['model'].predict_proba(X)[:, 1]
	return proba, apply_threshold(proba, bundle['threshold'])


# this function crosses the two calls into one of the four verdicts
def verdict(anticancer_prediction: int, hemolytic_prediction: int) -> str:
	return VERDICTS[(int(anticancer_prediction), int(hemolytic_prediction))]


# this function runs both classifiers on a table of peptides and builds the submission table
def build_submission(df: pd.DataFrame, models_dir: Path = MODELS_DIR) -> pd.DataFrame:
	if 'sequence' not in df.columns:
		raise ValueError('input csv needs a `sequence` column.')
	
	# keep the sequences exactly as given, since the judge matches on them
	sequences = df['sequence'].astype(str).tolist()
	
	anticancer_proba, anticancer_pred = score(sequences, load_bundle('anticancer', models_dir))
	hemolytic_proba, hemolytic_pred = score(sequences, load_bundle('hemolytic', models_dir))
	
	return pd.DataFrame({
		'sequence': sequences,
		'anticancer_probability': anticancer_proba,
		'anticancer_prediction': anticancer_pred,
		'hemolytic_probability': hemolytic_proba,
		'hemolytic_prediction': hemolytic_pred,
		'verdict': [verdict(a, h) for a, h in zip(anticancer_pred, hemolytic_pred)]
	})[SUBMISSION_COLUMNS]


# this function checks a submission against the rules and returns a list of problems (empty means valid)
def validate_submission(sub: pd.DataFrame, expected_rows: int | None = None) -> list:
	problems = []
	
	if list(sub.columns) != SUBMISSION_COLUMNS:
		return [f'columns must be exactly {SUBMISSION_COLUMNS}, got {list(sub.columns)}']
	if expected_rows is not None and len(sub) != expected_rows:
		problems.append(f'expected {expected_rows} rows, got {len(sub)}')
	if sub.isna().any().any():
		problems.append('submission has missing values')
	
	# probabilities must be between 0 and 1, predictions must be 0 or 1
	for task in ('anticancer', 'hemolytic'):
		if not sub[f'{task}_probability'].between(0, 1).all():
			problems.append(f'{task}_probability must be between 0 and 1')
		if not sub[f'{task}_prediction'].isin([0, 1]).all():
			problems.append(f'{task}_prediction must be 0 or 1')
	
	# the verdict must match the two predictions
	expected = [verdict(a, h) for a, h in zip(sub['anticancer_prediction'], sub['hemolytic_prediction'])]
	if sub['verdict'].tolist() != expected:
		problems.append('verdict does not match the predictions')
	
	return problems


# this function reads the peptides csv, scores them, validates and writes the submission
def run(input_csv: str, output_csv: str, models_dir: Path = MODELS_DIR) -> pd.DataFrame:
	start = time.perf_counter()
	df = pd.read_csv(input_csv)
	sub = build_submission(df, models_dir)
	
	# refuse to write a file the judging script would reject
	problems = validate_submission(sub, expected_rows=len(df))
	if problems:
		raise ValueError('invalid submission: ' + '; '.join(problems))
	
	Path(output_csv).parent.mkdir(parents=True, exist_ok=True)
	sub.to_csv(output_csv, index=False)
	
	print(f'scored {len(sub)} peptides in {time.perf_counter() - start:.1f}s -> {output_csv}')
	print(sub['verdict'].value_counts().to_string())
	return sub


# this function is the command line entry point
def main():
	parser = argparse.ArgumentParser(description='score peptides and write the submission csv')
	parser.add_argument('input_csv', help='csv with a `sequence` column (and optionally `id`)')
	parser.add_argument('--out', default='outputs/submission.csv')
	parser.add_argument('--models', default=str(MODELS_DIR))
	args = parser.parse_args()
	run(args.input_csv, args.out, Path(args.models))


if __name__ == '__main__':
	main()
