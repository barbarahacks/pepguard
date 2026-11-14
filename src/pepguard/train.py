import argparse
from pathlib import Path

import joblib
import pandas as pd
from sklearn.calibration import CalibratedClassifierCV
from sklearn.ensemble import RandomForestClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.model_selection import StratifiedGroupKFold, StratifiedKFold, cross_val_predict
from sklearn.pipeline import make_pipeline
from sklearn.preprocessing import StandardScaler
from sklearn.svm import SVC

from pepguard.data import load_training_set
from pepguard.feature_sets import build_features
from pepguard.groups import similarity_groups
from pepguard.metrics import compute_metrics
from pepguard.thresholds import tune_threshold
from pepguard.weights import exception_weights

# the two classifiers and the dataset each one learns from
TASKS = {
	'anticancer': 'data/initial/anticancer_train.csv',
	'hemolytic': 'data/initial/hemolytic_train.csv'
}

# how much to upweight the exception peptides, per task (1 means no weighting)
# the hemolytic set is dominated by "cationic = hemolytic", so the exceptions get extra weight
EXCEPTION_WEIGHT = {'hemolytic': 3.0}

# same seed everywhere so results can be repeated
SEED = 42

# where the trained models are saved
MODELS_DIR = Path('models')


# which feature set each candidate model uses (esm-2 is only used as a feature extractor)
CANDIDATE_FEATURES = {
	'logreg': 'hand',
	'random_forest': 'hand',
	'svm_rbf': 'hand',
	'esm150_logreg': 'esm150'
}


# this function builds the models we compare, each with its own scaler so scaling happens inside every cv fold
def candidate_models() -> dict:
	return {
		'logreg': make_pipeline(StandardScaler(), LogisticRegression(max_iter=2000, C=1.0)),
		'random_forest': make_pipeline(
			StandardScaler(), RandomForestClassifier(n_estimators=400, random_state=SEED, n_jobs=-1)
		),
		# the svm needs calibration to turn its scores into probabilities
		'svm_rbf': make_pipeline(
			StandardScaler(),
			CalibratedClassifierCV(SVC(kernel='rbf', C=2.0, gamma='scale'), ensemble=False)
		),
		# strong regularisation because the embeddings have 640 columns and we only have a few hundred peptides
		'esm150_logreg': make_pipeline(StandardScaler(), LogisticRegression(max_iter=5000, C=0.03))
	}


# this function gets out-of-fold probabilities: every peptide is scored by a model that never saw it
def cross_val_probabilities(model, X: pd.DataFrame, y, groups=None, folds: int = 5, sample_weight=None):
	# with groups, similar peptides stay on the same side of the split so cv can't peek at relatives
	if groups is not None:
		cv = StratifiedGroupKFold(n_splits=folds, shuffle=True, random_state=SEED)
	else:
		cv = StratifiedKFold(n_splits=folds, shuffle=True, random_state=SEED)
	
	# the weights are passed to the last step of the pipeline, which is the classifier
	params = {fit_weight_name(model): sample_weight} if sample_weight is not None else None
	return cross_val_predict(model, X, y, cv=cv, groups=groups, method='predict_proba', params=params)[:, 1]


# this function names the fit argument that carries sample weights to the classifier inside a pipeline
def fit_weight_name(model) -> str:
	return f'{model.steps[-1][0]}__sample_weight'


# this function compares all candidate models with cross validation and returns their metrics
def compare_models(features: dict, y, groups=None, folds: int = 5, sample_weight=None) -> dict:
	results = {}
	for name, model in candidate_models().items():
		# skip candidates whose feature set was not built for this run
		if CANDIDATE_FEATURES[name] not in features:
			continue
		
		X = features[CANDIDATE_FEATURES[name]]
		proba = cross_val_probabilities(model, X, y, groups, folds, sample_weight)
		results[name] = {'proba': proba, **compute_metrics(y, proba)}
	return results


# this function picks the model with the best cv accuracy, using auc to break ties
def pick_best(results: dict) -> str:
	return max(results, key=lambda name: (round(results[name]['accuracy'], 4), results[name]['auc']))


# this function trains one classifier end to end and saves it for the prediction script
def train_task(
	task: str, csv_path: str, models_dir: Path = MODELS_DIR, folds: int = 5, use_esm: bool = True
) -> dict:
	df = load_training_set(csv_path)
	y = df['label'].to_numpy()
	
	# build only the feature sets that some candidate needs (esm needs torch and takes a minute the first time)
	needed = {fs for fs in CANDIDATE_FEATURES.values() if use_esm or not fs.startswith('esm')}
	features = {fs: build_features(df['sequence'], fs) for fs in needed}
	hand = features['hand']
	
	# group similar peptides so the cv estimate is honest
	groups = similarity_groups(df['sequence'])
	
	# upweight exception peptides for the tasks that need it (the weights only depend on charge and label)
	weight = EXCEPTION_WEIGHT.get(task, 1.0)
	sample_weight = exception_weights(hand, y, weight) if weight != 1.0 else None
	
	# compare models fairly with grouped cross validation and keep the best one
	results = compare_models(features, y, groups, folds, sample_weight)
	best_name = pick_best(results)
	
	# tune the decision threshold on the winner's out-of-fold probabilities, never on the training fit
	threshold = tune_threshold(y, results[best_name]['proba'])
	tuned = compute_metrics(y, results[best_name]['proba'], threshold)
	
	# refit the winner on all the data, since only cv was used to judge it
	feature_set = CANDIDATE_FEATURES[best_name]
	X = features[feature_set]
	model = candidate_models()[best_name]
	if sample_weight is not None:
		model.fit(X, y, **{fit_weight_name(model): sample_weight})
	else:
		model.fit(X, y)
	
	# save everything predict.py needs, so nothing is retrained at the end
	models_dir.mkdir(parents=True, exist_ok=True)
	bundle = {
		'task': task,
		'model_name': best_name,
		'model': model,
		'feature_set': feature_set,
		'feature_columns': list(X.columns),
		'threshold': threshold,
		'cv_metrics': tuned
	}
	joblib.dump(bundle, models_dir / f'{task}.joblib')
	
	return {'results': results, 'best': best_name, 'threshold': threshold, 'tuned': tuned}


# this function reads the command line options
def parse_args(argv=None):
	parser = argparse.ArgumentParser(description='train the anticancer and hemolytic classifiers')
	parser.add_argument('--models', default=str(MODELS_DIR), help='folder to save the models in')
	parser.add_argument('--no-esm', action='store_true', help='skip esm-2 features (fallback if torch or the weights fail)')
	return parser.parse_args(argv)


# this function trains both classifiers and prints a comparison table
def main(argv=None):
	args = parse_args(argv)
	
	for task, csv_path in TASKS.items():
		print(f'\n--- {task} (grouped cv) ---')
		out = train_task(task, csv_path, models_dir=Path(args.models), use_esm=not args.no_esm)
		
		for name, res in out['results'].items():
			marker = '*' if name == out['best'] else ' '
			print(
				f'{marker} {name:14s} acc {res["accuracy"]:.3f} | sens {res["sensitivity"]:.3f} | '
				f'spec {res["specificity"]:.3f} | auc {res["auc"]:.3f} | mcc {res["mcc"]:.3f}'
			)
		
		tuned = out['tuned']
		print(f'  threshold {out["threshold"]:.2f} -> acc {tuned["accuracy"]:.3f} | sens {tuned["sensitivity"]:.3f} | spec {tuned["specificity"]:.3f}')
		print(f'saved {args.models}/{task}.joblib (best: {out["best"]}, features: {CANDIDATE_FEATURES[out["best"]]})')


if __name__ == '__main__':
	main()
