from pathlib import Path

import joblib
import pandas as pd
from sklearn.calibration import CalibratedClassifierCV
from sklearn.ensemble import RandomForestClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.model_selection import StratifiedKFold, cross_val_predict
from sklearn.pipeline import make_pipeline
from sklearn.preprocessing import StandardScaler
from sklearn.svm import SVC

from pepguard.data import load_training_set
from pepguard.features import build_feature_matrix
from pepguard.metrics import compute_metrics

# the two classifiers and the dataset each one learns from
TASKS = {
	'anticancer': 'data/initial/anticancer_train.csv',
	'hemolytic': 'data/initial/hemolytic_train.csv'
}

# same seed everywhere so results can be repeated
SEED = 42

# where the trained models are saved
MODELS_DIR = Path('models')


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
		)
	}


# this function gets out-of-fold probabilities: every peptide is scored by a model that never saw it
def cross_val_probabilities(model, X: pd.DataFrame, y, folds: int = 5):
	skf = StratifiedKFold(n_splits=folds, shuffle=True, random_state=SEED)
	return cross_val_predict(model, X, y, cv=skf, method='predict_proba')[:, 1]


# this function compares all candidate models with cross validation and returns their metrics
def compare_models(X: pd.DataFrame, y, folds: int = 5) -> dict:
	results = {}
	for name, model in candidate_models().items():
		proba = cross_val_probabilities(model, X, y, folds)
		results[name] = {'proba': proba, **compute_metrics(y, proba)}
	return results


# this function picks the model with the best cv accuracy, using auc to break ties
def pick_best(results: dict) -> str:
	return max(results, key=lambda name: (round(results[name]['accuracy'], 4), results[name]['auc']))


# this function trains one classifier end to end and saves it for the prediction script
def train_task(task: str, csv_path: str, models_dir: Path = MODELS_DIR, folds: int = 5) -> dict:
	df = load_training_set(csv_path)
	X = build_feature_matrix(df['sequence'])
	y = df['label'].to_numpy()
	
	# compare models fairly with cross validation and keep the best one
	results = compare_models(X, y, folds)
	best_name = pick_best(results)
	
	# refit the winner on all the data, since only cv was used to judge it
	model = candidate_models()[best_name].fit(X, y)
	
	# save everything predict.py needs, so nothing is retrained at the end
	models_dir.mkdir(parents=True, exist_ok=True)
	bundle = {
		'task': task,
		'model_name': best_name,
		'model': model,
		'feature_columns': list(X.columns),
		'threshold': 0.5,
		'cv_metrics': {k: v for k, v in results[best_name].items() if k != 'proba'}
	}
	joblib.dump(bundle, models_dir / f'{task}.joblib')
	
	return {'results': results, 'best': best_name}


# this function trains both classifiers and prints a comparison table
def main():
	for task, csv_path in TASKS.items():
		print(f'\n--- {task} ---')
		out = train_task(task, csv_path)
		
		for name, res in out['results'].items():
			marker = '*' if name == out['best'] else ' '
			print(
				f'{marker} {name:14s} acc {res["accuracy"]:.3f} | sens {res["sensitivity"]:.3f} | '
				f'spec {res["specificity"]:.3f} | auc {res["auc"]:.3f} | mcc {res["mcc"]:.3f}'
			)
		print(f'saved models/{task}.joblib (best: {out["best"]})')


if __name__ == '__main__':
	main()
