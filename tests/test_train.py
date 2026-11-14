import joblib
import numpy as np
import pandas as pd

from pepguard.features import build_feature_matrix
from pepguard.train import (
	candidate_models, compare_models, cross_val_probabilities, pick_best, train_task
)


# this function makes a tiny separable dataset: positives are all k/r, negatives are all d/e
def make_toy_csv(path, n: int = 40):
	rng = np.random.default_rng(0)
	pos = [''.join(rng.choice(list('KR'), size=rng.integers(8, 14))) for _ in range(n)]
	neg = [''.join(rng.choice(list('DE'), size=rng.integers(8, 14))) for _ in range(n)]
	pd.DataFrame({'sequence': pos + neg, 'label': [1] * n + [0] * n}).to_csv(path, index=False)


# this test checks that all three candidates exist and each carries its own scaler
def test_candidate_models():
	models = candidate_models()
	assert set(models) == {'logreg', 'random_forest', 'svm_rbf'}
	assert all(model.steps[0][0] == 'standardscaler' for model in models.values())


# this test checks that cv probabilities are valid and one per peptide
def test_cross_val_probabilities_shape_and_range(tmp_path):
	csv = tmp_path / 'toy.csv'
	make_toy_csv(csv)
	df = pd.read_csv(csv)
	X, y = build_feature_matrix(df['sequence']), df['label'].to_numpy()
	
	proba = cross_val_probabilities(candidate_models()['logreg'], X, y, folds=3)
	assert proba.shape == (len(df),)
	assert ((proba >= 0) & (proba <= 1)).all()


# this test checks that cv finds the obvious pattern and that pick_best returns a real model name
def test_compare_models_learns_a_clear_signal(tmp_path):
	csv = tmp_path / 'toy.csv'
	make_toy_csv(csv)
	df = pd.read_csv(csv)
	X, y = build_feature_matrix(df['sequence']), df['label'].to_numpy()
	
	results = compare_models(X, y, folds=3)
	best = pick_best(results)
	assert best in results
	assert results[best]['accuracy'] > 0.95


# this test checks that ties on accuracy are broken by auc
def test_pick_best_breaks_ties_with_auc():
	results = {'a': {'accuracy': 0.9, 'auc': 0.90}, 'b': {'accuracy': 0.9, 'auc': 0.95}}
	assert pick_best(results) == 'b'


# this test checks that a trained model is saved with everything predict.py needs
def test_train_task_saves_a_loadable_bundle(tmp_path):
	csv = tmp_path / 'toy.csv'
	make_toy_csv(csv)
	
	train_task('toy', str(csv), models_dir=tmp_path / 'models', folds=3)
	bundle = joblib.load(tmp_path / 'models' / 'toy.joblib')
	
	assert {'model', 'model_name', 'feature_columns', 'threshold', 'cv_metrics'} <= set(bundle)
	assert len(bundle['feature_columns']) == 429
	
	# the loaded model must score new peptides using the saved columns
	new = build_feature_matrix(['KRKRKR', 'DEDEDE'])[bundle['feature_columns']]
	proba = bundle['model'].predict_proba(new)[:, 1]
	assert proba[0] > 0.5 > proba[1]
