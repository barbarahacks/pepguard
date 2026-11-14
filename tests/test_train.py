import joblib
import numpy as np
import pandas as pd

from pepguard.features import build_feature_matrix
from pepguard.train import (
	CANDIDATE_FEATURES, candidate_models, compare_models, cross_val_probabilities, pick_best, train_task
)


# this test checks that all three candidates exist and each carries its own scaler
def test_candidate_models():
	models = candidate_models()
	assert set(models) == {'logreg', 'random_forest', 'svm_rbf', 'esm150_logreg'}
	assert set(models) == set(CANDIDATE_FEATURES)
	assert all(model.steps[0][0] == 'standardscaler' for model in models.values())


# this test checks that cv probabilities are valid and one per peptide
def test_cross_val_probabilities_shape_and_range(tmp_path, make_toy_csv):
	csv = tmp_path / 'toy.csv'
	make_toy_csv(csv)
	df = pd.read_csv(csv)
	X, y = build_feature_matrix(df['sequence']), df['label'].to_numpy()
	
	proba = cross_val_probabilities(candidate_models()['logreg'], X, y, folds=3)
	assert proba.shape == (len(df),)
	assert ((proba >= 0) & (proba <= 1)).all()


# this test checks that cv runs with groups and still gives one probability per peptide
def test_grouped_cv_runs_with_groups(tmp_path, make_toy_csv):
	csv = tmp_path / 'toy.csv'
	make_toy_csv(csv)
	df = pd.read_csv(csv)
	X, y = build_feature_matrix(df['sequence']), df['label'].to_numpy()
	
	# four peptides per group
	groups = np.arange(len(df)) // 4
	proba = cross_val_probabilities(candidate_models()['logreg'], X, y, groups=groups, folds=3)
	assert proba.shape == (len(df),)


# this test checks that cv finds the obvious pattern and that pick_best returns a real model name
def test_compare_models_learns_a_clear_signal(tmp_path, make_toy_csv):
	csv = tmp_path / 'toy.csv'
	make_toy_csv(csv)
	df = pd.read_csv(csv)
	X, y = build_feature_matrix(df['sequence']), df['label'].to_numpy()
	
	results = compare_models({'hand': X}, y, folds=3)
	best = pick_best(results)
	assert best in results
	assert results[best]['accuracy'] > 0.95


# this test checks that ties on accuracy are broken by auc
def test_pick_best_breaks_ties_with_auc():
	results = {'a': {'accuracy': 0.9, 'auc': 0.90}, 'b': {'accuracy': 0.9, 'auc': 0.95}}
	assert pick_best(results) == 'b'


# this test checks that a trained model is saved with everything predict.py needs
def test_train_task_saves_a_loadable_bundle(tmp_path, make_toy_csv):
	csv = tmp_path / 'toy.csv'
	make_toy_csv(csv)
	
	train_task('toy', str(csv), models_dir=tmp_path / 'models', folds=3, use_esm=False)
	bundle = joblib.load(tmp_path / 'models' / 'toy.joblib')
	
	assert {'model', 'model_name', 'feature_set', 'feature_columns', 'threshold', 'cv_metrics'} <= set(bundle)
	assert bundle['feature_set'] == 'hand'
	assert 0.05 <= bundle['threshold'] <= 0.95
	assert len(bundle['feature_columns']) == 429
	
	# the loaded model must score new peptides using the saved columns
	new = build_feature_matrix(['KRKRKR', 'DEDEDE'])[bundle['feature_columns']]
	proba = bundle['model'].predict_proba(new)[:, 1]
	assert proba[0] > 0.5 > proba[1]


# this test checks that cv and the final fit accept sample weights for every candidate model
def test_models_accept_sample_weights(tmp_path, make_toy_csv):
	csv = tmp_path / 'toy.csv'
	make_toy_csv(csv)
	df = pd.read_csv(csv)
	X, y = build_feature_matrix(df['sequence']), df['label'].to_numpy()
	weights = np.where(np.arange(len(y)) % 5 == 0, 3.0, 1.0)
	
	for name, model in candidate_models().items():
		proba = cross_val_probabilities(model, X, y, folds=3, sample_weight=weights)
		assert proba.shape == (len(y),), name
		assert ((proba >= 0) & (proba <= 1)).all(), name


# this test checks that the hemolytic task is weighted and anticancer is not
def test_only_hemolytic_is_weighted():
	from pepguard.train import EXCEPTION_WEIGHT
	assert EXCEPTION_WEIGHT.get('hemolytic', 1.0) > 1.0
	assert EXCEPTION_WEIGHT.get('anticancer', 1.0) == 1.0


# this test checks that candidates whose features were not built are skipped
def test_compare_models_skips_candidates_without_features(tmp_path, make_toy_csv):
	csv = tmp_path / 'toy.csv'
	make_toy_csv(csv)
	df = pd.read_csv(csv)
	X, y = build_feature_matrix(df['sequence']), df['label'].to_numpy()
	
	results = compare_models({'hand': X}, y, folds=3)
	assert 'esm150_logreg' not in results
	assert {'logreg', 'random_forest', 'svm_rbf'} == set(results)


# this test checks the command line defaults and the fallback flags
def test_parse_args():
	from pepguard.train import parse_args
	
	defaults = parse_args([])
	assert defaults.models == 'models'
	assert defaults.no_esm is False
	
	fallback = parse_args(['--no-esm', '--models', 'models_fallback'])
	assert fallback.no_esm is True
	assert fallback.models == 'models_fallback'
