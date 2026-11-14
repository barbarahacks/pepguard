import pandas as pd
import pytest

from pepguard.predict import (
	SUBMISSION_COLUMNS, build_submission, run, validate_submission, verdict
)
from pepguard.train import train_task


# this function makes a tiny separable dataset: positives are all k/r, negatives are all d/e
def make_toy_csv(path, n: int = 30):
	pos = ['KRKRKRKR' + 'K' * (i % 4) for i in range(n)]
	neg = ['DEDEDEDE' + 'D' * (i % 4) for i in range(n)]
	pd.DataFrame({'sequence': pos + neg, 'label': [1] * n + [0] * n}).to_csv(path, index=False)


# this fixture trains small anticancer and hemolytic models once and returns where they are saved
@pytest.fixture(scope='module')
def models_dir(tmp_path_factory):
	root = tmp_path_factory.mktemp('toy_models')
	csv = root / 'toy.csv'
	make_toy_csv(csv)
	for task in ('anticancer', 'hemolytic'):
		train_task(task, str(csv), models_dir=root / 'models', folds=3)
	return root / 'models'


# this test checks the verdict table against the four cases in the challenge pdf
def test_verdict_table():
	assert verdict(1, 0) == 'Selective'
	assert verdict(1, 1) == 'Works but toxic'
	assert verdict(0, 1) == "Toxic and doesn't work"
	assert verdict(0, 0) == 'Does nothing'


# this test checks our columns are exactly the ones in the organizers' template
def test_columns_match_the_official_template():
	template = pd.read_csv('data/submission_template.csv')
	assert list(template.columns) == SUBMISSION_COLUMNS


# this test checks the output has the right columns, row count and keeps sequences untouched
def test_build_submission_shape_and_sequences(models_dir):
	df = pd.DataFrame({'id': ['peptide_0001', 'peptide_0002'], 'sequence': ['KRKRKRKR', 'DEDEDEDE']})
	sub = build_submission(df, models_dir)
	
	assert list(sub.columns) == SUBMISSION_COLUMNS
	assert sub['sequence'].tolist() == ['KRKRKRKR', 'DEDEDEDE']
	assert validate_submission(sub, expected_rows=2) == []


# this test checks that a missing sequence column is rejected
def test_build_submission_needs_sequence_column(models_dir):
	with pytest.raises(ValueError, match='sequence'):
		build_submission(pd.DataFrame({'id': ['a']}), models_dir)


# this test checks that the toy model gives the obvious answer and a consistent verdict
def test_predictions_are_sensible(models_dir):
	sub = build_submission(pd.DataFrame({'sequence': ['KRKRKRKR', 'DEDEDEDE']}), models_dir)
	assert sub['anticancer_prediction'].tolist() == [1, 0]
	# both toy models learned the same thing, so the verdicts follow from the calls
	assert sub['verdict'].tolist() == ['Works but toxic', 'Does nothing']


# this test checks that a weird sequence still gets a row instead of crashing
def test_odd_sequences_still_get_a_row(models_dir):
	sub = build_submission(pd.DataFrame({'sequence': ['kkrr', '123', 'K']}), models_dir)
	assert len(sub) == 3
	assert validate_submission(sub, expected_rows=3) == []
	# the sequence is kept exactly as given, including lowercase
	assert sub['sequence'].tolist() == ['kkrr', '123', 'K']


# this test checks that the validator catches each kind of broken submission
def test_validate_submission_catches_problems(models_dir):
	good = build_submission(pd.DataFrame({'sequence': ['KRKRKRKR', 'DEDEDEDE']}), models_dir)
	
	assert 'columns must be exactly' in validate_submission(good.rename(columns={'verdict': 'v'}))[0]
	assert any('expected 5 rows' in p for p in validate_submission(good, expected_rows=5))
	
	bad_proba = good.copy()
	bad_proba.loc[0, 'anticancer_probability'] = 1.5
	assert any('between 0 and 1' in p for p in validate_submission(bad_proba))
	
	bad_verdict = good.copy()
	bad_verdict.loc[0, 'verdict'] = 'Selective'
	assert any('verdict does not match' in p for p in validate_submission(bad_verdict))


# this test rehearses the whole thing on the fake demo file and checks the written csv
def test_run_end_to_end_on_demo_data(models_dir, tmp_path):
	out = tmp_path / 'out' / 'submission.csv'
	run('data/demo/anticancer_demo.csv', str(out), models_dir)
	
	written = pd.read_csv(out)
	assert list(written.columns) == SUBMISSION_COLUMNS
	assert len(written) == 50
	assert validate_submission(written, expected_rows=50) == []
