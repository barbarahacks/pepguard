import pandas as pd
import pytest

from pepguard.data import clean_sequence, load_training_set


# this test checks that cleaning uppercases and drops bad letters
def test_clean_sequence_removes_invalid_letters():
	assert clean_sequence(' kwklX*fkk ') == 'KWKLFKK'


# this test checks that a fully invalid sequence becomes empty
def test_clean_sequence_all_invalid_is_empty():
	assert clean_sequence('123 ') == ''


# this test checks that a normal csv loads and is cleaned
def test_load_training_set_reads_and_cleans(tmp_path):
	csv = tmp_path / 'train.csv'
	pd.DataFrame({'sequence': ['kkll', 'BZ'], 'label': [1, 0]}).to_csv(csv, index=False)
	
	df = load_training_set(csv)
	
	# the second row has no valid letters so it is dropped
	assert df['sequence'].tolist() == ['KKLL']
	assert df['label'].tolist() == [1]


# this test checks that a csv without the label column is rejected
def test_load_training_set_missing_column(tmp_path):
	csv = tmp_path / 'train.csv'
	pd.DataFrame({'sequence': ['KK']}).to_csv(csv, index=False)
	
	with pytest.raises(ValueError, match='missing columns'):
		load_training_set(csv)


# this test checks that anything under a demo folder is refused
def test_load_training_set_refuses_demo(tmp_path):
	demo_dir = tmp_path / 'demo'
	demo_dir.mkdir()
	csv = demo_dir / 'anticancer_demo.csv'
	pd.DataFrame({'sequence': ['KK'], 'label': [1]}).to_csv(csv, index=False)
	
	with pytest.raises(ValueError, match='fake demo data'):
		load_training_set(csv)


# this test checks that the real training files load with balanced classes
def test_real_training_sets_load():
	anticancer = load_training_set('data/initial/anticancer_train.csv')
	hemolytic = load_training_set('data/initial/hemolytic_train.csv')
	
	assert len(anticancer) == 964
	assert len(hemolytic) == 618
	assert anticancer['label'].value_counts().nunique() == 1
