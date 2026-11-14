from pathlib import Path

import pandas as pd

# the only letters a valid peptide can contain
AMINO_ACIDS = 'ACDEFGHIKLMNPQRSTVWY'

# the columns every labeled dataset must have
REQUIRED_COLUMNS = ['sequence', 'label']


# this function cleans a peptide: strips spaces, uppercases it and drops non-standard letters
def clean_sequence(seq: str) -> str:
	return ''.join(c for c in str(seq).strip().upper() if c in AMINO_ACIDS)


# this function loads a labeled training csv and refuses the fake demo data
def load_training_set(path: str | Path) -> pd.DataFrame:
	path = Path(path)
	
	# the demo files are random fake data and must never be trained on
	if 'demo' in path.parts:
		raise ValueError(f'{path} is fake demo data and must not be used for training.')
	
	df = pd.read_csv(path)
	
	# make sure the expected columns exist
	missing = [col for col in REQUIRED_COLUMNS if col not in df.columns]
	if missing:
		raise ValueError(f'{path} is missing columns: {missing}')
	
	# clean every sequence and drop rows that end up empty
	df['sequence'] = df['sequence'].map(clean_sequence)
	df = df[df['sequence'].str.len() > 0].reset_index(drop=True)
	
	return df
