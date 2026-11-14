import pandas as pd

from pepguard.esm_features import esm_matrix
from pepguard.features import build_feature_matrix

# the named blocks of features a model can use. esm-2 is only a feature extractor here.
FEATURE_PARTS = {
	'hand': lambda seqs: build_feature_matrix(seqs),
	'esm35': lambda seqs: esm_matrix(seqs, 'esm2_t12_35M_UR50D'),
	'esm150': lambda seqs: esm_matrix(seqs, 'esm2_t30_150M_UR50D')
}


# this function builds a feature table from a name like 'hand', 'esm150' or 'hand+esm150'
def build_features(sequences, feature_set: str = 'hand') -> pd.DataFrame:
	seqs = [str(s) for s in sequences]
	
	# build each named block and put them side by side
	parts = [FEATURE_PARTS[name](seqs) for name in feature_set.split('+')]
	return pd.concat(parts, axis=1)
