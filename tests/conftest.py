import numpy as np
import pandas as pd
import pytest

from pepguard.groups import similarity_groups


# this fixture returns a function that writes a small, learnable toy dataset to a csv
@pytest.fixture(scope='session')
def make_toy_csv():
	# this function builds peptides that are mostly k/r (positives) or mostly d/e (negatives), padded with random letters
	def build(path, n: int = 40, seed: int = 0):
		rng = np.random.default_rng(seed)
		others = list('ACFGHILMNPQSTVWY')
		
		def peptide(core: str) -> str:
			length = int(rng.integers(10, 25))
			return ''.join(
				rng.choice(list(core)) if rng.random() < 0.7 else rng.choice(others) for _ in range(length)
			)
		
		pos = [peptide('KR') for _ in range(n)]
		neg = [peptide('DE') for _ in range(n)]
		df = pd.DataFrame({'sequence': pos + neg, 'label': [1] * n + [0] * n})
		df.to_csv(path, index=False)
		
		# the toy peptides must form enough groups for grouped cv to work
		assert len(np.unique(similarity_groups(df['sequence']))) >= 10
		return df
	
	return build
