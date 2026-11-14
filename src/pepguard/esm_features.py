import hashlib
import subprocess
import sys
import tempfile
from pathlib import Path

import numpy as np
import pandas as pd

from pepguard.embed import MODELS

CACHE_DIR = Path('data/cache')


# this function gets esm-2 features as a table, running the embedding in a separate process and caching the result
def esm_matrix(sequences, model_name: str = 'esm2_t12_35M_UR50D', cache_dir: Path = CACHE_DIR) -> pd.DataFrame:
	seqs = [str(s) for s in sequences]
	
	# the cache name depends on the model and the exact peptides, so a new peptide list never reuses an old file
	key = hashlib.sha1((model_name + '|' + '\n'.join(seqs)).encode()).hexdigest()[:16]
	path = Path(cache_dir) / f'{model_name}_{key}.npy'
	
	if not path.exists():
		path.parent.mkdir(parents=True, exist_ok=True)
		
		# torch + sklearn in one process can crash on macos, so embed in a child process
		with tempfile.TemporaryDirectory() as tmp:
			csv = Path(tmp) / 'in.csv'
			pd.DataFrame({'sequence': seqs}).to_csv(csv, index=False)
			subprocess.run(
				[sys.executable, '-m', 'pepguard.embed', '--in', str(csv), '--out', str(path), '--model', model_name],
				check=True
			)
	
	vectors = np.load(path)
	return pd.DataFrame(vectors, columns=[f'esm_{i}' for i in range(MODELS[model_name][1])])
