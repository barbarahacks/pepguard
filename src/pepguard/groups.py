import numpy as np
from rapidfuzz import fuzz, process
from scipy.sparse import csr_matrix
from scipy.sparse.csgraph import connected_components

# peptides at least this similar (0 to 1) are treated as the same family
SIMILARITY_CUTOFF = 0.6


# this function puts similar peptides in the same group, so cv never trains and tests on relatives
def similarity_groups(sequences, cutoff: float = SIMILARITY_CUTOFF) -> np.ndarray:
	seqs = [str(s) for s in sequences]
	
	# similarity between every pair of peptides, from 0 (nothing alike) to 1 (identical)
	sim = process.cdist(seqs, seqs, scorer=fuzz.ratio, dtype=np.float32, workers=-1) / 100
	
	# link every pair above the cutoff (a peptide is always identical to itself, so drop the diagonal)
	links = sim >= cutoff
	np.fill_diagonal(links, False)
	
	# friends of friends end up in one group (a chain a~b~c makes one family)
	_, labels = connected_components(csr_matrix(links), directed=False)
	return labels
