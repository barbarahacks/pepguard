import numpy as np
import pandas as pd

# a peptide with at least this net charge counts as cationic
CATIONIC_CHARGE = 3


# this function upweights the peptides that break the "cationic = hemolytic" rule, so the model can't just lean on charge
def exception_weights(X: pd.DataFrame, y, weight: float = 3.0, cutoff: int = CATIONIC_CHARGE) -> np.ndarray:
	y = np.asarray(y)
	cationic = (X['net_charge'] >= cutoff).to_numpy()
	
	# exceptions: cationic but safe, or not cationic but hemolytic
	exception = (cationic & (y == 0)) | (~cationic & (y == 1))
	return np.where(exception, weight, 1.0)
