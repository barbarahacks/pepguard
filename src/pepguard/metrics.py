import numpy as np
from sklearn.metrics import accuracy_score, matthews_corrcoef, roc_auc_score


# this function turns probabilities into 0/1 calls using a threshold
def apply_threshold(proba, threshold: float = 0.5) -> np.ndarray:
	return (np.asarray(proba) >= threshold).astype(int)


# this function scores predictions with the judging metrics plus sensitivity and specificity
def compute_metrics(y_true, proba, threshold: float = 0.5) -> dict:
	y_true = np.asarray(y_true)
	pred = apply_threshold(proba, threshold)
	
	# count the four outcomes of a yes/no call
	tp = int(np.sum((y_true == 1) & (pred == 1)))
	tn = int(np.sum((y_true == 0) & (pred == 0)))
	fp = int(np.sum((y_true == 0) & (pred == 1)))
	fn = int(np.sum((y_true == 1) & (pred == 0)))
	
	return {
		'accuracy': accuracy_score(y_true, pred),
		# sensitivity: share of real positives we caught
		'sensitivity': tp / (tp + fn) if (tp + fn) else float('nan'),
		# specificity: share of real negatives we correctly rejected
		'specificity': tn / (tn + fp) if (tn + fp) else float('nan'),
		'auc': roc_auc_score(y_true, proba),
		'mcc': matthews_corrcoef(y_true, pred),
		'tp': tp
	}
