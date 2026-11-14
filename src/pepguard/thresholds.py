import numpy as np

# the thresholds we try: 0.05, 0.06 ... 0.95
GRID = np.round(np.arange(0.05, 0.951, 0.01), 2)


# this function returns the accuracy at every threshold in the grid
def accuracy_curve(y_true, proba, grid=GRID) -> np.ndarray:
	y_true = np.asarray(y_true)
	proba = np.asarray(proba)
	return np.array([np.mean((proba >= t).astype(int) == y_true) for t in grid])


# this function picks the threshold with the best accuracy on out-of-fold probabilities
def tune_threshold(y_true, proba, grid=GRID, window: int = 7) -> float:
	acc = accuracy_curve(y_true, proba, grid)
	
	# smooth the curve so one lucky spike doesn't win, we want the middle of a good plateau
	padded = np.pad(acc, window // 2, mode='edge')
	smooth = np.convolve(padded, np.ones(window) / window, mode='valid')
	
	# best smoothed accuracy wins, and ties go to the threshold closest to 0.5
	best = max(range(len(grid)), key=lambda i: (round(smooth[i], 4), -abs(grid[i] - 0.5)))
	return float(grid[best])
