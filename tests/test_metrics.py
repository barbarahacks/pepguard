import numpy as np
import pytest

from pepguard.metrics import apply_threshold, compute_metrics


# this test checks that probabilities at or above the threshold become 1
def test_apply_threshold():
	assert apply_threshold([0.2, 0.5, 0.9], 0.5).tolist() == [0, 1, 1]
	assert apply_threshold([0.2, 0.5, 0.9], 0.8).tolist() == [0, 0, 1]


# this test checks perfect predictions give perfect scores
def test_compute_metrics_perfect():
	m = compute_metrics([0, 0, 1, 1], [0.1, 0.2, 0.8, 0.9])
	assert m['accuracy'] == 1.0
	assert m['sensitivity'] == 1.0
	assert m['specificity'] == 1.0
	assert m['auc'] == 1.0
	assert m['mcc'] == 1.0
	assert m['tp'] == 2


# this test checks the always-yes shortcut the organizers warned about scores badly
def test_compute_metrics_always_yes_is_penalised():
	m = compute_metrics([0, 0, 1, 1], [0.9, 0.9, 0.9, 0.9])
	assert m['accuracy'] == 0.5
	assert m['sensitivity'] == 1.0
	assert m['specificity'] == 0.0
	assert m['mcc'] == 0.0


# this test checks sensitivity and specificity on a mixed case
def test_compute_metrics_mixed():
	# 3 positives (2 caught), 3 negatives (2 rejected)
	m = compute_metrics([1, 1, 1, 0, 0, 0], [0.9, 0.8, 0.3, 0.1, 0.2, 0.7])
	assert m['sensitivity'] == pytest.approx(2 / 3)
	assert m['specificity'] == pytest.approx(2 / 3)
	assert m['accuracy'] == pytest.approx(4 / 6)
	assert m['tp'] == 2


# this test checks that moving the threshold trades sensitivity for specificity
def test_compute_metrics_threshold_tradeoff():
	y = np.array([1, 1, 0, 0])
	p = np.array([0.6, 0.4, 0.3, 0.1])
	assert compute_metrics(y, p, 0.5)['sensitivity'] == 0.5
	assert compute_metrics(y, p, 0.35)['sensitivity'] == 1.0
