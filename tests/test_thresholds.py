import numpy as np
import pytest

from pepguard.thresholds import accuracy_curve, tune_threshold


# this test checks the accuracy curve against hand-counted values
def test_accuracy_curve():
	y = [0, 0, 1, 1]
	p = [0.1, 0.4, 0.6, 0.9]
	acc = accuracy_curve(y, p, grid=[0.05, 0.5, 0.95])
	assert acc.tolist() == [0.5, 1.0, 0.5]


# this test checks that well separated classes get a threshold in the gap between them
def test_tune_threshold_lands_between_the_classes():
	rng = np.random.default_rng(0)
	y = np.array([0] * 100 + [1] * 100)
	p = np.concatenate([rng.uniform(0.0, 0.3, 100), rng.uniform(0.7, 1.0, 100)])
	assert 0.3 <= tune_threshold(y, p) <= 0.7


# this test checks that a model whose scores sit low gets a threshold below 0.5
def test_tune_threshold_moves_when_scores_are_shifted():
	rng = np.random.default_rng(0)
	y = np.array([0] * 100 + [1] * 100)
	p = np.concatenate([rng.uniform(0.0, 0.25, 100), rng.uniform(0.3, 0.55, 100)])
	assert tune_threshold(y, p) < 0.4


# this test checks that on pure noise we fall back toward 0.5 instead of chasing a spike
def test_tune_threshold_on_noise_stays_near_the_middle():
	rng = np.random.default_rng(1)
	y = rng.integers(0, 2, 400)
	p = rng.uniform(0, 1, 400)
	assert tune_threshold(y, p) == pytest.approx(0.5, abs=0.25)


# this test checks that the threshold is always one of the grid values
def test_tune_threshold_stays_inside_the_grid():
	t = tune_threshold([0, 1, 0, 1], [0.2, 0.8, 0.3, 0.7])
	assert 0.05 <= t <= 0.95
