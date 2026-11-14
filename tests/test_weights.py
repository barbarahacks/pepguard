import pandas as pd

from pepguard.weights import exception_weights


# this test checks that only the two exception groups get the extra weight
def test_exception_weights():
	X = pd.DataFrame({'net_charge': [5, 5, 0, 0]})
	y = [1, 0, 0, 1]
	# cationic+hemolytic (normal), cationic+safe (exception), plain+safe (normal), plain+hemolytic (exception)
	assert exception_weights(X, y, weight=3.0).tolist() == [1.0, 3.0, 1.0, 3.0]


# this test checks the cutoff for what counts as cationic
def test_exception_weights_cutoff():
	X = pd.DataFrame({'net_charge': [2, 3]})
	y = [0, 0]
	# charge 2 is not cationic, so a safe peptide is normal; charge 3 is cationic, so a safe peptide is an exception
	assert exception_weights(X, y, weight=4.0, cutoff=3).tolist() == [1.0, 4.0]


# this test checks that a weight of one changes nothing
def test_exception_weights_of_one_is_neutral():
	X = pd.DataFrame({'net_charge': [5, 0, 5]})
	assert exception_weights(X, [0, 1, 1], weight=1.0).tolist() == [1.0, 1.0, 1.0]
