import time

import numpy as np
import pytest

from pepguard.data import load_training_set
from pepguard.features import (
	aac_features, build_feature_matrix, charge_features, dpc_features,
	extract_features, hydrophobic_moment, hydrophobicity_features, length_features
)


# this test checks that amino acid composition adds up to one
def test_aac_sums_to_one():
	feats = aac_features('KWKLFKKIGAVLKVL')
	assert len(feats) == 20
	assert sum(feats.values()) == pytest.approx(1.0)


# this test checks the exact composition of a tiny peptide
def test_aac_values():
	feats = aac_features('KKA')
	assert feats['aac_K'] == pytest.approx(2 / 3)
	assert feats['aac_A'] == pytest.approx(1 / 3)
	assert feats['aac_W'] == 0


# this test checks that dipeptide composition has 400 entries that add up to one
def test_dpc_sums_to_one():
	feats = dpc_features('KWKLFKKIGAVLKVL')
	assert len(feats) == 400
	assert sum(feats.values()) == pytest.approx(1.0)


# this test checks that pairs are counted in order
def test_dpc_counts_ordered_pairs():
	feats = dpc_features('KKR')
	assert feats['dpc_KK'] == pytest.approx(0.5)
	assert feats['dpc_KR'] == pytest.approx(0.5)
	assert feats['dpc_RK'] == 0


# this test checks that a one-letter peptide has no pairs and does not crash
def test_dpc_single_residue_is_all_zero():
	assert sum(dpc_features('K').values()) == 0


# this test checks net charge counts k/r as positive and d/e as negative
def test_net_charge():
	assert charge_features('KKR')['net_charge'] == 3
	assert charge_features('KDE')['net_charge'] == -1
	assert charge_features('AAAA')['net_charge'] == 0


# this test checks charge density is the charge divided by length
def test_charge_density():
	assert charge_features('KKAA')['charge_density'] == pytest.approx(0.5)


# this test checks hydrophobicity of single letters against the scale
def test_avg_hydrophobicity_single_residue():
	assert hydrophobicity_features('I')['avg_hydrophobicity'] == pytest.approx(1.38)
	assert hydrophobicity_features('R')['avg_hydrophobicity'] == pytest.approx(-2.53)


# this test checks the hydrophobic and aromatic fractions
def test_hydrophobic_and_aromatic_fractions():
	feats = hydrophobicity_features('LLFK')
	assert feats['frac_hydrophobic'] == pytest.approx(0.75)
	assert feats['frac_aromatic'] == pytest.approx(0.25)


# this test checks that one residue's moment is just the size of its hydrophobicity
def test_hydrophobic_moment_single_residue():
	assert hydrophobic_moment('R')['hydrophobic_moment'] == pytest.approx(2.53)


# this test checks that a helix-like pattern has a big moment and a uniform one has none
def test_hydrophobic_moment_is_larger_when_amphipathic():
	# hydrophobic residues repeating every 3.6 positions line up on one helix face
	amphipathic = 'LKKLLKLLKKLLKLLKKL'

	# 18 identical residues cover exactly 5 full helix turns (18 * 100 = 1800 degrees), so the vectors cancel out
	uniform = 'LLLLLLLLLLLLLLLLLL'

	assert hydrophobic_moment(amphipathic)['hydrophobic_moment'] > 0.5
	assert hydrophobic_moment(uniform)['hydrophobic_moment'] == pytest.approx(0, abs=1e-9)


# this test checks the length feature
def test_length():
	assert length_features('KWKL')['length'] == 4


# this test checks that every sequence gives the same keys in the same order
def test_extract_features_same_keys_for_all():
	a = extract_features('KWKLFKKIGAVLKVL')
	b = extract_features('GG')
	assert list(a) == list(b)
	assert len(a) == 1 + 4 + 3 + 1 + 20 + 400


# this test checks that an invalid sequence falls back to a safe default
def test_extract_features_empty_sequence_guard():
	feats = extract_features('123')
	assert feats['length'] == 1
	assert all(np.isfinite(v) for v in feats.values())


# this test checks that we can pick only some feature groups
def test_extract_features_group_selection():
	feats = extract_features('KKLL', groups=('length', 'charge'))
	assert set(feats) == {'length', 'net_charge', 'charge_density', 'frac_positive', 'frac_negative'}


# this test checks the table shape and that nothing is missing
def test_build_feature_matrix_shape_and_no_nan():
	df = build_feature_matrix(['KWKLFKKIGAVLKVL', 'GG', 'RRRR'])
	assert df.shape == (3, 429)
	assert not df.isna().any().any()


# this test checks that the real training set builds fast enough for the final 650 peptides
def test_real_data_features_are_fast():
	df = load_training_set('data/initial/anticancer_train.csv')
	
	start = time.perf_counter()
	matrix = build_feature_matrix(df['sequence'])
	elapsed = time.perf_counter() - start
	
	assert matrix.shape == (len(df), 429)
	assert not matrix.isna().any().any()
	# 964 peptides should take a few seconds at most
	assert elapsed < 10
