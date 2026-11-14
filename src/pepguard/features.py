from itertools import product

import numpy as np
import pandas as pd

from pepguard.data import AMINO_ACIDS, clean_sequence

# every ordered pair of amino acids (400 of them), in a fixed order
DIPEPTIDES = [''.join(pair) for pair in product(AMINO_ACIDS, repeat=2)]

# position of each dipeptide in the 400-long vector, for fast lookup
DIPEPTIDE_INDEX = {pair: idx for idx, pair in enumerate(DIPEPTIDES)}

# hydrophobicity scale (positive = water-avoiding, negative = water-loving)
HYDROPHOBICITY = {
	'A': 0.62, 'R': -2.53, 'N': -0.78, 'D': -0.90, 'C': 0.29,
	'Q': -0.85, 'E': -0.74, 'G': 0.48, 'H': -0.40, 'I': 1.38,
	'L': 1.06, 'K': -1.50, 'M': 0.64, 'F': 1.19, 'P': 0.12,
	'S': -0.18, 'T': -0.05, 'W': 0.81, 'Y': 0.26, 'V': 1.08
}

# angle between neighbouring residues in an alpha helix (100 degrees)
HELIX_ANGLE = np.deg2rad(100)

# residue groups used for the fraction features
POSITIVE = 'KR'
NEGATIVE = 'DE'
HYDROPHOBIC = 'AILMFVWC'
AROMATIC = 'FWY'


# this function counts how much of the peptide is made of each of the 20 amino acids
def aac_features(seq: str) -> dict:
	n = len(seq)
	return {f'aac_{aa}': seq.count(aa) / n for aa in AMINO_ACIDS}


# this function counts how much of the peptide is made of each of the 400 amino acid pairs
def dpc_features(seq: str) -> dict:
	counts = np.zeros(len(DIPEPTIDES))
	
	# slide a window of two letters over the sequence
	for i in range(len(seq) - 1):
		counts[DIPEPTIDE_INDEX[seq[i:i + 2]]] += 1
	
	# a one-letter peptide has no pairs, so everything stays zero
	total = max(len(seq) - 1, 1)
	return {f'dpc_{pair}': counts[idx] / total for idx, pair in enumerate(DIPEPTIDES)}


# this function returns how strongly the peptide is positively charged (k/r pull toward cancer cells)
def charge_features(seq: str) -> dict:
	n = len(seq)
	positive = sum(seq.count(aa) for aa in POSITIVE)
	negative = sum(seq.count(aa) for aa in NEGATIVE)
	
	return {
		'net_charge': positive - negative,
		'charge_density': (positive - negative) / n,
		'frac_positive': positive / n,
		'frac_negative': negative / n
	}


# this function returns the average hydrophobicity and the share of hydrophobic and aromatic letters
def hydrophobicity_features(seq: str) -> dict:
	n = len(seq)
	return {
		'avg_hydrophobicity': sum(HYDROPHOBICITY[aa] for aa in seq) / n,
		'frac_hydrophobic': sum(seq.count(aa) for aa in HYDROPHOBIC) / n,
		'frac_aromatic': sum(seq.count(aa) for aa in AROMATIC) / n
	}


# this function measures amphipathicity: how much the peptide has a water-loving side and a water-avoiding side when coiled as a helix
def hydrophobic_moment(seq: str) -> dict:
	n = len(seq)
	angles = np.arange(n) * HELIX_ANGLE
	values = np.array([HYDROPHOBICITY[aa] for aa in seq])
	
	# add up each residue as a vector pointing at its helix angle, then take the length
	x = np.sum(values * np.cos(angles))
	y = np.sum(values * np.sin(angles))
	return {'hydrophobic_moment': float(np.hypot(x, y) / n)}


# this function returns the peptide length
def length_features(seq: str) -> dict:
	return {'length': len(seq)}


# the feature groups in the fixed order they are joined
FEATURE_GROUPS = {
	'length': length_features,
	'charge': charge_features,
	'hydrophobicity': hydrophobicity_features,
	'moment': hydrophobic_moment,
	'aac': aac_features,
	'dpc': dpc_features
}


# this function turns one sequence into a dict of numbers, always with the same keys
def extract_features(seq: str, groups: tuple = tuple(FEATURE_GROUPS)) -> dict:
	seq = clean_sequence(seq)
	
	# guard against a sequence that had no valid letters
	if len(seq) == 0:
		seq = 'A'
	
	feats = {}
	for group in groups:
		feats.update(FEATURE_GROUPS[group](seq))
	return feats


# this function turns many sequences into a table, one row per peptide
def build_feature_matrix(sequences, groups: tuple = tuple(FEATURE_GROUPS)) -> pd.DataFrame:
	return pd.DataFrame([extract_features(s, groups) for s in sequences])
