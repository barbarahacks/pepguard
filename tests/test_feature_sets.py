import numpy as np
import pandas as pd
import pytest

from pepguard.embed import CHECKPOINT_DIR, MODELS
from pepguard.esm_features import esm_matrix
from pepguard.feature_sets import build_features

# the esm tests need the downloaded weights and a few seconds to load the model
needs_esm = pytest.mark.skipif(
	not (CHECKPOINT_DIR / 'esm2_t6_8M_UR50D.pt').exists(), reason='esm-2 weights are not downloaded'
)


# this test checks the hand-crafted feature set has the expected size
def test_hand_features():
	assert build_features(['KKLL', 'GG'], 'hand').shape == (2, 429)


# this test checks that an unknown feature set is rejected
def test_unknown_feature_set_raises():
	with pytest.raises(KeyError):
		build_features(['KK'], 'nope')


# this test checks that the model table lists the sizes we rely on
def test_model_sizes():
	assert MODELS['esm2_t12_35M_UR50D'][1] == 480
	assert MODELS['esm2_t30_150M_UR50D'][1] == 640


# this test checks esm features have the right shape, are finite and are cached on the second call
@needs_esm
def test_esm_matrix_shape_and_cache(tmp_path):
	seqs = ['KWKLFKKIGAVLKVL', 'GG', 'DEDEDE']
	first = esm_matrix(seqs, 'esm2_t6_8M_UR50D', cache_dir=tmp_path)
	
	assert first.shape == (3, 320)
	assert np.isfinite(first.to_numpy()).all()
	assert len(list(tmp_path.glob('*.npy'))) == 1
	
	# the second call must reuse the file and give identical numbers
	second = esm_matrix(seqs, 'esm2_t6_8M_UR50D', cache_dir=tmp_path)
	pd.testing.assert_frame_equal(first, second)
	assert len(list(tmp_path.glob('*.npy'))) == 1


# this test checks different peptides get different embeddings and a different cache file
@needs_esm
def test_esm_matrix_different_peptides_differ(tmp_path):
	a = esm_matrix(['KWKLFKKIGAVLKVL', 'DEDEDE'], 'esm2_t6_8M_UR50D', cache_dir=tmp_path)
	assert not np.allclose(a.iloc[0], a.iloc[1])
	esm_matrix(['AAAA'], 'esm2_t6_8M_UR50D', cache_dir=tmp_path)
	assert len(list(tmp_path.glob('*.npy'))) == 2
