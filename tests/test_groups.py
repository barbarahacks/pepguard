import numpy as np

from pepguard.groups import similarity_groups


# this test checks that identical peptides land in the same group
def test_identical_peptides_share_a_group():
	groups = similarity_groups(['KWKLFKKIGAVLKVL', 'KWKLFKKIGAVLKVL'])
	assert groups[0] == groups[1]


# this test checks that a one-letter change keeps a peptide with its family
def test_near_duplicates_share_a_group():
	groups = similarity_groups(['KWKLFKKIGAVLKVL', 'KWKLFKKIGAVLKVA'])
	assert groups[0] == groups[1]


# this test checks that very different peptides are kept apart
def test_different_peptides_get_different_groups():
	groups = similarity_groups(['KWKLFKKIGAVLKVL', 'DDDDEEEEDDDDEEE'])
	assert groups[0] != groups[1]


# this test checks that friends of friends join one group (a chain, not just direct pairs)
def test_groups_are_transitive():
	# a~b and b~c are close, but a and c are further apart
	a, b, c = 'KWKLFKKIGAVLKVLKWKLF', 'KWKLFKKIGAVLKVLAAAAA', 'KWKLFKKIGAVLKVLAAAAAGGGGGGGG'
	groups = similarity_groups([a, b, c], cutoff=0.7)
	assert groups[0] == groups[1] == groups[2]


# this test checks that we get one label per peptide and that the real data has many groups
def test_one_label_per_peptide():
	groups = similarity_groups(['KK', 'DD', 'LL', 'KK'])
	assert groups.shape == (4,)
	assert groups[0] == groups[3]
	assert len(np.unique(groups)) == 3
