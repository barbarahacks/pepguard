import argparse
from pathlib import Path

import numpy as np
import pandas as pd

from pepguard.data import clean_sequence

# esm-2 models we can use: name -> (last layer, embedding size). the weights are already downloaded.
MODELS = {
	'esm2_t6_8M_UR50D': (6, 320),
	'esm2_t12_35M_UR50D': (12, 480),
	'esm2_t30_150M_UR50D': (30, 640)
}

# where torch keeps the downloaded weights
CHECKPOINT_DIR = Path.home() / '.cache' / 'torch' / 'hub' / 'checkpoints'


# this function turns peptides into esm-2 vectors: the average of the model's per-residue vectors
# esm-2 is used only as a feature extractor, our own classifier is trained on these vectors
def embed_sequences(sequences, model_name: str = 'esm2_t12_35M_UR50D', batch_size: int = 32) -> np.ndarray:
	# torch is imported here, only inside the embedding process, so it never loads next to sklearn
	import torch
	import esm
	
	layer, _ = MODELS[model_name]
	
	# the checkpoint pickles an argparse.Namespace, so allow that single class and nothing else
	with torch.serialization.safe_globals([argparse.Namespace]):
		model, alphabet = esm.pretrained.load_model_and_alphabet_local(str(CHECKPOINT_DIR / f'{model_name}.pt'))
	
	device = 'mps' if torch.backends.mps.is_available() else 'cpu'
	model = model.eval().to(device)
	batch_converter = alphabet.get_batch_converter()
	
	# clean the peptides the same way as everywhere else, with the same guard for empty ones
	seqs = [clean_sequence(s) or 'A' for s in sequences]
	
	# sort by length so each batch has little padding, and remember the original order
	order = sorted(range(len(seqs)), key=lambda i: len(seqs[i]))
	vectors = [None] * len(seqs)
	
	for start in range(0, len(order), batch_size):
		idx = order[start:start + batch_size]
		_, _, tokens = batch_converter([(str(i), seqs[i]) for i in idx])
		
		with torch.no_grad():
			reps = model(tokens.to(device), repr_layers=[layer])['representations'][layer].cpu()
		
		# token 0 is the start marker, so residues sit at 1..len; average only over the real residues
		for row, i in enumerate(idx):
			vectors[i] = reps[row, 1:len(seqs[i]) + 1].mean(0).numpy()
	
	return np.stack(vectors)


# this function is the command line entry point used by the subprocess
def main():
	parser = argparse.ArgumentParser(description='embed peptides with esm-2')
	parser.add_argument('--in', dest='input_csv', required=True, help='csv with a `sequence` column')
	parser.add_argument('--out', required=True, help='where to save the .npy matrix')
	parser.add_argument('--model', default='esm2_t12_35M_UR50D', choices=list(MODELS))
	args = parser.parse_args()
	
	sequences = pd.read_csv(args.input_csv)['sequence'].astype(str).tolist()
	np.save(args.out, embed_sequences(sequences, args.model))


if __name__ == '__main__':
	main()
