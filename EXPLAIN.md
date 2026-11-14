# how pepguard works (for the judges' questions)

## the problem in one line
given a peptide (a string of amino acid letters), predict two things: does it kill cancer cells, and does it damage red blood cells. the two answers are crossed into a verdict: Selective, Works but toxic, Toxic and doesn't work, Does nothing.

## the data
- anticancer set: 964 lab-tested peptides, 482 positive and 482 negative.
- hemolytic set: 618 peptides, 309 positive and 309 negative.
- the two sets come from different sources and each model is trained only on its own set. the fake demo files are never used for training (`load_training_set` refuses that folder).

## how we judge ourselves honestly
- many peptides in each set are near-copies of each other (421 of 964 anticancer peptides have a neighbour that is at least 80% similar). if a near-copy of a test peptide is in the training set, accuracy looks better than it is.
- so cross validation uses groups: peptides that are at least 60% similar (`src/pepguard/groups.py`) always stay on the same side of the split. normal cross validation said 0.76 and 0.95; grouped cross validation says 0.67 and 0.89 before our improvements. the grouped numbers are the honest ones.
- scaling is done inside each fold (a sklearn Pipeline), so no information leaks from the test fold.
- a shuffled-label check gave 0.50, so the pipeline itself is not leaking.

## the features (`src/pepguard/features.py`)
- length.
- charge: net charge (K and R count +1, D and E count -1), charge density, positive and negative fractions. cancer cells are more negatively charged, so positive peptides are pulled toward them.
- hydrophobicity: average hydrophobicity, hydrophobic fraction, aromatic fraction.
- hydrophobic moment: how well the peptide splits into a water-loving face and a water-avoiding face when coiled as a helix. this is what lets a peptide punch into a membrane.
- amino acid composition (20 numbers) and dipeptide composition (400 numbers, how often each pair of letters appears).

## the models (`src/pepguard/train.py`)
each run compares logistic regression, random forest, an rbf svm and logistic regression on esm-2 features. the winner per task, by grouped cross validation accuracy, is refit on all the data and saved.

| | winner | features | grouped-cv accuracy | sensitivity | specificity | auc |
|---|---|---|---|---|---|---|
| anticancer | random forest | hand-crafted | 0.668 | 0.51 | 0.82 | 0.72 |
| hemolytic | logistic regression | esm-2 (150M) | 0.955 | 0.93 | 0.98 | 0.99 |

## esm-2 (what it is and how we use it)
esm-2 is a general protein language model trained by others on millions of protein sequences. we use it only as a feature extractor: it turns each peptide into 640 numbers (the average of its per-residue vectors, `src/pepguard/embed.py`). we then train our own logistic regression on those numbers using only the labeled data we were given. we do not use any anticancer or hemolytic predictor and nothing is passed through from an outside tool.

## the hemolytic shortcut and how we reduced it
about 90% of the hemolytic labels follow the rule "positively charged means hemolytic". a model can score well by learning only that. the cases that matter for the Selective verdict are the opposite ones: cationic but safe (only 27 in training) and not cationic but hemolytic (84).
- we counted those exception peptides 3 times in training (`src/pepguard/weights.py`).
- esm-2 features carry information beyond charge. on the exceptions, accuracy went from 70% to 94% (cationic and safe) and from 36% to 81% (not cationic but hemolytic).

## decision thresholds
each classifier's cutoff is tuned on out-of-fold probabilities (`src/pepguard/thresholds.py`), smoothed so a lucky spike cannot win. the classes are balanced, so the cutoffs ended up near 0.5 (0.51 and 0.51).

## what we tried that did not help (anticancer)
extra features (terminal composition, charge moment, reduced alphabets), feature selection, gradient boosting, extra trees, svm and logistic regression with five different scalers, kNN, esm-2 features, and ensembles. all landed between 0.60 and 0.67 under grouped cross validation.

## limits we are upfront about
- anticancer accuracy is about 0.67 and it misses about half of the true positives (sensitivity 0.51).
- the hemolytic model has only 27 examples of cationic-but-safe peptides, so its Selective calls rest on little data.
- estimates are from cross validation. the final test set could be more or less similar to the training peptides.
