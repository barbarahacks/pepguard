# judge walkthrough (practice out loud)

say each answer in your own words, then check it against the answer below. the order is the order a judge would probably ask.

## 1. what does your project do?
two classifiers read a peptide's letters. one guesses whether it kills cancer cells, the other guesses whether it harms red blood cells. crossing the two answers gives a verdict, and **Selective** (kills cancer, spares blood) is the prize.

## 2. how do you turn letters into something a model can use?
two ways. first, 429 hand-made measurements: length, charge, how it avoids water, and its two-faced "moment". second, esm-2, a protein-reading model that gives 640 numbers per peptide. we only use esm-2 as a number-maker, and our own classifier learns from the labeled data we were given. *(`features.py`, `embed.py`)*

## 3. why do those numbers matter biologically?
cancer cells are more negative, so positive peptides (lots of K and R) get pulled toward them. water-avoiding parts let a peptide poke into the cell's oily skin. the two-faced shape punches the hole. the same traits also hit red blood cells, which is why selectivity is hard.

## 4. how do you know it isn't just memorising?
we test on peptides the model never saw. many peptides are near-twins, so we keep twins together in the same fold, otherwise accuracy looks inflated. normal cross validation said 0.76 and 0.95. the honest grouped numbers were 0.67 and 0.89 before our improvements. a shuffled-label check gave 0.50. *(`groups.py`)*

## 5. what are your results?
hemolytic about **0.95** accuracy, anticancer about **0.67**. say both numbers, and don't hide the second.

## 6. why is anticancer so much weaker?
hemolytic mostly follows an easy rule ("positive means harmful") and esm-2 captures the rest. anticancer has no such rule in our data. we tried boosting, a neural net, five scalers, extra features and ensembles, and all landed at the same ceiling, so it's the data and features and not the model.

## 7. is esm-2 allowed?
yes. the organizers said it's fine for feature extraction only. it turns a sequence into numbers, and nothing is routed through another predictor.

## 8. what's the weakest part of your work?
anticancer misses about half of the true positives (sensitivity 0.51). also, the hemolytic data has only 27 examples of positive-but-safe peptides, so its Selective calls rest on little data.

## 9. walk me through a line of code.
pick `tune_threshold` or `exception_weights`, both are short. the weights give 3x credit to the tricky cases (positive but safe, not positive but harmful), so the model can't just learn "positive = bad". `CODE_GUIDE.md` has demos you can run live.

## two things people usually get wrong
- **"scaling inside the pipeline"** means each fold only learns its ruler from its own training data, so nothing leaks from the test fold.
- **the threshold** is the "how sure before saying yes" line. ours landed at about 0.51 because the classes are balanced, so don't claim it did much.

## harder questions to be ready for
- *why not a deeper neural net?* we tried a small mlp on esm features and it matched the random forest (0.667 vs 0.671) without beating it. with about 1000 labeled peptides there isn't enough data for a big network to help.
- *why group at 60% similarity?* it's a judgement call. near-copies leak answers, and 0.6 kept enough distinct groups (428 and 393) for cross validation to work. we didn't tune it to look better.
- *what would you do with more time?* a sequence model (cnn or per-position esm features), more labeled cationic-but-safe peptides, and lab validation of the top Selective candidates.
- *what if the test peptides are very different from training?* then the numbers will be lower than our estimates. our grouped cross validation is the closest stand-in we have.
