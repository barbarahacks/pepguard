# understand pepguard like you're 5

read this top to bottom once. every part has the kid version first, then the real words you can say to a judge.

---

## the whole story

**kid version:** a peptide is a word spelled with 20 possible letters, like `KWKLFKKIGAVLKVL`. some words are like tiny keys that open a hole in a bad cell (cancer). but some keys also open holes in good cells (red blood cells), and that's bad. we taught a computer to look at a word and guess two things: "does it hurt cancer?" and "does it hurt good blood cells?". the best words say yes to the first and no to the second. we call those **Selective**.

**real words:** two binary classifiers on peptide sequences, trained on lab-tested data, whose two predictions are crossed into four verdicts.

```
 the word (peptide)
        |
        v
  [data.py]      clean it: uppercase, only the 20 real letters
        |
        v
  [features.py]  turn the word into numbers   ----or----   [embed.py] a protein "reading" model turns it into numbers
        |                                                         |
        +---------------------------+-----------------------------+
                                    v
                      [train.py]  practice, compare, pick the best, save it
                                    v
                      models/*.joblib   (the saved "brains")
                                    v
                      [predict.py]  new words in -> csv with 4 verdicts out
```

---

## file by file

### `data.py`: the clean-up helper
- **kid version:** before the computer looks at a word, we wipe off crumbs: make letters big, throw away anything that isn't one of the 20 real letters.
- **also:** it refuses to read anything in a folder called `demo`. the demo words are fake and random, so teaching the computer with them would be like studying from a made-up textbook.
- **point at:** `clean_sequence`, `load_training_set`.

### `features.py`: turning words into numbers
- **kid version:** a computer can't read letters, only numbers. so we measure each word like a doctor measures a kid: how long is it, how "positive" is it, how much does it hate water, and so on.
- **real words:** 429 numbers per peptide:
  - `length`
  - **charge** (K and R are +1, D and E are -1). cancer cells are slightly negative, and opposites attract, so positive peptides get pulled toward them.
  - **hydrophobicity** (how much it avoids water), which helps it poke into a cell's oily skin.
  - **hydrophobic moment**: if you coil the peptide like a spring, does one side hate water and the other love it? that two-faced shape is what lets it punch a hole.
  - **composition** (20 numbers): what fraction of the word is each letter.
  - **dipeptides** (400 numbers): how often each pair of neighbouring letters appears.
- **point at:** `extract_features` (builds one word's numbers) and `build_feature_matrix` (does it for many words). every word gets the same 429 columns in the same order.

### `embed.py` and `esm_features.py`: the friend who read a million books
- **kid version:** ESM-2 is a friend who has read millions of protein "words". we show it our word and it whispers 640 numbers describing what it thinks the word is like. we only borrow the whispers. our own computer brain still learns from our own labeled examples.
- **rule check:** ESM-2 is used **only to make numbers** (feature extraction). nothing is copied from another anticancer or hemolytic predictor.
- **why a separate process:** torch (the engine under ESM) and scikit-learn crash when they share one process on a mac, so `esm_features.py` runs the embedding in a helper process and saves the answer in `data/cache/` so it only has to do it once.
- **point at:** `embed_sequences` (the averaging), `esm_matrix` (the caching).

### `feature_sets.py`: choosing which numbers to use
- **kid version:** a menu. `hand` = our 429 measurements, `esm150` = the 640 whispers, `hand+esm150` = both. a model says which menu item it was trained on, and `predict.py` orders the same one later.
- **point at:** `build_features`.

### `groups.py`: no copying from your twin
- **kid version:** imagine a test where your answer sheet is almost the same as your twin's. if you study twin's sheet and then get tested on yours, you'll look smart for no reason. many peptides in our data are near-twins. so we put twins in the same "family" and make sure a family is never split between studying and testing.
- **real words:** peptides at least 60% similar are linked, and linked peptides form a group. cross validation never splits a group.
- **why it matters:** normal cross validation said anticancer 0.76 and hemolytic 0.95. with groups the honest numbers were 0.67 and 0.89 (before improvements). we choose to report the honest ones.
- **point at:** `similarity_groups`.

### `train.py`: practice, compare, pick the winner
- **kid version:** we hold a tournament. several different "students" (models) take practice tests. the one with the best score gets to be our brain.
- **real words:**
  - candidates: logistic regression, random forest, an rbf svm, and logistic regression on ESM numbers.
  - **cross validation** = practice exams: cut the data into 5 parts, study on 4, test on the 5th, rotate. every peptide gets tested by a model that never saw it.
  - the winner is **refit on all the data** and saved with its settings in one file (a "bundle": the model, which numbers it needs, and its cutoff).
- **scaling:** many models need all numbers on the same ruler (a length of 40 would drown out a fraction of 0.05). `StandardScaler` does that, and it sits **inside** the pipeline so each practice test only learns the ruler from its own studying data. no peeking.
- **point at:** `candidate_models`, `cross_val_probabilities`, `pick_best`, `train_task`.

### `weights.py`: extra credit for the tricky questions
- **kid version:** 9 out of 10 hemolytic peptides are just "positive = bad", so a lazy student learns only that. the tricky ones are the exceptions: positive but safe, or not positive but harmful. we tell the computer "get these wrong and you lose 3 points instead of 1".
- **point at:** `exception_weights`.

### `thresholds.py`: how sure before saying yes
- **kid version:** the model gives a number from 0 to 1, like "70% sure". a **threshold** is the line where we say "yes". 0.5 is the usual line. we tested other lines using the practice-test answers and picked the best one. the curve is smoothed so one lucky spike doesn't win.
- **honest note:** our classes are balanced, so the best line landed at about 0.5. this knob didn't give us much.
- **point at:** `tune_threshold`.

### `metrics.py`: the report card
- **kid version:** one score isn't enough. **sensitivity** = of the truly bad-for-cancer words, how many did we catch? **specificity** = of the harmless ones, how many did we correctly leave alone? a student who always answers "yes" gets sensitivity 100% and specificity 0%, which is a cheat, so we watch both. accuracy, AUC and MCC are in there too (the judges rank by accuracy first).
- **point at:** `compute_metrics`.

### `predict.py`: the exam day script
- **kid version:** new words arrive. we load the saved brains, measure each word the same way, ask both brains, and write a csv with the probability, a yes/no, and a verdict. then a safety inspector (`validate_submission`) checks the csv matches the official template before it lets us save it.
- **verdict table:** anticancer yes + hemolytic no = **Selective**; yes + yes = **Works but toxic**; no + yes = **Toxic and doesn't work**; no + no = **Does nothing**.
- **point at:** `score`, `verdict`, `build_submission`, `validate_submission`.

---

## try it live (all of these really run)

run from the project folder with the venv on.

```bash
# 1. a peptide turns into 429 numbers
python -c "from pepguard.features import extract_features as f; d=f('KWKLFKKIGAVLKVL'); print(len(d), d['length'], d['net_charge'], round(d['hydrophobic_moment'],2))"
# -> 429 15 5 0.41      (429 numbers, length 15, net charge +5, moment 0.41)

# 2. near-twins share a family
python -c "from pepguard.groups import similarity_groups as g; print(g(['KWKLFKKIGAVLKVL','KWKLFKKIGAVLKVA','DDDDEEEEDDDDEEE']).tolist())"
# -> [0, 0, 1]          (the first two are family 0, the third is alone)

# 3. exception weights: only the two tricky cases get x3
python -c "import pandas as pd; from pepguard.weights import exception_weights as w; print(w(pd.DataFrame({'net_charge':[5,5,0,0]}),[1,0,0,1]).tolist())"
# -> [1.0, 3.0, 1.0, 3.0]

# 4. score two peptides end to end
python -c "
import pandas as pd
from pepguard.predict import build_submission
print(build_submission(pd.DataFrame({'sequence':['KWKLFKKIGAVLKVL','DDDDEEEEDDDDEEE']})).round(3).to_string(index=False))"
```

what example 4 prints with our saved models:

| sequence | anticancer prob | hemolytic prob | verdict |
|---|---|---|---|
| KWKLFKKIGAVLKVL | 0.542 | 0.980 | Works but toxic |
| DDDDEEEEDDDDEEE | 0.565 | 0.000 | Selective |

read it like this: the hemolytic model is very sure (0.98 and 0.00), which matches the biology (a positive peptide hurts blood cells, a negative one doesn't). the **anticancer model is only about 67% accurate**, so probabilities near 0.5 mean "not sure". the second row shows that weakness: an all-negative peptide is very unlikely to kill cancer cells in reality, yet it scores 0.565. say this openly if you're asked. it is the honest limit of the anticancer data and features.

---

## if a judge asks...

| question | kid answer | where |
|---|---|---|
| why random forest for anticancer? | it was the best student in the tournament, and it doesn't care that our numbers are on different rulers. five other approaches landed at or below it. | `train.py` |
| is it overfitting? | we test on words the model never saw, and we keep near-twins together so it can't cheat. a shuffled-label test gave 0.50, so nothing is leaking. | `groups.py`, `train.py` |
| why is hemolytic so much better than anticancer? | hemolytic mostly follows "positive = harmful", which is easy to learn, and the ESM numbers capture the rest. anticancer has no such simple rule in our data. | `EXPLAIN.md` |
| what is ESM-2 and is it allowed? | a general protein-reading model. we only use it to make numbers from a sequence. our own classifier is trained on the given data. | `embed.py` |
| what would you do with more time? | try boosting and a small neural net, and ESM features per position instead of averaged. | `EXPLAIN.md` |
| what is the weakest part? | anticancer: about 0.67 accuracy and it misses about half of the real positives. | `EXPLAIN.md` |

## the 60-second pitch

"a peptide is just a string of letters. we turned each one into numbers, such as its charge and how it splits into water-loving and water-hating sides, and also asked a protein language model to describe it. then we trained two classifiers on the lab data, one for killing cancer and one for harming blood cells, and crossed them to find the **selective** ones. we were careful about honesty: similar peptides are never split between training and testing, so our numbers aren't inflated. hemolytic reaches about 95%. anticancer is about 67%, and we know exactly why it's harder."
