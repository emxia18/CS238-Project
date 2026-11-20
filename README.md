# Redactle Solver

Agents that play [Redactle](https://redactle.net) — guess words to un-redact a
Wikipedia article until its title is revealed — framed as sequential decision
making under uncertainty (CS238).

## Layout

```
redactle/               library
  paths.py              all data/result paths (anchored at the repo root)
  redact.py             tokenization, word matching, redaction rules
  env.py                RedactleEnv, Puzzle, Observation
  agents/
    base.py             Agent interface, RandomAgent, FrequencyAgent
    data.py             training-split vocabulary and count matrix (cached)
    belief.py           BeliefAgent: posterior over training articles
    lm.py               LMBeliefAgent: belief + DistilRoBERTa
scripts/                command-line entry points (run with python -m)
  vital.py              fetch the Vital Articles list -> data/vital_articles.json
  data.py               build data/articles.jsonl with train/val/test splits
  evaluate.py           evaluate an agent -> results/<agent>_<split>.csv
  figures.py            figures for the write-up -> figures/
  play.py               play in the terminal
tests/                  pytest suite
data/                   generated data (git-ignored except vital_articles.json)
results/                per-episode CSVs; results/sweeps/ holds tuning runs
figures/                generated figures
```

## Usage

Run everything from the repo root.

```bash
pip install -r requirements.txt

python -m scripts.data                        # one-time: build the dataset
python -m scripts.evaluate --agent belief     # random | frequency | belief | lm
python -m scripts.evaluate --agent belief --n 100 --set endgame_conf=0.5 --tag end0.5
python -m scripts.figures --agent lm          # figures from results/*.csv
python -m scripts.play --max-level 3          # play yourself
pytest                                        # tests
```

The first agent run builds the training vocabulary and count matrix under
`data/` (a few minutes); later runs load the cache.
