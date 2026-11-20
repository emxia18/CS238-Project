# Hyper-parameter sweeps

Belief-agent sweeps on the **first 100 validation articles**, used for tuning.
Configs were reconstructed from the original filenames (`base.csv`,
`end0.5.csv`, …); everything not listed is the `BeliefAgent` default
(`alpha=20, temperature=0.5, top_k=300, title_bonus=2, endgame_conf=None`).

| file | config | solved |
|---|---|---|
| `belief_val100_base.csv` | defaults | 69% |
| `belief_val100_endgame0.5.csv` | `endgame_conf=0.5` (`endgame_bonus=10`) | 75% |
| `belief_val100_endgame0.5_bonus30.csv` | `endgame_conf=0.5, endgame_bonus=30` | 76% |
| `belief_val100_endgame0.8.csv` | `endgame_conf=0.8` | 75% |
| `belief_val100_ownpen0.2.csv` | `own_title_penalty=0.2` | 69% |

n = 100, so the 95% interval on each rate is roughly ±9 points.

New sweeps record their config in the filename automatically:

    python -m scripts.evaluate --agent belief --n 100 \
        --set endgame_conf=0.5 --tag endgame0.5 --out-dir results/sweeps
