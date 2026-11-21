### Hello SharQs

## Leaderboard

[leaderboard/LEADERBOARD.md](leaderboard/LEADERBOARD.md): the best network found so far for each
QOBLIB 08 instance, by any method, with a QOBLIB `.sol` file per record. It shows the best per
instance, the best that came out of a quantum sampler, and the best gap per method, under a chart
of our gap to QOBLIB's best known value per instance (`leaderboard/gap_to_best_known.png`).
`leaderboard.py` re-checks and re-scores every topology with the routing LP, so a claimed score is
never trusted.

```bash
uv run python leaderboard.py show                 # rebuild and print the leaderboard
uv run python leaderboard.py scan                 # import the result files our tools write
uv run python leaderboard.py submit my.sol --method lp_local_search --label "what produced it"
```

From any script or notebook:

```python
import leaderboard
leaderboard.submit("network12", arcs, method="qaoa_partitioned", label="my idea")  # arcs: [(i, j), ...]
```

`scan` knows the evolution runs (`shinka_network/results`), post-processing runs and QAOA pools
(`shinka_postproc`), the LP local search reports and `.sol` files. Add a line to `SOURCES` in
`leaderboard.py` when a new tool writes topologies elsewhere. `shinka_network/eval_sizes.py`
evaluates surrogate programs on network12-16 and submits their topologies as they are found.
