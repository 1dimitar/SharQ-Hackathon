# QOBLIB network design

Open [main.ipynb](main.ipynb) from the repository root. Choose a network from 11 through 24 and run all cells to generate a solution, verify its routed traffic, and display the weighted directed graph. The notebook also verifies the archived submissions for networks 10 through 24.

By default, the notebook replays the selected run with its archived QAOA candidates. Set `FRESH_QAOA = True` to sample a new pool using the local or cloud backend. A fresh pool or a time-limited search can change the result.

- [Run configurations](experiments/workflow.py) pair each instance with its candidate pool, QUBO builder, classical search program, and routing budget.
- [shinka_network](experiments/shinka_network/) contains the evolved QUBO builders and their experiment history.
- [shinka_postproc](experiments/shinka_postproc/) contains the classical search generations, routing oracle, sampling code, and historical run reports.
- [data](data/) contains the notebook's demand matrix, candidate pools, leaderboard records, and archived solutions.
