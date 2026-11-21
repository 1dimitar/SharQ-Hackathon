# Hybrid Quantum-Classical Approach to Traffic Flow Optimization

Within this repository is a proposed hybrid quantum-classical approach to solving QOBLIB problem 8 involving network traffic optimization. All of the best known solutions were beaten by 3.42% at most. A combination of QAOA and classical optimizations are used to achieve this, together with a genetic algorithm-like approach via _ShinkaEvolve_ were the best solutions are found via an LLM.

Open [main.ipynb](main.ipynb) from the repository root. Choose a network from 11 through 24 and run all cells to generate a solution, verify its routed traffic, and display the weighted directed graph. The notebook also verifies the archived submissions for networks 10 through 24.

By default, the notebook replays the selected run with its archived QAOA candidates. Set `FRESH_QAOA = True` to sample a new pool using the local or cloud backend. A fresh pool or a time-limited search can change the result.
