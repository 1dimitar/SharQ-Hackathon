# QOBLIB 08 network design: hand-in

**Twelve instances, network13 to network24, beat QOBLIB's best known values, by 0.30% to 3.42%.** network10,
network11 and network12 match them. The largest instance, network24 (552 binary variables), is 0.63% below its
best known value. The largest quantum program is 20 qubits (Qoro cloud, `network12_quantum`).

All scores below are recomputed from the `.sol` files by the referee, `qoblib_tools.check_network_solution`.
Lower is better; ★ marks a score below QOBLIB's best known value.

| Folder | Instance | Binary variables | Largest quantum program (qubits) | Objective | Best known | Gap | Method |
|---|---|---:|---|---:|---:|---:|---|
| [network10](network10/RESULT.md) | network10 | 90 | none (classical) | 210800 | 210800 | +0.00% | Classical LP local search |
| [network11](network11/RESULT.md) | network11 | 110 | 16 (local Maestro) | 238334 | 238334 | +0.00% | QAOA (r3g128) + `init_run1_gen49.py` |
| [network12](network12/RESULT.md) | network12 | 132 | none (classical) | 276474 | 276474 | +0.00% | Classical LP local search |
| [network12_quantum](network12_quantum/RESULT.md) | network12 | 132 | 20 (Qoro cloud) | 277000 | 276474 | +0.19% | QAOA (r3g128) + `init_run1_gen49.py` |
| [network13](network13/RESULT.md) | network13 | 156 | 12 (local Maestro) | 299715 | 300620 | -0.30% ★ | QAOA (r3g128) + `init_run1_gen49.py` |
| [network14](network14/RESULT.md) | network14 | 182 | 12 (local Maestro) | 341375 | 343091 | -0.50% ★ | QAOA (r3g128) + `harvest_A_gen20.py` |
| [network15](network15/RESULT.md) | network15 | 210 | 12 (local Maestro) | 369720 | 370870 | -0.31% ★ | QAOA (r3g128) + `ckpt_A_gen20.py` |
| [network16](network16/RESULT.md) | network16 | 240 | 12 (local Maestro) | 395566 | 397908 | -0.59% ★ | QAOA (r2g181) + `harvest_A_gen20.py` |
| [network17](network17/RESULT.md) | network17 | 272 | 12 (local Maestro) | 431938 | 437000 | -1.16% ★ | QAOA (r2g181) + `harvest_A_gen33.py` |
| [network18](network18/RESULT.md) | network18 | 306 | 12 (local Maestro) | 466000 | 473375 | -1.56% ★ | QAOA (r3g128) + `ckpt_A_gen20.py` |
| [network19](network19/RESULT.md) | network19 | 342 | 12 (local Maestro) | 480000 | 492924 | -2.62% ★ | QAOA (r2g181) + `harvest_A_gen33.py` |
| [network20](network20/RESULT.md) | network20 | 380 | 12 (local Maestro) | 511769 | 528528 | -3.17% ★ | QAOA (r3g128) + `harvest_A_gen20.py` |
| [network21](network21/RESULT.md) | network21 | 420 | 16 (local Maestro) | 557053 | 569273 | -2.15% ★ | QAOA (r3g128) + `init_run1_gen49.py` |
| [network22](network22/RESULT.md) | network22 | 462 | 12 (local Maestro) | 597524 | 618135 | -3.33% ★ | QAOA (r3g128) + `harvest_A_gen20.py` |
| [network23](network23/RESULT.md) | network23 | 506 | 12 (local Maestro) | 636848 | 659369 | -3.42% ★ | QAOA (r2g181) + `harvest_A_gen20.py` |
| [network24](network24/RESULT.md) | network24 | 552 | 12 (local Maestro) | 659539 | 663688 | -0.63% ★ | QAOA (r3g128) + `harvest_A_gen20.py` |

Each folder holds the solution file `networkNN.sol` (QOBLIB's format) and `RESULT.md` with the exact settings and
the command that reproduces it. `network12/` is the best network12 result (classical); `network12_quantum/` is
the best network12 result from the quantum pipeline.

## Method

network11 and network13 to network24 come from one hybrid pipeline:

1. **QUBO.** One binary per possible link, n(n-1) in total. The QUBO has the degree rules (2 links out and 2 in
   per node) as penalties, plus a demand-based surrogate term. We evolved that surrogate with ShinkaEvolve (an
   LLM-driven program evolution): `r2g181` was scored by simulated annealing on network10, and `r3g128` by
   partitioned QAOA on Qoro's cloud on network12. Both are in `code/hamiltonians/`.
2. **Quantum sampling.** Divi's partitioned QAOA cuts the QUBO into blocks of at most 12, 16 or 20 variables
   (one circuit per block, so that many qubits) with p=2, a Monte Carlo angle optimiser (population 50,
   10 iterations) and 10000 shots. Beam search stitches the blocks into the 200 best full topologies, the
   "pool". The pools are in `code/pools/`.
3. **Classical post-processing.** A program evolved with ShinkaEvolve repairs the sampled topologies to valid
   ones and runs a local search that rewires links, scoring each candidate with the exact routing LP. Each run
   has a budget of 10000 to 25000 LP calls. The programs are in `code/postproc/`.
4. **Solution file.** `qoblib_tools.route_integral` finds integer flows that reach the LP score, and
   `qoblib_tools.write_network_solution` writes the `.sol` file.

network10 and network12 come from a classical LP local search (`shinka_network/analysis/lp_local_search.py`).

**What the quantum part contributes.** These results are hybrid: QAOA proposes starting topologies and the
classical LP search improves them. The quantum part is visible in one controlled comparison, on network12: at
2000 LP calls, post-processing reached a mean gap of +2.06% from QAOA pools against +2.70% from random starting
sets. We did not run that control on network13 to network24.

**Where the circuits ran.** The pools for network11 and network13 to network24 were sampled on the Maestro
simulator on a laptop. The `network12_quantum` pool (20-qubit circuits) ran on Qoro's cloud (`qoro_gpu`).

## How we checked the files

- `qoblib_tools.py` in our repository is byte-identical to the one in Qoro's hackathon repository
  (github.com/QoroQuantum/pushquantum-hackathon-2026), and our demand data is identical to QOBLIB's `demand.txt`.
- QOBLIB's own published solutions for network05 to network24, scored with the same referee, reproduce every
  published best known value exactly, so our scores and the best known values are on the same scale.
- Every `.sol` file here returns `valid` from the referee, with the objective in the table. On top of the
  referee's checks: no negative or fractional flows, exactly 2n links, no self-loops, and each score equals the
  routing-LP optimum of its topology.
- We have not yet run QOBLIB's official Rust checker on these files.

## Reproduce

Clone the repository and run the command in a folder's `RESULT.md` from `shinka_postproc/`. It re-runs the
post-processing on the saved QAOA pool. To re-sample a pool, use `shinka_postproc/make_pools.py` with the
settings in `RESULT.md`. The post-processing is deterministic for a given seed and LP budget as long as it
finishes inside its program's time guard: 3300 s for the `harvest_*` programs (our longest run took 1709 s),
140 s for `ckpt_A_gen20.py`. The network18 run hit that 140 s guard after 5593 LP calls, so network18 (and
possibly network15) depend on machine speed.
