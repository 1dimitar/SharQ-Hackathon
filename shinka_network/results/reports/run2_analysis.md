# Run 2 (network10, 200 generations, gpt-6-luna) - analysis of best program (gen 181)

Program: results/net10_luna_run2_200gen/best/main.py. Generated 2026-11-21.

## analysis/compare_network10.py
```
== 1. Coefficients
qoblib_tools linear: 60 nonzero in [-1.000, -0.193] | quadratic: 0 terms in [0.000, 0.000] | penalty weight 1.0
evolved      linear: 60 nonzero in [-1.000, -0.254] | quadratic: 480 terms in [-0.708, -0.087] | penalty weight 1.0
strongest evolved quadratic terms:
   x3,6 * x5,3: -0.708
   x5,10 * x10,6: -0.708
   x2,10 * x7,2: -0.686
   x3,10 * x7,3: -0.686
   x5,10 * x7,5: -0.680

== 2. Surrogate energy vs LP score (3000 random valid topologies)
qoblib_tools Spearman all 0.187 | best 10% 0.241 | optimum has lower energy than 100.0% of them
evolved      Spearman all 0.459 | best 10% 0.316 | optimum has lower energy than 100.0% of them

== 3. Annealing, 30 seeds x 300 reads -> repair -> LP
qoblib_tools best per seed: mean 232873 (+10.5%), min 226400, max 241400 | median-sample gap 38.4%
evolved      best per seed: mean 225750 (+7.1%), min 219125, max 233000 | median-sample gap 28.4%
Mann-Whitney, evolved best < qoblib_tools best: p = 1.3e-09

== 4. Sampler or surrogate? (1000 annealing reads)
qoblib_tools 347 of 1000 distinct valid samples have lower energy than the optimum (LP scores 230250-591000)
evolved      20 of 716 distinct valid samples have lower energy than the optimum (LP scores 244000-368500)

```

## analysis/compare_sizes.py (simulated annealing, 300 reads, seeds 1-3)
```
instance      BKV | method                      best per seed mean best gap median gap
network10  210800 | qoblib_tools       232875, 233667, 230667         10.2%      38.2%
network10  210800 | evolved            226286, 224084, 227000          7.1%      28.7%
network12  276474 | qoblib_tools       309750, 310000, 306250         11.6%      37.4%
network12  276474 | evolved            306556, 298800, 300182          9.2%      29.6%
network16  397908 | qoblib_tools       456600, 453572, 464084         15.1%      36.2%
network16  397908 | evolved            443280, 435625, 449286         11.3%      29.6%
network20  528528 | qoblib_tools       601286, 604286, 602125         14.0%      30.1%
network20  528528 | evolved            586234, 570566, 584500          9.8%      24.6%
network24  663688 | qoblib_tools       768778, 760734, 771286         15.6%      31.5%
network24  663688 | evolved            754534, 756556, 758858         14.0%      26.7%

```

## evaluate_qoro.py on network12, local Maestro simulator, 4 seeds per program
Programs: baseline = initial.py (qoblib_tools surrogate), net10_luna_run1 = run 1 best, net10_luna_run2_200gen = run 2 best.
```
qaoa_partitioned | baseline | combined_score: 0.7419365900378369 | best per seed: 358750 318500 344600 318286 
qaoa_partitioned | net10_luna_run1 | combined_score: 0.827593134848773 | best per seed: 323600 309112 304875 322000 
qaoa_partitioned | net10_luna_run2_200gen | combined_score: 0.837601487956912 | best per seed: 305200 305600 309715 307910 
pce | baseline | combined_score: 0.8044107585289918 | best per seed: 309000 309000 317000 310715 
pce | net10_luna_run1 | combined_score: 0.795695786191524 | best per seed: 317000 313500 330500 310715 
pce | net10_luna_run2_200gen | combined_score: 0.8002238656937578 | best per seed: 317000 309000 333000 310715 

```

## Qoro cloud smoke tests (network12, 1 seed)
- PCE, baseline: 501 circuits, 49 s, ~0.12 credits, combined score 0.7975 (best 315134).
- Partitioned QAOA, run 1 best: 6012 circuits, 110 s, ~0.51 credits, combined score 0.8233 (best 313000); the same program scored 0.8705 on the local simulator, so single runs are noisy.
