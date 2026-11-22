# Pitch script: 5 minutes, 7 slides (12 PDF pages with the click-steps)

Roughly 650 words at a calm pace. Each **[click]** is one page forward. The times are targets,
not hard limits.

Before the talk, refresh the numbers (harvest jobs may have added records):
`cd presentation && python3 make_numbers.py && latexmk -pdf pitch.tex`

---

## 1. Title (0:00 to 0:15)

"Hi, we're Team SharQ. Our pitch in one sentence: in hard network design, where you *start* your
search decides where you end up, and a quantum computer can tell you where to start."

## 2. The problem (0:15 to 1:30), 4 clicks

"Here are five German cities and how many people want to travel between each pair: Hamburg to
Berlin 42, Frankfurt to Cologne 44, and so on. We decide which one-way roads to build, then route
everyone. The score is the traffic on the **busiest road**. Lower is better."

**[click]** "The obvious design: connect neighbours in both directions. The busiest roads, in red,
carry 83."

**[click]** "A smarter mix brings it down to 74."

**[click]** "And this is the best of all 216 possible networks: 65.5. Look at the red roads: the
best design spreads the load, so five roads share the maximum."

**[click]** "All three follow one rule: every city has exactly **two roads out and two roads in**.
This is QOBLIB problem 08, network design. The five-city version is the real benchmark instance
network05. We only added the city names."

## 3. Search space (1:30 to 2:00)

"Five cities: 216 networks, easy. But the count explodes. With 20 cities there are 5.7 times 10 to
the 34 networks. Scoring one takes 10 milliseconds, so checking them all would take 10 to the 25
years. Optimal answers are known only up to 10 cities, and the benchmark goes up to 24."

## 4. Where you start (2:00 to 3:00), 1 click

"So instead we search: pick a starting network and walk downhill. Picture all networks as a
landscape, where height is the busiest-road load. Start at random and you roll into a shallow
valley and get stuck."

**[click]** "Our idea: let a quantum computer choose the starting points. We encode the problem as
a QUBO: one qubit per possible road, low energy for roads that carry heavy demand, and penalties
for breaking the two-in, two-out rule. We sample it with partitioned QAOA on Qoro's cloud, in
blocks of 12 to 20 qubits. The low-energy samples are networks that already sit in good valleys.
We measured this: the best QAOA start is 9 to 17 percent above the best known value, against 18
to 24 for random starts. After the same 500 LP calls of search, the QAOA starts are at 3.2 percent
and the random ones at 5.7."

## 5. ShinkaEvolve (3:00 to 4:00)

"That leaves two questions: which Hamiltonian should the circuit sample, and how exactly should we
walk downhill? Instead of tuning by hand, we let an LLM evolve the code with ShinkaEvolve. It
keeps an archive of programs. The LLM proposes a code change, we run it on QOBLIB instances
(QAOA, then the search, then the LP) and score it by the gap to the best known value.

We evolved two programs. The **Hamiltonian** raised QAOA sample quality from 0.74 to 0.84. The
**search** halved the gap on QAOA pools it had never seen. We re-tested every winner on held-out
data, and each evolution run cost less than four dollars in LLM calls."

## 6. Results (4:00 to 4:45)

"On 10 to 12 cities we match QOBLIB's best known values. From 13 up to 24 cities, on every single
instance, we **beat** QOBLIB's best known value, by up to 3.4 percent. On the largest instance,
24 cities and 552 binary variables, we go from 663,688 to 659,539. All twelve records start from
QAOA samples, and the official routing LP recomputes every number from our solution files."

## 7. Close (4:45 to 5:00)

"Quantum picks the valley; evolution learns the way down. Thank you!"

---

## Likely questions

- **Do you really need the quantum part?** The head start is clear at small search budgets (500
  LP calls: 3.2% vs 5.7%) and narrows at large ones (2000 calls: 2.1% vs 2.7%, ranges overlap).
  Measured on network12 with 2 held-out QAOA pools and 5 random start sets, so the samples are
  small. The QAOA starts are also much less diverse than random ones (about 60% shared links vs
  18%).
- **Is the city data real?** The numbers are QOBLIB network05 exactly; the city names are only
  for illustration.
- **Overfitting?** The search was evolved on network13-16 pools. network17-24 and later seeds
  were never seen in training, and the records there come from those unseen instances.
- **How long does a record take?** 10,000 to 25,000 LP calls per QAOA pool, at about 10 to 20 ms
  per LP for 20 to 24 cities, plus the QAOA sampling.
- **Why not bigger circuits?** Partitioning keeps every block at 20 qubits or fewer. At this size
  the cloud time is dominated by job upload and queueing, not simulation, and the GPU cluster was
  not faster.
- **What is the score unit?** QOBLIB's objective: the load on the busiest link times 1000. The
  5-city slide shows it divided by 1000 (65.5 = 65,500).
