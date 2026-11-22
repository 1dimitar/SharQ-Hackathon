#!/bin/sh
# Sequential GPU queue (one instance run at a time): network17 with 20-qubit QAOA blocks, then
# network18-24 with 20-qubit blocks if that beat the 12-qubit network17 result, else 12-qubit blocks.
cd "$(dirname "$0")/../.." || exit 1
P=../.venv/bin/python; F=results/net12_qaoa_cloud_run3/gen_128/main.py; L="run 3 recommended (gen 128)"
$P -u eval_sizes.py eval $F --label "$L" --sizes 17 --repeats 1 --parallel 1 --cluster qoro_gpu --cluster-size 20
CS=$($P - <<'PY'
import json
best12 = 544445.0  # network17 with 12-qubit blocks, gpu_queue.log 00:54
rows = [json.loads(l) for l in open("results/size_evals/evaluations.jsonl")]
runs = [r for e in rows if e.get("variant") == "cluster_size=20" for r in e["instances"] if r["n"] == 17]
print(20 if runs and min(r["best"] for r in runs) < best12 else 12)
PY
)
echo "[queue] network18-24 with cluster_size=$CS"
$P -u eval_sizes.py eval $F --label "$L" --sizes 18-24 --repeats 1 --parallel 1 --cluster qoro_gpu --cluster-size $CS
