#!/usr/bin/env bash
# Build the image and run the post-processing evolution in Docker. Arguments go to run_evo.py:
#   ./run_docker.sh --generations 100 --islands 4
set -euo pipefail
cd "$(dirname "$0")"
IMAGE=shinka-postproc
if [[ ! -f .env ]]; then
    echo "Missing shinka_postproc/.env with OPENAI_API_KEY=..." >&2
    exit 1
fi
docker build -f Dockerfile -t "$IMAGE" ..
mkdir -p results
docker run --rm \
    --name "shinka-postproc-$(date +%Y%m%d-%H%M%S)" \
    --env-file .env \
    -v "$PWD/pools:/app/shinka_postproc/pools:ro" \
    -v "$PWD/results:/app/shinka_postproc/results" \
    "$IMAGE" "$@"
