#!/usr/bin/env bash
# Build the image and run an evolution in Docker. Arguments go to run_evo.py, e.g.
#   ./run_docker.sh --suite net10 --generations 20 --islands 4
# Results land in shinka_network/results/ on the host.
set -euo pipefail
cd "$(dirname "$0")"

IMAGE=shinka-network
if [[ ! -f .env ]]; then
    echo "Missing shinka_network/.env with OPENAI_API_KEY=..." >&2
    exit 1
fi

docker build -f Dockerfile -t "$IMAGE" ..
mkdir -p results cache
docker run --rm \
    --name "shinka-network-$(date +%Y%m%d-%H%M%S)" \
    --env-file .env \
    -v "$PWD/results:/app/shinka_network/results" \
    -v "$PWD/cache:/app/shinka_network/cache" \
    "$IMAGE" "$@"
