# ShinkaEvolve run for the QOBLIB 08 network-design surrogate.
# Build from the repository root (run_docker.sh does this):
#   docker build -f shinka_network/Dockerfile -t shinka-network .
FROM ghcr.io/astral-sh/uv:python3.12-bookworm-slim

# git: ShinkaEvolve is installed from a pinned GitHub commit.
RUN apt-get update && apt-get install -y --no-install-recommends git \
    && rm -rf /var/lib/apt/lists/*

ENV UV_COMPILE_BYTECODE=1 \
    UV_LINK_MODE=copy \
    UV_PYTHON_DOWNLOADS=never \
    PYTHONUNBUFFERED=1 \
    PYTHONWARNINGS=ignore::SyntaxWarning

WORKDIR /app/shinka_network

# Dependencies first, so code changes do not reinstall them.
COPY shinka_network/pyproject.toml shinka_network/uv.lock ./
RUN uv sync --frozen --no-install-project
ENV PATH="/app/shinka_network/.venv/bin:$PATH"

COPY qoblib_tools.py /app/
COPY qoblib_data /app/qoblib_data
COPY shinka_network/*.py ./

# The OpenAI key is passed at run time (--env-file), never baked into the image.
ENTRYPOINT ["python", "run_evo.py"]
