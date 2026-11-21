# ShinkaEvolve run for the post-processing of QAOA output (QOBLIB 08). Separate from shinka_network/.
# Build from the repository root (run_docker.sh does this):
#   docker build -f shinka_postproc/Dockerfile -t shinka-postproc .
FROM ghcr.io/astral-sh/uv:python3.12-bookworm-slim

RUN apt-get update && apt-get install -y --no-install-recommends git \
    && rm -rf /var/lib/apt/lists/*

ENV UV_COMPILE_BYTECODE=1 \
    UV_LINK_MODE=copy \
    UV_PYTHON_DOWNLOADS=never \
    PYTHONUNBUFFERED=1 \
    PYTHONWARNINGS=ignore::SyntaxWarning

WORKDIR /app/shinka_postproc
COPY shinka_postproc/pyproject.toml shinka_postproc/uv.lock ./
RUN uv sync --frozen --no-install-project
ENV PATH="/app/shinka_postproc/.venv/bin:$PATH"

COPY qoblib_tools.py /app/
COPY qoblib_data /app/qoblib_data
COPY shinka_postproc/*.py ./

# Candidate pools and results are mounted at run time; the OpenAI key comes from --env-file.
ENTRYPOINT ["python", "run_evo.py"]
