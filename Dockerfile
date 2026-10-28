# Deposium GeoAI MCP Server
# Python 3.14: required by the pinned so4gp 1.1.2 CPU `patterns` extra.
# The compatibility probe verifies the public API + its scientific stack; the
# container build verifies the full GeoAI geospatial stack on the same runtime.
#
# Two stages keep build tools outside the runtime image:
# compilers, -dev libraries, uv and its caches stay in the builder; the runtime
# image receives the locked virtualenv, the cached DuckDB extensions and the app.
# Tools, public resources, scripts and zones are included in the runtime image.
#
# Both stages use the same base, pinned by digest so the measured base cannot move:
# python:3.14-slim index digest, read on 2026-10-15 (Python 3.14.8, Debian trixie;
# its linux/arm64 v8 manifest is
# sha256:7f47c8ffd4e70e88f7326b68e7df95109597ef66023a1052b7d1ec18390d2604).

# --------------------------------------------------------------------------- #
# Builder
# --------------------------------------------------------------------------- #
FROM python:3.14-slim@sha256:a2b82f3c48559aa0a8446d9af49826b6e2b2016f4cd2afabfe6013ec53729170 AS builder

# Build-time only: compilers and GDAL/GEOS/PROJ headers, should a locked
# dependency ever need a source build.
RUN apt-get update && apt-get install -y --no-install-recommends \
    build-essential \
    gdal-bin \
    libgdal-dev \
    libgeos-dev \
    libproj-dev \
    && rm -rf /var/lib/apt/lists/*

ENV CPLUS_INCLUDE_PATH=/usr/include/gdal \
    C_INCLUDE_PATH=/usr/include/gdal \
    GDAL_CONFIG=/usr/bin/gdal-config

# uv is a build tool: it is never copied into the runtime stage.
RUN pip install --no-cache-dir uv && uv --version

WORKDIR /app

# Copy dependency files and public package metadata required by hatchling.
# uv.lock pins the installation below to the exact locked versions.
COPY pyproject.toml uv.lock README.md LICENSE NOTICE CITATION.cff ./
COPY assets/ assets/
COPY reference/ reference/

# RW-2 frozen transport contract: the wheel force-includes these schemas as
# contracts/rw2 (see pyproject), so they must be in the build context before
# the package install below.
COPY tests/fixtures/rw2 tests/fixtures/rw2
# The archive panel also force-includes the canonical zone catalogue.
COPY zones/zones.geojson zones/zones.geojson
# Hatch packages `src`, so it must exist before installation. Otherwise the
# console-script entrypoints are created without an importable package.
COPY src/ src/

# One virtualenv, so exactly one directory moves to the runtime stage. Same base
# image in both stages, so its interpreter symlinks stay valid there.
RUN python -m venv /opt/venv
ENV PATH="/opt/venv/bin:$PATH"

# Install dependencies including the patterns extra (so4gp gradual mining).
# This is what makes the in-process discover_patterns/pattern_alert tools
# possible; it also pulls the scientific stack and CPU Torch through
# so4gp 1.1.2. The AI extra is not part of this image.
#
# Pinned to uv.lock: export the locked runtime deps (base + patterns, no dev) to a fully
# hash-pinned requirements file, install exactly those, then install the project itself with
# --no-deps so the lock — not a fresh resolve — governs every version in the deployed image.
# The pip interface does not use tool.uv.sources: explicitly choose the CPU
# Torch backend while preserving the exported versions and artifact hashes.
RUN uv export --frozen --no-dev --extra patterns --no-emit-project -o /tmp/requirements.txt \
    && uv pip install --python /opt/venv/bin/python --torch-backend cpu -r /tmp/requirements.txt \
    && uv pip install --python /opt/venv/bin/python --no-deps .

# Mesh requests run in killable subprocesses and may not download extensions
# inside their request deadline. Cache both extensions under the HOME the
# runtime stage uses, so DuckDB finds them there without a download.
ENV HOME=/opt/duckdb-home
RUN mkdir -p /opt/duckdb-home \
    && python -c "import duckdb; c=duckdb.connect(':memory:'); c.execute('INSTALL spatial'); c.execute('INSTALL httpfs'); c.close()"

# --------------------------------------------------------------------------- #
# Runtime (default target)
# --------------------------------------------------------------------------- #
FROM python:3.14-slim@sha256:a2b82f3c48559aa0a8446d9af49826b6e2b2016f4cd2afabfe6013ec53729170 AS runtime

# Runtime packages only: curl for the healthcheck and the GDAL CLI (`gdal-bin`,
# which brings the GDAL shared libraries). No compilers, no -dev packages.
RUN apt-get update && apt-get install -y --no-install-recommends \
    curl \
    gdal-bin \
    && rm -rf /var/lib/apt/lists/*

# GDAL network settings. HOME points at the cached DuckDB extensions, for the
# server and for the mesh subprocesses that inherit its environment.
ENV GDAL_HTTP_MERGE_CONSECUTIVE_RANGES=YES \
    GDAL_DISABLE_READDIR_ON_OPEN=EMPTY_DIR \
    GDAL_HTTP_MAX_RETRY=3 \
    GDAL_HTTP_TIMEOUT=30 \
    PATH="/opt/venv/bin:$PATH" \
    HOME=/opt/duckdb-home

COPY --from=builder /opt/venv /opt/venv
COPY --from=builder /opt/duckdb-home /opt/duckdb-home

WORKDIR /app

# `python -m src.server` runs from /app: the public README is read from here,
# and src/ includes the pinned runtime data and machine-readable schemas.
COPY README.md ./
COPY LICENSE NOTICE CITATION.cff ./
COPY assets/ assets/
COPY reference/ reference/
COPY src/ src/

# Copy runtime scripts and scientific zones.
COPY scripts/ scripts/
COPY zones/ zones/

# Create data directories
RUN mkdir -p /data/images /data/models

# Expose port
EXPOSE 5001

# Allow scientific dependencies time to initialize before the health check
HEALTHCHECK --interval=30s --timeout=10s --start-period=30s --retries=3 \
    CMD curl -f http://localhost:5001/health || exit 1

# Run server
CMD ["python", "-m", "src.server"]
