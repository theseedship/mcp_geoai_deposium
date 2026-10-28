# DEPOSIUM GeoAI

**Satellite observations, weather and spatial data — accessible through AI tools.**

GeoAI turns geospatial data into structured, traceable results that an AI client
can retrieve, analyse and explain. Ask for a weather forecast at a point, inspect
vegetation over an area, or look for relationships in environmental time series.
The server retrieves data and runs the calculations. Tools expose explicit
inputs and structured results, with source and limit metadata where available.

It is a standalone Python server implementing the
[Model Context Protocol (MCP)](https://modelcontextprotocol.io/): a common interface
that lets compatible AI applications call tools. Use it with your own MCP client,
over stdio, streamable HTTP or legacy SSE.

The maintained source and issue tracker are hosted in
[this repository](https://github.com/theseedship/mcp_geoai_deposium).

## Funded by FIERCE

<p>
  <a href="https://project-fierce.eu/">
    <img src="assets/fierce-logo.png" alt="FIERCE — Making Space for Green Solutions" width="140">
  </a>
  &nbsp;&nbsp;
  <a href="https://project-fierce.eu/about/">
    <img src="assets/eu-funded.jpg" alt="Funded by the European Union" width="287">
  </a>
</p>

**GeoAI is funded by the [FIERCE project](https://project-fierce.eu/)** through
its Financial Support Programme. This support brings together research,
geospatial engineering and market exploration to help DEPOSIUM work with Earth
Observation and environmental data.

FIERCE receives European Union support through Horizon Europe, Grant Agreement
**101177496**. See the [official project description](https://project-fierce.eu/about/).
The authors are responsible for this repository's content; it does not represent
an endorsement by the European Union or HADEA.

## Team and contributions

The project combines complementary research, engineering and market expertise.

| Contributor | Role in the project |
| --- | --- |
| **[Nicolas Geysse](https://github.com/nicolas-geysse)** — [DEPOSIUM / TSS](https://deposium.ai/) | Project lead and coordination; architecture and implementation of the MCP layer, product direction, integration into DEPOSIUM and qualification of user workflows. |
| **[Jonathan Labéjof, PhD](https://github.com/b3j0f)** — [BEJOF](https://www.bejof.org/) | Geospatial data pipelines, reproducible processing, forecast integration and evaluation infrastructure. |
| **[Dickson Owuor, PhD](https://github.com/owuordickson)** — [so4gp](https://pypi.org/project/so4gp/) | Scientific methods for gradual-pattern mining; library development, experimental comparisons and research validation. |
| **[Amandine Raynaud](https://www.linkedin.com/in/amandine-raynaud/)** — [CLELYA](https://clelya.fr/vision/) | Market study and innovation strategy: identify relevant sectors, practical problems and conditions for adoption. |

The scientific foundations include Dickson Owuor and Anne Laurent's work on
gradual-pattern mining and related collaborations. Method and software credits
are recorded in [CITATION.cff](CITATION.cff).

## Start here

| I want to… | Read |
| --- | --- |
| Understand the project and its research question | [Why GeoAI?](#why-geoai) |
| Meet the project team | [Team and contributions](#team-and-contributions) |
| See what I can do with it | [Capabilities](#capabilities) |
| Choose a tool or try a first request | [MCP tools: 16 tools, inputs and examples](reference/mcp-tools.md) |
| Connect my MCP client | [Run locally](#run-locally) |
| Run the server in a container | [Optional Docker deployment](#optional-docker-deployment) |
| Configure providers and storage | [Configuration](#configuration) |
| Understand the scientific scope | [Research and evidence](#research-and-evidence) |
| Develop, test or contribute | [Development](#development) |
| Cite the work or check reuse conditions | [Copy a citation or BibTeX](#citation), [licence](#licence) and [data attribution](NOTICE) |

## Why GeoAI?

[DEPOSIUM®](https://deposium.ai/), developed by
[The Seed Ship (TSS)](https://the-seed-ship.com/), helps people work with documents
and knowledge. GeoAI extends that direction to data with a location and a time:
satellite acquisitions, weather series, geographic features and their sources.

The guiding research question is:

> **Can we extract more knowledge with less computation?**

The work connects data preparation, temporal analysis, gradual-pattern mining
and AI-assisted exploration. It asks how to make complex environmental data
usable while keeping computation explicit and reproducible. Computational and
environmental efficiency are objectives to evaluate on defined workloads.

Within DEPOSIUM, these tools provide a foundation for connecting environmental
observations to documents and conversational workflows. This repository delivers
the independent GeoAI server; the consuming application supplies its interface,
document retrieval, scheduling and notification delivery.

## Capabilities

| Work with… | What GeoAI provides | Tools |
| --- | --- | --- |
| **Satellite observations** | Find scenes, obtain band URLs, compute NDVI profiles and vegetation indicators, compare two observation windows. | [Satellite and vegetation tools](reference/mcp-tools.md#satellite-imagery-and-vegetation) |
| **Weather and climate** | Daily forecasts for a point or zone, historical daily weather, monthly climate distributions and alignment with NDVI acquisitions. | [Weather and climate tools](reference/mcp-tools.md#weather-and-climate) |
| **Geographic layers** | Select GeoParquet features by intersection, containment or distance. | [Spatial queries](reference/mcp-tools.md#spatial-queries) |
| **Environmental time series** | Discover attributes that vary together, test their recurrence across zone folds and calculate experimental explanatory indices. | [Pattern tools](reference/mcp-tools.md#patterns-and-experimental-alerts) |
| **Solar profiles** | Validate or collect a bounded portfolio of hourly radiation data, with explicit provider-access configuration. | [Solar acquisition pilot](reference/mcp-tools.md#solar-acquisition-pilot) |

The server advertises **16 tools** with structured input/output contracts. An MCP
client can read `geoai://capabilities/tools` for the authoritative inventory and
`geoai://guide/agent` for workflow guidance. See the
[plain-language tool reference and first calls](reference/mcp-tools.md#first-calls).

## Research and evidence

The engineering layer makes data and calculations accessible through MCP. The
research layer investigates which interpretable relationships can be extracted
and whether they help on a defined task. Experimental comparisons, including
negative results, are kept reproducible through code, fixtures and source hashes.
The optional pattern tools use native **so4gp GRAANK and TGrad**.

A pattern describes an association to assess; it does not establish a cause or a
calibrated prediction. Explanatory indices are not event probabilities. NDVI
measures vegetation greenness, with separate validation needed for claims about
carbon or biodiversity. Weather results retain their actual provider, collection
time and missing-data limits; model issuance remains unknown when unavailable.

These foundations can support further work in agriculture, environmental
monitoring and renewable energy. New indicators and missing-data estimation need
a suitable dataset, an explicit method and their own validation protocol.

## Run locally

Python 3.14 and [uv](https://docs.astral.sh/uv/) are required by the locked CPU
pattern runtime. From this directory:

```sh
uv sync --locked --extra patterns
MCP_TRANSPORT=stdio uv run --no-sync geoai-server
```

An MCP client can spawn the same process. Replace the directory below with the
absolute path to this checkout:

```json
{
  "mcpServers": {
    "geoai": {
      "command": "uv",
      "args": ["--directory", "/absolute/path/to/geoai", "run", "--no-sync", "geoai-server"],
      "env": {"MCP_TRANSPORT": "stdio"}
    }
  }
}
```

For HTTP, choose a bearer secret of at least 32 characters and pass it to the
server and client:

```sh
export GEOAI_MESH_AUTH_TOKEN="$(python3 -c 'import secrets; print(secrets.token_urlsafe(32))')"
MCP_TRANSPORT=http HOST=127.0.0.1 PORT=5001 uv run --no-sync geoai-server
```

The MCP endpoint is `http://127.0.0.1:5001/mcp`; `/health` returns `OK`.
`GEOAI_MESH_AUTH_TOKEN` is a local server configuration name: any MCP HTTP client
can use it as an `Authorization: Bearer …` header. HTTP/SSE data-plane startup
requires this token. Stdio uses the parent process boundary instead.

## Optional Docker deployment

The supplied Compose file starts only the GeoAI server. It binds to loopback,
uses the locked CPU pattern dependencies and creates one data volume:

```sh
export GEOAI_MESH_AUTH_TOKEN="$(python3 -c 'import secrets; print(secrets.token_urlsafe(32))')"
docker compose up --build -d
```

No GPU, database or object-store service is launched. To use S3-backed tools,
provide your own endpoint and credentials. To change the published port, set
`GEOAI_HOST_PORT` in your shell, for example `5051`.

## Configuration

See `.env.example` for common server settings and `src/config.py` for the
complete validated settings. A local `.env` file is optional and ignored by Git.

| Setting | Purpose |
| --- | --- |
| `MCP_TRANSPORT` | `http` (default), `stdio` or `sse` |
| `HOST`, `PORT` | Network bind address and port; ignored for stdio |
| `GEOAI_MESH_AUTH_TOKEN` | Optional HTTP bearer; required when enabling data-plane access |
| `STAC_URL` | Satellite catalogue; defaults to public Earth Search |
| `S3_ENDPOINT`, `S3_ACCESS_KEY`, `S3_SECRET_KEY`, `S3_BUCKET` | Your S3-compatible storage, when needed |
| `ALLOW_LOCAL_PARQUET_SOURCE`, `LOCAL_PARQUET_SOURCE_ROOT` | Explicit opt-in to a confined local Parquet directory |
| `FORECAST_API_BASE_URL`, `FORECAST_API_KEY` | Forecast endpoint and optional matching API key |

Storage is optional for starting the server. Local Parquet access is disabled
by default; enabling it requires an absolute allowed directory other than `/`.
S3 paths retain explicit tenant/space prefixes. Do not treat a supplied path or
an evidence digest as caller authorization.

Solar collection has separate opt-in settings in `src/solar_pilot.py`; an
example is `examples/solar.env.example`. Use the provider access and endpoint
appropriate to your use. External APIs, imagery and datasets retain their own
access conditions, attribution and licences.

## Development

```sh
uv sync --locked --extra dev --extra patterns
uv run --no-sync ruff check .
uv run --no-sync ruff format --check .
OMP_NUM_THREADS=1 OPENBLAS_NUM_THREADS=1 MKL_NUM_THREADS=1 PYTHONHASHSEED=0 \
  uv run --no-sync pytest tests/
```

`tests/live_acceptance.py` is a separate opt-in harness requiring a running
server and configured test storage. The normal test suite runs offline with
synthetic or frozen fixtures. Packaged runtime examples retain their scientific
scope and provenance; they are not a certification of operational forecasts.

Public documentation is maintained in [this README](README.md) and the
[MCP tools reference](reference/mcp-tools.md). Private project documentation and
local research outputs remain excluded: `docs/`, `docs-private/`, `research-data/`
and `outputs/` are ignored; adding local documents does not expose them as MCP
resources.

## Licence

The existing MIT licence is preserved in `LICENSE`. Third-party libraries
and data keep their own licences and attribution; see `NOTICE` and `CITATION.cff` for
scientific references. Optional `ai` dependencies are outside the default CPU
image and have separate upstream requirements.

## Citation

If you use GeoAI in research, a report or another project, you can copy this
software citation:

```text
Nicolas Geysse, Jonathan Labéjof, Dickson Owuor & Amandine Raynaud (2026).
Deposium GeoAI (version 0.1.0) [Computer software].
https://deposium.ai/
Funded by FIERCE through its Financial Support Programme.
```

**BibTeX**

```bibtex
@software{deposium_geoai_2026,
  author  = {Geysse, Nicolas and Labéjof, Jonathan and Owuor, Dickson and Raynaud, Amandine},
  title   = {Deposium GeoAI},
  year    = {2026},
  version = {0.1.0},
  url     = {https://deposium.ai/},
  note    = {Funded by FIERCE through its Financial Support Programme.
             FIERCE receives European Union funding under Horizon Europe,
             Grant Agreement 101177496.}
}
```

For reproducibility, also record the commit SHA or release tag you used.
Machine-readable citation metadata and references for the methods and data
are available in [CITATION.cff](CITATION.cff).
