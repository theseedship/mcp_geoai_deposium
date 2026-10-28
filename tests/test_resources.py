import asyncio
import json
import re
import tomllib
from pathlib import Path

import pytest
from fastmcp import Client, FastMCP

import src.resources as resource_module
from src.contracts import tool_capabilities_json
from src.manifests import dataset_manifest_contract, dataset_manifest_json_schema
from src.resources import _DOC_RESOURCES, SERVER_INSTRUCTIONS, register_agent_resources

_REPO_ROOT = Path(__file__).resolve().parents[1]


async def _list_resources() -> list[str]:
    mcp = FastMCP("resource-test", instructions=SERVER_INSTRUCTIONS)
    register_agent_resources(mcp)
    async with Client(mcp) as client:
        resources = await client.list_resources()
        return sorted(str(resource.uri) for resource in resources)


async def _read_resource(uri: str) -> str:
    mcp = FastMCP("resource-test", instructions=SERVER_INSTRUCTIONS)
    register_agent_resources(mcp)
    async with Client(mcp) as client:
        contents = await client.read_resource(uri)
        return "\n".join(content.text for content in contents if hasattr(content, "text"))


async def _list_tools() -> list[str]:
    mcp = FastMCP("resource-test", instructions=SERVER_INSTRUCTIONS)
    register_agent_resources(mcp)
    async with Client(mcp) as client:
        tools = await client.list_tools()
        return sorted(tool.name for tool in tools)


async def _get_prompt(question: str) -> str:
    mcp = FastMCP("resource-test", instructions=SERVER_INSTRUCTIONS)
    register_agent_resources(mcp)
    async with Client(mcp) as client:
        prompt = await client.get_prompt("geoai_task", {"question": question})
        return "\n".join(message.content.text for message in prompt.messages)


def test_agent_resources_are_registered():
    uris = asyncio.run(_list_resources())

    assert "geoai://guide/agent" in uris
    assert {uri for uri in uris if uri.startswith("geoai://docs/")} == {"geoai://docs/readme"}
    assert "geoai://capabilities/tools" in uris
    assert "geoai://schemas/dataset-manifest-v1" in uris
    assert "geoai://contracts/dataset-manifest-v1" in uris


def test_agent_guide_explains_storage_free_and_data_plane_modes():
    guide = asyncio.run(_read_resource("geoai://guide/agent"))

    assert "forecast_query" in guide
    assert "discover_patterns" in guide
    assert "S3/RustFS" in guide
    assert "geoai://capabilities/tools" in guide
    assert "Good answer structure" in guide
    for tool_name in tool_capabilities_json()["tools"]:
        assert f"`{tool_name}`" in guide


def test_tool_capabilities_resource_matches_runtime_registry_without_adding_a_tool():
    payload = json.loads(asyncio.run(_read_resource("geoai://capabilities/tools")))

    assert payload == tool_capabilities_json()
    assert payload["tool_count"] == 16
    assert set(payload["tools"]) == set(tool_capabilities_json()["tools"])
    assert asyncio.run(_list_tools()) == []


def test_dataset_manifest_schema_resource_matches_runtime_model_without_adding_a_tool():
    payload = json.loads(asyncio.run(_read_resource("geoai://schemas/dataset-manifest-v1")))

    assert payload == dataset_manifest_json_schema()
    assert payload["$id"] == "https://deposium.ai/schemas/dataset-manifest/v1"
    assert asyncio.run(_list_tools()) == []


def test_dataset_manifest_contract_resource_includes_semantic_rules():
    payload = json.loads(asyncio.run(_read_resource("geoai://contracts/dataset-manifest-v1")))

    assert payload == dataset_manifest_contract()
    assert payload["validation_order"] == ["structural_schema", "semantic_rules"]
    assert asyncio.run(_list_tools()) == []


def test_markdown_docs_are_exposed_as_resources():
    readme = asyncio.run(_read_resource("geoai://docs/readme"))
    assert readme == (_REPO_ROOT / "README.md").read_text(encoding="utf-8")
    assert {relative_path for _, _, _, relative_path in _DOC_RESOURCES} == {"README.md"}


def test_geoai_task_prompt_guides_tool_use():
    prompt = asyncio.run(_get_prompt("Get a 7-day forecast for Nairobi"))

    assert "Get a 7-day forecast for Nairobi" in prompt
    assert "geoai://guide/agent" in prompt
    assert "storage-free tools" in prompt


def test_server_entrypoint_exposes_agent_resources():
    from src.server import mcp

    async def list_server_surface() -> tuple[list[str], list[str]]:
        async with Client(mcp) as client:
            resources = await client.list_resources()
            tools = await client.list_tools()
            return (
                sorted(str(resource.uri) for resource in resources),
                sorted(tool.name for tool in tools),
            )

    resources, tools = asyncio.run(list_server_surface())
    assert "geoai://guide/agent" in resources
    assert {uri for uri in resources if uri.startswith("geoai://docs/")} == {"geoai://docs/readme"}
    assert "geoai://capabilities/tools" in resources
    assert "geoai://schemas/dataset-manifest-v1" in resources
    assert "geoai://contracts/dataset-manifest-v1" in resources
    assert set(tools) == set(tool_capabilities_json()["tools"])
    assert "storage-free workflows" in mcp.instructions
    assert "geoai://contracts/dataset-manifest-v1" in mcp.instructions
    normalized_instructions = " ".join(mcp.instructions.split())
    assert "root-confined local Parquet path" in normalized_instructions
    assert "era5_ingest (when write_output=true)" in normalized_instructions
    assert "ndvi_ingest (when write_output=true)" in normalized_instructions
    instruction_tokens = set(re.findall(r"\b[a-z][a-z0-9_]*\b", mcp.instructions))
    missing_tools = set(tool_capabilities_json()["tools"]) - instruction_tokens
    assert not missing_tools


def _dockerfile_stages() -> list[tuple[str, str]]:
    """Return ``(base image, instructions without comments)`` for each Dockerfile stage."""
    dockerfile = (_REPO_ROOT / "Dockerfile").read_text(encoding="utf-8")
    instructions = "\n".join(
        line for line in dockerfile.splitlines() if not line.lstrip().startswith("#")
    )
    stages = re.split(r"(?m)^FROM ", instructions)[1:]
    return [(stage.split(None, 1)[0], stage) for stage in stages]


def test_docker_image_packages_exposed_doc_resources():
    pyproject = tomllib.loads((_REPO_ROOT / "pyproject.toml").read_text(encoding="utf-8"))
    force_include = pyproject["tool"]["hatch"]["build"]["targets"]["wheel"]["force-include"]
    (builder_base, builder), (runtime_base, runtime) = _dockerfile_stages()
    before_package_install = builder.split("uv pip install", 1)[0]

    # Builder: hatchling needs every force-included path and src/ before the locked install,
    # which selects CPU Torch explicitly.
    assert all(path in before_package_install for path in force_include)
    assert "COPY src/ src/" in before_package_install
    assert "--torch-backend cpu" in builder
    # Runtime: the public README, machine contracts, packaged data and package source.
    assert "COPY README.md ./" in runtime
    assert "COPY docs/" not in runtime
    assert "COPY docs/" not in builder
    assert "COPY src/ src/" in runtime
    assert "COPY --from=builder /opt/venv /opt/venv" in runtime
    assert "COPY --from=builder /opt/duckdb-home /opt/duckdb-home" in runtime
    # The cached extensions are found through HOME, by the server and its mesh subprocesses.
    assert "HOME=/opt/duckdb-home" in runtime
    assert 'CMD ["python", "-m", "src.server"]' in runtime
    assert "curl -f http://localhost:5001/health" in runtime
    assert "EXPOSE 5001" in runtime


def test_docker_runtime_stage_ships_no_build_tooling():
    (builder_base, _builder), (runtime_base, runtime) = _dockerfile_stages()

    # One base, pinned by digest, in both stages.
    assert "@sha256:" in builder_base
    assert builder_base == runtime_base
    assert not re.search(r"\buv\b", runtime)
    assert "-dev" not in runtime
    assert "build-essential" not in runtime


def test_wheel_packages_exposed_doc_resources():
    pyproject = tomllib.loads((_REPO_ROOT / "pyproject.toml").read_text(encoding="utf-8"))
    force_include = pyproject["tool"]["hatch"]["build"]["targets"]["wheel"]["force-include"]
    exposed_doc_paths = {relative_path for _, _, _, relative_path in _DOC_RESOURCES}

    assert force_include["README.md"] == "README.md"
    assert exposed_doc_paths <= set(force_include)
    assert all(force_include[path] == path for path in exposed_doc_paths)
    assert not any(path.startswith("docs/") for path in force_include)


@pytest.mark.parametrize(
    "relative_path",
    [
        "INSTALL.md",
        "MODES.md",
        "dev-harness.md",
        "TOOL_CAPABILITIES.md",
        "design/rustfs-storage-contract.md",
        "design/deposium-geoai-integration-contract.md",
        "design/deposium-mcps-geoai-handoff.md",
        "demo/private-evidence.md",
    ],
)
def test_local_private_documents_are_never_exposed(tmp_path, monkeypatch, relative_path):
    private = tmp_path / "docs" / relative_path
    private.parent.mkdir(parents=True, exist_ok=True)
    private.write_text("PRIVATE_DOCUMENT_SENTINEL", encoding="utf-8")
    (tmp_path / "README.md").write_text("Public project overview", encoding="utf-8")
    monkeypatch.setattr(resource_module, "_REPO_ROOT", tmp_path)

    uris = asyncio.run(_list_resources())
    assert {uri for uri in uris if uri.startswith("geoai://docs/")} == {"geoai://docs/readme"}
    assert asyncio.run(_read_resource("geoai://docs/readme")) == "Public project overview"
    guide = asyncio.run(_read_resource("geoai://guide/agent"))
    assert "PRIVATE_DOCUMENT_SENTINEL" not in guide
    assert "geoai://docs/" not in guide
    assert "geoai://docs/" not in SERVER_INSTRUCTIONS
