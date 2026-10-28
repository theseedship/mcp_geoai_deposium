"""GeoAI HTTP mesh authentication tests."""

from __future__ import annotations

import asyncio

from src.auth import MeshSharedSecretVerifier, build_mesh_auth


def test_mesh_shared_secret_verifier_accepts_only_the_exact_token() -> None:
    secret = "geoai-mesh-test-secret-at-least-32-characters"
    verifier = MeshSharedSecretVerifier(secret)

    accepted = asyncio.run(verifier.verify_token(secret))
    rejected = asyncio.run(verifier.verify_token(secret + "-wrong"))

    assert accepted is not None
    assert accepted.client_id == "geoai-client"
    assert accepted.scopes == ["geoai:mesh"]
    assert accepted.token == "verified"
    assert rejected is None


def test_empty_mesh_secret_leaves_standalone_server_unprotected() -> None:
    assert build_mesh_auth("") is None
    assert build_mesh_auth("   ") is None
