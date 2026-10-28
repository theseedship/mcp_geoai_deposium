"""Service-to-service authentication for HTTP data-plane deployments."""

from __future__ import annotations

import hmac

from fastmcp.server.auth import AccessToken, TokenVerifier


class MeshSharedSecretVerifier(TokenVerifier):
    """Verify one opaque bearer secret without storing caller identity claims."""

    def __init__(self, secret: str) -> None:
        super().__init__(required_scopes=["geoai:mesh"])
        self._secret = secret.encode("utf-8")

    async def verify_token(self, token: str) -> AccessToken | None:
        if not hmac.compare_digest(token.encode("utf-8"), self._secret):
            return None
        return AccessToken(
            token="verified",
            client_id="geoai-client",
            scopes=["geoai:mesh"],
            claims={"auth_type": "shared_secret"},
        )


def build_mesh_auth(secret: str) -> MeshSharedSecretVerifier | None:
    normalized = secret.strip()
    return MeshSharedSecretVerifier(normalized) if normalized else None
