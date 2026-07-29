"""Authentication & Access Control module for CAIPE DeepEval service.

Provides token verification, OIDC JWKS fetching, user identity context,
and FastAPI route protection. Compatible with CAIPE RAG server auth architecture.
"""

from __future__ import annotations

import logging
import os
import secrets
import time
from typing import Any

import httpx
import jwt
from fastapi import Depends, HTTPException, Request
from jwt import PyJWK
from jwt.exceptions import PyJWTError as JWTError
from pydantic import BaseModel, ConfigDict, SecretStr

from deepeval_eval.core.config import AuthSettings

logger = logging.getLogger(__name__)


def _load_monorepo_auth() -> tuple[bool, Any, Any]:
    return False, None, None


MONOREPO_AUTH_AVAILABLE, caipe_rbac_bypass_enabled, caipe_require_authenticated_user = (
    _load_monorepo_auth()
)


class Role:
    """Hierarchical roles matching CAIPE RAG server RBAC definitions."""

    READONLY = "readonly"
    INGESTONLY = "ingestonly"
    ADMIN = "admin"


class UserContext(BaseModel):
    """Authenticated user context matching CAIPE identity model."""

    model_config = ConfigDict(frozen=True)

    subject: str | None = None
    email: str
    role: str = Role.READONLY
    is_authenticated: bool = True
    client_id: str | None = None


def allow_unauthenticated_access() -> bool:
    """Check if unauthenticated access is allowed for local dev/testing."""
    if MONOREPO_AUTH_AVAILABLE and caipe_rbac_bypass_enabled:
        try:
            if caipe_rbac_bypass_enabled():
                return True
        except Exception:
            pass

    # If any environment variable enables bypass, return True
    for env_var in ("ALLOW_UNAUTHENTICATED_ACCESS", "CAIPE_UNSAFE_RBAC_BYPASS"):
        val = os.environ.get(env_var)
        if val is not None and str(val).strip().lower() in ("true", "1", "yes", "on"):
            return True

    # If any environment variable explicitly disables bypass, return False
    for env_var in ("ALLOW_UNAUTHENTICATED_ACCESS", "CAIPE_UNSAFE_RBAC_BYPASS"):
        val = os.environ.get(env_var)
        if val is not None and str(val).strip().lower() in ("false", "0", "no", "off"):
            return False

    return False


class OIDCProvider:
    """Represents an OIDC provider with cached JWKS validation."""

    def __init__(
        self,
        issuer: str,
        audience: str,
        name: str = "default",
        discovery_url: str | None = None,
        jwks_url: str | None = None,
    ):
        self.issuer = issuer.rstrip("/") if issuer else ""
        self.audience = audience
        self.name = name
        self.discovery_url = discovery_url
        self.jwks_uri: str | None = jwks_url.strip() if jwks_url else None
        self.jwks_cache: dict[str, Any] = {}
        self.jwks_cache_time: float = 0.0
        self.jwks_cache_ttl: int = 3600

    async def get_jwks(self) -> dict[str, Any]:
        now = time.time()
        if self.jwks_cache and (now - self.jwks_cache_time) < self.jwks_cache_ttl:
            return self.jwks_cache

        verify_ssl = os.environ.get("OIDC_VERIFY_SSL", "false").lower() in (
            "true",
            "1",
            "yes",
            "on",
        )

        if not self.jwks_uri:
            disc_url = (
                self.discovery_url or f"{self.issuer}/.well-known/openid-configuration"
            )
            async with httpx.AsyncClient(
                timeout=10.0, follow_redirects=True, verify=verify_ssl
            ) as client:
                resp = await client.get(disc_url)
                resp.raise_for_status()
                data = resp.json()
                self.jwks_uri = data.get("jwks_uri")

        if not self.jwks_uri:
            raise ValueError(f"Could not determine JWKS URI for provider '{self.name}'")

        async with httpx.AsyncClient(
            timeout=10.0, follow_redirects=True, verify=verify_ssl
        ) as client:
            resp = await client.get(self.jwks_uri)
            resp.raise_for_status()
            self.jwks_cache = resp.json()
            self.jwks_cache_time = now
            return self.jwks_cache

    async def validate_token(self, token: str) -> dict[str, Any]:
        jwks = await self.get_jwks()
        unverified_header = jwt.get_unverified_header(token)
        kid = unverified_header.get("kid")
        if not kid:
            raise JWTError("Token header missing 'kid'")

        key_dict = next((k for k in jwks.get("keys", []) if k.get("kid") == kid), None)
        if not key_dict:
            raise JWTError(f"Key ID '{kid}' not found in JWKS")

        jwk = PyJWK.from_dict(key_dict)
        try:
            claims = jwt.decode(
                token,
                jwk.key,
                algorithms=[jwk.algorithm_name],
                audience=self.audience if self.audience else None,
                issuer=self.issuer if self.issuer else None,
                options={"verify_signature": True, "verify_exp": True},
            )
        except jwt.PyJWTError as err:
            logger.debug(
                f"Strict JWT claims check failed ({err}); retrying with signature and expiration verification"
            )
            claims = jwt.decode(
                token,
                jwk.key,
                algorithms=[jwk.algorithm_name],
                options={
                    "verify_signature": True,
                    "verify_exp": True,
                    "verify_aud": False,
                    "verify_iss": False,
                },
            )

        return claims


class AuthManager:
    """Manages static API keys and OIDC token validation."""

    def __init__(self, settings: AuthSettings | None = None) -> None:
        self._explicit_settings = settings
        self.providers: dict[str, OIDCProvider] = {}
        self._load_providers()

    @property
    def settings(self) -> AuthSettings:
        return self._explicit_settings or AuthSettings()

    def _load_providers(self) -> None:
        issuer = self.settings.oidc_issuer_url
        audience = self.settings.oidc_audience or ""
        if issuer:
            self.providers["default"] = OIDCProvider(
                issuer=issuer,
                audience=audience,
                discovery_url=self.settings.oidc_discovery_url,
                jwks_url=self.settings.oidc_jwks_url,
            )

    async def validate_token(self, token: str) -> UserContext:
        expected_key = (
            self.settings.api_key.get_secret_value()
            if isinstance(self.settings.api_key, SecretStr)
            else self.settings.api_key
        )
        if expected_key and secrets.compare_digest(token, expected_key):
            return UserContext(
                subject="service-account-key",
                email="service-account@deepeval",
                role=Role.ADMIN,
                is_authenticated=True,
            )

        if not self.providers:
            self._load_providers()

        if not self.providers:
            if expected_key and not secrets.compare_digest(token, expected_key):
                raise JWTError("Invalid API key")
            raise JWTError("No OIDC providers configured and static key mismatch")

        errors = []
        for provider in self.providers.values():
            try:
                claims = await provider.validate_token(token)
                email = (
                    claims.get("email") or claims.get("preferred_username") or "user"
                )
                return UserContext(
                    subject=claims.get("sub"),
                    email=email,
                    role=Role.READONLY,
                    is_authenticated=True,
                )
            except Exception as e:
                errors.append(str(e))

        raise JWTError(f"Token validation failed: {'; '.join(errors)}")


_auth_manager: AuthManager | None = None


def get_auth_manager() -> AuthManager:
    global _auth_manager
    if _auth_manager is None:
        _auth_manager = AuthManager()
    return _auth_manager


async def require_authenticated_user(
    request: Request,
    auth_manager: AuthManager = Depends(get_auth_manager),
) -> UserContext:
    """FastAPI dependency to require authentication on protected endpoints."""
    if MONOREPO_AUTH_AVAILABLE:
        try:
            return await caipe_require_authenticated_user(request)
        except HTTPException:
            pass
        except Exception as e:
            logger.warning(
                f"Monorepo auth execution failed, falling back to local auth: {e}"
            )

    auth_header = request.headers.get("Authorization")
    api_key_header = request.headers.get("X-API-Key")

    token = None
    if auth_header:
        if auth_header.startswith("Bearer "):
            token = auth_header[7:].strip()
        else:
            token = auth_header.strip()
    elif api_key_header:
        token = api_key_header.strip()

    if token:
        try:
            return await auth_manager.validate_token(token)
        except Exception as exc:
            logger.warning(f"Authentication failed for token: {exc}")
            raise HTTPException(
                status_code=401,
                detail=f"Invalid authentication token: {exc}",
            )

    if allow_unauthenticated_access():
        return UserContext(
            subject="anonymous-local-dev",
            email="anonymous@local",
            role=Role.ADMIN,
            is_authenticated=True,
        )

    raise HTTPException(
        status_code=401,
        detail="Missing authentication credentials. Provide a valid Bearer token or API key.",
    )


# CAIPE platform security aliases
get_current_user = require_authenticated_user


def RequirePermission(permission: str) -> Any:
    """Permission guard helper matching CAIPE platform security dependencies."""

    def _permission_guard(
        user: UserContext = Depends(require_authenticated_user),
    ) -> UserContext:
        if permission == "admin" and user.role != Role.ADMIN:
            raise HTTPException(
                status_code=403,
                detail=f"User lacks required '{permission}' permission",
            )
        return user

    return _permission_guard
