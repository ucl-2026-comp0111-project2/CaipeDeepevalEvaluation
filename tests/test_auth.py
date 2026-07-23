from __future__ import annotations

import os
from unittest.mock import AsyncMock, MagicMock, patch

import pytest
from fastapi import HTTPException

from deepeval_eval.auth import (
    AuthManager,
    OIDCProvider,
    RequirePermission,
    Role,
    UserContext,
    _load_monorepo_auth,
    allow_unauthenticated_access,
    get_auth_manager,
    require_authenticated_user,
)


def test_user_context_contract_positive():
    """Verify UserContext fields and immutability."""
    user = UserContext(
        subject="sub_123",
        email="test@example.com",
        role=Role.ADMIN,
        is_authenticated=True,
        client_id="client_1",
    )
    assert user.subject == "sub_123"
    assert user.email == "test@example.com"
    assert user.role == Role.ADMIN
    assert user.is_authenticated is True
    assert user.client_id == "client_1"


def test_user_context_immutability_negative():
    """Verify UserContext is frozen and rejects field mutation."""
    user = UserContext(email="test@example.com", is_authenticated=True)
    with pytest.raises((TypeError, Exception)):
        user.email = "other@example.com"  # type: ignore[misc]


def test_load_monorepo_auth_positive():
    """Verify _load_monorepo_auth returns tuple indicating status."""
    avail, bypass, req_user = _load_monorepo_auth()
    assert isinstance(avail, bool)


def test_load_monorepo_auth_import_error_negative():
    """Verify _load_monorepo_auth handles missing modules gracefully."""
    with patch("importlib.import_module", side_effect=ImportError("No module")):
        avail, bypass, req_user = _load_monorepo_auth()
        assert avail is False
        assert bypass is None
        assert req_user is None


def test_allow_unauthenticated_access_toggle():
    """Verify allow_unauthenticated_access respects environment variable settings."""
    os.environ["ALLOW_UNAUTHENTICATED_ACCESS"] = "true"
    assert allow_unauthenticated_access() is True

    os.environ["ALLOW_UNAUTHENTICATED_ACCESS"] = "false"
    os.environ.pop("CAIPE_UNSAFE_RBAC_BYPASS", None)
    assert allow_unauthenticated_access() is False

    os.environ.pop("ALLOW_UNAUTHENTICATED_ACCESS", None)
    os.environ["CAIPE_UNSAFE_RBAC_BYPASS"] = "1"
    assert allow_unauthenticated_access() is True

    os.environ.pop("ALLOW_UNAUTHENTICATED_ACCESS", None)
    os.environ.pop("CAIPE_UNSAFE_RBAC_BYPASS", None)


def test_allow_unauthenticated_access_monorepo_positive():
    """Verify allow_unauthenticated_access returns True when monorepo bypass is enabled."""
    with (
        patch("deepeval_eval.auth.MONOREPO_AUTH_AVAILABLE", True),
        patch("deepeval_eval.auth.caipe_rbac_bypass_enabled", return_value=True),
    ):
        assert allow_unauthenticated_access() is True


def test_allow_unauthenticated_access_monorepo_exception_negative():
    """Verify allow_unauthenticated_access handles monorepo bypass exception gracefully."""
    with (
        patch("deepeval_eval.auth.MONOREPO_AUTH_AVAILABLE", True),
        patch(
            "deepeval_eval.auth.caipe_rbac_bypass_enabled",
            side_effect=RuntimeError("Bypass error"),
        ),
        patch.dict("os.environ", {"ALLOW_UNAUTHENTICATED_ACCESS": "false"}),
    ):
        assert allow_unauthenticated_access() is False


@pytest.mark.asyncio
async def test_oidc_provider_get_jwks_cached_positive():
    """Verify OIDCProvider returns cached JWKS if not expired."""
    provider = OIDCProvider(
        issuer="https://issuer.com",
        audience="aud",
        jwks_url="https://issuer.com/jwks",
    )
    provider.jwks_cache = {"keys": [{"kid": "k1"}]}
    provider.jwks_cache_time = 9999999999.0

    jwks = await provider.get_jwks()
    assert jwks == {"keys": [{"kid": "k1"}]}


@pytest.mark.asyncio
async def test_oidc_provider_get_jwks_http_positive():
    """Verify OIDCProvider fetches discovery and JWKS over HTTP."""
    provider = OIDCProvider(issuer="https://issuer.com", audience="aud")

    mock_disc_resp = MagicMock()
    mock_disc_resp.json.return_value = {"jwks_uri": "https://issuer.com/jwks"}
    mock_disc_resp.raise_for_status = MagicMock()

    mock_jwks_resp = MagicMock()
    mock_jwks_resp.json.return_value = {"keys": [{"kid": "key1"}]}
    mock_jwks_resp.raise_for_status = MagicMock()

    mock_client = AsyncMock()
    mock_client.get.side_effect = [mock_disc_resp, mock_jwks_resp]
    mock_client.__aenter__.return_value = mock_client
    mock_client.__aexit__.return_value = None

    with patch("httpx.AsyncClient", return_value=mock_client):
        jwks = await provider.get_jwks()
        assert jwks == {"keys": [{"kid": "key1"}]}


@pytest.mark.asyncio
async def test_oidc_provider_get_jwks_missing_uri_negative():
    """Verify OIDCProvider raises ValueError if discovery returns no jwks_uri."""
    provider = OIDCProvider(issuer="https://issuer.com", audience="aud")

    mock_disc_resp = MagicMock()
    mock_disc_resp.json.return_value = {}
    mock_disc_resp.raise_for_status = MagicMock()

    mock_client = AsyncMock()
    mock_client.get.return_value = mock_disc_resp
    mock_client.__aenter__.return_value = mock_client
    mock_client.__aexit__.return_value = None

    with (
        patch("httpx.AsyncClient", return_value=mock_client),
        pytest.raises(ValueError, match="Could not determine JWKS URI"),
    ):
        await provider.get_jwks()


@pytest.mark.asyncio
async def test_oidc_provider_validate_token_positive():
    """Verify OIDCProvider validates valid token successfully."""
    provider = OIDCProvider(
        issuer="https://issuer.com",
        audience="aud",
        jwks_url="https://issuer.com/jwks",
    )
    provider.get_jwks = AsyncMock(
        return_value={"keys": [{"kid": "k1", "kty": "RSA", "n": "abc", "e": "AQAB"}]}
    )

    with (
        patch("jwt.get_unverified_header", return_value={"kid": "k1"}),
        patch(
            "jwt.PyJWK.from_dict",
            return_value=MagicMock(key="pubkey", algorithm_name="RS256"),
        ),
        patch(
            "jwt.decode",
            return_value={
                "sub": "s1",
                "email": "user@test.com",
                "preferred_username": "p_user",
            },
        ),
    ):
        claims = await provider.validate_token("valid_token")
        assert claims["email"] == "user@test.com"


@pytest.mark.asyncio
async def test_oidc_provider_validate_token_missing_kid_negative():
    """Verify OIDCProvider raises JWTError when token header is missing kid."""
    provider = OIDCProvider(
        issuer="https://issuer.com",
        audience="aud",
        jwks_url="https://issuer.com/jwks",
    )
    provider.get_jwks = AsyncMock(return_value={"keys": []})

    with (
        patch("jwt.get_unverified_header", return_value={}),
        pytest.raises(Exception, match="missing 'kid'"),
    ):
        await provider.validate_token("token_no_kid")


@pytest.mark.asyncio
async def test_oidc_provider_validate_token_kid_not_found_negative():
    """Verify OIDCProvider raises JWTError when kid is not found in JWKS."""
    provider = OIDCProvider(
        issuer="https://issuer.com",
        audience="aud",
        jwks_url="https://issuer.com/jwks",
    )
    provider.get_jwks = AsyncMock(return_value={"keys": [{"kid": "k1"}]})

    with (
        patch("jwt.get_unverified_header", return_value={"kid": "k99"}),
        pytest.raises(Exception, match="not found in JWKS"),
    ):
        await provider.validate_token("token_unknown_kid")


@pytest.mark.asyncio
async def test_auth_manager_static_api_key_positive():
    """Verify AuthManager validates matching DEEPEVAL_API_KEY static token."""
    os.environ["DEEPEVAL_API_KEY"] = "secret_key_123"
    am = AuthManager()

    user = await am.validate_token("secret_key_123")
    assert user.is_authenticated is True
    assert user.role == Role.ADMIN
    assert "service-account" in user.email

    os.environ.pop("DEEPEVAL_API_KEY", None)


@pytest.mark.asyncio
async def test_auth_manager_static_api_key_negative():
    """Verify AuthManager raises error for mismatched static API key."""
    os.environ["DEEPEVAL_API_KEY"] = "secret_key_123"
    am = AuthManager()

    with pytest.raises(Exception):
        await am.validate_token("invalid_key_456")

    os.environ.pop("DEEPEVAL_API_KEY", None)


@pytest.mark.asyncio
async def test_auth_manager_provider_load_positive():
    """Verify AuthManager loads provider when env vars set."""
    with patch.dict(
        "os.environ",
        {"OIDC_ISSUER_URL": "https://auth.com", "OIDC_AUDIENCE": "deepeval"},
    ):
        am = AuthManager()
        assert "default" in am.providers


@pytest.mark.asyncio
async def test_auth_manager_oidc_token_positive():
    """Verify AuthManager validates OIDC token using provider."""
    am = AuthManager()
    mock_provider = AsyncMock()
    mock_provider.validate_token.return_value = {
        "sub": "user_1",
        "email": "user@domain.com",
    }
    am.providers = {"default": mock_provider}

    user = await am.validate_token("oidc_token_123")
    assert user.subject == "user_1"
    assert user.email == "user@domain.com"


@pytest.mark.asyncio
async def test_auth_manager_all_providers_fail_negative():
    """Verify AuthManager raises error when all providers fail."""
    am = AuthManager()
    mock_provider = AsyncMock()
    mock_provider.validate_token.side_effect = Exception("Expired token")
    am.providers = {"default": mock_provider}

    with pytest.raises(Exception, match="Token validation failed"):
        await am.validate_token("bad_token")


def test_get_auth_manager_positive():
    """Verify get_auth_manager returns cached singleton AuthManager."""
    am1 = get_auth_manager()
    am2 = get_auth_manager()
    assert am1 is am2


@pytest.mark.asyncio
async def test_require_authenticated_user_unauthenticated_mode():
    """Verify require_authenticated_user returns dev user context when unauthenticated access is allowed."""
    os.environ["ALLOW_UNAUTHENTICATED_ACCESS"] = "true"
    request = MagicMock()
    request.headers = {}

    user = await require_authenticated_user(request)
    assert user.is_authenticated is True
    assert user.email == "anonymous@local"

    os.environ.pop("ALLOW_UNAUTHENTICATED_ACCESS", None)


@pytest.mark.asyncio
async def test_require_authenticated_user_bearer_token_positive():
    """Verify require_authenticated_user extracts Bearer token and validates."""
    request = MagicMock()
    request.headers = {"Authorization": "Bearer valid_bearer_123"}
    mock_am = AsyncMock()
    mock_am.validate_token.return_value = UserContext(subject="s1", email="u@test.com")

    user = await require_authenticated_user(request, auth_manager=mock_am)
    assert user.email == "u@test.com"
    mock_am.validate_token.assert_called_once_with("valid_bearer_123")


@pytest.mark.asyncio
async def test_require_authenticated_user_api_key_header_positive():
    """Verify require_authenticated_user extracts X-API-Key header and validates."""
    request = MagicMock()
    request.headers = {"X-API-Key": "my_api_key"}
    mock_am = AsyncMock()
    mock_am.validate_token.return_value = UserContext(subject="s1", email="u@test.com")

    user = await require_authenticated_user(request, auth_manager=mock_am)
    assert user.email == "u@test.com"
    mock_am.validate_token.assert_called_once_with("my_api_key")


@pytest.mark.asyncio
async def test_require_authenticated_user_invalid_token_negative():
    """Verify require_authenticated_user raises 401 when validate_token fails."""
    request = MagicMock()
    request.headers = {"Authorization": "Bearer invalid_token"}
    mock_am = AsyncMock()
    mock_am.validate_token.side_effect = Exception("Token expired")

    with pytest.raises(HTTPException) as exc_info:
        await require_authenticated_user(request, auth_manager=mock_am)

    assert exc_info.value.status_code == 401
    assert "Invalid authentication token" in exc_info.value.detail


@pytest.mark.asyncio
async def test_require_authenticated_user_monorepo_positive():
    """Verify require_authenticated_user calls caipe_require_authenticated_user if MONOREPO_AUTH_AVAILABLE."""
    request = MagicMock()
    mock_user = UserContext(subject="m1", email="m@test.com")
    mock_require_user = AsyncMock(return_value=mock_user)
    with (
        patch("deepeval_eval.auth.MONOREPO_AUTH_AVAILABLE", True),
        patch("deepeval_eval.auth.caipe_require_authenticated_user", mock_require_user),
    ):
        user = await require_authenticated_user(request)
        assert user.subject == "m1"


@pytest.mark.asyncio
async def test_require_authenticated_user_monorepo_http_exception_fallback_positive():
    """Verify require_authenticated_user falls back when monorepo auth raises HTTPException."""
    request = MagicMock()
    request.headers = {}
    with (
        patch("deepeval_eval.auth.MONOREPO_AUTH_AVAILABLE", True),
        patch(
            "deepeval_eval.auth.caipe_require_authenticated_user",
            side_effect=HTTPException(status_code=401),
        ),
        patch.dict("os.environ", {"ALLOW_UNAUTHENTICATED_ACCESS": "true"}),
    ):
        user = await require_authenticated_user(request)
        assert user.email == "anonymous@local"


@pytest.mark.asyncio
async def test_require_authenticated_user_rejected_negative():
    """Verify require_authenticated_user raises 401 when auth is missing and unauthenticated access is disabled."""
    os.environ["ALLOW_UNAUTHENTICATED_ACCESS"] = "false"
    os.environ.pop("CAIPE_UNSAFE_RBAC_BYPASS", None)
    request = MagicMock()
    request.headers = {}

    with pytest.raises(HTTPException) as exc_info:
        await require_authenticated_user(request)

    assert exc_info.value.status_code == 401
    os.environ.pop("ALLOW_UNAUTHENTICATED_ACCESS", None)


def test_require_permission_positive():
    """Verify RequirePermission guard returns passed user."""
    guard = RequirePermission("admin:read")
    user = UserContext(subject="u1", email="admin@test.com", role=Role.ADMIN)
    res = guard(user)
    assert res == user
