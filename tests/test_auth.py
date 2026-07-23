from __future__ import annotations

import os
from unittest.mock import MagicMock

import pytest
from fastapi import HTTPException

from deepeval_eval.auth import (
    AuthManager,
    Role,
    UserContext,
    allow_unauthenticated_access,
    require_authenticated_user,
)


def test_user_context_contract_positive():
    """Verify UserContext fields and immutability."""
    user = UserContext(
        subject="sub_123",
        email="test@example.com",
        role=Role.ADMIN,
        is_authenticated=True,
    )
    assert user.subject == "sub_123"
    assert user.email == "test@example.com"
    assert user.role == Role.ADMIN
    assert user.is_authenticated is True


def test_user_context_immutability_negative():
    """Verify UserContext is frozen and rejects field mutation."""
    user = UserContext(email="test@example.com", is_authenticated=True)
    with pytest.raises((TypeError, Exception)):
        user.email = "other@example.com"  # type: ignore[misc]


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

    # Reset
    os.environ.pop("ALLOW_UNAUTHENTICATED_ACCESS", None)
    os.environ.pop("CAIPE_UNSAFE_RBAC_BYPASS", None)


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
