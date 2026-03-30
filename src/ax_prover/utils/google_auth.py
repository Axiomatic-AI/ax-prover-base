"""Google Cloud authentication utilities."""

import google.auth
import google.auth.transport.requests
import google.oauth2.id_token
import httpx

from .logging import get_logger

logger = get_logger(__name__)


def get_auth_token(server_url: str) -> str:
    """Get a fresh ID token for Cloud Run authentication.

    Uses the VM metadata server to fetch an ID token.
    This works automatically on GCP VMs with attached service accounts.

    Args:
        server_url: The target audience (Cloud Run service URL)

    Returns:
        Fresh ID token

    Raises:
        google.auth.exceptions.DefaultCredentialsError: If not running on GCP VM
    """
    logger.debug(f"Fetching ID token for audience: {server_url}")
    request = google.auth.transport.requests.Request()
    token = google.oauth2.id_token.fetch_id_token(request, server_url)
    logger.debug("Successfully fetched ID token")
    return token


class VertexAIAuth(httpx.Auth):
    """Auto-refreshing Google OAuth2 auth for Vertex AI dedicated endpoints.

    Uses Application Default Credentials. On GCP, picks up the attached service account
    automatically. Locally, requires `gcloud auth application-default login`.
    """

    def __init__(self) -> None:
        self._credentials, _ = google.auth.default(
            scopes=["https://www.googleapis.com/auth/cloud-platform"]
        )

    def auth_flow(self, request: httpx.Request):  # type: ignore[override]
        if not self._credentials.valid:
            self._credentials.refresh(google.auth.transport.requests.Request())
        request.headers["Authorization"] = f"Bearer {self._credentials.token}"
        yield request
