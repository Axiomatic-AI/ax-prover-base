"""Google Cloud authentication utilities."""

import google.auth.transport.requests
import google.oauth2.id_token

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
