"""Standalone script to test the LEAN_SEARCH_TOOL_TYPE tool with a single query.

Usage:
    python scripts/test_lean_search.py "continuity of functions"
    python scripts/test_lean_search.py "Mathlib.Topology.Basic" --server-url http://localhost:8080
    python scripts/test_lean_search.py "prime" --max-results 10
"""

import argparse
import asyncio
import sys

from ax_prover.tools.lean_search import (
    DEFAULT_LEAN_SEARCH_URL,
    SearchLeanSearchConfig,
    lean_search,
    lean_search_session_manager,
    warmup_lean_search,
)


async def run(query: str, config: SearchLeanSearchConfig) -> str:
    async with lean_search_session_manager():
        await warmup_lean_search(config)
        return await lean_search(query, config)


def main() -> None:
    parser = argparse.ArgumentParser(description="Test the LeanSearch tool with a single query.")
    parser.add_argument("query", help="Search term (module path or natural language).")
    parser.add_argument("--server-url", default=DEFAULT_LEAN_SEARCH_URL)
    parser.add_argument("--max-results", type=int, default=6)
    parser.add_argument("--timeout", type=int, default=60)
    parser.add_argument("--max-retries", type=int, default=3)
    parser.add_argument("--retry-delay", type=int, default=2)
    args = parser.parse_args()

    config = SearchLeanSearchConfig(
        server_url=args.server_url,
        max_results=args.max_results,
        timeout=args.timeout,
        max_retries=args.max_retries,
        retry_delay=args.retry_delay,
    )

    result = asyncio.run(run(args.query, config))
    sys.stdout.write(result + "\n")


if __name__ == "__main__":
    main()
