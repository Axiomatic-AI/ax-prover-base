import asyncio
from lean_explore.search import Service
import argparse

async def main():
    parser = argparse.ArgumentParser(description="Test the LeanSearch tool with a single query.")
    parser.add_argument("query", help="Search term (module path or natural language).")
    args = parser.parse_args()

    service = Service()  # spins up a default SearchEngine

    response = await service.search(
        query=args.query,
        limit=10,
        rerank_top=50,
        packages=["Mathlib"],
    )

    print(response)

    # for result in response.results:
    #     print(result.name, "-", result.module)
    #
    # # Look up one declaration by id
    # first = response.results[0]
    # same = await service.get_by_id(first.id)

asyncio.run(main())