# Contributing to ax-prover

Thanks for your interest in contributing! Whether it's a bug report, a feature idea, or a pull request — all contributions are welcome.

## Ways to Contribute

- **Report bugs** — open an [issue](https://github.com/Axiomatic-AI/ax-prover-base/issues) with steps to reproduce, expected vs actual behavior, and your environment (OS, Python version, Lean version)
- **Suggest features** — open an issue describing the use case and proposed solution
- **Improve documentation** — fix typos, clarify instructions, or add examples
- **Submit code** — fix a bug, implement a feature, or improve test coverage

## Development Setup

```bash
# Clone the repository
git clone https://github.com/Axiomatic-AI/ax-prover-base.git
cd ax-prover

# Create and activate a virtual environment
python -m venv .venv
source .venv/bin/activate  # On Windows: .venv\Scripts\activate

# Install with dev dependencies
pip install -e ".[dev]"

# Set up pre-commit hooks (runs ruff formatting on every commit)
pre-commit install
```

## Running Tests

```bash
# Run unit tests
.venv/bin/pytest tests/unit

# Run a specific test file
.venv/bin/pytest tests/unit/utils/test_files.py

# Run a single test
.venv/bin/pytest tests/unit/utils/test_files.py::test_function_name
```

## Code Style

We use [Ruff](https://docs.astral.sh/ruff/) for formatting and linting, enforced automatically by pre-commit hooks.

To format manually:

```bash
ruff format .
ruff check --fix .
```

## Submitting Changes

1. Fork the repository and create a branch from `main`
2. Make your changes
3. Ensure tests pass: `.venv/bin/pytest tests/unit`
4. Ensure code is formatted: `ruff format --check .`
5. Open a pull request against `main`

CI will run linting and tests automatically on your PR. All checks must pass before merging.

## Questions?

If you're unsure about anything, feel free to open an issue and ask. We're happy to help.
