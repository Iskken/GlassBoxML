#!/bin/bash
# Run the GlassBoxML test suite.

set -e  # exit immediately if any command fails

# Run from the repo root so pytest picks up pyproject.toml and the tests/ directory
cd "$(dirname "$0")/.."

# Run the test suite
pytest
