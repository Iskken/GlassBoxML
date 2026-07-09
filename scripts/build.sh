#!/bin/bash
# Build the GlassBoxML webapp Docker image.

set -e  # exit immediately if any command fails

# Run from the repo root so the Dockerfile's COPY paths (glassboxml/, webapp/) resolve correctly
cd "$(dirname "$0")/.."

# Build the image and tag it "glassboxml-webapp"
docker build -t glassboxml-webapp .
