#!/bin/bash
# Run the containerized GlassBoxML webapp locally.

set -e  # exit immediately if any command fails

# Start a container from the "glassboxml-webapp" image (see build.sh),
# mapping container port 8000 to the same port on the host
docker run -p 8000:8000 glassboxml-webapp
