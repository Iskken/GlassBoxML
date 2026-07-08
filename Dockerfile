# Slim official Python base image (small footprint, no unnecessary OS tooling)
FROM python:3.11-slim

# Prevent .pyc files and force stdout/stderr to be unbuffered (logs show up immediately)
ENV PYTHONDONTWRITEBYTECODE=1 \
    PYTHONUNBUFFERED=1

WORKDIR /app

# Copy only dependency manifests first so Docker can cache the pip install
# layer and skip re-installing when just the source code changes
COPY pyproject.toml ./
COPY webapp/requirements.txt ./webapp/requirements.txt

# Copy the source needed to build/install the glassboxml package and run the webapp
COPY glassboxml/ ./glassboxml/
COPY webapp/ ./webapp/

# Install glassboxml (editable) and the webapp's own dependencies (fastapi, uvicorn)
RUN pip install --no-cache-dir -e . \
    && pip install --no-cache-dir -r webapp/requirements.txt

# Document the port uvicorn will listen on (informational; doesn't publish it)
EXPOSE 8000

# Start the FastAPI app with uvicorn, bound to all interfaces so it's reachable
# from outside the container
CMD ["uvicorn", "webapp.app:app", "--host", "0.0.0.0", "--port", "8000"]
