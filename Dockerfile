FROM python:3.12-slim

# Install poetry and clear cache
RUN apt-get update \
    && apt-get install -y curl build-essential \
    && curl -sSL https://install.python-poetry.org | python3 - \
    && apt-get purge -y curl build-essential \
    && apt-get autoremove -y \
    && rm -rf /var/lib/apt/lists/* \
    && rm -rf /root/.cache

ENV PATH="/root/.local/bin:$PATH"

WORKDIR /app
COPY pyproject.toml poetry.lock* /app/

# Configure, install Poetry project, and clear cache
RUN poetry config virtualenvs.create false \
    && poetry config installer.max-workers 1 \
    && poetry install --no-interaction --no-ansi --no-root --only=main \
    && poetry cache clear --all pypi \
    && rm -rf /root/.cache

COPY . .

# Expose ports for streamlit and fastAPI
EXPOSE 8501
EXPOSE 8000

