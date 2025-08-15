FROM python:3.12-slim

# Install system dependencies for building packages
RUN apt-get update \
    && apt-get install -y curl build-essential \
    && rm -rf /var/lib/apt/lists/*

# Install Poetry
RUN curl -sSL https://install.python-poetry.org | python3 -
ENV PATH="/root/.local/bin:$PATH"

# Set working directory
WORKDIR /app

# Copy only pyproject.toml and poetry.lock first (to leverage Docker cache)
COPY pyproject.toml poetry.lock* /app/

# Install dependencies
RUN poetry config virtualenvs.create false \
    && poetry install --no-interaction --no-ansi --no-root

# Copy application code
COPY . .

# Default command
CMD ["streamlit", "run", "streamlit_app.py", "--server.address", "0.0.0.0"]
