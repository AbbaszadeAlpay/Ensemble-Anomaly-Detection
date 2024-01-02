FROM python:3.10-slim

WORKDIR /app

RUN pip install poetry

RUN poetry config virtualenvs.create false

COPY pyproject.toml /app/

RUN poetry install --with main

COPY . /app

EXPOSE 8041

CMD ["uvicorn", "src.model_deployment.api:app", "--host", "0.0.0.0",  "--port", "8041"]