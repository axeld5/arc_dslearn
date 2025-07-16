FROM python:3.11-slim
WORKDIR /work
COPY . /work
RUN pip install --upgrade pip uv && uv pip install .[dev]
CMD ["python", "-m", "pytest"]
