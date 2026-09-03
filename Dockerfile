# x86_64 to match the EC2 build host. Pinned digest-free but version-pinned base;
# the pipeline itself is stdlib-only, so the image stays small.
FROM python:3.11-slim-bookworm

WORKDIR /app

COPY requirements.txt .
RUN pip install --no-cache-dir -r requirements.txt

COPY src/ src/
COPY eval/ eval/
COPY scripts/ scripts/
COPY Makefile .
# The CIViC snapshot is copied in rather than pulled at build time: the whole
# point of a dated snapshot is that the image and the results refer to the same
# corpus. Rebuild the image to move to a newer snapshot.
COPY data/snapshots/ data/snapshots/
COPY data/eval_sets/ data/eval_sets/
COPY results/ results/

EXPOSE 8000
HEALTHCHECK --interval=30s --timeout=5s --start-period=20s \
  CMD python -c "import urllib.request;urllib.request.urlopen('http://localhost:8000/health')"

CMD ["python", "-m", "uvicorn", "src.api:app", "--host", "0.0.0.0", "--port", "8000"]
