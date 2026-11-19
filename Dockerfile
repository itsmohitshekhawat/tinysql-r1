# TinySQL-R1 demo: small CPU-only image. Training does not happen here (see train/); this serves the page,
# runs SQL on the read-only sample databases and calls the frontier model over an API.
FROM python:3.12-slim
ENV PYTHONDONTWRITEBYTECODE=1 PYTHONUNBUFFERED=1 PIP_NO_CACHE_DIR=1
WORKDIR /app
COPY requirements.txt .
RUN pip install -r requirements.txt
COPY app ./app
COPY tinysql/__init__.py tinysql/sqlenv.py ./tinysql/
COPY data ./data
RUN useradd -r -u 10001 tinysql && mkdir -p /var/lib/tinysql && chown tinysql /var/lib/tinysql && chmod -R a+rX /app
USER tinysql
ENV BUDGET_DB=/var/lib/tinysql/budget.sqlite
EXPOSE 8000
HEALTHCHECK --interval=60s --timeout=5s CMD python -c "import urllib.request;urllib.request.urlopen('http://127.0.0.1:8000/health')"
CMD ["uvicorn", "app.main:app", "--host", "0.0.0.0", "--port", "8000", "--proxy-headers", "--forwarded-allow-ips", "*"]
