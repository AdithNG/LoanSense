# LoanSense: Streamlit UI (train from UI or mount models/)
FROM python:3.11-slim

WORKDIR /app

COPY requirements.txt .
RUN pip install --no-cache-dir -r requirements.txt

COPY app.py .
COPY src/ src/
COPY scripts/ scripts/

# Default: run Streamlit. Override to run API: CMD ["uvicorn", "src.api.main:app", "--host", "0.0.0.0", "--port", "8000"]
EXPOSE 8501
CMD ["streamlit", "run", "app.py", "--server.port=8501", "--server.address=0.0.0.0"]
