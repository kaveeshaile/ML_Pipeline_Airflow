FROM apache/airflow:2.8.1
COPY requiement.txt .
RUN pip install -r requiement.txt