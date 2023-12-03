FROM python:3.10
ENV PYTHONUNBUFFERED 1
COPY requirements.txt .
WORKDIR /app
COPY . /app/
RUN pip install -r requirements.txt
EXPOSE 5000
CMD ["python3", "run_servers.py"]
