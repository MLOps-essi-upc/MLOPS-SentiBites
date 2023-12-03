from subprocess import Popen

# Start uvicorn
uvicorn_cmd = ["uvicorn", "--host", "0.0.0.0", "--port", "8000", "src.app.api:app"]
uvicorn_process = Popen(uvicorn_cmd)

# Start gunicorn
gunicorn_cmd = ["gunicorn", "-b", "0.0.0.0:5000", "src.app.app:app"]
gunicorn_process = Popen(gunicorn_cmd)

# Wait for the processes to finish
uvicorn_process.wait()
gunicorn_process.wait()
