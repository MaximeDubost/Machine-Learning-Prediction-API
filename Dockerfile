FROM tiangolo/uvicorn-gunicorn-fastapi:python3.7

COPY . /app

RUN pip3 install -r requirements.txt

CMD ["gunicorn", "-w 4","-k uvicorn.workers.UvicornWorker", "main:app"]