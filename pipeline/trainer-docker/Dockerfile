FROM python:3.8.10-slim

COPY . /app
WORKDIR /app

RUN pip install -U pip
RUN pip install -r requirements.txt

ENTRYPOINT ["python", "trainer.py"]