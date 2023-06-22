FROM python:3.8.10

ENV LANG C.UTF-8
COPY requirements.txt /tmp/requirements.txt
RUN pip install -U pip
RUN pip install -r /tmp/requirements.txt

COPY servers/mlflow/before-run /app/before-run
COPY servers/mlflow/image_metadata.json /app/image_metadata.json
COPY servers/mlflow/MLFlowServer.py /app/MLFlowServer.py
COPY servers/mlflow/pip_env_create.py /app/pip_env_create.py

RUN chmod +x /app/before-run
RUN chmod +x /app/pip_env_create.py
RUN chmod +r /app/MLFlowServer.py

WORKDIR /app

EXPOSE 5000
EXPOSE 9000

ENV MODEL_NAME MLFlowServer
ENV SERVICE_TYPE MODEL
ENV PERSISTENCE 0

ENV PYTHONPATH /app:${PYTHONPATH}

CMD bash before-run && \
    exec seldon-core-microservice ${MODEL_NAME} --service-type ${SERVICE_TYPE} --persistence ${PERSISTENCE}
