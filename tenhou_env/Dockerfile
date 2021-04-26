FROM rayproject/ray-ml:latest-cpu

WORKDIR /app/

COPY requirements.txt /requirements.txt
COPY config.yaml /config.yaml

RUN pip install --no-cache-dir -r /requirements.txt
RUN pip install google-api-python-client==1.7.8

COPY trainer /trainer
COPY tenhou_env/project .
COPY models /models
RUN export PYTHONPATH=PYTHONPATH:/

USER root