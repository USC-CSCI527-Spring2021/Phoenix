FROM tensorflow/tensorflow:2.4.1

RUN useradd -ms /bin/bash docker-user

WORKDIR /app/

COPY tenhou_env/requirements /requirements
RUN pip install --no-cache-dir -r /requirements/dev.txt

COPY trainer /trainer
COPY tenhou_env/project .

USER docker-user