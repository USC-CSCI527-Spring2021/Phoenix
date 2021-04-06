FROM tensorflow/tensorflow:2.4.1

WORKDIR /
COPY extract_features /extract_features
COPY logs_parser /logs_parser
COPY trainer /trainer
COPY models /models
COPY tenhou_env /tenhou_env
COPY requirements.txt /requirements.txt
COPY setup.py /setup.py
RUN pip install --no-cache-dir -r /requirements.txt

ENTRYPOINT ["python","-m", "trainer.task"]
