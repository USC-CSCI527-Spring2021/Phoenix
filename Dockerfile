FROM apache/beam_python3.7_sdk:2.28.0

COPY requirements.txt /requirements.txt
RUN pip install --no-cache-dir -r /requirements.txt
COPY ./extract_features .
COPY ./logs_parser .
COPY ./pipeline.py /pipeline.py
