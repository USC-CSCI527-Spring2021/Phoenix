FROM rayproject/ray-ml:latest-cpu

WORKDIR /home/ray/
RUN git clone https://github.com/USC-CSCI527-Spring2021/Phoenix.git && cd Phoenix && git fetch && git checkout experiment_RL
#
#COPY requirements.txt ./requirements.txt
#COPY config.yaml ./config.yaml
#COPY .gitignore ./.gitignore
#COPY .git/ ./.git
#
WORKDIR /home/ray/Phoenix
RUN pip install --no-cache-dir -r ./requirements.txt
RUN pip install google-api-python-client==1.7.8
#
#COPY extract_features ./extract_features
#COPY logs_parser ./logs_parser
#COPY trainer ./trainer
#COPY tenhou_env/project ./tenhou_env/project
COPY models ./models
RUN export PYTHONPATH=PYTHONPATH:/home/ray/Phoenix

USER ray