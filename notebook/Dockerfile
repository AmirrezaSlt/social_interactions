FROM tensorflow/tensorflow:1.13.2-py3-jupyter
COPY ./requirements.txt /
RUN pip install -r /requirements.txt && \ 
    rm -rf /tf/*
WORKDIR /tf