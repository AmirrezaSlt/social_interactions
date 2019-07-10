FROM tensorflow/tensorflow:latest-py3
WORKDIR /app
COPY . .
RUN pip install -r ./requirements.txt
RUN apt-get -y install git
# CMD CMD tail -f /dev/null