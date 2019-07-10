FROM tensorflow/tensorflow:latest-py3
COPY . .
RUN pip install -r ./requirements.txt
RUN apt-get -y install git
WORKDIR /main
# CMD CMD tail -f /dev/null