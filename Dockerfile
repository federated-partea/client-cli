FROM python

MAINTAINER Julian Späth

RUN apt-get update
RUN apt-get upgrade -y
RUN apt-get install ffmpeg libsm6 libxext6  -y
RUN pip3 install --upgrade pip

ADD requirements.txt /requirements.txt
RUN pip3 install -r requirements.txt

ADD partea /partea
ADD pet /pet
ADD main.py /main.py
ADD serialize.py /serialize.py

ENTRYPOINT ["python3", "-u", "./main.py"]