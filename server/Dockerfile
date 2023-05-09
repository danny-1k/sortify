FROM python:3.9.6

ENV SERVER_PORT 3000

WORKDIR /backend

RUN apt-get -y update && apt-get -y upgrade && apt-get install -y ffmpeg

COPY ./requirements.txt .

RUN pip install --upgrade pip

RUN pip install -r requirements.txt

COPY . .

EXPOSE $SERVER_PORT

CMD gunicorn -b 0.0.0.0:$SERVER_PORT app:app
