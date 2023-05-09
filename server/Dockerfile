FROM python:3

WORKDIR /backend

COPY ./requirements.txt .

RUN pip install --upgrade pip

RUN pip install -r requirements.txt

COPY . .

EXPOSE $SERVER_PORT

CMD [ "gunicorn", "-b", "0.0.0.0:$SERVER_PORT", "app:app"]