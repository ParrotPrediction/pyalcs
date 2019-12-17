FROM python:3.7

RUN mkdir /code
COPY . /code
WORKDIR /code

RUN pip install -r requirements.txt

CMD ["make", "test"]
