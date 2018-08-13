FROM python:3.7

RUN apt-get update && apt-get install -y swig cmake

RUN mkdir /code
COPY . /code
WORKDIR /code

RUN pip install -r requirements.txt --ignore-installed six

CMD ["make", "test"]
