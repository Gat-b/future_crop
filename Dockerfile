FROM python:3.10.6-buster


COPY requirements_api.txt requirements.txt

RUN pip install --upgrade pip
RUN pip install -r requirements.txt

COPY future_crop future_crop
COPY setup.py setup.py


RUN pip install .

COPY yield_forecasts yield_forecasts

ENV PLATFORM=docker

CMD uvicorn future_crop.api.api:app --host 127.0.0.1 --port $PORT