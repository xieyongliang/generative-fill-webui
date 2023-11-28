FROM python:3.10-slim

RUN mkdir -p /opt/ml/code
COPY . /opt/ml/code
RUN pip install -r /opt/ml/code/requirements.txt

WORKDIR /opt/ml/code

ENTRYPOINT ["python3", "ui.py"]