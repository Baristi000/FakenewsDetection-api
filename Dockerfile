FROM python:3.8

COPY . /.

WORKDIR /

RUN pip install -r r.txt &&\
    mkdir ai_services/checkpoints

CMD ["python", "main.py"]