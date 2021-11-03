FROM python:3.8

COPY . /.

WORKDIR /

RUN pip install -r r.txt --upgrade pip &&\
    mkdir ai_services/checkpoints

CMD ["python", "main.py"]