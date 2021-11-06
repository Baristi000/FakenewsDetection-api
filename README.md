# Fake news detection api

## Requirement:

- Docker OR Python > 3.x

## Setup api:

- RUN:

```bash
git clone https://github.com/Baristi000/FakenewsDetection-api.git && \
    cd FakenewsDetection-api && \
    pip install -r requirements/r.txt
```

## Run api:

```bash
python main.py
```

## By docker:

Enable gpu:

```bash
docker run  --gpus all -p 8001:8001 baristi000/fake_new_detection_api:0.1.0
```

No gpu:

```bash
docker run -p 8001:8001 baristi000/fake_new_detection_api:0.1.0
```
