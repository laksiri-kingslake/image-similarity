FROM python:3.11-slim
# FROM python:3.8-bullseye

WORKDIR /app

COPY requirements.txt requirements.txt 

RUN python3 -m pip install -U --user -r requirements.txt 

COPY . .

EXPOSE 6010

CMD ["python", "app-ml.py", "input.png", "images" ]