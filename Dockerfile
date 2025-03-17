FROM python:3.11.7

WORKDIR /app

COPY . /app

RUN pip3 install -r requirements.txt

CMD ["uvicorn","deployment.api:app","--host=0.0.0.0", "--port=7860"]