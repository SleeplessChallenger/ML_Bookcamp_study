FROM python:3

ENV PYTHONUNBUFFERED=TRUE

WORKDIR /app
EXPOSE 8085

ADD . .

RUN pip install -r requirements.txt

COPY . .

ENTRYPOINT ["gunicorn", "--bind", "0.0.0.0:8085", "app:app"]
