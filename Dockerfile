FROM python:3.12

WORKDIR /app

COPY app/requirements.txt .

RUN pip install -r requirements.txt

COPY app/app.py .
COPY app/baner.png .

EXPOSE 8503

CMD ["streamlit", "run", "app.py"]