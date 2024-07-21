FROM python:3.12

WORKDIR /app

COPY app/requirements.txt .

RUN pip install -r requirements.txt

COPY app/app.py .

EXPOSE 8501

CMD ["streamlit", "run", "app.py"]