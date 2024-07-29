# Toxic comment classification with BERT

## Project description

This project aims to develop a model capable of identifying and classifying various levels of toxicity in comments. Utilizing the capabilities of BERT (Bidirectional Encoder Representations from Transformers) for text analysis, we create a classifier that helps moderators effectively detect and handle toxic comments, improving the quality of online communities.

## Technical requirements

- **Programming language:** Python
- **Libraries:**
  - PyTorch for deep learning
  - Transformers for using the BERT model

## Methodology

### Data preprocessing:

- Clean text from unnecessary symbols.
- Convert text into a format suitable for BERT processing.

### Using BERT:

- Apply a pre-trained BERT model to obtain contextualized word embeddings for deeper understanding of the context and meaning of words in comments.

### Classification:

- Create a classification model based on BERT embeddings to identify and classify text toxicity by labels.

## Goals

- **Primary goal:** Develop a model for effective classification of toxic comments, capable of determining the degree and type of toxicity.
- **Moderation improvement:** Provide a tool to simplify moderation and create a healthier environment for online dialogues.

## Expected outcomes

- **Classifier:** A model that can accurately distinguish between toxic and non-toxic comments, as well as identify different levels and types of toxicity.
- **Interface:** Develop a user-friendly interface for easy access to the model's functionality.

## Project structure
```bash
.
├── app
│ ├── app.py
│ ├── baner.png
│ ├── requirements.txt
├── model
│ ├── bert_toxic_model.pth
├── data
│ ├── train.csv
├── train.ipynb
├── Dockerfile
├── docker-compose.yml
└── README.md
```


## Installation and running

### 1. Clone the repository

```bash
git clone https://github.com/yourusername/toxic-comment-classification.git
cd toxic-comment-classification
```
### 2. Prepare the dataset
Download the dataset from the [Jigsaw Toxic Comment Classification Challenge](https://www.kaggle.com/c/jigsaw-toxic-comment-classification-challenge/data) and place it in the data directory:

data/train.csv

### 3. Train the model
Before running the web application, you need to train the model. Use train.ipynb to create and save the model in the model folder.

### 4. Build the docker container
```bash
docker build --no-cache -t fp_web .
```
### 5. Run the docker container
```bash
docker-compose up
```
### 6. Access the web application
After starting the container, the web application will be available at: http://localhost:8501

## Configuring traefik for production environment
To use with your own domain, edit docker-compose.yml by adding the appropriate settings for Traefik:
```bash
services:
  web:
    image: fp_web
    restart: always
    ports:
      - "8501:8501"
    volumes:
      - ./model:/app/model
    environment:
      - MODEL_PATH=/app/model/bert_toxic_model.pth
    labels:
      - "traefik.enable=true"
      - "traefik.http.routers.web.rule=Host(`yourdomain.com`)"
      - "traefik.http.services.web.loadbalancer.server.port=8501"
    networks:
      - app-network

networks:
  app-network:
    external: true

```
Dockerfile
```bash
FROM python:3.12

WORKDIR /app

COPY app/requirements.txt .

RUN pip install -r requirements.txt

COPY app/app.py .
COPY app/baner.png .

EXPOSE 8501

CMD ["streamlit", "run", "app.py"]
```
## Contributing
Please follow the standard process for creating pull requests if you want to contribute to this project. First, open an Issue to discuss the changes you wish to make. Then fork the repository, create a branch for your changes, commit, and open a pull request.
