
import torch
from transformers import BertTokenizer, BertForSequenceClassification
import streamlit as st
import matplotlib.pyplot as plt
import numpy as np

MAX_LEN = 128
CLASS_NAMES = ['toxic', 'severe_toxic', 'obscene', 'threat', 'insult', 'identity_hate']

model = BertForSequenceClassification.from_pretrained('bert-base-uncased', num_labels=6)
model.load_state_dict(torch.load('model/bert_toxic_model.pth', map_location=torch.device('cpu')))
model.eval()

tokenizer = BertTokenizer.from_pretrained('bert-base-uncased')

st.title('Toxicity classification model')
st.markdown("## Final project")
st.markdown("Classification of toxic comments using BERT")
st.markdown("performers:")
st.markdown("- Serhii Klymenko")
st.markdown("- Roman Sydorenko")
st.markdown("- Pavlo Taradayka")
st.markdown('<hr>', unsafe_allow_html=True)

input_text = st.text_area("Enter text for classification")

if st.button('Classify'):
    inputs = tokenizer(input_text, return_tensors='pt', truncation=True, padding=True, max_length=MAX_LEN)
    with torch.no_grad():
        outputs = model(**inputs)
    predictions = outputs.logits.sigmoid().cpu().numpy()[0]

    sorted_predictions = sorted(((label, f"{prob*100:.1f}%") for label, prob in zip(CLASS_NAMES, predictions) if prob > 0.10), key=lambda x: float(x[1][:-1]))

    st.write('Predictions:')
    for label, prob in sorted_predictions:
        st.write(f"{label}: {prob}")

    if sorted_predictions:
        labels, values = zip(*[(label, float(prob[:-1])) for label, prob in sorted_predictions])
        fig, ax = plt.subplots()
        ax.barh(list(range(len(labels))), list(values), color='skyblue')
        ax.set_yticks(list(range(len(labels))))
        ax.set_yticklabels(labels)
        ax.set_xlabel('Prediction probability')
        ax.set_title('Toxicity class predictions')
        st.pyplot(fig)
    else:
        st.write("No predictions exceed 10%")


