import pickle
import streamlit as st
import tensorflow as tf
from transformers import BertTokenizer, TFBertForSequenceClassification
import zipfile
import shutil
import os

# Завантаження моделі з директорії
model_dir = 'model_save'

# Завантаження токенізатора
tokenizer = BertTokenizer.from_pretrained(model_dir)

# Завантаження моделі
model = TFBertForSequenceClassification.from_pretrained(model_dir)

model.classifier.activation = tf.keras.activations.sigmoid
def predict_comment(comment, model, tokenizer):
    encoding = tokenizer.encode_plus(
        comment,
        add_special_tokens=True,
        max_length=512,
        return_token_type_ids=False,
        padding='max_length',
        return_attention_mask=True,
        return_tensors='tf',
        truncation=True
    )

    input_ids = encoding['input_ids']
    attention_mask = encoding['attention_mask']

    preds = model.predict({'input_ids': input_ids, 'attention_mask': attention_mask})[0][0]

    return {
        'toxic': preds[0],
        'severe_toxic': preds[1],
        'obscene': preds[2],
        'threat': preds[3],
        'insult': preds[4],
        'identity_hate': preds[5]
    }

comment = "You're so stupid and useless. No one wants you around here."
text_class = predict_comment(comment, model, tokenizer)
print(text_class)


# streamlit app
st.title("Text Classification with BERT")

user_input = st.text_area("Enter your text here:")

if st.button("Classify"):
    if user_input:
        prediction = predict_comment(user_input, model, tokenizer)
        st.write("Classification Results:")
        for label, score in prediction.items():
            st.write(f"{label}: {score:.4f}")
    else:
        st.write("Please enter text to classify.")

