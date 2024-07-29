
import torch
from transformers import BertTokenizer, BertForSequenceClassification
import streamlit as st
import matplotlib.pyplot as plt
import numpy as np
import base64

MAX_LEN = 128
CLASS_NAMES = ['toxic', 'severe_toxic', 'obscene', 'threat', 'insult', 'identity_hate']

model = BertForSequenceClassification.from_pretrained('bert-base-uncased', num_labels=6)
model.load_state_dict(torch.load('model/bert_toxic_model.pth', map_location=torch.device('cpu')))
model.eval()

tokenizer = BertTokenizer.from_pretrained('bert-base-uncased')

def load_image(image_file):
    with open(image_file, "rb") as f:
        return base64.b64encode(f.read()).decode()
    
    
background_image_base64 = load_image('baner.png')

st.set_page_config(layout="wide", initial_sidebar_state="expanded")

# def set_dark_theme():
#     st.markdown(
#         """
#         <style>
#         [data-testid="stAppViewContainer"] {
#             background-color: #000000;
#             color: #e0e0e0;
#         }
#         [data-testid="stSidebar"] {
#             background-color: #1e1e1e;
#         }
#         [data-testid="stMarkdownContainer"],
#         [data-testid="stTextArea"],
#         [data-testid="stTextInput"] {
#             color: #e0e0e0;
#         }
#         [data-testid="stButton"] > button {
#             background-color: #4CAF50;
#             color: #fff;
#         }
#         [data-testid="stImage"] {
#             background-color: #121212;
#         }
#         [data-testid="stMarkdownBlock"] {
#             color: #e0e0e0;
#         }
#         </style>
#         """,
#         unsafe_allow_html=True,
#     )

# set_dark_theme()

# st.markdown(
#     f"""
#     <style>
#     .full-width-image {{
#         background-image: url('data:image/png;base64,{background_image_base64}');
#         background-size: cover;
#         background-position: center;
#         height: 250px;
#         width: 100%;
#         border-bottom: 1px solid #e0e0e0;
#         overflow: hidden;
#     }}
#     .spacer {{
#         height: 40px;
#     }}
#     </style>
#     <div class="full-width-image"></div>
#     <div class="spacer"></div>
#     """,
#     unsafe_allow_html=True
# )

def set_dark_theme():
    st.markdown(
        """
        <style>
        [data-testid="stAppViewContainer"] {
            background-color: #000000;
            color: #e0e0e0;
        }
        [data-testid="stSidebar"] {
            background-color: #1e1e1e;
        }
        [data-testid="stMarkdownContainer"],
        [data-testid="stTextArea"],
        [data-testid="stTextInput"] {
            color: #e0e0e0;
        }
        [data-testid="stButton"] > button {
            background-color: #4CAF50;
            color: #fff;
        }
        [data-testid="stImage"] {
            background-color: #121212;
        }
        [data-testid="stMarkdownBlock"] {
            color: #e0e0e0;
        }
        @media (max-width: 600px) {
            .full-width-image {
                height: auto;
                max-height: 150px;
            }
            .spacer {
                height: 20px;
            }
            [data-testid="stTextArea"] {
                height: 200px;
            }
            [data-testid="stButton"] > button {
                width: 100%;
            }
        }
        </style>
        """,
        unsafe_allow_html=True,
    )

set_dark_theme()

st.markdown(
    f"""
    <style>
    .full-width-image {{
        background-image: url('data:image/png;base64,{background_image_base64}');
        background-size: cover;
        background-position: center;
        height: 250px;
        width: 100%;
        border-bottom: 1px solid #e0e0e0;
        overflow: hidden;
    }}
    .spacer {{
        height: 40px;
    }}
    @media (max-width: 600px) {{
        .full-width-image {{
            height: auto;
            max-height: 150px;
        }}
        .spacer {{
            height: 20px;
        }}
    }}
    </style>
    <div class="full-width-image"></div>
    <div class="spacer"></div>
    """,
    unsafe_allow_html=True
)



container = st.container()
col1, col2 = container.columns([1, 1])

with col1:
    input_text = st.text_area("Enter text for classification", height=300)

with col2:
    st.write("Result")
    placeholder = st.empty()
    with placeholder.container():
        st.markdown(
            """
            <div style="background-color: white; height: 300px; display: flex; align-items: center; justify-content: center; border: 1px solid #e0e0e0; margin-top: -14px;">
                <p style="color: gray;">Waiting for input...</p>
            </div>
            """,
            unsafe_allow_html=True,
        )

if st.button('Classify'):
    inputs = tokenizer(input_text, return_tensors='pt', truncation=True, padding=True, max_length=MAX_LEN)
    with torch.no_grad():
        outputs = model(**inputs)
    predictions = outputs.logits.sigmoid().cpu().numpy()[0]

    sorted_predictions = sorted(((label, f"{prob*100:.1f}%") for label, prob in zip(CLASS_NAMES, predictions) if prob > 0.10), key=lambda x: float(x[1][:-1]))

    with col2:
        if sorted_predictions:
            labels, values = zip(*[(label, float(prob[:-1])) for label, prob in sorted_predictions])
            fig, ax = plt.subplots(figsize=(5, 0.8))  # Adjust the size of the figure here

            num_bars = len(labels)
            bar_height = 0.3  # Adjust height of bars to be smaller
            bar_spacing = 0.3  # Adjust spacing between bars

            bars = ax.barh(list(range(num_bars)), list(values), color='skyblue', height=bar_height)

            ax.set_yticks(list(range(num_bars)))
            ax.set_yticklabels(labels, fontsize=8)  # Adjust font size for labels
            ax.set_xlabel('Prediction Probability', fontsize=8)  # Adjust font size for x-axis label
            ax.set_title('Toxicity Class Predictions', fontsize=10)  # Adjust font size for title

            ax.spines['top'].set_visible(False)
            ax.spines['right'].set_visible(False)

            for bar, value in zip(bars, values):
                width = bar.get_width()
                ax.text(width, bar.get_y() + bar.get_height() / 2, f'{value:.1f}%', 
                        va='center', ha='left', color='black', fontsize=8)  # Adjust font size for text on bars

            ax.tick_params(axis='x', labelsize=8)
            fig.subplots_adjust(left=0.25, right=0.75, top=0.85, bottom=0.2)

            placeholder.empty()  # Clear the placeholder
            placeholder.pyplot(fig)
        else:
            placeholder.empty()  # Clear the placeholder
            placeholder.write("No predictions exceed 10%")