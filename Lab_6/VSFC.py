import numpy as np
import torch
import streamlit as st
from PIL import Image
from transformers import AutoTokenizer, AutoModelForSequenceClassification, Trainer
from vncorenlp import VnCoreNLP

vncorenlp = VnCoreNLP("vncorenlp/VnCoreNLP-1.1.1.jar", annotators="wseg", max_heap_size='-Xmx500m') 

phobert = AutoModelForSequenceClassification.from_pretrained("model/UIT-VSFC", num_labels = 3)
tokenizer_phobert = AutoTokenizer.from_pretrained("vinai/phobert-base",use_fast=False)

class BuildDataset(torch.utils.data.Dataset):
    def __init__(self, encodings, labels):
        self.encodings = encodings
        self.labels = labels

    def __getitem__(self, idx):
        item = {key: torch.tensor(val[idx]) for key, val in self.encodings.items()}
        item['labels'] = torch.tensor(self.labels[idx])
        return item

    def __len__(self):
        return len(self.labels)
    
def preprocess(text, tokenized=True, lowercased=False):
    text = text.lower() if lowercased else text
    if tokenized:
        pre_text = ""
        sentences = vncorenlp.tokenize(text)
        for sentence in sentences:
            pre_text += " ".join(sentence)
        text = pre_text
    return text

def sigmoid_array(x): 
    return 1 / (1 + np.exp(-x))

def predict(text, lowercased=False, tokenized=False):
    labels = {
        0: "Tiêu Cực",
        1: "Trung Tính",
        2: "Tích Cực"
    }
    p_text = preprocess(text, lowercased=lowercased, tokenized=tokenized)

    text = tokenizer_phobert([p_text], truncation=True, padding=True, max_length=100)
    text = BuildDataset(text, [0])
    y_pred = Trainer(model=phobert).predict(text).predictions
    y_pred = sigmoid_array(y_pred)

    return labels[np.argmax(y_pred, axis=-1)[0]]

def VSFC():
    st.title("Vietnamese Students Feedback")
    sentence = st.text_input('Enter Students Feedback:')
    
    if st.button('Predict'):
        labels = predict(sentence)
        st.write('Predicted:', labels)
        if(labels=="Tích Cực"):
            image = Image.open('data/images/positive.png')
        elif(labels=="Tiêu Cực"):
            image = Image.open('data/images/negative.png')
        else:
            image = Image.open('data/images/neutral.png')
        st.image(image, caption=labels)