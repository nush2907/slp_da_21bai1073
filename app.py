import streamlit as st
import torch
from transformers import BertTokenizer, BertForSequenceClassification

# Load the model and tokenizer
# Load the model and tokenizer
model = BertForSequenceClassification.from_pretrained("anushkam2003/my_model")
tokenizer = BertTokenizer.from_pretrained("anushkam2003/my_model")

# Set the app title
st.title("Question Duplication Predictor")

# User input
question1 = st.text_input("Enter Question 1:")
question2 = st.text_input("Enter Question 2:")

if st.button("Predict"):
    if question1 and question2:
        # Prepare the input for the model
        inputs = tokenizer.encode_plus(
            question1,
            question2,
            add_special_tokens=True,
            return_tensors='pt',
            padding='max_length',
            max_length=512,
            truncation=True
        )

        input_ids = inputs['input_ids']
        attention_mask = inputs['attention_mask']

        # Make prediction
        with torch.no_grad():
            outputs = model(input_ids, attention_mask=attention_mask)
            logits = outputs.logits
            prediction = torch.argmax(logits, dim=1).item()

        # Display result
        if prediction == 1:
            st.success("The questions are likely duplicates.")
        else:
            st.success("The questions are not duplicates.")
    else:
        st.error("Please enter both questions.")
