import streamlit as st
import torch
from transformers import AutoTokenizer, AutoModelForSequenceClassification, pipeline

# Load models and tokenizer
from transformers import AutoTokenizer, AutoModel
tokenizer = AutoTokenizer.from_pretrained("emilyalsentzer/Bio_ClinicalBERT")
model = AutoModel.from_pretrained("emilyalsentzer/Bio_ClinicalBERT")


explainer = pipeline("text-generation", model="gpt2")

# Define functions
def analyze_symptoms(symptoms):
    inputs = tokenizer(symptoms, return_tensors="pt")
    outputs = model(**inputs)
    predictions = torch.argmax(outputs.logits, dim=-1)
    diagnosis = "Possible diagnosis based on symptoms."
    return diagnosis

def explain_diagnosis(diagnosis):
    explanation = explainer(f"Explain in simple terms: {diagnosis}")[0]['generated_text']
    return explanation

# Streamlit App Interface
st.title("Healthcare AI Assistant")
st.write("Enter your symptoms, and the assistant will suggest a possible diagnosis.")

symptoms = st.text_area("Enter symptoms:")

if st.button("Analyze Symptoms"):
    diagnosis = analyze_symptoms(symptoms)
    st.write("Suggested Diagnosis:", diagnosis)
    explanation = explain_diagnosis(diagnosis)
    st.write("Explanation:", explanation)
