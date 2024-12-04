import streamlit as st
import numpy as np

def load_metrics(file_path):
    with open(file_path, 'r') as f:
        return eval(f.read())

def load_error_analysis(file_path):
    with open(file_path, 'r') as f:
        return f.read()

st.title('Model Comparison: Baseline vs. LoRA vs. LoRA + Adapter')

# Load training and evaluation metrics
baseline_training = load_metrics('training_metrics_baseline.txt')
lora_training = load_metrics('training_metrics_lora.txt')
lora_adapter_training = load_metrics('training_metrics_lora_adapter.txt')

baseline_eval = load_metrics('evaluation_baseline.txt')
lora_eval = load_metrics('evaluation_lora.txt')
lora_adapter_eval = load_metrics('evaluation_lora_adapter.txt')

# Load error analysis results
error_analysis_results = load_error_analysis('error_analysis_results.txt')

# Display Training Metrics
st.header('Training Metrics')
st.write("**Baseline Training:**", baseline_training)
st.write("**LoRA Training:**", lora_training)
st.write("**LoRA + Adapter Training:**", lora_adapter_training)

# Display Evaluation Metrics
st.header('Evaluation Metrics')
st.write("**Baseline Evaluation (ROUGE):**", baseline_eval)
st.write("**LoRA Evaluation (ROUGE):**", lora_eval)
st.write("**LoRA + Adapter Evaluation (ROUGE):**", lora_adapter_eval)

# Display Error Analysis
st.header('Error Analysis Results')
st.text_area("Detailed Error Analysis", error_analysis_results, height=500)
