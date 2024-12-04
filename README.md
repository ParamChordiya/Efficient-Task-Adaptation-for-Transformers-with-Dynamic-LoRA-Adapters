# Project: Efficient Task Adaptation for Transformers with Dynamic LoRA-Adapters

This project implements and compares baseline, LoRA, and LoRA + Dynamic Adapter models for sequence-to-sequence learning tasks using the CNN/DailyMail dataset. Below are the instructions to preprocess data, train models, evaluate performance, analyze errors, and launch the Streamlit app for results visualization.

---

## Prerequisites

Ensure you have the following installed:
- Python 3.7 or higher
- Required Python packages (install via `requirements.txt` if available):
  ```bash
  pip install -r requirements.txt
  ```

## Step-by-Step Instructions
1. Data Preprocessing
Preprocess the dataset to prepare it for model training:
```bash
python data_preprocessing.py --dataset_name cnn_dailymail --tokenizer_name t5-base
```

2. Model Training
Train the models using the preprocessed dataset:

Baseline Model:
```bash
python model_training.py --model_name t5-small --tokenizer_name t5-small
```

LoRA Model:
```bash
python model_training.py --model_name t5-small --tokenizer_name t5-small --use_lora
```

LoRA + Dynamic Adapter Model:
```bash
python model_training.py --model_name t5-small --tokenizer_name t5-small --use_lora --use_adapter
```

3. Evaluation
Evaluate each model on the validation dataset:
Baseline Model:
```bash
python evaluation.py --model_dir ./saved_model_baseline --tokenizer_name t5-small --output_file evaluation_baseline.txt
```
LoRA Model:
```bash
python evaluation.py --model_dir ./saved_model_lora --tokenizer_name t5-small --output_file evaluation_lora.txt
```
LoRA + Dynamic Adapter Model:
```bash
python evaluation.py --model_dir ./saved_model_lora_adapter --tokenizer_name t5-small --output_file evaluation_lora_adapter.txt
```

4. Error Analysis
Perform error analysis on the LoRA + Dynamic Adapter model:
```bash
python error_analysis.py --model_dir ./saved_model_lora_adapter --tokenizer_name t5-small --output_file error_analysis_results.txt
```

5. Launch Streamlit App
Visualize and compare results using the Streamlit app:
```bash
streamlit run app.py
```