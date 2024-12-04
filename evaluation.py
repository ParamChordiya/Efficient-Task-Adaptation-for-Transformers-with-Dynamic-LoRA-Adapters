import argparse
import evaluate
from transformers import AutoTokenizer, AutoModelForSeq2SeqLM
from datasets import load_dataset
import torch

def generate_summary(batch, model, tokenizer, max_input_length=128, max_target_length=64):
    inputs = tokenizer(
        batch['article'],
        return_tensors='pt',
        truncation=True,
        padding="max_length",
        max_length=max_input_length
    ).to(model.device)

    outputs = model.generate(
        inputs['input_ids'],
        max_length=max_target_length,
        num_beams=4,
        early_stopping=True
    )

    batch['predicted_summary'] = tokenizer.batch_decode(outputs, skip_special_tokens=True)
    return batch

def evaluate_model(model_dir, dataset_name, tokenizer_name, dataset_config_name):
    model = AutoModelForSeq2SeqLM.from_pretrained(model_dir)
    tokenizer = AutoTokenizer.from_pretrained(tokenizer_name)

    dataset = load_dataset(dataset_name, dataset_config_name)
    tokenized_dataset = dataset['validation'].select(range(200))  # Smaller subset for evaluation

    model.to('cuda' if torch.cuda.is_available() else 'cpu')

    results = tokenized_dataset.map(
        lambda batch: generate_summary(batch, model, tokenizer),
        batched=True,
        batch_size=4
    )

    rouge = evaluate.load('rouge')

    scores = rouge.compute(
        predictions=results['predicted_summary'],
        references=results['highlights']
    )

    return scores

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description='Evaluation Script')
    parser.add_argument('--model_dir', type=str, required=True, help='Directory of the saved model')
    parser.add_argument('--dataset_name', type=str, default='cnn_dailymail', help='Dataset name')
    parser.add_argument('--tokenizer_name', type=str, default='t5-small', help='Tokenizer name')
    parser.add_argument('--dataset_config_name', type=str, default='3.0.0', help='Dataset configuration name')
    parser.add_argument('--output_file', type=str, required=True, help='File to save evaluation metrics')
    args = parser.parse_args()

    rouge_scores = evaluate_model(args.model_dir, args.dataset_name, args.tokenizer_name, args.dataset_config_name)

    with open(args.output_file, 'w') as f:
        f.write(str(rouge_scores))
