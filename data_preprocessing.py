import argparse
from datasets import load_dataset
from transformers import AutoTokenizer

def preprocess_data(dataset_name, tokenizer_name, dataset_config_name=None, max_input_length=128, max_target_length=64):
    if dataset_config_name:
        dataset = load_dataset(dataset_name, dataset_config_name)
    else:
        dataset = load_dataset(dataset_name)
    tokenizer = AutoTokenizer.from_pretrained(tokenizer_name)

    def preprocess_function(examples):
        inputs = [doc for doc in examples['article']]
        model_inputs = tokenizer(
            inputs,
            max_length=max_input_length,
            truncation=True,
            padding="max_length"
        )

        labels = tokenizer(
            examples['highlights'],
            max_length=max_target_length,
            truncation=True,
            padding="max_length"
        )

        model_inputs['labels'] = labels['input_ids']
        model_inputs['labels'] = [
            [(label if label != tokenizer.pad_token_id else -100) for label in labels_example]
            for labels_example in model_inputs['labels']
        ]

        return model_inputs

    tokenized_datasets = dataset.map(
        preprocess_function,
        batched=True,
        remove_columns=dataset['train'].column_names
    )
    return tokenized_datasets, tokenizer

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description='Data Preprocessing Script')
    parser.add_argument('--dataset_name', type=str, default='cnn_dailymail', help='Name of the dataset')
    parser.add_argument('--tokenizer_name', type=str, default='t5-base', help='Tokenizer name')
    parser.add_argument('--dataset_config_name', type=str, default='3.0.0', help='Dataset configuration name')
    args = parser.parse_args()

    tokenized_datasets, tokenizer = preprocess_data(
        args.dataset_name,
        args.tokenizer_name,
        dataset_config_name=args.dataset_config_name
    )