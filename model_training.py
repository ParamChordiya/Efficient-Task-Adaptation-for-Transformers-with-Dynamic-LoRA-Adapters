import argparse
import time
import torch
from torch import nn
from transformers import AutoModelForSeq2SeqLM, Trainer, TrainingArguments
from peft import LoraConfig, get_peft_model
from data_preprocessing import preprocess_data

class DynamicAdapter(nn.Module):
    def __init__(self, embedding_dim, hidden_dim=64):
        super(DynamicAdapter, self).__init__()
        self.adapter_fc = nn.Linear(embedding_dim, hidden_dim)
        self.activation = nn.ReLU()
        self.output_fc = nn.Linear(hidden_dim, embedding_dim)

    def forward(self, inputs):
        adapted = self.activation(self.adapter_fc(inputs))
        outputs = self.output_fc(adapted)
        return outputs + inputs  

def train_model(model_name, tokenizer_name, train_dataset, eval_dataset, training_args, use_lora=False, use_adapter=False):
    model = AutoModelForSeq2SeqLM.from_pretrained(model_name)

    if use_lora:
        lora_config = LoraConfig(
            r=8,
            lora_alpha=16,
            target_modules=['q', 'v'],
            lora_dropout=0.1,
            bias='none'
        )
        model = get_peft_model(model, lora_config)

    if use_adapter:
        embedding_dim = model.config.d_model
        adapter_layer = DynamicAdapter(embedding_dim)
        model.encoder.add_module("dynamic_adapter", adapter_layer)

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model.to(device)

    trainer = Trainer(
        model=model,
        args=training_args,
        train_dataset=train_dataset,
        eval_dataset=eval_dataset
    )

    start_time = time.time()
    trainer.train()
    end_time = time.time()

    training_time = end_time - start_time
    max_gpu_memory = torch.cuda.max_memory_allocated() / 1e6  # in MB

    metrics = {
        'training_time': training_time,
        'max_gpu_memory': max_gpu_memory
    }

    return model, metrics

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description='Model Training Script')
    parser.add_argument('--model_name', type=str, default='t5-small', help='Model name')
    parser.add_argument('--tokenizer_name', type=str, default='t5-small', help='Tokenizer name')
    parser.add_argument('--dataset_name', type=str, default='cnn_dailymail', help='Dataset name')
    parser.add_argument('--dataset_config_name', type=str, default='3.0.0', help='Dataset configuration name')
    parser.add_argument('--use_lora', action='store_true', help='Use LoRA for fine-tuning')
    parser.add_argument('--use_adapter', action='store_true', help='Use Dynamic Adapter for fine-tuning')
    args = parser.parse_args()

    tokenized_datasets, tokenizer = preprocess_data(
        args.dataset_name,
        args.tokenizer_name,
        dataset_config_name=args.dataset_config_name,
        max_input_length=128,
        max_target_length=64
    )

    small_train_dataset = tokenized_datasets['train'].shuffle(seed=42).select(range(1000))
    small_eval_dataset = tokenized_datasets['validation'].shuffle(seed=42).select(range(200))

    training_args = TrainingArguments(
        output_dir='./results',
        evaluation_strategy='steps',
        eval_steps=100,
        learning_rate=5e-5,
        per_device_train_batch_size=2,
        per_device_eval_batch_size=2,
        num_train_epochs=5,
        weight_decay=0.01,
        save_total_limit=1,
        fp16=torch.cuda.is_available(),
        logging_dir='./logs',
        logging_steps=10,
        gradient_accumulation_steps=2
    )

    model, metrics = train_model(
        args.model_name,
        args.tokenizer_name,
        small_train_dataset,
        small_eval_dataset,
        training_args,
        use_lora=args.use_lora,
        use_adapter=args.use_adapter
    )

    # Save the model
    if args.use_lora and args.use_adapter:
        save_dir = './saved_model_lora_adapter'
        metrics_file = 'training_metrics_lora_adapter.txt'
    elif args.use_lora:
        save_dir = './saved_model_lora'
        metrics_file = 'training_metrics_lora.txt'
    else:
        save_dir = './saved_model_baseline'
        metrics_file = 'training_metrics_baseline.txt'

    model.save_pretrained(save_dir)
    tokenizer.save_pretrained(save_dir)

    with open(metrics_file, 'w') as f:
        f.write(str(metrics))
