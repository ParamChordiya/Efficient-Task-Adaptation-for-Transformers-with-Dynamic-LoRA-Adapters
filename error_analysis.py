import torch
from transformers import AutoModelForSeq2SeqLM, AutoTokenizer

def load_model_and_tokenizer(model_dir, tokenizer_name):
    model = AutoModelForSeq2SeqLM.from_pretrained(model_dir)
    tokenizer = AutoTokenizer.from_pretrained(tokenizer_name)
    model.to('cuda' if torch.cuda.is_available() else 'cpu')
    return model, tokenizer

def generate_summary(input_text, model, tokenizer, max_input_length=512, max_target_length=128):
    inputs = tokenizer(
        input_text, 
        return_tensors="pt", 
        truncation=True, 
        padding="max_length", 
        max_length=max_input_length
    ).to(model.device)

    outputs = model.generate(
        inputs["input_ids"],
        max_length=max_target_length,
        num_beams=4,
        early_stopping=True
    )

    return tokenizer.decode(outputs[0], skip_special_tokens=True)

def test_failure_patterns(model, tokenizer):
    test_cases = {
        "Overly Long Articles": {
            "input": " ".join(["This is a very long article." for _ in range(200)]),
            "description": "Test model's handling of overly long articles exceeding token limits."
        },
        "Ambiguous Contexts": {
            "input": "The project was started in 1999. It was successful. Meanwhile, other projects failed. The main goal was education.",
            "description": "Test model's ability to identify the main theme among multiple topics."
        },
        "Complex Syntax": {
            "input": "Despite the report being filed late, which was unexpected considering the strict deadlines, the conclusion was accepted without hesitation by all members.",
            "description": "Test model's handling of sentences with complex structures."
        },
        "Out-of-Domain Examples": {
            "input": "The Higgs boson particle, crucial for understanding the mass of subatomic particles, was discovered at CERN using the LHC in 2012.",
            "description": "Test model's ability to summarize niche scientific content."
        },
        "Contextual Misalignment": {
            "input": "The financial crisis of 2008, also known as the Great Recession, and the crisis of 1929 share many similarities. However, the responses differed significantly.",
            "description": "Test model's ability to focus on relevant context when there is lexical overlap."
        },
        "Ambiguous Pronoun References": {
            "input": "John told Peter he needed to finish the report before the deadline. He agreed and started working immediately.",
            "description": "Test model's handling of ambiguous pronouns."
        },
        "Extreme Length Variability": {
            "input": "The cat sat on the mat.",
            "description": "Test model's handling of very short articles."
        },
        "Shifting Focal Points": {
            "input": "The politician started by discussing healthcare policies but quickly moved to personal anecdotes about his childhood.",
            "description": "Test model's ability to summarize articles with shifting focal points."
        },
        "Rare Vocabulary": {
            "input": "The article delved into the intricacies of chromatophore functionality in cephalopods, highlighting their ability to change color via neuromuscular activation.",
            "description": "Test model's ability to handle domain-specific vocabulary."
        },
        "Indirect Speech": {
            "input": "The professor stated that the theory was groundbreaking. However, she warned that further research was necessary.",
            "description": "Test model's handling of indirect speech."
        },
    }

    results = {}
    for case_name, case_data in test_cases.items():
        print(f"Testing: {case_name}")
        summary = generate_summary(case_data["input"], model, tokenizer)
        results[case_name] = {
            "description": case_data["description"],
            "input": case_data["input"],
            "summary": summary
        }
        print(f"Input: {case_data['input'][:100]}...")
        print(f"Summary: {summary}\n")
    
    return results


def save_results(results, output_file):
    with open(output_file, "w") as f:
        for case_name, case_data in results.items():
            f.write(f"=== {case_name} ===\n")
            f.write(f"Description: {case_data['description']}\n")
            f.write(f"Input: {case_data['input']}\n")
            f.write(f"Summary: {case_data['summary']}\n\n")

if __name__ == "__main__":
    import argparse

    parser = argparse.ArgumentParser(description="Error Analysis Script")
    parser.add_argument("--model_dir", type=str, required=True, help="Directory of the saved model")
    parser.add_argument("--tokenizer_name", type=str, required=True, help="Tokenizer name")
    parser.add_argument("--output_file", type=str, required=True, help="File to save error analysis results")
    args = parser.parse_args()

    model, tokenizer = load_model_and_tokenizer(args.model_dir, args.tokenizer_name)
    results = test_failure_patterns(model, tokenizer)
    save_results(results, args.output_file)
    print(f"Error analysis results saved to {args.output_file}")
