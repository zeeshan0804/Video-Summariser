import torch
from transformers import T5Tokenizer, T5ForConditionalGeneration
import argparse

def load_model(model_path, model_name='t5-small'):
    tokenizer = T5Tokenizer.from_pretrained(model_name)
    model = T5ForConditionalGeneration.from_pretrained(model_name)
    model.load_state_dict(torch.load(model_path))
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    model.to(device)
    model.eval()
    return model, tokenizer, device

def generate_summary(model, tokenizer, device, text, max_length=150):
    inputs = tokenizer.encode("summarize: " + text, return_tensors="pt", max_length=512, truncation=True).to(device)
    summary_ids = model.generate(inputs, max_length=max_length, min_length=30, length_penalty=2.0, num_beams=4, early_stopping=True)
    summary = tokenizer.decode(summary_ids[0], skip_special_tokens=True)
    return summary

def main(input_file, model_path, output_file):
    model, tokenizer, device = load_model(model_path)
    
    with open(input_file, 'r') as file:
        text = file.read()

    summary = generate_summary(model, tokenizer, device, text)

    with open(output_file, 'w') as file:
        file.write(summary)

    print(f"Summary written to {output_file}")

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Generate summary from a text file using a fine-tuned T5 model.")
    parser.add_argument('--input_file', type=str, required=True, help="Path to the input text file.")
    parser.add_argument('--model_path', type=str, required=True, help="Path to the fine-tuned model file.")
    parser.add_argument('--output_file', type=str, required=True, help="Path to save the generated summary.")
    
    args = parser.parse_args()
    main(args.input_file, args.model_path, args.output_file)
