from transformers import BartForConditionalGeneration, BartTokenizer
from rouge_score import rouge_scorer
import torch
import pandas as pd
from torch.utils.data import DataLoader

# Load model and tokenizer from HuggingFace
model_name = "chinhon/bart-large-cnn-summarizer_03"
tokenizer = BartTokenizer.from_pretrained(model_name)
model = BartForConditionalGeneration.from_pretrained(model_name)
device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
model.to(device)

def evaluate_huggingface_model(model, tokenizer, val_df):
    model.eval()
    rouge = rouge_scorer.RougeScorer(['rouge1', 'rouge2', 'rougeL'], use_stemmer=True)
    all_hypotheses = []
    all_references = []

    with torch.no_grad():
        for _, row in val_df.iterrows():
            inputs = tokenizer(row['article'], max_length=1024, truncation=True, 
                             padding='max_length', return_tensors='pt')
            input_ids = inputs['input_ids'].to(device)
            attention_mask = inputs['attention_mask'].to(device)

            summary_ids = model.generate(input_ids=input_ids, 
                                       attention_mask=attention_mask,
                                       max_length=150,
                                       min_length=40,
                                       length_penalty=2.0,
                                       num_beams=4,
                                       early_stopping=True)
            
            decoded_summary = tokenizer.decode(summary_ids[0], skip_special_tokens=True)
            all_hypotheses.append(decoded_summary)
            all_references.append(row['summary'])

    rouge_scores = {key: 0 for key in ['rouge1', 'rouge2', 'rougeL']}
    for hyp, ref in zip(all_hypotheses, all_references):
        scores = rouge.score(hyp, ref)
        print(scores)
        for key in scores:
            rouge_scores[key] += scores[key].fmeasure

    for key in rouge_scores:
        rouge_scores[key] /= len(all_hypotheses)

    print(f"ROUGE Scores: {rouge_scores}")
    return rouge_scores

# Load your validation data
df = pd.read_csv('preprocessed_data.csv')
val_df = df.sample(n=100, random_state=42)  # Using 100 samples for evaluation

# Run evaluation
rouge_scores = evaluate_huggingface_model(model, tokenizer, val_df)