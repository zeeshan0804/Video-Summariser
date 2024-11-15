import os
import pandas as pd
import torch.nn as nn
import torch
from torch.utils.data import DataLoader
from transformers import AdamW
from rouge_score import rouge_scorer
from model import TextSummarizer, SummarizationDataset
from model3 import EnhancedBartSummarizer, TextSummarizer
from transformers import BartForConditionalGeneration, BartTokenizer
import matplotlib.pyplot as plt
import argparse

def train(model, train_loader, optimizer):
    model.train()
    total_loss = 0
    for batch in train_loader:
        optimizer.zero_grad()
        input_ids = batch['input_ids'].to(model.device)
        attention_mask = batch['attention_mask'].to(model.device)
        labels = batch['labels'].to(model.device)

        logits = model(input_ids=input_ids, attention_mask=attention_mask, labels=labels)
        loss = nn.CrossEntropyLoss()(logits.view(-1, logits.size(-1)), labels.view(-1))
        total_loss += loss.item()

        loss.backward()
        optimizer.step()

    avg_train_loss = total_loss / len(train_loader)
    print(f"Training Loss: {avg_train_loss}")
    return avg_train_loss

def evaluate(model, val_loader):
    model.eval()
    total_loss = 0
    rouge = rouge_scorer.RougeScorer(['rouge1', 'rougeL'], use_stemmer=True)
    all_hypotheses = []
    all_references = []

    with torch.no_grad():
        for batch in val_loader:
            input_ids = batch['input_ids'].to(model.device)
            attention_mask = batch['attention_mask'].to(model.device)
            labels = batch['labels'].to(model.device)

            logits = model(input_ids=input_ids, attention_mask=attention_mask, labels=labels)
            loss = nn.CrossEntropyLoss()(logits.view(-1, logits.size(-1)), labels.view(-1))
            total_loss += loss.item()

            summaries = model.generate(input_ids=input_ids, attention_mask=attention_mask)
            decoded_summaries = [model.tokenizer.decode(s, skip_special_tokens=True) for s in summaries]
            decoded_labels = [model.tokenizer.decode(l, skip_special_tokens=True) for l in labels]

            all_hypotheses.extend(decoded_summaries)
            all_references.extend(decoded_labels)

    avg_val_loss = total_loss / len(val_loader)
    rouge_scores = {key: 0 for key in rouge.score(all_hypotheses[0], all_references[0]).keys()}
    for hyp, ref in zip(all_hypotheses, all_references):
        scores = rouge.score(hyp, ref)
        for key in scores:
            rouge_scores[key] += scores[key].fmeasure

    for key in rouge_scores:
        rouge_scores[key] /= len(all_hypotheses)

    print(f"Validation Loss: {avg_val_loss}")
    print(f"ROUGE Scores: {rouge_scores}")
    return avg_val_loss, rouge_scores

def load_model(model_path, model_type='bart'):
    if model_type == 'bart':
        tokenizer = BartTokenizer.from_pretrained('facebook/bart-base')
        model = EnhancedBartSummarizer('facebook/bart-base')
    else:
        raise ValueError("Unsupported model type. Use 'bart'")

    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    model.load_state_dict(torch.load(model_path, map_location=device))
    model.to(device)
    model.eval()
    return model, tokenizer, device

def generate_summary(model, tokenizer, device, text, model_type='bart', max_length=150):
    prefix = "summarize: "
    
    inputs = tokenizer.encode(prefix + text, return_tensors="pt", max_length=512, truncation=True).to(device)
    summary_ids = model.generate(inputs, max_length=max_length, min_length=30, length_penalty=2.0, num_beams=4, early_stopping=True)
    summary = tokenizer.decode(summary_ids[0], skip_special_tokens=True)
    return summary

if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument('--model_type', type=str, choices=['bart'], default='bart')
    args = parser.parse_args()

    # Load preprocessed data
    df = pd.read_csv('preprocessed_data.csv')

    # Use only 20% of the dataset for fine-tuning
    sample_df = df.sample(frac=0.5, random_state=42)

    # Print the size of the dataset
    print(f"Total dataset size: {len(df)}")
    print(f"Sampled dataset size: {len(sample_df)}")

    # Split data into training and validation sets
    train_df = sample_df.sample(frac=0.9, random_state=42)
    val_df = sample_df.drop(train_df.index)

    # Print the size of the training and validation sets
    print(f"Training set size: {len(train_df)}")
    print(f"Validation set size: {len(val_df)}")

    # Create datasets
    summarizer = TextSummarizer(model_name='facebook/bart-base')
    train_dataset = SummarizationDataset(train_df, summarizer.tokenizer)
    val_dataset = SummarizationDataset(val_df, summarizer.tokenizer)
    
    train_loader = DataLoader(train_dataset, batch_size=8, shuffle=True)
    val_loader = DataLoader(val_dataset, batch_size=8)

    model_path = 'enhanced_bart_model_epoch_15.pt'
    if os.path.exists(model_path):
        print(f"Model file {model_path} found. Loading and evaluating the model.")
        summarizer.model.load_state_dict(torch.load(model_path, map_location=summarizer.device))
        evaluate(summarizer.model, val_loader)
    else:
        print(f"Model file {model_path} not found. Training the model.")
        summarizer.fine_tune(train_dataset, val_dataset, epochs=15, batch_size=8, learning_rate=1e-5)