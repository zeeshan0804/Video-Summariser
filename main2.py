import os
import pandas as pd
import torch
from torch.utils.data import DataLoader
from transformers import AdamW
from rouge_score import rouge_scorer
from transformers import BartForConditionalGeneration, BartTokenizer, AdamW
from model2 import BartSummarizer, SummarizationDataset

def train(model, train_loader, optimizer):
    model.model.train()
    total_loss = 0
    for batch in train_loader:
        optimizer.zero_grad()
        input_ids = batch['input_ids'].to(model.device)
        attention_mask = batch['attention_mask'].to(model.device)
        labels = batch['labels'].to(model.device)

        outputs = model.model(input_ids=input_ids, attention_mask=attention_mask, labels=labels)
        logits = outputs.logits
        predictions = model.tokenizer.batch_decode(logits.argmax(dim=-1), skip_special_tokens=True)
        references = model.tokenizer.batch_decode(labels, skip_special_tokens=True)

        rouge_loss = sum(model.compute_rouge_loss(pred, ref) for pred, ref in zip(predictions, references)) / len(predictions)
        total_loss += rouge_loss.item()

        rouge_loss.backward()
        optimizer.step()

    avg_train_loss = total_loss / len(train_loader)
    print(f"Training Loss: {avg_train_loss}")

def evaluate(model, val_loader):
    model.model.eval()
    total_loss = 0
    rouge = rouge_scorer.RougeScorer(['rouge1', 'rougeL'], use_stemmer=True)
    all_hypotheses = []
    all_references = []

    with torch.no_grad():
        for batch in val_loader:
            input_ids = batch['input_ids'].to(model.device)
            attention_mask = batch['attention_mask'].to(model.device)
            labels = batch['labels'].to(model.device)

            outputs = model.model(input_ids=input_ids, attention_mask=attention_mask, labels=labels)
            logits = outputs.logits
            predictions = model.tokenizer.batch_decode(logits.argmax(dim=-1), skip_special_tokens=True)
            references = model.tokenizer.batch_decode(labels, skip_special_tokens=True)

            rouge_loss = sum(model.compute_rouge_loss(pred, ref) for pred, ref in zip(predictions, references)) / len(predictions)
            total_loss += rouge_loss.item()

            all_hypotheses.extend(predictions)
            all_references.extend(references)

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

def load_model(model_path, model_name='facebook/bart-base'):
    tokenizer = BartTokenizer.from_pretrained(model_name)
    model = BartForConditionalGeneration.from_pretrained(model_name)
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    model.load_state_dict(torch.load(model_path, map_location=device))
    model.to(device)
    model.eval()
    return model, tokenizer, device

if __name__ == "__main__":
    # Load preprocessed data
    df = pd.read_csv('preprocessed_data.csv')

    # Use only 20% of the dataset for fine-tuning
    sample_df = df.sample(frac=0.4, random_state=42)

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
    summarizer = BartSummarizer()
    train_dataset = SummarizationDataset(train_df, summarizer.tokenizer)
    val_dataset = SummarizationDataset(val_df, summarizer.tokenizer)
    
    train_loader = DataLoader(train_dataset, batch_size=8, shuffle=True)
    val_loader = DataLoader(val_dataset, batch_size=8)

    model_path = 'bart_model_epoch_15.pt'
    if os.path.exists(model_path):
        print(f"Model file {model_path} found. Loading and evaluating the model.")
        summarizer.model.load_state_dict(torch.load(model_path, map_location=summarizer.device))
        evaluate(summarizer, val_loader)
    else:
        print(f"Model file {model_path} not found. Training the model.")
        summarizer.fine_tune(train_dataset, val_dataset, epochs=15, batch_size=8, learning_rate=2e-5)
        evaluate(summarizer, val_loader)