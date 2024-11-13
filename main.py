import pandas as pd
import torch
from torch.utils.data import DataLoader
from transformers import AdamW
from rouge_score import rouge_scorer
from model import TextSummarizer, SummarizationDataset

def train(model, train_loader, optimizer):
    model.model.train()
    total_loss = 0
    for batch in train_loader:
        optimizer.zero_grad()
        input_ids = batch['input_ids'].to(model.device)
        attention_mask = batch['attention_mask'].to(model.device)
        labels = batch['labels'].to(model.device)

        outputs = model.model(input_ids=input_ids, attention_mask=attention_mask, labels=labels)
        loss = outputs.loss
        total_loss += loss.item()

        loss.backward()
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
            loss = outputs.loss
            total_loss += loss.item()

            # Generate summaries
            summaries = model.model.generate(input_ids=input_ids, attention_mask=attention_mask)
            decoded_summaries = [model.tokenizer.decode(s, skip_special_tokens=True) for s in summaries]
            decoded_labels = [model.tokenizer.decode(l, skip_special_tokens=True) for l in labels]

            all_hypotheses.extend(decoded_summaries)
            all_references.extend(decoded_labels)

    avg_val_loss = total_loss / len(val_loader)
    rouge_scores = rouge.get_scores(all_hypotheses, all_references, avg=True)
    print(f"Validation Loss: {avg_val_loss}")
    print(f"ROUGE Scores: {rouge_scores}")

if __name__ == "__main__":
    # Load preprocessed data
    df = pd.read_csv('preprocessed_data.csv')

    # Split data into training and validation sets
    train_df = df.sample(frac=0.8, random_state=42)
    val_df = df.drop(train_df.index)

    # Create datasets
    summarizer = TextSummarizer()
    train_dataset = SummarizationDataset(train_df, summarizer.tokenizer)
    val_dataset = SummarizationDataset(val_df, summarizer.tokenizer)

    # Create data loaders
    train_loader = DataLoader(train_dataset, batch_size=8, shuffle=True)
    val_loader = DataLoader(val_dataset, batch_size=8)

    # Fine-tune the model
    optimizer = AdamW(summarizer.model.parameters(), lr=5e-5)
    epochs = 3

    for epoch in range(epochs):
        print(f"Epoch {epoch + 1}/{epochs}")
        # evaluate(summarizer, val_loader)
        train(summarizer, train_loader, optimizer)
        evaluate(summarizer, val_loader)
