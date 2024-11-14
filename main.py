import os
import pandas as pd
import torch
from torch.utils.data import DataLoader
from transformers import AdamW
from rouge_score import rouge_scorer
from transformers import T5ForConditionalGeneration, T5Tokenizer, AdamW
from model import TextSummarizer, SummarizationDataset

def train(model, train_loader, optimizer):
    model.encoder.train()
    model.decoder.train()
    total_loss = 0
    for batch in train_loader:
        optimizer.zero_grad()
        input_ids = batch['input_ids'].to(model.device)
        attention_mask = batch['attention_mask'].to(model.device)
        labels = batch['labels'].to(model.device)

        encoder_outputs = model.encoder.encoder(input_ids=input_ids, attention_mask=attention_mask)
        decoder_outputs = model.decoder(
            input_ids=labels,
            encoder_outputs=encoder_outputs,
            labels=labels
        )
        loss = decoder_outputs.loss
        total_loss += loss.item()

        loss.backward()
        optimizer.step()

    avg_train_loss = total_loss / len(train_loader)
    print(f"Training Loss: {avg_train_loss}")

def evaluate(model, val_loader):
    model.encoder.eval()
    model.decoder.eval()
    total_loss = 0
    rouge = rouge_scorer.RougeScorer(['rouge1', 'rougeL'], use_stemmer=True)
    all_hypotheses = []
    all_references = []

    with torch.no_grad():
        for batch in val_loader:
            input_ids = batch['input_ids'].to(model.device)
            attention_mask = batch['attention_mask'].to(model.device)
            labels = batch['labels'].to(model.device)

            encoder_outputs = model.encoder.encoder(input_ids=input_ids, attention_mask=attention_mask)
            decoder_outputs = model.decoder(
                input_ids=labels,
                encoder_outputs=encoder_outputs,
                labels=labels
            )
            loss = decoder_outputs.loss
            total_loss += loss.item()

            summaries = model.decoder.generate(input_ids=labels, encoder_outputs=encoder_outputs)
            decoded_summaries = [model.tokenizer.decode(s, skip_special_tokens=True) for s in summaries]
            decoded_labels = [model.tokenizer.decode(l, skip_special_tokens=True) for l in labels]

            all_hypotheses.extend(decoded_summaries)
            all_references.extend(decoded_labels)

    avg_val_loss = total_loss / len(val_loader)
    rouge_scores = {key: 0 for key in rouge.score(decoded_summaries[0], decoded_labels[0]).keys()}
    for hyp, ref in zip(all_hypotheses, all_references):
        scores = rouge.score(hyp, ref)
        for key in scores:
            rouge_scores[key] += scores[key].fmeasure

    for key in rouge_scores:
        rouge_scores[key] /= len(all_hypotheses)

    print(f"Validation Loss: {avg_val_loss}")
    print(f"ROUGE Scores: {rouge_scores}")

def load_model(encoder_path, decoder_path, model_name='t5-small'):
    tokenizer = T5Tokenizer.from_pretrained(model_name)
    encoder = T5ForConditionalGeneration.from_pretrained(model_name)
    decoder = T5ForConditionalGeneration.from_pretrained(model_name)
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    encoder.load_state_dict(torch.load(encoder_path, map_location=device))
    decoder.load_state_dict(torch.load(decoder_path, map_location=device))
    encoder.to(device)
    decoder.to(device)
    encoder.eval()
    decoder.eval()
    return encoder, decoder, tokenizer, device

if __name__ == "__main__":
    # Load preprocessed data
    df = pd.read_csv('preprocessed_data.csv')

    # Use only 20% of the dataset for fine-tuning
    sample_df = df.sample(frac=0.2, random_state=42)

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
    summarizer = TextSummarizer()
    train_dataset = SummarizationDataset(train_df, summarizer.tokenizer)
    val_dataset = SummarizationDataset(val_df, summarizer.tokenizer)
    
    train_loader = DataLoader(train_dataset, batch_size=8, shuffle=True)
    val_loader = DataLoader(val_dataset, batch_size=8)

    encoder_path = 'encoder_epoch_15.pt'
    decoder_path = 'decoder_epoch_15.pt'
    if os.path.exists(encoder_path) and os.path.exists(decoder_path):
        print(f"Model files {encoder_path} and {decoder_path} found. Loading and evaluating the models.")
        summarizer.encoder.load_state_dict(torch.load(encoder_path, map_location=summarizer.device))
        summarizer.decoder.load_state_dict(torch.load(decoder_path, map_location=summarizer.device))
        evaluate(summarizer, val_loader)
    else:
        print(f"Model files {encoder_path} and {decoder_path} not found. Training the models.")
        summarizer.fine_tune(train_dataset, val_dataset, epochs=15, batch_size=8, learning_rate=1e-5)
        evaluate(summarizer, val_loader)