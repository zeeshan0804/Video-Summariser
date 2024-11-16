from transformers import AutoModelForSeq2SeqLM, AutoTokenizer
from rouge_score import rouge_scorer
import torch
import pandas as pd
from torch.utils.data import DataLoader
from model import SummarizationDataset

def test_flan_model(model_name, test_loader):
    """
    Function to evaluate Flan-T5 model on test data using ROUGE scores.
    
    Args:
        model_name (str): Hugging Face model name or path for Flan-T5 model.
        test_loader (DataLoader): DataLoader for test dataset.

    Returns:
        dict: ROUGE scores (ROUGE-1, ROUGE-L).
    """
    # Load model and tokenizer
    tokenizer = AutoTokenizer.from_pretrained(model_name)
    model = AutoModelForSeq2SeqLM.from_pretrained(model_name)
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    model.to(device)
    model.eval()

    # Initialize ROUGE scorer
    rouge = rouge_scorer.RougeScorer(['rouge1', 'rougeL'], use_stemmer=True)

    # Lists to store hypotheses and references
    all_hypotheses = []
    all_references = []

    # Evaluate model
    with torch.no_grad():
        for batch in test_loader:
            input_ids = batch['input_ids'].to(model.device)
            attention_mask = batch['attention_mask'].to(model.device)
            labels = batch['labels'].to(model.device)

            # # Tokenize inputs and move to device
            # inputs = tokenizer(input_texts, return_tensors="pt", padding=True, truncation=True, max_length=512).to(device)

            # # # Generate summaries
            # summaries = model.generate(
            #     inputs.input_ids, 
            #     max_length=150, 
            #     num_beams=4, 
            #     early_stopping=True
            # )

            # Decode generated summaries
            # decoded_summaries = [tokenizer.decode(s, skip_special_tokens=True) for s in summaries]
            summaries = model.generate(input_ids=input_ids, attention_mask=attention_mask)
            decoded_summaries = [tokenizer.decode(s, skip_special_tokens=True) for s in summaries]
            decoded_labels = [tokenizer.decode(l, skip_special_tokens=True) for l in labels]

            # Append results
            all_hypotheses.extend(decoded_summaries)
            all_references.extend(decoded_labels)

    # Calculate ROUGE scores
    rouge_scores = {key: 0 for key in rouge.score(all_hypotheses[0], all_references[0]).keys()}
    for hyp, ref in zip(all_hypotheses, all_references):
        scores = rouge.score(hyp, ref)
        for key in scores:
            rouge_scores[key] += scores[key].fmeasure

    for key in rouge_scores:
        rouge_scores[key] /= len(all_hypotheses)

    print(f"ROUGE Scores: {rouge_scores}")
    return rouge_scores

if __name__ == "__main__":
    # Load preprocessed dataset
    df = pd.read_csv('preprocessed_data.csv')

    # Use the full dataset
    sample_df = df.sample(frac=1, random_state=42)

    # Split data into training and validation sets
    train_df = sample_df.sample(frac=0.9, random_state=42)
    val_df = sample_df.drop(train_df.index)

    print(f"Training set size: {len(train_df)}")
    print(f"Validation set size: {len(val_df)}")

    # Create datasets
    model_name = "jordiclive/flan-t5-3b-summarizer"
    tokenizer = AutoTokenizer.from_pretrained(model_name)
    train_dataset = SummarizationDataset(train_df, tokenizer)
    val_dataset = SummarizationDataset(val_df, tokenizer)

    train_loader = DataLoader(train_dataset, batch_size=8, shuffle=True)
    val_loader = DataLoader(val_dataset, batch_size=8)

    # Test the model
    rouge_scores = test_flan_model(model_name, val_loader)
    print(rouge_scores)
