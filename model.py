import torch
from torch.utils.data import Dataset, DataLoader
from transformers import T5ForConditionalGeneration, T5Tokenizer, AdamW
from rouge_score import rouge_scorer


class TextSummarizer:
    def __init__(self, model_name='t5-small'):
        self.tokenizer = T5Tokenizer.from_pretrained(model_name)
        self.model = T5ForConditionalGeneration.from_pretrained(model_name)
        self.device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
        self.model.to(self.device)

    def summarize(self, text, max_length=150):
        inputs = self.tokenizer.encode("summarize: " + text, return_tensors="pt", max_length=512, truncation=True).to(self.device)
        summary_ids = self.model.generate(inputs, max_length=max_length, min_length=30, length_penalty=2.0, num_beams=4, early_stopping=True)
        summary = self.tokenizer.decode(summary_ids[0], skip_special_tokens=True)
        return summary

    def fine_tune(self, train_dataset, val_dataset, epochs=3, batch_size=8, learning_rate=5e-5):
        train_loader = DataLoader(train_dataset, batch_size=batch_size, shuffle=True)
        val_loader = DataLoader(val_dataset, batch_size=batch_size)

        optimizer = AdamW(self.model.parameters(), lr=learning_rate)
        self.model.train()

        for epoch in range(epochs):
            total_loss = 0
            for batch in train_loader:
                optimizer.zero_grad()
                input_ids = batch['input_ids'].to(self.device)
                attention_mask = batch['attention_mask'].to(self.device)
                labels = batch['labels'].to(self.device)

                outputs = self.model(input_ids=input_ids, attention_mask=attention_mask, labels=labels)
                loss = outputs.loss
                total_loss += loss.item()

                loss.backward()
                optimizer.step()

            avg_train_loss = total_loss / len(train_loader)
            print(f"Epoch {epoch + 1}, Training Loss: {avg_train_loss}")

            self.evaluate(val_loader)
            self.save_model(epoch + 1)

    def evaluate(self,val_loader):
        self.model.eval()
        total_loss = 0
        rouge = rouge_scorer.RougeScorer(['rouge1', 'rougeL'], use_stemmer=True)
        all_hypotheses = []
        all_references = []

        with torch.no_grad():
            for batch in val_loader:
                input_ids = batch['input_ids'].to(self.device)
                attention_mask = batch['attention_mask'].to(self.device)
                labels = batch['labels'].to(self.device)

                outputs = self.model(input_ids=input_ids, attention_mask=attention_mask, labels=labels)
                loss = outputs.loss
                total_loss += loss.item()

                # Generate summaries
                summaries = self.model.generate(input_ids=input_ids, attention_mask=attention_mask)
                decoded_summaries = [self.tokenizer.decode(s, skip_special_tokens=True) for s in summaries]
                decoded_labels = [self.tokenizer.decode(l, skip_special_tokens=True) for l in labels]

                all_hypotheses.extend(decoded_summaries)
                all_references.extend(decoded_labels)

        avg_val_loss = total_loss / len(val_loader)
        rouge_scores = rouge.get_scores(all_hypotheses, all_references, avg=True)
        print(f"Validation Loss: {avg_val_loss}")
        print(f"ROUGE Scores: {rouge_scores}")

    def save_model(self, epoch):
        model_save_path = f"t5_summarizer_epoch_{epoch}.pt"
        torch.save(self.model.state_dict(), model_save_path)
        print(f"Model saved to {model_save_path}")

class SummarizationDataset(Dataset):
    def __init__(self, dataframe, tokenizer, max_length=512):
        self.dataframe = dataframe
        self.tokenizer = tokenizer
        self.max_length = max_length

    def __len__(self):
        return len(self.dataframe)

    def __getitem__(self, idx):
        article = self.dataframe.iloc[idx]['article']
        summary = self.dataframe.iloc[idx]['summary']

        inputs = self.tokenizer.encode_plus(
            "summarize: " + article,
            max_length=self.max_length,
            padding='max_length',
            truncation=True,
            return_tensors="pt"
        )

        labels = self.tokenizer.encode_plus(
            summary,
            max_length=self.max_length,
            padding='max_length',
            truncation=True,
            return_tensors="pt"
        )

        return {
            'input_ids': inputs['input_ids'].flatten(),
            'attention_mask': inputs['attention_mask'].flatten(),
            'labels': labels['input_ids'].flatten()
        }

# # Example usage
# if __name__ == "__main__":
#     import pandas as pd

#     # Load preprocessed data
#     df = pd.read_csv('preprocessed_data.csv')

#     # Split data into training and validation sets
#     train_df = df.sample(frac=0.8, random_state=42)
#     val_df = df.drop(train_df.index)

#     # Create datasets
#     tokenizer = T5Tokenizer.from_pretrained('t5-small')
#     train_dataset = SummarizationDataset(train_df, tokenizer)
#     val_dataset = SummarizationDataset(val_df, tokenizer)

#     # Fine-tune the model
#     summarizer = TextSummarizer()
#     summarizer.fine_tune(train_dataset, val_dataset)
