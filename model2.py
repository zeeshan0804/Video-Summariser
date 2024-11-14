import torch
from torch.utils.data import Dataset, DataLoader
from transformers import BartForConditionalGeneration, BartTokenizer, AdamW
from rouge_score import rouge_scorer
from transformers import get_linear_schedule_with_warmup

class BartSummarizer:
    def __init__(self, model_name='facebook/bart-base'):
        self.tokenizer = BartTokenizer.from_pretrained(model_name)
        self.model = BartForConditionalGeneration.from_pretrained(model_name)
        self.device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
        self.model.to(self.device)

    def summarize(self, text, max_length=150):
        inputs = self.tokenizer.encode("summarize: " + text, return_tensors="pt", max_length=512, truncation=True).to(self.device)
        summary_ids = self.model.generate(
            inputs,
            max_length=max_length,
            min_length=30,
            length_penalty=2.0,
            num_beams=4,
            early_stopping=True
        )
        summary = self.tokenizer.decode(summary_ids[0], skip_special_tokens=True)
        return summary

    def compute_rouge_loss(self, predictions, references):
        scorer = rouge_scorer.RougeScorer(['rouge1', 'rougeL'], use_stemmer=True)
        rouge_scores = scorer.score(predictions, references)
        rouge_loss = 1 - (rouge_scores['rouge1'].fmeasure + rouge_scores['rougeL'].fmeasure) / 2
        return rouge_loss

    def fine_tune(self, train_dataset, val_dataset, epochs=3, batch_size=8, learning_rate=5e-5):
        train_loader = DataLoader(train_dataset, batch_size=batch_size, shuffle=True)
        val_loader = DataLoader(val_dataset, batch_size=batch_size)

        optimizer = AdamW(self.model.parameters(), lr=learning_rate)
        total_steps = len(train_loader) * epochs
        scheduler = get_linear_schedule_with_warmup(optimizer, num_warmup_steps=0, num_training_steps=total_steps)
        self.model.train()

        for epoch in range(epochs):
            total_loss = 0
            for batch in train_loader:
                optimizer.zero_grad()
                input_ids = batch['input_ids'].to(self.device)
                attention_mask = batch['attention_mask'].to(self.device)
                labels = batch['labels'].to(self.device)

                outputs = self.model(input_ids=input_ids, attention_mask=attention_mask, labels=labels)
                logits = outputs.logits
                predictions = self.tokenizer.batch_decode(logits.argmax(dim=-1), skip_special_tokens=True)
                references = self.tokenizer.batch_decode(labels, skip_special_tokens=True)

                rouge_loss = sum(self.compute_rouge_loss(pred, ref) for pred, ref in zip(predictions, references)) / len(predictions)
                total_loss += rouge_loss.item()

                rouge_loss.backward()
                optimizer.step()
                scheduler.step()

            avg_train_loss = total_loss / len(train_loader)
            print(f"Epoch {epoch + 1}, Training Loss: {avg_train_loss}")

            self.evaluate(val_loader)
            self.save_model(epoch + 1)

    def evaluate(self, val_loader):
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
                logits = outputs.logits
                predictions = self.tokenizer.batch_decode(logits.argmax(dim=-1), skip_special_tokens=True)
                references = self.tokenizer.batch_decode(labels, skip_special_tokens=True)

                rouge_loss = sum(self.compute_rouge_loss(pred, ref) for pred, ref in zip(predictions, references)) / len(predictions)
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

    def save_model(self, epoch):
        model_save_path = f"bart_model_epoch_{epoch}.pt"
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