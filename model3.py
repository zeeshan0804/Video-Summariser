import torch
import torch.nn as nn
from torch.utils.data import Dataset, DataLoader
from transformers import BartForConditionalGeneration, BartTokenizer, AdamW
from rouge_score import rouge_scorer
from transformers import get_linear_schedule_with_warmup
import matplotlib.pyplot as plt

class EnhancedBartSummarizer(nn.Module):
    def __init__(self, model_name='facebook/bart-base'):
        super(EnhancedBartSummarizer, self).__init__()
        self.bart = BartForConditionalGeneration.from_pretrained(model_name)
        self.cnn = nn.Conv1d(in_channels=self.bart.config.d_model, out_channels=512, kernel_size=3, padding=1)
        self.lstm = nn.LSTM(input_size=512, hidden_size=256, num_layers=1, batch_first=True)
        self.fc = nn.Linear(256, self.bart.config.vocab_size)
        self.activation = nn.ReLU()

    def forward(self, input_ids, attention_mask, labels=None):
        outputs = self.bart(input_ids=input_ids, attention_mask=attention_mask, labels=labels)
        hidden_states = outputs.encoder_last_hidden_state
        hidden_states = hidden_states.permute(0, 2, 1)  # Change shape for CNN
        cnn_output = self.activation(self.cnn(hidden_states))
        cnn_output = cnn_output.permute(0, 2, 1)  # Change shape back for LSTM
        lstm_output, _ = self.lstm(cnn_output)
        logits = self.fc(lstm_output)
        return logits

    def generate(self, input_ids, attention_mask, max_length=150, min_length=30, length_penalty=2.0, num_beams=4, early_stopping=True):
        return self.bart.generate(input_ids=input_ids, attention_mask=attention_mask, max_length=max_length, min_length=min_length, length_penalty=length_penalty, num_beams=num_beams, early_stopping=early_stopping)

class TextSummarizer:
    def __init__(self, model_name='facebook/bart-base'):
        self.tokenizer = BartTokenizer.from_pretrained(model_name)
        self.model = EnhancedBartSummarizer(model_name)
        self.device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
        self.model.to(self.device)

    def summarize(self, text, max_length=150):
        inputs = self.tokenizer.encode("summarize: " + text, return_tensors="pt", max_length=512, truncation=True).to(self.device)
        summary_ids = self.model.generate(
            input_ids=inputs,
            attention_mask=inputs.ne(self.tokenizer.pad_token_id).long(),
            max_length=max_length,
            min_length=30,
            length_penalty=2.0,
            num_beams=4,
            early_stopping=True
        )
        summary = self.tokenizer.decode(summary_ids[0], skip_special_tokens=True)
        return summary

    def fine_tune(self, train_dataset, val_dataset, epochs=3, batch_size=8, learning_rate=5e-5, early_stopping_patience=3):
        train_loader = DataLoader(train_dataset, batch_size=batch_size, shuffle=True)
        val_loader = DataLoader(val_dataset, batch_size=batch_size)

        optimizer = AdamW(self.model.parameters(), lr=learning_rate)
        total_steps = len(train_loader) * epochs
        scheduler = get_linear_schedule_with_warmup(optimizer, num_warmup_steps=0, num_training_steps=total_steps)
        # self.model.train()

        best_rougeL = 0
        patience_counter = 0

        train_losses = []
        val_losses = []
        rouge1_scores = []
        rougeL_scores = []

        for epoch in range(epochs):
            self.model.train()
            total_loss = 0
            for batch in train_loader:
                optimizer.zero_grad()
                input_ids = batch['input_ids'].to(self.device)
                attention_mask = batch['attention_mask'].to(self.device)
                labels = batch['labels'].to(self.device)

                logits = self.model(input_ids=input_ids, attention_mask=attention_mask, labels=labels)
                loss = nn.CrossEntropyLoss()(logits.view(-1, logits.size(-1)), labels.view(-1))
                total_loss += loss.item()

                loss.backward()
                optimizer.step()
                scheduler.step()

            avg_train_loss = total_loss / len(train_loader)
            print(f"Epoch {epoch + 1}, Training Loss: {avg_train_loss}")

            val_loss, rouge_scores = self.evaluate(val_loader)
            print(f"Validation Loss: {val_loss}")
            print(f"ROUGE Scores: {rouge_scores}")

            train_losses.append(avg_train_loss)
            val_losses.append(val_loss)
            rouge1_scores.append(rouge_scores['rouge1'])
            rougeL_scores.append(rouge_scores['rougeL'])

            if rouge_scores['rougeL'] > best_rougeL:
                best_rougeL = rouge_scores['rougeL']
                patience_counter = 0
                self.save_model(epoch + 1)
            else:
                patience_counter += 1

            if patience_counter >= early_stopping_patience:
                print("Early stopping triggered")
                break

        # Plot training and validation losses
        plt.figure(figsize=(10, 5))
        plt.plot(range(1, len(train_losses) + 1), train_losses, label='Training Loss')
        plt.plot(range(1, len(val_losses) + 1), val_losses, label='Validation Loss')
        plt.xlabel('Epochs')
        plt.ylabel('Loss')
        plt.title('Training and Validation Losses')
        plt.legend()
        plt.show()

        # Plot ROUGE scores
        plt.figure(figsize=(10, 5))
        plt.plot(range(1, len(rouge1_scores) + 1), rouge1_scores, label='ROUGE-1')
        plt.plot(range(1, len(rougeL_scores) + 1), rougeL_scores, label='ROUGE-L')
        plt.xlabel('Epochs')
        plt.ylabel('ROUGE Score')
        plt.title('ROUGE Scores')
        plt.legend()
        plt.show()

    def evaluate(self, val_loader):
        self.model.eval()
        total_loss = 0
        all_hypotheses = []
        all_references = []

        with torch.no_grad():
            for batch in val_loader:
                input_ids = batch['input_ids'].to(self.device)
                attention_mask = batch['attention_mask'].to(self.device)
                labels = batch['labels'].to(self.device)

                logits = self.model(input_ids=input_ids, attention_mask=attention_mask, labels=labels)
                loss = nn.CrossEntropyLoss()(logits.view(-1, logits.size(-1)), labels.view(-1))
                total_loss += loss.item()

                summaries = self.model.generate(input_ids=input_ids, attention_mask=attention_mask)
                decoded_summaries = [self.tokenizer.decode(s, skip_special_tokens=True) for s in summaries]
                decoded_labels = [self.tokenizer.decode(l, skip_special_tokens=True) for l in labels]

                all_hypotheses.extend(decoded_summaries)
                all_references.extend(decoded_labels)

        avg_val_loss = total_loss / len(val_loader)
        rouge = rouge_scorer.RougeScorer(['rouge1', 'rougeL'], use_stemmer=True)
        rouge_scores = {key: 0 for key in rouge.score(all_hypotheses[0], all_references[0]).keys()}
        for hyp, ref in zip(all_hypotheses, all_references):
            scores = rouge.score(hyp, ref)
            for key in scores:
                rouge_scores[key] += scores[key].fmeasure

        for key in rouge_scores:
            rouge_scores[key] /= len(all_hypotheses)

        return avg_val_loss, rouge_scores

    def save_model(self, epoch):
        model_save_path = f"enhanced_bart_model_epoch_{epoch}.pt"
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