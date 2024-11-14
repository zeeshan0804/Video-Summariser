import torch
from torch.utils.data import Dataset, DataLoader
from transformers import T5ForConditionalGeneration, T5Tokenizer, AdamW
from rouge_score import rouge_scorer
from transformers import get_linear_schedule_with_warmup
<<<<<<< HEAD


class TextSummarizer:
    def __init__(self, model_name='t5-base'):
        self.tokenizer = T5Tokenizer.from_pretrained(model_name)
        self.model = T5ForConditionalGeneration.from_pretrained(model_name)
=======

class TextSummarizer:
    def __init__(self, encoder_model_name='t5-small', decoder_model_name='t5-small'):
        self.tokenizer = T5Tokenizer.from_pretrained(encoder_model_name)
        self.encoder = T5ForConditionalGeneration.from_pretrained(encoder_model_name)
        self.decoder = T5ForConditionalGeneration.from_pretrained(decoder_model_name)
>>>>>>> cd94812 (Revert changes to specific files)
        self.device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
        self.encoder.to(self.device)
        self.decoder.to(self.device)

    def summarize(self, text, max_length=150):
        inputs = self.tokenizer.encode("summarize: " + text, return_tensors="pt", max_length=512, truncation=True).to(self.device)
        encoder_outputs = self.encoder.encoder(inputs)
        decoder_input_ids = torch.tensor([[self.tokenizer.pad_token_id]], device=self.device)
        summary_ids = self.decoder.generate(
            input_ids=decoder_input_ids,
            encoder_outputs=encoder_outputs,
            max_length=max_length,
            min_length=30,
            length_penalty=2.0,
            num_beams=4,
            early_stopping=True
        )
        summary = self.tokenizer.decode(summary_ids[0], skip_special_tokens=True)
        return summary

    from transformers import get_linear_schedule_with_warmup

    def fine_tune(self, train_dataset, val_dataset, epochs=3, batch_size=8, learning_rate=5e-5):
        train_loader = DataLoader(train_dataset, batch_size=batch_size, shuffle=True)
        val_loader = DataLoader(val_dataset, batch_size=batch_size)

<<<<<<< HEAD
        optimizer = AdamW(self.model.parameters(), lr=learning_rate)
        total_steps = len(train_loader) * epochs
        scheduler = get_linear_schedule_with_warmup(optimizer, num_warmup_steps=0, num_training_steps=total_steps)
        self.model.train()
=======
        optimizer = AdamW(list(self.encoder.parameters()) + list(self.decoder.parameters()), lr=learning_rate)
        total_steps = len(train_loader) * epochs
        scheduler = get_linear_schedule_with_warmup(optimizer, num_warmup_steps=0, num_training_steps=total_steps)
        self.encoder.train()
        self.decoder.train()
>>>>>>> cd94812 (Revert changes to specific files)

        for epoch in range(epochs):
            total_loss = 0
            for batch in train_loader:
                optimizer.zero_grad()
                input_ids = batch['input_ids'].to(self.device)
                attention_mask = batch['attention_mask'].to(self.device)
                labels = batch['labels'].to(self.device)

                encoder_outputs = self.encoder.encoder(input_ids=input_ids, attention_mask=attention_mask)
                decoder_outputs = self.decoder(
                    input_ids=labels,
                    encoder_outputs=encoder_outputs,
                    labels=labels
                )
                loss = decoder_outputs.loss
                total_loss += loss.item()

                loss.backward()
                optimizer.step()
                scheduler.step()

            avg_train_loss = total_loss / len(train_loader)
            print(f"Epoch {epoch + 1}, Training Loss: {avg_train_loss}")

            self.evaluate(val_loader)
            self.save_model(epoch + 1)
            
    def evaluate(self, val_loader):
        self.encoder.eval()
        self.decoder.eval()
        total_loss = 0
        rouge = rouge_scorer.RougeScorer(['rouge1', 'rougeL'], use_stemmer=True)
        all_hypotheses = []
        all_references = []

        with torch.no_grad():
            for batch in val_loader:
                input_ids = batch['input_ids'].to(self.device)
                attention_mask = batch['attention_mask'].to(self.device)
                labels = batch['labels'].to(self.device)

                encoder_outputs = self.encoder.encoder(input_ids=input_ids, attention_mask=attention_mask)
                decoder_outputs = self.decoder(
                    input_ids=labels,
                    encoder_outputs=encoder_outputs,
                    labels=labels
                )
                loss = decoder_outputs.loss
                total_loss += loss.item()

                summaries = self.decoder.generate(input_ids=labels, encoder_outputs=encoder_outputs)
                decoded_summaries = [self.tokenizer.decode(s, skip_special_tokens=True) for s in summaries]
                decoded_labels = [self.tokenizer.decode(l, skip_special_tokens=True) for l in labels]

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

    def save_model(self, epoch):
        encoder_save_path = f"encoder_epoch_{epoch}.pt"
        decoder_save_path = f"decoder_epoch_{epoch}.pt"
        torch.save(self.encoder.state_dict(), encoder_save_path)
        torch.save(self.decoder.state_dict(), decoder_save_path)
        print(f"Encoder model saved to {encoder_save_path}")
        print(f"Decoder model saved to {decoder_save_path}")

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