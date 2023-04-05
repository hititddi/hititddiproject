import numpy as np
import pandas as pd
import torch
from torch.utils.data import Dataset, DataLoader
from transformers import BertTokenizer, BertForSequenceClassification, AdamW, get_linear_schedule_with_warmup
from sklearn.metrics import f1_score, confusion_matrix
from sklearn.model_selection import train_test_split
import seaborn as sns
import matplotlib.pyplot as plt

data = pd.read_csv("teknofest_train_final.csv",delimiter='|')

import string
def preprocess_text(text):
    text = text.lower()  # Harfleri küçük hale getirelim
    text = text.translate(str.maketrans("", "", string.punctuation))  # Noktalama işaretlerini kaldırın
    return text

# Veri kümesindeki metinleri ön işleme gerçekleştirelim
data["text"] = data["text"].apply(preprocess_text)

# "is_offensive" sütunu 1 olan ve "target" sütunu "OTHER" olan metinleri veri setinden çıkaralım
data = data[(data['is_offensive'] == 0) | (data['target'] != 'OTHER')]
# "is_offensive" sütunu 0 olan ve "target" sütunu "OTHER" olmayan metinleri veri setinden çıkaralım
data = data[(data['is_offensive'] == 1) | (data['target'] == 'OTHER')]
# Tek harften oluşan metinleri veri setinden çıkaralım
data = data[data['text'].str.len() > 1]

train_texts, temp_texts, train_labels, temp_labels = train_test_split(data['text'], data['is_offensive'], test_size=0.2, random_state=42)

valid_texts, test_texts, valid_labels, test_labels = train_test_split(temp_texts, temp_labels, test_size=0.5, random_state=42)

tokenizer = BertTokenizer.from_pretrained("dbmdz/bert-base-turkish-cased")

class OffensiveTextDataset(Dataset):
    def __init__(self, texts, labels, tokenizer, max_length=128):
        self.texts = texts
        self.labels = labels
        self.tokenizer = tokenizer
        self.max_length = max_length

    def __len__(self):
        return len(self.texts)

    def __getitem__(self, idx):
        text = self.texts.iloc[idx]
        label = self.labels.iloc[idx]
        inputs = self.tokenizer.encode_plus(
            text,
            add_special_tokens=True,
            max_length=self.max_length,
            padding="max_length",
            truncation=True,
            return_tensors="pt",
        )
        return {"input_ids": inputs["input_ids"].squeeze(), "attention_mask": inputs["attention_mask"].squeeze(), "label": torch.tensor(label)}

train_dataset = OffensiveTextDataset(train_texts, train_labels, tokenizer)
valid_dataset = OffensiveTextDataset(valid_texts, valid_labels, tokenizer)
test_dataset = OffensiveTextDataset(test_texts, test_labels, tokenizer)

train_loader = DataLoader(train_dataset, batch_size=64, shuffle=True)
valid_loader = DataLoader(valid_dataset, batch_size=64, shuffle=False)
test_loader = DataLoader(test_dataset, batch_size=64, shuffle=False)


model = BertForSequenceClassification.from_pretrained("dbmdz/bert-base-turkish-cased", num_labels=2).to("cuda")

optimizer = AdamW(model.parameters(), lr=1e-5,weight_decay=0.1)
total_steps = len(train_loader) * 3
scheduler = get_linear_schedule_with_warmup(optimizer, num_warmup_steps=0, num_training_steps=total_steps)


def train_epoch(model, loader, optimizer, scheduler):
    model.train()
    total_loss = 0
    for batch in loader:
        optimizer.zero_grad()
        input_ids = batch["input_ids"].to("cuda")
        attention_mask = batch["attention_mask"].to("cuda")
        labels = batch["label"].to("cuda")
        outputs = model(input_ids, attention_mask=attention_mask, labels=labels)
        loss = outputs.loss
        total_loss += loss.item()
        loss.backward()
        torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
        optimizer.step()
        scheduler.step()
    return total_loss / len(loader)




def evaluate(model, loader):
    model.eval()
    total_loss = 0
    all_preds, all_labels = [], []
    with torch.no_grad():
        for batch in loader:
            input_ids = batch["input_ids"].to("cuda")
            attention_mask = batch["attention_mask"].to("cuda")
            labels = batch["label"].to("cuda")
            outputs = model(input_ids, attention_mask=attention_mask, labels=labels)
            loss = outputs.loss
            total_loss += loss.item()
            preds = torch.argmax(outputs.logits, dim=-1)
            all_preds.extend(preds.cpu().numpy())
            all_labels.extend(labels.cpu().numpy())
    return total_loss / len(loader), f1_score(all_labels, all_preds), all_preds, all_labels

def train(model, train_loader, valid_loader, optimizer, scheduler, num_epochs=5):
    train_losses, valid_losses = [], []
    for epoch in range(num_epochs):
        train_loss = train_epoch(model, train_loader, optimizer, scheduler)
        valid_loss, valid_f1, _, _ = evaluate(model, valid_loader)
        train_losses.append(train_loss)
        valid_losses.append(valid_loss)
        print(f"Epoch {epoch + 1}/{num_epochs}: train_loss={train_loss}, valid_loss={valid_loss}, valid_f1={valid_f1}")

    return train_losses, valid_losses


torch.cuda.empty_cache()
model.to("cuda")

train_losses, valid_losses = train(model, train_loader, valid_loader, optimizer, scheduler, num_epochs=3)




x_ticks = np.arange(1, len(train_losses) + 1)

plt.plot(x_ticks,train_losses, label="Training Loss")
plt.plot(x_ticks,valid_losses, label="Validation Loss")
plt.xlabel("Epoch")
plt.ylabel("Loss")
plt.legend()
plt.show()


test_loss, test_f1, test_preds, test_labels = evaluate(model, test_loader)
print(f"Test Loss: {test_loss}, Test F1 Score: {test_f1}")

conf_matrix = confusion_matrix(test_labels, test_preds)
sns.heatmap(conf_matrix, annot=True, fmt="d", cmap="Blues")
plt.xlabel("Predicted")
plt.ylabel("Actual")
plt.show()



filtre = data["target"] != "OTHER"
data = data[filtre]



rain_texts, temp_texts, train_targets, temp_targets = train_test_split(data['text'], data['target'], test_size=0.2, random_state=42)

valid_texts, test_texts, valid_targets, test_targets = train_test_split(temp_texts, temp_targets, test_size=0.5, random_state=42)

class_mapping = {"RACIST": 0, "SEXIST": 1, "PROFANITY": 2, "INSULT": 3}
train_labels = train_targets.map(class_mapping)
valid_labels = valid_targets.map(class_mapping)
test_labels = test_targets.map(class_mapping)

def evaluate(model, loader, average="macro"):
    model.eval()
    total_loss = 0
    all_preds, all_labels = [], []
    with torch.no_grad():
        for batch in loader:
            input_ids = batch["input_ids"].to("cuda")
            attention_mask = batch["attention_mask"].to("cuda")
            labels = batch["label"].to("cuda")
            outputs = model(input_ids, attention_mask=attention_mask, labels=labels)
            loss = outputs.loss
            total_loss += loss.item()
            preds = torch.argmax(outputs.logits, dim=-1)
            all_preds.extend(preds.cpu().numpy())
            all_labels.extend(labels.cpu().numpy())
    return total_loss / len(loader), f1_score(all_labels, all_preds, average=average), all_preds, all_labels

class MultiClassOffensiveTextDataset(Dataset):
    def __init__(self, texts, labels, tokenizer, max_length=128):
        self.texts = texts
        self.labels = labels
        self.tokenizer = tokenizer
        self.max_length = max_length

    def __len__(self):
        return len(self.texts)

    def __getitem__(self, idx):
        text = self.texts.iloc[idx]
        label = self.labels.iloc[idx]
        inputs = self.tokenizer.encode_plus(
            text,
            add_special_tokens=True,
            max_length=self.max_length,
            padding="max_length",
            truncation=True,
            return_tensors="pt",
        )
        return {"input_ids": inputs["input_ids"].squeeze(), "attention_mask": inputs["attention_mask"].squeeze(), "label": torch.tensor(label)}

multi_train_dataset = MultiClassOffensiveTextDataset(train_texts, train_labels, tokenizer)
multi_valid_dataset = MultiClassOffensiveTextDataset(valid_texts, valid_labels, tokenizer)
multi_test_dataset = MultiClassOffensiveTextDataset(test_texts, test_labels, tokenizer)

multi_train_loader = DataLoader(multi_train_dataset, batch_size=64, shuffle=True)
multi_valid_loader = DataLoader(multi_valid_dataset, batch_size=64, shuffle=False)
multi_test_loader = DataLoader(multi_test_dataset, batch_size=64, shuffle=False)

multi_model = BertForSequenceClassification.from_pretrained("dbmdz/bert-base-turkish-cased", num_labels=4).to("cuda")

multi_optimizer = AdamW(multi_model.parameters(), lr=2e-5, weight_decay=0.1)
multi_total_steps = len(multi_train_loader) * 3
multi_scheduler = get_linear_schedule_with_warmup(multi_optimizer, num_warmup_steps=0, num_training_steps=multi_total_steps)

torch.cuda.empty_cache()
multi_model.to("cuda")



multi_train_losses, multi_valid_losses = train(multi_model, multi_train_loader, multi_valid_loader, multi_optimizer, multi_scheduler, num_epochs=3)


x_ticks = np.arange(1, len(multi_train_losses) + 1)

plt.plot(x_ticks,multi_train_losses, label="Training Loss")
plt.plot(x_ticks,multi_valid_losses, label="Validation Loss")
plt.xlabel("Epoch")
plt.ylabel("Loss")
plt.legend()
plt.show()

multi_test_loss, multi_test_f1, multi_test_preds, multi_test_labels = evaluate(multi_model, multi_test_loader)
print(f"Test Loss: {multi_test_loss}, Test F1 Score: {multi_test_f1}")
class_labels=["RACIST","SEXIST","PROFANITY","INSULT"]
multi_conf_matrix = confusion_matrix(multi_test_labels, multi_test_preds)
sns.heatmap(multi_conf_matrix, annot=True, fmt="d", cmap="Blues",xticklabels=class_labels, yticklabels=class_labels)
plt.xlabel("Predicted")
plt.ylabel("Actual")
plt.show()


binary_model_path = "binary_model.pt"
torch.save(model.state_dict(), binary_model_path)

multi_model_path = "multi_model.pt"
torch.save(multi_model.state_dict(), multi_model_path)