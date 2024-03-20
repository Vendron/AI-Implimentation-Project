import sys
sys.path.append("..")

from data_loading import get_spam_dataset

from collections import Counter
import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import Dataset, DataLoader
from sklearn.metrics import accuracy_score
from sklearn.model_selection import train_test_split
import json

class SpamDataset(Dataset):
    def __init__(self, encodings, labels):
        self.encodings = encodings
        self.labels = labels

    def __getitem__(self, idx):
        item = {"input_ids": torch.tensor(self.encodings[idx]), "labels": torch.tensor(self.labels[idx])}

        return item

    def __len__(self):

        return len(self.labels)

class SimpleClassifier(nn.Module):
    def __init__(self, vocab_size, embed_dim, num_classes):
        super(SimpleClassifier, self).__init__()
        self.embedding = nn.Embedding(vocab_size, embed_dim)
        self.fc = nn.Linear(embed_dim, num_classes)

    def forward(self, input_ids):
        embedded = self.embedding(input_ids).mean(dim = 1)
        output = self.fc(embedded)

        return output

def build_vocabulary(texts, vocab_size = 30000):
    token_counter = Counter()
    
    for text in texts:
        tokens = text.split()
        token_counter.update(tokens)

    most_common_tokens = token_counter.most_common(vocab_size)
    vocab = {}

    for idx, (token, _) in enumerate(most_common_tokens, start = 2):
        vocab[token] = idx

    vocab["[PAD]"] = 0
    vocab["[UNK]"] = 1
    
    return vocab

def tokenize_texts(texts, vocab, max_length = 512):
    tokenized_texts = []

    for text in texts:
        tokens = text.split()[:max_length]
        token_ids = []

        for token in tokens:
            token_id = vocab.get(token, vocab["[UNK]"])
            token_ids.append(token_id)
        
        num_padding = max_length - len(token_ids)

        for _ in range(num_padding):
            token_ids.append(vocab["[PAD]"])

        tokenized_texts.append(token_ids)
    
    return tokenized_texts

def train_model(train_loader, val_loader, vocab):
    model = SimpleClassifier(vocab_size = len(vocab), embed_dim = 128, num_classes = 2)
    optimizer = optim.Adam(model.parameters(), lr = 0.001)
    criterion = nn.CrossEntropyLoss()

    for epoch in range(20):
        model.train()

        for batch in train_loader:
            optimizer.zero_grad()
            outputs = model(batch["input_ids"])
            loss = criterion(outputs, batch["labels"])
            loss.backward()
            optimizer.step()
        
        model.eval()
        true_labels, predictions = [], []

        with torch.no_grad():
            for batch in val_loader:
                outputs = model(batch["input_ids"])
                _, predicted = torch.max(outputs, 1)
                true_labels.extend(batch["labels"].numpy())
                predictions.extend(predicted.numpy())

        accuracy = accuracy_score(true_labels, predictions)
        print(f"Epoch {epoch + 1}, Validation Accuracy: {accuracy}")

    return model

def save_vocab(vocab, model):
    vocab_path = "vocab.json"

    with open(vocab_path, "w") as vocab_file:
        json.dump(vocab, vocab_file)

    torch.save(model.state_dict(), "spam_classifier_model.pth")

def main():
    df = get_spam_dataset()
    df.rename(columns={"Body": "text", "Label": "label"}, inplace=True)

    train_df, val_df = train_test_split(df, test_size = 0.2, random_state = 42)

    vocab = build_vocabulary(train_df["text"])
    train_tokenized_texts = tokenize_texts(train_df["text"], vocab)
    val_tokenized_texts = tokenize_texts(val_df["text"], vocab)

    train_dataset = SpamDataset(train_tokenized_texts, train_df["label"].tolist())
    val_dataset = SpamDataset(val_tokenized_texts, val_df["label"].tolist())
    train_loader = DataLoader(train_dataset, batch_size = 8, shuffle = True)
    val_loader = DataLoader(val_dataset, batch_size = 8, shuffle = False)

    model = train_model(train_loader, val_loader, vocab)
    save_vocab(vocab, model)

if __name__ == "__main__":
    main()
