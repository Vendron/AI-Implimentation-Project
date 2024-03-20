import torch
import torch.nn as nn
import json

class SimpleClassifier(nn.Module):
    def __init__(self, vocab_size, embed_dim, num_classes):
        super(SimpleClassifier, self).__init__()
        self.embedding = nn.Embedding(vocab_size, embed_dim)
        self.fc = nn.Linear(embed_dim, num_classes)

    def forward(self, input_ids):
        embedded = self.embedding(input_ids).mean(dim = 1)
        output = self.fc(embedded)

        return output

def load_model(model_path, vocab_size, embed_dim = 128, num_classes = 2):
    model = SimpleClassifier(vocab_size, embed_dim, num_classes)
    model.load_state_dict(torch.load(model_path))
    model.eval()

    return model

def load_vocab(vocab_path):
    with open(vocab_path, "r") as vocab_file:
        vocab = json.load(vocab_file)

    return vocab

def tokenize_texts(texts, vocab, max_length = 512):
    tokenized_texts = []

    for text in texts:
        tokens = text.split()[:max_length]
        token_ids = []

        for token in tokens:
            token_id = vocab.get(token, vocab["[UNK]"])
            token_ids.append(token_id)

        token_ids += [vocab["[PAD]"]] * (max_length - len(token_ids))
        tokenized_texts.append(token_ids)

    return tokenized_texts

def predict(texts, model, vocab):
    tokenized_texts = tokenize_texts(texts, vocab)
    input_ids = torch.tensor(tokenized_texts)
    predictions = []

    with torch.no_grad():
        outputs = model(input_ids)
        predicted_labels = torch.argmax(outputs, dim=1)

        for label in predicted_labels:
            if label == 0:
                predictions.append("Non-spam")
            else:
                predictions.append("Spam")
    return predictions

def main():
    model_path = "spam_classifier_model.pth"
    vocab_path = "vocab.json"

    vocab = load_vocab(vocab_path)
    model = load_model(model_path, vocab_size=len(vocab))

    choice = input("Enter 1 to upload a .txt file or 2 to paste the email text directly: ")

    if choice == "1":
        file_path = input("Enter the path to your .txt file: ")
        try:
            with open(file_path, "r", encoding="utf-8") as file:
                email_text = file.read()
        except Exception as e:
            print(f"Failed to read file: {e}")
            return
    elif choice == '2':
        email_text = input("Paste the email text here: ")
    else:
        print("Invalid choice. Please enter 1 or 2.")
        return

    predictions = predict([email_text], model, vocab)
    print(f"Email: '{email_text}'\nPrediction: {predictions[0]}")

if __name__ == "__main__":
    main()
