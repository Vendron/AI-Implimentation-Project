import torch
from transformers import AutoTokenizer, AutoModelForSequenceClassification

def predict_spam(email_texts):
    model_path = "./spam_model"
    tokenizer = AutoTokenizer.from_pretrained(model_path)
    model = AutoModelForSequenceClassification.from_pretrained(model_path)
    results = []

    inputs = tokenizer(
        email_texts,
        padding = True,
        truncation = True,
        max_length = 512,
        return_tensors = "pt"
    )

    for key, val in inputs.items():
        inputs[key] = val.to(model.device)
    
    with torch.no_grad():
        outputs = model(**inputs)
        predictions = outputs.logits.argmax(dim=-1)

    for pred in predictions:
        if pred == 1:
            results.append("spam")
        else:
            results.append("not spam")
    return results


def main():
    choice = input("Enter 1 to upload a .txt file or 2 to paste the email text directly: ")

    if choice == "1":
        file_path = input("Enter the path to your .txt file: ")
        try:
            with open(file_path, "r", encoding="utf-8") as file:
                email_text = file.read()
                predictions = predict_spam([email_text])
                print(f"Email: '{email_text}'\nPrediction: {predictions[0]}")
        except Exception as e:
            print(f"Failed to read file: {e}")
    elif choice == '2':
        email_text = input("Paste the email text here: ")
        predictions = predict_spam([email_text])
        print(f"Email: '{email_text}'\nPrediction: {predictions[0]}")
    else:
        print("Invalid choice. Please enter 1 or 2.")

if __name__ == "__main__":
    main()

