import sys
sys.path.append("..")

from data_loading import get_spam_dataset
from transformers import AutoModelForSequenceClassification, AutoTokenizer, Trainer, TrainingArguments
from sklearn.model_selection import train_test_split
from datasets import Dataset

def get_and_prepare_data():
    df = get_spam_dataset()
    df.rename(columns={"Body": "text", "Label": "label"}, inplace = True)

    return df

def split_dataset(df):

    return train_test_split(df, test_size = 0.2, random_state = 42)

def tokenize_function(examples, tokenizer):

    return tokenizer(examples["text"], padding = "max_length", truncation = True, max_length = 512)

def tokenize_data(dataframe):
    tokenizer = AutoTokenizer.from_pretrained("distilbert-base-uncased")

    return tokenize_function(dataframe.to_dict(orient = "list"), tokenizer)

def create_dataset(train_encodings, test_encodings, train_df, test_df):
    train_dataset = Dataset.from_dict({**train_encodings, "labels": train_df["label"].tolist()})
    test_dataset = Dataset.from_dict({**test_encodings, "labels": test_df["label"].tolist()})

    return train_dataset, test_dataset

def train_and_save_model(train_dataset, test_dataset):
    model = AutoModelForSequenceClassification.from_pretrained("distilbert-base-uncased", num_labels = 2)

    training_args = TrainingArguments(
        output_dir = "./results",
        num_train_epochs = 20,
        per_device_train_batch_size = 8,
        warmup_steps = 500,
        weight_decay = 0.01,
        logging_dir = "./logs",
        logging_steps = 10,
    )
    trainer = Trainer(
        model = model,
        args = training_args,
        train_dataset = train_dataset,
        eval_dataset = test_dataset,
    )
    trainer.train()
    model_path = "./spam_model"
    model.save_pretrained(model_path)
    tokenizer = AutoTokenizer.from_pretrained("distilbert-base-uncased")
    tokenizer.save_pretrained(model_path)

def main():
    df = get_and_prepare_data()
    train_df, test_df = split_dataset(df)
    train_encodings = tokenize_data(train_df)
    test_encodings = tokenize_data(test_df)
    train_dataset, test_dataset = create_dataset(train_encodings, test_encodings, train_df, test_df)
    train_and_save_model(train_dataset, test_dataset)

if __name__ == "__main__":
    main()
