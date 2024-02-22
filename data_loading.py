import pandas as pd


def get_spam_dataset(path: str = "../data/") -> pd.core.frame.DataFrame:
    # Load all the data
    # Columns: 'Unnamed: 0.1', 'Unnamed: 0', 'Body', 'Label'
    enron_spam_subset_df = pd.read_csv(path + "enronSpamSubset.csv")
    # Columns: 'Unnamed: 0', 'Body', 'Label'
    ling_spam_df = pd.read_csv(path + "lingSpam.csv")
    # Columns: 'Unnamed: 0', 'Body', 'Label'], dtype='object'
    complete_spam_assassin_df = pd.read_csv(path + "completeSpamAssassin.csv")
    # Columns: 'Unnamed: 0', 'label', 'text', 'label_num'
    spam_ham_dataset_df = pd.read_csv(path + "spam_ham_dataset.csv")
    # Columns: 'email', 'label'
    spam_or_not_spam_df = pd.read_csv(path + "spam_or_not_spam.csv")
    # Columns: 'Category', 'Message'
    spam_df = pd.read_csv(path + "spam.csv")

    # Preprocessing the data
    # enron spam data preprocessing
    enron_spam_subset_df.drop(['Unnamed: 0', 'Unnamed: 0.1'], axis=1, inplace=True)
    # ling spam data preprocessing
    ling_spam_df.drop(['Unnamed: 0'], axis=1, inplace=True)
    # complete spam assassin data preprocessing
    complete_spam_assassin_df.drop(['Unnamed: 0'], axis=1, inplace=True)
    # ham spam dataset preprocessing
    spam_ham_dataset_df.drop(['Unnamed: 0', 'label'], axis=1, inplace=True)
    spam_ham_dataset_df = spam_ham_dataset_df.rename(columns={'text': 'Body', 'label_num': 'Label'})
    # spam or not spam dataset preprocessing
    spam_or_not_spam_df=spam_or_not_spam_df.rename(columns={'email': 'Body', 'label': 'Label'})
    # spam data preprocessing
    spam_df = spam_df.rename(columns={'Message': 'Body', 'Category': 'Label'})
    spam_df['Label'] = spam_df['Label'].map({'ham': 0, 'spam': 1})
    spam_df = spam_df[['Body', 'Label']]

    df=pd.concat([enron_spam_subset_df, ling_spam_df, complete_spam_assassin_df, spam_ham_dataset_df, spam_or_not_spam_df, spam_df])
    # Merging the dataset
    df.dropna(subset=['Body'], inplace=True)
    return 
