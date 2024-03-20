import matplotlib.pyplot as plt
import nltk
import string

from collections import Counter
from nltk.tokenize import word_tokenize
from nltk.corpus import stopwords
from wordcloud import WordCloud

from data_loading import get_spam_dataset

nltk.download('punkt')
nltk.download('stopwords')

df = get_spam_dataset(path="./data/")


def create_spam_ham_ratio():
    label_counts = df['Label'].value_counts().reset_index()
    label_counts.columns = ['Label', 'Count']

    color_map = {0: 'blue', 1: 'red'}
    colors = label_counts['Label'].map(color_map)

    plt.figure(figsize=(8, 6))
    plt.bar(label_counts['Label'], label_counts['Count'], color=colors)

    plt.title("Count of Labels")
    plt.xlabel("Label")
    plt.ylabel("Count")
    plt.xticks(ticks=label_counts['Label'], labels=['Non-Spam', 'Spam'])

    plt.savefig("./plots/spam_ham_ratio.png")


def generate_wordcloud_all_dataset():
    all_text = ' '.join(df['Body'])
    tokens = word_tokenize(all_text)

    stop_words = set(stopwords.words('english'))
    table = str.maketrans('', '', string.punctuation)
    filtered_tokens = [word.translate(table) for word in tokens if word.lower() not in stop_words and word.isalnum()]

    word_freq = Counter(filtered_tokens)

    top_words = word_freq.most_common(20)

    wordcloud = WordCloud(width=800, height=400, background_color='white').generate_from_frequencies(word_freq)

    plt.figure(figsize=(20, 8))  # Increase the total figure size.

    plt.subplot(1, 2, 1)
    plt.bar(*zip(*top_words))
    plt.xlabel('Words')
    plt.ylabel('Frequency')
    plt.title('Most Frequent Words in Body Column')
    plt.xticks(rotation=45, ha='right')

    plt.subplot(1, 2, 2)
    plt.imshow(wordcloud, interpolation='bilinear')
    plt.axis('off')
    plt.title('Word Cloud of Body Column')

    plt.savefig("./plots/wordcloud_all_dataset.png")


def generate_ham_wordcloud():
    df_ham = df[df['Label'] == 0]

    all_text = ' '.join(df_ham['Body'])

    tokens = word_tokenize(all_text)

    stop_words = set(stopwords.words('english'))
    table = str.maketrans('', '', string.punctuation)
    filtered_tokens = [word.translate(table) for word in tokens if word.lower() not in stop_words and word.isalnum()]

    word_freq = Counter(filtered_tokens)

    top_words = word_freq.most_common(20)
    wordcloud = WordCloud(width=800, height=400, background_color='white').generate_from_frequencies(word_freq)

    plt.figure(figsize=(20, 8))  # Increase the total figure size.

    plt.subplot(1, 2, 1)
    plt.bar(*zip(*top_words))
    plt.xlabel('Words')
    plt.ylabel('Frequency')
    plt.title('Most Frequent Words in Body Column')
    plt.xticks(rotation=45, ha='right')

    plt.subplot(1, 2, 2)
    plt.imshow(wordcloud, interpolation='bilinear')
    plt.axis('off')
    plt.title('Word Cloud of Body Column')

    plt.savefig("./plots/wordcloud_ham.png")


def generate_spam_wordcloud():
    df_spam = df[df['Label'] == 1]

    all_text = ' '.join(df_spam['Body'])

    tokens = word_tokenize(all_text)

    stop_words = set(stopwords.words('english'))
    table = str.maketrans('', '', string.punctuation)
    filtered_tokens = [word.translate(table) for word in tokens if word.lower() not in stop_words and word.isalnum()]

    word_freq = Counter(filtered_tokens)

    top_words = word_freq.most_common(20)
    wordcloud = WordCloud(width=800, height=400, background_color='white').generate_from_frequencies(word_freq)

    plt.figure(figsize=(20, 8))  # Increase the total figure size.

    plt.subplot(1, 2, 1)
    plt.bar(*zip(*top_words))
    plt.xlabel('Words')
    plt.ylabel('Frequency')
    plt.title('Most Frequent Words in Body Column')
    plt.xticks(rotation=45, ha='right')

    plt.subplot(1, 2, 2)
    plt.imshow(wordcloud, interpolation='bilinear')
    plt.axis('off')
    plt.title('Word Cloud of Body Column')

    plt.savefig("./plots/wordcloud_spam.png")


create_spam_ham_ratio()
generate_wordcloud_all_dataset()
generate_ham_wordcloud()
generate_spam_wordcloud()