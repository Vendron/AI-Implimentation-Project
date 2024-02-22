import sys

sys.path.append("..")

from data_loading import get_spam_dataset

df = get_spam_dataset()
print(df.info())
