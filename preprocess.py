# preprocess.py

import pandas as pd
import random
from nltk.corpus import wordnet
import nltk

nltk.download('wordnet')

# 🔁 Synonym replacement function
def get_synonyms(word):
    synonyms = set()
    for syn in wordnet.synsets(word):
        for lemma in syn.lemmas():
            synonym = lemma.name().replace('_', ' ')
            if synonym.lower() != word.lower():
                synonyms.add(synonym)
    return list(synonyms)

def synonym_augment(text):
    words = text.split()
    new_words = []
    for word in words:
        synonyms = get_synonyms(word)
        if synonyms and random.random() < 0.2:
            new_words.append(random.choice(synonyms))
        else:
            new_words.append(word)
    return ' '.join(new_words)

# ✅ Load and preprocess your dataset
def load_dataset(path):
    df = pd.read_csv(path)

    # Rename if necessary
    df.rename(columns={'your_text_column': 'cleaned_text', 'your_label_column': 'Label'}, inplace=True)

    # Convert labels if needed
    if df['Label'].dtype == object:
        label_map = {'Negative': 0, 'Neutral': 1, 'Positive': 2}
        df['Label'] = df['Label'].map(label_map)

    # Apply synonym augmentation
    augmented_texts = df['cleaned_text'].apply(synonym_augment)
    augmented_df = df.copy()
    augmented_df['cleaned_text'] = augmented_texts

    # Combine original and augmented
    df = pd.concat([df, augmented_df], ignore_index=True)
    return df
