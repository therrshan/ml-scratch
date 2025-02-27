import os
import numpy as np
import pandas as pd
import nltk
import string
from collections import Counter


class NG20Parser:
    def __init__(self, dataset_dir="data/20NG", subset=True, clean=True, normalize=True, vectorize=True):
        self.dataset_dir = dataset_dir
        self.subset = subset
        self.clean = clean
        self.normalize = normalize
        self.vectorize = vectorize
        self.df = None
        self.tf_matrix = None

    def parse(self):
        data = []
        target = []

        train_path = os.path.join(self.dataset_dir, "20news-bydate-train")
        test_path = os.path.join(self.dataset_dir, "20news-bydate-test")

        datasets = [train_path, test_path]
        if self.subset:
            majority_subset = [
                "alt.atheism",
                "sci.med",
                "sci.electronics",
                "comp.graphics",
                "talk.politics.guns",
                "sci.crypt",
            ]
        else:
            majority_subset = None

        for dataset in datasets:
            for category in os.listdir(dataset):
                if majority_subset is None or category in majority_subset:
                    category_path = os.path.join(dataset, category)
                    if os.path.isdir(category_path):
                        for document in os.listdir(category_path):
                            document_path = os.path.join(category_path, document)
                            with open(document_path, "r", errors="ignore") as f:
                                text = f.read()
                                data.append(text)
                            target.append(category)

        self.df = pd.DataFrame({'text': data, 'target': target})

        if self.clean:
            self.df['tokens'] = self.df['text'].apply(self.clean_text)
        if self.normalize:
            self.df['normalized_tokens'] = self.df['tokens'].apply(self.normalize_tokens)
        if self.vectorize:
            if not self.normalize:
                print(f"Normalize the text first to vectorize")
            self.tf_matrix = self.vectorize_tokens(self.df['normalized_tokens'])

        return self.df, self.tf_matrix

    def clean_text(self, text):

        nltk.download('punkt', quiet=True)
        nltk.download('stopwords', quiet=True)

        tokens = nltk.tokenize.word_tokenize(text)
        tokens = [word.lower() for word in tokens]
        table = str.maketrans('', '', string.punctuation)
        stripped = [word.translate(table) for word in tokens]
        words = [word for word in stripped if word.isalpha()]
        stop_words = set(nltk.corpus.stopwords.words('english'))
        words = [word for word in words if word not in stop_words]
        return words

    @staticmethod
    def normalize_tokens(self, tokens):
        term_freq = Counter(tokens)
        total_terms = len(tokens)
        normalized_tf = {term: freq / total_terms for term, freq in term_freq.items()}
        return normalized_tf

    @staticmethod
    def vectorize_tokens(self, tokens):
        vocab = set()
        for token_dict in tokens:
            vocab.update(token_dict.keys())

        vocab = sorted(vocab)
        vocab_dict = {word: idx for idx, word in enumerate(vocab)}

        res = np.zeros((len(tokens), len(vocab)))
        for i, token_dict in enumerate(tokens):
            for word, weight in token_dict.items():
                res[i, vocab_dict[word]] = weight

        return res

if __name__ == "__main__":
    # Test or script-specific code
    parser = NG20Parser(subset=True, clean=True, normalize=True, vectorize=True)
    df, tf_matrix = parser.parse()
    print(df.head())
