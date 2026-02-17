from collections import Counter
import pickle
import re

class Vocab:
    def __init__(self, sentences, min_freq=1):
        self.specials = ['<pad>', '<sos>', '<eos>', '<unk>']
        """
        <pad>: for padding sentences
        <sos>: start of sentence
        <eos>: end of sentence
        <unk>: unknown word
        get added to vocabulary first always
        """
        self.freqs = Counter()
        for sentence in sentences:
            tokens = self.tokenize(sentence)
            self.freqs.update(tokens)
        # tokenizes every sentance and count how often word appears
        self.itos = self.specials.copy()
        for word, freq in self.freqs.items():
            if freq >= min_freq and word not in self.itos:
                self.itos.append(word)
            """
            itos = index → string (a list)
            - Adds all words with frequency ≥ min_freq to the vocabulary
            - Result: every word has a position in itos, e.g. itos[4] = "cat"
            """
        self.stoi = {word: i for i, word in enumerate(self.itos)}

    def tokenize(self, sentence):
        return re.findall(r"\w+|[^\w\s]", sentence.lower(), re.UNICODE)

    def numericalize(self, sentence, max_len):
        tokens = self.tokenize(sentence)[:max_len - 1]
        tokens.append('<eos>')
        ids = [self.stoi.get(token, self.stoi['<unk>']) for token in tokens]
        return ids

    def decode(self, ids):
        tokens = [self.itos[i] for i in ids if i < len(self.itos)]
        return tokens

    def __len__(self):
        return len(self.itos)
    """
    
    Tokenizes the sentence
    Truncates to max_len - 1
    Adds <eos> at the end
    Looks up each token’s ID in the vocab
    If token is unknown, uses <unk>'s ID
    """
    
    def save(self, filename):
        with open(filename, 'wb') as f:
            pickle.dump(self, f)

    @staticmethod
    def load(filename):
        with open(filename, 'rb') as f:
            return pickle.load(f)
    
    
    
# Example test
if __name__ == "__main__":
    sample_sentences = ["i love cats", "you love dogs", "cats love fish"]
    tokens = [word for sent in sample_sentences for word in sent.lower().split()]

    vocab = Vocab(tokens)
    print("Vocab size:", len(vocab))
    print("itos:", vocab.itos)
    print("stoi:", vocab.stoi)

    test_sent = "i love fish"
    print("Numericalized:", vocab.numericalize(test_sent, max_len=6))
    print("Decoded:", vocab.decode(vocab.numericalize(test_sent, max_len=6)))


