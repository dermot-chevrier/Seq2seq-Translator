# save_vocab_test.py
from vocab import Vocab

sentences = ["hello world", "bonjour le monde"]
vocab = Vocab(sentences)
vocab.save("test_vocab.pkl")

v2 = Vocab.load("test_vocab.pkl")
print(v2.itos[:10])
