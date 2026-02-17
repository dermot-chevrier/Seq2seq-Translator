import torch

class TranslationDataset(torch.utils.data.Dataset):
    def __init__(self, src_sentences, tgt_sentences, src_vocab, tgt_vocab, max_len=30):
        self.src_sentences = src_sentences
        self.tgt_sentences = tgt_sentences
        self.src_vocab = src_vocab
        self.tgt_vocab = tgt_vocab
        self.max_len = max_len
    
    """
    initializes the dataset with:

    src_sentences: List of strings (Eng)

    tgt_sentences: List of strings (Fr tanslations)

    src_vocab & tgt_vocab: Handle tokenization and mapping words to numbers

    max_len: Max sentence length (used for truncation)

    Stores them as instance variables
    """

    def __len__(self):
        return len(self.src_sentences)
    # PyTorch know how many items are in the dataset for indexing.

    def __getitem__(self, idx):
        src = self.src_vocab.numericalize(self.src_sentences[idx], self.max_len)
        tgt = self.tgt_vocab.numericalize(self.tgt_sentences[idx], self.max_len)
        return torch.tensor(src), torch.tensor(tgt)
    
    """
    Retrieves and converts a single sentence pair at index idx.

    Steps:

    src_vocab.numericalize(...) → converts the source sentence into a list of integers (tokens), adding <eos> and truncating if necessary.

    Same for tgt.

    Converts both lists to PyTorch tensors so they can be used in training.

    Return: A tuple of two tensors: one for the source and one for the target sentence.
    """

    def save(self, filename):
        torch.save({
            'src_sentences': self.src_sentences,
            'tgt_sentences': self.tgt_sentences,
            'max_len': self.max_len
        }, filename)
    # Save just the raw sentence data and max_len to a file
    @staticmethod
    def load(filename, src_vocab, tgt_vocab):
        data = torch.load(filename)
        return TranslationDataset(
            data['src_sentences'],
            data['tgt_sentences'],
            src_vocab,
            tgt_vocab,
            data['max_len']
        )
    #Load from a file and return a new TranslationDataset using the saved sentences and vocabularies.


def collate_fn(batch, pad_idx):
    src_batch, tgt_batch = zip(*batch)
    # batch is a list of (src_tensor, tgt_tensor) tuples
    # zip(*batch) separates them into two lists
    src_lens = [len(s) for s in src_batch]
    tgt_lens = [len(t) for t in tgt_batch]

    max_src_len = max(src_lens)
    max_tgt_len = max(tgt_lens)

    padded_src = torch.full((len(src_batch), max_src_len), pad_idx, dtype=torch.long)
    padded_tgt = torch.full((len(tgt_batch), max_tgt_len), pad_idx, dtype=torch.long)
    #Creates a padded 2D tensor where each row is a sentence
    #Fills with <pad> token index (e.g. 0)
    #
    for i, (src_seq, tgt_seq) in enumerate(zip(src_batch, tgt_batch)):
        padded_src[i, :len(src_seq)] = src_seq
        padded_tgt[i, :len(tgt_seq)] = tgt_seq
    #Copies actual tokens into the padded matrix
   

    src_mask = (padded_src != pad_idx)
    tgt_mask = (padded_tgt != pad_idx)
     # Boolean mask showing where padding is not present
    return padded_src, padded_tgt, src_mask, tgt_mask
    # mask help model ignore padding when computing attention or loss