import torch
import torch.nn as nn
from torch.utils.data import DataLoader
import pandas as pd
from sklearn.model_selection import train_test_split

from vocab import Vocab
from dataset import TranslationDataset, collate_fn
from model import EncoderRNN, DecoderRNN, BahdanauAttention
from model import Seq2Seq

# 1. loading and preprocessing data
df = pd.read_csv("rec05_small_en_fr.csv")
src_list = df["EN"].tolist()
tgt_list = df["FR"].tolist()

# 2. building vocabularies
src_vocab = Vocab(src_list, min_freq=2)
tgt_vocab = Vocab(tgt_list, min_freq=2)

# saveing vocabularies
src_vocab.save("src_vocab.pkl")
tgt_vocab.save("tgt_vocab.pkl")

# 3. splitting into train and test
src_train, src_test, tgt_train, tgt_test = train_test_split(
    src_list, tgt_list, test_size=0.2, random_state=42
)

# 4. screateing datasets
train_dataset = TranslationDataset(src_train, tgt_train, src_vocab, tgt_vocab)
test_dataset = TranslationDataset(src_test, tgt_test, src_vocab, tgt_vocab)

# 5. saveing datasets for later evaluation
train_dataset.save("train_dataset.pt")
test_dataset.save("test_dataset.pt")

# 6. createing DataLoader for training
loader = DataLoader(train_dataset, batch_size=32, shuffle=True,
                    collate_fn=lambda x: collate_fn(x, pad_idx=src_vocab.stoi['<pad>']))

# 7. defineing staert model parameters
INPUT_DIM = len(src_vocab)
OUTPUT_DIM = len(tgt_vocab)
ENC_EMB_DIM = 64
DEC_EMB_DIM = 64
HID_DIM = 128
N_LAYERS = 2
ENC_DROPOUT = 0.5
DEC_DROPOUT = 0.5

# 8. initialize model
attention = BahdanauAttention(enc_hid_dim=HID_DIM, dec_hid_dim=HID_DIM)
encoder = EncoderRNN(INPUT_DIM, ENC_EMB_DIM, HID_DIM, n_layers=N_LAYERS, dropout=ENC_DROPOUT)
decoder = DecoderRNN(OUTPUT_DIM, DEC_EMB_DIM, HID_DIM, HID_DIM, attention, n_layers=N_LAYERS, dropout=DEC_DROPOUT)
model = Seq2Seq(encoder, decoder, device='cpu', pad_idx=src_vocab.stoi['<pad>']).to('cpu')

# 9. training components
optimizer = torch.optim.Adam(model.parameters(), lr=0.001)
criterion = nn.CrossEntropyLoss(ignore_index=tgt_vocab.stoi['<pad>'])

# 10. training loop
for epoch in range(1, 16):
    model.train()
    epoch_loss = 0
    for src_batch, tgt_batch, src_mask, tgt_mask in loader:
        optimizer.zero_grad()
        output = model(src_batch, tgt_batch, teacher_forcing_ratio=0.5)


        output_dim = output.shape[-1]
        output = output[:, 1:].reshape(-1, output_dim)
        tgt = tgt_batch[:, 1:].reshape(-1)

        loss = criterion(output, tgt)
        loss.backward()
        optimizer.step()

        epoch_loss += loss.item()

    print(f"Epoch {epoch} | Loss: {epoch_loss / len(loader):.4f}")

from evaluation import evaluate_bleu
from dataset import TranslationDataset

# loading test dataset again (or using existing one)
test_dataset = TranslationDataset.load("test_dataset.pt", src_vocab, tgt_vocab)

# runing evaluation on the test set, batch by batch
# So createing a DataLoader with batch size 1 
test_loader = torch.utils.data.DataLoader(
    test_dataset, batch_size=1, shuffle=False,
    collate_fn=lambda x: collate_fn(x, pad_idx=src_vocab.stoi['<pad>'])
)

bleu_score = evaluate_bleu(model, test_loader, src_vocab, tgt_vocab, device='cpu')
print(f"Final BLEU score on test set: {bleu_score:.4f}")
