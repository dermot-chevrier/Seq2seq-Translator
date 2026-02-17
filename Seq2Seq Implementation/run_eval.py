import torch
from evaluation import evaluate_bleu, plot_attention
from vocab import Vocab
from dataset import TranslationDataset, collate_fn
from model import EncoderRNN, DecoderRNN, BahdanauAttention, Seq2Seq
from torch.utils.data import DataLoader

device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

# Loading vocab and test data
src_vocab = Vocab.load('src_vocab.pkl')  # Or however you save/load vocab
tgt_vocab = Vocab.load('tgt_vocab.pkl')

test_dataset = TranslationDataset.load('test_dataset.pt', src_vocab, tgt_vocab)


attention = BahdanauAttention(enc_hid_dim=128, dec_hid_dim=128)
encoder = EncoderRNN(len(src_vocab), 64, 128, n_layers=1, dropout=0.5).to(device)
decoder = DecoderRNN(len(tgt_vocab), 64, 128, 128, attention, n_layers=1, dropout=0.5).to(device)

model = Seq2Seq(encoder, decoder, device=device, pad_idx=src_vocab.stoi['<pad>']).to(device)

# Load trained weights
model.load_state_dict(torch.load('best_model.pt', map_location=device))

# Evaluate BLEU
evaluate_bleu(model, test_dataset, src_vocab, tgt_vocab, device)

# Visualize attention for a few samples
for i in range(3):
    src_seq, tgt_seq, _, _ = test_dataset[i]
    src_seq_batch = src_seq.unsqueeze(0).to(device)

    output_indices, attentions = model.greedy_decode(src_seq_batch, max_len=50)
    output_indices = output_indices.squeeze(0).cpu().numpy()
    attentions = attentions.squeeze(0).cpu().numpy()  # (tgt_len, src_len)

    # Convert to tokens (truncate at <eos>)
    src_tokens = [src_vocab.itos[idx.item()] for idx in src_seq]
    tgt_tokens = []
    for idx in output_indices:
        token = tgt_vocab.itos[idx]
        if token == '<eos>':
            break
        tgt_tokens.append(token)

    plot_attention(attentions[:len(tgt_tokens), :len(src_tokens)], src_tokens, tgt_tokens)
