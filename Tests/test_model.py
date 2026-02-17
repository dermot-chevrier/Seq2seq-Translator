import torch
from model import EncoderRNN, BahdanauAttention, DecoderRNN, Seq2Seq
from vocab import Vocab  # adjust import if you used a different name/location

# Dummy small vocab sizes
SRC_VOCAB_SIZE = 100
TGT_VOCAB_SIZE = 120
PAD_IDX = 0

# Embedding and hidden dimensions
EMB_DIM = 16
HID_DIM = 32
DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")

# Dummy batch of input/output (batch_size=2, seq_len=10)
src = torch.randint(1, SRC_VOCAB_SIZE, (2, 10)).to(DEVICE)
tgt = torch.randint(1, TGT_VOCAB_SIZE, (2, 12)).to(DEVICE)
tgt[:, 0] = 1  # <sos> token at the start

# Build model components
encoder = EncoderRNN(SRC_VOCAB_SIZE, EMB_DIM, HID_DIM, dropout=0.1).to(DEVICE)
attention = BahdanauAttention(HID_DIM, HID_DIM).to(DEVICE)
decoder = DecoderRNN(TGT_VOCAB_SIZE, EMB_DIM, HID_DIM, HID_DIM, attention, dropout=0.1).to(DEVICE)

model = Seq2Seq(encoder, decoder, DEVICE, pad_idx=PAD_IDX).to(DEVICE)

# Forward pass
outputs = model(src, tgt, teacher_forcing_ratio=0.75)
print("Output shape:", outputs.shape)  # Expect: (batch_size, tgt_len, TGT_VOCAB_SIZE)
