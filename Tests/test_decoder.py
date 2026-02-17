# test_decoder.py

import torch
from model import BahdanauAttention, DecoderRNN

# Constants
BATCH_SIZE = 4
SRC_LEN = 10
TGT_VOCAB_SIZE = 50
ENC_HID_DIM = 64
DEC_HID_DIM = 64
EMB_DIM = 32

# Dummy data
encoder_outputs = torch.randn(BATCH_SIZE, SRC_LEN, ENC_HID_DIM * 2)
decoder_hidden = torch.randn(1, BATCH_SIZE, DEC_HID_DIM)
input_token = torch.randint(0, TGT_VOCAB_SIZE, (BATCH_SIZE,))
mask = torch.ones(BATCH_SIZE, SRC_LEN).int()

# Instantiate attention and decoder
attention = BahdanauAttention(enc_hid_dim=ENC_HID_DIM, dec_hid_dim=DEC_HID_DIM)
decoder = DecoderRNN(
    output_dim=TGT_VOCAB_SIZE,
    emb_dim=EMB_DIM,
    enc_hid_dim=ENC_HID_DIM,
    dec_hid_dim=DEC_HID_DIM,
    attention=attention,
    n_layers=1,
    dropout=0.1
)

# Run decoder step
pred_logits, new_hidden, attn_weights = decoder(input_token, decoder_hidden, encoder_outputs, mask)

# Check shapes
print("Prediction logits:", pred_logits.shape)
print("New hidden state:", new_hidden.shape)
print("Attention weights:", attn_weights.shape)
