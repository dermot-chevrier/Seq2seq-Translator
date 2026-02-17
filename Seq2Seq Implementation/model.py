
import torch
import torch.nn as nn

"""
EncoderRNN: reads the input sentence (e.g., English)

BahdanauAttention: helps the decoder focus on relevant parts

DecoderRNN: generates the output sentence (e.g., French)

Seq2Seq: wraps the encoder and decoder together into one full model
"""

class EncoderRNN(nn.Module):
    def __init__(self, input_dim, emb_dim, hid_dim, n_layers=1, dropout=0.5, rnn_type='gru'):
        """
        Takes an input sentence (word IDs) and produces a context that summarizes its meaning
        bidirectional GRU
        Reads the sentence forward and backward

        Helps capture full context"""
        super().__init__()
        self.embedding = nn.Embedding(input_dim, emb_dim)
        self.rnn_type = rnn_type.lower()
        self.n_layers = n_layers
        self.hid_dim = hid_dim

        if self.rnn_type == 'gru':
            self.rnn = nn.GRU(emb_dim, hid_dim, n_layers, dropout=dropout, batch_first=True, bidirectional=True)
        elif self.rnn_type == 'lstm':
            self.rnn = nn.LSTM(emb_dim, hid_dim, n_layers, dropout=dropout, batch_first=True, bidirectional=True)
        else:
            raise ValueError("Unsupported RNN type")

        self.dropout = nn.Dropout(dropout)

    def forward(self, src, src_mask=None):
        # src: (batch_size, src_len)
        embedded = self.dropout(self.embedding(src))  # (batch_size, src_len, emb_dim)
        #print("embedded shape:", embedded.shape)
        outputs, hidden = self.rnn(embedded)  # outputs: (batch_size, src_len, hid_dim*2)
        return outputs, hidden  # outputs: contains hidden states for all time steps

#hidden is the final hidden# state s, used to start the decoder
    
class BahdanauAttention(nn.Module):
    """
    Lets the decoder focus on relevant words in the input sentence at each decoding step.
    Combines the current decoder hidden state + all encoder outputs

    Scores how well each encoder output matches the decoder’s needs
    Produces attention weights (softmax scores over encoder words)

    Returns a vector of weights (one per source word) — higher = more attention
    """
    def __init__(self, enc_hid_dim, dec_hid_dim):
        super().__init__()
        self.attn = nn.Linear(enc_hid_dim * 2 + dec_hid_dim, dec_hid_dim)
        self.v = nn.Linear(dec_hid_dim, 1, bias=False)

    def forward(self, decoder_hidden, encoder_outputs, mask=None):
        # decoder_hidden: (batch, dec_hid_dim)
        # encoder_outputs: (batch, src_len, enc_hid_dim*2)
        batch_size = encoder_outputs.size(0)
        src_len = encoder_outputs.size(1)

        # Repeats decoder hidden state ""src_len" times for each batch
        decoder_hidden = decoder_hidden.unsqueeze(1).repeat(1, src_len, 1)

        # concat: (batch, src_len, enc_hid_dim*2 + dec_hid_dim)
        energy = torch.tanh(self.attn(torch.cat((decoder_hidden, encoder_outputs), dim=2)))  # (batch, src_len, dec_hid_dim)
        attention = self.v(energy).squeeze(2)  # (batch, src_len)

        if mask is not None:
            attention = attention.masked_fill(mask == 0, -1e10)

        return torch.softmax(attention, dim=1)  # (batch, src_len)
    

class DecoderRNN(nn.Module):
    """
    Embeds the input token (previous word)

        Uses attention to get a context vector

        Concatenates embedding + context, and feeds into GRU

        Predicts next token with a linear layer"""
    def __init__(self, output_dim, emb_dim, enc_hid_dim, dec_hid_dim, attention, n_layers=1, dropout=0.5, rnn_type='gru'):
        super().__init__()
        self.output_dim = output_dim
        self.attention = attention
        self.rnn_type = rnn_type.lower()

        self.embedding = nn.Embedding(output_dim, emb_dim)
        self.rnn = nn.GRU(enc_hid_dim * 2 + emb_dim, dec_hid_dim, n_layers, dropout=dropout, batch_first=True)
        self.fc_out = nn.Linear(enc_hid_dim * 2 + dec_hid_dim + emb_dim, output_dim)
        self.dropout = nn.Dropout(dropout)

    def forward(self, input_token, decoder_hidden, encoder_outputs, mask=None):
        # input_token: (batch,)
        # decoder_hidden: (n_layers, batch, dec_hid_dim)
        # encoder_outputs: (batch, src_len, enc_hid_dim*2)

        input_token = input_token.unsqueeze(1)  # (batch, 1)
        embedded = self.dropout(self.embedding(input_token))  # (batch, 1, emb_dim)

        # Gets attention weights (batch, src_len)
        attn_weights = self.attention(decoder_hidden[-1], encoder_outputs, mask)  # uses last layer of hidden

        # Computes context vector (batch, 1, enc_hid_dim*2)
        attn_weights = attn_weights.unsqueeze(1)
        context = torch.bmm(attn_weights, encoder_outputs)

        # Concatenates embedded + context
        rnn_input = torch.cat((embedded, context), dim=2)  # (batch, 1, emb+ctx)

        # Passes through RNN
        output, hidden = self.rnn(rnn_input, decoder_hidden)  # output: (batch, 1, dec_hid_dim)

        # Predicts next token
        output = output.squeeze(1)      # (batch, dec_hid_dim)
        context = context.squeeze(1)    # (batch, enc_hid_dim*2)
        embedded = embedded.squeeze(1)  # (batch, emb_dim)
        pred_input = torch.cat((output, context, embedded), dim=1)
        prediction = self.fc_out(pred_input)  # (batch, output_dim)

        return prediction, hidden, attn_weights.squeeze(1)  # (batch, vocab_size), (n_layers, batch, dec_hid), (batch, src_len)
"""
Returns:

prediction: next word probabilities

hidden: updated hidden state

attention: weights over encoder outputs (optional for plotting)
"""


class Seq2Seq(nn.Module):#Wraps everything encoder + attention + decoder  into a full sequence-to-sequence model
    def __init__(self, encoder, decoder, device, pad_idx=0):
        super().__init__()
        self.encoder = encoder
        self.decoder = decoder
        self.device = device
        self.pad_idx = pad_idx

    def create_mask(self, src):
        return (src != self.pad_idx).to(self.device)

    def forward(self, src, tgt, teacher_forcing_ratio=0.5):
        batch_size = src.size(0)
        tgt_len = tgt.size(1)
        tgt_vocab_size = self.decoder.output_dim

        """
        Feeds the target sentence to the decoder step-by-step

        Uses teacher forcing: sometimes uses the true previous word, sometimes the models guess
        """

        outputs = torch.zeros(batch_size, tgt_len, tgt_vocab_size).to(self.device)

        encoder_outputs, hidden = self.encoder(src)
        mask = self.create_mask(src)

        # Prepares decoder initial hidden state (uses last layer of encoder)
        if isinstance(hidden, tuple):  # LSTM
            hidden = hidden[0]  # Only useings hidden state, not cell state

        # Suming up bidirectional hidden states
        if self.encoder.rnn.bidirectional:
            hidden = hidden.view(self.encoder.n_layers, 2, batch_size, self.encoder.hid_dim)
            hidden = hidden.sum(dim=1)

        input_token = tgt[:, 0]  # Starting with <sos> token

        for t in range(1, tgt_len):
            output, hidden, _ = self.decoder(input_token, hidden, encoder_outputs, mask)
            outputs[:, t] = output
            top1 = output.argmax(1)
            input_token = tgt[:, t] if torch.rand(1).item() < teacher_forcing_ratio else top1

        return outputs
    
    def greedy_decode(self, src, src_vocab, tgt_vocab, max_len=50, device='cpu'):

        """
        Used during evaluation (e.g., BLEU score)

        Starts with <sos>, predicts one word at a time until <eos> or max length
        """
        self.eval()
        src = src.to(device)
        src_mask = (src != src_vocab.stoi['<pad>']).to(device)

        with torch.no_grad():
            encoder_outputs, hidden = self.encoder(src)
            if isinstance(hidden, tuple):  
                hidden = hidden[0]

            if self.encoder.rnn.bidirectional:
                batch_size = hidden.size(1)
                n_layers = self.encoder.n_layers
                hid_dim = self.encoder.hid_dim
                hidden = hidden.view(n_layers, 2, batch_size, hid_dim).sum(dim=1)

            input_token = torch.tensor([tgt_vocab.stoi['<sos>']], device=device)

            decoded_tokens = []
            attentions = []  
            mask = src_mask

            for _ in range(max_len):
                output, hidden, attention = self.decoder(input_token, hidden, encoder_outputs, mask)
                pred_token = output.argmax(1).item()
                decoded_tokens.append(pred_token)

                if attention is not None:
                    attentions.append(attention.squeeze(0).cpu())  

                if pred_token == tgt_vocab.stoi['<eos>']:
                    break

                input_token = torch.tensor([pred_token], device=device)

        # we staccking attentions into (tgt_len, src_len)
        attentions = torch.stack(attentions, dim=0).numpy() if attentions else None

        return decoded_tokens, attentions



