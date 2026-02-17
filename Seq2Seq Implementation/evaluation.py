import torch
import evaluate
import matplotlib.pyplot as plt
import numpy as np

bleu_metric = evaluate.load("bleu")

def evaluate_bleu(model, dataloader, src_vocab, tgt_vocab, device):
    model.eval()
    references = []
    predictions = []
    plotted = False

    with torch.no_grad():
        for batch in dataloader: # loops over test data and transl 1 snetance at a time du to batchsize 1
            src_seq, tgt_seq = batch[0].to(device), batch[1].to(device)  # src_seq shape: (batch_size=1, seq_len)

            output_indices, attentions = model.greedy_decode(src_seq, src_vocab, tgt_vocab, max_len=50, device=device)


            # If output_indices is a tensor with batch dimension:
            if isinstance(output_indices, torch.Tensor):
                output_indices = output_indices.squeeze(0).cpu().tolist()
            # If it's already a list we dont have to do sheeeet

            pred_tokens = []
            for idx in output_indices:
                token = tgt_vocab.itos[idx]
                if token == '<eos>':
                    break
                pred_tokens.append(token)
                # converts pedicted IDs --> french words -- stops at <eos>
                # SAMe here:
            tgt_seq = tgt_seq.squeeze(0).cpu().tolist()
            tgt_tokens = []
            for idx in tgt_seq:
                token = tgt_vocab.itos[idx] 
                if token == '<eos>':
                    break
                tgt_tokens.append(token)

            #assert isinstance(tgt_tokens, list), f"tgt_tokens is not a list: {type(tgt_tokens)}"
            #assert all(isinstance(tok, str) for tok in tgt_tokens), f"tgt_tokens contains non-strings: {tgt_tokens}"


            predictions.append(" ".join(pred_tokens))
            references.append([" ".join(tgt_tokens)])   # list of references (can be multiple refs per sent)

            if not plotted and attentions is not None:
                src_tokens = [src_vocab.itos[idx] for idx in src_seq.squeeze(0).cpu().tolist()]
                
                if isinstance(attentions, torch.Tensor):
                    # PyTorch tensor: checks dims with .dim() and squeezes if needed
                    if attentions.dim() == 3 and attentions.size(0) == 1:
                        attentions = attentions.squeeze(0)
                    attentions_np = attentions.cpu().numpy()
                else:
                    # NumPy array: uses .ndim
                    if attentions.ndim == 3 and attentions.shape[0] == 1:
                        attentions = attentions.squeeze(0)
                    attentions_np = attentions

                plot_attention(attentions_np, src_tokens, pred_tokens, save_path="attention_plot.png")

                plotted = True

            
            #print(f"Current pred_tokens: {pred_tokens}")
            #print(f"Current tgt_tokens: {[tgt_tokens]}")

    bleu_score = bleu_metric.compute(predictions=predictions, references=references)
    # uses hugginface eval to calc bleu--> bleu compares predicted translation to reference translatuin using n-gram overlap
    print("Example prediction:", predictions[0])
    print("Example reference:", references[0])
    print("Type of predictions:", type(predictions), "and element type:", type(predictions[0]))
    print("Type of references:", type(references), "and element type:", type(references[0]))

    print(f"BLEU score: {bleu_score['bleu']:.4f}")
    return bleu_score['bleu']


def plot_attention(attentions, src_sentence, tgt_sentence, save_path=None):
    fig, ax = plt.subplots(figsize=(10,10))
    cax = ax.matshow(attentions, cmap='bone')
    fig.colorbar(cax)

    # Sets tick positions explicitly
    ax.set_xticks(range(len(src_sentence) + 1))
    ax.set_yticks(range(len(tgt_sentence) + 1))

    # Then think set tick labels
    ax.set_xticklabels([''] + src_sentence, rotation=90)
    ax.set_yticklabels([''] + tgt_sentence)

    ax.xaxis.set_ticks_position('bottom')

    plt.xlabel('Source Sentence')
    plt.ylabel('Target Sentence')
    plt.title('Attention Heatmap')

    if save_path:
        plt.savefig(save_path)
    else:
        plt.show()

        