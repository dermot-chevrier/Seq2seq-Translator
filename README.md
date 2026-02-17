# Seq2seq-Translator
Attention based Sequence to Sequence model for machine translation, implemented in PyTorch.  
Includes training pipeline, teacher forcing, BLEU evaluation, and full encoder decoder architecture with additive attention.

---

## Overview
This project implements a **Sequence-to-Sequence (Seq2Seq)** translation model using an Encoder Decoder architecture with Attention.

The model takes sentences in a source language and translates them into a target language, focusing on:

- sequence modeling  
- recurrent neural networks (GRU / LSTM)  
- attention mechanisms  
- training stability  
- evaluation using BLEU score  

This project demonstrates practical skills in:

- Deep Learning for Natural Language Processing  
- Encoder Decoder architectures  
- Attention mechanisms  
- PyTorch training pipelines  
- Tokenization and text preprocessing  
- Model evaluation and debugging  

---

## Features
- Encoder–Decoder architecture using GRU or LSTM  
- Additive (Bahdanau-style) attention  
- Teacher forcing for efficient training  
- BLEU score evaluation  
- Sentence tokenization and preprocessing  
- Checkpoint saving/loading  
- Modular codebase for easy experimentation  
- Unit tests for training and inference behavior  

---
## Repository Structure

    Seq2seq-Translator/
    ├── Seq2Seq Implementation/
    │   ├── dataset.py                
    │   ├── evaluation.py
    │   ├── model.py
    │   ├── run_eval.py
    │   ├── train.py
    │   └── vocab.py
    │
    ├── Tests/
    │   ├── save_vocab_test.py
    │   ├── test_decoder.py
    │   └── test_model.py           
    │
    ├── README.md
    └── requirements.txt

---

## Dataset
This project uses a parallel text dataset for translation (e.g., English → German, English → French).  
The dataset is not included in this repository.
