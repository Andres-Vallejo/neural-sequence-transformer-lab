# Neural Sequence Transformer Lab

Deep learning portfolio project for sequence modeling and NLP. The project downloads the public AG News dataset, creates reproducible train/validation/test splits, and classifies news stories using multiple neural architectures:

1. EmbeddingBag baseline for fast text classification.
2. BiLSTM encoder for contextual sequence representation.
3. Denoising sequence autoencoder for representation learning.
4. Transformer Encoder classifier with token embeddings and positional encoding.

## Executive Scenario

A media analytics team needs to route incoming news stories into topical desks such as world, sports, business, and science/technology. This repository demonstrates how neural sequence models can convert raw text into actionable labels for editorial triage automation.

## Models Included

- FastTextLikeClassifier: embedding encoder with mean pooling.
- BiLSTMClassifier: recurrent encoder with bidirectional hidden states.
- SequenceAutoencoder: encoder-decoder model for reconstruction pretraining.
- TransformerTextClassifier: token embeddings, positional embeddings, TransformerEncoder, classifier head.

## Repository Structure

- data/raw: downloaded AG News train/test CSV files, ignored by git.
- configs/default.yaml: training configuration.
- src/dataset.py: AG News downloader, tokenizer, vocabulary, dataset, dataloaders, and train/validation/test splitting.
- src/models.py: neural architectures.
- src/train.py: training loop for classification and autoencoder objectives.
- src/evaluate.py: metrics export.
- reports/model_card.md: model notes and recommended experiments.

## Quick Start

```bash
pip install -r requirements.txt
python src/train.py --model embedding --epochs 25
python src/train.py --model bilstm --epochs 25
python src/train.py --model transformer --epochs 25
python src/train.py --model autoencoder --epochs 25
python src/evaluate.py --model transformer
```

## Skills Demonstrated

Text preprocessing, vocabulary encoding, neural encoders, recurrent models, autoencoders, Transformer encoders, Adam optimization, early stopping, PyTorch training loops, evaluation, and model documentation.
