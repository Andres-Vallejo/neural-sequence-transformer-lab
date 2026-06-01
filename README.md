# Neural Sequence Transformer Lab

Deep learning portfolio project for sequence modeling and NLP. The project classifies customer support messages using multiple neural architectures:

1. EmbeddingBag baseline for fast text classification.
2. BiLSTM encoder for contextual sequence representation.
3. Denoising sequence autoencoder for representation learning.
4. Transformer Encoder classifier with token embeddings and positional encoding.

## Executive Scenario

A support operations team needs to route incoming tickets by intent and urgency. This repository demonstrates how neural sequence models can convert raw text into actionable labels for triage automation.

## Models Included

- FastTextLikeClassifier: embedding encoder with mean pooling.
- BiLSTMClassifier: recurrent encoder with bidirectional hidden states.
- SequenceAutoencoder: encoder-decoder model for reconstruction pretraining.
- TransformerTextClassifier: token embeddings, positional embeddings, TransformerEncoder, classifier head.

## Repository Structure

- data/support_tickets.csv: synthetic labeled support messages.
- configs/default.yaml: training configuration.
- src/dataset.py: tokenizer, vocabulary, dataset, dataloaders.
- src/models.py: neural architectures.
- src/train.py: training loop for classification and autoencoder objectives.
- src/evaluate.py: metrics export.
- reports/model_card.md: model notes and recommended experiments.

## Quick Start

pip install -r requirements.txt
python src/train.py --model embedding --epochs 10
python src/train.py --model bilstm --epochs 10
python src/train.py --model transformer --epochs 10
python src/train.py --model autoencoder --epochs 10
python src/evaluate.py --model transformer

## Skills Demonstrated

Text preprocessing, vocabulary encoding, neural encoders, recurrent models, autoencoders, Transformer encoders, PyTorch training loops, evaluation, and model documentation.
