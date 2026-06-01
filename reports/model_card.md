# Model Card: Neural Sequence Transformer Lab

## Intended Use

Educational and portfolio demonstration of neural sequence models for customer support ticket triage.

## Dataset

Synthetic support messages labeled by intent and urgency. The dataset is intentionally small so the full code is easy to inspect; production use would require thousands of labeled examples.

## Architectures

- Embedding baseline for fast classification.
- BiLSTM encoder for contextual sequence modeling.
- Sequence autoencoder for representation learning.
- Transformer encoder classifier for attention-based modeling.

## Evaluation

The evaluation script exports classification report and confusion matrix CSV files. For production, add calibration, drift monitoring, and human review for low-confidence predictions.

## Suggested Experiments

- Compare embedding, BiLSTM, and transformer validation accuracy.
- Pretrain the autoencoder and transfer encoder representations.
- Add urgency as a second classification head for multi-task learning.
- Replace the custom tokenizer with a pretrained tokenizer and fine-tune a Hugging Face model.
