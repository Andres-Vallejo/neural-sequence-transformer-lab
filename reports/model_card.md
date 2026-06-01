# Model Card: Neural Sequence Transformer Lab

## Intended Use

Educational and portfolio demonstration of neural sequence models for news topic classification.

## Dataset

The default pipeline downloads the public AG News CSV dataset, maps the four labeled classes, and creates reproducible train, validation, and test loaders. The configuration caps sample counts so the GitHub Actions training job stays practical while preserving a real labeled data workflow.

## Architectures

- Embedding baseline for fast classification.
- BiLSTM encoder for contextual sequence modeling.
- Sequence autoencoder for representation learning.
- Transformer encoder classifier for attention-based modeling.

## Evaluation

The evaluation script exports classification report and confusion matrix CSV files from the held-out test split. For production, add calibration, drift monitoring, and human review for low-confidence predictions.

## Suggested Experiments

- Compare embedding, BiLSTM, and transformer test accuracy.
- Pretrain the autoencoder and transfer encoder representations.
- Increase max_train_samples and train longer for higher benchmark accuracy.
- Replace the custom tokenizer with a pretrained tokenizer and fine-tune a Hugging Face model.
