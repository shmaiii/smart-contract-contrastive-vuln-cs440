# SemanticBERT - 440 Project

SemanticBERT is a contrastive learning approach to smart contract vulnerability detection. It extends a CodeBERT encoder with a projection head trained using triplet loss alongside a binary vulnerability classifier, enabling more robust and semantically consistent representations compared to a standard fine-tuned baseline.

### File Structure:
```
.
├── datasets/
│   ├── data_preprocessing.ipynb    # Notebook for cleaning, chunking and tokenizing raw Solidity code
│   ├── sample_train.pt             # Small sample training set for local development
│   └── sample_val.pt               # Small sample validation set for local development
├── evaluation/
│   ├── eval_datasets.py            # Dataset loaders for test sets
│   ├── evaluate.py                 # Evaluator class: metrics, invariance, failure analysis
│   └── semanticbert-eval.ipynb     # Full evaluation notebook (baseline vs SemanticBERT)
├── models/
│   ├── baseline/
│   │   ├── baseline.py             # Standard CodeBERT classifier architecture
│   │   └── smart_datasets.py       # Dataset utilities for the baseline model
│   └── semantic_bert/
│       ├── codebert_contrastive.py # SemanticBERT architecture (encoder + projection + classifier)
│       ├── contrastive_dataset.py  # Triplet dataset with hard negative mining
│       └── train_contrastive.py    # Training script for SemanticBERT
├── train/
│   ├── train-baseline.ipynb        # Training pipeline for the baseline model (Kaggle)
│   └── train-semanticbert.ipynb    # Training pipeline for SemanticBERT (Kaggle)
├── utils/
│   └── plot_training.py            # Parse Kaggle training logs and plot loss/validation curves
├── requirements.txt                # Project dependencies
└── README.md
```

### External Resources

* **Dataset:** [Download via Google Drive](https://drive.google.com/drive/folders/17T1DUDj6FonNTRlHqtYWny6Yuy1qO20O?usp=sharing)
* **SemanticBERT Model Weights:** [Download via Google Drive](https://drive.google.com/drive/folders/10zn1oegeTDq_txPQjisA0fy8KwJsChGq?usp=drive_link)
* **Baseline Model Weights:** [Download via Google Drive](https://drive.google.com/drive/folders/18veB7j1HN2GsqaaDa8WDPK47LWGrOCYk?usp=drive_link)
