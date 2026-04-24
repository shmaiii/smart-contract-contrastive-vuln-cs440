# SemanticBERT - 440 Project
### File Structure:
.
├── datasets/
│   ├── contrastive_dataset.py     # Custom Dataset class for contrastive pairs
│   └── data_preprocessing.ipynb    # Notebook for cleaning, chunking and tokenizing raw Solidity code
├── evaluation/
│   ├── eval_datasets.py           # Dataset loaders for test sets
│   ├── evaluate.py                # Script for calculating general metrics and interpretability scores
│   └── semanticbert-eval.ipynb    # Evaluation notebook
├── models/
│   ├── baseline/
│   │   ├── baseline.py            # Standard CodeBERT architecture for classification
│   │   └── smart_datasets.py      # Dataset utilities specific to the baseline model
│   └── codebert_contrastive.py     # SemanticBERT architecture with contrastive loss
├── train/
│   ├── train-baseline.ipynb       # Training pipeline for the baseline model
│   └── train_contrastive.py       # Training script for the SemanticBERT model
├── utils/                         # Helper functions and logging utilities
├── requirements.txt               # Project dependencies
└── README.md

### External Resources

* **Dataset:** [Download via Google Drive](https://drive.google.com/drive/folders/17T1DUDj6FonNTRlHqtYWny6Yuy1qO20O?usp=sharing)
* **SemanticBERT Model Weights:** [Download via Google Drive](https://drive.google.com/drive/folders/10zn1oegeTDq_txPQjisA0fy8KwJsChGq?usp=drive_link)
* **Baseline Model Weights:** [Download via Google Drive](https://drive.google.com/drive/folders/18veB7j1HN2GsqaaDa8WDPK47LWGrOCYk?usp=drive_link)
