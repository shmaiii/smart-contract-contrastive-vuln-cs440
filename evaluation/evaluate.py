import numpy as np
import torch
import json
import pandas as pd
from collections import defaultdict

from sklearn.metrics import (
    precision_recall_fscore_support,
    accuracy_score,
    precision_recall_curve,
    roc_auc_score,
    auc,
    confusion_matrix,
    ConfusionMatrixDisplay
)

import matplotlib.pyplot as plt


class Evaluator:
    def __init__(self, model, device, is_contrastive=False, model_name="Model", tokenizer=None):
        self.model = model
        self.device = device
        self.is_contrastive = is_contrastive
        self.model_name = model_name
        self.tokenizer = tokenizer # Added tokenizer for decoding if text isn't in test_raw

        self.results = None

    def evaluate(self, data_loader):
        self.model.eval()
        all_labels, all_probs = [], []
        
        print(f"\n--- Evaluating: {self.model_name} ---")

        with torch.no_grad():
            for batch in data_loader:
                inputs = batch['input_ids'].to(self.device)
                mask = batch['attention_mask'].to(self.device)
                labels = batch['label'].to(self.device)

                if self.is_contrastive:
                    _, probs = self.model(inputs, mask)
                else:
                    logits = self.model(inputs, mask)
                    probs = torch.sigmoid(logits)

                all_labels.extend(labels.cpu().numpy())
                all_probs.extend(probs.cpu().numpy())

        all_labels = np.array(all_labels)
        all_probs = np.array(all_probs)

        threshold = self._find_best_threshold(all_labels, all_probs)
        preds = (all_probs > threshold).astype(int)

        precision, recall, f1, _ = precision_recall_fscore_support(
            all_labels, preds, average='binary'
        )
        prec, rec, _ = precision_recall_curve(all_labels, all_probs)

        self.results = {
            "accuracy": accuracy_score(all_labels, preds),
            "precision": precision,
            "recall": recall,
            "f1": f1,
            "roc_auc": roc_auc_score(all_labels, all_probs),
            "pr_auc": auc(rec, prec),
            "threshold": threshold,
            "raw_probs": all_probs,
            "raw_labels": all_labels
        }

        return self.results

    def evaluate_invariance(self, eval_data, batch_size=32):
        assert hasattr(self.model, "encode_chunks"), \
            "Model must support embedding extraction"

        distances = []
        for start in range(0, len(eval_data), batch_size):
            batch = eval_data[start:start+batch_size]

            anchor_ids = torch.stack([
                torch.tensor(x["anchor_input_ids"]) for x in batch
            ]).to(self.device)

            pos_ids = torch.stack([
                torch.tensor(x["pos_input_ids"]) for x in batch
            ]).to(self.device)

            anchor_mask = anchor_ids.ne(1).long()
            pos_mask = pos_ids.ne(1).long()

            with torch.no_grad():
                anchor_emb = self.model.encode_chunks(anchor_ids, anchor_mask)
                pos_emb = self.model.encode_chunks(pos_ids, pos_mask)

            cos_sim = (anchor_emb * pos_emb).sum(dim=-1)
            cos_dist = 1 - cos_sim
            distances.extend(cos_dist.cpu().numpy())

        distances = np.array(distances)
        print(f"\n[{self.model_name}] Invariance - Mean distance: {distances.mean():.4f}")
        return distances

    def _find_best_threshold(self, labels, probs):
        precision, recall, thresholds = precision_recall_curve(labels, probs)
        f1_scores = 2 * (precision * recall) / (precision + recall + 1e-8)
        best_idx = np.argmax(f1_scores)
        return thresholds[min(best_idx, len(thresholds) - 1)]

    def compare(self, other):
        assert self.results and other.results
        p1, r1, _ = precision_recall_curve(self.results['raw_labels'], self.results['raw_probs'])
        p2, r2, _ = precision_recall_curve(other.results['raw_labels'], other.results['raw_probs'])

        plt.figure(figsize=(8, 6))
        plt.plot(r1, p1, label=f"{self.model_name} (AUC: {self.results['pr_auc']:.3f})")
        plt.plot(r2, p2, label=f"{other.model_name} (AUC: {other.results['pr_auc']:.3f})")
        plt.xlabel("Recall")
        plt.ylabel("Precision")
        plt.title("PR Curve Comparison")
        plt.legend()
        plt.grid(alpha=0.3)
        plt.show()

    def failure_analysis(self, test_raw, top_n=3):
        probs = self.results['raw_probs']
        labels = self.results['raw_labels']

        fp = np.where((labels == 0) & (probs > 0.8))[0]
        fn = np.where((labels == 1) & (probs < 0.2))[0]
        tp = np.where((labels == 1) & (probs > 0.9))[0]

        categories = [("False Positive (FP)", fp), ("False Negative (FN)", fn), ("True Positive (TP)", tp)]

        print(f"\n=== Failure Analysis: {self.model_name} ===")
        for name, idxs in categories:
            print(f"\n{name}:")
            if len(idxs) == 0:
                print("No examples found.")
                continue
            
            for i in idxs[:top_n]:
                print(f"Confidence Score: {probs[i]:.4f}")
                
                # Robust key checking to avoid KeyError: 'code'
                if 'source_code' in test_raw[i]:
                    print(test_raw[i]['source_code'][:300])
                elif 'code' in test_raw[i]:
                    print(test_raw[i]['code'][:300])
                elif self.tokenizer and 'anchor_input_ids' in test_raw[i]:
                    # Decode tokens if raw text isn't available
                    text = self.tokenizer.decode(test_raw[i]['anchor_input_ids'], skip_special_tokens=True)
                    print(text[:300])
                else:
                    print("[Code snippet not found in test_raw]")
                
                print("-" * 40)

    def save_results(self, prefix):
        summary = {k: v for k, v in self.results.items() if "raw" not in k}
        with open(f"{prefix}_metrics.json", "w") as f:
            json.dump(summary, f, indent=4, default=float)
        pd.DataFrame([summary]).to_csv(f"{prefix}_metrics.csv", index=False)
        np.save(f"{prefix}_probs.npy", self.results["raw_probs"])
        np.save(f"{prefix}_labels.npy", self.results["raw_labels"])
        print(f"[{self.model_name}] Results saved.")

    def aggregate_contracts(self, contract_ids, strategy="max"):
        # Logic for mapping chunk-level results back to full Smart Contracts
        probs, labels = self.results['raw_probs'], self.results['raw_labels']
        threshold = self.results['threshold']

        grouped, gt = defaultdict(list), {}
        for p, cid, l in zip(probs, contract_ids, labels):
            grouped[cid].append(p)
            gt[cid] = l

        final_preds, final_labels = [], []
        for cid, p_list in grouped.items():
            score = max(p_list) if strategy == "max" else np.mean(p_list)
            final_preds.append(1 if score > threshold else 0)
            final_labels.append(gt[cid])

        f1 = precision_recall_fscore_support(final_labels, final_preds, average='binary')[2]
        print(f"\n[{self.model_name}] Contract-Level (Strategy: {strategy})")
        print(f"F1: {f1:.4f} | Acc: {accuracy_score(final_labels, final_preds):.4f}")
        return final_labels, final_preds
