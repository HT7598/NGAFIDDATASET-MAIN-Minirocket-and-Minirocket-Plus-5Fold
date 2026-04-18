import argparse
import json
import os
import sys
from pathlib import Path

import joblib
import numpy as np
import pandas as pd
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import accuracy_score, f1_score, roc_auc_score

from ngafiddataset.dataset.dataset import NGAFID_Dataset_Manager
from ngafiddataset.dataset.utils import get_slice


PROJECT_ROOT = Path(__file__).resolve().parent
MINIROCKET_CODE_DIR = PROJECT_ROOT / "minirocket-main" / "code"
if str(MINIROCKET_CODE_DIR) not in sys.path:
    sys.path.insert(0, str(MINIROCKET_CODE_DIR))

from minirocket_multivariate import fit as minirocket_fit  # noqa: E402
from minirocket_multivariate import transform as minirocket_transform  # noqa: E402


def set_seed(seed: int = 42) -> None:
    np.random.seed(seed)
    os.environ["PYTHONHASHSEED"] = str(seed)


def scale_and_clean(data: np.ndarray, mins: np.ndarray, maxs: np.ndarray) -> np.ndarray:
    denom = maxs - mins
    denom = np.where(denom == 0, 1.0, denom)
    data = (data - mins) / denom
    return np.nan_to_num(data, copy=False).astype(np.float32)


def examples_to_numpy(examples, mins: np.ndarray, maxs: np.ndarray):
    x = np.asarray([example["data"] for example in examples], dtype=np.float32)
    y = np.asarray([example["before_after"] for example in examples], dtype=np.int32)
    ids = np.asarray([example["id"] for example in examples])

    x = scale_and_clean(x, mins, maxs)
    x = np.transpose(x, (0, 2, 1)).astype(np.float32)
    return x, y, ids


def evaluate_fold(y_true: np.ndarray, y_prob: np.ndarray):
    y_pred = (y_prob >= 0.5).astype(np.int32)
    metrics = {
        "accuracy": float(accuracy_score(y_true, y_pred)),
        "f1": float(f1_score(y_true, y_pred, zero_division=0)),
        "roc_auc": float("nan"),
    }

    if len(np.unique(y_true)) > 1:
        metrics["roc_auc"] = float(roc_auc_score(y_true, y_prob))

    return metrics, y_pred


def run_fold(dm, fold: int, args, output_dir: Path):
    train_examples = get_slice(dm.data_dict, fold=fold, reverse=True)
    val_examples = get_slice(dm.data_dict, fold=fold, reverse=False)

    train_x, train_y, _ = examples_to_numpy(train_examples, dm.mins, dm.maxs)
    val_x, val_y, val_ids = examples_to_numpy(val_examples, dm.mins, dm.maxs)

    set_seed(args.seed + fold)
    minirocket_parameters = minirocket_fit(
        train_x,
        num_features=args.num_features,
        max_dilations_per_kernel=args.max_dilations_per_kernel,
    )

    train_features = minirocket_transform(train_x, minirocket_parameters).astype(np.float32)
    val_features = minirocket_transform(val_x, minirocket_parameters).astype(np.float32)

    classifier = LogisticRegression(
        max_iter=args.max_iter,
        random_state=args.seed + fold,
        solver="liblinear",
    )
    classifier.fit(train_features, train_y)

    y_prob = classifier.predict_proba(val_features)[:, 1]
    metrics, y_pred = evaluate_fold(val_y, y_prob)
    metrics["fold"] = fold
    metrics["n_train"] = len(train_examples)
    metrics["n_val"] = len(val_examples)
    metrics["num_features"] = int(train_features.shape[1])

    fold_dir = output_dir / f"fold_{fold}"
    fold_dir.mkdir(parents=True, exist_ok=True)

    pd.DataFrame(
        {
            "id": val_ids,
            "y_true": val_y,
            "y_prob": y_prob,
            "y_pred": y_pred,
        }
    ).to_csv(fold_dir / "predictions.csv", index=False)

    with open(fold_dir / "metrics.json", "w", encoding="utf-8") as f:
        json.dump(metrics, f, ensure_ascii=False, indent=2)

    joblib.dump(
        {
            "classifier": classifier,
            "minirocket_parameters": minirocket_parameters,
        },
        fold_dir / "minirocket_artifacts.joblib",
    )

    return metrics



def parse_args():
    parser = argparse.ArgumentParser(description="NGAFID MiniRocket before/after 5-fold cross validation")
    parser.add_argument("--dataset-name", default="2days")
    parser.add_argument("--dataset-dir", default=".")
    parser.add_argument("--output-root", default="outputs")
    parser.add_argument("--model-name", default="minirocket")
    parser.add_argument("--num-features", type=int, default=10000)
    parser.add_argument("--max-dilations-per-kernel", type=int, default=32)
    parser.add_argument("--max-iter", type=int, default=1000)
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--only-fold", type=int, default=None, choices=[0, 1, 2, 3, 4])
    return parser.parse_args()



def main():
    args = parse_args()
    set_seed(args.seed)

    output_dir = Path(args.output_root) / args.model_name / "before_after_cv"
    output_dir.mkdir(parents=True, exist_ok=True)

    dm = NGAFID_Dataset_Manager(args.dataset_name, destination=args.dataset_dir)
    dm.data_dict = dm.construct_data_dictionary(numpy=True)

    folds = [args.only_fold] if args.only_fold is not None else list(range(5))
    results = []

    for fold in folds:
        print(f"\n===== Running MiniRocket fold {fold} =====")
        metrics = run_fold(dm, fold, args, output_dir)
        print(
            "Fold {fold}: accuracy={accuracy:.4f}, f1={f1:.4f}, roc_auc={roc_auc:.4f}".format(
                **metrics
            )
        )
        results.append(metrics)

    results_df = pd.DataFrame(results).sort_values("fold")
    results_df.to_csv(output_dir / "cv_results.csv", index=False)

    summary = {
        "model_name": args.model_name,
        "accuracy_mean": float(results_df["accuracy"].mean()),
        "accuracy_std": float(results_df["accuracy"].std(ddof=1)) if len(results_df) > 1 else 0.0,
        "f1_mean": float(results_df["f1"].mean()),
        "f1_std": float(results_df["f1"].std(ddof=1)) if len(results_df) > 1 else 0.0,
        "roc_auc_mean": float(results_df["roc_auc"].mean()),
        "roc_auc_std": float(results_df["roc_auc"].std(ddof=1)) if len(results_df) > 1 else 0.0,
        "folds": folds,
        "num_features": args.num_features,
        "max_dilations_per_kernel": args.max_dilations_per_kernel,
    }

    with open(output_dir / "summary.json", "w", encoding="utf-8") as f:
        json.dump(summary, f, ensure_ascii=False, indent=2)

    print("\n===== MiniRocket cross-validation summary =====")
    print(results_df[["fold", "accuracy", "f1", "roc_auc"]])
    print(
        "accuracy = {accuracy_mean:.4f} ± {accuracy_std:.4f}\n"
        "f1       = {f1_mean:.4f} ± {f1_std:.4f}\n"
        "roc_auc  = {roc_auc_mean:.4f} ± {roc_auc_std:.4f}".format(**summary)
    )


if __name__ == "__main__":
    main()
