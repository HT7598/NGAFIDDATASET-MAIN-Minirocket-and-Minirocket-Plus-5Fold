import argparse
import json
import os
import numpy as np
import pandas as pd
import torch
from sklearn.metrics import accuracy_score, f1_score, roc_auc_score
from ngafiddataset.dataset.dataset import NGAFID_Dataset_Manager
from ngafiddataset.dataset.utils import get_slice, replace_nan_w_zero
from tsai.basics import *
from tsai.models.MINIROCKET_Pytorch import MiniRocketFeatures, get_minirocket_features, MiniRocketHead
from tsai.tscore import *

# 固定随机种子，保证可复现
def set_seed(seed=42):
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
    os.environ["PYTHONHASHSEED"] = str(seed)

# 评估函数
def evaluate_fold(y_true, y_prob):
    y_pred = (y_prob >= 0.5).astype(np.int32)
    y_true = np.asarray(y_true).astype(np.int32)
    metrics = {
        "accuracy": float(accuracy_score(y_true, y_pred)),
        "f1": float(f1_score(y_true, y_pred, zero_division=0)),
        "roc_auc": float("nan"),
    }
    if len(np.unique(y_true)) > 1:
        metrics["roc_auc"] = float(roc_auc_score(y_true, y_prob))
    return metrics, y_prob, y_pred

# 单折训练
def run_fold(dm, fold, args, output_dir):
    # 严格复用你之前的折划分逻辑
    train_examples = get_slice(dm.data_dict, fold=fold, reverse=True)
    val_examples = get_slice(dm.data_dict, fold=fold, reverse=False)

    # 数据提取
    train_X = np.array([x["data"] for x in train_examples], dtype=np.float32)
    val_X = np.array([x["data"] for x in val_examples], dtype=np.float32)
    train_Y = np.array([x["before_after"] for x in train_examples], dtype=np.int64)
    val_Y = np.array([x["before_after"] for x in val_examples], dtype=np.int64)

    # 数据预处理：归一化+填充NaN（和论文一致）
    train_X = (train_X - dm.mins) / (dm.maxs - dm.mins)
    val_X = (val_X - dm.mins) / (dm.maxs - dm.mins)
    train_X = np.nan_to_num(train_X, copy=False)
    val_X = np.nan_to_num(val_X, copy=False)

    # ==================== MiniRocket 核心 ====================
    set_seed(args.seed + fold)
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    
    # 特征提取
    mrf = MiniRocketFeatures(train_X.shape[1], train_X.shape[2]).to(device)
    mrf.fit(train_X, chunksize=args.batch_size)
    
    # 获取MiniRocket特征
    train_feat = get_minirocket_features(train_X, mrf, chunksize=args.batch_size, to_np=True)
    val_feat = get_minirocket_features(val_X, mrf, chunksize=args.batch_size, to_np=True)

    # 构建数据加载器
    tfms = [None, TSClassification()]
    batch_tfms = TSStandardize(by_sample=True)
    dls = get_ts_dls(train_feat, train_Y, valid_data=(val_feat, val_Y), 
                    tfms=tfms, batch_tfms=batch_tfms, bs=args.batch_size)

    # 构建模型
    model = build_ts_model(MiniRocketHead, dls=dls)
    learn = Learner(dls, model, metrics=accuracy, 
                   cbs=[EarlyStoppingCallback(patience=args.patience), SaveModelCallback()])

    # 训练
    learn.fit_one_cycle(args.epochs, lr_max=args.learning_rate)

    # 预测
    y_prob, _ = learn.get_preds(dl=dls.valid)
    y_prob = y_prob.numpy().reshape(-1)
    metrics, y_prob, y_pred = evaluate_fold(val_Y, y_prob)

    # 保存结果
    fold_dir = output_dir / f"fold_{fold}"
    fold_dir.mkdir(parents=True, exist_ok=True)
    
    # 保存预测
    pred_df = pd.DataFrame({
        "id": [x["id"] for x in val_examples],
        "y_true": val_Y.tolist(),
        "y_prob": y_prob.tolist(),
        "y_pred": y_pred.tolist()
    })
    pred_df.to_csv(fold_dir / "predictions.csv", index=False)
    
    # 保存指标
    metrics["fold"] = fold
    metrics["n_train"] = len(train_examples)
    metrics["n_val"] = len(val_examples)
    with open(fold_dir / "metrics.json", "w", encoding="utf-8") as f:
        json.dump(metrics, f, ensure_ascii=False, indent=2)

    return metrics

# 命令行参数
def parse_args():
    parser = argparse.ArgumentParser(description="NGAFID MiniRocket before/after 5-fold CV")
    parser.add_argument("--dataset-name", default="2days")
    parser.add_argument("--dataset-dir", default=".")
    parser.add_argument("--output-root", default="outputs")
    parser.add_argument("--epochs", type=int, default=200)
    parser.add_argument("--batch-size", type=int, default=64)
    parser.add_argument("--learning-rate", type=float, default=2.5e-5)
    parser.add_argument("--patience", type=int, default=5)
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--only-fold", type=int, default=None, choices=[0,1,2,3,4])
    return parser.parse_args()

# 主函数
def main():
    args = parse_args()
    set_seed(args.seed)
    output_dir = Path(args.output_root) / "minirocket" / "before_after_cv"
    output_dir.mkdir(parents=True, exist_ok=True)

    # 加载数据集（和你之前的代码完全一致）
    dm = NGAFID_Dataset_Manager(args.dataset_name, destination=args.dataset_dir)
    dm.data_dict = dm.construct_data_dictionary(numpy=True)

    # 5折训练
    folds = [args.only_fold] if args.only_fold is not None else list(range(5))
    results = []
    for fold in folds:
        print(f"\n===== Running MiniRocket Fold {fold} =====")
        metrics = run_fold(dm, fold, args, output_dir)
        print(f"Fold {fold}: Acc={metrics['accuracy']:.4f}, F1={metrics['f1']:.4f}, AUC={metrics['roc_auc']:.4f}")
        results.append(metrics)

    # 汇总结果
    results_df = pd.DataFrame(results).sort_values("fold")
    results_df.to_csv(output_dir / "cv_results.csv", index=False)
    
    summary = {
        "accuracy_mean": float(results_df["accuracy"].mean()),
        "accuracy_std": float(results_df["accuracy"].std(ddof=1)) if len(results_df)>1 else 0.0,
        "f1_mean": float(results_df["f1"].mean()),
        "f1_std": float(results_df["f1"].std(ddof=1)) if len(results_df)>1 else 0.0,
        "roc_auc_mean": float(results_df["roc_auc"].mean()),
        "roc_auc_std": float(results_df["roc_auc"].std(ddof=1)) if len(results_df)>1 else 0.0,
    }
    with open(output_dir / "summary.json", "w", encoding="utf-8") as f:
        json.dump(summary, f, ensure_ascii=False, indent=2)

    # 打印最终结果
    print("\n===== MiniRocket 5折CV汇总 =====")
    print(results_df[["fold","accuracy","f1","roc_auc"]])
    print(f"\nAccuracy: {summary['accuracy_mean']:.4f} ± {summary['accuracy_std']:.4f}")
    print(f"F1: {summary['f1_mean']:.4f} ± {summary['f1_std']:.4f}")
    print(f"ROC-AUC: {summary['roc_auc_mean']:.4f} ± {summary['roc_auc_std']:.4f}")

if __name__ == "__main__":
    main()